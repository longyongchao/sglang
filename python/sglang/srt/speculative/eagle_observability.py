"""Eagle sub-segment profiling infrastructure.

Mirrors the DSpark/DFlash pattern with segments appropriate to Eagle's
multi-step draft-forward → tree-build → target-verify → sample cycle,
including per-step GPU timing arrays.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from enum import Enum
from typing import Callable, ContextManager, Iterator, Optional, Union

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_NULL_SEGMENT = nullcontext()

ALL_COMPONENTS_TOKEN = "all"

INFO_DUMP_MAX_RECORDS = 200_000
INFO_DUMP_MAX_STEP_CPU_SECONDS = 1.0


class EagleInfoComponent(str, Enum):
    CORE = "core"
    STEP_CPU_TIME = "step_cpu_time"
    STEP_GPU_TIME = "step_gpu_time"
    DRAFT_GPU_TIME = "draft_gpu_time"
    DRAFT_PREPARE_GPU_TIME = "draft_prepare_gpu_time"
    DRAFT_FORWARD_STEPS_GPU_TIME = "draft_forward_steps_gpu_time"
    DRAFT_PER_STEP_GPU_TIME = "draft_per_step_gpu_time"
    DRAFT_TREE_BUILD_GPU_TIME = "draft_tree_build_gpu_time"
    TARGET_VERIFY_GPU_TIME = "target_verify_gpu_time"
    VERIFY_SAMPLE_GPU_TIME = "verify_sample_gpu_time"
    DRAFT_EXTEND_GPU_TIME = "draft_extend_gpu_time"


class EagleInfoSegment(str, Enum):
    STEP = "step"
    DRAFT = "draft"
    DRAFT_PREPARE = "draft_prepare"
    DRAFT_FORWARD_STEPS = "draft_forward_steps"
    DRAFT_PER_STEP = "draft_per_step"
    DRAFT_TREE_BUILD = "draft_tree_build"
    TARGET_VERIFY = "target_verify"
    VERIFY_SAMPLE = "verify_sample"
    DRAFT_EXTEND = "draft_extend"


def resolve_enabled_components() -> set[EagleInfoComponent]:
    """Read SGLANG_SPEC_DEBUG_DUMP (unified), fallback SGLANG_DSPARK_DEBUG_DUMP (legacy)."""
    raw = envs.SGLANG_SPEC_DEBUG_DUMP.get() or envs.SGLANG_DSPARK_DEBUG_DUMP.get()
    components = _resolve_components(raw)
    if EagleInfoComponent.DRAFT_GPU_TIME in components:
        components |= {
            EagleInfoComponent.DRAFT_PREPARE_GPU_TIME,
            EagleInfoComponent.DRAFT_FORWARD_STEPS_GPU_TIME,
            EagleInfoComponent.DRAFT_PER_STEP_GPU_TIME,
            EagleInfoComponent.DRAFT_TREE_BUILD_GPU_TIME,
        }
    return components


def _resolve_components(raw: tuple[str, ...]) -> set[EagleInfoComponent]:
    tokens = {token.strip() for token in raw if token.strip()}
    if not tokens:
        return set()
    if ALL_COMPONENTS_TOKEN in tokens:
        return set(EagleInfoComponent)
    try:
        return {EagleInfoComponent(token) for token in tokens}
    except ValueError as exc:
        valid = [c.value for c in EagleInfoComponent]
        raise ValueError(
            f"Invalid SGLANG_SPEC_DEBUG_DUMP token in {sorted(tokens)}; "
            f"valid: {valid} or '{ALL_COMPONENTS_TOKEN}'."
        ) from exc


class EagleDecodeStepRecord(msgspec.Struct, omit_defaults=True):
    forward_ct: int
    bs: int = -1
    step_cpu_ms: Optional[float] = None
    step_gpu_ms: Optional[float] = None
    draft_gpu_ms: Optional[float] = None
    draft_prepare_gpu_ms: Optional[float] = None
    draft_forward_steps_gpu_ms: Optional[float] = None
    draft_per_step_gpu_ms: Optional[list[float]] = None
    draft_tree_build_gpu_ms: Optional[float] = None
    target_verify_gpu_ms: Optional[float] = None
    verify_sample_gpu_ms: Optional[float] = None
    draft_extend_gpu_ms: Optional[float] = None


class _PendingStep(msgspec.Struct):
    forward_ct: int
    bs: int
    step_cpu_ms: Optional[float]
    segment_events: dict[EagleInfoSegment, tuple[torch.cuda.Event, torch.cuda.Event]]
    per_step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]]


_SEGMENT_TO_COMPONENT: dict[EagleInfoSegment, EagleInfoComponent] = {
    EagleInfoSegment.STEP: EagleInfoComponent.STEP_GPU_TIME,
    EagleInfoSegment.DRAFT: EagleInfoComponent.DRAFT_GPU_TIME,
    EagleInfoSegment.DRAFT_PREPARE: EagleInfoComponent.DRAFT_PREPARE_GPU_TIME,
    EagleInfoSegment.DRAFT_FORWARD_STEPS: EagleInfoComponent.DRAFT_FORWARD_STEPS_GPU_TIME,
    EagleInfoSegment.DRAFT_TREE_BUILD: EagleInfoComponent.DRAFT_TREE_BUILD_GPU_TIME,
    EagleInfoSegment.TARGET_VERIFY: EagleInfoComponent.TARGET_VERIFY_GPU_TIME,
    EagleInfoSegment.VERIFY_SAMPLE: EagleInfoComponent.VERIFY_SAMPLE_GPU_TIME,
    EagleInfoSegment.DRAFT_EXTEND: EagleInfoComponent.DRAFT_EXTEND_GPU_TIME,
}

_SEGMENT_TO_FIELD: dict[EagleInfoSegment, str] = {
    EagleInfoSegment.STEP: "step_gpu_ms",
    EagleInfoSegment.DRAFT: "draft_gpu_ms",
    EagleInfoSegment.DRAFT_PREPARE: "draft_prepare_gpu_ms",
    EagleInfoSegment.DRAFT_FORWARD_STEPS: "draft_forward_steps_gpu_ms",
    EagleInfoSegment.DRAFT_TREE_BUILD: "draft_tree_build_gpu_ms",
    EagleInfoSegment.TARGET_VERIFY: "target_verify_gpu_ms",
    EagleInfoSegment.VERIFY_SAMPLE: "verify_sample_gpu_ms",
    EagleInfoSegment.DRAFT_EXTEND: "draft_extend_gpu_ms",
}


class EagleInfoDumper:
    def __init__(
        self,
        *,
        components: set[Union[EagleInfoComponent, str]],
        num_steps: int,
        attn_tp_rank: int,
        device: torch.device,
        max_records: int = INFO_DUMP_MAX_RECORDS,
        max_step_cpu_seconds: float = INFO_DUMP_MAX_STEP_CPU_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.num_steps = int(num_steps)
        self.attn_tp_rank = int(attn_tp_rank)
        self.device = device
        self._clock = clock
        self._max_step_cpu_seconds = max_step_cpu_seconds

        self._components: set[EagleInfoComponent] = {
            EagleInfoComponent(c) for c in components
        }
        self.enabled = bool(self._components) and self.attn_tp_rank == 0

        self._records: deque[EagleDecodeStepRecord] = deque(maxlen=max_records)
        self._pending: Optional[_PendingStep] = None
        self._prev_stamp: Optional[float] = None
        self._current_segments: dict[
            EagleInfoSegment, tuple[torch.cuda.Event, torch.cuda.Event]
        ] = {}
        self._open_segments: dict[EagleInfoSegment, torch.cuda.Event] = {}
        self._current_per_step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    @property
    def profiling_active(self) -> bool:
        """Always False — profiling no longer bypasses CUDA graph fold."""
        return False

    def begin_step(self) -> None:
        if not self.enabled:
            return
        self._current_segments = {}
        self._open_segments = {}
        self._current_per_step_events = []
        if EagleInfoComponent.STEP_GPU_TIME in self._components:
            self._open_segment(EagleInfoSegment.STEP)

    def segment(self, name: Union[EagleInfoSegment, str]) -> ContextManager[None]:
        if not self.enabled:
            return _NULL_SEGMENT
        seg = EagleInfoSegment(name)
        comp = _SEGMENT_TO_COMPONENT.get(seg)
        if comp is None or comp not in self._components:
            return _NULL_SEGMENT
        return self._active_segment(seg)

    @contextmanager
    def _active_segment(self, segment: EagleInfoSegment) -> Iterator[None]:
        self._open_segment(segment)
        try:
            yield
        finally:
            self._close_segment(segment)

    def open_per_step_segment(self) -> None:
        """Mark the start of one draft forward step for per-step timing."""
        if not self.enabled:
            return
        if EagleInfoComponent.DRAFT_PER_STEP_GPU_TIME not in self._components:
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._open_segments[EagleInfoSegment.DRAFT_PER_STEP] = start

    def close_per_step_segment(self) -> None:
        """Mark the end of one draft forward step for per-step timing."""
        if not self.enabled:
            return
        start = self._open_segments.pop(EagleInfoSegment.DRAFT_PER_STEP, None)
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._current_per_step_events.append((start, end))

    def observe_decode_step(self, forward_ct: int, bs: int) -> None:
        if not self.enabled:
            return
        if EagleInfoComponent.STEP_GPU_TIME in self._components:
            self._close_segment(EagleInfoSegment.STEP)

        now = self._clock()
        step_cpu_ms = self._step_cpu_ms(now=now)
        self._drain_pending()

        self._pending = _PendingStep(
            forward_ct=int(forward_ct),
            bs=int(bs),
            step_cpu_ms=step_cpu_ms,
            segment_events=self._current_segments,
            per_step_events=self._current_per_step_events,
        )
        self._current_segments = {}
        self._current_per_step_events = []
        self._prev_stamp = now

    def flush(self) -> None:
        if not self.enabled:
            return
        self._drain_pending()

    def clear(self) -> None:
        self._records.clear()
        self._pending = None
        self._prev_stamp = None
        self._current_segments = {}
        self._open_segments = {}
        self._current_per_step_events = []

    def dump(self) -> Optional[dict]:
        if not self.enabled:
            return None
        self.flush()
        return {
            "num_steps": self.num_steps,
            "components": sorted(c.value for c in self._components),
            "records": [msgspec.to_builtins(r) for r in self._records],
        }

    def _open_segment(self, segment: EagleInfoSegment) -> None:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._open_segments[segment] = start

    def _close_segment(self, segment: EagleInfoSegment) -> None:
        start = self._open_segments.pop(segment, None)
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._current_segments[segment] = (start, end)

    def _drain_pending(self) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return

        record = EagleDecodeStepRecord(forward_ct=pending.forward_ct)
        if EagleInfoComponent.CORE in self._components:
            record.bs = pending.bs
        if EagleInfoComponent.STEP_CPU_TIME in self._components:
            record.step_cpu_ms = pending.step_cpu_ms
        for seg, field in _SEGMENT_TO_FIELD.items():
            comp = _SEGMENT_TO_COMPONENT[seg]
            if comp in self._components:
                ms = self._segment_ms(pending, seg)
                if ms is not None:
                    setattr(record, field, ms)
        # Per-step array
        if (
            EagleInfoComponent.DRAFT_PER_STEP_GPU_TIME in self._components
            and pending.per_step_events
        ):
            per_step_ms = []
            for start, end in pending.per_step_events:
                end.synchronize()
                elapsed = start.elapsed_time(end)
                if elapsed > self._max_step_cpu_seconds * 1000.0:
                    per_step_ms.append(None)
                else:
                    per_step_ms.append(round(elapsed, 4))
            record.draft_per_step_gpu_ms = per_step_ms
        self._records.append(record)

    def _step_cpu_ms(self, *, now: float) -> Optional[float]:
        prev = self._prev_stamp
        if prev is None:
            return None
        step_cpu = now - prev
        if not (0.0 < step_cpu <= self._max_step_cpu_seconds):
            return None
        return round(step_cpu * 1000.0, 4)

    def _segment_ms(
        self, pending: _PendingStep, segment: EagleInfoSegment
    ) -> Optional[float]:
        events = pending.segment_events.get(segment)
        if events is None:
            return None
        start, end = events
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
        if elapsed_ms > self._max_step_cpu_seconds * 1000.0:
            return None
        return round(elapsed_ms, 4)


class EagleStepObservers:
    """Lightweight facade wrapping EagleInfoDumper for the worker's decode path."""

    def __init__(
        self,
        *,
        device: torch.device,
        num_steps: int,
        attn_tp_rank: int,
    ) -> None:
        self._info_dumper = EagleInfoDumper(
            components=resolve_enabled_components(),
            num_steps=num_steps,
            attn_tp_rank=attn_tp_rank,
            device=device,
        )

    @property
    def profiling_active(self) -> bool:
        return self._info_dumper.profiling_active

    def begin_step(self) -> None:
        self._info_dumper.begin_step()

    def segment(self, name: Union[EagleInfoSegment, str]) -> ContextManager[None]:
        return self._info_dumper.segment(name)

    def open_per_step_segment(self) -> None:
        self._info_dumper.open_per_step_segment()

    def close_per_step_segment(self) -> None:
        self._info_dumper.close_per_step_segment()

    def observe_decode_step(self, forward_ct: int, bs: int) -> None:
        self._info_dumper.observe_decode_step(forward_ct, bs)

    def dump_info_records(self) -> Optional[dict]:
        return self._info_dumper.dump()

    def clear_info_records(self) -> None:
        self._info_dumper.clear()
