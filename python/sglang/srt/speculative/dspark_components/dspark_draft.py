from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Callable, ContextManager, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_sampled_draft_graph import (
    SampledDraftGraphRunner,
    sampled_markov_rollout,
    sampled_rollout_ineligible_reason,
    sampled_rollout_max_bs,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context

logger = logging.getLogger(__name__)


class DraftBlockResult(msgspec.Struct, frozen=True):
    draft_tokens: torch.Tensor
    corrected_logits: Optional[torch.Tensor]
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor
    deterministic_draft: bool = False


class DraftForwardResult(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    raw_hidden: torch.Tensor
    draft_hidden_3d: torch.Tensor
    can_run_graph: bool


class DraftProposal(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    draft_block: DraftBlockResult
    draft_hidden: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor] = None
    confidence_tap: Optional[torch.Tensor] = None
    folded: bool = False
    # The sampled rollout ran in the draft graph: draft tokens sit in the
    # sampler's ``out`` and q in its static logits buffer for this step.
    sampled_folded: bool = False


def greedy_step_sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
    del step_idx
    return torch.argmax(step_logits, dim=-1)


class _SampledFoldBuffers(msgspec.Struct):
    """Static buffers of the sampled draft rollout folded into the draft graph.

    Captured temporaries come from the graph mempool and are unreadable after
    replay, so everything the verifier reads, and every per-request input
    staged before replay, lives here.
    """

    # [max_bs, gamma, V] corrected logits (q for the verifier), written in-graph
    logits: torch.Tensor
    # staged before each replay; padded rows keep earlier valid values
    temperatures: torch.Tensor  # [max_bs] fp32
    greedy_mask: torch.Tensor  # [max_bs] int32
    # scratch
    latent: torch.Tensor  # [max_bs, R]
    exp_noise: torch.Tensor  # [max_bs, V] fp32
    out_val: torch.Tensor  # [max_bs, num_v_blocks] fp32
    out_idx: torch.Tensor  # [max_bs, num_v_blocks] int32
    step_tokens: torch.Tensor  # [max_bs] int64


class SampledDraftIO(msgspec.Struct, frozen=True):
    """Externally owned outputs / staged inputs of the sampled draft fold.

    The DSpark verify epilogue allocates these before the target graphs are
    captured so that its in-graph rejection sampling can read q and the draft
    temperatures from the same storage the draft graph writes.
    """

    logits: torch.Tensor  # [max_bs, gamma, V]
    temperatures: torch.Tensor  # [max_bs] fp32
    greedy_mask: torch.Tensor  # [max_bs] int32

    @classmethod
    def allocate(cls, *, max_bs: int, gamma: int, vocab_size: int, dtype, device):
        return cls(
            logits=torch.zeros((max_bs, gamma, vocab_size), dtype=dtype, device=device),
            temperatures=torch.ones((max_bs,), dtype=torch.float32, device=device),
            greedy_mask=torch.ones((max_bs,), dtype=torch.int32, device=device),
        )

    def fits(self, *, max_bs: int, gamma: int, vocab_size: int, dtype) -> bool:
        return (
            self.logits.shape[0] >= max_bs
            and tuple(self.logits.shape[1:]) == (gamma, vocab_size)
            and self.logits.dtype == dtype
        )


class DsparkDraftSampler:
    """Draft proposal run inside the draft cuda graph (capture tail hook).

    Greedy by default. In sampled mode (``sampled_max_bs`` given, draft
    temperature > 0) it replays the fast-sampling VanillaMarkov rollout of
    ``sample_draft_block`` in-graph and keeps the corrected logits in static
    buffers for the rejection-sampling verifier, so the base-logits GEMM and
    the gamma-step rollout no longer run outside the graph.
    """

    def __init__(
        self,
        *,
        model,
        gamma,
        max_bs,
        device,
        confidence_fn=None,
        out=None,
        sampled_max_bs: Optional[int] = None,
        sampled_io: Optional[SampledDraftIO] = None,
    ):
        self.model = model
        self.markov_head = model.markov_head
        self.gamma = int(gamma)
        if out is not None:
            assert out.shape == (int(max_bs) * self.gamma,) and out.dtype == torch.int64
            self.out = out
        else:
            self.out = torch.empty(
                (int(max_bs) * self.gamma,), dtype=torch.int64, device=device
            )
        self.confidence_fn = confidence_fn
        self.confidence_out = (
            torch.empty((int(max_bs), self.gamma), dtype=torch.float32, device=device)
            if confidence_fn is not None
            else None
        )
        self.sampled = sampled_max_bs is not None
        self.max_bs = int(sampled_max_bs) if self.sampled else int(max_bs)
        # Whether the sampled buffers are the caller's ``sampled_io``.
        self.shares_io = False
        # Set during capture when the base logits do not match the Markov head;
        # the proposal then keeps the eager sampled path.
        self._sampled_disabled = False
        # captured draft graph bs -> whether its tail ran the sampled rollout
        self._captured_bs: dict[int, bool] = {}
        if self.sampled:
            # detach(): same storage, but out= ops and kernel args must not
            # see a requires_grad Parameter.
            self._embed_table = self.markov_head.markov_w1.weight.detach()
            self._fc2_weight = self.markov_head.markov_w2.weight.detach()
            # Plain tensors regardless of the caller's grad mode: tensors made
            # under inference_mode cannot be updated in place outside it, and
            # inputs are staged from the scheduler, which may run either way.
            with torch.inference_mode(False):
                self._sampled_buffers = self._build_sampled_buffers(
                    device, sampled_io
                )
            self._fused_scratch = None
        else:
            self._sampled_buffers = None
            # Fused kernel scratch buffers
            self._fused_scratch = self._try_build_scratch(
                model=model, max_bs=max_bs, device=device
            )

    def _build_sampled_buffers(
        self, device, sampled_io: Optional[SampledDraftIO]
    ) -> _SampledFoldBuffers:
        from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
            MARKOV_BLOCK_V,
        )

        weight = self.markov_head.markov_w2.weight
        vocab_size, dtype = int(weight.shape[0]), weight.dtype
        num_v_blocks = (vocab_size + MARKOV_BLOCK_V - 1) // MARKOV_BLOCK_V
        max_bs = self.max_bs
        self.shares_io = sampled_io is not None and sampled_io.fits(
            max_bs=max_bs, gamma=self.gamma, vocab_size=vocab_size, dtype=dtype
        )
        io = (
            sampled_io
            if self.shares_io
            else SampledDraftIO.allocate(
                max_bs=max_bs,
                gamma=self.gamma,
                vocab_size=vocab_size,
                dtype=dtype,
                device=device,
            )
        )
        return _SampledFoldBuffers(
            logits=io.logits[:max_bs],
            temperatures=io.temperatures[:max_bs],
            greedy_mask=io.greedy_mask[:max_bs],
            latent=torch.empty(
                (max_bs, int(self.markov_head.markov_rank)), dtype=dtype, device=device
            ),
            exp_noise=torch.empty(
                (max_bs, vocab_size), dtype=torch.float32, device=device
            ),
            out_val=torch.empty(
                (max_bs, num_v_blocks), dtype=torch.float32, device=device
            ),
            out_idx=torch.empty((max_bs, num_v_blocks), dtype=torch.int32, device=device),
            step_tokens=torch.zeros((max_bs,), dtype=torch.int64, device=device),
        )

    def serves_sampled(self, bs: int) -> bool:
        """Whether the draft graph a batch of ``bs`` pads to (the smallest
        captured size >= bs) ran the sampled rollout."""
        if not self.sampled or self._sampled_disabled:
            return False
        padded = [b for b in self._captured_bs if b >= bs]
        return bool(padded) and self._captured_bs[min(padded)]

    def stage_sampled_inputs(
        self, *, temperatures: torch.Tensor, greedy_mask: torch.Tensor
    ) -> None:
        """Copy this step's per-request inputs in before the draft replay."""
        bs = temperatures.shape[0]
        self._sampled_buffers.temperatures[:bs].copy_(temperatures)
        self._sampled_buffers.greedy_mask[:bs].copy_(greedy_mask)

    def sampled_logits(self, bs: int) -> torch.Tensor:
        return self._sampled_buffers.logits[:bs]

    @staticmethod
    def _try_build_scratch(*, model, max_bs, device):
        try:
            from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
                MarkovFusedScratch,
            )

            markov_head = model.markov_head
            vocab_size = markov_head.vocab_size
            markov_rank = markov_head.markov_rank
            is_rnn = markov_head.markov_head_type == "rnn"
            hidden_size = getattr(markov_head, "hidden_size", 0)
            # Determine dtype from the embedding weight
            dtype = markov_head.markov_w1.weight.dtype
            return MarkovFusedScratch(
                max_bs=int(max_bs),
                vocab_size=vocab_size,
                markov_rank=markov_rank,
                hidden_size=hidden_size,
                device=device,
                dtype=dtype,
                is_rnn=is_rnn,
            )
        except Exception:
            return None

    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.gamma
        if self.sampled:
            # Called once per captured draft graph size; propose() looks up the
            # graph a batch pads to before trusting the folded outputs.
            serves = not self._sampled_disabled and bs <= self.max_bs
            self._captured_bs[bs] = serves
            if not serves:
                return
        base_logits, confidence_tap = self.model.compute_base_logits(hidden_states)
        base_logits = base_logits.view(bs, self.gamma, -1)
        anchor = input_ids.view(bs, self.gamma)[:, 0]

        if self.sampled:
            buffers = self._sampled_buffers
            if (
                base_logits.shape[2] != buffers.logits.shape[2]
                or base_logits.dtype != buffers.logits.dtype
            ):
                self._sampled_disabled = True
                logger.warning(
                    "DSpark sampled draft fold disabled: base logits %s/%s do "
                    "not match the markov head; the sampled draft stays eager.",
                    tuple(base_logits.shape),
                    base_logits.dtype,
                )
                self._captured_bs[bs] = False
                return
            draft_tokens = self.out[: bs * self.gamma].view(bs, self.gamma)
            # Padded rows read temperatures / greedy flags left by earlier
            # steps (buffers start as ones), so they never produce NaN.
            sampled_markov_rollout(
                embed_table=self._embed_table,
                fc2_weight=self._fc2_weight,
                base_logits=base_logits,
                logits_out=buffers.logits[:bs],
                anchor_tokens=anchor,
                temperatures=buffers.temperatures[:bs],
                greedy_mask=buffers.greedy_mask[:bs],
                latent=buffers.latent[:bs],
                exp_noise=buffers.exp_noise[:bs],
                out_val=buffers.out_val[:bs],
                out_idx=buffers.out_idx[:bs],
                step_tokens=buffers.step_tokens[:bs],
                draft_tokens=draft_tokens,
            )
        elif (
            self._fused_scratch is not None
            and self.markov_head.markov_head_type in ("vanilla", "rnn")
        ):
            draft_tokens = self.markov_head.sample_block_fused(
                base_logits,
                first_prev_tokens=anchor,
                hidden_states=hidden_states.view(bs, self.gamma, -1),
                scratch=self._fused_scratch,
            )
        else:
            draft_tokens, _ = self.markov_head.sample_block(
                base_logits,
                first_prev_tokens=anchor,
                hidden_states=hidden_states.view(bs, self.gamma, -1),
                sampler=greedy_step_sampler,
            )
        if not self.sampled:
            self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        if self.confidence_out is not None:
            confidence = self.confidence_fn(
                draft_hidden=hidden_states.view(bs, self.gamma, -1),
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
                confidence_tap=confidence_tap,
            )
            self.confidence_out[:bs].copy_(confidence)


def maybe_build_draft_sampler(
    *,
    draft_model,
    gamma: int,
    max_bs: int,
    device,
    tp_rank: int,
    confidence_fn=None,
    out=None,
    draft_temperature: Optional[float] = None,
    capture_bs=None,
    dp_moe_sync: bool = False,
    sampled_io: Optional[SampledDraftIO] = None,
) -> Optional[DsparkDraftSampler]:
    """Build the graph-folded draft sampler, or return None (with the reason
    logged) when the draft model cannot support folding and the proposal must
    stay eager.

    A draft that may sample (draft temperature None or > 0) gets the sampled
    mode, which also produces the corrected logits rejection sampling needs;
    a deterministic draft gets the greedy mode."""

    def _eager(reason):
        if tp_rank == 0:
            logger.info("DSpark draft proposal kept eager (reason=%s).", reason)
        return None

    if gamma <= 0:
        return _eager("gamma<=0")
    if not hasattr(draft_model, "compute_base_logits"):
        return _eager("no compute_base_logits")
    if getattr(draft_model, "markov_head", None) is None:
        return _eager("no markov head")

    sampled_max_bs = None
    if draft_temperature is None or draft_temperature > 0:
        sampled_max_bs, sampled_reason = plan_sampled_fold_max_bs(
            draft_model=draft_model,
            gamma=gamma,
            capture_bs=capture_bs or [max_bs],
            dp_moe_sync=dp_moe_sync,
        )
        if sampled_reason is not None and draft_temperature is not None:
            # Every batch samples the draft and needs corrected logits, which
            # the greedy fold does not produce; capturing it anyway would
            # replay and discard the rollout on every decode step.
            return _eager(f"sampled draft not foldable: {sampled_reason}")
        if sampled_reason is not None and tp_rank == 0:
            logger.info(
                "DSpark sampled draft not folded (reason=%s); only fully greedy "
                "batches use the folded draft.",
                sampled_reason,
            )

    if tp_rank == 0:
        if sampled_max_bs is not None:
            logger.info(
                "DSpark draft proposal folded into the draft cuda graph "
                "(sampled rollout with corrected logits, bs<=%d).",
                sampled_max_bs,
            )
        else:
            logger.info("DSpark draft greedy proposal folded into the draft cuda graph.")
    return DsparkDraftSampler(
        model=draft_model,
        gamma=gamma,
        max_bs=max_bs,
        device=device,
        confidence_fn=confidence_fn,
        out=out,
        sampled_max_bs=sampled_max_bs,
        sampled_io=sampled_io,
    )


def plan_sampled_fold_max_bs(
    *, draft_model, gamma: int, capture_bs, dp_moe_sync: bool
) -> tuple[Optional[int], Optional[str]]:
    """Largest draft graph bs the sampled fold covers, or (None, reason).

    Pure check without allocations, shared by the draft sampler build and the
    verify epilogue, which must size its buffers before the draft graphs exist.
    """
    reason = _sampled_fold_ineligible_reason(
        draft_model=draft_model, dp_moe_sync=dp_moe_sync
    )
    if reason is not None:
        return None, reason
    max_bs = sampled_rollout_max_bs(
        capture_bs=capture_bs, gamma=gamma, markov_head=draft_model.markov_head
    )
    if max_bs is None:
        return None, "no capture bs fits the static buffer budget"
    return max_bs, None


def _sampled_fold_ineligible_reason(*, draft_model, dp_moe_sync: bool):
    if not envs.SGLANG_DSPARK_ENABLE_SAMPLED_DRAFT_GRAPH.get():
        return "SGLANG_DSPARK_ENABLE_SAMPLED_DRAFT_GRAPH=0"
    if not envs.SGLANG_DSPARK_FAST_SAMPLING.get():
        # The folded rollout is the exponential-race sampler only.
        return "SGLANG_DSPARK_FAST_SAMPLING=0"
    return sampled_rollout_ineligible_reason(
        draft_model=draft_model, dp_moe_sync=dp_moe_sync
    )


def make_next_draft_input(
    *,
    bonus_tokens: torch.Tensor,
    new_seq_lens: torch.Tensor,
) -> DFlashDraftInputV2:
    return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens)


def resolve_greedy_mask(
    *,
    bs: int,
    sampling_info,
    device: torch.device,
) -> torch.Tensor:
    if sampling_info is None:
        return torch.ones(bs, dtype=torch.bool, device=device)
    return (sampling_info.top_ks <= 1).view(-1)


def sample_draft_block(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_hidden: torch.Tensor,
    sampling_info,
    markov_head,
    device: torch.device,
    draft_temperature: Optional[float] = None,
    sampled_graph: Optional[SampledDraftGraphRunner] = None,
) -> DraftBlockResult:
    bs = base_logits.shape[0]
    deterministic_draft = draft_temperature == 0
    greedy_mask = (
        torch.ones(bs, dtype=torch.bool, device=device)
        if deterministic_draft
        else resolve_greedy_mask(bs=bs, sampling_info=sampling_info, device=device)
    )
    any_sampling = (
        not deterministic_draft
        and sampling_info is not None
        and not sampling_info.is_all_greedy
    )
    fast_sampling = envs.SGLANG_DSPARK_FAST_SAMPLING.get()

    if draft_temperature is not None:
        temperatures = torch.full(
            (bs,), max(float(draft_temperature), 1e-5), dtype=torch.float32, device=device
        )
    elif sampling_info is None:
        temperatures = torch.ones(bs, dtype=torch.float32, device=device)
    else:
        temperatures = (
            sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
        )

    if sampled_graph is not None and any_sampling and fast_sampling:
        # Same rollout as the fast-sampling eager loop below, replayed from a
        # CUDA graph; None means this batch is not covered, so fall through.
        graph_out = sampled_graph.run(
            base_logits=base_logits,
            anchor_tokens=anchor_tokens,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
        )
        if graph_out is not None:
            draft_tokens, corrected_logits = graph_out
            return DraftBlockResult(
                draft_tokens=draft_tokens,
                corrected_logits=corrected_logits,
                greedy_mask=greedy_mask,
                temperatures=temperatures,
                deterministic_draft=deterministic_draft,
            )

    if not any_sampling:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            return torch.argmax(step_logits, dim=-1)

    else:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            if fast_sampling:
                exp_noise = torch.empty(
                    step_logits.shape, dtype=torch.float32, device=step_logits.device
                ).exponential_(1)
                return SampleStepTokens.execute(
                    step_logits=step_logits,
                    temperatures=temperatures,
                    greedy_mask=greedy_mask,
                    exp_noise=exp_noise,
                )
            else:
                probs = torch.softmax(
                    step_logits.float() / temperatures[:, None], dim=-1
                )
                argmax_tokens = torch.argmax(step_logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return torch.where(greedy_mask, argmax_tokens, sampled_tokens)

    draft_tokens, corrected_logits = markov_head.sample_block(
        base_logits,
        first_prev_tokens=anchor_tokens,
        hidden_states=draft_hidden,
        sampler=sampler,
    )
    return DraftBlockResult(
        draft_tokens=draft_tokens,
        corrected_logits=corrected_logits,
        greedy_mask=greedy_mask,
        temperatures=temperatures,
        deterministic_draft=deterministic_draft,
    )


class DraftBlockProposer:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        gamma: int,
        mask_token_id: int,
        draft_block_spec_info,
        dp_moe_sync: bool = False,
        draft_temperature: Optional[float] = None,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.gamma = gamma
        self._mask_token_id = mask_token_id
        self._draft_block_spec_info = draft_block_spec_info
        self._draft_sampler = None
        self._sampled_graph: Optional[SampledDraftGraphRunner] = None
        self._dp_moe_sync = dp_moe_sync
        self._draft_temperature = draft_temperature
        self._segment_fn: Optional[Callable[[str], ContextManager[None]]] = None
        self._profiling_active_fn: Optional[Callable[[], bool]] = None

    def attach_draft_sampler(self, draft_sampler) -> None:
        self._draft_sampler = draft_sampler

    def attach_sampled_graph(
        self, sampled_graph: Optional[SampledDraftGraphRunner]
    ) -> None:
        self._sampled_graph = sampled_graph

    def attach_segment_fn(
        self,
        segment_fn: Callable[[str], ContextManager[None]],
        profiling_active_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._segment_fn = segment_fn
        self._profiling_active_fn = profiling_active_fn

    def _base_logits_context(self):
        if self._dp_moe_sync:
            return draft_tp_context(get_parallel().attn_tp_group)
        return nullcontext()

    def _resolve_draft_temperatures(self, *, bs, sampling_info, device):
        """Same per-request draft temperatures as ``sample_draft_block``."""
        if self._draft_temperature is not None:
            return torch.full(
                (bs,),
                max(float(self._draft_temperature), 1e-5),
                dtype=torch.float32,
                device=device,
            )
        if sampling_info is None:
            return torch.ones(bs, dtype=torch.float32, device=device)
        return sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)

    def propose(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        target_model,
        sampling_info,
    ) -> DraftProposal:
        embed_module = target_model.get_input_embeddings()
        _seg = self._segment_fn
        draft_sampler = self._draft_sampler
        deterministic_draft = self._draft_temperature == 0
        requests_all_greedy = sampling_info is None or sampling_info.is_all_greedy
        all_greedy = deterministic_draft or requests_all_greedy
        # Sampled mode: the draft graph tail runs the sampled rollout on inputs
        # staged now, before the replay inside _run_forward.
        sampled_fold = draft_sampler is not None and draft_sampler.serves_sampled(bs)
        if sampled_fold:
            temperatures = self._resolve_draft_temperatures(
                bs=bs, sampling_info=sampling_info, device=device
            )
            greedy_mask = resolve_greedy_mask(
                bs=bs, sampling_info=sampling_info, device=device
            )
            draft_sampler.stage_sampled_inputs(
                temperatures=temperatures, greedy_mask=greedy_mask.to(torch.int32)
            )
        with _seg("draft_backbone") if _seg else nullcontext():
            fwd = self._run_forward(
                batch=batch,
                draft_input=draft_input,
                verify_window=verify_window,
                bs=bs,
                device=device,
                embed_module=embed_module,
            )
        draft_block_ids = fwd.draft_block_ids

        folded_confidence = None
        confidence_tap = None
        folded = False
        sampled_folded = False
        draft_block = None
        if (
            sampled_fold
            and fwd.can_run_graph
            and not requests_all_greedy
            and not deterministic_draft
        ):
            # The rejection-sampling verifier reads q from the static logits
            # buffer within this step. draft_tokens is cloned: observers read
            # it on side streams, and the next replay overwrites ``out``.
            # ``folded`` stays False, as it gates the greedy verify epilogue.
            sampled_folded = True
            draft_block = DraftBlockResult(
                draft_tokens=draft_sampler.out[: bs * self.gamma]
                .view(bs, self.gamma)
                .clone(),
                corrected_logits=draft_sampler.sampled_logits(bs),
                greedy_mask=greedy_mask,
                temperatures=temperatures,
                deterministic_draft=False,
            )
            if draft_sampler.confidence_out is not None:
                folded_confidence = draft_sampler.confidence_out[:bs]
        elif (
            draft_sampler is not None
            and fwd.can_run_graph
            and all_greedy
            and (sampled_fold or not draft_sampler.sampled)
        ):
            # Fold (skip corrected_logits) when either:
            # 1. All requests are greedy (no rejection sampling needed), OR
            # 2. Draft is deterministic (draft_temp=0): verification uses
            #    target-only path (deterministic_draft=True → draft_probs=None),
            #    so corrected_logits are not needed regardless of target temp.
            if requests_all_greedy or deterministic_draft:
                folded = True
                temperatures = self._resolve_draft_temperatures(
                    bs=bs, sampling_info=sampling_info, device=device
                )
                draft_block = DraftBlockResult(
                    draft_tokens=draft_sampler.out[: bs * self.gamma].view(bs, self.gamma),
                    corrected_logits=None,
                    greedy_mask=(
                        torch.ones(bs, dtype=torch.bool, device=device)
                        if deterministic_draft
                        else resolve_greedy_mask(
                            bs=bs, sampling_info=sampling_info, device=device
                        )
                    ),
                    temperatures=temperatures,
                    deterministic_draft=deterministic_draft,
                )
                if draft_sampler.confidence_out is not None:
                    folded_confidence = draft_sampler.confidence_out[:bs]
        if draft_block is None:
            with self._base_logits_context(), (
                _seg("draft_base_logits") if _seg else nullcontext()
            ):
                base_logits, confidence_tap = self.draft_model.compute_base_logits(
                    fwd.raw_hidden
                )
                base_logits = base_logits.view(bs, self.gamma, -1)
            with _seg("draft_markov_sample") if _seg else nullcontext():
                draft_block = sample_draft_block(
                    base_logits=base_logits,
                    anchor_tokens=draft_block_ids[:, 0],
                    draft_hidden=fwd.draft_hidden_3d,
                    sampling_info=sampling_info,
                    markov_head=self.draft_model.markov_head,
                    device=device,
                    draft_temperature=self._draft_temperature,
                    sampled_graph=self._sampled_graph,
                )
        return DraftProposal(
            draft_block_ids=draft_block_ids,
            draft_block=draft_block,
            draft_hidden=fwd.draft_hidden_3d,
            confidence=folded_confidence,
            confidence_tap=confidence_tap,
            folded=folded,
            sampled_folded=sampled_folded,
        )

    def run_idle_participation(self, batch: ScheduleBatch) -> None:
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
        device = self.draft_model_runner.device
        empty_long = torch.empty((0,), dtype=torch.int64, device=device)
        idle_batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=empty_long,
            req_pool_indices=empty_long,
            seq_lens=empty_long,
            out_cache_loc=empty_long,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            positions=empty_long,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._fill_dp_moe_sync_metadata(idle_batch, batch)
        with torch.inference_mode():
            self.draft_model_runner.forward(idle_batch)

    def _run_forward(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        embed_module,
    ) -> DraftForwardResult:
        gamma = self.gamma
        prefix_lens = batch.seq_lens
        positions_2d = verify_window.positions_2d
        verify_cache_loc_2d = verify_window.verify_cache_loc_2d

        draft_block_ids = torch.full(
            (bs, gamma), int(self._mask_token_id), dtype=torch.long, device=device
        )
        draft_block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))
        draft_positions = positions_2d[:, :gamma].reshape(-1)
        draft_cache_loc = verify_cache_loc_2d[:, :gamma].reshape(-1)

        draft_owns_embed = hasattr(self.draft_model, "forward_embed")
        draft_input_embeds: Optional[torch.Tensor] = None
        if not draft_owns_embed:
            noise_embedding = embed_module(draft_block_ids)
            draft_input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        if batch.seq_lens_cpu is not None:
            draft_seq_lens_cpu = batch.seq_lens_cpu + gamma
            draft_seq_lens_sum = int(draft_seq_lens_cpu.sum())
        elif draft_input.reserved_seq_lens_cpu is not None:
            draft_seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
        else:
            raise RuntimeError("DSpark decode expected batch.seq_lens_cpu, got None")

        draft_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=draft_block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            out_cache_loc=draft_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=draft_seq_lens_cpu,
            positions=draft_positions,
            input_embeds=draft_input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._fill_dp_moe_sync_metadata(draft_forward_batch, batch)
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(draft_forward_batch)
        logits_output = draft_out.logits_output
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            raise RuntimeError("DSpark draft model returned no hidden states.")
        draft_hidden_3d = raw_hidden.view(bs, gamma, -1)
        return DraftForwardResult(
            draft_block_ids=draft_block_ids,
            raw_hidden=raw_hidden,
            draft_hidden_3d=draft_hidden_3d,
            can_run_graph=draft_out.can_run_graph,
        )

    def _fill_dp_moe_sync_metadata(
        self, forward_batch: ForwardBatch, batch: ScheduleBatch
    ) -> None:
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
        gnt, gnt_logprob = (
            self._draft_block_spec_info.get_spec_adjusted_global_num_tokens(batch)
        )
        device = self.draft_model_runner.device
        forward_batch.global_num_tokens_cpu = gnt
        forward_batch.global_num_tokens_for_logprob_cpu = gnt_logprob
        forward_batch.global_num_tokens_gpu = torch.tensor(gnt, dtype=torch.int64).to(
            device, non_blocking=True
        )
        forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
            gnt_logprob, dtype=torch.int64
        ).to(device, non_blocking=True)
        forward_batch.can_run_dp_cuda_graph = batch.can_run_dp_cuda_graph
