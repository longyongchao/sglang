"""CUDA graph for the sampled DSpark draft rollout (rejection-sampling path).

When the draft samples (draft temperature > 0 and the batch is not all greedy),
the verifier needs the corrected logits q, so the folded greedy sampler cannot
be used and ``sample_draft_block`` runs the gamma-step Markov loop eagerly:
per step a linear, an add, the noise draw, two sampling kernels, plus the
stack/cat of all steps, all launched from the host. This module captures that
loop as one CUDA graph per padded batch size, with every step reduced to
``index_select`` + ``exponential_`` + ``fused_markov_bias_sample``, writing the
corrected logits in place over a static copy of the base logits.

Scope is kept to the exact eager semantics it replaces: vanilla Markov head,
the dense ``compute_base_logits`` (which stays eager outside the graph), and
the fast exponential-race sampler. Anything else returns None and the caller
stays on the eager loop.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Static buffers are held for the lifetime of the worker; cap them the same way
# as the Domino rejection-sampling rollout graph.
_SAMPLED_DRAFT_GRAPH_BUDGET_MB = 1024


class _SampledDraftBuffers(msgspec.Struct):
    max_bs: int
    # [max_bs, gamma, V]: base logits copied in before each replay, overwritten
    # in place by the corrected logits. Each kernel tile reads its base block
    # before storing to the same addresses, so the in-place update is safe.
    logits: torch.Tensor
    # inputs, copied in before each replay
    anchor_tokens: torch.Tensor  # [max_bs] int64
    temperatures: torch.Tensor  # [max_bs] fp32
    greedy_mask: torch.Tensor  # [max_bs] int32
    # outputs
    draft_tokens: torch.Tensor  # [max_bs, gamma] int64
    # scratch
    latent: torch.Tensor  # [max_bs, R]
    exp_noise: torch.Tensor  # [max_bs, V] fp32
    out_val: torch.Tensor  # [max_bs, num_v_blocks] fp32
    out_idx: torch.Tensor  # [max_bs, num_v_blocks] int32
    step_tokens: torch.Tensor  # [max_bs] int64


class SampledDraftGraphRunner:
    """Lazily captured CUDA graphs for the sampled Markov rollout.

    One set of static buffers is shared by the graphs of every padded batch
    size; it is sized for the largest padded batch seen so far and grows (with
    all graphs dropped and recaptured on demand) when a larger one arrives.
    """

    def __init__(
        self,
        *,
        markov_head,
        gamma: int,
        capture_bs: Sequence[int],
        max_bs: int,
        device: torch.device,
    ) -> None:
        from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
            MARKOV_BLOCK_V,
        )

        self.gamma = int(gamma)
        self.vocab_size = int(markov_head.markov_w2.weight.shape[0])
        self.markov_rank = int(markov_head.markov_rank)
        self.device = device
        # detach(): same storage (graph pointers stay valid), but out= ops and
        # kernel args must not see a requires_grad Parameter.
        self._embed_table = markov_head.markov_w1.weight.detach()
        self._fc2_weight = markov_head.markov_w2.weight.detach()
        self.dtype = self._fc2_weight.dtype
        self._num_v_blocks = (self.vocab_size + MARKOV_BLOCK_V - 1) // MARKOV_BLOCK_V
        self._capture_bs = sorted({int(b) for b in capture_bs if 0 < int(b) <= max_bs})
        self._buffers: Optional[_SampledDraftBuffers] = None
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._pool = None
        self._disabled = False

    @staticmethod
    def bytes_per_request(
        *, gamma: int, vocab_size: int, markov_rank: int, dtype
    ) -> int:
        from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
            MARKOV_BLOCK_V,
        )

        elt = torch.empty((), dtype=dtype).element_size()
        num_v_blocks = (vocab_size + MARKOV_BLOCK_V - 1) // MARKOV_BLOCK_V
        return (
            gamma * vocab_size * elt  # logits (base in, corrected out)
            + vocab_size * 4  # exp_noise
            + num_v_blocks * 8  # out_val + out_idx
            + markov_rank * elt  # latent
            + gamma * 8  # draft_tokens
            + 32  # anchor / temperature / greedy / step token
        )

    @property
    def max_bs(self) -> int:
        return self._capture_bs[-1] if self._capture_bs else 0

    def _padded_bs(self, bs: int) -> Optional[int]:
        for capture_bs in self._capture_bs:
            if capture_bs >= bs:
                return capture_bs
        return None

    def run(
        self,
        *,
        base_logits: torch.Tensor,
        anchor_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return ``(draft_tokens, corrected_logits)``, or None when this batch
        is not covered and the caller must run the eager loop."""
        if self._disabled or torch.cuda.is_current_stream_capturing():
            return None
        if (
            not base_logits.is_cuda
            or base_logits.device != self.device
            or base_logits.dim() != 3
            or base_logits.shape[1] != self.gamma
            or base_logits.shape[2] != self.vocab_size
            or base_logits.dtype != self.dtype
        ):
            return None
        bs = base_logits.shape[0]
        padded_bs = self._padded_bs(bs)
        if padded_bs is None:
            return None

        # Buffers are plain tensors regardless of the caller's grad mode:
        # tensors created under inference_mode cannot be updated in place
        # outside it, and the scheduler may run with either mode.
        with torch.inference_mode(False), torch.no_grad():
            if not self._ensure_buffers(padded_bs):
                return None
            graph = self._graphs.get(padded_bs)
            if graph is None:
                # Capture before the inputs are copied in: warm-up runs update
                # the logits buffer in place, so it is refilled afterwards.
                try:
                    graph = self._capture(padded_bs)
                except torch.cuda.OutOfMemoryError:
                    self._disable(f"out of memory capturing bs={padded_bs}")
                    return None
            buffers = self._buffers
            buffers.logits[:bs].copy_(base_logits)
            if padded_bs > bs:
                # Padded rows would otherwise accumulate the Markov bias across
                # replays. Their anchors / temperatures / greedy flags keep
                # earlier valid values (buffers start as zeros / ones), so they
                # can neither index out of range nor produce NaN.
                buffers.logits[bs:padded_bs].zero_()
            buffers.anchor_tokens[:bs].copy_(anchor_tokens)
            buffers.temperatures[:bs].copy_(temperatures)
            buffers.greedy_mask[:bs].copy_(greedy_mask)
            graph.replay()
            # draft_tokens is read asynchronously on side streams by the
            # observers (info dumper d2h copy, block-accept bundle), so it must
            # not alias a buffer the next replay overwrites. corrected_logits
            # is only consumed on this stream within the current step.
            return (
                buffers.draft_tokens[:bs].clone(),
                buffers.logits[:bs],
            )

    def _ensure_buffers(self, padded_bs: int) -> bool:
        if self._buffers is not None and self._buffers.max_bs >= padded_bs:
            return True
        # Growing: the old graphs bake in the old buffer pointers. Drain the
        # stream before dropping them so no in-flight replay outlives its pool.
        torch.cuda.synchronize()
        self._graphs = {}
        self._pool = None
        self._buffers = None
        try:
            self._buffers = self._allocate_buffers(padded_bs)
        except torch.cuda.OutOfMemoryError:
            self._disable(f"out of memory allocating static buffers for bs={padded_bs}")
            return False
        self._pool = torch.cuda.graph_pool_handle()
        return True

    def _disable(self, reason: str) -> None:
        self._disabled = True
        self._graphs = {}
        self._pool = None
        self._buffers = None
        logger.warning(
            "DSpark sampled draft graph disabled (%s); the sampled draft stays "
            "eager.",
            reason,
        )

    def _allocate_buffers(self, max_bs: int) -> _SampledDraftBuffers:
        device, dtype = self.device, self.dtype
        gamma, vocab_size = self.gamma, self.vocab_size
        return _SampledDraftBuffers(
            max_bs=max_bs,
            logits=torch.zeros((max_bs, gamma, vocab_size), dtype=dtype, device=device),
            anchor_tokens=torch.zeros((max_bs,), dtype=torch.int64, device=device),
            temperatures=torch.ones((max_bs,), dtype=torch.float32, device=device),
            greedy_mask=torch.ones((max_bs,), dtype=torch.int32, device=device),
            draft_tokens=torch.zeros((max_bs, gamma), dtype=torch.int64, device=device),
            latent=torch.empty((max_bs, self.markov_rank), dtype=dtype, device=device),
            exp_noise=torch.empty(
                (max_bs, vocab_size), dtype=torch.float32, device=device
            ),
            out_val=torch.empty(
                (max_bs, self._num_v_blocks), dtype=torch.float32, device=device
            ),
            out_idx=torch.empty(
                (max_bs, self._num_v_blocks), dtype=torch.int32, device=device
            ),
            step_tokens=torch.zeros((max_bs,), dtype=torch.int64, device=device),
        )

    def _capture(self, padded_bs: int) -> torch.cuda.CUDAGraph:
        # Warm-up compiles the Triton kernels for this BLOCK_B outside capture.
        # The caller copies this step's inputs in only after capture returns.
        for _ in range(2):
            self._rollout(padded_bs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            self._rollout(padded_bs)
        self._graphs[padded_bs] = graph
        logger.info(
            "DSpark sampled draft rollout captured: bs=%d gamma=%d "
            "(static buffers sized for bs=%d).",
            padded_bs,
            self.gamma,
            self._buffers.max_bs,
        )
        return graph

    def _rollout(self, padded_bs: int) -> None:
        """The captured body: ``sample_draft_block``'s eager loop on static
        buffers, one fused scoring + sampling kernel per step."""
        buffers = self._buffers
        logits = buffers.logits[:padded_bs]
        sampled_markov_rollout(
            embed_table=self._embed_table,
            fc2_weight=self._fc2_weight,
            base_logits=logits,
            logits_out=logits,
            anchor_tokens=buffers.anchor_tokens[:padded_bs],
            temperatures=buffers.temperatures[:padded_bs],
            greedy_mask=buffers.greedy_mask[:padded_bs],
            latent=buffers.latent[:padded_bs],
            exp_noise=buffers.exp_noise[:padded_bs],
            out_val=buffers.out_val[:padded_bs],
            out_idx=buffers.out_idx[:padded_bs],
            step_tokens=buffers.step_tokens[:padded_bs],
            draft_tokens=buffers.draft_tokens[:padded_bs],
        )


def sampled_markov_rollout(
    *,
    embed_table: torch.Tensor,
    fc2_weight: torch.Tensor,
    base_logits: torch.Tensor,
    logits_out: torch.Tensor,
    anchor_tokens: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    latent: torch.Tensor,
    exp_noise: torch.Tensor,
    out_val: torch.Tensor,
    out_idx: torch.Tensor,
    step_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> None:
    """Capture-safe sampled VanillaMarkov rollout over ``[B, gamma, V]`` base
    logits: per step ``index_select`` + ``exponential_`` +
    ``fused_markov_bias_sample``. Writes the corrected logits to ``logits_out``
    (may alias ``base_logits``) and the drawn tokens to ``draft_tokens``."""
    from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
        fused_markov_bias_sample,
    )

    prev_tokens = anchor_tokens
    for step_idx in range(base_logits.shape[1]):
        torch.index_select(embed_table, 0, prev_tokens, out=latent)
        # Fresh Exp(1) noise per step, as the eager fast sampler draws it;
        # inside a graph the Philox offset advances on every replay.
        exp_noise.exponential_(1)
        fused_markov_bias_sample(
            latent=latent,
            fc2_weight=fc2_weight,
            base_logits=base_logits[:, step_idx, :],
            logits_out=logits_out[:, step_idx, :],
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
            out_val=out_val,
            out_idx=out_idx,
            final_token=step_tokens,
        )
        draft_tokens[:, step_idx].copy_(step_tokens)
        prev_tokens = step_tokens


def sampled_rollout_ineligible_reason(*, draft_model, dp_moe_sync: bool) -> Optional[str]:
    """Why the sampled Markov rollout cannot replay ``draft_model``'s eager
    sampled draft (None when it can)."""
    if dp_moe_sync:
        return "dp moe sync"

    from torch import nn

    from sglang.srt.models.dspark import DSparkDraftMixin, VanillaMarkov

    # The rollout replays VanillaMarkov.apply_step_logits on top of the dense
    # compute_base_logits; subclasses (gated / rnn heads) and models with their
    # own base logits (deepseek_v4_dspark) carry different math.
    markov_head = getattr(draft_model, "markov_head", None)
    if type(markov_head) is not VanillaMarkov:
        return f"markov head {type(markov_head).__name__} is not vanilla"
    compute_base_logits = getattr(type(draft_model), "compute_base_logits", None)
    if compute_base_logits is not DSparkDraftMixin.compute_base_logits:
        return "custom compute_base_logits"
    w1, w2 = markov_head.markov_w1, markov_head.markov_w2
    if type(w1) is not nn.Embedding or type(w2) is not nn.Linear or w2.bias is not None:
        return "non-plain markov projections"
    if w1.weight.dtype != w2.weight.dtype or w1.weight.device.type != "cuda":
        return "markov weights dtype/device mismatch"
    if w1.weight.shape[0] < w2.weight.shape[0]:
        # index_select does not clamp out-of-range ids; every drafted id is
        # < vocab_size, so the table must cover that range.
        return "markov embedding narrower than vocab"
    return None


def sampled_rollout_max_bs(
    *, capture_bs: Sequence[int], gamma: int, markov_head
) -> Optional[int]:
    """Largest capture batch size whose static rollout buffers fit the budget."""
    per_request = SampledDraftGraphRunner.bytes_per_request(
        gamma=gamma,
        vocab_size=int(markov_head.markov_w2.weight.shape[0]),
        markov_rank=int(markov_head.markov_rank),
        dtype=markov_head.markov_w2.weight.dtype,
    )
    budget_bytes = _SAMPLED_DRAFT_GRAPH_BUDGET_MB * 1024 * 1024
    fitting_bs = [
        int(b) for b in capture_bs if 0 < int(b) * per_request <= budget_bytes
    ]
    return max(fitting_bs) if fitting_bs else None


def maybe_build_sampled_draft_graph(
    *,
    draft_model,
    gamma: int,
    capture_bs: Sequence[int],
    device,
    tp_rank: int,
    dp_moe_sync: bool,
    draft_temperature: Optional[float],
) -> Optional[SampledDraftGraphRunner]:
    """Build the sampled draft rollout graph runner, or return None (with the
    reason logged) so that every sampled draft stays on the eager loop."""

    def _eager(reason):
        if tp_rank == 0:
            logger.info("DSpark sampled draft rollout kept eager (reason=%s).", reason)
        return None

    if not envs.SGLANG_DSPARK_ENABLE_SAMPLED_DRAFT_GRAPH.get():
        return _eager("SGLANG_DSPARK_ENABLE_SAMPLED_DRAFT_GRAPH=0")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return _eager("not cuda")
    if gamma <= 0:
        return _eager("gamma<=0")
    if draft_temperature is not None and draft_temperature == 0:
        return _eager("deterministic draft never samples")
    reason = sampled_rollout_ineligible_reason(
        draft_model=draft_model, dp_moe_sync=dp_moe_sync
    )
    if reason is not None:
        return _eager(reason)

    markov_head = draft_model.markov_head
    max_bs = sampled_rollout_max_bs(
        capture_bs=capture_bs, gamma=gamma, markov_head=markov_head
    )
    if max_bs is None:
        return _eager("no capture bs fits the static buffer budget")
    w2 = markov_head.markov_w2
    if tp_rank == 0:
        per_request = SampledDraftGraphRunner.bytes_per_request(
            gamma=gamma,
            vocab_size=int(w2.weight.shape[0]),
            markov_rank=int(markov_head.markov_rank),
            dtype=w2.weight.dtype,
        )
        logger.info(
            "DSpark sampled draft rollout will use cuda graphs for bs<=%d "
            "(%.1f MiB static buffers per request, %d MiB budget).",
            max_bs,
            per_request / (1024 * 1024),
            _SAMPLED_DRAFT_GRAPH_BUDGET_MB,
        )
    return SampledDraftGraphRunner(
        markov_head=markov_head,
        gamma=gamma,
        capture_bs=capture_bs,
        max_bs=max_bs,
        device=w2.weight.device,
    )
