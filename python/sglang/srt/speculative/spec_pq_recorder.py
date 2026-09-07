"""Token-level p(x)/q(x) dump shared by all speculative decoding families.

Enabled by pointing ``SGLANG_SPEC_DUMP_PQ_PATH`` at a directory. Every
speculative worker process appends one JSONL file there, with one record per
drafted token: the drafted token id, the target log-prob ``lp``, the draft
log-prob ``lq`` (``None`` when the drafter kept no logits), and whether the
token was accepted by verification.

Each record also carries diagnostics that are free to compute once the target
row has been normalized: the drafted token's rank under the target ``rk``, the
target's own argmax token ``ta`` and its log-prob ``lpa``, the target entropy
``ent`` (the position's intrinsic difficulty, independent of the drafter), and
the drafted token's output-relative position ``pos``. When ``bonus_tokens`` is
provided, the final bonus token and its output position are attached to exactly
one draft row per request; no bonus fields are emitted otherwise.

``dent`` mirrors ``ent`` on the draft side: the entropy of the drafter's own
belief, i.e. its logits at temperature 1 rather than at the sampling
temperature. It is emitted only for rows whose draft temperature is non-zero,
because a greedy drafter keeps no distribution to measure. Drafters with a
truncated vocabulary report the entropy over the columns they actually own; the
value is comparable to ``ent`` in scale but not over an identical support.

Chain layout convention shared by EAGLE3 (topk==1), DFlash/DFlare/DPace/Domino
and DSpark: for request ``b``, drafted token ``j`` (0-based) sits at candidate
slot ``j + 1``, its target logits row is ``b * rows_per_request + j``, and it is
accepted iff ``j < num_correct_drafts[b]``. The bonus owner is the frontier row
``j == num_correct_drafts[b]`` for rejection, or the last draft row for full
acceptance; ``bonus_pos`` records the bonus position after the final draft in
the latter case.
"""

from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path
from typing import Optional

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Entropy needs one extra vocab-sized buffer per chunk on top of the normalized
# rows, so keep the chunk small enough to stay well inside the KV-cache slack.
_GATHER_CHUNK_ROWS = 128
_FLUSH_EVERY_RECORDS = 4096
# Floor for probabilities before taking a log; a truncated draft vocab leaves
# exact zeros behind.
_MIN_PROB = 1e-30


class SpecPqRecord(msgspec.Struct, omit_defaults=True):
    rid: str
    ct: int  # target forward counter
    pos: int  # index of this token in the request's output, 0-based
    i: int  # position inside the drafted chain, 0-based
    tok: int  # drafted token id
    acc: int  # 1 when this drafted token was accepted
    nc: int  # num_correct_drafts of the whole chain
    lp: float  # log p_target(tok) at the request's target temperature
    rk: int  # tokens strictly more likely than tok under p_target; 0 == argmax
    ta: int  # argmax token of p_target at this position
    lpa: float  # log p_target(ta), i.e. the target's own top-1 log-prob
    ent: float  # entropy of p_target at this position, in nats
    lq: Optional[float] = None  # log q_draft(tok); None when unavailable
    tt: float = 1.0  # target temperature used for lp (greedy reported as 1.0)
    dt: float = 0.0  # draft temperature; 0.0 means the drafter was greedy
    # entropy of the drafter's T=1 belief at this position, in nats; None when
    # the drafter was greedy or kept no distribution
    dent: Optional[float] = None
    bonus_token: Optional[int] = None  # final bonus token emitted after drafts
    bonus_pos: Optional[int] = None  # output position of the bonus token


def spec_dump_pq_path() -> str:
    return envs.SGLANG_SPEC_DUMP_PQ_PATH.get()


def spec_dump_pq_enabled() -> bool:
    return bool(spec_dump_pq_path())


def _iter_row_chunks(
    logits: torch.Tensor,
    row_indices: torch.Tensor,
    temperatures: torch.Tensor,
):
    """Yield (start, end, log_softmax(logits[rows] / temp)) chunk by chunk."""
    for start in range(0, row_indices.numel(), _GATHER_CHUNK_ROWS):
        end = start + _GATHER_CHUNK_ROWS
        rows = logits[row_indices[start:end]].to(torch.float32)
        rows = rows / temperatures[start:end, None]
        yield start, end, torch.log_softmax(rows, dim=-1)


def gather_row_logprobs(
    *,
    logits: torch.Tensor,
    row_indices: torch.Tensor,
    token_indices: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """log softmax(logits[row] / temp)[token] for every (row, token) pair."""
    chunks = []
    for start, end, logprobs in _iter_row_chunks(logits, row_indices, temperatures):
        chunks.append(
            logprobs.gather(dim=1, index=token_indices[start:end, None]).squeeze(1)
        )
    return torch.cat(chunks)


def gather_row_entropy(
    *,
    logits: torch.Tensor,
    row_indices: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Entropy in nats of softmax(logits[row] / temp) for every row."""
    chunks = [
        -(logprobs.exp() * logprobs).sum(dim=-1)
        for _, _, logprobs in _iter_row_chunks(logits, row_indices, temperatures)
    ]
    return torch.cat(chunks)


def belief_entropy_from_probs(
    *, probs: torch.Tensor, temperatures: torch.Tensor
) -> torch.Tensor:
    """Entropy in nats of the T=1 belief behind a temperature-scaled proposal.

    ``softmax(logits / T)`` pins ``logits`` down up to an additive constant, so
    ``T * log q`` is a valid logit row and ``softmax(T * log q)`` recovers the
    drafter's belief. Columns that a truncated draft vocab left at exactly zero
    keep ~zero mass and contribute nothing to the sum.
    """
    chunks = []
    for start in range(0, probs.shape[0], _GATHER_CHUNK_ROWS):
        end = start + _GATHER_CHUNK_ROWS
        rows = probs[start:end].to(torch.float32).clamp_min(_MIN_PROB)
        belief_logits = torch.log(rows) * temperatures[start:end, None]
        logprobs = torch.log_softmax(belief_logits, dim=-1)
        chunks.append(-(logprobs.exp() * logprobs).sum(dim=-1))
    return torch.cat(chunks)


class RowStats(msgspec.Struct):
    """Per-row view of the target distribution at one drafted position."""

    logp: torch.Tensor  # log p(drafted token)
    rank: torch.Tensor  # tokens strictly more likely than the drafted one
    argmax_token: torch.Tensor
    argmax_logp: torch.Tensor
    entropy: torch.Tensor  # nats


def gather_row_stats(
    *,
    logits: torch.Tensor,
    row_indices: torch.Tensor,
    token_indices: torch.Tensor,
    temperatures: torch.Tensor,
) -> RowStats:
    """gather_row_logprobs plus the reductions that share the same pass."""
    logp, rank, argmax_token, argmax_logp, entropy = [], [], [], [], []
    for start, end, logprobs in _iter_row_chunks(logits, row_indices, temperatures):
        token_logp = logprobs.gather(dim=1, index=token_indices[start:end, None])
        logp.append(token_logp.squeeze(1))
        rank.append((logprobs > token_logp).sum(dim=-1))
        top_logp, top_token = logprobs.max(dim=-1)
        argmax_token.append(top_token)
        argmax_logp.append(top_logp)
        entropy.append(-(logprobs.exp() * logprobs).sum(dim=-1))
    return RowStats(
        logp=torch.cat(logp),
        rank=torch.cat(rank),
        argmax_token=torch.cat(argmax_token),
        argmax_logp=torch.cat(argmax_logp),
        entropy=torch.cat(entropy),
    )


def _effective_temperatures(
    temperatures: Optional[torch.Tensor], *, bs: int, device: torch.device
) -> torch.Tensor:
    """Greedy rows have temperature 0; report their log-probs at T=1 instead."""
    if temperatures is None:
        return torch.ones(bs, dtype=torch.float32, device=device)
    flat = temperatures.detach().reshape(-1).to(torch.float32)
    return torch.where(flat > 0, flat, torch.ones_like(flat))


class SpecPqRecorder:
    def __init__(
        self, *, dump_dir: str, max_records: int, tag: str, tp_rank: int
    ) -> None:
        self.path = Path(dump_dir) / f"{tag}.rank{tp_rank}.pid{os.getpid()}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = int(max_records)
        self._buffer: list[bytes] = []
        self._num_records = 0
        self._warned_full = False
        atexit.register(self.flush)

    @property
    def is_full(self) -> bool:
        return self.max_records > 0 and self._num_records >= self.max_records

    def record_chain(
        self,
        *,
        forward_ct: int,
        reqs: list,  # Req objects; only .rid and .origin_input_ids are read
        prefix_lens: torch.Tensor,  # [bs] seq len before this iteration's tokens
        draft_tokens: torch.Tensor,  # [bs, num_draft]
        num_correct_drafts: torch.Tensor,  # [bs]
        target_logits: torch.Tensor,  # [bs * rows_per_request, vocab]
        rows_per_request: int,
        bonus_tokens: Optional[torch.Tensor] = None,  # [bs] final bonus tokens
        target_temperatures: Optional[torch.Tensor] = None,  # [bs] or [bs, 1]
        draft_logits: Optional[torch.Tensor] = None,  # [bs, num_draft, vocab]
        # [bs, num_draft, vocab], already-normalized proposal distribution
        draft_probs: Optional[torch.Tensor] = None,
        draft_temperature: float = 0.0,  # scalar fallback
        draft_temperatures: Optional[torch.Tensor] = None,  # [bs], wins over scalar
        draft_vocab_start: int = 0,  # draft logits column 0 == this global token id
    ) -> None:
        if self.is_full:
            if not self._warned_full:
                self._warned_full = True
                self.flush()
                logger.warning(
                    "spec p/q dump reached %d records, stopping: %s",
                    self.max_records,
                    self.path,
                )
            return

        bs = len(reqs)
        if bs == 0 or draft_tokens.numel() == 0:
            return
        num_draft = draft_tokens.shape[-1]
        device = draft_tokens.device
        draft_flat = draft_tokens.reshape(bs * num_draft).to(torch.long)

        # Candidate slot 0 holds the already-sampled bootstrap token, so the first
        # drafted token lands one past the sequence prefix.
        prefix_len_list = prefix_lens.reshape(bs).tolist()
        start_positions = [
            int(prefix_len_list[b]) - len(reqs[b].origin_input_ids) + 1
            for b in range(bs)
        ]

        target_temps = _effective_temperatures(
            target_temperatures, bs=bs, device=device
        )
        target_rows = (
            (torch.arange(bs, device=device) * int(rows_per_request))[:, None]
            + torch.arange(num_draft, device=device)[None, :]
        ).reshape(-1)
        stats = gather_row_stats(
            logits=target_logits,
            row_indices=target_rows,
            token_indices=draft_flat,
            temperatures=torch.repeat_interleave(target_temps, num_draft),
        )

        if draft_temperatures is not None:
            draft_temps = draft_temperatures.detach().reshape(bs).to(torch.float32)
        else:
            draft_temps = torch.full(
                (bs,), float(draft_temperature), dtype=torch.float32, device=device
            )
        draft_temps_full = torch.repeat_interleave(draft_temps, num_draft)
        draft_entropy: Optional[torch.Tensor] = None
        if draft_logits is not None:
            draft_flat_local = draft_flat - int(draft_vocab_start)
            draft_logits_flat = draft_logits.reshape(bs * num_draft, -1)
            draft_rows = torch.arange(bs * num_draft, device=device)
            logq = gather_row_logprobs(
                logits=draft_logits_flat,
                row_indices=draft_rows,
                token_indices=draft_flat_local,
                temperatures=torch.where(
                    draft_temps_full > 0,
                    draft_temps_full,
                    torch.ones_like(draft_temps_full),
                ),
            )
            # A greedy drafter puts all its mass on the drafted token: log q = 0.
            logq = torch.where(draft_temps_full > 0, logq, torch.zeros_like(logq))
            logq_list: list[Optional[float]] = [round(v, 6) for v in logq.tolist()]
            draft_entropy = gather_row_entropy(
                logits=draft_logits_flat,
                row_indices=draft_rows,
                temperatures=torch.ones_like(draft_temps_full),
            )
        elif draft_probs is not None:
            # Already the normalized proposal distribution that was sampled from.
            draft_probs_flat = draft_probs.reshape(bs * num_draft, -1).to(torch.float32)
            q = draft_probs_flat.gather(
                dim=1, index=(draft_flat - int(draft_vocab_start))[:, None]
            ).squeeze(1)
            logq_list = [
                round(v, 6) for v in torch.log(q.clamp_min(_MIN_PROB)).tolist()
            ]
            draft_entropy = belief_entropy_from_probs(
                probs=draft_probs_flat, temperatures=draft_temps_full
            )
        else:
            logq_list = [0.0 if t <= 0 else None for t in draft_temps_full.tolist()]

        if draft_entropy is None:
            draft_entropy_list: list[Optional[float]] = [None] * (bs * num_draft)
        else:
            # A greedy drafter has no distribution worth measuring.
            draft_entropy_list = [
                None if temp <= 0 else round(value, 6)
                for temp, value in zip(
                    draft_temps_full.tolist(), draft_entropy.tolist()
                )
            ]

        logp_list = [round(v, 6) for v in stats.logp.tolist()]
        rank_list = stats.rank.tolist()
        argmax_token_list = stats.argmax_token.tolist()
        argmax_logp_list = [round(v, 6) for v in stats.argmax_logp.tolist()]
        entropy_list = [round(v, 6) for v in stats.entropy.tolist()]
        token_list = draft_flat.tolist()
        num_correct_list = num_correct_drafts.reshape(bs).tolist()
        bonus_token_list = (
            None if bonus_tokens is None else bonus_tokens.reshape(bs).tolist()
        )
        target_temp_list = target_temps.tolist()
        draft_temp_list = draft_temps.tolist()

        for b in range(bs):
            num_correct = int(num_correct_list[b])
            bonus_owner = (
                None if bonus_token_list is None else min(num_correct, num_draft - 1)
            )
            for j in range(num_draft):
                flat_idx = b * num_draft + j
                has_bonus = j == bonus_owner
                self._buffer.append(
                    msgspec.json.encode(
                        SpecPqRecord(
                            rid=str(reqs[b].rid),
                            ct=int(forward_ct),
                            pos=start_positions[b] + j,
                            i=j,
                            tok=int(token_list[flat_idx]),
                            acc=1 if j < num_correct else 0,
                            nc=num_correct,
                            lp=logp_list[flat_idx],
                            rk=int(rank_list[flat_idx]),
                            ta=int(argmax_token_list[flat_idx]),
                            lpa=argmax_logp_list[flat_idx],
                            ent=entropy_list[flat_idx],
                            lq=logq_list[flat_idx],
                            tt=round(float(target_temp_list[b]), 6),
                            dt=round(float(draft_temp_list[b]), 6),
                            dent=draft_entropy_list[flat_idx],
                            bonus_token=(
                                int(bonus_token_list[b]) if has_bonus else None
                            ),
                            bonus_pos=(
                                start_positions[b] + num_correct if has_bonus else None
                            ),
                        )
                    )
                )
        self._num_records += bs * num_draft
        if len(self._buffer) >= _FLUSH_EVERY_RECORDS:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        payload = b"\n".join(self._buffer) + b"\n"
        self._buffer.clear()
        with open(self.path, "ab") as f:
            f.write(payload)


def create_spec_pq_recorder(*, tag: str, tp_rank: int) -> Optional[SpecPqRecorder]:
    dump_dir = spec_dump_pq_path()
    if not dump_dir:
        return None
    recorder = SpecPqRecorder(
        dump_dir=dump_dir,
        max_records=envs.SGLANG_SPEC_DUMP_PQ_MAX_RECORDS.get(),
        tag=tag,
        tp_rank=tp_rank,
    )
    logger.info("spec p/q dump enabled, writing to %s", recorder.path)
    return recorder
