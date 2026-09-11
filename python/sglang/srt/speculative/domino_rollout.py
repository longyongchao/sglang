import logging
import os
from contextlib import nullcontext
from typing import Callable, ContextManager, Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.speculative.dflash_utils import sample_dflash_draft_tokens
from sglang.srt.speculative.domino_helper import DFlashDominoHelper
from sglang.srt.speculative.domino_kernels import (
    fused_gru_cell_from_table,
    fused_silu_fc2_argmax,
    fused_silu_fc2_argmax_with_value,
    fused_silu_fc2_candidate_argmax,
    fused_silu_fc2_candidate_argmax_with_value,
)

logger = logging.getLogger(__name__)

_DOMINO_BLOCK_V = 128
_DOMINO_BLOCK_M = 32
_DOMINO_SCORE_NUM_WARPS = 4
_DOMINO_SCORE_NUM_STAGES = 3
_DOMINO_CANDIDATE_BLOCK_C = 512
_DOMINO_CANDIDATE_POOL_SIZE = 0  # Disable candidate pool; use full-vocab scoring like DSpark
_DOMINO_TP_SCORER_CUSHION_MB = 2048
_DOMINO_TP_GRU_TABLE_CUSHION_MB = 1024


class DFlashDominoRollout:
    """Domino-specific draft-token rollout for DFLASH speculative decoding."""

    def __init__(
        self,
        *,
        domino_helper: DFlashDominoHelper,
        block_size: int,
        target_vocab_size: int,
        global_argmax_from_local_logits,
        global_argmax_from_local_max,
    ) -> None:
        self.domino_helper = domino_helper
        self.block_size = int(block_size)
        self.target_vocab_size = int(target_vocab_size)
        self.global_argmax_from_local_logits = global_argmax_from_local_logits
        self.global_argmax_from_local_max = global_argmax_from_local_max

    def _domino_rollout_draft_block_tp_eager(
        self,
        *,
        z: torch.Tensor,
        bonus_tokens: torch.Tensor,
        target_model,
        lm_head,
        org_vocab_start: int,
        num_org: int,
        num_org_padded: int,
        state: dict,
        segment_fn: Optional[Callable[[str], ContextManager]] = None,
    ) -> torch.Tensor:
        """Domino rollout for TP>1.

        Each rank scores its local vocab shard, then synchronizes the winning
        token after every step because the selected token feeds the next GRU
        state. This path keeps the TP collectives explicit while using fused
        local scoring and table-based GRU input lookup when available.
        """

        bs, num_draft, hidden_size = z.shape
        device = z.device
        _seg = segment_fn if segment_fn is not None else lambda _: nullcontext()
        weight = lm_head.weight
        weight_dtype = weight.dtype
        embed_module = target_model.get_input_embeddings()
        domino_helper = self.domino_helper
        if domino_helper is None:
            raise RuntimeError("DFLASH Domino rollout called without a Domino helper.")
        valid_vocab_size = int(
            getattr(lm_head, "org_vocab_size", 0)
            or getattr(self, "target_vocab_size", 0)
            or int(state["fc2_weight"].shape[0])
        )
        if valid_vocab_size <= 0:
            raise RuntimeError(
                f"DFLASH TP replicated scorer cannot resolve valid vocab size: {valid_vocab_size}."
            )
        if valid_vocab_size > int(state["fc2_weight"].shape[0]):
            raise RuntimeError(
                "DFLASH Domino fc2 vocab is smaller than the target org vocab: "
                f"fc2_vocab={int(state['fc2_weight'].shape[0])}, "
                f"target_vocab={valid_vocab_size}."
            )
        full_lm_head_weight = self._get_domino_tp_full_lm_head_weight(
            local_lm_head_weight=weight,
            org_vocab_start=org_vocab_start,
            num_org_padded=num_org_padded,
            valid_vocab_size=valid_vocab_size,
        )
        use_replicated_scorer = full_lm_head_weight is not None
        gru_input_table = domino_helper.get_gru_input_proj_table(embed_module.weight)
        full_gru_input_table = self._get_domino_tp_full_gru_input_table(
            local_gru_input_table=gru_input_table,
            org_vocab_start=org_vocab_start,
            num_org_padded=num_org_padded,
        )

        z_for_dtype = z.to(weight_dtype) if z.dtype != weight_dtype else z
        score_weight = full_lm_head_weight if use_replicated_scorer else weight[:num_org]
        score_vocab_size = valid_vocab_size if use_replicated_scorer else num_org
        with _seg("draft_base_logits"):
            base_logits = torch.matmul(
                z_for_dtype.reshape(bs * num_draft, hidden_size),
                score_weight.T,
            ).view(bs, num_draft, score_vocab_size)

            out = torch.empty((bs, num_draft), dtype=torch.long, device=device)
            if use_replicated_scorer:
                slot_1 = torch.argmax(base_logits[:, 0, :], dim=-1).to(torch.long)
            else:
                slot_1 = self.global_argmax_from_local_logits(
                    local_logits=base_logits[:, 0, :],
                    local_vocab_start=org_vocab_start,
                )
            out[:, 0].copy_(slot_1)

            z_proj_all = torch.nn.functional.linear(z, state["w_z"], state["b1"])

        gru_h = self._init_domino_prefix_gru_hidden(
            prefix_ids=torch.stack([bonus_tokens.to(torch.long), slot_1], dim=1),
            z_dtype=z.dtype,
            device=device,
            state=state,
            full_gru_input_table=full_gru_input_table,
            local_gru_input_table=gru_input_table,
            org_vocab_start=org_vocab_start,
            num_org=num_org,
        )

        if use_replicated_scorer:
            fc2_w, fc2_b = self._slice_domino_fc2(
                state=state,
                start=0,
                length=valid_vocab_size,
            )
        else:
            fc2_w, fc2_b = self._slice_domino_fc2(
                state=state,
                start=org_vocab_start,
                length=num_org,
            )
        candidate_pool_size = self._resolve_domino_candidate_pool_size(
            score_vocab_size,
            warn_on_disable=False,
        )
        use_fused_score = z.is_cuda
        candidate_ids = None
        candidate_token_ids = None
        candidate_fc2_w = None
        candidate_fc2_b = None
        if candidate_pool_size > 0 and num_draft > 1:
            pool_source = base_logits[:, 1:, :]
            pool_logits = pool_source.max(dim=1).values
            candidate_ids = torch.topk(
                pool_logits, k=candidate_pool_size, dim=-1
            ).indices.contiguous()
            if not use_fused_score:
                candidate_token_ids = candidate_ids.to(torch.int64)
                if not use_replicated_scorer:
                    candidate_token_ids = candidate_token_ids + int(org_vocab_start)
                candidate_fc2_w = fc2_w.index_select(0, candidate_ids.reshape(-1)).view(
                    bs, candidate_pool_size, -1
                )
                if fc2_b is not None:
                    candidate_fc2_b = fc2_b.index_select(
                        0, candidate_ids.reshape(-1)
                    ).view(bs, candidate_pool_size)

        if use_fused_score:
            block_v = _DOMINO_BLOCK_V
            block_m = _DOMINO_BLOCK_M
            score_num_warps = _DOMINO_SCORE_NUM_WARPS
            score_num_stages = _DOMINO_SCORE_NUM_STAGES
            candidate_block_c = _DOMINO_CANDIDATE_BLOCK_C
            num_score_blocks = (
                (candidate_pool_size + candidate_block_c - 1) // candidate_block_c
                if candidate_ids is not None
                else (score_vocab_size + block_v - 1) // block_v
            )
            score_val_buf = torch.empty(
                (bs, num_score_blocks), dtype=torch.float32, device=device
            )
            score_idx_buf = torch.empty(
                (bs, num_score_blocks), dtype=torch.int32, device=device
            )
            local_tok_buf = torch.empty((bs,), dtype=torch.long, device=device)
            local_val_buf = (
                None
                if use_replicated_scorer
                else torch.empty((bs,), dtype=torch.float32, device=device)
            )

        G = int(state["gru_hidden_size"])
        emb_dim = int(state["w_z"].shape[0])
        w_s_hh_T = state.get("w_s_hh_T", None)
        use_fused_gru = z.is_cuda and full_gru_input_table is not None
        gru_h_next = torch.empty_like(gru_h) if use_fused_gru else None
        with _seg("draft_gru_correct"):
            for k in range(1, num_draft):
                gh = None
                if w_s_hh_T is not None and k + 1 < num_draft:
                    sh = torch.matmul(gru_h, w_s_hh_T)
                    s_proj = sh[:, :emb_dim]
                    gh = sh[:, emb_dim : emb_dim + 3 * G]
                else:
                    s_proj = torch.nn.functional.linear(gru_h, state["w_s"], None)
                if use_fused_score:
                    if candidate_ids is None:
                        if use_replicated_scorer:
                            fused_silu_fc2_argmax(
                                z_proj=z_proj_all[:, k, :],
                                s_proj=s_proj,
                                fc2_weight=fc2_w,
                                fc2_bias=fc2_b,
                                base_logits=base_logits[:, k, :],
                                out_val=score_val_buf,
                                out_idx=score_idx_buf,
                                final_token=local_tok_buf,
                                block_v=block_v,
                                block_m=block_m,
                                num_warps=score_num_warps,
                                num_stages=score_num_stages,
                            )
                        else:
                            assert local_val_buf is not None
                            fused_silu_fc2_argmax_with_value(
                                z_proj=z_proj_all[:, k, :],
                                s_proj=s_proj,
                                fc2_weight=fc2_w,
                                fc2_bias=fc2_b,
                                base_logits=base_logits[:, k, :],
                                out_val=score_val_buf,
                                out_idx=score_idx_buf,
                                final_token=local_tok_buf,
                                final_value=local_val_buf,
                                block_v=block_v,
                                block_m=block_m,
                                num_warps=score_num_warps,
                                num_stages=score_num_stages,
                            )
                    else:
                        if use_replicated_scorer:
                            fused_silu_fc2_candidate_argmax(
                                z_proj=z_proj_all[:, k, :],
                                s_proj=s_proj,
                                fc2_weight=fc2_w,
                                fc2_bias=fc2_b,
                                base_logits=base_logits[:, k, :],
                                candidate_ids=candidate_ids,
                                out_val=score_val_buf,
                                out_idx=score_idx_buf,
                                final_token=local_tok_buf,
                                block_c=candidate_block_c,
                                block_m=block_m,
                                num_warps=score_num_warps,
                                num_stages=score_num_stages,
                            )
                        else:
                            assert local_val_buf is not None
                            fused_silu_fc2_candidate_argmax_with_value(
                                z_proj=z_proj_all[:, k, :],
                                s_proj=s_proj,
                                fc2_weight=fc2_w,
                                fc2_bias=fc2_b,
                                base_logits=base_logits[:, k, :],
                                candidate_ids=candidate_ids,
                                out_val=score_val_buf,
                                out_idx=score_idx_buf,
                                final_token=local_tok_buf,
                                final_value=local_val_buf,
                                block_c=candidate_block_c,
                                block_m=block_m,
                                num_warps=score_num_warps,
                                num_stages=score_num_stages,
                            )
                    if use_replicated_scorer:
                        tok = local_tok_buf.to(torch.long)
                    else:
                        tok = self.global_argmax_from_local_max(
                            local_max=local_val_buf,
                            global_ids=local_tok_buf.to(torch.int64) + int(org_vocab_start),
                        )
                elif candidate_ids is None:
                    mid = torch.nn.functional.silu(z_proj_all[:, k, :] + s_proj)
                    bias_local = torch.nn.functional.linear(mid, fc2_w, fc2_b)
                    logits_k = base_logits[:, k, :] + bias_local.to(base_logits.dtype)
                    if use_replicated_scorer:
                        tok = torch.argmax(logits_k, dim=-1).to(torch.long)
                    else:
                        tok = self.global_argmax_from_local_logits(
                            local_logits=logits_k,
                            local_vocab_start=org_vocab_start,
                        )
                else:
                    assert candidate_fc2_w is not None
                    assert candidate_token_ids is not None
                    mid = torch.nn.functional.silu(z_proj_all[:, k, :] + s_proj)
                    bias_local = torch.bmm(
                        candidate_fc2_w, mid.unsqueeze(-1)
                    ).squeeze(-1)
                    if candidate_fc2_b is not None:
                        bias_local = bias_local + candidate_fc2_b
                    logits_k = (
                        torch.gather(base_logits[:, k, :], 1, candidate_ids)
                        + bias_local.to(base_logits.dtype)
                    )
                    if use_replicated_scorer:
                        local_arg = torch.argmax(logits_k, dim=-1)
                        tok = torch.gather(
                            candidate_token_ids, 1, local_arg.unsqueeze(1)
                        ).view(-1)
                    else:
                        tok = self.global_argmax_from_local_logits(
                            local_logits=logits_k,
                            local_vocab_start=org_vocab_start,
                            local_token_ids=candidate_token_ids,
                        )
                out[:, k].copy_(tok)
                if k + 1 < num_draft:
                    if use_fused_gru and gh is not None:
                        assert gru_h_next is not None
                        assert full_gru_input_table is not None
                        fused_gru_cell_from_table(
                            tok_full=tok,
                            gru_input_table=full_gru_input_table,
                            gh=gh,
                            gh_bias=state["b_hh"],
                            h_state=gru_h,
                            h_out=gru_h_next,
                        )
                        gru_h, gru_h_next = gru_h_next, gru_h
                    else:
                        if full_gru_input_table is not None:
                            gi = self._domino_lookup_gru_input_proj_full(
                                token_ids=tok,
                                full_gru_input_table=full_gru_input_table,
                            )
                        else:
                            gi = self._domino_lookup_gru_input_proj_tp(
                                token_ids=tok,
                                local_gru_input_table=gru_input_table,
                                org_vocab_start=org_vocab_start,
                                num_org=num_org,
                            )
                        if gh is None:
                            gru_h = self._domino_manual_gru_step_from_input_proj(
                                gi=gi,
                                gru_h=gru_h,
                                state=state,
                            )
                        else:
                            gru_h = self._domino_manual_gru_step_from_projections(
                                gi=gi,
                                gh=gh,
                                gru_h=gru_h,
                                state=state,
                            )

        return out

    def _get_domino_tp_full_lm_head_weight(
        self,
        *,
        local_lm_head_weight: torch.Tensor,
        org_vocab_start: int,
        num_org_padded: int,
        valid_vocab_size: int,
    ) -> Optional[torch.Tensor]:
        """Replicate the target lm_head org-vocab shards for Domino TP rollout.

        Domino's GRU state depends on the exact token sampled at each draft step.
        If the scorer remains vocab-parallel, every step needs a TP global argmax
        before the next GRU step can run. Replicating the full org-vocab lm_head
        lets each rank compute the same winning token locally and removes that
        per-step collective. This is an optimization-only path and can be disabled
        with DFLASH_DOMINO_TP_REPLICATE_SCORER=0.
        """

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        if tp_size == 1:
            return local_lm_head_weight[:valid_vocab_size]

        mode = os.environ.get("DFLASH_DOMINO_TP_REPLICATE_SCORER", "auto").lower()
        if mode in {"0", "false", "off", "disable", "disabled"}:
            return None
        if mode not in {"1", "true", "on", "enable", "enabled", "auto"}:
            if not getattr(self, "_domino_tp_replicate_scorer_bad_mode_warned", False):
                logger.warning(
                    "Ignoring invalid DFLASH_DOMINO_TP_REPLICATE_SCORER=%r; "
                    "expected auto/1/0. Falling back to auto.",
                    mode,
                )
                self._domino_tp_replicate_scorer_bad_mode_warned = True
            mode = "auto"

        if num_org_padded <= 0:
            return None
        expected_org_start = int(tp_group.rank_in_group) * int(num_org_padded)
        if int(org_vocab_start) != expected_org_start:
            if not getattr(self, "_domino_tp_replicate_scorer_layout_warned", False):
                logger.warning(
                    "DFLASH TP replicated scorer disabled because vocab shard layout "
                    "is not direct padded-org concat: org_vocab_start=%d, expected=%d.",
                    int(org_vocab_start),
                    expected_org_start,
                )
                self._domino_tp_replicate_scorer_layout_warned = True
            return None

        full_padded_vocab_size = tp_size * int(num_org_padded)
        if int(valid_vocab_size) > full_padded_vocab_size:
            raise RuntimeError(
                "DFLASH TP replicated scorer valid vocab exceeds padded org vocab: "
                f"valid_vocab_size={int(valid_vocab_size)}, "
                f"full_padded_vocab_size={full_padded_vocab_size}."
            )
        if int(local_lm_head_weight.shape[0]) < int(num_org_padded):
            raise RuntimeError(
                "DFLASH TP lm_head shard is smaller than the padded org-vocab shard: "
                f"weight_rows={int(local_lm_head_weight.shape[0])}, "
                f"num_org_padded={int(num_org_padded)}."
            )

        hidden_size = int(local_lm_head_weight.shape[-1])
        local_org_weight = local_lm_head_weight[: int(num_org_padded)].contiguous()
        cache_key = (
            local_lm_head_weight.data_ptr(),
            int(num_org_padded),
            int(valid_vocab_size),
            hidden_size,
            local_lm_head_weight.dtype,
            local_lm_head_weight.device,
            tp_size,
        )
        cached = getattr(self, "_domino_tp_full_lm_head_weight_cache", None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["weight"]

        required_bytes = (
            tp_size
            * int(num_org_padded)
            * hidden_size
            * local_lm_head_weight.element_size()
        )
        if mode == "auto" and local_lm_head_weight.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(local_lm_head_weight.device)
            cushion = _DOMINO_TP_SCORER_CUSHION_MB
            if int(free_bytes) < required_bytes + cushion * 1024 * 1024:
                if not getattr(self, "_domino_tp_replicate_scorer_mem_warned", False):
                    logger.warning(
                        "DFLASH TP replicated scorer disabled by auto memory check: "
                        "need %.2f GiB + %d MiB cushion, free %.2f GiB. "
                        "Set DFLASH_DOMINO_TP_REPLICATE_SCORER=1 to force or 0 to silence.",
                        required_bytes / (1024**3),
                        cushion,
                        int(free_bytes) / (1024**3),
                    )
                    self._domino_tp_replicate_scorer_mem_warned = True
                return None

        full_padded_weight = torch.empty(
            (full_padded_vocab_size, hidden_size),
            dtype=local_lm_head_weight.dtype,
            device=local_lm_head_weight.device,
        )
        tp_group.all_gather_into_tensor(full_padded_weight, local_org_weight)
        full_weight = full_padded_weight[: int(valid_vocab_size)]
        self._domino_tp_full_lm_head_weight_cache = {
            "key": cache_key,
            "weight": full_weight,
            "full_padded_weight": full_padded_weight,
        }
        if not getattr(self, "_domino_tp_replicate_scorer_logged", False):
            logger.info(
                "DFLASH TP replicated scorer enabled. lm_head_shape=%s, memory=%.2f GiB",
                tuple(full_weight.shape),
                required_bytes / (1024**3),
            )
            self._domino_tp_replicate_scorer_logged = True
        return full_weight

    def _get_domino_tp_full_gru_input_table(
        self,
        *,
        local_gru_input_table: torch.Tensor,
        org_vocab_start: int,
        num_org_padded: int,
    ) -> Optional[torch.Tensor]:
        """Replicate the padded org-vocab GRU input table across TP ranks.

        In the TP eager rollout the selected draft token is global, so the next
        GRU input projection currently requires one all-reduce per generated
        draft token. Gathering the padded org-vocab table once lets every rank
        index by global token id directly. The gathered layout is TP-concatenated
        padded org shards, so valid base token ids map to the same row id.
        """

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        if tp_size == 1:
            return local_gru_input_table

        mode = os.environ.get("DFLASH_DOMINO_TP_REPLICATE_GRU_TABLE", "auto").lower()
        if mode in {"0", "false", "off", "disable", "disabled"}:
            return None
        if mode not in {"1", "true", "on", "enable", "enabled", "auto"}:
            if not getattr(self, "_domino_tp_replicate_gru_table_bad_mode_warned", False):
                logger.warning(
                    "Ignoring invalid DFLASH_DOMINO_TP_REPLICATE_GRU_TABLE=%r; "
                    "expected auto/1/0. Falling back to auto.",
                    mode,
                )
                self._domino_tp_replicate_gru_table_bad_mode_warned = True
            mode = "auto"

        if num_org_padded <= 0:
            return None
        expected_org_start = int(tp_group.rank_in_group) * int(num_org_padded)
        if int(org_vocab_start) != expected_org_start:
            if not getattr(self, "_domino_tp_replicate_gru_table_layout_warned", False):
                logger.warning(
                    "DFLASH TP replicated GRU table disabled because vocab shard layout "
                    "is not direct padded-org concat: org_vocab_start=%d, expected=%d.",
                    int(org_vocab_start),
                    expected_org_start,
                )
                self._domino_tp_replicate_gru_table_layout_warned = True
            return None
        if int(local_gru_input_table.shape[0]) < int(num_org_padded):
            raise RuntimeError(
                "DFLASH TP GRU input table is smaller than the padded org-vocab shard: "
                f"table_rows={int(local_gru_input_table.shape[0])}, "
                f"num_org_padded={int(num_org_padded)}."
            )

        width = int(local_gru_input_table.shape[-1])
        local_org_table = local_gru_input_table[: int(num_org_padded)].contiguous()
        cache_key = (
            local_gru_input_table.data_ptr(),
            int(num_org_padded),
            width,
            local_gru_input_table.dtype,
            local_gru_input_table.device,
            tp_size,
        )
        cached = getattr(self, "_domino_tp_full_gru_input_table_cache", None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["table"]

        required_bytes = (
            tp_size
            * int(num_org_padded)
            * width
            * local_gru_input_table.element_size()
        )
        if mode == "auto" and local_gru_input_table.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(local_gru_input_table.device)
            # Keep a modest cushion for verify/draft temporary buffers. The table
            # is an optimization, so do not risk turning a viable run into OOM.
            cushion = _DOMINO_TP_GRU_TABLE_CUSHION_MB
            if int(free_bytes) < required_bytes + cushion * 1024 * 1024:
                if not getattr(self, "_domino_tp_replicate_gru_table_mem_warned", False):
                    logger.warning(
                        "DFLASH TP replicated GRU table disabled by auto memory check: "
                        "need %.2f GiB + %d MiB cushion, free %.2f GiB. "
                        "Set DFLASH_DOMINO_TP_REPLICATE_GRU_TABLE=1 to force or 0 to silence.",
                        required_bytes / (1024**3),
                        cushion,
                        int(free_bytes) / (1024**3),
                    )
                    self._domino_tp_replicate_gru_table_mem_warned = True
                return None

        full_table = torch.empty(
            (tp_size * int(num_org_padded), width),
            dtype=local_gru_input_table.dtype,
            device=local_gru_input_table.device,
        )
        tp_group.all_gather_into_tensor(full_table, local_org_table)
        self._domino_tp_full_gru_input_table_cache = {
            "key": cache_key,
            "table": full_table,
        }
        if not getattr(self, "_domino_tp_replicate_gru_table_logged", False):
            logger.info(
                "DFLASH TP replicated GRU input table enabled. shape=%s, memory=%.2f GiB",
                tuple(full_table.shape),
                required_bytes / (1024**3),
            )
            self._domino_tp_replicate_gru_table_logged = True
        return full_table

    def _domino_lookup_gru_input_proj_full(
        self,
        *,
        token_ids: torch.Tensor,
        full_gru_input_table: torch.Tensor,
    ) -> torch.Tensor:
        flat_ids = token_ids.to(torch.long).reshape(-1)
        gi = torch.index_select(full_gru_input_table, 0, flat_ids).contiguous()
        return gi.view(*token_ids.shape, full_gru_input_table.shape[-1])

    def _domino_lookup_gru_input_proj_tp(
        self,
        *,
        token_ids: torch.Tensor,
        local_gru_input_table: torch.Tensor,
        org_vocab_start: int,
        num_org: int,
    ) -> torch.Tensor:
        """Lookup token GRU input projections from vocab shards and all-reduce."""

        if num_org <= 0:
            raise RuntimeError("DFLASH TP GRU input lookup got an empty vocab shard.")

        flat_ids = token_ids.to(torch.long).reshape(-1)
        in_local = (flat_ids >= int(org_vocab_start)) & (
            flat_ids < int(org_vocab_start + num_org)
        )
        local_idx = (flat_ids - int(org_vocab_start)).clamp_(0, int(num_org) - 1)
        local_gi = torch.index_select(
            local_gru_input_table, 0, local_idx
        ).contiguous()
        local_gi.masked_fill_(~in_local.unsqueeze(-1), 0)
        global_gi = get_tp_group().all_reduce(local_gi)
        return global_gi.view(*token_ids.shape, local_gru_input_table.shape[-1])

    def _resolve_domino_candidate_pool_size(
        self,
        vocab_size: int,
        *,
        warn_on_disable: bool,
    ) -> int:
        candidate_pool_size = max(
            0,
            int(
                os.environ.get(
                    "DFLASH_DOMINO_CANDIDATE_POOL",
                    str(_DOMINO_CANDIDATE_POOL_SIZE),
                )
            ),
        )
        if candidate_pool_size > 0 and candidate_pool_size >= int(vocab_size):
            if warn_on_disable and not getattr(
                self, "_domino_candidate_full_vocab_warned", False
            ):
                logger.warning(
                    "DFLASH_DOMINO_CANDIDATE_POOL=%d is >= local vocab size %d; "
                    "using full-vocab Domino scoring instead.",
                    candidate_pool_size,
                    int(vocab_size),
                )
                self._domino_candidate_full_vocab_warned = True
            candidate_pool_size = 0
        return candidate_pool_size

    def _slice_domino_fc2(
        self,
        *,
        state: dict,
        start: int,
        length: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        end = int(start) + int(length)
        fc2_w = state["fc2_weight"][int(start) : end].contiguous()
        fc2_b = (
            state["fc2_bias"][int(start) : end].contiguous()
            if state["fc2_bias"] is not None
            else None
        )
        return fc2_w, fc2_b

    def _init_domino_prefix_gru_hidden(
        self,
        *,
        prefix_ids: torch.Tensor,
        z_dtype: torch.dtype,
        device: torch.device,
        state: dict,
        embed_module=None,
        full_gru_input_table: Optional[torch.Tensor] = None,
        local_gru_input_table: Optional[torch.Tensor] = None,
        org_vocab_start: int = 0,
        num_org: int = 0,
    ) -> torch.Tensor:
        """Initialize the Domino prefix GRU state from [verified, first draft]."""

        if full_gru_input_table is None and local_gru_input_table is None:
            if embed_module is None:
                raise RuntimeError(
                    "DFLASH Domino prefix init requires embeddings or a GRU input table."
                )
            prefix_embeds = embed_module(prefix_ids).to(z_dtype)
            return self.domino_helper.init_gru_hidden(prefix_embeds)

        if full_gru_input_table is not None:
            prefix_gi = self._domino_lookup_gru_input_proj_full(
                token_ids=prefix_ids,
                full_gru_input_table=full_gru_input_table,
            )
        else:
            if local_gru_input_table is None:
                raise RuntimeError("DFLASH Domino prefix init missing local GRU table.")
            prefix_gi = self._domino_lookup_gru_input_proj_tp(
                token_ids=prefix_ids,
                local_gru_input_table=local_gru_input_table,
                org_vocab_start=org_vocab_start,
                num_org=num_org,
            )

        gru_h = torch.zeros(
            (int(prefix_ids.shape[0]), int(state["gru_hidden_size"])),
            dtype=prefix_gi.dtype,
            device=device,
        )
        for t_idx in range(int(prefix_ids.shape[1])):
            gru_h = self._domino_manual_gru_step_from_input_proj(
                gi=prefix_gi[:, t_idx, :],
                gru_h=gru_h,
                state=state,
            )
        return gru_h

    def _domino_manual_gru_step_from_input_proj(
        self,
        *,
        gi: torch.Tensor,
        gru_h: torch.Tensor,
        state: dict,
    ) -> torch.Tensor:
        """Run one GRU step from precomputed input projection gates."""

        G = int(state["gru_hidden_size"])
        gh = torch.nn.functional.linear(gru_h, state["w_hh"], state["b_hh"])
        r = torch.sigmoid(gi[:, :G] + gh[:, :G])
        z_gate = torch.sigmoid(gi[:, G : 2 * G] + gh[:, G : 2 * G])
        n = torch.tanh(gi[:, 2 * G :] + r * gh[:, 2 * G :])
        return (1.0 - z_gate) * n + z_gate * gru_h

    def _domino_manual_gru_step_from_projections(
        self,
        *,
        gi: torch.Tensor,
        gh: torch.Tensor,
        gru_h: torch.Tensor,
        state: dict,
    ) -> torch.Tensor:
        """Run one GRU step when both input and hidden projections are ready."""

        if state["b_hh"] is not None:
            gh = gh + state["b_hh"]
        G = int(state["gru_hidden_size"])
        r = torch.sigmoid(gi[:, :G] + gh[:, :G])
        z_gate = torch.sigmoid(gi[:, G : 2 * G] + gh[:, G : 2 * G])
        n = torch.tanh(gi[:, 2 * G :] + r * gh[:, 2 * G :])
        return (1.0 - z_gate) * n + z_gate * gru_h

    def _run_domino_rollout_loop(
        self,
        *,
        out_target: torch.Tensor,
        num_draft: int,
        org_vocab_start: int,
        gru_h: torch.Tensor,
        z_proj: torch.Tensor,
        base_logits: torch.Tensor,
        w_sh_T: torch.Tensor,
        sh_buf: torch.Tensor,
        s_proj_buf: torch.Tensor,
        gh_buf: torch.Tensor,
        h_new_buf: torch.Tensor,
        argmax_val_buf: torch.Tensor,
        argmax_idx_buf: torch.Tensor,
        local_tok_buf: torch.Tensor,
        fc2_w_shard: torch.Tensor,
        fc2_b_shard: Optional[torch.Tensor],
        b_hh_static: Optional[torch.Tensor],
        gru_input_table: torch.Tensor,
        candidate_ids: Optional[torch.Tensor],
    ) -> None:
        '''Capture-safe rollout loop over draft slots 2..block_size-1.

        Shared by the lazy standalone loop graph (eager-prologue fallback)
        and the DominoDraftSampler tail hook that folds the whole rollout
        into the draft cuda graph.
        '''

        h_state = gru_h
        for k in range(1, num_draft):
            torch.matmul(h_state, w_sh_T, out=sh_buf)
            if candidate_ids is not None:
                fused_silu_fc2_candidate_argmax(
                    z_proj=z_proj[:, k, :],
                    s_proj=s_proj_buf,
                    fc2_weight=fc2_w_shard,
                    fc2_bias=fc2_b_shard,
                    base_logits=base_logits[:, k, :],
                    candidate_ids=candidate_ids,
                    out_val=argmax_val_buf,
                    out_idx=argmax_idx_buf,
                    final_token=local_tok_buf,
                    block_c=_DOMINO_CANDIDATE_BLOCK_C,
                    block_m=_DOMINO_BLOCK_M,
                    num_warps=_DOMINO_SCORE_NUM_WARPS,
                    num_stages=_DOMINO_SCORE_NUM_STAGES,
                )
            else:
                fused_silu_fc2_argmax(
                    z_proj=z_proj[:, k, :],
                    s_proj=s_proj_buf,
                    fc2_weight=fc2_w_shard,
                    fc2_bias=fc2_b_shard,
                    base_logits=base_logits[:, k, :],
                    out_val=argmax_val_buf,
                    out_idx=argmax_idx_buf,
                    final_token=local_tok_buf,
                    block_v=_DOMINO_BLOCK_V,
                    block_m=_DOMINO_BLOCK_M,
                    num_warps=_DOMINO_SCORE_NUM_WARPS,
                    num_stages=_DOMINO_SCORE_NUM_STAGES,
                )
            tok_full = local_tok_buf + org_vocab_start
            out_target[:, k] = tok_full
            if k + 1 < num_draft:
                fused_gru_cell_from_table(
                    tok_full=tok_full,
                    gru_input_table=gru_input_table,
                    gh=gh_buf,
                    gh_bias=b_hh_static,
                    h_state=h_state,
                    h_out=h_new_buf,
                )
                h_state = h_new_buf

    def _get_or_capture_domino_loop_graph(
        self,
        *,
        bs: int,
        num_draft: int,
        emb_dim: int,
        gru_hidden: int,
        num_org: int,
        org_vocab_start: int,
        z_dtype: torch.dtype,
        logits_dtype: torch.dtype,
        device: torch.device,
        state: dict,
        gru_input_table: torch.Tensor,
        candidate_pool_size: int = 0,
    ):
        """Capture or return the CUDA graph for the Domino rollout loop."""

        pool = getattr(self, "_domino_loop_graph_pool", None)
        if pool is None:
            pool = {}
            self._domino_loop_graph_pool = pool

        candidate_pool_size = max(0, int(candidate_pool_size))
        use_candidate_pool = candidate_pool_size > 0
        pool_key = (
            bs,
            num_draft,
            emb_dim,
            gru_hidden,
            num_org,
            org_vocab_start,
            z_dtype,
            logits_dtype,
            use_candidate_pool,
            candidate_pool_size,
        )
        entry = pool.get(pool_key)
        if entry is not None:
            return entry

        G = gru_hidden
        z_proj_buf = torch.empty(
            (bs, num_draft, emb_dim), dtype=z_dtype, device=device
        )
        base_logits_buf = torch.empty(
            (bs, num_draft, num_org), dtype=logits_dtype, device=device
        )
        gru_h_buf = torch.empty((bs, G), dtype=z_dtype, device=device)
        out_buf = torch.empty((bs, num_draft), dtype=torch.long, device=device)

        block_v = _DOMINO_BLOCK_V
        candidate_block_c = _DOMINO_CANDIDATE_BLOCK_C
        num_score_blocks = (
            (candidate_pool_size + candidate_block_c - 1) // candidate_block_c
            if use_candidate_pool
            else (num_org + block_v - 1) // block_v
        )

        w_sh_T = state["w_s_hh_T"]
        sh_buf = torch.empty((bs, emb_dim + 3 * G), dtype=z_dtype, device=device)
        s_proj_buf = sh_buf[:, :emb_dim]
        gh_buf = sh_buf[:, emb_dim:]
        h_new_buf = torch.empty((bs, G), dtype=z_dtype, device=device)
        argmax_val_buf = torch.empty(
            (bs, num_score_blocks), dtype=torch.float32, device=device
        )
        argmax_idx_buf = torch.empty(
            (bs, num_score_blocks), dtype=torch.int32, device=device
        )
        candidate_ids_buf = (
            torch.empty((bs, candidate_pool_size), dtype=torch.long, device=device)
            if use_candidate_pool
            else None
        )
        if candidate_ids_buf is not None:
            candidate_ids_buf.zero_()
        local_tok_buf = torch.empty((bs,), dtype=torch.long, device=device)
        fc2_w_shard, fc2_b_shard = self._slice_domino_fc2(
            state=state,
            start=org_vocab_start,
            length=num_org,
        )
        b_hh_static = state["b_hh"]

        static_refs = [
            z_proj_buf,
            base_logits_buf,
            gru_h_buf,
            out_buf,
            sh_buf,
            h_new_buf,
            argmax_val_buf,
            argmax_idx_buf,
            local_tok_buf,
            fc2_w_shard,
            w_sh_T,
            gru_input_table,
        ]
        if fc2_b_shard is not None:
            static_refs.append(fc2_b_shard)
        if b_hh_static is not None:
            static_refs.append(b_hh_static)
        if candidate_ids_buf is not None:
            static_refs.append(candidate_ids_buf)

        def run_loop(out_target):
            self._run_domino_rollout_loop(
                out_target=out_target,
                num_draft=num_draft,
                org_vocab_start=org_vocab_start,
                gru_h=gru_h_buf,
                z_proj=z_proj_buf,
                base_logits=base_logits_buf,
                w_sh_T=w_sh_T,
                sh_buf=sh_buf,
                s_proj_buf=s_proj_buf,
                gh_buf=gh_buf,
                h_new_buf=h_new_buf,
                argmax_val_buf=argmax_val_buf,
                argmax_idx_buf=argmax_idx_buf,
                local_tok_buf=local_tok_buf,
                fc2_w_shard=fc2_w_shard,
                fc2_b_shard=fc2_b_shard,
                b_hh_static=b_hh_static,
                gru_input_table=gru_input_table,
                candidate_ids=candidate_ids_buf if use_candidate_pool else None,
            )

        warmup_out = torch.empty((bs, num_draft), dtype=torch.long, device=device)
        for _ in range(2):
            run_loop(warmup_out)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_loop(out_buf)

        entry = {
            "graph": graph,
            "z_proj_buf": z_proj_buf,
            "base_logits_buf": base_logits_buf,
            "gru_h_buf": gru_h_buf,
            "out_buf": out_buf,
            "_static_refs": static_refs,
        }
        if candidate_ids_buf is not None:
            entry["candidate_ids_buf"] = candidate_ids_buf
        pool[pool_key] = entry
        logger.warning(
            "[DFLASH Domino] captured CUDA graph (%s) for rollout loop "
            "bs=%d num_draft=%d",
            "Triton-candidate-pool" if use_candidate_pool else "Triton-fused",
            bs,
            num_draft,
        )
        return entry

    def build_capture_sampler(
        self,
        *,
        target_model,
        lm_head,
        num_org: int,
        org_vocab_start: int,
        capture_bs_list,
    ) -> Optional["DominoDraftSampler"]:
        '''Build the tail hook that folds the whole rollout into the draft
        cuda graph. The caller checks fold eligibility (tp=1, dense lm_head,
        no added vocab).'''

        num_draft = self.block_size - 1
        # The in-graph base_logits temporary dominates fold memory. It is
        # allocated inside the draft capture, so it comes from the shared
        # cuda-graph memory pool: because captures run in decreasing batch
        # size order, the pool is sized by the largest captured batch size
        # rather than by their sum. Fold every captured batch size so no
        # batch size silently falls back to the eager rollout.
        capture_bs = [int(b) for b in capture_bs_list]
        if not capture_bs:
            logger.warning(
                "[DFLASH Domino] rollout fold disabled: empty capture batch "
                "size list."
            )
            return None
        fold_max_bs = max(capture_bs)
        fold_logits_mib = (
            fold_max_bs
            * num_draft
            * int(num_org)
            * lm_head.weight.element_size()
            / (1024 * 1024)
        )
        state = self.domino_helper.get_rollout_state()
        embed_module = target_model.get_input_embeddings()
        gru_input_table = self.domino_helper.get_gru_input_proj_table(
            embed_module.weight
        )
        candidate_pool_size = self._resolve_domino_candidate_pool_size(
            int(num_org),
            warn_on_disable=True,
        )
        logger.info(
            "[DFLASH Domino] rollout folded into the draft cuda graph "
            "(fold_max_bs=%d, candidate_pool_size=%d, base_logits=%.0f MiB).",
            fold_max_bs,
            candidate_pool_size,
            fold_logits_mib,
        )
        return DominoDraftSampler(
            rollout=self,
            weight=lm_head.weight,
            num_org=int(num_org),
            org_vocab_start=int(org_vocab_start),
            state=state,
            gru_input_table=gru_input_table,
            candidate_pool_size=candidate_pool_size,
            shift_label=bool(
                getattr(self.domino_helper.draft_model, "shift_label", False)
            ),
            max_bs=fold_max_bs,
        )

    def rollout_draft_block(
        self,
        *,
        draft_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        target_model,
        lm_head,
        return_logits: bool = False,
        draft_temperature: float = 0.0,
        segment_fn: Optional[Callable[[str], ContextManager]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sequential Domino rollout to produce `block_size - 1` draft tokens.

        Algorithm mirrors dflash benchmark.py:244-264 for both
        shift_label=True and shift_label=False. SGLang's draft block reserves
        slot 0 for the current verified token and emits slots 1..block_size-1:
          1. Select z so z[:, 0] predicts slot_1 under the checkpoint's label
             alignment.
          2. slot_1 = argmax(base_logits[0]).
          3. Initialize prefix_gru on embeddings of [bonus_tokens, slot_1] to get gru_h.
          4. For k = 1..block_size-2:
                 bias = embed_proj(cat(z_k, gru_h))
                 slot_{k+1} = argmax(base_logits[k] + bias)
                 gru_h = prefix_gru_step(embed(slot_{k+1}))

        Implementation note: the per-step embed_proj/GRU calls have non-trivial
        Python + cuDNN launch overhead.  We pre-split fc1.weight along its input
        axis so z @ W_z (the part that doesn't depend on the GRU state) is
        batched outside the loop, and we replace nn.GRU(seq_len=1) with a manual
        GRU cell to skip cuDNN setup.

        Args:
            draft_hidden: [B, block_size, hidden_size] draft model output (post-norm).
            bonus_tokens: [B] current verified token per request (int64).
            target_model: the SGLang target model (for embed_tokens).
            lm_head: vocab-parallel lm_head (used to compute dense base logits).
            return_logits: if True, return per-step logits for rejection sampling.

        Returns:
            Tuple of ([B, block_size - 1] int64 draft tokens, logits or None).
            logits shape: [B, block_size - 1, vocab] when return_logits=True and TP=1.
        """
        bs, total_slots, hidden_size = draft_hidden.shape
        if total_slots != self.block_size:
            raise RuntimeError(
                f"DFLASH Domino expected draft_hidden block dim={self.block_size}, "
                f"got {total_slots}."
            )
        num_draft = self.block_size - 1  # 15 for block_size=16
        if num_draft <= 0:
            raise RuntimeError(
                f"DFLASH Domino requires block_size > 1, got {self.block_size}."
            )

        device = draft_hidden.device
        tp_size = int(get_tp_group().world_size)

        weight = lm_head.weight
        shard = lm_head.shard_indices
        num_added = int(shard.num_added_elements)
        if num_added != 0:
            raise NotImplementedError(
                "DFLASH Domino rollout does not yet handle added-vocab lm_head shards."
            )
        org_vocab_start = int(shard.org_vocab_start_index)
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        if num_org <= 0:
            raise RuntimeError("DFLASH lm_head has empty base vocab shard.")

        draft_model = self.domino_helper.draft_model
        embed_module = target_model.get_input_embeddings()
        domino_helper = self.domino_helper
        if domino_helper is None:
            raise RuntimeError("DFLASH Domino rollout called without a Domino helper.")

        # shift_label=True: draft_hidden[:, i, :] predicts token at position i+1.
        # shift_label=False: draft_hidden[:, i, :] predicts token at position i.
        # For the draft block, position 0 is bonus_tokens; positions 1..block_size-1
        # are the draft slots.  With shift_label we need draft_hidden[:, 0, :]
        # to predict slot_1, so we slice [:num_draft]; otherwise we slice [1:].
        if getattr(draft_model, "shift_label", False):
            z = draft_hidden[:, :num_draft, :].contiguous()  # [B, num_draft, hidden]
        else:
            z = draft_hidden[:, 1:, :].contiguous()  # [B, num_draft, hidden]

        state = domino_helper.get_rollout_state()
        _seg = segment_fn if segment_fn is not None else lambda _: nullcontext()
        if tp_size != 1:
            out = self._domino_rollout_draft_block_tp_eager(
                z=z,
                bonus_tokens=bonus_tokens,
                target_model=target_model,
                lm_head=lm_head,
                org_vocab_start=org_vocab_start,
                num_org=num_org,
                num_org_padded=num_org_padded,
                state=state,
                segment_fn=_seg,
            )
            # TP>1: logits are sharded, cannot return full logits cheaply.
            return out, None

        candidate_pool_size = self._resolve_domino_candidate_pool_size(
            num_org,
            warn_on_disable=True,
        )

        G = state["gru_hidden_size"]
        emb_dim = int(state["w_z"].shape[0])
        z_for_dtype = z.to(weight.dtype) if z.dtype != weight.dtype else z
        gru_input_table = domino_helper.get_gru_input_proj_table(
            embed_module.weight
        )

        # Original eager-prologue path.
        with _seg("draft_base_logits"):
            z_flat = z_for_dtype.reshape(bs * num_draft, hidden_size)
            base_logits = torch.matmul(z_flat, weight[:num_org].T).view(
                bs, num_draft, num_org
            )

            candidate_ids = None
            if candidate_pool_size > 0:
                pool_source = (
                    base_logits[:, 1:, :]
                    if int(base_logits.shape[1]) > 1 else base_logits[:, :1, :]
                )
                pool_logits = pool_source.max(dim=1).values
                candidate_ids = torch.topk(
                    pool_logits, k=candidate_pool_size, dim=-1
                ).indices.contiguous()

            slot_local_arg = torch.argmax(base_logits[:, 0, :], dim=-1)
            slot_1 = (slot_local_arg + org_vocab_start).to(torch.long)

            z_proj_all = torch.nn.functional.linear(z, state["w_z"], state["b1"])

        if return_logits:
            # Eager path: the verifier needs the actual per-step logits, which
            # the fused argmax kernels deliberately never materialize, so this
            # loop cannot be a cuda graph. Everything that does not depend on
            # the logit values is still shared with _run_domino_rollout_loop:
            # one GEMM produces both the GRU s-projection and its hidden
            # projection, and the GRU cell is a single fused kernel. A draft
            # slot therefore costs a handful of launches and, thanks to
            # sample_dflash_draft_tokens, no host synchronization.
            logits_buf = torch.empty(
                bs, num_draft, num_org, dtype=base_logits.dtype, device=device
            )
            # Slot 0: pure base_logits (no GRU bias for the first draft token).
            logits_buf[:, 0, :] = base_logits[:, 0, :]
            out = torch.empty((bs, num_draft), dtype=torch.long, device=device)
            # Sample slot_1 according to draft_temperature for consistency with RS.
            if draft_temperature > 0:
                local_id_0 = sample_dflash_draft_tokens(
                    logits=base_logits[:, 0, :],
                    draft_temperature=draft_temperature,
                )
                slot_1 = local_id_0 + org_vocab_start
            out[:, 0].copy_(slot_1)

            # Initialize GRU with the (possibly re-sampled) slot_1.
            gru_h = self._init_domino_prefix_gru_hidden(
                prefix_ids=torch.stack([bonus_tokens.to(torch.long), slot_1], dim=1),
                z_dtype=z.dtype,
                device=device,
                state=state,
                full_gru_input_table=gru_input_table,
            )

            fc2_w_shard, fc2_b_shard = self._slice_domino_fc2(
                state=state, start=org_vocab_start, length=num_org
            )
            w_sh_T = state["w_s_hh_T"]
            sh_buf = torch.empty(
                (bs, emb_dim + 3 * G), dtype=gru_h.dtype, device=device
            )
            s_proj_buf = sh_buf[:, :emb_dim]
            gh_buf = sh_buf[:, emb_dim:]
            h_new_buf = torch.empty_like(gru_h)
            with _seg("draft_gru_correct"):
                h_state = gru_h
                for k in range(1, num_draft):
                    torch.matmul(h_state, w_sh_T, out=sh_buf)
                    mid = torch.nn.functional.silu(z_proj_all[:, k, :] + s_proj_buf)
                    bias = torch.nn.functional.linear(mid, fc2_w_shard, fc2_b_shard)
                    logits_k = logits_buf[:, k, :]
                    torch.add(
                        base_logits[:, k, :],
                        bias.to(base_logits.dtype),
                        out=logits_k,
                    )
                    if draft_temperature > 0:
                        local_id_k = sample_dflash_draft_tokens(
                            logits=logits_k, draft_temperature=draft_temperature
                        )
                    else:
                        local_id_k = torch.argmax(logits_k, dim=-1).to(torch.long)
                    tok = local_id_k + org_vocab_start
                    out[:, k].copy_(tok)
                    if k + 1 < num_draft:
                        fused_gru_cell_from_table(
                            tok_full=tok,
                            gru_input_table=gru_input_table,
                            gh=gh_buf,
                            gh_bias=state["b_hh"],
                            h_state=h_state,
                            h_out=h_new_buf,
                        )
                        h_state = h_new_buf
            return out, logits_buf

        with _seg("draft_gru_correct"):
            gru_h = self._init_domino_prefix_gru_hidden(
                prefix_ids=torch.stack([bonus_tokens.to(torch.long), slot_1], dim=1),
                z_dtype=z.dtype,
                device=device,
                state=state,
                full_gru_input_table=gru_input_table,
            )

            graph_entry = self._get_or_capture_domino_loop_graph(
                bs=bs,
                num_draft=num_draft,
                emb_dim=emb_dim,
                gru_hidden=G,
                num_org=num_org,
                org_vocab_start=org_vocab_start,
                z_dtype=z.dtype,
                logits_dtype=base_logits.dtype,
                device=device,
                state=state,
                gru_input_table=gru_input_table,
                candidate_pool_size=candidate_pool_size,
            )
            graph_entry["z_proj_buf"].copy_(z_proj_all)
            graph_entry["base_logits_buf"].copy_(base_logits)
            if candidate_ids is not None:
                graph_entry["candidate_ids_buf"].copy_(candidate_ids)
            graph_entry["gru_h_buf"].copy_(gru_h)
            graph_entry["out_buf"][:, 0].copy_(slot_1)
            graph_entry["graph"].replay()
            out = graph_entry["out_buf"]
        return out, None


class DominoDraftSampler:
    '''Capture-safe Domino rollout run inside the draft cuda graph.

    Registered as a capture tail hook (mirrors
    dflash_worker_v2._DflashDraftSampler): the base-logits GEMM, candidate
    pool topk, prefix-GRU init, and the fused rollout loop are all recorded
    into the draft graph, so the per-step eager prologue and the standalone
    loop-graph replay disappear. The worker reads `out` after replay.
    tp=1 / no-added-vocab only; other setups stay eager in the worker.
    '''

    def __init__(
        self,
        *,
        rollout: DFlashDominoRollout,
        weight: torch.Tensor,
        num_org: int,
        org_vocab_start: int,
        state: dict,
        gru_input_table: torch.Tensor,
        candidate_pool_size: int,
        shift_label: bool,
        max_bs: int,
    ) -> None:
        self.rollout = rollout
        self.weight = weight
        self.block_size = int(rollout.block_size)
        self.num_draft = self.block_size - 1
        self.num_org = int(num_org)
        self.org_vocab_start = int(org_vocab_start)
        self.state = state
        self.gru_input_table = gru_input_table
        self.candidate_pool_size = max(0, int(candidate_pool_size))
        self.shift_label = bool(shift_label)
        self.max_bs = int(max_bs)
        self._warned_over_max_bs = False
        self.emb_dim = int(state["w_z"].shape[0])
        self.num_score_blocks = (
            (self.candidate_pool_size + _DOMINO_CANDIDATE_BLOCK_C - 1)
            // _DOMINO_CANDIDATE_BLOCK_C
            if self.candidate_pool_size > 0
            else (self.num_org + _DOMINO_BLOCK_V - 1) // _DOMINO_BLOCK_V
        )
        self.fc2_w_shard, self.fc2_b_shard = rollout._slice_domino_fc2(
            state=state,
            start=self.org_vocab_start,
            length=self.num_org,
        )
        # Draft tokens for slots 1..block_size-1: written in-graph, read by
        # the worker after replay.
        self.out = torch.empty(
            (self.max_bs, self.num_draft),
            dtype=torch.int64,
            device=weight.device,
        )

    def __call__(self, hidden_states: torch.Tensor, input_ids=None) -> None:
        bs = hidden_states.shape[0] // self.block_size
        if bs > self.max_bs:
            # Should be unreachable: max_bs covers every captured batch size.
            # Warn loudly because the worker silently runs the eager rollout
            # here, which changes the measured draft latency.
            if not self._warned_over_max_bs:
                self._warned_over_max_bs = True
                logger.warning(
                    "[DFLASH Domino] rollout fold skipped for bs=%d > "
                    "fold_max_bs=%d; falling back to the eager rollout.",
                    bs,
                    self.max_bs,
                )
            return
        rollout = self.rollout
        state = self.state
        num_draft = self.num_draft
        device = hidden_states.device

        hidden = hidden_states.view(bs, self.block_size, -1)
        if self.shift_label:
            z = hidden[:, :num_draft, :].contiguous()
        else:
            z = hidden[:, 1:, :].contiguous()
        z_score = z.to(self.weight.dtype) if z.dtype != self.weight.dtype else z

        base_logits = torch.matmul(
            z_score.reshape(bs * num_draft, z_score.shape[-1]),
            self.weight[: self.num_org].T,
        ).view(bs, num_draft, self.num_org)

        candidate_ids = None
        if self.candidate_pool_size > 0:
            pool_source = (
                base_logits[:, 1:, :] if num_draft > 1 else base_logits[:, :1, :]
            )
            pool_logits = pool_source.max(dim=1).values
            candidate_ids = torch.topk(
                pool_logits, k=self.candidate_pool_size, dim=-1
            ).indices.contiguous()

        slot_1 = (
            torch.argmax(base_logits[:, 0, :], dim=-1) + self.org_vocab_start
        ).to(torch.long)
        bonus_tokens = input_ids.view(bs, self.block_size)[:bs, 0].to(torch.long)

        gru_h = rollout._init_domino_prefix_gru_hidden(
            prefix_ids=torch.stack([bonus_tokens, slot_1], dim=1),
            z_dtype=z.dtype,
            device=device,
            state=state,
            full_gru_input_table=self.gru_input_table,
        )
        if gru_h.dtype != z.dtype:
            gru_h = gru_h.to(z.dtype)

        z_proj = torch.nn.functional.linear(z, state["w_z"], state["b1"])

        gru_hidden = int(state["gru_hidden_size"])
        sh_buf = torch.empty(
            (bs, self.emb_dim + 3 * gru_hidden), dtype=z.dtype, device=device
        )
        h_new_buf = torch.empty((bs, gru_hidden), dtype=z.dtype, device=device)
        argmax_val_buf = torch.empty(
            (bs, self.num_score_blocks), dtype=torch.float32, device=device
        )
        argmax_idx_buf = torch.empty(
            (bs, self.num_score_blocks), dtype=torch.int32, device=device
        )
        local_tok_buf = torch.empty((bs,), dtype=torch.long, device=device)

        out_slice = self.out[:bs]
        out_slice[:, 0].copy_(slot_1)
        rollout._run_domino_rollout_loop(
            out_target=out_slice,
            num_draft=num_draft,
            org_vocab_start=self.org_vocab_start,
            gru_h=gru_h,
            z_proj=z_proj,
            base_logits=base_logits,
            w_sh_T=state["w_s_hh_T"],
            sh_buf=sh_buf,
            s_proj_buf=sh_buf[:, : self.emb_dim],
            gh_buf=sh_buf[:, self.emb_dim :],
            h_new_buf=h_new_buf,
            argmax_val_buf=argmax_val_buf,
            argmax_idx_buf=argmax_idx_buf,
            local_tok_buf=local_tok_buf,
            fc2_w_shard=self.fc2_w_shard,
            fc2_b_shard=self.fc2_b_shard,
            b_hh_static=state["b_hh"],
            gru_input_table=self.gru_input_table,
            candidate_ids=candidate_ids,
        )
