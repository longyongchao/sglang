from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)

logger = logging.getLogger(__name__)

StepSampler = Callable[[torch.Tensor, int], torch.Tensor]


def gather_and_crop_vocab(
    local_logits: torch.Tensor, lm_head: nn.Module
) -> torch.Tensor:
    full_logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
    return full_logits[..., : int(lm_head.org_vocab_size)]


def run_markov_block(
    head: nn.Module,
    base_logits: torch.Tensor,
    *,
    first_prev_tokens: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    sampler: StepSampler,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, proposal_len = base_logits.shape[:2]
    if proposal_len == 0:
        empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
        return empty, base_logits

    sampled_tokens = []
    corrected_logits = []
    prev_tokens = first_prev_tokens.long()
    for step_idx in range(proposal_len):
        step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
        step_logits = head.apply_step_logits(
            base_logits[:, step_idx, :],
            token_ids=prev_tokens,
            hidden_states=step_hidden,
        )
        next_tokens = sampler(step_logits, step_idx)
        sampled_tokens.append(next_tokens)
        corrected_logits.append(step_logits.unsqueeze(1))
        prev_tokens = next_tokens
    return (
        torch.stack(sampled_tokens, dim=1),
        torch.cat(corrected_logits, dim=1),
    )


class VanillaMarkov(nn.Module):

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
            )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids, hidden_states)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return run_markov_block(
            self,
            base_logits,
            first_prev_tokens=first_prev_tokens,
            hidden_states=hidden_states,
            sampler=sampler,
        )

    def sample_block_fused(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        scratch: "MarkovFusedScratch",
    ) -> torch.Tensor:
        """Fused greedy sampling: embed[prev_tok] @ w2.T + base -> argmax, per step.

        Returns draft_tokens [B, proposal_len]. No corrected_logits (not needed
        in the folded graph path).
        """
        from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
            fused_markov_bias_argmax,
        )

        batch_size, proposal_len = base_logits.shape[:2]
        device = base_logits.device
        if proposal_len == 0:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)

        # detach(): 同一块 storage（cuda graph 里的指针不变），但脱离 autograd。
        # decode 的 cuda graph 捕获没有关梯度（只有 prefill runner 带 @torch.no_grad），
        # 而权重仍是 requires_grad 的 Parameter，index_select 的 out= 形式在 grad mode
        # 下会直接抛错。
        embed_table = self.markov_w1.weight.detach()  # [V_emb, R]
        fc2_weight = self.markov_w2.weight   # [V, R]
        latent = scratch.latent_buf[:batch_size, : self.markov_rank]
        # index_select 不像 kernel 内的查表那样静默钳位越界 token，所以这里一次性
        # 校验 argmax 的值域（< base_logits 的词表宽度）落在 embedding 表内。
        if base_logits.shape[-1] > embed_table.shape[0]:
            raise ValueError(
                "DSpark markov embedding table is narrower than the scored vocab: "
                f"V_emb={embed_table.shape[0]}, V={base_logits.shape[-1]}."
            )

        prev_tokens = first_prev_tokens.long()
        sampled_tokens = []
        for step_idx in range(proposal_len):
            # 先把 embedding 行物化成连续的 [B, R]：交给 kernel 按 token id 现查
            # 会让同一行被该步 B*ceil(V/BLOCK_V) 个 CTA 各 gather 一次。
            torch.index_select(embed_table, 0, prev_tokens, out=latent)
            fused_markov_bias_argmax(
                latent=latent,
                fc2_weight=fc2_weight,
                base_logits=base_logits[:, step_idx, :],
                out_val=scratch.out_val[:batch_size],
                out_idx=scratch.out_idx[:batch_size],
                final_token=scratch.tok_buf[:batch_size],
            )
            prev_tokens = scratch.tok_buf[:batch_size]
            sampled_tokens.append(prev_tokens.clone())
        return torch.stack(sampled_tokens, dim=1)


class GatedMarkovHead(VanillaMarkov):

    markov_head_type = "gated"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.gate_proj = nn.Linear(int(hidden_size) + markov_rank, markov_rank)

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("GatedMarkovHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(
            dtype=prev_embeddings.dtype
        )
        return self.project_bias(gate * prev_embeddings)


class RNNHead(VanillaMarkov):

    markov_head_type = "rnn"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.hidden_size = int(hidden_size)
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(2 * markov_rank + self.hidden_size, 3 * markov_rank)

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits
        leading_shape = base_logits.shape[:-2]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            state, bias = self._rnn_step(state, prev_emb, hidden_states[..., k, :])
            output_logits.append(base_logits[..., k, :] + bias)
        return torch.stack(output_logits, dim=-2)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty = torch.empty(
                batch_size, 0, dtype=torch.long, device=base_logits.device
            )
            return empty, base_logits

        state = torch.zeros(
            batch_size,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )
        sampled_tokens = []
        corrected_logits = []
        prev_tokens = first_prev_tokens.long()
        for step_idx in range(proposal_len):
            prev_emb = self.get_prev_embeddings(prev_tokens)
            state, bias = self._rnn_step(state, prev_emb, hidden_states[:, step_idx, :])
            step_logits = base_logits[:, step_idx, :] + bias
            next_tokens = sampler(step_logits, step_idx)
            sampled_tokens.append(next_tokens)
            corrected_logits.append(step_logits.unsqueeze(1))
            prev_tokens = next_tokens
        return (
            torch.stack(sampled_tokens, dim=1),
            torch.cat(corrected_logits, dim=1),
        )

    def sample_block_fused(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        scratch: "MarkovFusedScratch",
    ) -> torch.Tensor:
        """Fused greedy RNN sampling: joint_proj (cuBLAS) + fused_rnn_cell + fused_markov_bias_argmax.

        Returns draft_tokens [B, proposal_len].
        """
        from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
            fused_dspark_rnn_cell,
            fused_markov_bias_argmax,
        )

        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        batch_size, proposal_len = base_logits.shape[:2]
        device = base_logits.device
        if proposal_len == 0:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)

        fc2_weight = self.markov_w2.weight   # [V, R]
        R = self.markov_rank

        # Initialize RNN state to zeros
        state = scratch.rnn_state_buf[:batch_size, :R].zero_()
        new_state = scratch.rnn_new_state_buf[:batch_size, :R]
        output_emb = scratch.latent_buf[:batch_size, :R]
        proj_buf = scratch.rnn_proj_buf[:batch_size]

        prev_tokens = first_prev_tokens.long()
        sampled_tokens = []
        for step_idx in range(proposal_len):
            # Build joint_proj input: cat([state, prev_emb, hidden])
            prev_emb = self.get_prev_embeddings(prev_tokens)
            joint_input = torch.cat(
                [state, prev_emb, hidden_states[:, step_idx, :]], dim=-1
            )
            # cuBLAS matmul: joint_proj linear
            proj_buf.copy_(
                torch.nn.functional.linear(
                    joint_input, self.joint_proj.weight, self.joint_proj.bias,
                )
            )
            # Fused RNN cell
            fused_dspark_rnn_cell(
                proj_out=proj_buf,
                state=state,
                new_state=new_state,
                output_emb=output_emb,
            )
            # Swap state buffers
            state, new_state = new_state, state
            # Fused argmax: output_emb @ fc2_weight.T + base_logits -> argmax
            fused_markov_bias_argmax(
                latent=output_emb,
                fc2_weight=fc2_weight,
                base_logits=base_logits[:, step_idx, :],
                out_val=scratch.out_val[:batch_size],
                out_idx=scratch.out_idx[:batch_size],
                final_token=scratch.tok_buf[:batch_size],
            )
            prev_tokens = scratch.tok_buf[:batch_size]
            sampled_tokens.append(prev_tokens.clone())
        return torch.stack(sampled_tokens, dim=1)


def build_markov_head(config) -> Optional[nn.Module]:
    # markov_rank / markov_head_type may live in the nested dspark_config dict
    # (SpecForge checkpoints) rather than as top-level HF config attributes.
    parsed = parse_dspark_draft_config(draft_hf_config=config)
    markov_rank = int(parsed.markov_rank or 0)
    if markov_rank <= 0:
        raise ValueError(
            "DSpark requires markov_rank > 0 (the Markov head is the core of the "
            f"semi-AR draft); got markov_rank={markov_rank}."
        )
    markov_head_type = str(parsed.markov_head_type or "vanilla").lower()
    vocab_size = int(config.vocab_size)
    hidden_size = int(config.hidden_size)
    if markov_head_type == "vanilla":
        return VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size
        )
    if markov_head_type == "rnn":
        return RNNHead(
            vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size
        )
    raise ValueError(f"Unsupported DSpark markov_head_type={markov_head_type!r}.")


class DSparkConfidenceHead(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        markov_rank: int,
        with_markov: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.with_markov = bool(with_markov)
        input_dim = int(hidden_size) + (int(markov_rank) if self.with_markov else 0)
        self.proj = nn.Linear(input_dim, 1, bias=bias, dtype=dtype)
        self.register_buffer(
            "sts_temperatures", torch.ones((), dtype=torch.float32), persistent=False
        )
        self._last_confidence_raw: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embed_stack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.with_markov:
            if markov_embed_stack is None:
                raise ValueError(
                    "DSparkConfidenceHead(with_markov=True) requires markov_embed_stack."
                )
            features = torch.cat(
                [hidden_states, markov_embed_stack.to(dtype=hidden_states.dtype)],
                dim=-1,
            )
        else:
            features = hidden_states
        features = features.to(dtype=self.proj.weight.dtype)
        return self.proj(features).squeeze(-1)

    def apply_sts(self, confidence_raw: torch.Tensor) -> torch.Tensor:
        self._last_confidence_raw = confidence_raw
        return torch.sigmoid(confidence_raw.float() / self.sts_temperatures)


def build_confidence_head(config) -> Optional[nn.Module]:
    if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
        return None
    from sglang.srt.speculative.dspark_components.dspark_config import (
        _get_dspark_config,
    )

    nested = _get_dspark_config(config)
    parsed = parse_dspark_draft_config(draft_hf_config=config)
    if "enable_confidence_head" not in nested and not hasattr(
        config, "enable_confidence_head"
    ):
        logger.warning(
            "DSpark draft config has no enable_confidence_head field; treating the "
            "confidence head as enabled."
        )
    hidden_size = int(config.hidden_size)
    markov_rank = int(parsed.markov_rank or 0)
    with_markov = bool(
        nested.get(
            "confidence_head_with_markov",
            getattr(config, "confidence_head_with_markov", markov_rank > 0),
        )
    )
    if with_markov and markov_rank <= 0:
        raise ValueError(
            "DSpark confidence_head_with_markov requires markov_rank > 0, "
            f"got markov_rank={markov_rank}."
        )
    return DSparkConfidenceHead(
        hidden_size=hidden_size,
        markov_rank=markov_rank,
        with_markov=with_markov,
    )


_DSPARK_SKIPPED_WEIGHT_PREFIXES = (
    "embed_tokens.",
    "lm_head.",
    "rotary_emb.",
)


class DSparkDraftMixin:

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark draft requires markov_rank > 0, "
                f"got markov_rank={dspark_config.markov_rank}."
            )
        self.gamma = int(dspark_config.resolve_gamma(default=self.block_size))
        self.markov_head = build_markov_head(config)
        self.confidence_head = build_confidence_head(config)
        self.lm_head: Optional[nn.Module] = None

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        del embed_tokens
        self.lm_head = lm_head

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.lm_head is None:
            raise ValueError(
                "DSpark dense draft requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        weight = self.lm_head.weight
        if hidden.dtype != weight.dtype:
            hidden = hidden.to(weight.dtype)
        local_logits = torch.matmul(hidden, weight.T)
        base_logits = gather_and_crop_vocab(local_logits, self.lm_head)
        return base_logits, None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        markov_weights = []
        confidence_weights = []
        backbone_weights = []
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if any(name.startswith(p) for p in _DSPARK_SKIPPED_WEIGHT_PREFIXES):
                continue
            if name.startswith("confidence_head."):
                if self.confidence_head is None:
                    continue
                confidence_weights.append((name, loaded_weight))
            elif name.startswith("markov_head."):
                markov_weights.append((name, loaded_weight))
            else:
                backbone_weights.append((name, loaded_weight))

        super().load_weights(backbone_weights)

        for name, loaded_weight in markov_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DSpark unexpected markov weight {name!r} not found in model "
                    f"parameters (known markov params require a {type(self.markov_head).__name__} head)."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        self._load_confidence_weights(
            confidence_weights=confidence_weights, params_dict=params_dict
        )

    def _load_confidence_weights(
        self,
        *,
        confidence_weights: list,
        params_dict: dict,
    ) -> None:
        if self.confidence_head is None:
            return
        loaded_names = set()
        for name, loaded_weight in confidence_weights:
            if name not in params_dict:
                raise ValueError(
                    f"DSpark unexpected confidence weight {name!r} not found in "
                    "model parameters."
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_names.add(name)

        confidence_param_names = {
            name for name in params_dict if name.startswith("confidence_head.")
        }
        missing = confidence_param_names - loaded_names
        if missing:
            raise ValueError(
                f"DSpark confidence head is enabled but the checkpoint is missing "
                f"{sorted(missing)}. Provide a checkpoint with trained confidence weights, "
                f"or disable the confidence head (enable_confidence_head=False)."
            )

    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        fused_kv_helper=None,
    ) -> None:
        """Write the target hidden states into every draft layer's KV cache.

        With ``fused_kv_helper`` (a ``FusedKVMaterializeHelper``) all layers
        share one K/V GEMM plus one fused k_norm+RoPE kernel; otherwise each
        layer runs its own projection, norm and RoPE.
        """
        ctx_hidden = self.project_target_hidden(target_hidden)
        prefix_valid = cache_loc_2d is not None and commit_lens is not None

        def write_layer_kv(layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
            attn = self.layers[layer_idx].self_attn.attn
            if prefix_valid:
                pool.set_kv_buffer_prefix_valid(
                    attn, cache_loc_2d, commit_lens, k, v, attn.k_scale, attn.v_scale
                )
            else:
                pool.set_kv_buffer(attn, cache_loc, k, v, attn.k_scale, attn.v_scale)

        if fused_kv_helper is not None:
            fused_kv_helper.materialize(
                ctx_hidden, positions, write_layer_kv=write_layer_kv
            )
            return
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            k, v = attn.kv_proj_only(ctx_hidden)
            k = attn.apply_k_norm(k)
            k = attn.apply_k_rope(positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            write_layer_kv(layer_idx, k, v)

class DSparkDraftModel(DSparkDraftMixin, DFlashDraftModel):

    pass


class Qwen3DSparkModel(DSparkDraftModel):
    pass


# DSpark checkpoints produced by SpecForge use DSparkDraftModel as their
# architecture name; Qwen3DSparkModel is retained for newer renamed configs.
EntryClass = [DSparkDraftModel, Qwen3DSparkModel]
