# DFlare draft model: a DFlash variant with two structural changes, both only
# affecting how target-context hidden states are materialized into the draft
# KV cache (the noise-token forward path is identical to DFlash):
#   1. Per-layer context fusion: instead of one shared fc(T*H -> H) projection,
#      a learnable layer_fusion_weights[D, T] matrix (softmax over T) fuses the
#      T captured target layers into a distinct ctx hidden per draft layer.
#   2. Separate context K/V projections: ctx K/V use dedicated
#      k_proj_target / v_proj_target weights; noise K/V keep using qkv_proj.

from __future__ import annotations

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.models.dflash import (
    DFlashAttention,
    DFlashDecoderLayer,
    DFlashDraftModel,
    DFlashMLP,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu


class DFlareAttention(DFlashAttention):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__(config=config, layer_id=layer_id)
        if is_npu():
            raise NotImplementedError(
                "DFlare is not supported on NPU: the NPU context-KV path "
                "(forward_prepare_npu) projects ctx hidden with qkv_proj, which is "
                "incorrect for DFlare's separate k_proj_target/v_proj_target."
            )
        tp_size = int(get_parallel().tp_size)
        if tp_size > self.total_num_kv_heads or self.total_num_kv_heads % tp_size != 0:
            raise NotImplementedError(
                "DFlare k_proj_target/v_proj_target require tp_size to divide "
                f"total_num_kv_heads (kv-head replication is not supported). "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )

        attention_bias = bool(getattr(config, "attention_bias", False))
        # Output dim is contiguous per kv head, so ColumnParallelLinear's even
        # split matches QKVParallelLinear's KV shard layout.
        self.k_proj_target = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
            prefix="k_proj_target",
        )
        self.v_proj_target = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
            prefix="v_proj_target",
        )

    def kv_proj_only(self, hidden_states: torch.Tensor):
        """Project ctx hidden to K/V with the dedicated context projections.

        k_norm / RoPE / KV-pool writes are shared with the parent (they match
        training, which applies k_norm to concat(ctx, noise) uniformly).
        """
        k, _ = self.k_proj_target(hidden_states)
        v, _ = self.v_proj_target(hidden_states)
        return k, v


class DFlareDecoderLayer(DFlashDecoderLayer):
    # Same as DFlashDecoderLayer.__init__ but with DFlareAttention; skip the
    # parent __init__ to avoid constructing a throwaway DFlashAttention.
    def __init__(self, config, layer_id: int) -> None:
        nn.Module.__init__(self)
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DFlareAttention(config=config, layer_id=layer_id)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(config=config)


class DFlareDraftModel(DFlashDraftModel):
    # The fused KV materialization kernel assumes one shared ctx hidden and
    # slices K/V weights from qkv_proj; neither holds for DFlare. The worker
    # reads this flag and falls back to the per-layer eager path.
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        num_layers = int(config.num_hidden_layers)
        self.layers = nn.ModuleList(
            [DFlareDecoderLayer(config=config, layer_id=i) for i in range(num_layers)]
        )
        # DFlare checkpoints have no fc weight; per-layer fusion replaces it.
        del self.fc
        self.layer_fusion_weights = nn.Parameter(
            torch.empty(num_layers, self.num_context_features)
        )

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Fuse concatenated target-layer hidden states per draft layer.

        Input [N, T*H]; output [N, D, H] (one ctx hidden per draft layer).
        """
        hidden_size = int(self.config.hidden_size)
        expected = self.num_context_features * hidden_size
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={hidden_size}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        stacked = target_hidden.view(-1, self.num_context_features, hidden_size)
        fusion_probs = torch.softmax(self.layer_fusion_weights, dim=1)
        fused = torch.einsum("nth,dt->ndh", stacked, fusion_probs)
        return self.hidden_norm(fused)

    def prepare_context_hidden_for_kv(
        self, layer: DFlareDecoderLayer, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        return ctx_hidden[:, layer.self_attn.attn.layer_id, :]


EntryClass = DFlareDraftModel
