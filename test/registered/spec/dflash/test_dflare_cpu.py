"""CPU numerics tests for the DFlare draft model (sglang.srt.models.dflare).

Checks that the sglang-side context-KV materialization pipeline
(project_target_hidden -> per-layer prepare_context_hidden_for_kv ->
kv_proj_only -> apply_k_norm) matches a reference computation implemented
inline with training-side semantics (SpecForge specforge/modeling/draft/dflare.py:
softmax layer fusion + hidden_norm + dedicated k_proj_target/v_proj_target,
with k_norm shared between ctx and noise K). RoPE and KV-pool writes are
shared DFlash code and are not re-checked here.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
from sglang.srt.models.dflare import DFlareDraftModel
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN_SIZE = 16
HEAD_DIM = 4
NUM_HEADS = 4
NUM_KV_HEADS = 2
NUM_DRAFT_LAYERS = 2
TARGET_LAYER_IDS = [0, 2, 4]  # T = 3 context features
RMS_NORM_EPS = 1e-6


def _config():
    return SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_DRAFT_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=32,
        hidden_act="silu",
        attention_bias=False,
        rms_norm_eps=RMS_NORM_EPS,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling=None,
        num_target_layers=5,
        block_size=8,
        vocab_size=64,
        layer_types=["full_attention"] * NUM_DRAFT_LAYERS,
        dflash_config={
            "mask_token_id": 63,
            "target_layer_ids": TARGET_LAYER_IDS,
        },
    )


def _checkpoint_weights(generator):
    """Random weights using the training checkpoint's key names."""

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.1

    weights = {
        "layer_fusion_weights": randn(NUM_DRAFT_LAYERS, len(TARGET_LAYER_IDS)),
        "hidden_norm.weight": 1.0 + randn(HIDDEN_SIZE),
        "norm.weight": 1.0 + randn(HIDDEN_SIZE),
    }
    q_size = NUM_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    for i in range(NUM_DRAFT_LAYERS):
        prefix = f"layers.{i}."
        weights[prefix + "self_attn.q_proj.weight"] = randn(q_size, HIDDEN_SIZE)
        weights[prefix + "self_attn.k_proj.weight"] = randn(kv_size, HIDDEN_SIZE)
        weights[prefix + "self_attn.v_proj.weight"] = randn(kv_size, HIDDEN_SIZE)
        weights[prefix + "self_attn.k_proj_target.weight"] = randn(kv_size, HIDDEN_SIZE)
        weights[prefix + "self_attn.v_proj_target.weight"] = randn(kv_size, HIDDEN_SIZE)
        weights[prefix + "self_attn.o_proj.weight"] = randn(HIDDEN_SIZE, q_size)
        weights[prefix + "self_attn.q_norm.weight"] = 1.0 + randn(HEAD_DIM)
        weights[prefix + "self_attn.k_norm.weight"] = 1.0 + randn(HEAD_DIM)
        weights[prefix + "mlp.gate_proj.weight"] = randn(32, HIDDEN_SIZE)
        weights[prefix + "mlp.up_proj.weight"] = randn(32, HIDDEN_SIZE)
        weights[prefix + "mlp.down_proj.weight"] = randn(HIDDEN_SIZE, 32)
        weights[prefix + "input_layernorm.weight"] = 1.0 + randn(HIDDEN_SIZE)
        weights[prefix + "post_attention_layernorm.weight"] = 1.0 + randn(HIDDEN_SIZE)
    return weights


def _rms_norm(x, weight, eps=RMS_NORM_EPS):
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    return (x32 * torch.rsqrt(variance + eps) * weight.to(torch.float32)).to(x.dtype)


class TestDFlareContextKV(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # RotaryEmbedding reads global server args during construction.
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(0)
        with get_parallel().override(tp_size=1, tp_rank=0):
            cls.model = DFlareDraftModel(_config())
        # Platform dispatch follows the installed torch build, but all test
        # tensors live on CPU; force the native math path (e.g. on CUDA hosts).
        for module in cls.model.modules():
            if isinstance(module, MultiPlatformOp):
                module._forward_method = module.forward_native
        generator = torch.Generator().manual_seed(42)
        cls.weights = _checkpoint_weights(generator)
        cls.model.load_weights(iter(cls.weights.items()))

    def test_model_structure(self):
        self.assertFalse(self.model.supports_fused_context_kv)
        self.assertFalse(hasattr(self.model, "fc"))
        state_dict = self.model.state_dict()
        self.assertIn("layer_fusion_weights", state_dict)
        self.assertIn("layers.0.self_attn.k_proj_target.weight", state_dict)
        self.assertIn("layers.0.self_attn.v_proj_target.weight", state_dict)
        # The noise-path fused qkv_proj must be untouched by *_proj_target keys.
        qkv = self.model.layers[0].self_attn.qkv_proj.weight
        expected_qkv = torch.cat(
            [
                self.weights["layers.0.self_attn.q_proj.weight"],
                self.weights["layers.0.self_attn.k_proj.weight"],
                self.weights["layers.0.self_attn.v_proj.weight"],
            ],
            dim=0,
        )
        torch.testing.assert_close(qkv, expected_qkv)

    def test_dflash_base_hook_is_identity(self):
        ctx_hidden = torch.randn(3, HIDDEN_SIZE)
        out = DFlashDraftModel.prepare_context_hidden_for_kv(
            self.model, self.model.layers[0], ctx_hidden
        )
        self.assertIs(out, ctx_hidden)

    def test_context_kv_matches_training_semantics(self):
        num_tokens = 6
        num_ctx_features = len(TARGET_LAYER_IDS)
        target_hidden = torch.randn(num_tokens, num_ctx_features * HIDDEN_SIZE)

        # --- sglang side: project + per-layer kv_proj_only + k_norm ---
        ctx_hidden = self.model.project_target_hidden(target_hidden)
        self.assertEqual(
            tuple(ctx_hidden.shape), (num_tokens, NUM_DRAFT_LAYERS, HIDDEN_SIZE)
        )
        sglang_kv = []
        for layer in self.model.layers:
            attn = layer.self_attn
            layer_ctx_hidden = self.model.prepare_context_hidden_for_kv(
                layer, ctx_hidden
            )
            k, v = attn.kv_proj_only(layer_ctx_hidden)
            k = attn.apply_k_norm(k)
            sglang_kv.append((k, v))

        # --- reference: training-side semantics, inline ---
        stacked = target_hidden.view(num_tokens, num_ctx_features, HIDDEN_SIZE)
        fusion_probs = torch.softmax(self.weights["layer_fusion_weights"], dim=1)
        fused = torch.einsum("nth,dt->ndh", stacked, fusion_probs)
        fused = _rms_norm(fused, self.weights["hidden_norm.weight"])
        for layer_id, (k, v) in enumerate(sglang_kv):
            layer_hidden = fused[:, layer_id, :]
            prefix = f"layers.{layer_id}."
            k_ref = (
                layer_hidden @ self.weights[prefix + "self_attn.k_proj_target.weight"].T
            )
            v_ref = (
                layer_hidden @ self.weights[prefix + "self_attn.v_proj_target.weight"].T
            )
            k_ref = _rms_norm(
                k_ref.view(-1, HEAD_DIM),
                self.weights[prefix + "self_attn.k_norm.weight"],
            ).view(num_tokens, NUM_KV_HEADS * HEAD_DIM)
            torch.testing.assert_close(k, k_ref, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(v, v_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
