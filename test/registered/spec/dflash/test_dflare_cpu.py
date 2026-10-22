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
from unittest.mock import Mock

import torch

from sglang.kernels.ops.speculative.fused_kv_materialize import (
    DFlareGroupedKVMaterializeHelper,
    _fused_norm_rope_stacked,
)
from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
from sglang.srt.models.dflare import DFlareDraftModel
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
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

    def _make_fused_worker(self, projected_hidden):
        class FakeGroupedHelper:
            def __init__(self, num_layers):
                self.num_layers = num_layers
                self.ctx_hidden = None
                self.positions = None
                self.layer_kv = []

            def materialize(self, *, ctx_hidden, positions, write_layer_kv):
                self.ctx_hidden = ctx_hidden
                self.positions = positions
                for layer_idx in range(self.num_layers):
                    cache_k = torch.full(
                        (ctx_hidden.shape[0], 2), float(layer_idx + 1)
                    )
                    cache_v = torch.full(
                        (ctx_hidden.shape[0], 2), float(10 + layer_idx)
                    )
                    self.layer_kv.append((cache_k, cache_v))
                    write_layer_kv(layer_idx, cache_k, cache_v)

        attns = [
            SimpleNamespace(k_scale=0.25 + layer_idx, v_scale=0.5 + layer_idx)
            for layer_idx in range(projected_hidden.shape[1])
        ]
        draft_model = SimpleNamespace(
            supports_grouped_context_kv=True,
            project_target_hidden=Mock(return_value=projected_hidden),
            layers=[
                SimpleNamespace(self_attn=SimpleNamespace(attn=attn))
                for attn in attns
            ],
        )
        token_to_kv_pool = SimpleNamespace(
            set_kv_buffer=Mock(), set_kv_buffer_prefix_valid=Mock()
        )
        grouped_helper = FakeGroupedHelper(len(attns))

        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker.device = torch.device("cpu")
        worker.model_runner = SimpleNamespace(device=worker.device)
        worker.draft_model = draft_model
        worker.draft_model_runner = SimpleNamespace(
            token_to_kv_pool=token_to_kv_pool
        )
        worker._use_fused_kv_materialize = True
        worker._fused_kv_helper = grouped_helper
        return worker, draft_model, grouped_helper, token_to_kv_pool, attns

    def test_grouped_fused_append_writes_committed_kv(self):
        num_tokens, num_layers, hidden_size = 3, 2, 4
        target_hidden = torch.randn(num_tokens, hidden_size)
        projected_hidden = torch.randn(num_tokens, num_layers, hidden_size)
        positions = torch.tensor([4, 5, 6], dtype=torch.int64)
        cache_loc = torch.tensor([14, 15, 16], dtype=torch.int64)
        worker, draft_model, helper, pool, attns = self._make_fused_worker(
            projected_hidden
        )

        worker._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
        )

        self.assertTrue(draft_model.supports_grouped_context_kv)
        draft_model.project_target_hidden.assert_called_once_with(target_hidden)
        self.assertEqual(helper.ctx_hidden.shape, (num_tokens, num_layers, hidden_size))
        torch.testing.assert_close(helper.ctx_hidden, projected_hidden)
        torch.testing.assert_close(helper.positions, positions)
        self.assertEqual(pool.set_kv_buffer.call_count, num_layers)
        pool.set_kv_buffer_prefix_valid.assert_not_called()
        for layer_idx, call in enumerate(pool.set_kv_buffer.call_args_list):
            cache_k, cache_v = helper.layer_kv[layer_idx]
            self.assertIs(call.args[0], attns[layer_idx])
            torch.testing.assert_close(call.args[1], cache_loc)
            torch.testing.assert_close(call.args[2], cache_k)
            torch.testing.assert_close(call.args[3], cache_v)
            self.assertEqual(call.args[4], attns[layer_idx].k_scale)
            self.assertEqual(call.args[5], attns[layer_idx].v_scale)

    def test_grouped_fused_append_writes_prefix_valid_kv(self):
        num_tokens, num_layers, hidden_size = 6, 2, 4
        target_hidden = torch.randn(num_tokens, hidden_size)
        projected_hidden = torch.randn(num_tokens, num_layers, hidden_size)
        positions = torch.tensor([2, 3, 4, 8, 9, 10], dtype=torch.int64)
        cache_loc_2d = torch.tensor([[20, 21, 22], [30, 31, 32]])
        cache_loc = cache_loc_2d.flatten()
        commit_lens = torch.tensor([2, 1], dtype=torch.int32)
        worker, draft_model, helper, pool, attns = self._make_fused_worker(
            projected_hidden
        )

        worker._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
        )

        self.assertTrue(draft_model.supports_grouped_context_kv)
        draft_model.project_target_hidden.assert_called_once_with(target_hidden)
        self.assertEqual(helper.ctx_hidden.shape, (num_tokens, num_layers, hidden_size))
        torch.testing.assert_close(helper.ctx_hidden, projected_hidden)
        torch.testing.assert_close(helper.positions, positions)
        self.assertEqual(pool.set_kv_buffer_prefix_valid.call_count, num_layers)
        pool.set_kv_buffer.assert_not_called()
        for layer_idx, call in enumerate(
            pool.set_kv_buffer_prefix_valid.call_args_list
        ):
            cache_k, cache_v = helper.layer_kv[layer_idx]
            self.assertIs(call.args[0], attns[layer_idx])
            torch.testing.assert_close(call.args[1], cache_loc_2d)
            torch.testing.assert_close(call.args[2], commit_lens)
            torch.testing.assert_close(call.args[3], cache_k)
            torch.testing.assert_close(call.args[4], cache_v)
            self.assertEqual(call.args[5], attns[layer_idx].k_scale)
            self.assertEqual(call.args[6], attns[layer_idx].v_scale)

    def test_fusion_probs_cache_tracks_weight_reload(self):
        model = self.model
        weights = model.layer_fusion_weights
        original = weights.detach().clone()
        try:
            first = model._layer_fusion_probs()
            self.assertIs(model._layer_fusion_probs(), first)
            torch.testing.assert_close(first, torch.softmax(original, dim=1))
            # Reload through the regular weight loader (param.data.copy_).
            new_weights = torch.randn_like(original)
            model.load_weights(iter([("layer_fusion_weights", new_weights)]))
            torch.testing.assert_close(
                model._layer_fusion_probs(), torch.softmax(new_weights, dim=1)
            )
            # Replacing the storage (as .to()/.cuda() does) also invalidates.
            weights.data = original.clone()
            torch.testing.assert_close(
                model._layer_fusion_probs(), torch.softmax(original, dim=1)
            )
        finally:
            model.load_weights(iter([("layer_fusion_weights", original)]))

    def test_model_structure(self):
        self.assertFalse(self.model.supports_fused_context_kv)
        self.assertTrue(self.model.supports_grouped_context_kv)
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

    def test_grouped_projection_matches_per_layer_projections(self):
        num_tokens = 5
        ctx_hidden = torch.randn(num_tokens, NUM_DRAFT_LAYERS, HIDDEN_SIZE)
        first_attn = self.model.layers[0].self_attn
        helper = DFlareGroupedKVMaterializeHelper(
            layers=self.model.layers,
            rotary_emb=first_attn.rotary_emb,
            num_kv_heads=first_attn.num_kv_heads,
            head_dim=first_attn.head_dim,
            device=torch.device("cpu"),
        )
        helper._ensure_workspace(num_tokens, ctx_hidden.dtype)
        proj_out = helper._project_ctx_hidden(
            ctx_hidden,
            helper._proj_workspace[:num_tokens].view(num_tokens, NUM_DRAFT_LAYERS, -1),
        )

        for layer_idx, layer in enumerate(self.model.layers):
            k, v = layer.self_attn.kv_proj_only(ctx_hidden[:, layer_idx])
            torch.testing.assert_close(
                proj_out[:, layer_idx], torch.cat((k, v), dim=-1)
            )

        with self.assertRaisesRegex(ValueError, "Invalid ctx_hidden shape"):
            helper._validate_ctx_hidden(ctx_hidden[:, 0])

    def test_empty_grouped_materialize_skips_callback(self):
        first_attn = self.model.layers[0].self_attn
        helper = DFlareGroupedKVMaterializeHelper(
            layers=self.model.layers,
            rotary_emb=first_attn.rotary_emb,
            num_kv_heads=first_attn.num_kv_heads,
            head_dim=first_attn.head_dim,
            device=torch.device("cpu"),
        )

        def unexpected_write(*args):
            self.fail(f"write callback invoked with {args}")

        helper.materialize(
            torch.empty(0, NUM_DRAFT_LAYERS, HIDDEN_SIZE),
            torch.empty(0, dtype=torch.long),
            unexpected_write,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_norm_rope_stacked_cuda(self):
        num_tokens, num_layers = 4, 3
        num_kv_heads, head_dim, rotary_dim = 2, 8, 6
        kv_size = num_kv_heads * head_dim
        positions = torch.tensor([7, 1, 9, 4], device="cuda")

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                generator = torch.Generator(device="cuda").manual_seed(123)
                kv = torch.randn(
                    num_tokens,
                    num_layers,
                    2 * kv_size,
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                weights = torch.randn(
                    num_layers,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                eps = torch.tensor([1e-6, 2e-5, 1e-4], device="cuda")
                angles = torch.randn(
                    12,
                    rotary_dim // 2,
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                )
                cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)

                actual_k, actual_v = _fused_norm_rope_stacked(
                    kv,
                    weights,
                    eps,
                    cos_sin_cache,
                    positions,
                    num_kv_heads,
                    head_dim,
                    rotary_dim,
                )

                reference_k = []
                reference_v = []
                cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
                for layer_id in range(num_layers):
                    k = kv[:, layer_id, :kv_size].view(
                        num_tokens, num_kv_heads, head_dim
                    )
                    k32 = k.float()
                    k32 = k32 * torch.rsqrt(
                        k32.square().mean(dim=-1, keepdim=True) + eps[layer_id]
                    )
                    k32 = k32 * weights[layer_id].float()
                    first, second = k32[..., :rotary_dim].chunk(2, dim=-1)
                    rotated = torch.cat(
                        (
                            first * cos[:, None] - second * sin[:, None],
                            second * cos[:, None] + first * sin[:, None],
                            k32[..., rotary_dim:],
                        ),
                        dim=-1,
                    )
                    reference_k.append(rotated.to(dtype))
                    reference_v.append(
                        kv[:, layer_id, kv_size:].view(
                            num_tokens, num_kv_heads, head_dim
                        )
                    )

                atol, rtol = (
                    (2e-3, 2e-3)
                    if dtype == torch.float16
                    else (2e-2, 2e-2)
                )
                torch.testing.assert_close(
                    actual_k, torch.stack(reference_k), atol=atol, rtol=rtol
                )
                torch.testing.assert_close(
                    actual_v, torch.stack(reference_v), atol=0, rtol=0
                )

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
