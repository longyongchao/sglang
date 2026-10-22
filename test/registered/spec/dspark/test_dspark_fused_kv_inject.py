"""DSpark target-hidden KV inject: fused (one GEMM + fused norm/RoPE) vs per-layer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    build_fused_kv_helper,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)

HIDDEN, HEAD_DIM, HEADS, KV_HEADS, LAYERS = 64, 16, 4, 2, 3


def _model(dtype):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    cfg = SimpleNamespace(
        hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM, intermediate_size=128,
        hidden_act="silu", attention_bias=False, rms_norm_eps=1e-6,
        max_position_embeddings=512, rope_theta=10000.0, rope_scaling=None,
        num_target_layers=4, block_size=4, vocab_size=64,
        layer_types=["full_attention"] * LAYERS,
        dflash_config={"mask_token_id": 63, "target_layer_ids": [0, 2]},
    )  # fmt: skip
    torch.manual_seed(0)
    with get_parallel().override(tp_size=1, tp_rank=0):
        model = DFlashDraftModel(cfg)
    for p in model.parameters():
        p.data.normal_(0, 0.2)
    for layer in model.layers:
        layer.self_attn.k_norm.weight.data.add_(1.0)
    return model.cuda().to(dtype).eval()


class _Pool:
    """Records the K/V each layer writes, keyed by cache slot."""

    def __init__(self, n_slots):
        shape = (LAYERS, n_slots, KV_HEADS, HEAD_DIM)
        self.k = torch.zeros(shape, device="cuda", dtype=torch.float32)
        self.v = torch.zeros_like(self.k)
        self._layer = {}

    def bind(self, model):
        self._layer = {id(l.self_attn.attn): i for i, l in enumerate(model.layers)}

    def set_kv_buffer(self, attn, loc, k, v, k_scale, v_scale):
        i = self._layer[id(attn)]
        self.k[i, loc] = k.float()
        self.v[i, loc] = v.float()

    def set_kv_buffer_prefix_valid(self, attn, loc_2d, commit_lens, k, v, ks, vs):
        stride = loc_2d.shape[1]
        valid = (torch.arange(stride, device="cuda")[None] < commit_lens[:, None]).reshape(-1)
        self.set_kv_buffer(attn, loc_2d.reshape(-1)[valid], k[valid], v[valid], ks, vs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("prefix_valid", [False, True])
def test_fused_inject_matches_per_layer(dtype, prefix_valid):
    model = _model(dtype)
    helper, reason = build_fused_kv_helper(model, device="cuda", max_position_hint=512)
    assert helper is not None, reason
    bs, stride = 5, 4
    n = bs * stride
    in_dim = model.fc.in_features if hasattr(model, "fc") else HIDDEN
    target_hidden = torch.randn(n, in_dim, device="cuda", dtype=dtype)
    positions = (
        torch.randint(0, 400, (bs, 1), device="cuda") + torch.arange(stride, device="cuda")
    ).reshape(-1)
    cache_loc = torch.randperm(64, device="cuda")[:n]
    kw = dict(target_hidden=target_hidden, positions=positions, cache_loc=cache_loc)
    if prefix_valid:
        kw.update(
            cache_loc_2d=cache_loc.view(bs, stride),
            commit_lens=torch.randint(0, stride + 1, (bs,), device="cuda", dtype=torch.int32),
        )
    pools = []
    for fused in (None, helper):
        pool = _Pool(64)
        pool.bind(model)
        with torch.inference_mode():
            DSparkDraftMixin.write_target_hidden_kv(model, pool=pool, fused_kv_helper=fused, **kw)
        pools.append(pool)
    ref, got = pools
    tol = 2e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(got.k, ref.k, atol=tol, rtol=tol)
    torch.testing.assert_close(got.v, ref.v, atol=tol, rtol=tol)
    assert ref.k.abs().sum() > 0


def test_ineligible_model_reports_reason():
    attn = SimpleNamespace(qkv_proj=None, attn=SimpleNamespace(), rotary_emb=None)
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
    helper, reason = build_fused_kv_helper(model, device="cuda", max_position_hint=16)
    assert helper is None and "layer=0" in reason


def test_injector_builds_helper_before_kv_pool_exists():
    # At DSpark worker init the draft runner has no token_to_kv_pool yet.
    from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
        TargetHiddenKvInjector,
    )

    model = _model(torch.bfloat16)
    injector = TargetHiddenKvInjector(
        draft_model=model,
        draft_model_runner=SimpleNamespace(),
        model_runner=SimpleNamespace(model_config=SimpleNamespace(context_len=512)),
        device="cuda",
        verify_num_draft_tokens=4,
        block_pos_offsets=torch.arange(4, device="cuda"),
    )
    assert injector._fused_kv_helper is not None
