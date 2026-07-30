"""Domino 打分 kernel（BLOCK_B 批分块 + tl.dot）与 eager 参考的一致性。

重点覆盖批分块引入的新失败模式：
- B 跨多个批 tile。注意 ``_score_block_b`` 随 bs 变化（<=16 -> 16，<=32 -> 32，
  否则 64），所以 grid dim0 要到 3 才需要 B >= 129；小于这个值时把 ``offs_b``
  的 tile 偏移写死也测不出来。
- strided 的 z_proj / base_logits——production 传的是 ``[:, k, :]`` 切片
- V 末尾的 ragged tile
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.speculative.domino_kernels import (
    fused_silu_fc2_argmax,
    fused_silu_fc2_argmax_with_value,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
BLOCK_V = 128
NUM_DRAFT = 4  # 模拟 production 的 z_proj_all[:, k, :] 布局
STEP_K = 2

# 129 是必须的：grid dim0 = ceil(B / 64)，只有到 3 个批 tile 才会暴露
# offs_b 的 tile 偏移错误（B=65 时 pid_b=0 会把 pid_b=1 该算的行重算一遍）。
BATCH_SIZES = (1, 16, 17, 40, 64, 65, 129)


def _make(bs, m, v, dtype, *, with_bias, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def rand(*shape, d=dtype):
        return torch.randn(*shape, generator=g, device=DEVICE, dtype=d)

    z_all = rand(bs, NUM_DRAFT, m)
    base_all = rand(bs, NUM_DRAFT, v)
    return {
        "z_proj": z_all[:, STEP_K, :],
        "s_proj": rand(bs, m),
        "fc2_weight": rand(v, m).mul_(0.05),
        "fc2_bias": rand(v, d=torch.float32) if with_bias else None,
        "base_logits": base_all[:, STEP_K, :],
    }


def _scratch(bs, v):
    num_v_blocks = (v + BLOCK_V - 1) // BLOCK_V
    return {
        "out_val": torch.empty((bs, num_v_blocks), dtype=torch.float32, device=DEVICE),
        "out_idx": torch.empty((bs, num_v_blocks), dtype=torch.int32, device=DEVICE),
        "final_token": torch.empty((bs,), dtype=torch.int64, device=DEVICE),
    }


def _reference(args):
    """kernel 的精度边界：mid 落回 fc2 精度，其后在 fp32 里累加。"""
    z = args["z_proj"].float()
    s = args["s_proj"].float()
    mid = torch.nn.functional.silu(z + s).to(args["fc2_weight"].dtype)
    ref = mid.float() @ args["fc2_weight"].float().t()
    if args["fc2_bias"] is not None:
        ref = ref + args["fc2_bias"].float()
    return ref + args["base_logits"].float()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton kernels")
class TestDominoScoreKernel(CustomTestCase):
    def test_planted_argmax_is_exact(self):
        """在 base_logits 里种一个远离其他值的峰值，argmax 结果必须逐位一致。

        与 MMA 舍入无关，专门验证批分块 / 掩码 / stride / tile 索引算术，
        以及 pass-2 reduce 是否覆盖了全部 num_v_blocks。
        """
        m, v = 256, 8200  # v 不是 BLOCK_V 的整数倍 -> 末尾 ragged tile
        for dtype in (torch.bfloat16, torch.float16):
            for bs in BATCH_SIZES:
                for with_bias in (False, True):
                    with self.subTest(dtype=dtype, bs=bs, with_bias=with_bias):
                        args = _make(bs, m, v, dtype, with_bias=with_bias, seed=bs)
                        planted = torch.randint(
                            0, v, (bs,), device=DEVICE, dtype=torch.int64
                        )
                        rows = torch.arange(bs, device=DEVICE)
                        args["base_logits"][rows, planted] = 1000.0
                        buf = _scratch(bs, v)
                        fused_silu_fc2_argmax(**args, **buf)
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(buf["final_token"], planted))

    def test_with_value_variant_matches(self):
        m, v = 256, 8200
        dtype = torch.bfloat16
        for bs in BATCH_SIZES:
            with self.subTest(bs=bs):
                args = _make(bs, m, v, dtype, with_bias=False, seed=100 + bs)
                planted = torch.randint(0, v, (bs,), device=DEVICE, dtype=torch.int64)
                rows = torch.arange(bs, device=DEVICE)
                args["base_logits"][rows, planted] = 1000.0
                buf = _scratch(bs, v)
                final_value = torch.empty((bs,), dtype=torch.float32, device=DEVICE)
                fused_silu_fc2_argmax_with_value(
                    **args, **buf, final_value=final_value
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(buf["final_token"], planted))
                ref = _reference(args)
                torch.testing.assert_close(
                    final_value, ref[rows, planted], rtol=2e-2, atol=2e-2
                )

    def test_real_shape_matches_fp32_reference(self):
        """生产形状（M=256, V=151936, num_v_blocks=1187）下与 fp32 参考对齐。

        bf16 MMA 与 fp32 参考在接近平局的行上允许分歧，所以只在 top1/top2
        间距大于 MMA 误差量级的行上要求 index 完全一致。
        """
        m, v, bs = 256, 151936, 40
        args = _make(bs, m, v, torch.bfloat16, with_bias=False, seed=7)
        buf = _scratch(bs, v)
        final_value = torch.empty((bs,), dtype=torch.float32, device=DEVICE)
        fused_silu_fc2_argmax_with_value(**args, **buf, final_value=final_value)
        torch.cuda.synchronize()

        ref = _reference(args)
        top2 = ref.topk(2, dim=-1)
        gap = top2.values[:, 0] - top2.values[:, 1]
        tol = 5e-2
        decisive = gap > tol
        self.assertGreater(int(decisive.sum()), 0, "构造的样本全是平局，测试无效")
        self.assertTrue(
            torch.equal(
                buf["final_token"][decisive], top2.indices[:, 0][decisive].long()
            )
        )
        rows = torch.arange(bs, device=DEVICE)
        torch.testing.assert_close(
            final_value, ref[rows, buf["final_token"]], rtol=tol, atol=tol
        )

    def test_cuda_graph_capture_replay(self):
        m, v, bs = 256, 8200, 40
        args = _make(bs, m, v, torch.bfloat16, with_bias=False, seed=3)
        planted = torch.randint(0, v, (bs,), device=DEVICE, dtype=torch.int64)
        args["base_logits"][torch.arange(bs, device=DEVICE), planted] = 1000.0
        buf = _scratch(bs, v)
        fused_silu_fc2_argmax(**args, **buf)  # warm up JIT before capture
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_silu_fc2_argmax(**args, **buf)
        buf["final_token"].zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(buf["final_token"], planted))


if __name__ == "__main__":
    unittest.main()
