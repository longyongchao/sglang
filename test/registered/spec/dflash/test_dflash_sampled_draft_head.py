"""DFLASH 采样草稿头折进 draft cuda graph。

``draft_temperature > 0`` 时 ``_DflashDraftSampler`` 在图内算 logits、用 exp-race
抽样，并把 logits 留在静态 buffer 里给拒绝采样验证当 q。这里检查 graph replay 后
logits 与 eager GEMM 一致、抽到的 token 有正概率、噪声随 replay 重抽。
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDflashSampledDraftHead(CustomTestCase):
    BS, BLOCK, H, V = 3, 5, 64, 4096
    DTYPE = torch.bfloat16

    def _build(self, *, anchor_prediction, max_bs=None):
        from sglang.srt.speculative.dflash_worker_v2 import _DflashDraftSampler

        g = torch.Generator(device=DEVICE).manual_seed(11)
        weight = torch.randn(
            self.V, self.H, generator=g, device=DEVICE, dtype=self.DTYPE
        ).mul_(4.0 * self.H**-0.5)
        hidden = torch.randn(
            self.BS * self.BLOCK, self.H, generator=g, device=DEVICE, dtype=self.DTYPE
        )
        sampler = _DflashDraftSampler(
            weight=weight,
            block_size=self.BLOCK,
            num_org=self.V,
            org_vocab_start=0,
            max_bs=self.BS if max_bs is None else max_bs,
            anchor_prediction=anchor_prediction,
            draft_temperature=1.0,
        )
        return sampler, weight, hidden

    def _check(self, anchor_prediction):
        sampler, weight, hidden = self._build(anchor_prediction=anchor_prediction)
        num_draft = self.BLOCK if anchor_prediction else self.BLOCK - 1
        n = self.BS * num_draft
        sampler(hidden)  # warmup：编译 triton、填 row-args 缓存
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sampler(hidden)

        hs = hidden.view(self.BS, self.BLOCK, self.H)
        if not anchor_prediction:
            hs = hs[:, 1:, :]
        ref_logits = hs.reshape(-1, self.H) @ weight.T

        draws = []
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            logits = sampler.logits[:n].clone()
            tokens = sampler.out[:n].clone()
            draws.append(tokens)
            self.assertTrue(torch.equal(logits, ref_logits))
            probs = torch.softmax(logits.float(), dim=-1)
            self.assertTrue((probs.gather(1, tokens[:, None]) > 0).all())
        self.assertFalse(
            torch.equal(draws[0], draws[1]),
            "两次 replay 抽到同一串 token，说明噪声没有跟着 replay 重抽",
        )

    def test_graph_sampled_head(self):
        self._check(anchor_prediction=False)

    def test_graph_sampled_head_anchor_prediction(self):
        self._check(anchor_prediction=True)

    def test_over_max_bs_is_noop(self):
        sampler, _, hidden = self._build(anchor_prediction=False, max_bs=1)
        sampler.out.fill_(-1)
        sampler(hidden)
        torch.cuda.synchronize()
        self.assertTrue((sampler.out == -1).all())


if __name__ == "__main__":
    unittest.main()
