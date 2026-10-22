"""chain 拒绝采样直接读 target / 草稿 logits，与先落稠密 fp32 probs 的老路径逐位等价。

老路径：``softmax(logits.float() / T)`` 落稠密 probs（q 再 scatter 进
``zeros(bs, num_slots, vocab)`` 的 ``[draft_vocab_start, +draft_vocab)`` 区间），再走
``chain_speculative_sampling_triton``。新路径在 kernel 里按 ``exp(l / T - lse)`` 现算
p / q。两者在同一份 coin 下 accept 长度与 predicts 必须一致。
"""

from __future__ import annotations

import types
import unittest

import torch

from sglang.srt.speculative.dspark_components.kernels import dspark_accept
from sglang.srt.speculative.reject_sampling import (
    chain_speculative_sampling_fused_triton,
    chain_speculative_sampling_triton,
    row_logsumexp,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")


def _buffers(bs, num_slots):
    return (
        torch.full((bs * num_slots,), -1, dtype=torch.int32, device=DEVICE),
        torch.full((bs, num_slots), -1, dtype=torch.int32, device=DEVICE),
        torch.zeros((bs,), dtype=torch.int32, device=DEVICE),
    )


def _per_row_temp(temperature, bs):
    if isinstance(temperature, torch.Tensor):
        return temperature.reshape(bs, 1, 1).float()
    return temperature


def _dense_q(draft_logits, num_slots, vocab, start, temperature):
    bs, rows, dv = draft_logits.shape
    t = _per_row_temp(temperature, bs)
    probs = torch.softmax(draft_logits.float() / t, dim=-1)
    dense = torch.zeros(bs, num_slots, vocab, dtype=torch.float32, device=DEVICE)
    end = min(start + dv, vocab)
    dense[:, :rows, start:end] = probs[:, :, : end - start]
    return dense


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton kernels")
class TestChainRSFromLogits(CustomTestCase):
    def _case(
        self, *, bs, num_slots, vocab, draft_vocab, start, rows, temperature, seed,
        target_from_logits=False, target_temperature=1.0,
    ):  # fmt: skip
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        draft_logits = torch.randn(
            bs, rows, draft_vocab, generator=g, device=DEVICE
        ).mul_(3.0).to(torch.bfloat16)
        dense_q = _dense_q(draft_logits, num_slots, vocab, start, temperature)
        # target 与 q 有重叠也有差异，accept/reject 两条路都会走到
        target_logits = torch.randn(bs, num_slots, vocab, generator=g, device=DEVICE)
        target_logits[:, :rows, start : start + draft_vocab] += draft_logits[
            :, :, : min(draft_vocab, vocab - start)
        ].float() / _per_row_temp(temperature, bs)
        target_logits = target_logits.to(torch.bfloat16)
        target_probs = torch.softmax(
            target_logits.float() / _per_row_temp(target_temperature, bs), dim=-1
        ).contiguous()
        candidates = torch.zeros(bs, num_slots, dtype=torch.int64, device=DEVICE)
        for k in range(1, num_slots):
            row = k - 1
            src = dense_q[:, row] if row < rows else target_probs[:, row]
            candidates[:, k] = torch.multinomial(src, 1, generator=g).squeeze(1)
        retrieve_index = (
            torch.arange(bs * num_slots, dtype=torch.int64, device=DEVICE)
            .view(bs, num_slots)
            .contiguous()
        )
        coins = torch.rand(bs, num_slots, generator=g, device=DEVICE)
        final_coins = torch.rand(bs, generator=g, device=DEVICE)

        ref = _buffers(bs, num_slots)
        chain_speculative_sampling_triton(
            predicts=ref[0],
            accept_index=ref[1],
            accept_token_num=ref[2],
            candidates=candidates,
            retrive_index=retrieve_index,
            retrive_next_token=None,
            retrive_next_sibling=None,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=final_coins,
            target_probs=target_probs,
            draft_probs=dense_q,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )
        if target_from_logits:
            target_kwargs = dict(
                target_logits=target_logits, target_temperature=target_temperature
            )
        else:
            target_kwargs = dict(target_probs=target_probs)
        out = _buffers(bs, num_slots)
        chain_speculative_sampling_fused_triton(
            predicts=out[0],
            accept_index=out[1],
            accept_token_num=out[2],
            candidates=candidates,
            retrive_index=retrieve_index,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=final_coins,
            draft_logits=draft_logits,
            draft_vocab_start=start,
            draft_temperature=temperature,
            **target_kwargs,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out[2], ref[2]), (out[2], ref[2]))
        self.assertTrue(torch.equal(out[1], ref[1]))
        self.assertTrue(torch.equal(out[0], ref[0]))
        return ref[2]

    def test_full_vocab_matches_dense(self):
        correct = self._case(
            bs=64, num_slots=6, vocab=9000, draft_vocab=9000, start=0, rows=5,
            temperature=1.0, seed=1,
        )  # fmt: skip
        # 用例要同时覆盖全部接受与中途拒绝
        self.assertTrue((correct == 5).any() and (correct < 5).any(), correct)

    def test_vocab_shard_offset_and_temperature(self):
        self._case(
            bs=32, num_slots=5, vocab=8192, draft_vocab=3000, start=2500, rows=4,
            temperature=0.7, seed=2,
        )  # fmt: skip

    def test_draft_vocab_overhangs_target_vocab(self):
        """草稿词表越过目标词表尾部：softmax 仍按整行归一，越界部分被截掉。"""
        self._case(
            bs=16, num_slots=4, vocab=5000, draft_vocab=3000, start=4000, rows=3,
            temperature=1.0, seed=3,
        )  # fmt: skip

    def test_missing_rows_are_zero_q(self):
        self._case(
            bs=16, num_slots=5, vocab=4096, draft_vocab=4096, start=0, rows=2,
            temperature=1.0, seed=4,
        )  # fmt: skip

    def test_target_and_draft_from_logits_per_request_temperature(self):
        bs = 48
        g = torch.Generator(device=DEVICE).manual_seed(6)
        t_temps = torch.rand(bs, 1, generator=g, device=DEVICE) * 1.2 + 0.4
        d_temps = torch.rand(bs, generator=g, device=DEVICE) * 1.2 + 0.4
        correct = self._case(
            bs=bs, num_slots=6, vocab=9000, draft_vocab=9000, start=0, rows=5,
            temperature=d_temps, seed=7, target_from_logits=True,
            target_temperature=t_temps,
        )  # fmt: skip
        self.assertTrue((correct == 5).any() and (correct < 5).any(), correct)

    def test_target_from_logits_scalar_temperature(self):
        self._case(
            bs=32, num_slots=5, vocab=8192, draft_vocab=8192, start=0, rows=4,
            temperature=1.0, seed=8, target_from_logits=True, target_temperature=1.0,
        )  # fmt: skip

    def test_row_logsumexp(self):
        g = torch.Generator(device=DEVICE).manual_seed(5)
        x = torch.randn(3, 4, 10000, generator=g, device=DEVICE).mul_(5).bfloat16()
        x[0, 0, :5000] = float("-inf")  # 首块整块 -inf
        lse = row_logsumexp(x, 0.8)
        ref = torch.logsumexp(x.float() / 0.8, dim=-1)
        torch.testing.assert_close(lse, ref, rtol=1e-5, atol=1e-5)

        temps = torch.tensor([0.5, 1.0, 1.7], device=DEVICE)
        lse = row_logsumexp(x, temps)
        ref = torch.logsumexp(x.float() / temps.view(3, 1, 1), dim=-1)
        torch.testing.assert_close(lse, ref, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton kernels")
class TestDSparkAcceptSamplingFromLogits(CustomTestCase):
    """DSpark ``AcceptSampling``：draft_logits 路径与 SoftmaxTemp 稠密 probs 路径一致。"""

    def _run(self, *, seed, **kwargs):
        torch.manual_seed(seed)  # 同一份 coin
        out = dspark_accept.AcceptSampling.execute(**kwargs)
        torch.cuda.synchronize()
        return tuple(x.clone() for x in out)

    def test_matches_dense_softmax_path(self):
        bs, gamma, vocab = 32, 5, 12000
        t = gamma + 1
        g = torch.Generator(device=DEVICE).manual_seed(11)
        temps = torch.rand(bs, 1, generator=g, device=DEVICE) * 1.2 + 0.4
        d_temps = temps.reshape(bs)
        draft_logits = (
            torch.randn(bs, gamma, vocab, generator=g, device=DEVICE)
            .mul_(3.0)
            .bfloat16()
        )
        target_logits = torch.randn(bs, t, vocab, generator=g, device=DEVICE)
        target_logits[:, :gamma] += draft_logits.float()
        target_logits = target_logits.bfloat16().view(bs * t, vocab)
        draft_probs = dspark_accept.SoftmaxTemp.execute(
            logits=draft_logits.reshape(bs * gamma, vocab),
            temperatures=d_temps,
            rows_per_request=gamma,
        ).view(bs, gamma, vocab)
        candidates = torch.zeros(bs, t, dtype=torch.int64, device=DEVICE)
        candidates[:, 1:] = torch.multinomial(
            draft_probs.view(-1, vocab), 1, generator=g
        ).view(bs, gamma)
        common = dict(
            candidates=candidates,
            target_logits=target_logits,
            sampling_info=types.SimpleNamespace(
                need_top_k_sampling=False,
                need_top_p_sampling=False,
                temperatures=temps,
            ),
            draft_input=types.SimpleNamespace(max_top_k=0, uniform_top_k_value=None),
            gamma=gamma,
            verify_num_draft_tokens=t,
        )
        cutoff = torch.randint(1, t + 1, (bs,), device=DEVICE, generator=g).int()
        for cutoff_verify_lens in (None, cutoff):
            ref = self._run(
                seed=12, draft_probs=draft_probs,
                cutoff_verify_lens=cutoff_verify_lens, **common,
            )  # fmt: skip
            got = self._run(
                seed=12, draft_probs=None, draft_logits=draft_logits,
                draft_temperatures=d_temps,
                cutoff_verify_lens=cutoff_verify_lens, **common,
            )  # fmt: skip
            for a, b in zip(got, ref):
                self.assertTrue(torch.equal(a, b), (a, b))
            if cutoff_verify_lens is None:
                correct_len = ref[0]
                self.assertTrue(
                    (correct_len == gamma).any() and (correct_len < gamma).any(),
                    correct_len,
                )


if __name__ == "__main__":
    unittest.main()
