"""Domino 拒绝采样 rollout：融合采样 kernel 与整条 graph 的等价性。

覆盖两个入口：

- ``fused_silu_fc2_sample``——把 fc2 GEMM、logits 落盘和抽样合成一趟。抽样必须用
  写回 ``logits_out`` 的那份数，否则验证阶段读到的 q(x) 和被抽中的 token 来自两
  份不同的 logits。噪声由 caller 用 ``Tensor.exponential_`` 批量生成。
- ``_get_or_capture_domino_rs_loop_graph``——``return_logits=True`` 这条路径以前
  一直是 eager 的，因为里面有 host sync；sync 去掉之后形状在 ``(bs, num_draft)``
  固定，可以整条 capture。
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.speculative.domino_kernels import fused_silu_fc2_sample
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
BLOCK_V = 128


def _scratch(bs, v, block_v=BLOCK_V):
    num_v_blocks = (v + block_v - 1) // block_v
    return {
        "out_val": torch.empty((bs, num_v_blocks), dtype=torch.float32, device=DEVICE),
        "out_idx": torch.empty((bs, num_v_blocks), dtype=torch.int32, device=DEVICE),
        "final_token": torch.empty((bs,), dtype=torch.int64, device=DEVICE),
    }


def _args(bs, m, v, seed, dtype=torch.bfloat16, with_bias=True):
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g, device=DEVICE, dtype=dtype)

    return {
        "z_proj": rand(bs, m),
        "s_proj": rand(bs, m),
        "fc2_weight": rand(v, m).mul_(0.05),
        "fc2_bias": rand(v) if with_bias else None,
        "base_logits": rand(bs, v),
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton kernels")
class TestFusedSiluFc2Sample(CustomTestCase):
    def test_logits_match_eager(self):
        """``logits_out`` 必须等于 eager 的 ``base_logits + fc2(silu(z + s))``。"""
        bs, m, v = 3, 256, 8200  # v 非 BLOCK_V 整数倍 -> 末尾 ragged tile
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                args = _args(bs, m, v, seed=hash(dtype) % 1000, dtype=dtype)
                mid = torch.nn.functional.silu(
                    args["z_proj"].float() + args["s_proj"].float()
                ).to(dtype)
                ref = (
                    mid.float() @ args["fc2_weight"].float().t()
                    + args["fc2_bias"].float()
                    + args["base_logits"].float()
                )

                buf = _scratch(bs, v)
                logits_out = torch.empty((bs, v), dtype=dtype, device=DEVICE)
                fused_silu_fc2_sample(
                    **args,
                    logits_out=logits_out,
                    exp_noise=None,
                    inv_temperature=0.0,
                    **buf,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(logits_out.float(), ref, rtol=2e-2, atol=2e-2)
                # greedy 分支必须就是写出去的那份 logits 的 argmax
                self.assertTrue(
                    torch.equal(buf["final_token"], logits_out.argmax(dim=-1))
                )

    def test_sampling_leaves_logits_untouched(self):
        """抽样只影响 token，不能污染验证阶段要读的 q。"""
        bs, m, v = 4, 256, 8200
        dtype = torch.bfloat16
        args = _args(bs, m, v, seed=21, dtype=dtype, with_bias=False)
        buf = _scratch(bs, v)

        greedy_logits = torch.empty((bs, v), dtype=dtype, device=DEVICE)
        fused_silu_fc2_sample(
            **args,
            logits_out=greedy_logits,
            exp_noise=None,
            inv_temperature=0.0,
            **buf,
        )
        noise = torch.empty((bs, v), dtype=torch.float32, device=DEVICE).exponential_(1)
        sampled_logits = torch.empty((bs, v), dtype=dtype, device=DEVICE)
        fused_silu_fc2_sample(
            **args,
            logits_out=sampled_logits,
            exp_noise=noise,
            inv_temperature=1.0,
            **buf,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(sampled_logits, greedy_logits))
        # 抽中的 token 必须落在正概率的支撑集上
        probs = torch.softmax(sampled_logits.float(), dim=-1)
        self.assertTrue((probs.gather(1, buf["final_token"].unsqueeze(1)) > 0).all())

    def test_sampling_needs_noise(self):
        bs, m, v = 1, 64, 256
        args = _args(bs, m, v, seed=1, with_bias=False)
        with self.assertRaises(ValueError):
            fused_silu_fc2_sample(
                **args,
                logits_out=torch.empty((bs, v), dtype=torch.bfloat16, device=DEVICE),
                exp_noise=None,
                inv_temperature=1.0,
                **_scratch(bs, v),
            )

    def test_draw_matches_indexed_exponential_race(self):
        """给定同一份噪声，kernel 的抽样必须逐个等于 ``argmax_i p_i / E_i``。"""
        bs, m, v = 8, 128, 4096
        dtype = torch.bfloat16
        temperature = 0.9
        args = _args(bs, m, v, seed=33, dtype=dtype, with_bias=True)
        buf = _scratch(bs, v)
        logits_out = torch.empty((bs, v), dtype=dtype, device=DEVICE)
        noise = torch.empty((bs, v), dtype=torch.float32, device=DEVICE).exponential_(1)
        fused_silu_fc2_sample(
            **args,
            logits_out=logits_out,
            exp_noise=noise,
            inv_temperature=1.0 / temperature,
            **buf,
        )
        torch.cuda.synchronize()
        probs = torch.softmax(logits_out.float() / temperature, dim=-1)
        self.assertTrue(torch.equal(buf["final_token"], probs.div(noise).argmax(dim=-1)))

    def test_draw_matches_softmax_frequency(self):
        """重复抽样，经验频率对齐 kernel 自己写出的 logits 的 softmax。"""
        bs, m, v, reps = 1, 128, 1024, 4000
        dtype = torch.bfloat16
        args = _args(bs, m, v, seed=34, dtype=dtype, with_bias=False)
        buf = _scratch(bs, v)
        logits_out = torch.empty((bs, v), dtype=dtype, device=DEVICE)
        noise = torch.empty((bs, v), dtype=torch.float32, device=DEVICE)
        tokens = []
        for _ in range(reps):
            noise.exponential_(1)
            fused_silu_fc2_sample(
                **args,
                logits_out=logits_out,
                exp_noise=noise,
                inv_temperature=1.0,
                **buf,
            )
            tokens.append(buf["final_token"].clone())
        empirical = torch.bincount(torch.cat(tokens), minlength=v).double() / reps
        probs = torch.softmax(logits_out[0].float(), dim=-1).double()
        top = torch.topk(probs, 16).indices
        self.assertLess((empirical[top] - probs[top]).abs().max().item(), 0.03)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton kernels")
class TestDominoRSRolloutGraph(CustomTestCase):
    """整条拒绝采样 rollout 被 capture 成一张 graph 后仍与 eager 等价。"""

    BS, ND, V, EMB, G = 2, 4, 4096, 128, 96
    DTYPE = torch.bfloat16

    def _state(self, seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)

        def rand(*shape):
            return torch.randn(*shape, generator=g, device=DEVICE, dtype=self.DTYPE)

        return {
            "fc2_weight": rand(self.V, self.EMB).div_(self.EMB**0.5),
            "fc2_bias": rand(self.V).mul_(0.1),
            "w_s_hh_T": rand(self.G, self.EMB + 3 * self.G).div_(self.G**0.5),
            "b_hh": rand(3 * self.G).mul_(0.1),
        }, rand(self.V, 3 * self.G).div_(self.G**0.5)

    def _capture(self, inv_temperature, seed):
        from sglang.srt.speculative.domino_rollout import DFlashDominoRollout

        state, table = self._state(seed)
        rollout = DFlashDominoRollout(
            domino_helper=None,
            block_size=self.ND + 1,
            target_vocab_size=self.V,
            global_argmax_from_local_logits=None,
            global_argmax_from_local_max=None,
        )
        entry = rollout._get_or_capture_domino_rs_loop_graph(
            bs=self.BS,
            num_draft=self.ND,
            emb_dim=self.EMB,
            gru_hidden=self.G,
            num_org=self.V,
            org_vocab_start=0,
            z_dtype=self.DTYPE,
            logits_dtype=self.DTYPE,
            inv_temperature=inv_temperature,
            device=DEVICE,
            state=state,
            gru_input_table=table,
        )
        self.assertIsNotNone(entry, "capture 被 pool budget 拒绝了，测试形状选得太大")
        entry["_rollout"] = rollout
        return state, table, entry

    def _inputs(self, seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)

        def rand(*shape):
            return torch.randn(*shape, generator=g, device=DEVICE, dtype=self.DTYPE)

        return (
            rand(self.BS, self.ND, self.EMB),
            rand(self.BS, self.ND, self.V).mul_(2.0),
            rand(self.BS, self.G),
            torch.randint(0, self.V, (self.BS,), generator=g, device=DEVICE),
        )

    def _replay(self, entry, z_proj, base_logits, gru_h, slot_1):
        entry["z_proj_buf"].copy_(z_proj)
        entry["base_logits_buf"].copy_(base_logits)
        entry["gru_h_buf"].copy_(gru_h)
        entry["out_buf"][:, 0].copy_(slot_1)
        entry["graph"].replay()
        torch.cuda.synchronize()
        return entry["out_buf"].clone(), entry["logits_buf"].clone()

    def _eager(self, state, table, z_proj, base_logits, gru_h):
        """`rollout_draft_block` 里保留的那条 eager 参考循环，greedy 分支。"""
        from sglang.srt.speculative.domino_kernels import fused_gru_cell_from_table

        logits = torch.empty(self.BS, self.ND, self.V, device=DEVICE, dtype=self.DTYPE)
        logits[:, 0, :] = base_logits[:, 0, :]
        out = torch.zeros(self.BS, self.ND, dtype=torch.long, device=DEVICE)
        sh = torch.empty(
            self.BS, self.EMB + 3 * self.G, device=DEVICE, dtype=self.DTYPE
        )
        h_new = torch.empty_like(gru_h)
        h_state = gru_h
        for k in range(1, self.ND):
            torch.matmul(h_state, state["w_s_hh_T"], out=sh)
            mid = torch.nn.functional.silu(z_proj[:, k, :] + sh[:, : self.EMB])
            bias = torch.nn.functional.linear(
                mid, state["fc2_weight"], state["fc2_bias"]
            )
            torch.add(base_logits[:, k, :], bias.to(self.DTYPE), out=logits[:, k, :])
            tokens = torch.argmax(logits[:, k, :], dim=-1).to(torch.long)
            out[:, k] = tokens
            if k + 1 < self.ND:
                fused_gru_cell_from_table(
                    tok_full=tokens,
                    gru_input_table=table,
                    gh=sh[:, self.EMB :],
                    gh_bias=state["b_hh"],
                    h_state=h_state,
                    h_out=h_new,
                )
                h_state = h_new
        return out, logits

    def test_greedy_graph_matches_eager_loop(self):
        state, table, entry = self._capture(0.0, seed=61)
        z_proj, base_logits, gru_h, slot_1 = self._inputs(62)
        out, logits = self._replay(entry, z_proj, base_logits, gru_h, slot_1)
        ref_out, ref_logits = self._eager(state, table, z_proj, base_logits, gru_h)

        torch.testing.assert_close(
            logits.float(), ref_logits.float(), rtol=2e-2, atol=2e-2
        )
        self.assertTrue(torch.equal(out[:, 1:], ref_out[:, 1:]))
        # slot 0 由 caller 填，graph 不能碰；它的 logits 就是 base_logits
        self.assertTrue(torch.equal(out[:, 0], slot_1))
        self.assertTrue(torch.equal(logits[:, 0, :], base_logits[:, 0, :]))

        replayed = self._replay(entry, z_proj, base_logits, gru_h, slot_1)
        self.assertTrue(torch.equal(out, replayed[0]))
        self.assertTrue(torch.equal(logits, replayed[1]))

    def test_sampling_graph_redraws_per_replay(self):
        """``exponential_`` 在 graph 里也要跟着 replay 推进 philox offset。"""
        _, _, entry = self._capture(1.0, seed=71)
        z_proj, base_logits, gru_h, slot_1 = self._inputs(72)
        out_a, logits_a = self._replay(entry, z_proj, base_logits, gru_h, slot_1)
        out_b, logits_b = self._replay(entry, z_proj, base_logits, gru_h, slot_1)

        self.assertFalse(
            torch.equal(out_a[:, 1:], out_b[:, 1:]),
            "两次 replay 抽到同一串 token，说明噪声没有跟着 replay 重抽",
        )
        # slot 1 的 logits 只依赖 gru_h，不依赖抽样结果，必须逐位复现
        self.assertTrue(torch.equal(logits_a[:, 1, :], logits_b[:, 1, :]))
        # 后续 slot 依赖抽中的 token，所以允许变；但抽中的 token 必须有正概率
        for k in range(1, self.ND):
            probs = torch.softmax(logits_a[:, k, :].float(), dim=-1)
            self.assertTrue(
                (probs.gather(1, out_a[:, k : k + 1]) > 0).all(),
                f"slot {k} 抽到了零概率 token",
            )

    def test_in_place_logits_match_graph(self):
        """折进 draft graph 的采样 rollout 让 base_logits 与 logits_target 共用一块
        存储：kernel 在同一个 program 里先读 base tile 再写回同一位置，结果必须和分开
        两块 buffer 的 graph 路径逐位一致。"""
        state, table, entry = self._capture(0.0, seed=81)
        z_proj, base_logits, gru_h, slot_1 = self._inputs(82)
        ref_out, ref_logits = self._replay(entry, z_proj, base_logits, gru_h, slot_1)

        rollout = entry["_rollout"]
        fc2_w, fc2_b = rollout._slice_domino_fc2(
            state=state, start=0, length=self.V
        )
        num_v_blocks = (self.V + BLOCK_V - 1) // BLOCK_V
        shared = base_logits.clone()
        out = torch.empty(self.BS, self.ND, dtype=torch.long, device=DEVICE)
        out[:, 0].copy_(slot_1)
        sh = torch.empty(
            self.BS, self.EMB + 3 * self.G, device=DEVICE, dtype=self.DTYPE
        )
        rollout._run_domino_rs_rollout_loop(
            out_target=out,
            logits_target=shared,
            num_draft=self.ND,
            org_vocab_start=0,
            gru_h=gru_h.clone(),
            z_proj=z_proj,
            base_logits=shared,
            w_sh_T=state["w_s_hh_T"],
            sh_buf=sh,
            s_proj_buf=sh[:, : self.EMB],
            gh_buf=sh[:, self.EMB :],
            h_new_buf=torch.empty_like(gru_h),
            race_val_buf=torch.empty(
                (self.BS, num_v_blocks), dtype=torch.float32, device=DEVICE
            ),
            race_idx_buf=torch.empty(
                (self.BS, num_v_blocks), dtype=torch.int32, device=DEVICE
            ),
            local_tok_buf=torch.empty((self.BS,), dtype=torch.long, device=DEVICE),
            exp_noise_buf=None,
            inv_temperature=0.0,
            fc2_w_shard=fc2_w,
            fc2_b_shard=fc2_b,
            b_hh_static=state["b_hh"],
            gru_input_table=table,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(shared, ref_logits))
        self.assertTrue(torch.equal(out, ref_out))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton kernels")
class TestDominoSampledDraftSampler(CustomTestCase):
    """采样版 ``DominoDraftSampler`` 整条折进 cuda graph：抽到的 token 与写出的 q
    必须自洽——拿抽中的 token 做 teacher forcing 重算每个 slot 的 logits，应当和
    graph 写进 ``sampler.logits`` 的那份一致。"""

    BS, ND, H, V, EMB, G = 3, 4, 64, 4096, 128, 96
    DTYPE = torch.bfloat16

    def _build(self, seed):
        from sglang.srt.speculative.domino_rollout import (
            DFlashDominoRollout,
            DominoDraftSampler,
        )

        g = torch.Generator(device=DEVICE).manual_seed(seed)

        def rand(*shape, scale=1.0):
            return torch.randn(
                *shape, generator=g, device=DEVICE, dtype=self.DTYPE
            ).mul_(scale)

        w_hh = rand(3 * self.G, self.G, scale=self.G**-0.5)
        w_s = rand(self.EMB, self.G, scale=self.G**-0.5)
        state = {
            "w_z": rand(self.EMB, self.H, scale=self.H**-0.5),
            "b1": rand(self.EMB, scale=0.1),
            "gru_hidden_size": self.G,
            "w_hh": w_hh,
            "b_hh": rand(3 * self.G, scale=0.1),
            "w_s_hh_T": torch.cat([w_s, w_hh], dim=0).T.contiguous(),
            "fc2_weight": rand(self.V, self.EMB, scale=self.EMB**-0.5),
            "fc2_bias": rand(self.V, scale=0.1),
        }
        rollout = DFlashDominoRollout(
            domino_helper=None,
            block_size=self.ND + 1,
            target_vocab_size=self.V,
            global_argmax_from_local_logits=None,
            global_argmax_from_local_max=None,
        )
        sampler = DominoDraftSampler(
            rollout=rollout,
            weight=rand(self.V, self.H, scale=self.H**-0.5).mul_(4.0),
            num_org=self.V,
            org_vocab_start=0,
            state=state,
            gru_input_table=rand(self.V, 3 * self.G, scale=0.5),
            candidate_pool_size=0,
            shift_label=False,
            max_bs=self.BS,
            draft_temperature=1.0,
        )
        hidden = rand(self.BS * (self.ND + 1), self.H)
        input_ids = torch.randint(
            0, self.V, (self.BS * (self.ND + 1),), generator=g, device=DEVICE
        )
        return rollout, sampler, state, hidden, input_ids

    def _teacher_forced_logits(self, rollout, sampler, state, hidden, input_ids, out):
        z = hidden.view(self.BS, self.ND + 1, self.H)[:, 1:, :]
        base = (z.reshape(-1, self.H) @ sampler.weight.T).view(
            self.BS, self.ND, self.V
        )
        ref = base.clone()
        bonus = input_ids.view(self.BS, self.ND + 1)[:, 0]
        h = rollout._init_domino_prefix_gru_hidden(
            prefix_ids=torch.stack([bonus, out[:, 0]], dim=1),
            z_dtype=self.DTYPE,
            device=DEVICE,
            state=state,
            full_gru_input_table=sampler.gru_input_table,
        )
        z_proj = torch.nn.functional.linear(z, state["w_z"], state["b1"])
        for k in range(1, self.ND):
            sh = h @ state["w_s_hh_T"]
            mid = torch.nn.functional.silu(z_proj[:, k, :] + sh[:, : self.EMB])
            bias = torch.nn.functional.linear(
                mid, state["fc2_weight"], state["fc2_bias"]
            )
            ref[:, k, :] = (base[:, k, :].float() + bias.float()).to(self.DTYPE)
            h = rollout._domino_manual_gru_step_from_projections(
                gi=sampler.gru_input_table[out[:, k]],
                gh=sh[:, self.EMB :],
                gru_h=h,
                state=state,
            )
        return ref

    def test_graph_writes_consistent_q(self):
        rollout, sampler, state, hidden, input_ids = self._build(91)
        sampler(hidden, input_ids)  # warmup：填缓存、编译 triton
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sampler(hidden, input_ids)

        draws = []
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            out = sampler.out[: self.BS].clone()
            logits = sampler.logits[: self.BS].clone()
            draws.append(out)
            ref = self._teacher_forced_logits(
                rollout, sampler, state, hidden, input_ids, out
            )
            torch.testing.assert_close(
                logits.float(), ref.float(), rtol=3e-2, atol=3e-2
            )
            for k in range(self.ND):
                probs = torch.softmax(logits[:, k, :].float(), dim=-1)
                self.assertTrue(
                    (probs.gather(1, out[:, k : k + 1]) > 0).all(),
                    f"slot {k} 抽到了零概率 token",
                )
        self.assertFalse(
            torch.equal(draws[0], draws[1]),
            "两次 replay 抽到同一串 token，说明噪声没有跟着 replay 重抽",
        )


if __name__ == "__main__":
    unittest.main()
