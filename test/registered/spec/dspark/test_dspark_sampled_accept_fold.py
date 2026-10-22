"""验证 DSpark verify epilogue 在 target graph 内的门控拒绝采样 accept。"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    SampledAcceptSpec,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_accept import (
    ChainVerifyStaticBuffers,
    accept_greedy_triton,
    gated_accept_sampling_from_logits,
)
from sglang.srt.speculative.reject_sampling import (
    chain_speculative_sampling_fused_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)


def _reference_rs(*, coins, max_bs, candidates, target_logits, t_temps, q_logits,
                  d_temps, verify_lens):  # fmt: skip
    """用 gated 版本实际抽到的 coin 重跑无门控 chain RS，再按 verify_lens 截断。"""
    bs, slots = candidates.shape
    predicts = torch.zeros(bs * slots, dtype=torch.int32, device="cuda")
    accept_index = torch.zeros(bs, slots, dtype=torch.int32, device="cuda")
    accept_num = torch.zeros(bs, dtype=torch.int32, device="cuda")
    chain_speculative_sampling_fused_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_num,
        candidates=candidates,
        retrive_index=torch.arange(bs * slots, device="cuda").view(bs, slots),
        uniform_samples=coins[: bs * slots].view(bs, slots),
        uniform_samples_for_final_sampling=coins[max_bs * slots : max_bs * slots + bs],
        target_logits=target_logits.view(bs, slots, -1),
        target_temperature=t_temps,
        draft_logits=q_logits,
        draft_temperature=d_temps,
    )
    capped = torch.minimum(accept_num.long(), verify_lens.long() - 1)
    pos = accept_index.long().gather(1, capped.clamp_min(0)[:, None])[:, 0]
    return capped, predicts.long()[pos], accept_num.long() - capped


def _case(bs, slots, vocab, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    target = torch.randn(bs * slots, vocab, device="cuda", dtype=dtype, generator=g)
    q = torch.randn(bs, slots - 1, vocab, device="cuda", dtype=dtype, generator=g)
    # 候选 token 取 q 的 argmax，保证接受长度有分布
    cands = torch.cat(
        [torch.zeros(bs, 1, dtype=torch.long, device="cuda"), q.argmax(-1)], dim=1
    )
    verify_lens = torch.randint(1, slots + 1, (bs,), device="cuda", generator=g)
    t_temps = torch.rand(bs, device="cuda", generator=g).add_(0.5)
    d_temps = torch.rand(bs, device="cuda", generator=g).add_(0.5)
    return target, q, cands, verify_lens, t_temps, d_temps


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gate_on", [False, True])
def test_gated_accept_matches_reference_under_graph(dtype, gate_on):
    bs, slots, vocab = 5, 4, 300
    target, q, cands, verify_lens, t_temps, d_temps = _case(bs, slots, vocab, dtype, 0)
    buffers = ChainVerifyStaticBuffers.allocate(max_bs=8, slots=slots, device="cuda")
    gate = torch.zeros(1, dtype=torch.int32, device="cuda")

    def run():
        g_len, g_bonus, g_trim = accept_greedy_triton(
            candidates=cands, target_logits=target, verify_num_draft_tokens=slots,
            cutoff_verify_lens=verify_lens,
        )  # fmt: skip
        outs = gated_accept_sampling_from_logits(
            gate=gate, candidates=cands, target_logits=target,
            target_temperatures=t_temps, draft_logits=q, draft_temperatures=d_temps,
            verify_lens=verify_lens, buffers=buffers, greedy_len=g_len,
            greedy_bonus=g_bonus, greedy_trim=g_trim,
        )  # fmt: skip
        return (g_len, g_bonus, g_trim), outs

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        greedy_out, sampled_out = run()
    gate.fill_(1 if gate_on else 0)
    seen = set()
    for _ in range(6):
        graph.replay()
        torch.cuda.synchronize()
        if not gate_on:
            for a, b in zip(sampled_out, greedy_out):
                assert torch.equal(a.long(), b.long())
            continue
        ref = _reference_rs(
            coins=buffers.coins, max_bs=8, candidates=cands, target_logits=target,
            t_temps=t_temps, q_logits=q, d_temps=d_temps, verify_lens=verify_lens,
        )  # fmt: skip
        for a, b in zip(sampled_out, ref):
            assert torch.equal(a.long(), b)
        assert torch.all(sampled_out[0] <= verify_lens - 1)
        seen.add(tuple(sampled_out[1].tolist()))
    if gate_on:
        assert len(seen) > 1  # coin 随 replay 刷新


def _strided_candidates(input_ids, verify_lens, stride):
    rows = []
    start = 0
    for vl in verify_lens.tolist():
        row = torch.zeros(stride, dtype=torch.long, device="cuda")
        row[:vl] = input_ids[start : start + vl]
        rows.append(row)
        start += vl
    return torch.stack(rows)


def test_epilogue_sampled_accept_in_graph():
    """epilogue 在 graph 内：gate=1 时 accept 与同 coin 的参考 RS 一致，gate=0 回到
    greedy；graph slot 数超过 draft IO 行数时多出的 padding 行保持 greedy。"""
    torch.manual_seed(21)
    dtype = torch.bfloat16
    graph_bs, stride, vocab, hidden = 6, 4, 257, 32
    epilogue = DsparkVerifyEpilogue(
        max_bs=graph_bs,
        verify_num_draft_tokens=stride,
        device="cuda",
        sampled_accept=SampledAcceptSpec(max_bs=4, vocab_size=vocab, logits_dtype=dtype),
    )
    assert epilogue.serves_sampled_accept(4) and not epilogue.serves_sampled_accept(5)
    io = epilogue.sampled_draft_io
    assert io.logits.shape == (4, stride - 1, vocab)
    n_tok = graph_bs * stride
    compact_logits = torch.zeros(n_tok, vocab, device="cuda", dtype=dtype)
    compact_hidden = torch.zeros(n_tok, hidden, device="cuda", dtype=dtype)
    input_ids = torch.zeros(n_tok, dtype=torch.long, device="cuda")
    seq_lens = torch.full((graph_bs,), 10, dtype=torch.long, device="cuda")
    pool_idx = torch.zeros(graph_bs, dtype=torch.long, device="cuda")

    def call():
        epilogue(
            compact_logits=compact_logits, compact_hidden=compact_hidden,
            input_ids=input_ids, seq_lens=seq_lens, req_pool_indices=pool_idx,
            bs=graph_bs,
        )  # fmt: skip

    epilogue.begin_step(None, armed=False)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()

    bs = 3
    seen = set()
    for step in range(8):
        sampled = step % 4 != 3
        verify_lens = torch.randint(1, stride + 1, (bs,), device="cuda")
        total = int(verify_lens.sum())
        compact_logits[:total].copy_(torch.randn(total, vocab, device="cuda") * 2)
        q = torch.randn(bs, stride - 1, vocab, device="cuda", dtype=dtype) * 2
        io.logits[:bs].copy_(q)
        input_ids[:total].copy_(torch.randint(0, vocab, (total,), device="cuda"))
        cands = _strided_candidates(input_ids, verify_lens, stride)
        cands[:, 1:] = torch.where(
            torch.rand(bs, stride - 1, device="cuda") < 0.7, q.argmax(-1), cands[:, 1:]
        )
        # 候选须与 compact input_ids 一致
        start = 0
        for i, vl in enumerate(verify_lens.tolist()):
            input_ids[start : start + vl].copy_(cands[i, :vl])
            start += vl
        cands = _strided_candidates(input_ids, verify_lens, stride)
        d_temps = torch.rand(bs, device="cuda").add_(0.5)
        t_temps = torch.rand(bs, device="cuda").add_(0.5)
        io.temperatures[:bs].copy_(d_temps)
        epilogue.begin_step(
            verify_lens, armed=False,
            sampled_temperatures=t_temps[:, None] if sampled else None,
        )  # fmt: skip
        graph.replay()
        torch.cuda.synchronize()
        acc = epilogue.read_accept(bs)
        strided = epilogue.strided_logits[: bs * stride]
        if sampled:
            ref_len, ref_bonus, ref_trim = _reference_rs(
                coins=epilogue._sampled.chain.coins, max_bs=4, candidates=cands,
                target_logits=strided, t_temps=t_temps, q_logits=q, d_temps=d_temps,
                verify_lens=verify_lens,
            )  # fmt: skip
            seen.add(tuple(acc.bonus.tolist()))
        else:
            ref_len, ref_bonus, ref_trim = accept_greedy_triton(
                candidates=cands, target_logits=strided, verify_num_draft_tokens=stride,
                cutoff_verify_lens=verify_lens,
            )  # fmt: skip
        assert torch.equal(acc.correct_len.long(), ref_len.long())
        assert torch.equal(acc.bonus.long(), ref_bonus.long())
        assert torch.equal(acc.cap_trim_lens.long(), ref_trim.long())
        assert torch.equal(acc.commit_lens.long(), ref_len.long() + 1)
        rows = torch.arange(bs, device="cuda")
        assert torch.equal(acc.out_tokens[rows, acc.correct_len.long()], acc.bonus)
    assert len(seen) > 1


def test_draft_sampler_shares_epilogue_io():
    from sglang.srt.models.dspark import VanillaMarkov
    from sglang.srt.speculative.dspark_components.dspark_draft import (
        DsparkDraftSampler,
        SampledDraftIO,
    )

    class _Model:
        markov_head = VanillaMarkov(vocab_size=96, markov_rank=8).cuda().bfloat16()

    gamma = 3
    io = SampledDraftIO.allocate(
        max_bs=4, gamma=gamma, vocab_size=96, dtype=torch.bfloat16, device="cuda"
    )
    sampler = DsparkDraftSampler(
        model=_Model(), gamma=gamma, max_bs=8, device="cuda", sampled_max_bs=4,
        sampled_io=io,
    )  # fmt: skip
    assert sampler.shares_io
    assert sampler.sampled_logits(4).data_ptr() == io.logits.data_ptr()
    assert sampler._sampled_buffers.temperatures.data_ptr() == io.temperatures.data_ptr()
    # 形状不符时自行分配，不能共享
    other = DsparkDraftSampler(
        model=_Model(), gamma=gamma + 1, max_bs=8, device="cuda", sampled_max_bs=4,
        sampled_io=io,
    )  # fmt: skip
    assert not other.shares_io
