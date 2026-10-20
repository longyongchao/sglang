"""验证 DSpark 采样草稿 rollout 的 fused 采样 kernel 与 CUDA graph 路径。"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.models.dspark import GatedMarkovHead, VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_sampled_draft_graph import (
    SampledDraftGraphRunner,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
    MARKOV_BLOCK_V,
    fused_markov_bias_sample,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)

DTYPES = [torch.float16, torch.bfloat16]


def _assert_logits_close(actual, expected):
    tol = 2e-2 if actual.dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


def _assert_rollout_consistent(head, base, anchor, tokens, logits, greedy_rows):
    """每步 corrected logits 以上一步抽中的 token 为条件，且 greedy 行取其 argmax。"""
    prev = anchor
    with torch.no_grad():
        for k in range(base.shape[1]):
            expected = head.apply_step_logits(
                base[:, k], token_ids=prev, hidden_states=None
            )
            _assert_logits_close(logits[:, k], expected)
            argmax = logits[:, k].argmax(dim=-1)
            assert torch.equal(tokens[greedy_rows, k], argmax[greedy_rows])
            prev = tokens[:, k]


def _scratch(bs, vocab_size):
    num_v_blocks = (vocab_size + MARKOV_BLOCK_V - 1) // MARKOV_BLOCK_V
    return (
        torch.empty((bs, num_v_blocks), dtype=torch.float32, device="cuda"),
        torch.empty((bs, num_v_blocks), dtype=torch.int32, device="cuda"),
        torch.empty((bs,), dtype=torch.int64, device="cuda"),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bs", [1, 5, 40])
@pytest.mark.parametrize("in_place", [False, True])
def test_fused_sample_matches_reference(dtype, bs, in_place):
    torch.manual_seed(3)
    vocab_size, rank = 1000, 48
    head = VanillaMarkov(vocab_size=vocab_size, markov_rank=rank).cuda().to(dtype)
    base = torch.randn(bs, vocab_size, device="cuda", dtype=dtype)
    prev = torch.randint(0, vocab_size, (bs,), device="cuda")
    temperatures = torch.rand(bs, device="cuda").add_(0.3)
    greedy = (torch.arange(bs, device="cuda") % 3 == 0).to(torch.int32)
    noise = torch.empty(bs, vocab_size, device="cuda").exponential_(1)

    with torch.no_grad():
        expected_logits = head.apply_step_logits(
            base, token_ids=prev, hidden_states=None
        )
        latent = head.markov_w1.weight[prev].contiguous()

    logits_out = base.clone() if in_place else torch.empty_like(base)
    base_in = logits_out if in_place else base
    out_val, out_idx, tokens = _scratch(bs, vocab_size)
    fused_markov_bias_sample(
        latent=latent,
        fc2_weight=head.markov_w2.weight.detach(),
        base_logits=base_in,
        logits_out=logits_out,
        temperatures=temperatures,
        greedy_mask=greedy,
        exp_noise=noise,
        out_val=out_val,
        out_idx=out_idx,
        final_token=tokens,
    )
    # fc2 的累加顺序与 cuBLAS 不同，存储精度上可能差 1 ulp。
    _assert_logits_close(logits_out, expected_logits)
    # 抽样必须基于 kernel 实际写出的那份 logits（验证端据此得到 q）。
    stored = logits_out.float()
    key = torch.where(
        greedy.bool()[:, None], stored, stored / temperatures[:, None] - noise.log()
    )
    assert torch.equal(tokens, key.argmax(dim=-1))


def _eager_rollout(head, base_logits, anchor):
    with torch.no_grad():
        return head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            hidden_states=None,
            sampler=lambda logits, _: logits.argmax(dim=-1),
        )


def _runner(head, gamma, capture_bs):
    return SampledDraftGraphRunner(
        markov_head=head,
        gamma=gamma,
        capture_bs=capture_bs,
        max_bs=max(capture_bs),
        device=torch.device("cuda", torch.cuda.current_device()),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bs", [1, 3, 8])
def test_graph_greedy_rows_match_eager(dtype, bs):
    """全 greedy 批次不依赖噪声：多次 replay（含 padding 行）都与 eager 一致。"""
    torch.manual_seed(5)
    vocab_size, rank, gamma = 777, 32, 6
    head = VanillaMarkov(vocab_size=vocab_size, markov_rank=rank).cuda().to(dtype)
    runner = _runner(head, gamma, capture_bs=[1, 4, 8])
    all_rows = torch.ones(bs, dtype=torch.bool, device="cuda")
    for _ in range(3):
        base = torch.randn(bs, gamma, vocab_size, device="cuda", dtype=dtype)
        anchor = torch.randint(0, vocab_size, (bs,), device="cuda")
        out = runner.run(
            base_logits=base,
            anchor_tokens=anchor,
            temperatures=torch.ones(bs, device="cuda"),
            greedy_mask=all_rows,
        )
        assert out is not None
        tokens, logits = out
        assert tokens.shape == (bs, gamma) and logits.shape == base.shape
        _assert_rollout_consistent(head, base, anchor, tokens, logits, all_rows)
        expected_tokens, _ = _eager_rollout(head, base, anchor)
        assert torch.equal(tokens, expected_tokens)


def test_graph_mixed_rows_are_consistent_and_vary():
    """混合批次：q 与被抽中的 token 来自同一份 logits，且每次 replay 噪声都会刷新。"""
    torch.manual_seed(9)
    dtype = torch.bfloat16
    vocab_size, rank, gamma, bs = 512, 16, 4, 6
    head = VanillaMarkov(vocab_size=vocab_size, markov_rank=rank).cuda().to(dtype)
    runner = _runner(head, gamma, capture_bs=[8])
    base = torch.randn(bs, gamma, vocab_size, device="cuda", dtype=dtype)
    anchor = torch.randint(0, vocab_size, (bs,), device="cuda")
    temperatures = torch.full((bs,), 1.0, device="cuda")
    greedy = torch.arange(bs, device="cuda") % 2 == 0

    draws = []
    for _ in range(4):
        tokens, logits = runner.run(
            base_logits=base,
            anchor_tokens=anchor,
            temperatures=temperatures,
            greedy_mask=greedy,
        )
        assert torch.all((tokens >= 0) & (tokens < vocab_size))
        _assert_rollout_consistent(head, base, anchor, tokens, logits, greedy)
        draws.append(tokens)
    sampled = ~greedy
    assert any(not torch.equal(draws[0][sampled], d[sampled]) for d in draws[1:])
    # 返回的 draft tokens 不能与下一次 replay 共享存储。
    assert all(d.data_ptr() != draws[0].data_ptr() for d in draws[1:])


def test_graph_sampling_distribution():
    """单步、小词表下经验分布逼近 softmax(logits / T)。"""
    torch.manual_seed(1)
    dtype = torch.float16
    vocab_size, rank, gamma, bs = 8, 16, 1, 64
    head = VanillaMarkov(vocab_size=vocab_size, markov_rank=rank).cuda().to(dtype)
    with torch.no_grad():
        head.markov_w2.weight.zero_()
    runner = _runner(head, gamma, capture_bs=[bs])
    row = torch.randn(vocab_size, device="cuda", dtype=dtype)
    base = row.expand(bs, gamma, vocab_size).contiguous()
    temperature = 0.7
    counts = torch.zeros(vocab_size, device="cuda")
    for _ in range(200):
        tokens, _ = runner.run(
            base_logits=base,
            anchor_tokens=torch.zeros(bs, dtype=torch.long, device="cuda"),
            temperatures=torch.full((bs,), temperature, device="cuda"),
            greedy_mask=torch.zeros(bs, dtype=torch.bool, device="cuda"),
        )
        counts += torch.bincount(tokens.view(-1), minlength=vocab_size)
    empirical = counts / counts.sum()
    expected = torch.softmax(row.float() / temperature, dim=-1)
    assert (empirical - expected).abs().max().item() < 0.02


def test_graph_returns_none_when_not_covered():
    dtype = torch.bfloat16
    vocab_size, rank, gamma = 64, 8, 3
    head = VanillaMarkov(vocab_size=vocab_size, markov_rank=rank).cuda().to(dtype)
    runner = _runner(head, gamma, capture_bs=[2])
    kwargs = dict(
        anchor_tokens=torch.zeros(4, dtype=torch.long, device="cuda"),
        temperatures=torch.ones(4, device="cuda"),
        greedy_mask=torch.zeros(4, dtype=torch.bool, device="cuda"),
    )
    # bs 超出捕获范围
    base = torch.randn(4, gamma, vocab_size, device="cuda", dtype=dtype)
    assert runner.run(base_logits=base, **kwargs) is None
    # dtype 不一致
    base = torch.randn(2, gamma, vocab_size, device="cuda", dtype=torch.float16)
    small = {k: v[:2] for k, v in kwargs.items()}
    assert runner.run(base_logits=base, **small) is None
    # 覆盖范围内的批次走 graph
    base = torch.randn(2, gamma, vocab_size, device="cuda", dtype=dtype)
    assert runner.run(base_logits=base, **small) is not None


def test_builder_rejects_non_vanilla_heads():
    from sglang.srt.speculative.dspark_components.dspark_sampled_draft_graph import (
        maybe_build_sampled_draft_graph,
    )

    class _Model:
        markov_head = GatedMarkovHead(vocab_size=32, markov_rank=8, hidden_size=16)

    assert (
        maybe_build_sampled_draft_graph(
            draft_model=_Model(),
            gamma=4,
            capture_bs=[1, 2],
            device="cuda",
            tp_rank=0,
            dp_moe_sync=False,
            draft_temperature=1.0,
        )
        is None
    )
