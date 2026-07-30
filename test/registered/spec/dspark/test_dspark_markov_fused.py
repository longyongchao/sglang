"""验证 DSpark Markov fused 贪心采样与 production 路径一致。"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from sglang.srt.models.dspark import GatedMarkovHead, RNNHead, VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DsparkDraftSampler,
    greedy_step_sampler,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
    MarkovFusedScratch,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)

DTYPES = [torch.float16, torch.bfloat16]


def _scratch(head, batch_size, dtype):
    return MarkovFusedScratch(
        max_bs=batch_size,
        vocab_size=head.vocab_size,
        markov_rank=head.markov_rank,
        hidden_size=getattr(head, "hidden_size", 0),
        device=torch.device("cuda"),
        dtype=dtype,
        is_rnn=head.markov_head_type == "rnn",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size,proposal_len", [(1, 8), (4, 15), (129, 4)])
@pytest.mark.parametrize("head_cls", [VanillaMarkov, RNNHead])
def test_production_sample_block_parity(
    dtype, batch_size, proposal_len, head_cls
):
    torch.manual_seed(42)
    vocab_size, rank, hidden_size = 1024, 64, 128
    kwargs = dict(vocab_size=vocab_size, markov_rank=rank)
    if head_cls is RNNHead:
        kwargs["hidden_size"] = hidden_size
    head = head_cls(**kwargs).cuda().to(dtype).eval()
    base_logits = torch.randn(
        batch_size, proposal_len, vocab_size, device="cuda", dtype=dtype
    )
    first_prev = torch.randint(0, vocab_size, (batch_size,), device="cuda")
    hidden_states = (
        torch.randn(
            batch_size, proposal_len, hidden_size, device="cuda", dtype=dtype
        )
        if head_cls is RNNHead
        else None
    )

    eager_tokens, _ = head.sample_block(
        base_logits,
        first_prev_tokens=first_prev,
        hidden_states=hidden_states,
        sampler=greedy_step_sampler,
    )
    fused_tokens = head.sample_block_fused(
        base_logits,
        first_prev_tokens=first_prev,
        hidden_states=hidden_states,
        scratch=_scratch(head, batch_size, dtype),
    )
    assert torch.equal(eager_tokens, fused_tokens)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tie_indices", [(3, 7), (3, 515)])
def test_native_dtype_tie_chooses_smallest_index(dtype, tie_indices):
    vocab_size, rank = 1024, 8
    head = VanillaMarkov(vocab_size=vocab_size, markov_rank=rank).cuda().to(dtype)
    with torch.no_grad():
        head.markov_w1.weight.zero_()
        head.markov_w2.weight.zero_()
    base_logits = torch.zeros(1, 1, vocab_size, device="cuda", dtype=dtype)
    base_logits[..., tie_indices[0]] = 1
    base_logits[..., tie_indices[1]] = 1
    first_prev = torch.zeros(1, device="cuda", dtype=torch.long)

    eager_tokens, _ = head.sample_block(
        base_logits,
        first_prev_tokens=first_prev,
        hidden_states=None,
        sampler=greedy_step_sampler,
    )
    fused_tokens = head.sample_block_fused(
        base_logits,
        first_prev_tokens=first_prev,
        hidden_states=None,
        scratch=_scratch(head, 1, dtype),
    )
    assert eager_tokens.item() == min(tie_indices)
    assert torch.equal(eager_tokens, fused_tokens)


class _DraftModel:
    def __init__(self, head, base_logits):
        self.markov_head = head
        self.base_logits = base_logits

    def compute_base_logits(self, hidden_states):
        del hidden_states
        return self.base_logits, None


@pytest.mark.parametrize("head_cls", [VanillaMarkov, RNNHead])
def test_vanilla_and_rnn_dispatch_to_fused(head_cls):
    dtype = torch.bfloat16
    vocab_size, rank, hidden_size, gamma = 32, 8, 16, 2
    kwargs = dict(vocab_size=vocab_size, markov_rank=rank)
    if head_cls is RNNHead:
        kwargs["hidden_size"] = hidden_size
    head = head_cls(**kwargs).cuda().to(dtype).eval()
    base_logits = torch.randn(gamma, vocab_size, device="cuda", dtype=dtype)
    model = _DraftModel(head, base_logits)
    sampler = DsparkDraftSampler(
        model=model, gamma=gamma, max_bs=1, device=torch.device("cuda")
    )
    original = head.sample_block_fused
    head.sample_block_fused = Mock(wraps=original)
    hidden = torch.randn(gamma, hidden_size, device="cuda", dtype=dtype)
    sampler(hidden, torch.zeros(gamma, device="cuda", dtype=torch.long))
    head.sample_block_fused.assert_called_once()


def test_gated_dispatch_uses_eager_sampling():
    torch.manual_seed(7)
    dtype = torch.bfloat16
    vocab_size, rank, hidden_size, gamma = 32, 8, 16, 3
    head = GatedMarkovHead(
        vocab_size=vocab_size, markov_rank=rank, hidden_size=hidden_size
    ).cuda().to(dtype).eval()
    base_logits = torch.randn(gamma, vocab_size, device="cuda", dtype=dtype)
    hidden = torch.randn(gamma, hidden_size, device="cuda", dtype=dtype)
    input_ids = torch.randint(0, vocab_size, (gamma,), device="cuda")
    expected, _ = head.sample_block(
        base_logits.view(1, gamma, -1),
        first_prev_tokens=input_ids.view(1, gamma)[:, 0],
        hidden_states=hidden.view(1, gamma, -1),
        sampler=greedy_step_sampler,
    )
    head.sample_block_fused = Mock(side_effect=AssertionError("must not fuse gated"))
    sampler = DsparkDraftSampler(
        model=_DraftModel(head, base_logits),
        gamma=gamma,
        max_bs=1,
        device=torch.device("cuda"),
    )
    sampler(hidden, input_ids)
    head.sample_block_fused.assert_not_called()
    assert torch.equal(sampler.out[:gamma].view(1, gamma), expected)


@pytest.mark.parametrize("head_cls", [VanillaMarkov, RNNHead])
def test_draft_sampler_batched_matches_eager(head_cls):
    """bs>1 走 fused：anchor 是 input_ids.view(bs, gamma)[:, 0] 的跨步视图，
    kernel 必须按 stride 读取，否则除第 0 行外都会取到错误的 prev token。"""
    torch.manual_seed(13)
    dtype = torch.bfloat16
    vocab_size, rank, hidden_size, gamma, bs = 512, 32, 64, 15, 8
    mask_token_id = vocab_size - 1
    kwargs = dict(vocab_size=vocab_size, markov_rank=rank)
    if head_cls is RNNHead:
        kwargs["hidden_size"] = hidden_size
    head = head_cls(**kwargs).cuda().to(dtype).eval()

    base_logits = torch.randn(bs * gamma, vocab_size, device="cuda", dtype=dtype)
    hidden = torch.randn(bs * gamma, hidden_size, device="cuda", dtype=dtype)
    # production 布局：每行只有第 0 列是真实 bonus token，其余是 mask token
    ids_2d = torch.full((bs, gamma), mask_token_id, device="cuda", dtype=torch.long)
    ids_2d[:, 0] = torch.randint(0, vocab_size - 1, (bs,), device="cuda")
    input_ids = ids_2d.flatten()

    expected, _ = head.sample_block(
        base_logits.view(bs, gamma, -1),
        first_prev_tokens=ids_2d[:, 0].contiguous(),
        hidden_states=hidden.view(bs, gamma, -1),
        sampler=greedy_step_sampler,
    )
    sampler = DsparkDraftSampler(
        model=_DraftModel(head, base_logits),
        gamma=gamma,
        max_bs=bs,
        device=torch.device("cuda"),
    )
    sampler(hidden, input_ids)
    assert torch.equal(sampler.out[: bs * gamma].view(bs, gamma), expected)


@pytest.mark.parametrize("head_cls", [VanillaMarkov, RNNHead, GatedMarkovHead])
def test_draft_sampler_cuda_graph_capture_replay(head_cls):
    torch.manual_seed(11)
    dtype = torch.bfloat16
    vocab_size, rank, hidden_size, gamma = 64, 8, 16, 3
    kwargs = dict(vocab_size=vocab_size, markov_rank=rank)
    if head_cls is not VanillaMarkov:
        kwargs["hidden_size"] = hidden_size
    head = head_cls(**kwargs).cuda().to(dtype).eval()
    base_logits = torch.randn(gamma, vocab_size, device="cuda", dtype=dtype)
    sampler = DsparkDraftSampler(
        model=_DraftModel(head, base_logits),
        gamma=gamma,
        max_bs=1,
        device=torch.device("cuda"),
    )
    hidden = torch.randn(gamma, hidden_size, device="cuda", dtype=dtype)
    input_ids = torch.zeros(gamma, device="cuda", dtype=torch.long)
    sampler(hidden, input_ids)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampler(hidden, input_ids)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all((sampler.out[:gamma] >= 0) & (sampler.out[:gamma] < vocab_size))
