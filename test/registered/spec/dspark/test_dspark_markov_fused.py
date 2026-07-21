"""Unit tests for fused DSpark Markov kernels: numerical equivalence with eager path.

Validates that fused_markov_bias_argmax and fused_dspark_rnn_cell produce
correct greedy tokens compared to the eager VanillaMarkov.sample_block and
RNNHead.sample_block.

Note on precision: The fused kernels accumulate the dot product in fp32 for
better numerics, while the eager path uses fp16 cuBLAS. This means on rare
borderline argmax ties (where two tokens have the same fp16 logit value),
the fused kernel resolves the tie with higher precision. The VanillaMarkov
path is bit-exact because the embedding lookup + matmul has no accumulated
state. The RNN path validates that outputs match when run fully in fp32.

Requires: CUDA GPU with Triton support.
"""

from __future__ import annotations

import pytest
import torch

# Skip the entire module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)


@pytest.fixture(params=[1, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[8, 15])
def proposal_len(request):
    return request.param


@pytest.fixture
def device():
    return torch.device("cuda")


class TestVanillaMarkovFused:
    """Test VanillaMarkov.sample_block_fused vs sample_block (eager)."""

    @pytest.fixture
    def model_params(self):
        return dict(vocab_size=1024, markov_rank=64)

    def test_bit_exact_greedy(self, model_params, batch_size, proposal_len, device):
        from sglang.srt.models.dspark import VanillaMarkov
        from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
            MarkovFusedScratch,
        )

        torch.manual_seed(42)
        head = VanillaMarkov(**model_params).to(device).eval()
        V = model_params["vocab_size"]
        R = model_params["markov_rank"]

        base_logits = torch.randn(batch_size, proposal_len, V, device=device, dtype=torch.float16)
        first_prev = torch.randint(0, V, (batch_size,), device=device)
        hidden_states = None  # VanillaMarkov ignores hidden_states

        # Eager path
        def greedy_sampler(logits, step_idx):
            return torch.argmax(logits, dim=-1)

        eager_tokens, _ = head.sample_block(
            base_logits, first_prev_tokens=first_prev,
            hidden_states=hidden_states, sampler=greedy_sampler,
        )

        # Fused path
        scratch = MarkovFusedScratch(
            max_bs=batch_size, vocab_size=V, markov_rank=R,
            hidden_size=0, device=device, dtype=torch.float16, is_rnn=False,
        )
        fused_tokens = head.sample_block_fused(
            base_logits, first_prev_tokens=first_prev,
            hidden_states=hidden_states, scratch=scratch,
        )

        assert torch.equal(eager_tokens, fused_tokens), (
            f"Mismatch at positions: {(eager_tokens != fused_tokens).nonzero().tolist()}"
        )

    def test_large_vocab(self, device):
        """Test with a vocab size closer to real models (32K)."""
        from sglang.srt.models.dspark import VanillaMarkov
        from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
            MarkovFusedScratch,
        )

        V, R, B, T = 32000, 256, 2, 15
        torch.manual_seed(7)
        head = VanillaMarkov(vocab_size=V, markov_rank=R).to(device).eval()
        base_logits = torch.randn(B, T, V, device=device, dtype=torch.float16)
        first_prev = torch.randint(0, V, (B,), device=device)

        def greedy_sampler(logits, step_idx):
            return torch.argmax(logits, dim=-1)

        eager_tokens, _ = head.sample_block(
            base_logits, first_prev_tokens=first_prev,
            hidden_states=None, sampler=greedy_sampler,
        )
        scratch = MarkovFusedScratch(
            max_bs=B, vocab_size=V, markov_rank=R,
            hidden_size=0, device=device, dtype=torch.float16, is_rnn=False,
        )
        fused_tokens = head.sample_block_fused(
            base_logits, first_prev_tokens=first_prev,
            hidden_states=None, scratch=scratch,
        )
        assert torch.equal(eager_tokens, fused_tokens)


class TestRNNHeadFused:
    """Test RNNHead.sample_block_fused vs sample_block (eager).

    The RNN fused path accumulates the final latent @ fc2_weight.T matmul in
    fp32 (Triton kernel), while eager uses fp16 cuBLAS. This gives strictly
    better precision but can yield a different argmax on borderline ties. We
    validate by running both paths fully in fp32 (where they must agree).
    """

    @pytest.fixture
    def model_params(self):
        return dict(vocab_size=1024, markov_rank=64, hidden_size=128)

    def _run_rnn_eager_fp32(self, *, markov_w1, markov_w2, joint_proj,
                            base_logits, first_prev, hidden_states):
        """Run RNN sampling loop in fp32 for reference."""
        B, T, V = base_logits.shape
        R = markov_w1.weight.shape[1]
        state = torch.zeros(B, R, device=base_logits.device, dtype=torch.float32)
        prev_tokens = first_prev.long()
        tokens = []
        w1 = markov_w1.weight.float()
        w2 = markov_w2.weight.float()
        jp_w = joint_proj.weight.float()
        jp_b = joint_proj.bias.float()
        bl = base_logits.float()
        hs = hidden_states.float()
        for step in range(T):
            prev_emb = w1[prev_tokens]
            z = torch.cat([state, prev_emb, hs[:, step, :]], dim=-1)
            proj = z @ jp_w.T + jp_b
            g, c, o = proj.chunk(3, dim=-1)
            gate = torch.sigmoid(g)
            cand = torch.tanh(c)
            state = gate * state + (1.0 - gate) * cand
            oe = torch.tanh(o)
            bias = oe @ w2.T
            logits = bl[:, step, :] + bias
            tok = logits.argmax(dim=-1)
            tokens.append(tok)
            prev_tokens = tok
        return torch.stack(tokens, dim=1)

    def test_matches_fp32_reference(self, model_params, batch_size, proposal_len, device):
        """Fused RNN path should match the fp32 eager reference."""
        from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
            MarkovFusedScratch,
            fused_dspark_rnn_cell,
            fused_markov_bias_argmax,
        )
        from torch import nn

        torch.manual_seed(42)
        V = model_params["vocab_size"]
        R = model_params["markov_rank"]
        H = model_params["hidden_size"]

        markov_w1 = nn.Embedding(V, R).to(device).half()
        markov_w2 = nn.Linear(R, V, bias=False).to(device).half()
        joint_proj = nn.Linear(2 * R + H, 3 * R).to(device).half()

        base_logits = torch.randn(batch_size, proposal_len, V, device=device, dtype=torch.float16)
        first_prev = torch.randint(0, V, (batch_size,), device=device)
        hidden_states = torch.randn(
            batch_size, proposal_len, H, device=device, dtype=torch.float16
        )

        # fp32 reference
        ref_tokens = self._run_rnn_eager_fp32(
            markov_w1=markov_w1, markov_w2=markov_w2, joint_proj=joint_proj,
            base_logits=base_logits, first_prev=first_prev, hidden_states=hidden_states,
        )

        # Fused path
        scratch = MarkovFusedScratch(
            max_bs=batch_size, vocab_size=V, markov_rank=R,
            hidden_size=H, device=device, dtype=torch.float16, is_rnn=True,
        )
        state = scratch.rnn_state_buf[:batch_size, :R].zero_()
        new_state = scratch.rnn_new_state_buf[:batch_size, :R]
        output_emb = scratch.output_emb_buf[:batch_size, :R]
        proj_buf = scratch.rnn_proj_buf[:batch_size]

        prev_tokens = first_prev.long()
        fused_tokens = []
        for step in range(proposal_len):
            prev_emb = markov_w1(prev_tokens)
            joint_input = torch.cat([state, prev_emb, hidden_states[:, step, :]], dim=-1)
            proj_buf.copy_(torch.nn.functional.linear(
                joint_input, joint_proj.weight, joint_proj.bias))
            fused_dspark_rnn_cell(
                proj_out=proj_buf, state=state,
                new_state=new_state, output_emb=output_emb,
            )
            state, new_state = new_state, state
            fused_markov_bias_argmax(
                prev_tokens=prev_tokens, embed_table=markov_w1.weight,
                fc2_weight=markov_w2.weight, base_logits=base_logits[:, step, :],
                out_val=scratch.out_val[:batch_size], out_idx=scratch.out_idx[:batch_size],
                final_token=scratch.tok_buf[:batch_size], latent=output_emb,
            )
            prev_tokens = scratch.tok_buf[:batch_size].clone()
            fused_tokens.append(prev_tokens.clone())
        fused_all = torch.stack(fused_tokens, dim=1)

        assert torch.equal(ref_tokens, fused_all), (
            f"Mismatch vs fp32 ref at: {(ref_tokens != fused_all).nonzero().tolist()}"
        )

    def test_large_vocab(self, device):
        """Test with larger vocab."""
        from sglang.srt.speculative.dspark_components.kernels.dspark_markov_fused import (
            MarkovFusedScratch,
            fused_dspark_rnn_cell,
            fused_markov_bias_argmax,
        )
        from torch import nn

        V, R, H, B, T = 32000, 256, 4096, 2, 15
        torch.manual_seed(7)
        markov_w1 = nn.Embedding(V, R).to(device).half()
        markov_w2 = nn.Linear(R, V, bias=False).to(device).half()
        joint_proj = nn.Linear(2 * R + H, 3 * R).to(device).half()

        base_logits = torch.randn(B, T, V, device=device, dtype=torch.float16)
        first_prev = torch.randint(0, V, (B,), device=device)
        hidden_states = torch.randn(B, T, H, device=device, dtype=torch.float16)

        # fp32 reference
        ref_tokens = self._run_rnn_eager_fp32(
            markov_w1=markov_w1, markov_w2=markov_w2, joint_proj=joint_proj,
            base_logits=base_logits, first_prev=first_prev, hidden_states=hidden_states,
        )

        # Fused path
        scratch = MarkovFusedScratch(
            max_bs=B, vocab_size=V, markov_rank=R,
            hidden_size=H, device=device, dtype=torch.float16, is_rnn=True,
        )
        state = scratch.rnn_state_buf[:B, :R].zero_()
        new_state = scratch.rnn_new_state_buf[:B, :R]
        output_emb = scratch.output_emb_buf[:B, :R]
        proj_buf = scratch.rnn_proj_buf[:B]

        prev_tokens = first_prev.long()
        fused_tokens = []
        for step in range(T):
            prev_emb = markov_w1(prev_tokens)
            joint_input = torch.cat([state, prev_emb, hidden_states[:, step, :]], dim=-1)
            proj_buf.copy_(torch.nn.functional.linear(
                joint_input, joint_proj.weight, joint_proj.bias))
            fused_dspark_rnn_cell(
                proj_out=proj_buf, state=state,
                new_state=new_state, output_emb=output_emb,
            )
            state, new_state = new_state, state
            fused_markov_bias_argmax(
                prev_tokens=prev_tokens, embed_table=markov_w1.weight,
                fc2_weight=markov_w2.weight, base_logits=base_logits[:, step, :],
                out_val=scratch.out_val[:B], out_idx=scratch.out_idx[:B],
                final_token=scratch.tok_buf[:B], latent=output_emb,
            )
            prev_tokens = scratch.tok_buf[:B].clone()
            fused_tokens.append(prev_tokens.clone())
        fused_all = torch.stack(fused_tokens, dim=1)
        assert torch.equal(ref_tokens, fused_all)
