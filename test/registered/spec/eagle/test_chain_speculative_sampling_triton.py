"""Reference tests for the classic (Leviathan) chain rejection-sampling kernel.

The kernel under test is `chain_speculative_sampling_triton`, the only path that
reads the draft proposal distribution `q`. Row alignment (slot `step` verifies
against `target_probs[step - 1]` / `draft_probs[step - 1]`) degrades *silently*
when broken -- a shifted `q` still yields plausible text -- so the expected
values here are computed by an independent port of the DeepSpec evaluator
(`deepspec/eval/base_evaluator.py`, `deepspec/utils/sampling.py`) with the
randomness pinned, and compared token by token.

Two deliberate divergences from DeepSpec are encoded below:

1. Degenerate residual (`relu(p - q)` sums to 0 on rejection): DeepSpec falls
   back to `target_probs`, the kernel falls back to `VOCAB_SIZE - 1`.
2. DeepSpec clamps `q` to 1e-8 and compares `rand < min(1, p / q)`; the kernel
   compares `coin * q < p`. Equivalent for `q > 0`, and both reject `p == 0`.
"""

import unittest

import torch

from sglang.srt.speculative.reject_sampling import chain_speculative_sampling_triton
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-a")


def deepspec_chain_reference(
    candidates,
    retrieve_index,
    coins,
    final_coins,
    target_probs,
    draft_probs,
):
    """Leviathan chain verification, ported from the DeepSpec evaluator.

    `accept_prob = min(1, p / q)` and `residual = relu(p - q) / Z` are DeepSpec's
    formulation. Residual sampling uses the inverse CDF (first token whose
    cumulative mass strictly exceeds `final_coin * Z`) instead of DeepSpec's
    `multinomial`, so that a fixed coin pins a single expected token.

    Returns `(predicts, accept_index, num_correct_drafts)` on CPU.
    """
    bs, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    predicts = torch.full((bs * num_slots,), -1, dtype=torch.int32)
    accept_index = torch.full((bs, num_slots), -1, dtype=torch.int32)
    num_correct_drafts = torch.zeros((bs,), dtype=torch.int32)

    for b in range(bs):
        prob_row = 0
        last_global_idx = int(retrieve_index[b, 0])
        accept_index[b, 0] = last_global_idx
        num_accept = 0
        all_accepted = True

        for step in range(1, num_slots):
            draft_token = int(candidates[b, step])
            p = float(target_probs[b, prob_row, draft_token])
            q = float(draft_probs[b, prob_row, draft_token])
            accept_prob = min(1.0, p / max(q, 1e-8))
            if float(coins[b, step - 1]) < accept_prob:
                num_accept += 1
                prob_row = step
                predicts[last_global_idx] = draft_token
                last_global_idx = int(retrieve_index[b, step])
                accept_index[b, num_accept] = last_global_idx
            else:
                all_accepted = False
                break

        num_correct_drafts[b] = num_accept

        if all_accepted:
            residual = target_probs[b, prob_row].clone().float()
        else:
            residual = (
                (target_probs[b, prob_row] - draft_probs[b, prob_row])
                .clamp_min(0.0)
                .float()
            )
        threshold = float(final_coins[b]) * float(residual.sum())
        # Kernel fallback when nothing exceeds the threshold (degenerate mass).
        final_token = vocab_size - 1
        cumulative = 0.0
        for token in range(vocab_size):
            cumulative += float(residual[token])
            if cumulative > threshold:
                final_token = token
                break
        predicts[last_global_idx] = final_token

    return predicts, accept_index, num_correct_drafts


def run_kernel(
    candidates, retrieve_index, coins, final_coins, target_probs, draft_probs
):
    bs, num_slots = candidates.shape
    device = target_probs.device
    predicts = torch.full((bs * num_slots,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_slots), -1, dtype=torch.int32, device=device)
    num_correct_drafts = torch.zeros((bs,), dtype=torch.int32, device=device)
    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=num_correct_drafts,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=final_coins,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    return predicts, accept_index, num_correct_drafts


def _chain_index(bs, num_slots, device):
    return (
        torch.arange(bs * num_slots, dtype=torch.int64, device=device)
        .reshape(bs, num_slots)
        .contiguous()
    )


class TestChainSpeculativeSamplingTriton(CustomTestCase):
    """bs=3, num_steps=3 (4 slots), vocab=8. All masses are multiples of 1/4 so
    the sequential Python cumsum and the kernel's blocked `tl.cumsum` agree
    exactly in fp32."""

    def _pinned_case(self):
        vocab_size = 8
        num_slots = 4
        bs = 3

        candidates = torch.tensor(
            [
                [5, 1, 2, 3],  # accepts all three drafts
                [5, 4, 1, 1],  # rejects the first draft
                [7, 2, 6, 0],  # accepts one, rejects the second
            ],
            dtype=torch.int64,
        )
        target_probs = torch.zeros(bs, num_slots, vocab_size)
        draft_probs = torch.full((bs, num_slots - 1, vocab_size), 1.0 / vocab_size)

        # Batch 0: p >= q on every proposed token, so every coin accepts.
        target_probs[0, 0, [1, 2, 3]] = torch.tensor([0.5, 0.25, 0.25])
        draft_probs[0, 0] = 0.0
        draft_probs[0, 0, [1, 5, 6]] = torch.tensor([0.5, 0.25, 0.25])
        target_probs[0, 1, [2, 3]] = torch.tensor([0.75, 0.25])
        draft_probs[0, 1] = 0.0
        draft_probs[0, 1, [2, 3]] = torch.tensor([0.5, 0.5])
        target_probs[0, 2, 3] = 1.0
        draft_probs[0, 2] = 0.0
        draft_probs[0, 2, 3] = 1.0
        # Bonus comes from pure p on the last row (all drafts accepted).
        target_probs[0, 3, [0, 7]] = torch.tensor([0.5, 0.5])

        # Batch 1: p[4]=0.25 vs q[4]=0.75 -> accept_prob 1/3, coin 0.9 rejects.
        target_probs[1, 0, [3, 4, 5]] = torch.tensor([0.25, 0.25, 0.5])
        draft_probs[1, 0] = 0.0
        draft_probs[1, 0, [4, 5]] = torch.tensor([0.75, 0.25])

        # Batch 2: accept token 2 (p == q == 0.5), then reject token 6.
        target_probs[2, 0, [2, 6, 7]] = torch.tensor([0.5, 0.25, 0.25])
        draft_probs[2, 0] = 0.0
        draft_probs[2, 0, [2, 6]] = torch.tensor([0.5, 0.5])
        target_probs[2, 1, [0, 6]] = torch.tensor([0.5, 0.5])
        draft_probs[2, 1] = 0.0
        draft_probs[2, 1, 6] = 1.0

        coins = torch.tensor(
            [
                [0.9, 0.9, 0.9, 0.0],
                [0.9, 0.0, 0.0, 0.0],
                [0.5, 0.75, 0.0, 0.0],
            ]
        )
        final_coins = torch.tensor([0.8, 0.3, 0.5])
        return candidates, coins, final_coins, target_probs, draft_probs

    def test_matches_deepspec_chain_reference(self):
        candidates, coins, final_coins, target_probs, draft_probs = self._pinned_case()
        bs, num_slots = candidates.shape
        retrieve_index = _chain_index(bs, num_slots, "cpu")

        expected = deepspec_chain_reference(
            candidates, retrieve_index, coins, final_coins, target_probs, draft_probs
        )

        actual = run_kernel(
            candidates.cuda(),
            retrieve_index.cuda(),
            coins.cuda(),
            final_coins.cuda(),
            target_probs.cuda(),
            draft_probs.cuda(),
        )

        self.assertEqual(actual[0].cpu().tolist(), expected[0].tolist())
        self.assertEqual(actual[1].cpu().tolist(), expected[1].tolist())
        self.assertEqual(actual[2].cpu().tolist(), expected[2].tolist())

        # Guard the reference itself: hand-computed outcomes per batch.
        predicts, accept_index, num_correct_drafts = (t.cpu() for t in actual)
        self.assertEqual(num_correct_drafts.tolist(), [3, 0, 1])
        # Batch 0 emits its three drafts, bonus from pure p (0.8 * 1.0 -> token 7).
        self.assertEqual(predicts[0:4].tolist(), [1, 2, 3, 7])
        self.assertEqual(accept_index[0].tolist(), [0, 1, 2, 3])
        # Batch 1 rejects immediately; residual relu(p-q) = [.,.,.,0.25,0,0.25,.,.]
        # normalizes over Z=0.5, and 0.3 * 0.5 = 0.15 < 0.25 lands on token 3.
        self.assertEqual(int(predicts[4]), 3)
        self.assertEqual(accept_index[1].tolist(), [4, -1, -1, -1])
        # Batch 2 accepts token 2, then residual at row 1 is all mass on token 0.
        self.assertEqual(predicts[8:10].tolist(), [2, 0])
        self.assertEqual(accept_index[2].tolist(), [8, 9, -1, -1])

    def test_multiplicative_and_ratio_accept_tests_agree(self):
        """`coin * q < p` (kernel) vs `coin < min(1, p / q)` (DeepSpec) -- the
        second deliberate divergence. Equivalent on the pinned data."""
        candidates, coins, final_coins, target_probs, draft_probs = self._pinned_case()
        bs, num_slots = candidates.shape
        for b in range(bs):
            prob_row = 0
            for step in range(1, num_slots):
                token = int(candidates[b, step])
                p = float(target_probs[b, prob_row, token])
                q = float(draft_probs[b, prob_row, token])
                coin = float(coins[b, step - 1])
                self.assertEqual(
                    coin * q < p, coin < min(1.0, p / max(q, 1e-8)), (b, step)
                )
                if not coin * q < p:
                    break
                prob_row = step

    def test_row_alignment_is_not_shift_invariant(self):
        """A one-row shift of `q` must change the outcome, otherwise the pinned
        case above could not detect a misaligned `draft_probs`."""
        candidates, coins, final_coins, target_probs, draft_probs = self._pinned_case()
        bs, num_slots = candidates.shape
        retrieve_index = _chain_index(bs, num_slots, "cuda")
        shifted = draft_probs.roll(shifts=1, dims=1)

        aligned = run_kernel(
            candidates.cuda(),
            retrieve_index,
            coins.cuda(),
            final_coins.cuda(),
            target_probs.cuda(),
            draft_probs.cuda(),
        )
        misaligned = run_kernel(
            candidates.cuda(),
            retrieve_index,
            coins.cuda(),
            final_coins.cuda(),
            target_probs.cuda(),
            shifted.cuda(),
        )
        self.assertNotEqual(aligned[2].cpu().tolist(), misaligned[2].cpu().tolist())

    def test_degenerate_residual_falls_back_to_last_token(self):
        """Divergence 1: zero residual mass on rejection. Unreachable with two
        normalized distributions (`p <= q` everywhere forces `p == q`, which
        always accepts), so it is pinned with an all-zero target row. DeepSpec
        would fall back to `target_probs`; the kernel emits `VOCAB_SIZE - 1`."""
        vocab_size = 8
        candidates = torch.tensor([[5, 4]], dtype=torch.int64).cuda()
        retrieve_index = _chain_index(1, 2, "cuda")
        target_probs = torch.zeros(1, 2, vocab_size).cuda()
        draft_probs = torch.zeros(1, 1, vocab_size).cuda()
        draft_probs[0, 0, 4] = 1.0
        coins = torch.zeros(1, 2).cuda()
        final_coins = torch.zeros(1).cuda()

        predicts, accept_index, num_correct_drafts = run_kernel(
            candidates, retrieve_index, coins, final_coins, target_probs, draft_probs
        )
        self.assertEqual(int(num_correct_drafts[0]), 0)
        self.assertEqual(int(predicts[0]), vocab_size - 1)
        self.assertEqual(accept_index[0].cpu().tolist(), [0, -1])

    def test_first_emitted_token_matches_target_distribution(self):
        """Unbiasedness: with one draft step the token emitted at slot 0 is the
        accepted draft or a residual draw, and must be distributed as `p`."""
        torch.manual_seed(0)
        vocab_size = 16
        num_slots = 2
        bs = 200_000
        device = "cuda"

        p = torch.softmax(torch.randn(vocab_size), dim=-1).to(device)
        q = torch.softmax(torch.randn(vocab_size), dim=-1).to(device)

        target_probs = p.expand(bs, num_slots, vocab_size).contiguous()
        draft_probs = q.expand(bs, num_slots - 1, vocab_size).contiguous()
        candidates = torch.zeros(bs, num_slots, dtype=torch.int64, device=device)
        candidates[:, 1] = torch.multinomial(q, bs, replacement=True)
        retrieve_index = _chain_index(bs, num_slots, device)
        coins = torch.rand(bs, num_slots, device=device)
        final_coins = torch.rand(bs, device=device)

        predicts, _, _ = run_kernel(
            candidates, retrieve_index, coins, final_coins, target_probs, draft_probs
        )
        emitted = predicts.reshape(bs, num_slots)[:, 0].to(torch.int64)
        empirical = torch.bincount(emitted, minlength=vocab_size).float() / bs
        total_variation = 0.5 * (empirical - p).abs().sum().item()
        self.assertLess(total_variation, 0.01, f"{total_variation=}")


if __name__ == "__main__":
    unittest.main()
