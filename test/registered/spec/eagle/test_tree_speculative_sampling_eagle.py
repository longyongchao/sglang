import unittest

import torch
from sgl_kernel import tree_speculative_sampling_eagle
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-a")


def official_reference(candidates, next_token, next_sibling, coins, bonus_coins, probs, depth):
    bs, width = candidates.shape
    predicts = torch.full((bs * width,), -1, dtype=torch.int32)
    accept_index = torch.full((bs, depth), -1, dtype=torch.int32)
    counts = torch.zeros(bs, dtype=torch.int32)
    for b in range(bs):
        base = b * width
        current = 0
        output = base
        accept_index[b, 0] = output
        rejected = torch.zeros_like(probs[b])
        for level in range(1, depth):
            node = int(next_token[b, current])
            remaining = 1.0
            found = False
            while node != -1:
                token = int(candidates[b, node])
                p = float(probs[b, current, token])
                if float(coins[b, node]) * remaining <= p:
                    predicts[output] = token
                    counts[b] += 1
                    output = base + node
                    accept_index[b, level] = output
                    current = node
                    found = True
                    break
                rejected[current, token] = p
                remaining = max(remaining - p, 0.0)
                node = int(next_sibling[b, node])
            if not found:
                break
        residual = probs[b, current].clone()
        if int(counts[b]) != depth - 1:
            residual -= rejected[current]
            residual.clamp_(min=0)
        threshold = float(bonus_coins[b]) * float(residual.sum())
        cumulative = 0.0
        valid = torch.nonzero(residual > 0).flatten()
        sampled = int(valid[-1]) if valid.numel() else residual.numel() - 1
        for token, value in enumerate(residual):
            cumulative += float(value)
            if value > 0 and cumulative >= threshold:
                sampled = token
                break
        predicts[output] = sampled
    return predicts, accept_index, counts


class TestTreeSpeculativeSamplingEagle(CustomTestCase):
    def test_matches_official_tree_reference(self):
        candidates = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]])
        next_token = torch.tensor([[1, 4, 6, -1, -1, -1, -1]] * 2)
        next_sibling = torch.tensor([[-1, 2, 3, -1, 5, -1, -1]] * 2)
        coins = torch.tensor([[0, .9, .9, .2, 0, 0, 0], [0, .5, 0, 0, .4, 0, 0]])
        bonus_coins = torch.tensor([.8, .3])
        probs = torch.zeros(2, 7, 10)
        probs[0, 0, 1:4] = torch.tensor([.2, .3, .15])
        probs[0, 0, 9] = .35
        probs[0, 3, 9] = 1
        probs[1, 0, 1] = .5
        probs[1, 0, 8] = .5
        probs[1, 1, 4] = .4
        probs[1, 1, 9] = .6
        probs[1, 4, 7] = 1
        expected = official_reference(
            candidates, next_token, next_sibling, coins, bonus_coins, probs, 3
        )
        device = "cuda"
        candidates = candidates.to(device=device, dtype=torch.int64)
        next_token = next_token.to(device=device, dtype=torch.int64)
        next_sibling = next_sibling.to(device=device, dtype=torch.int64)
        coins = coins.to(device=device, dtype=torch.float32)
        bonus_coins = bonus_coins.to(device=device, dtype=torch.float32)
        probs = probs.to(device=device, dtype=torch.float32)
        predicts = torch.full((14,), -1, dtype=torch.int32, device=device)
        accept_index = torch.full((2, 3), -1, dtype=torch.int32, device=device)
        counts = torch.zeros(2, dtype=torch.int32, device=device)
        tree_speculative_sampling_eagle(
            predicts, accept_index, counts, candidates,
            torch.arange(14, device=device, dtype=torch.int64).reshape(2, 7),
            next_token, next_sibling, coins, bonus_coins, probs,
            torch.zeros_like(probs),
        )
        self.assertEqual(predicts.cpu().tolist(), expected[0].tolist())
        self.assertEqual(accept_index.cpu().tolist(), expected[1].tolist())
        self.assertEqual(counts.cpu().tolist(), expected[2].tolist())
        self.assertEqual(int(predicts[3]), 9)
        self.assertEqual(int(predicts[11]), 7)

    def test_reference_suppresses_duplicate_siblings(self):
        ordered = [3, 4, 3, 5, 4]
        self.assertEqual(list(dict.fromkeys(ordered)), [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
