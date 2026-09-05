import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_utils import (
    renorm_draft_probs,
    resolve_rejection_sampling_mode,
    scatter_draft_probs_to_target_vocab,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEagleRejectionModes(CustomTestCase):
    def _worker_for_mapping(self, mapping, draft_vocab=4, target_vocab=8):
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.hot_token_id = mapping
        head = torch.empty(draft_vocab, 2)
        worker.draft_runner = SimpleNamespace(
            model=SimpleNamespace(get_embed_and_head=lambda: (None, head))
        )
        worker.target_worker = SimpleNamespace(
            model_config=SimpleNamespace(vocab_size=target_vocab)
        )
        return worker

    def test_valid_mapping(self):
        self._worker_for_mapping(torch.tensor([7, 1, 5, 3]))._validate_hot_token_id()

    def test_missing_mapping_is_allowed_for_equal_vocab(self):
        self._worker_for_mapping(None, draft_vocab=8)._validate_hot_token_id()

    def test_mapping_validation_errors(self):
        cases = [
            torch.tensor([[0, 1], [2, 3]]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            torch.tensor([0, 1, 2]),
            torch.tensor([-1, 1, 2, 3]),
            torch.tensor([0, 1, 2, 8]),
            torch.tensor([0, 1, 1, 3]),
        ]
        for mapping in cases:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                self._worker_for_mapping(mapping)._validate_hot_token_id()

    def _worker_for_decode_extend(self, *, classic, eagle3, topk):
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.use_classic_rejection_sampling = classic
        worker.use_eagle3_rejection_sampling = eagle3
        worker.topk = topk
        worker.device = "cpu"
        worker.speculative_draft_temperature = None
        worker.speculative_num_draft_tokens = 1
        worker.cuda_graph_runner_for_draft_extend = None
        worker.plan_stream = None
        worker.plan_stream_ctx = nullcontext()
        return worker

    def _decode_extend_inputs(self):
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        output = SimpleNamespace(next_token_logits=logits, hidden_states=None)
        next_input = SimpleNamespace(draft_probs=torch.ones(1))
        result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=None),
            accept_lens=torch.ones(1, dtype=torch.int32),
            next_token_ids=torch.zeros(1, dtype=torch.int32),
            next_draft_input=next_input,
        )
        batch = SimpleNamespace(
            seq_lens=torch.ones(1, dtype=torch.int32), sampling_info=SimpleNamespace()
        )
        return logits, output, next_input, result, batch

    def test_tree_rejection_scales_each_branch_by_request_temperature(self):
        logits = torch.tensor(
            [
                [2.0, 0.0],
                [4.0, 0.0],
                [6.0, 0.0],
                [2.0, 0.0],
                [4.0, 0.0],
                [6.0, 0.0],
            ]
        )
        sampling_info = SimpleNamespace(temperatures=torch.tensor([[1.0], [2.0]]))

        actual = renorm_draft_probs(logits, sampling_info, True)
        expanded_temperatures = torch.tensor(
            [[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]]
        )
        expected = torch.softmax(logits / expanded_temperatures, dim=-1)

        torch.testing.assert_close(actual, expected)

    def test_tree_rejection_rejects_non_integral_branch_count(self):
        logits = torch.zeros(5, 2)
        sampling_info = SimpleNamespace(temperatures=torch.ones(2, 1))

        with self.assertRaisesRegex(ValueError, "must be a multiple"):
            renorm_draft_probs(logits, sampling_info, True)

    def test_draft_temperature_zero_ignores_request_temperature(self):
        logits = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
        sampling_info = SimpleNamespace(temperatures=torch.tensor([[10.0]]))

        actual = renorm_draft_probs(logits, sampling_info, True, 0.0)

        torch.testing.assert_close(actual, torch.softmax(logits, dim=-1))

    def test_positive_draft_temperature_overrides_request_temperature(self):
        logits = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
        sampling_info = SimpleNamespace(temperatures=torch.tensor([[10.0]]))

        actual = renorm_draft_probs(logits, sampling_info, True, 2.0)

        torch.testing.assert_close(actual, torch.softmax(logits / 2.0, dim=-1))

    def test_eagle3_decode_extend_uses_topk_without_q(self):
        worker = self._worker_for_decode_extend(classic=False, eagle3=True, topk=3)
        logits, output, next_input, result, batch = self._decode_extend_inputs()
        worker.prepare_for_draft_extend = lambda *args: SimpleNamespace()
        worker.draft_runner = SimpleNamespace(
            forward=lambda _: SimpleNamespace(logits_output=output), canary_manager=None
        )
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.renorm_draft_probs",
            return_value=torch.softmax(logits, -1),
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2.fast_topk",
            return_value=(torch.ones(1, 3), torch.tensor([[0, 1, 2]])),
        ) as topk, patch(
            "sglang.srt.speculative.eagle_worker_v2.fast_sample"
        ) as sample:
            worker._draft_extend_for_decode(batch, result)
        topk.assert_called_once()
        sample.assert_not_called()
        self.assertIsNone(next_input.draft_probs)

    def test_classic_decode_extend_samples_and_keeps_q(self):
        worker = self._worker_for_decode_extend(classic=True, eagle3=False, topk=1)
        logits, output, next_input, result, batch = self._decode_extend_inputs()
        probs = torch.softmax(logits, -1)
        worker.prepare_for_draft_extend = lambda *args: SimpleNamespace()
        worker.draft_runner = SimpleNamespace(
            forward=lambda _: SimpleNamespace(logits_output=output), canary_manager=None
        )
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.renorm_draft_probs",
            return_value=probs,
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2.fast_topk"
        ) as topk, patch(
            "sglang.srt.speculative.eagle_worker_v2.fast_sample",
            return_value=(torch.ones(1, 1), torch.tensor([[0]])),
        ) as sample:
            worker._draft_extend_for_decode(batch, result)
        sample.assert_called_once()
        topk.assert_not_called()
        self.assertIs(next_input.draft_probs, probs)

    def _worker_for_verify_probs(self, *, topk, hot_token_id, target_vocab=8):
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.topk = topk
        worker.hot_token_id = hot_token_id
        worker.target_vocab_size = target_vocab
        return worker

    def test_build_verify_draft_probs_lifts_rows_in_order(self):
        """Row i must stay the distribution slot i + 1 was sampled from; the chain
        kernel reads row `step - 1` when verifying slot `step`."""
        worker = self._worker_for_verify_probs(
            topk=1, hot_token_id=torch.tensor([7, 1, 5, 3])
        )
        rows = [
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),  # proposes target token 7
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),  # proposes target token 3
        ]
        draft_tokens = torch.tensor([[7, 3]])

        out = worker._build_verify_draft_probs(rows, draft_tokens)

        self.assertEqual(out.shape, (1, 2, 8))
        self.assertEqual(out[0, 0, 7].item(), 1.0)
        self.assertEqual(out[0, 1, 3].item(), 1.0)
        self.assertEqual(out[0, 0, 3].item(), 0.0)

    def test_build_verify_draft_probs_without_token_map(self):
        worker = self._worker_for_verify_probs(
            topk=1, hot_token_id=None, target_vocab=4
        )
        rows = [torch.tensor([[0.25, 0.25, 0.25, 0.25]])]

        out = worker._build_verify_draft_probs(rows, torch.tensor([[0]]))

        torch.testing.assert_close(out, rows[0].unsqueeze(1))

    def test_build_verify_draft_probs_rejects_tree_topology(self):
        worker = self._worker_for_verify_probs(
            topk=4, hot_token_id=None, target_vocab=4
        )
        with self.assertRaisesRegex(ValueError, "speculative_eagle_topk=1"):
            worker._build_verify_draft_probs(
                [torch.ones(1, 4) / 4], torch.tensor([[0]])
            )


class TestRejectionSamplingModeResolution(CustomTestCase):
    """Truth table for `resolve_rejection_sampling_mode`.

    The rows with `draft_temperature in (None, 0.0)` are the zero-regression
    guarantee: EAGLE3 keeps the official target-only tree kernel there, at both
    topk == 1 and topk > 1, exactly as before the classic chain was wired up.
    """

    def test_truth_table(self):
        # (use_rejection_sampling, is_eagle3, topk, dt) -> (official, classic)
        cases = {}
        for is_eagle3 in (False, True):
            for topk in (1, 4):
                for dt in (None, 0.0, 1.0):
                    cases[(False, is_eagle3, topk, dt)] = (False, False)
        for topk in (1, 4):
            for dt in (None, 0.0, 1.0):
                # Classic EAGLE has no second kernel; it always reads q.
                cases[(True, False, topk, dt)] = (False, True)
        for dt in (None, 0.0):
            # dt == 0 selects tokens by argmax while returning a temperature-1
            # q, so it must not reach the `coin * q < p` accept test.
            cases[(True, True, 1, dt)] = (True, False)
            cases[(True, True, 4, dt)] = (True, False)
        cases[(True, True, 1, 1.0)] = (False, True)
        # A tree has no single q per slot, so dt cannot switch the kernel.
        cases[(True, True, 4, 1.0)] = (True, False)

        self.assertEqual(len(cases), 24)
        for (use_rs, is_eagle3, topk, dt), expected in cases.items():
            with self.subTest(use_rs=use_rs, is_eagle3=is_eagle3, topk=topk, dt=dt):
                self.assertEqual(
                    resolve_rejection_sampling_mode(
                        use_rejection_sampling=use_rs,
                        is_eagle3=is_eagle3,
                        topk=topk,
                        draft_temperature=dt,
                    ),
                    expected,
                )

    def test_modes_are_mutually_exclusive(self):
        for use_rs in (False, True):
            for is_eagle3 in (False, True):
                for topk in (1, 2, 8):
                    for dt in (None, 0.0, 0.5, 100.0):
                        official, classic = resolve_rejection_sampling_mode(
                            use_rejection_sampling=use_rs,
                            is_eagle3=is_eagle3,
                            topk=topk,
                            draft_temperature=dt,
                        )
                        self.assertFalse(official and classic)
                        self.assertEqual(use_rs, official or classic)


class TestScatterDraftProbsToTargetVocab(CustomTestCase):
    """The reduced draft vocabulary is lifted into the target vocabulary exactly:
    the draft can only propose hot tokens, so `q == 0` off-support is the true
    proposal mass, not an approximation."""

    MAPPING = torch.tensor([7, 1, 5, 3])  # deliberately holey and unsorted

    def test_every_position_matches(self):
        draft_probs = torch.tensor(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.0, 0.25, 0.25]],
                [[1.0, 0.0, 0.0, 0.0], [0.4, 0.3, 0.2, 0.1]],
            ]
        )
        lifted = scatter_draft_probs_to_target_vocab(draft_probs, self.MAPPING, 8)

        self.assertEqual(lifted.shape, (2, 2, 8))
        expected = torch.zeros(2, 2, 8)
        for j, target_token in enumerate(self.MAPPING.tolist()):
            expected[..., target_token] = draft_probs[..., j]
        torch.testing.assert_close(lifted, expected)
        # Mass is preserved, and only mapped columns are non-zero.
        torch.testing.assert_close(lifted.sum(-1), draft_probs.sum(-1))
        cold = [t for t in range(8) if t not in self.MAPPING.tolist()]
        self.assertTrue(bool((lifted[..., cold] == 0).all()))

    def test_forces_fp32_so_eager_and_graph_paths_agree(self):
        lifted = scatter_draft_probs_to_target_vocab(
            torch.rand(1, 1, 4, dtype=torch.bfloat16), self.MAPPING, 8
        )
        self.assertEqual(lifted.dtype, torch.float32)

    def test_residual_keeps_full_target_mass_on_cold_tokens(self):
        """`relu(p - q)` on the lifted q: cold tokens keep all of p, so the
        residual distribution stays a valid recovery distribution."""
        q = scatter_draft_probs_to_target_vocab(
            torch.tensor([[[0.5, 0.5, 0.0, 0.0]]]), self.MAPPING, 8
        )
        p = torch.full((1, 1, 8), 0.125)
        residual = (p - q).clamp_min(0.0)

        cold = [t for t in range(8) if t not in self.MAPPING.tolist()]
        torch.testing.assert_close(residual[..., cold], p[..., cold])
        # Tokens 7 and 1 carry q = 0.5 > p = 0.125, so they are fully consumed.
        self.assertEqual(residual[0, 0, 7].item(), 0.0)
        self.assertEqual(residual[0, 0, 1].item(), 0.0)
        # Tokens 5 and 3 are hot but had zero draft mass, so p survives there.
        torch.testing.assert_close(residual[0, 0, 5], p[0, 0, 5])
        torch.testing.assert_close(residual[0, 0, 3], p[0, 0, 3])


if __name__ == "__main__":
    unittest.main()
