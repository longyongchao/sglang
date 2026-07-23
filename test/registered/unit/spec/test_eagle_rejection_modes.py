import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_utils import renorm_draft_probs
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


if __name__ == "__main__":
    unittest.main()
