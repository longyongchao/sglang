"""CPU unit tests for speculative p/q recorder bonus and draft-entropy fields."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.speculative.spec_pq_recorder import SpecPqRecorder
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeRequest:
    def __init__(self, rid: str) -> None:
        self.rid = rid
        self.origin_input_ids = [1, 2]


def _entropy(logits_row: torch.Tensor) -> float:
    """Reference entropy in nats of softmax over one logits row."""
    logprobs = torch.log_softmax(logits_row.to(torch.float64), dim=-1)
    return float(-(logprobs.exp() * logprobs).sum())


class TestSpecPqRecorder(CustomTestCase):
    def _record_jsonl(
        self,
        *,
        num_correct_drafts: torch.Tensor,
        bonus_tokens: torch.Tensor | None,
        **record_chain_kwargs,
    ) -> list[dict]:
        num_draft = 3
        reqs = [_FakeRequest("req-0"), _FakeRequest("req-1"), _FakeRequest("req-2")]
        with tempfile.TemporaryDirectory() as dump_dir:
            recorder = SpecPqRecorder(
                dump_dir=dump_dir, max_records=0, tag="test", tp_rank=0
            )
            recorder.record_chain(
                forward_ct=7,
                reqs=reqs,
                prefix_lens=torch.tensor([10, 20, 30]),
                draft_tokens=torch.tensor(
                    [[0, 1, 2], [3, 4, 5], [1, 2, 3]], dtype=torch.long
                ),
                num_correct_drafts=num_correct_drafts,
                target_logits=torch.arange(
                    len(reqs) * num_draft * 8, dtype=torch.float32
                ).reshape(len(reqs) * num_draft, 8),
                rows_per_request=num_draft,
                bonus_tokens=bonus_tokens,
                **record_chain_kwargs,
            )
            recorder.flush()
            return [
                json.loads(line)
                for line in Path(recorder.path).read_text().splitlines()
            ]

    def test_bonus_has_one_owner_per_request(self):
        records = self._record_jsonl(
            num_correct_drafts=torch.tensor([0, 2, 3]),
            bonus_tokens=torch.tensor([100, 101, 102], dtype=torch.int32),
        )

        self.assertEqual(len(records), 9)
        starts = [9, 19, 29]
        expected_num_correct = [0, 2, 3]
        expected_draft_tokens = [[0, 1, 2], [3, 4, 5], [1, 2, 3]]
        for request_idx, (start, num_correct) in enumerate(
            zip(starts, expected_num_correct)
        ):
            request_records = records[request_idx * 3 : (request_idx + 1) * 3]
            self.assertEqual([record["i"] for record in request_records], [0, 1, 2])
            self.assertEqual(
                [record["tok"] for record in request_records],
                expected_draft_tokens[request_idx],
            )
            self.assertEqual(
                [record["pos"] for record in request_records],
                [start, start + 1, start + 2],
            )
            self.assertTrue(
                all(record["nc"] == num_correct for record in request_records)
            )
            self.assertEqual(
                [record["acc"] for record in request_records],
                [int(i < num_correct) for i in range(3)],
            )

            expected_owner = [0, 2, 2][request_idx]
            self.assertEqual(
                sum("bonus_token" in record for record in request_records), 1
            )
            self.assertEqual(
                sum("bonus_pos" in record for record in request_records), 1
            )
            for i, record in enumerate(request_records):
                if i == expected_owner:
                    self.assertEqual(record["bonus_token"], 100 + request_idx)
                    self.assertEqual(record["bonus_pos"], start + num_correct)
                else:
                    self.assertNotIn("bonus_token", record)
                    self.assertNotIn("bonus_pos", record)

    def test_omitted_bonus_preserves_draft_only_schema(self):
        records = self._record_jsonl(
            num_correct_drafts=torch.tensor([1, 1, 1]), bonus_tokens=None
        )

        self.assertEqual(len(records), 9)
        for record in records:
            self.assertNotIn("bonus_token", record)
            self.assertNotIn("bonus_pos", record)
            self.assertIn("tok", record)
            self.assertIn("lp", record)
            self.assertIn("rk", record)
            self.assertIn("ta", record)
            self.assertIn("lpa", record)
            self.assertIn("ent", record)
            self.assertIn("lq", record)
            self.assertIn("acc", record)
            self.assertIn("nc", record)
            self.assertNotIn("dent", record)

    def test_draft_entropy_is_belief_entropy_and_skips_greedy_rows(self):
        torch.manual_seed(0)
        draft_logits = torch.randn(3, 3, 8)
        records = self._record_jsonl(
            num_correct_drafts=torch.tensor([1, 1, 1]),
            bonus_tokens=None,
            draft_logits=draft_logits,
            # Row 0 drafts greedily; the other two sample at temperatures that
            # must not leak into the reported belief entropy.
            draft_temperatures=torch.tensor([0.0, 0.5, 2.0]),
        )

        self.assertEqual(len(records), 9)
        for request_idx in range(3):
            request_records = records[request_idx * 3 : (request_idx + 1) * 3]
            for i, record in enumerate(request_records):
                if request_idx == 0:
                    self.assertNotIn("dent", record)
                    continue
                self.assertAlmostEqual(
                    record["dent"], _entropy(draft_logits[request_idx, i]), places=5
                )

    def test_draft_entropy_recovers_belief_from_scaled_probs(self):
        torch.manual_seed(1)
        draft_logits = torch.randn(3, 3, 8)
        draft_temperature = 0.5
        records = self._record_jsonl(
            num_correct_drafts=torch.tensor([1, 1, 1]),
            bonus_tokens=None,
            draft_probs=torch.softmax(draft_logits / draft_temperature, dim=-1),
            draft_temperature=draft_temperature,
        )

        self.assertEqual(len(records), 9)
        for request_idx in range(3):
            request_records = records[request_idx * 3 : (request_idx + 1) * 3]
            for i, record in enumerate(request_records):
                self.assertAlmostEqual(
                    record["dent"], _entropy(draft_logits[request_idx, i]), places=4
                )
                self.assertGreater(
                    record["dent"],
                    _entropy(draft_logits[request_idx, i] / draft_temperature),
                )


if __name__ == "__main__":
    unittest.main()
