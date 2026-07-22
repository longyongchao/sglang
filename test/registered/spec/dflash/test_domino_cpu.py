import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models.dflash import DFlashDraftModel
import sglang.srt.speculative.dflash_worker_v2 as dflash_worker_v2
import sglang.srt.speculative.domino_rollout as domino_rollout
from sglang.srt.speculative.dflash_utils import (
    is_dflash_domino_projector,
    parse_dflash_draft_config,
)
from sglang.srt.speculative.domino_rollout import (
    _domino_fold_candidate_logits,
    _domino_fold_per_bs_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(projector_type=None, **dflash_fields):
    dflash_config = {
        "block_size": 16,
        "target_layer_ids": [0],
        "projector_type": projector_type,
        **dflash_fields,
    }
    return SimpleNamespace(
        hidden_size=8,
        num_hidden_layers=1,
        num_target_layers=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
        dflash_config=dflash_config,
    )


class _FakeLayer(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))


class _MockTPGroup:
    def __init__(self, outputs):
        self.world_size = int(outputs[0].shape[0])
        self.outputs = outputs
        self.collective_count = 0

    def all_gather_into_tensor(self, output, _input):
        self.assert_collective_buffers(output, _input)
        source = self.outputs[self.collective_count]
        output.copy_(source.reshape_as(output))
        self.collective_count += 1

    @staticmethod
    def assert_collective_buffers(output, input_):
        if not output.is_contiguous() or not input_.is_contiguous():
            raise AssertionError("all_gather_into_tensor buffers must be contiguous")


class TestTPGlobalArgmax(CustomTestCase):
    @staticmethod
    def _worker():
        worker = dflash_worker_v2.DFlashWorkerV2.__new__(
            dflash_worker_v2.DFlashWorkerV2
        )
        worker._draft_greedy_gathered_max_buf = None
        worker._draft_greedy_gathered_ids_buf = None
        worker._draft_greedy_gathered_packed_buf = None
        worker._draft_greedy_packed_buf = None
        return worker

    def test_fp32_packed_argmax_random_ties_and_padding(self):
        generator = torch.Generator().manual_seed(17)
        rank_max = torch.randn(3, 11, generator=generator)
        rank_max[1, 3] = rank_max[0, 3]
        rank_max[2, 3] = rank_max[0, 3] - 1
        rank_ids = torch.randint(0, 97, (3, 11), generator=generator)
        rank_ids[2, 7] = 127  # A padded-shard ID remains exactly representable.
        gathered = torch.stack((rank_max, rank_ids.to(torch.float32)), dim=-1)
        group = _MockTPGroup([gathered])
        worker = self._worker()

        with patch.object(dflash_worker_v2, "get_tp_group", return_value=group):
            actual = worker._global_argmax_from_local_max(
                local_max=rank_max[0],
                global_ids=rank_ids[0],
                global_vocab_size=128,
            )

        best_rank = rank_max.argmax(dim=0)
        expected = rank_ids.gather(0, best_rank.unsqueeze(0)).squeeze(0)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(group.collective_count, 1)
        self.assertEqual(actual[3], rank_ids[0, 3])

    def test_fp32_boundary_retains_two_collectives(self):
        rank_max = torch.tensor([[1.0, 5.0, 4.0], [2.0, 5.0, 3.0]])
        rank_ids = torch.tensor([[10, (1 << 24) - 1, 12], [20, 21, 22]])
        group = _MockTPGroup([rank_max, rank_ids])
        worker = self._worker()

        with patch.object(dflash_worker_v2, "get_tp_group", return_value=group):
            actual = worker._global_argmax_from_local_max(
                local_max=rank_max[0],
                global_ids=rank_ids[0],
                global_vocab_size=1 << 24,
            )

        self.assertEqual(actual.tolist(), [20, (1 << 24) - 1, 12])
        self.assertEqual(group.collective_count, 2)
    def test_gather_buffers_remain_contiguous_when_batch_shrinks(self):
        rank_max_large = torch.tensor([[1.0, 5.0, 4.0, 2.0], [2.0, 3.0, 6.0, 1.0]])
        rank_ids_large = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
        rank_max_small = rank_max_large[:, :2]
        rank_ids_small = rank_ids_large[:, :2]

        for global_vocab_size in (128, 1 << 24):
            if global_vocab_size < (1 << 24):
                large_outputs = [
                    torch.stack(
                        (rank_max_large, rank_ids_large.to(torch.float32)), dim=-1
                    )
                ]
                small_outputs = [
                    torch.stack(
                        (rank_max_small, rank_ids_small.to(torch.float32)), dim=-1
                    )
                ]
            else:
                large_outputs = [rank_max_large, rank_ids_large]
                small_outputs = [rank_max_small, rank_ids_small]

            worker = self._worker()
            with patch.object(
                dflash_worker_v2,
                "get_tp_group",
                side_effect=[_MockTPGroup(large_outputs), _MockTPGroup(small_outputs)],
            ):
                worker._global_argmax_from_local_max(
                    local_max=rank_max_large[0],
                    global_ids=rank_ids_large[0],
                    global_vocab_size=global_vocab_size,
                )
                output_buffer = (
                    worker._draft_greedy_gathered_packed_buf
                    if global_vocab_size < (1 << 24)
                    else worker._draft_greedy_tp_gathered_max_buf
                )
                output_ptr = output_buffer.data_ptr()
                actual = worker._global_argmax_from_local_max(
                    local_max=rank_max_small[0],
                    global_ids=rank_ids_small[0],
                    global_vocab_size=global_vocab_size,
                )
                current_output_buffer = (
                    worker._draft_greedy_gathered_packed_buf
                    if global_vocab_size < (1 << 24)
                    else worker._draft_greedy_tp_gathered_max_buf
                )
                self.assertEqual(current_output_buffer.data_ptr(), output_ptr)

            best_rank = rank_max_small.argmax(dim=0)
            expected = rank_ids_small.gather(0, best_rank.unsqueeze(0)).squeeze(0)
            self.assertTrue(torch.equal(actual, expected))


class TestDominoCandidateFold(CustomTestCase):
    def test_candidate_fold_matches_full_logits_and_pool_definition(self):
        torch.manual_seed(7)
        z = torch.randn(2, 4, 3)
        weight = torch.randn(11, 3)
        compact, candidate_ids, _, slot_1 = _domino_fold_candidate_logits(
            z, weight, candidate_pool_size=5, org_vocab_start=13
        )
        full = torch.matmul(z.reshape(-1, 3), weight.T).view(2, 4, 11)
        expected_pool = full[:, 1:, :].max(dim=1).values
        expected_ids = torch.argsort(
            expected_pool, dim=-1, descending=True, stable=True
        )[:, :5]
        self.assertTrue(torch.equal(candidate_ids, expected_ids))
        self.assertTrue(
            torch.equal(
                compact,
                torch.gather(full, 2, candidate_ids[:, None, :].expand(2, 4, 5)),
            )
        )
        self.assertTrue(torch.equal(slot_1, full[:, 0, :].argmax(dim=-1) + 13))

    def test_candidate_fold_uses_lowest_global_ids_for_ties(self):
        z = torch.ones(1, 3, 2)
        weight = torch.zeros(9, 2)
        _, candidate_ids, _, slot_1 = _domino_fold_candidate_logits(
            z, weight, candidate_pool_size=4, org_vocab_start=20
        )
        self.assertEqual(candidate_ids.tolist(), [[0, 1, 2, 3]])
        self.assertEqual(slot_1.tolist(), [20])

    def test_candidate_fold_crosses_vocab_chunk_boundary(self):
        with patch.object(domino_rollout, "_DOMINO_CANDIDATE_VOCAB_CHUNK", 3):
            z = torch.arange(2 * 3 * 2, dtype=torch.float32).view(2, 3, 2)
            weight = torch.arange(7 * 2, dtype=torch.float32).view(7, 2)
            compact, candidate_ids, _, slot_1 = _domino_fold_candidate_logits(
                z, weight, candidate_pool_size=4, org_vocab_start=5
            )
        full = torch.matmul(z.reshape(-1, 2), weight.T).view(2, 3, 7)
        expected_ids = torch.argsort(
            full[:, 1:, :].max(dim=1).values, dim=-1, descending=True, stable=True
        )[:, :4]
        self.assertTrue(torch.equal(candidate_ids, expected_ids))
        self.assertTrue(
            torch.equal(
                compact,
                torch.gather(full, 2, candidate_ids[:, None, :].expand(2, 3, 4)),
            )
        )
        self.assertTrue(torch.equal(slot_1, full[:, 0, :].argmax(dim=-1) + 5))

    def test_capture_budget_accounts_for_chunk_and_compact_logits(self):
        element_size = torch.empty((), dtype=torch.float16).element_size()
        candidate_bytes = _domino_fold_per_bs_bytes(7, 10000, 17, element_size)
        full_bytes = _domino_fold_per_bs_bytes(7, 10000, 0, element_size)
        self.assertEqual(candidate_bytes, 7 * (4096 + 17) * element_size)
        self.assertEqual(full_bytes, 7 * 10000 * element_size)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_graph_capture_replay_avoids_full_candidate_logits(self):
        torch.manual_seed(11)
        device = torch.device("cuda")
        dtype = torch.float16
        batch, draft_step, hidden, vocab_size, pool_size = 8, 8, 4, 1_100_000, 17
        z = torch.randn(batch, draft_step, hidden, device=device, dtype=dtype)
        weight = torch.randn(vocab_size, hidden, device=device, dtype=dtype)
        z_static = z.clone()

        eager = _domino_fold_candidate_logits(
            z, weight, candidate_pool_size=pool_size, org_vocab_start=23
        )
        del eager
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before_capture = torch.cuda.memory_allocated(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _domino_fold_candidate_logits(
                z_static, weight, candidate_pool_size=pool_size, org_vocab_start=23
            )
        z_static.copy_(z)
        graph.replay()
        torch.cuda.synchronize()
        replay = tuple(output.clone() for output in captured)
        eager = _domino_fold_candidate_logits(
            z, weight, candidate_pool_size=pool_size, org_vocab_start=23
        )

        z_static.copy_(z + 0.25)
        graph.replay()
        torch.cuda.synchronize()
        replay_changed = tuple(output.clone() for output in captured)
        eager_changed = _domino_fold_candidate_logits(
            z + 0.25,
            weight,
            candidate_pool_size=pool_size,
            org_vocab_start=23,
        )

        for replay_output, eager_output in zip(replay, eager):
            self.assertTrue(torch.equal(replay_output, eager_output))
        for replay_output, eager_output in zip(replay_changed, eager_changed):
            self.assertTrue(torch.equal(replay_output, eager_output))

        element_size = torch.empty((), dtype=dtype).element_size()
        full_bytes = batch * draft_step * vocab_size * element_size
        compact_bytes = _domino_fold_per_bs_bytes(
            batch, vocab_size, pool_size, element_size
        )
        input_bytes = z.numel() * z.element_size() + weight.numel() * weight.element_size()
        self.assertGreater(full_bytes, 128 * 1024**2)
        self.assertLess(compact_bytes, 128 * 1024**2)
        peak_allocated_delta = torch.cuda.max_memory_allocated(device) - allocated_before_capture
        self.assertLess(peak_allocated_delta, full_bytes + input_bytes)
        print(
            f"old full FP16 bytes={full_bytes}, "
            f"peak allocated delta={peak_allocated_delta}"
        )


class TestDFlashDominoConfig(CustomTestCase):
    def test_projector_is_exactly_domino(self):
        self.assertTrue(is_dflash_domino_projector("domino"))
        for projector_type in (None, "dspark", "prism", "causal_v5"):
            self.assertFalse(is_dflash_domino_projector(projector_type))

    def test_parse_vanilla_and_dspark_without_domino_fields(self):
        for projector_type in (None, "dspark"):
            parsed = parse_dflash_draft_config(
                draft_hf_config=_config(projector_type)
            )
            self.assertEqual(parsed.projector_type, projector_type)
            self.assertIsNone(parsed.gru_hidden_dim)
            self.assertIsNone(parsed.emb_dim)

    def test_parse_domino_nested_fields(self):
        parsed = parse_dflash_draft_config(
            draft_hf_config=_config(
                "domino",
                pure_draft_prefix_len=1,
                gru_hidden_dim=1024,
                emb_dim=256,
            )
        )
        self.assertEqual(parsed.pure_draft_prefix_len, 1)
        self.assertEqual(parsed.gru_hidden_dim, 1024)
        self.assertEqual(parsed.emb_dim, 256)

    def test_domino_requires_dimensions_and_prefix(self):
        with self.assertRaises(ValueError):
            parse_dflash_draft_config(
                draft_hf_config=_config("domino", pure_draft_prefix_len=1)
            )
        with self.assertRaises(NotImplementedError):
            parse_dflash_draft_config(
                draft_hf_config=_config(
                    "domino", gru_hidden_dim=16, emb_dim=8
                )
            )

    def test_model_allocates_domino_modules_only_for_domino(self):
        with patch("sglang.srt.models.dflash.DFlashDecoderLayer", _FakeLayer):
            vanilla = DFlashDraftModel(_config())
            domino = DFlashDraftModel(
                _config(
                    "domino",
                    pure_draft_prefix_len=1,
                    gru_hidden_dim=16,
                    emb_dim=8,
                )
            )

        self.assertFalse(hasattr(vanilla, "prefix_gru"))
        self.assertFalse(hasattr(vanilla, "embed_proj"))
        self.assertTrue(hasattr(domino, "prefix_gru"))
        self.assertTrue(hasattr(domino, "embed_proj"))
        self.assertIn("prefix_gru.weight_ih_l0", domino.state_dict())
        self.assertIn("embed_proj.0.weight", domino.state_dict())
        self.assertIn("embed_proj.2.weight", domino.state_dict())


if __name__ == "__main__":
    unittest.main()
