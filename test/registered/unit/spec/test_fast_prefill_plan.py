"""Equivalence tests for the sync-free `fast_prefill_plan`.

`fast_prefill_plan` replaces FlashInfer's `BatchPrefillWithPagedKVCacheWrapper.plan`
in fixed-width speculative prefill CUDA graphs: upstream plan() does blocking
`.to("cpu")` copies to build host scheduling metadata, while fast_prefill_plan
takes that metadata as host-known args and reaches `_cached_module.plan` with no
readback.

Correctness is proven end-to-end: the same fixed-width attention, planned two
independent ways (upstream plan() vs fast_prefill_plan), must yield the SAME
`run()` output on identical q/kv. A mutation check reverses the kv_indices handed
to fast_prefill_plan and asserts the output DIVERGES, so we know the output is
sensitive to the metadata under test and the equivalence is not vacuous.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    fast_prefill_plan,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

try:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class _TargetVerifyMode:
    def is_decode_or_idle(self):
        return False

    def is_target_verify(self):
        return True

    def is_dllm_extend(self):
        return False

    def is_draft_extend_v2(self):
        return False


@unittest.skipUnless(_HAS_FLASHINFER, "requires flashinfer")
class TestFastPrefillPlanInstallation(CustomTestCase):
    def _run_capture(
        self, *, seq_lens_cpu, spec_info, backend_name="fa2", dispatch_reason=None
    ):
        bs = 2
        original_begin_forward = MagicMock()
        wrapper = SimpleNamespace(begin_forward=original_begin_forward)

        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend.prefill_backend = backend_name
        backend.dispatch_reason = dispatch_reason
        backend.prefill_cuda_graph_metadata = {bs: [wrapper]}
        backend.draft_extend_cuda_graph_metadata = {}
        backend.indices_updater_prefill = MagicMock()
        backend._prepare_cuda_graph_metadata = MagicMock()
        backend.use_sliding_window_kv_pool = False

        def assert_initial_plan_precedes_install(*args, **kwargs):
            self.assertIs(wrapper.begin_forward, original_begin_forward)

        backend.indices_updater_prefill.update.side_effect = (
            assert_initial_plan_precedes_install
        )
        forward_batch = SimpleNamespace(
            batch_size=bs,
            req_pool_indices=torch.arange(bs),
            seq_lens=torch.tensor([7, 11], dtype=torch.int32),
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=18,
            encoder_lens=None,
            forward_mode=_TargetVerifyMode(),
            spec_info=spec_info,
            positions=torch.arange(8),
            out_cache_loc=None,
        )

        backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
        return wrapper, original_begin_forward

    def test_fixed_width_target_verify_installs_after_initial_plan(self):
        wrapper, original = self._run_capture(
            seq_lens_cpu=torch.tensor([7, 11], dtype=torch.int32),
            spec_info=SimpleNamespace(num_tokens_per_req=4, ragged_verify_layout=None),
        )

        self.assertIsNot(wrapper.begin_forward, original)
        self.assertIs(wrapper.begin_forward.func, fast_prefill_plan)
        self.assertIs(wrapper.begin_forward.args[0], wrapper)

    def test_custom_mask_target_verify_keeps_normal_plan(self):
        wrapper, original = self._run_capture(
            seq_lens_cpu=torch.tensor([7, 11], dtype=torch.int32),
            spec_info=SimpleNamespace(
                num_tokens_per_req=4,
                ragged_verify_layout=None,
                custom_mask=torch.ones(8, dtype=torch.bool),
            ),
        )

        self.assertIs(wrapper.begin_forward, original)

    def test_ineligible_target_verify_keeps_normal_plan(self):
        cases = (
            (torch.tensor([7, 11]), None, "fa2", None),
            (
                None,
                SimpleNamespace(num_tokens_per_req=4, ragged_verify_layout=None),
                "fa2",
                None,
            ),
            (
                torch.tensor([7, 11]),
                SimpleNamespace(num_tokens_per_req=0, ragged_verify_layout=None),
                "fa2",
                None,
            ),
            (
                torch.tensor([7, 11]),
                SimpleNamespace(num_tokens_per_req=4, ragged_verify_layout=None),
                "fa3",
                None,
            ),
            (
                torch.tensor([7, 11]),
                SimpleNamespace(num_tokens_per_req=4, ragged_verify_layout=None),
                "fa2",
                "sliding-window",
            ),
        )
        for seq_lens_cpu, spec_info, backend_name, dispatch_reason in cases:
            with self.subTest(
                seq_lens_cpu=seq_lens_cpu,
                num_tokens_per_req=getattr(spec_info, "num_tokens_per_req", None),
                backend_name=backend_name,
                dispatch_reason=dispatch_reason,
            ):
                wrapper, original = self._run_capture(
                    seq_lens_cpu=seq_lens_cpu,
                    spec_info=spec_info,
                    backend_name=backend_name,
                    dispatch_reason=dispatch_reason,
                )
                self.assertIs(wrapper.begin_forward, original)


# Fixed-width layout: constant qo (num_tokens_per_req per req), page_size 1.
# Non-uniform seq_lens make cumsum non-trivial, so a wrong per-row kv split is
# caught instead of hidden by equal lengths.
NUM_TOKENS_PER_REQ = 8
SEQ_LENS = [37, 12, 89]
NUM_QO_HEADS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.float16


@unittest.skipUnless(_HAS_FLASHINFER, "requires flashinfer")
class TestFastPrefillPlan(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = "cuda"
        bs = len(SEQ_LENS)
        self.bs = bs
        seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int32, device=self.device)

        # Device inputs in the exact layout draft-extend feeds plan().
        self.qo_indptr = torch.arange(
            0,
            (bs + 1) * NUM_TOKENS_PER_REQ,
            step=NUM_TOKENS_PER_REQ,
            dtype=torch.int32,
            device=self.device,
        )
        self.kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        self.kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        self.total_kv = int(self.kv_indptr[-1].item())
        self.total_q = int(self.qo_indptr[-1].item())
        self.kv_indices = torch.arange(
            self.total_kv, dtype=torch.int32, device=self.device
        )
        self.last_page_len = torch.ones(bs, dtype=torch.int32, device=self.device)

        # Host metadata the fast path is handed (page_size==1 -> token-level).
        seq_lens_cpu = seq_lens.cpu()
        self.qo_indptr_host = self.qo_indptr.cpu()
        self.kv_indptr_host = self.kv_indptr.cpu()
        self.kv_lens_host = seq_lens_cpu
        self.max_q_len = NUM_TOKENS_PER_REQ
        self.max_kv_len = int(seq_lens_cpu.max())

        # EAGLE-style tree mask: every verify token sees the cached prefix and
        # only its ancestors (plus itself) within the fixed-width draft block.
        parents = (-1, 0, 0, 1, 1, 2, 2, 3)
        draft_mask = torch.eye(
            NUM_TOKENS_PER_REQ, dtype=torch.bool, device=self.device
        )
        for token, parent in enumerate(parents):
            while parent >= 0:
                draft_mask[token, parent] = True
                parent = parents[parent]
        custom_masks = []
        for seq_len in SEQ_LENS:
            prefix_len = seq_len - NUM_TOKENS_PER_REQ
            mask = torch.ones(
                NUM_TOKENS_PER_REQ,
                seq_len,
                dtype=torch.bool,
                device=self.device,
            )
            mask[:, prefix_len:] = draft_mask
            custom_masks.append(mask.reshape(-1))
        self.custom_mask = torch.cat(custom_masks)

        self.workspace = torch.empty(
            384 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )

        # Shared random q/kv so both code paths attend over identical data.
        self.q = torch.randn(
            self.total_q, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=self.device
        )
        # page_size == 1 -> [num_pages, 1, num_kv_heads, head_dim] (NHD).
        self.k_cache = torch.randn(
            self.total_kv, 1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=self.device
        )
        self.v_cache = torch.randn(
            self.total_kv, 1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=self.device
        )

    def _new_wrapper(self, *, use_custom_mask=False):
        bs = self.bs
        extra = (
            {
                "custom_mask_buf": torch.zeros(
                    self.custom_mask.numel(), dtype=torch.uint8, device=self.device
                ),
                "mask_indptr_buf": torch.zeros(
                    bs + 1, dtype=torch.int32, device=self.device
                ),
            }
            if use_custom_mask
            else {}
        )
        return BatchPrefillWithPagedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_cuda_graph=True,
            backend="fa2",
            qo_indptr_buf=torch.zeros(bs + 1, dtype=torch.int32, device=self.device),
            paged_kv_indptr_buf=torch.zeros(
                bs + 1, dtype=torch.int32, device=self.device
            ),
            paged_kv_indices_buf=torch.zeros(
                self.total_kv, dtype=torch.int32, device=self.device
            ),
            paged_kv_last_page_len_buf=torch.ones(
                bs, dtype=torch.int32, device=self.device
            ),
            **extra,
        )

    def _real_plan(self, w, *, custom_mask=None):
        w.plan(
            self.qo_indptr,
            self.kv_indptr,
            self.kv_indices,
            self.last_page_len,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            1,  # page_size
            custom_mask=custom_mask,
            causal=custom_mask is None,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )

    def _forward(self, w):
        return w.run(self.q, (self.k_cache, self.v_cache))

    def _out_upstream(self, *, custom_mask=None):
        """Ground truth: a wrapper planned only by upstream plan()."""
        w = self._new_wrapper(use_custom_mask=custom_mask is not None)
        self._real_plan(w, custom_mask=custom_mask)
        return self._forward(w)

    def _out_fast(self, *, kv_indices=None, custom_mask=None):
        """Same attention, planned via the host-known fast path. One real plan()
        first populates `_cached_module` (mirrors capture), then fast_prefill_plan
        re-plans from host metadata."""
        if kv_indices is None:
            kv_indices = self.kv_indices
        w = self._new_wrapper(use_custom_mask=custom_mask is not None)
        self._real_plan(w, custom_mask=custom_mask)
        fast_prefill_plan(
            w,
            self.qo_indptr,
            self.kv_indptr,
            kv_indices,
            self.last_page_len,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            1,
            custom_mask=custom_mask,
            causal=custom_mask is None,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
            qo_indptr_host=self.qo_indptr_host,
            kv_indptr_host=self.kv_indptr_host,
            kv_lens_host=self.kv_lens_host,
            max_q_len=self.max_q_len,
            max_kv_len=self.max_kv_len,
        )
        return self._forward(w)

    def test_fast_plan_matches_upstream(self):
        # Two genuinely independent plan paths over identical q/kv must produce
        # the same attention output.
        out_upstream = self._out_upstream()
        out_fast = self._out_fast()
        torch.testing.assert_close(out_fast, out_upstream, rtol=0, atol=0)

    def test_fast_plan_matches_upstream_with_eagle_tree_custom_mask(self):
        out_upstream = self._out_upstream(custom_mask=self.custom_mask)
        out_fast = self._out_fast(custom_mask=self.custom_mask)
        torch.testing.assert_close(out_fast, out_upstream, rtol=0, atol=0)

    def test_mutation_changes_output(self):
        """Guards against a vacuous test: the kv_indices fast_prefill_plan installs
        select which physical KV slots the kernel gathers, so reversing them must
        change the attention output. If it did not, the equivalence assertion
        would not be exercising the metadata fast_prefill_plan is responsible for."""
        out_upstream = self._out_upstream()
        reversed_kv = torch.flip(self.kv_indices, dims=[0]).contiguous()
        out_wrong = self._out_fast(kv_indices=reversed_kv)
        self.assertFalse(
            torch.allclose(out_wrong, out_upstream, rtol=1e-3, atol=1e-3),
            "output unchanged under reversed kv_indices; test lacks discriminating power",
        )


if __name__ == "__main__":
    unittest.main()
