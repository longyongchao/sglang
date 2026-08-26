"""Correctness tests for the FlashInfer EAGLE tree-verify optimizations.

Covers, at the kernel/wrapper level (no server):
1. The Triton mask-pack kernel vs flashinfer.segment_packbits (bit-exact),
   including zero-prefix (split draft block) and cuda-graph padding rows.
2. "prepacked" verify attention vs the legacy bool-custom-mask path
   (same fa2 kernel + identical packed mask => bit-exact outputs).
3. "split" verify (prefix paged attention + draft-block ragged attention +
   merge_state) vs the legacy path (numerical tolerance).
4. CUDA-graph replay of the prepacked path vs eager legacy reference.
5. "chain" verify (topk=1: mask-free causal plan) vs the legacy path with an
   equivalent causal-chain custom mask, eager and under CUDA-graph replay.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, suite="base-b-test-1-gpu-large")


def _make_random_tree_masks(bs, draft_token_num, device, seed):
    """Random EAGLE-like tree masks: row i attends itself + a random subset of
    ancestors; row 0 (the bonus/root token) attends only itself."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    masks = torch.zeros(bs, draft_token_num, draft_token_num, dtype=torch.bool)
    for b in range(bs):
        for i in range(draft_token_num):
            masks[b, i, i] = True
            if i > 0:
                parent = int(torch.randint(0, i, (1,), generator=gen))
                # union of the parent's row makes a valid ancestor chain
                masks[b, i] |= masks[b, parent]
    return masks.to(device)


def _reference_full_mask_flat(tree_masks, seq_lens, num_padded_reqs=0, pad_seq_len=1):
    """Flat bool FULL mask (prefix all-visible + tree block), the layout the
    legacy path feeds to FlashInfer plan(). Padded requests are all-visible."""
    bs, D, _ = tree_masks.shape
    chunks = []
    for b in range(bs + num_padded_reqs):
        S = int(seq_lens[b]) if b < bs else pad_seq_len
        if b < bs:
            row = torch.cat(
                [
                    torch.ones(D, S, dtype=torch.bool, device=tree_masks.device),
                    tree_masks[b],
                ],
                dim=1,
            )
        else:
            row = torch.ones(D, S + D, dtype=torch.bool, device=tree_masks.device)
        chunks.append(row.flatten())
    return torch.cat(chunks)


class TestPackTreeVerifyMask(CustomTestCase):
    """Bit-exact parity between the Triton pack kernel and segment_packbits."""

    def _run_case(self, bs, draft_token_num, max_seq_len, seed, num_padded_reqs=0):
        from flashinfer.quantization import segment_packbits

        from sglang.srt.layers.attention.flashinfer_tree_verify import (
            compute_tree_verify_mask_byte_indptr,
            pack_tree_verify_masks,
        )

        device = "cuda"
        gen = torch.Generator(device="cpu").manual_seed(seed)
        seq_lens_cpu = torch.randint(
            1, max_seq_len + 1, (bs,), generator=gen, dtype=torch.int64
        )
        if num_padded_reqs:
            seq_lens_cpu = torch.cat(
                [seq_lens_cpu, torch.ones(num_padded_reqs, dtype=torch.int64)]
            )
        total_bs = bs + num_padded_reqs
        tree_masks = _make_random_tree_masks(bs, draft_token_num, device, seed)
        seq_lens = seq_lens_cpu.to(device)

        # Reference: legacy bool FULL mask -> segment_packbits.
        full_mask = _reference_full_mask_flat(
            tree_masks, seq_lens_cpu, num_padded_reqs=num_padded_reqs
        )
        kv_lens = seq_lens_cpu + draft_token_num
        bit_indptr = torch.zeros(total_bs + 1, dtype=torch.int64)
        torch.cumsum(kv_lens * draft_token_num, dim=0, out=bit_indptr[1:])
        ref_packed, ref_byte_indptr = segment_packbits(
            full_mask, bit_indptr.to(device), bitorder="little"
        )

        byte_indptr_cpu = compute_tree_verify_mask_byte_indptr(
            kv_lens, draft_token_num
        )
        self.assertTrue(
            torch.equal(byte_indptr_cpu.to(torch.int64), ref_byte_indptr.cpu().to(torch.int64)),
            "byte indptr mismatch vs segment_packbits",
        )

        byte_indptr = byte_indptr_cpu.to(device)
        out = torch.full(
            (int(byte_indptr_cpu[-1]),), 0xAB, dtype=torch.uint8, device=device
        )
        max_seg_bytes = int(
            (byte_indptr_cpu[1:] - byte_indptr_cpu[:-1]).max()
        )
        pack_tree_verify_masks(
            tree_masks,
            seq_lens,
            byte_indptr,
            out,
            batch_size=total_bs,
            draft_token_num=draft_token_num,
            max_seg_bytes=max_seg_bytes,
        )
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(out, ref_packed),
            f"packed mask mismatch: {int((out != ref_packed).sum())} / {out.numel()} bytes differ",
        )

    def test_pack_matches_segment_packbits(self):
        for bs, D, max_s, seed in [
            (1, 1, 1, 0),
            (1, 2, 3, 1),
            (3, 4, 17, 2),
            (5, 8, 129, 3),
            (7, 16, 333, 4),
            (2, 63, 1000, 5),  # odd draft width, non-byte-aligned rows
            (4, 64, 2048, 6),
            (16, 8, 512, 7),
        ]:
            with self.subTest(bs=bs, D=D, max_seq_len=max_s):
                self._run_case(bs, D, max_s, seed)

    def test_pack_with_cuda_graph_padding_rows(self):
        # Padded requests (b >= num_tree_reqs) must be all-visible segments.
        for bs, pad, D, max_s, seed in [
            (2, 2, 8, 100, 10),
            (3, 5, 16, 50, 11),
            (1, 7, 4, 9, 12),
        ]:
            with self.subTest(bs=bs, pad=pad, D=D):
                self._run_case(bs, D, max_s, seed, num_padded_reqs=pad)

    def test_pack_zero_prefix_draft_block(self):
        # The split mode packs the draft x draft block with no prefix.
        from flashinfer.quantization import segment_packbits

        from sglang.srt.layers.attention.flashinfer_tree_verify import (
            compute_tree_verify_mask_byte_indptr,
            pack_tree_verify_masks,
        )

        device = "cuda"
        for bs, D, seed in [(1, 1, 20), (2, 7, 21), (4, 8, 22), (3, 64, 23), (5, 33, 24)]:
            with self.subTest(bs=bs, D=D):
                tree_masks = _make_random_tree_masks(bs, D, device, seed)
                bit_indptr = torch.arange(
                    0, (bs + 1) * D * D, D * D, dtype=torch.int64, device=device
                )
                ref_packed, ref_byte_indptr = segment_packbits(
                    tree_masks.flatten(), bit_indptr, bitorder="little"
                )
                byte_indptr_cpu = compute_tree_verify_mask_byte_indptr(
                    torch.full((bs,), D, dtype=torch.int64), D
                )
                self.assertTrue(
                    torch.equal(
                        byte_indptr_cpu.to(torch.int64),
                        ref_byte_indptr.cpu().to(torch.int64),
                    )
                )
                byte_indptr = byte_indptr_cpu.to(device)
                out = torch.full(
                    (int(byte_indptr_cpu[-1]),), 0xAB, dtype=torch.uint8, device=device
                )
                pack_tree_verify_masks(
                    tree_masks,
                    None,  # zero prefix
                    byte_indptr,
                    out,
                    batch_size=bs,
                    draft_token_num=D,
                    max_seg_bytes=(D * D + 7) // 8,
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(out, ref_packed))


def _torch_verify_reference(
    q, prefix_kv, draft_kv, tree_masks, seq_lens_cpu, sm_scale
):
    """fp32 tree-verify attention reference (GQA, per request)."""
    bs, D, _ = tree_masks.shape
    num_qo_heads = q.shape[1]
    num_kv_heads = prefix_kv[0].shape[2]  # [bs, max_s, H_kv, d]
    group = num_qo_heads // num_kv_heads
    outs = []
    for b in range(bs):
        S = int(seq_lens_cpu[b])
        qb = q[b * D : (b + 1) * D].float()  # [D, Hq, d]
        kb = torch.cat([prefix_kv[0][b][:S], draft_kv[0][b * D : (b + 1) * D]])
        vb = torch.cat([prefix_kv[1][b][:S], draft_kv[1][b * D : (b + 1) * D]])
        kb = kb.float().repeat_interleave(group, dim=1)  # [S+D, Hq, d]
        vb = vb.float().repeat_interleave(group, dim=1)
        mask = torch.cat(
            [
                torch.ones(D, S, dtype=torch.bool, device=q.device),
                tree_masks[b],
            ],
            dim=1,
        )  # [D, S+D]
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * sm_scale
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, vb))
    return torch.cat(outs, dim=0)  # [bs*D, Hq, d] fp32


class _VerifyAttnCase:
    """Shared fixture: paged KV pool + tree masks + flat kv indices."""

    def __init__(
        self, bs, D, max_s, seed, device="cuda", dtype=torch.bfloat16, chain=False
    ):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.bs, self.D, self.device, self.dtype = bs, D, device, dtype
        self.num_qo_heads, self.num_kv_heads, self.head_dim = 8, 2, 128
        self.sm_scale = self.head_dim**-0.5
        self.seq_lens_cpu = torch.randint(
            1, max_s + 1, (bs,), generator=gen, dtype=torch.int64
        )
        if chain:
            # topk=1: the tree degenerates to a chain, whose mask is exactly
            # the lower-triangular causal block.
            self.tree_masks = (
                torch.tril(torch.ones(D, D, dtype=torch.bool, device=device))
                .unsqueeze(0)
                .expand(bs, D, D)
                .contiguous()
            )
        else:
            self.tree_masks = _make_random_tree_masks(bs, D, device, seed)
        max_s_pad = int(self.seq_lens_cpu.max())
        self.q = torch.randn(
            bs * D, self.num_qo_heads, self.head_dim, generator=gen
        ).to(device=device, dtype=dtype)
        # Per-request prefix K/V, padded to max_s (reference slices per S).
        self.prefix_k = torch.randn(
            bs, max_s_pad, self.num_kv_heads, self.head_dim, generator=gen
        ).to(device=device, dtype=dtype)
        self.prefix_v = torch.randn(
            bs, max_s_pad, self.num_kv_heads, self.head_dim, generator=gen
        ).to(device=device, dtype=dtype)
        self.draft_k = torch.randn(
            bs * D, self.num_kv_heads, self.head_dim, generator=gen
        ).to(device=device, dtype=dtype)
        self.draft_v = torch.randn(
            bs * D, self.num_kv_heads, self.head_dim, generator=gen
        ).to(device=device, dtype=dtype)

        # Paged pool (page_size=1), shuffled locations. Sized by the upper
        # bound (not the drawn lens) so cases with equal (bs, max_s, D) share
        # pool shapes -- required by the cuda-graph replay test.
        total = int(self.seq_lens_cpu.sum()) + bs * D
        pool_size = bs * (max_s + D) + 16
        perm = torch.randperm(pool_size, generator=gen)[:total].to(device)
        self.pool_k = torch.zeros(
            pool_size, 1, self.num_kv_heads, self.head_dim, device=device, dtype=dtype
        )
        self.pool_v = torch.zeros_like(self.pool_k)
        kv_indices, prefix_indices = [], []
        cursor = 0
        for b in range(bs):
            S = int(self.seq_lens_cpu[b])
            locs_prefix = perm[cursor : cursor + S]
            locs_draft = perm[cursor + S : cursor + S + D]
            cursor += S + D
            self.pool_k[locs_prefix, 0] = self.prefix_k[b][:S]
            self.pool_v[locs_prefix, 0] = self.prefix_v[b][:S]
            self.pool_k[locs_draft, 0] = self.draft_k[b * D : (b + 1) * D]
            self.pool_v[locs_draft, 0] = self.draft_v[b * D : (b + 1) * D]
            kv_indices.append(torch.cat([locs_prefix, locs_draft]))
            prefix_indices.append(locs_prefix)
        self.kv_indices = torch.cat(kv_indices).to(torch.int32)
        self.prefix_kv_indices = torch.cat(prefix_indices).to(torch.int32)
        kv_lens = self.seq_lens_cpu + D
        self.kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        self.kv_indptr[1:] = torch.cumsum(kv_lens.to(device), dim=0)
        self.prefix_kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        self.prefix_kv_indptr[1:] = torch.cumsum(self.seq_lens_cpu.to(device), dim=0)
        self.qo_indptr = torch.arange(
            0, (bs + 1) * D, D, dtype=torch.int32, device=device
        )
        self.last_page_len = torch.ones(bs, dtype=torch.int32, device=device)
        self.full_mask = _reference_full_mask_flat(
            self.tree_masks, self.seq_lens_cpu
        )
        self.reference = _torch_verify_reference(
            self.q,
            (self.prefix_k, self.prefix_v),
            (self.draft_k, self.draft_v),
            self.tree_masks,
            self.seq_lens_cpu,
            self.sm_scale,
        )

    def plan_kwargs(self):
        return dict(
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=1,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

    def run_legacy(self, workspace):
        import flashinfer

        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        wrapper.plan(
            self.qo_indptr,
            self.kv_indptr,
            self.kv_indices,
            self.last_page_len,
            custom_mask=self.full_mask,
            **self.plan_kwargs(),
        )
        return wrapper.run(self.q, (self.pool_k, self.pool_v))

    def assert_close_to_reference(self, testcase, out, tag, atol=2e-2, rtol=2e-2):
        diff = (out.float() - self.reference).abs()
        ref_scale = self.reference.abs().clamp_min(1.0)
        testcase.assertTrue(
            bool((diff <= atol + rtol * ref_scale).all()),
            f"{tag}: max abs diff {diff.max().item():.4f} vs fp32 reference",
        )


class TestVerifyAttentionModes(CustomTestCase):
    """Attention-level parity of the prepacked and split verify modes."""

    @classmethod
    def setUpClass(cls):
        cls.workspace = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )

    def test_prepacked_matches_legacy_custom_mask(self):
        import flashinfer

        from sglang.srt.layers.attention.flashinfer_tree_verify import (
            compute_tree_verify_mask_byte_indptr,
            pack_tree_verify_masks,
        )

        for bs, D, max_s, seed in [(2, 4, 33, 40), (3, 8, 100, 41), (1, 16, 7, 42)]:
            with self.subTest(bs=bs, D=D, max_s=max_s):
                case = _VerifyAttnCase(bs, D, max_s, seed)
                o_legacy = case.run_legacy(self.workspace)
                case.assert_close_to_reference(self, o_legacy, "legacy")

                # Prepacked eager flow: plan without a mask, then attach the
                # Triton-packed mask buffers (mirrors the backend eager path).
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace, "NHD", backend="fa2"
                )
                wrapper.plan(
                    case.qo_indptr,
                    case.kv_indptr,
                    case.kv_indices,
                    case.last_page_len,
                    **case.plan_kwargs(),
                )
                kv_lens = case.seq_lens_cpu + D
                byte_indptr_cpu = compute_tree_verify_mask_byte_indptr(kv_lens, D)
                packed = torch.empty(
                    int(byte_indptr_cpu[-1]), dtype=torch.uint8, device="cuda"
                )
                byte_indptr = byte_indptr_cpu.to("cuda")
                pack_tree_verify_masks(
                    case.tree_masks,
                    case.seq_lens_cpu.to("cuda"),
                    byte_indptr,
                    packed,
                    batch_size=bs,
                    draft_token_num=D,
                    max_seg_bytes=int((kv_lens.max() * D + 7) // 8),
                )
                wrapper._custom_mask_buf = packed
                wrapper._mask_indptr_buf = byte_indptr
                o_prepacked = wrapper.run(case.q, (case.pool_k, case.pool_v))
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(o_prepacked, o_legacy),
                    "prepacked path is not bit-identical to the legacy "
                    "custom-mask path",
                )

    def test_split_matches_reference(self):
        import flashinfer

        from sglang.srt.layers.attention.flashinfer_backend import _safe_merge_state

        for bs, D, max_s, seed in [(2, 4, 33, 50), (3, 8, 100, 51), (1, 16, 7, 52)]:
            with self.subTest(bs=bs, D=D, max_s=max_s):
                case = _VerifyAttnCase(bs, D, max_s, seed)
                o_legacy = case.run_legacy(self.workspace)

                paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace, "NHD", backend="fa2"
                )
                paged.plan(
                    case.qo_indptr,
                    case.prefix_kv_indptr,
                    case.prefix_kv_indices,
                    case.last_page_len,
                    causal=False,
                    **case.plan_kwargs(),
                )
                o_prefix, lse_prefix = paged.run(
                    case.q, (case.pool_k, case.pool_v), return_lse=True
                )

                ragged = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    self.workspace, "NHD", backend="fa2"
                )
                ragged.plan(
                    case.qo_indptr,
                    case.qo_indptr,
                    case.num_qo_heads,
                    case.num_kv_heads,
                    case.head_dim,
                    custom_mask=case.tree_masks.flatten(),
                    q_data_type=case.dtype,
                )
                o_drafts, lse_drafts = ragged.run(
                    case.q, case.draft_k, case.draft_v, return_lse=True
                )
                o_split, _ = _safe_merge_state(
                    o_prefix, lse_prefix, o_drafts, lse_drafts
                )
                torch.cuda.synchronize()
                case.assert_close_to_reference(self, o_split, "split")
                diff = (o_split.float() - o_legacy.float()).abs().max().item()
                self.assertLess(
                    diff, 3e-2, f"split vs legacy max abs diff {diff:.4f}"
                )

    def test_chain_matches_legacy_causal_mask(self):
        """topk=1 verify: a mask-free causal plan must reproduce the legacy
        path fed with the equivalent causal-chain custom mask."""
        import flashinfer

        for bs, D, max_s, seed in [(2, 4, 33, 70), (3, 8, 100, 71), (1, 16, 7, 72)]:
            with self.subTest(bs=bs, D=D, max_s=max_s):
                case = _VerifyAttnCase(bs, D, max_s, seed, chain=True)
                o_legacy = case.run_legacy(self.workspace)
                case.assert_close_to_reference(self, o_legacy, "legacy-chain")

                # Chain flow (mirrors _plan_eagle_verify_optimized "chain"):
                # same [prefix + drafts] KV layout, no mask, causal=True.
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace, "NHD", backend="fa2"
                )
                wrapper.plan(
                    case.qo_indptr,
                    case.kv_indptr,
                    case.kv_indices,
                    case.last_page_len,
                    causal=True,
                    **case.plan_kwargs(),
                )
                o_chain = wrapper.run(case.q, (case.pool_k, case.pool_v))
                torch.cuda.synchronize()
                case.assert_close_to_reference(self, o_chain, "chain")
                self.assertTrue(
                    torch.equal(o_chain, o_legacy),
                    "chain (causal, mask-free) is not bit-identical to the "
                    "legacy causal-chain custom-mask path",
                )


class TestPrepackedVerifyCudaGraph(CustomTestCase):
    """CUDA-graph replay of the prepacked flow: capture with a dummy all-ones
    mask via the real plan(), then per-replay refresh with fast_prefill_plan +
    the Triton pack kernel, and compare replays against the legacy eager path."""

    def test_graph_replay_matches_eager(self):
        import flashinfer

        from sglang.srt.layers.attention.flashinfer_backend import fast_prefill_plan
        from sglang.srt.layers.attention.flashinfer_tree_verify import (
            compute_tree_verify_mask_byte_indptr,
            pack_tree_verify_masks,
        )

        device = "cuda"
        bs, D, max_s = 3, 8, 64
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        base = _VerifyAttnCase(bs, D, max_s, seed=60)

        cap = bs * (max_s + D) * 2
        qo_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        kv_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        kv_indices_buf = torch.zeros(cap, dtype=torch.int32, device=device)
        last_page_len_buf = torch.ones(bs, dtype=torch.int32, device=device)
        mask_buf = torch.zeros((cap * D + 7) // 8 + bs, dtype=torch.uint8, device=device)
        mask_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)

        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            backend="fa2",
            use_cuda_graph=True,
            qo_indptr_buf=qo_indptr_buf,
            paged_kv_indptr_buf=kv_indptr_buf,
            paged_kv_indices_buf=kv_indices_buf,
            paged_kv_last_page_len_buf=last_page_len_buf,
            custom_mask_buf=mask_buf,
            mask_indptr_buf=mask_indptr_buf,
        )
        # Capture-time plan: real plan() with a dummy all-ones bool mask, as
        # the backend does with the cuda-graph dummy EagleVerifyInput.
        kv_indices_dev = torch.zeros(cap, dtype=torch.int32, device=device)
        n = base.kv_indices.numel()
        kv_indices_dev[:n] = base.kv_indices
        dummy_mask = torch.ones(
            int((base.seq_lens_cpu + D).sum()) * D, dtype=torch.bool, device=device
        )
        wrapper.plan(
            base.qo_indptr,
            base.kv_indptr,
            kv_indices_dev,
            base.last_page_len,
            custom_mask=dummy_mask,
            **base.plan_kwargs(),
        )

        q_static = base.q.clone()
        pool_k = base.pool_k
        pool_v = base.pool_v
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            o_static = wrapper.run(q_static, (pool_k, pool_v))

        for seed in (61, 62, 63):
            case = _VerifyAttnCase(bs, D, max_s, seed=seed)
            o_ref = case.run_legacy(workspace)
            case.assert_close_to_reference(self, o_ref, f"legacy seed={seed}")

            # Replay-side refresh, mirroring _plan_eagle_verify_optimized.
            q_static.copy_(case.q)
            pool_k.copy_(case.pool_k)
            pool_v.copy_(case.pool_v)
            kv_indices_dev[: case.kv_indices.numel()] = case.kv_indices
            kv_lens_cpu = (case.seq_lens_cpu + D).to(torch.int32)
            byte_indptr_cpu = compute_tree_verify_mask_byte_indptr(kv_lens_cpu, D)
            mask_indptr_buf.copy_(byte_indptr_cpu, non_blocking=True)
            pack_tree_verify_masks(
                case.tree_masks,
                case.seq_lens_cpu.to(device),
                mask_indptr_buf,
                mask_buf,
                batch_size=bs,
                draft_token_num=D,
                max_seg_bytes=int((kv_lens_cpu.max() * D + 7) // 8),
            )
            kv_indptr_host = torch.zeros(bs + 1, dtype=torch.int32)
            kv_indptr_host[1:] = torch.cumsum(kv_lens_cpu, dim=0)
            fast_prefill_plan(
                wrapper,
                case.qo_indptr,
                case.kv_indptr,
                kv_indices_dev,
                case.last_page_len,
                base.num_qo_heads,
                base.num_kv_heads,
                base.head_dim,
                1,
                q_data_type=base.dtype,
                kv_data_type=base.dtype,
                non_blocking=True,
                qo_indptr_host=torch.arange(0, (bs + 1) * D, D, dtype=torch.int32),
                kv_indptr_host=kv_indptr_host,
                kv_lens_host=kv_lens_cpu,
                max_q_len=D,
                max_kv_len=int(kv_lens_cpu.max()),
            )
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(o_static, o_ref),
                f"graph replay (seed={seed}) not bit-identical to eager legacy",
            )


class TestVerifyAttnModeResolution(CustomTestCase):
    """Resolution matrix of _resolve_verify_attn_mode, incl. topk=1
    auto-chain: unset env or prepacked/split upgrade to "chain"; an
    explicitly pinned "legacy" is honored; chain needs topk=1."""

    def _resolve(self, env_value, topk):
        from types import SimpleNamespace

        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )

        backend = SimpleNamespace(
            skip_prefill=False,
            dispatch_reason=None,
            prefill_backend="fa2",
            is_multimodal=False,
            enable_mis=False,
            is_dllm_model=False,
            enable_deterministic=False,
        )
        model_runner = SimpleNamespace(
            is_draft_worker=False,
            spec_algorithm=SimpleNamespace(
                is_eagle=lambda: True, is_frozen_kv_mtp=lambda: False
            ),
            server_args=SimpleNamespace(
                speculative_eagle_topk=topk, enable_two_batch_overlap=False
            ),
            kv_cache_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
        )
        env = envs.SGLANG_FLASHINFER_VERIFY_ATTN_MODE
        was_set, prev = env.is_set(), env.get()
        try:
            if env_value is None:
                env.clear()
            else:
                env.set(env_value)
            return FlashInferAttnBackend._resolve_verify_attn_mode(
                backend, model_runner
            ).value
        finally:
            env.set(prev) if was_set else env.clear()

    def test_resolution_matrix(self):
        for env_value, topk, expected in [
            (None, 1, "chain"),         # unset -> auto-chain
            (None, 4, "legacy"),        # unset, tree -> legacy default
            ("legacy", 1, "legacy"),    # explicit legacy opts out
            ("prepacked", 1, "chain"),  # auto-chain wins over prepacked
            ("split", 1, "chain"),      # auto-chain wins over split
            ("prepacked", 4, "prepacked"),
            ("chain", 1, "chain"),
            ("chain", 4, "legacy"),     # chain needs topk=1 -> fallback
        ]:
            with self.subTest(env=env_value, topk=topk):
                self.assertEqual(self._resolve(env_value, topk), expected)


class TestChainVerifyCudaGraph(CustomTestCase):
    """CUDA-graph replay of the chain flow: capture a mask-free causal plan
    (bakes MaskMode.CAUSAL), refresh per replay with fast_prefill_plan
    (causal=True, sync-free), and compare replays against the eager legacy
    path fed with the equivalent causal-chain custom mask."""

    def test_graph_replay_matches_eager(self):
        import flashinfer

        from sglang.srt.layers.attention.flashinfer_backend import fast_prefill_plan

        device = "cuda"
        bs, D, max_s = 3, 8, 64
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        base = _VerifyAttnCase(bs, D, max_s, seed=80, chain=True)

        cap = bs * (max_s + D) * 2
        qo_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        kv_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        kv_indices_buf = torch.zeros(cap, dtype=torch.int32, device=device)
        last_page_len_buf = torch.ones(bs, dtype=torch.int32, device=device)

        # No custom_mask_buf / mask_indptr_buf: run() must pick MaskMode.CAUSAL.
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            backend="fa2",
            use_cuda_graph=True,
            qo_indptr_buf=qo_indptr_buf,
            paged_kv_indptr_buf=kv_indptr_buf,
            paged_kv_indices_buf=kv_indices_buf,
            paged_kv_last_page_len_buf=last_page_len_buf,
        )
        kv_indices_dev = torch.zeros(cap, dtype=torch.int32, device=device)
        kv_indices_dev[: base.kv_indices.numel()] = base.kv_indices
        # Capture-time plan: real plan() with causal=True and no mask, as the
        # backend does for the cuda-graph dummy input under "chain".
        wrapper.plan(
            base.qo_indptr,
            base.kv_indptr,
            kv_indices_dev,
            base.last_page_len,
            causal=True,
            **base.plan_kwargs(),
        )

        q_static = base.q.clone()
        pool_k = base.pool_k
        pool_v = base.pool_v
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            o_static = wrapper.run(q_static, (pool_k, pool_v))

        for seed in (81, 82, 83):
            case = _VerifyAttnCase(bs, D, max_s, seed=seed, chain=True)
            o_ref = case.run_legacy(workspace)
            case.assert_close_to_reference(self, o_ref, f"legacy-chain seed={seed}")

            # Replay-side refresh, mirroring the "chain" branch of
            # _plan_eagle_verify_optimized: sync-free plan, causal, no mask.
            q_static.copy_(case.q)
            pool_k.copy_(case.pool_k)
            pool_v.copy_(case.pool_v)
            kv_indices_dev[: case.kv_indices.numel()] = case.kv_indices
            kv_lens_cpu = (case.seq_lens_cpu + D).to(torch.int32)
            kv_indptr_host = torch.zeros(bs + 1, dtype=torch.int32)
            kv_indptr_host[1:] = torch.cumsum(kv_lens_cpu, dim=0)
            fast_prefill_plan(
                wrapper,
                case.qo_indptr,
                case.kv_indptr,
                kv_indices_dev,
                case.last_page_len,
                base.num_qo_heads,
                base.num_kv_heads,
                base.head_dim,
                1,
                causal=True,
                q_data_type=base.dtype,
                kv_data_type=base.dtype,
                non_blocking=True,
                qo_indptr_host=torch.arange(0, (bs + 1) * D, D, dtype=torch.int32),
                kv_indptr_host=kv_indptr_host,
                kv_lens_host=kv_lens_cpu,
                max_q_len=D,
                max_kv_len=int(kv_lens_cpu.max()),
            )
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(o_static, o_ref),
                f"chain graph replay (seed={seed}) not bit-identical to "
                "eager legacy",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
