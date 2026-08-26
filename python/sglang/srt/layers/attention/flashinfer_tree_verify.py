# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FlashInfer EAGLE target-verify tree-mask helpers.

Implements the two optimized verify-attention modes selected by
``SGLANG_FLASHINFER_VERIFY_ATTN_MODE``:

- ``prepacked``: a Triton kernel writes the tree mask directly in FlashInfer's
  bit-packed custom-mask layout (per-request byte-aligned segments, bit index
  ``qo_idx * kv_len + kv_idx``, LSB-first within each byte). This replaces the
  bool FULL_MASK memset + ``segment_packbits`` (which contains a blocking
  ``.item()`` sync) that FlashInfer's ``plan()`` runs on every verify step.
- ``split``: verify attention is split into prefix attention (no mask) and
  draft-block attention (draft x draft tree mask only, packed by the same
  kernel with a zero prefix), merged with ``merge_state``.

The packed layout must match ``flashinfer.quantization.segment_packbits`` with
``bitorder="little"`` exactly; ``mask_indptr`` entries are byte offsets (see
``DefaultAttention::LogitsMask`` in flashinfer's ``variants.cuh``).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs


class FlashInferVerifyAttnMode(str, Enum):
    LEGACY = "legacy"
    PREPACKED = "prepacked"
    SPLIT = "split"
    # topk=1 only: the draft "tree" degenerates to a chain appended right
    # after the prefix, so target verify is plain causal attention over
    # [prefix + drafts]; no tree mask is built or read at all.
    CHAIN = "chain"


def read_flashinfer_verify_attn_mode() -> FlashInferVerifyAttnMode:
    value = envs.SGLANG_FLASHINFER_VERIFY_ATTN_MODE.get()
    for mode in FlashInferVerifyAttnMode:
        if value == mode.value:
            return mode
    raise ValueError(
        f"invalid SGLANG_FLASHINFER_VERIFY_ATTN_MODE={value!r}; expected one of "
        f"{', '.join(repr(m.value) for m in FlashInferVerifyAttnMode)}"
    )


def compute_tree_verify_mask_byte_indptr(
    kv_lens_cpu: torch.Tensor, draft_token_num: int
) -> torch.Tensor:
    """Byte offsets of each request's packed-mask segment.

    ``kv_lens_cpu``: per-request mask row width in bits (for the prepacked full
    mask this is ``seq_len + draft_token_num``; for the split draft block it is
    ``draft_token_num``). Returns an int32 CPU tensor of shape ``[bs + 1]``.
    Segment i holds ``draft_token_num * kv_lens[i]`` bits packed into
    ``ceil(bits / 8)`` bytes, matching ``segment_packbits``'s byte-aligned
    output layout.
    """
    kv_lens = kv_lens_cpu.to(torch.int64)
    seg_bytes = (kv_lens * draft_token_num + 7) // 8
    indptr = torch.zeros(kv_lens.numel() + 1, dtype=torch.int64)
    torch.cumsum(seg_bytes, dim=0, out=indptr[1:])
    total = int(indptr[-1])
    if total >= 2**31:
        # segment_packbits (the legacy path) has the same int32 limit, so this
        # is parity, not a new restriction.
        raise ValueError(
            f"packed verify mask needs {total} bytes, exceeding the int32 "
            "mask_indptr range supported by FlashInfer"
        )
    return indptr.to(torch.int32)


@triton.jit
def _pack_tree_verify_mask_kernel(
    tree_mask_ptr,  # bool [num_tree_reqs, D, D], contiguous
    seq_lens_ptr,  # int [bs] prefix lens; unused when HAS_PREFIX=False
    mask_byte_indptr_ptr,  # int32 [bs + 1], byte offsets into out
    out_ptr,  # uint8 packed mask
    num_tree_reqs,  # requests with a real tree-mask row; b >= this => all-visible
    D: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    blk = tl.program_id(1)

    byte_start = tl.load(mask_byte_indptr_ptr + b).to(tl.int64)
    byte_end = tl.load(mask_byte_indptr_ptr + b + 1).to(tl.int64)
    num_bytes = byte_end - byte_start

    offs = blk * BLOCK + tl.arange(0, BLOCK)
    inbound = offs < num_bytes

    if HAS_PREFIX:
        prefix_len = tl.load(seq_lens_ptr + b).to(tl.int64)
    else:
        prefix_len = tl.full((), 0, dtype=tl.int64)
    kv_len = prefix_len + D

    has_tree = b < num_tree_reqs

    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for t in tl.static_range(0, 8):
        g = offs.to(tl.int64) * 8 + t  # bit index within this segment
        qi = g // kv_len
        ki = g % kv_len
        valid = inbound & (qi < D)
        is_tree_col = ki >= prefix_len
        col = ki - prefix_len
        tree_bit = tl.load(
            tree_mask_ptr + (b.to(tl.int64) * D + qi) * D + col,
            mask=valid & is_tree_col & has_tree,
            other=0,
        ).to(tl.int32)
        # Prefix columns are always visible; padded requests (no tree row)
        # get an all-visible segment so their (discarded) outputs stay finite.
        bit = tl.where(is_tree_col & has_tree, tree_bit, 1)
        bit = tl.where(valid, bit, 0)
        acc |= bit << t  # little bitorder: element 8j+t -> bit t of byte j
    tl.store(out_ptr + byte_start + offs, acc.to(tl.uint8), mask=inbound)


def pack_tree_verify_masks(
    tree_masks: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    mask_byte_indptr: torch.Tensor,
    out: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    max_seg_bytes: int,
) -> None:
    """Pack QLEN_ONLY tree masks into FlashInfer's custom-mask layout.

    - ``tree_masks``: bool, ``num_tree_reqs * D * D`` elements (flat or 3D).
      Requests ``b >= num_tree_reqs`` (cuda-graph padding) get all-visible
      segments.
    - ``seq_lens``: int device tensor ``[>= batch_size]`` of prefix lengths,
      or None for prefix-free packing (the split draft block).
    - ``mask_byte_indptr``: int32 device tensor ``[>= batch_size + 1]``.
    - ``out``: uint8 device buffer, at least ``mask_byte_indptr[batch_size]``
      bytes.
    - ``max_seg_bytes``: host-known upper bound of any single segment's byte
      count (sizes the launch grid without a device sync).
    """
    assert tree_masks.dtype == torch.bool and tree_masks.is_contiguous()
    assert out.dtype == torch.uint8
    num_tree_reqs = tree_masks.numel() // (draft_token_num * draft_token_num)
    if batch_size == 0 or max_seg_bytes == 0:
        return
    BLOCK = 256
    grid = (batch_size, triton.cdiv(max_seg_bytes, BLOCK))
    _pack_tree_verify_mask_kernel[grid](
        tree_masks,
        seq_lens if seq_lens is not None else tree_masks,  # dummy ptr when unused
        mask_byte_indptr,
        out,
        num_tree_reqs,
        D=draft_token_num,
        HAS_PREFIX=seq_lens is not None,
        BLOCK=BLOCK,
    )
