"""Fused Triton kernels for the DSpark Markov head sampling loop.

Reduces per-step kernel count for the greedy sampling path:
- Vanilla: from ~6 ops to 2 kernels (fused_markov_bias_argmax, two passes)
- RNN: from ~11 ops to 4 kernels (cuBLAS matmul + fused_rnn_cell +
  fused_markov_bias_argmax two passes)

Both kernels are CUDA-graph friendly (no host-device sync, fixed shapes).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fused embedding lookup @ markov_w2.T + base_logits -> argmax
# ---------------------------------------------------------------------------


@triton.jit
def _dspark_fused_markov_bias_argmax_kernel(
    # Input: either embed_table lookup by token id, or pre-computed latent
    embed_table_ptr,  # [V_emb, R] embedding table (markov_w1.weight)
    prev_tokens_ptr,  # [B] int64 previous token ids
    latent_ptr,  # [B, R] pre-computed latent (used when FROM_TABLE=False)
    # Projection weight
    fc2_ptr,  # [V, R] markov_w2.weight
    # Base logits to add
    base_ptr,  # [B, V] base_logits for current step
    # Output scratch
    out_val_ptr,  # [B, num_v_blocks] fp32
    out_idx_ptr,  # [B, num_v_blocks] int32
    # Dimensions
    R,  # markov_rank (hidden dim of embedding)
    V,  # vocab_size
    V_EMB,  # embedding table vocab size (for bounds check)
    # Strides
    stride_et_v, stride_et_r,
    stride_latent_b, stride_latent_r,
    stride_f2_v, stride_f2_r,
    stride_base_b, stride_base_v,
    stride_outv_b, stride_outv_n,
    stride_outi_b, stride_outi_n,
    # Compile-time constants
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
    FROM_TABLE: tl.constexpr,
):
    """Pass 1: per-tile matmul + base_logits + local argmax."""
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    # Load the latent vector [R] for this batch element
    if FROM_TABLE:
        tok_id_raw = tl.load(prev_tokens_ptr + pid_b)
        tok_in_range = (tok_id_raw >= 0) & (tok_id_raw < V_EMB)
        tok_id = tl.where(tok_in_range, tok_id_raw, 0)

    # Accumulate dot product: latent[R] @ fc2_weight[BLOCK_V, R].T
    acc = tl.zeros([BLOCK_V], dtype=tl.float32)

    for r_start in range(0, R, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R

        # Load latent slice [BLOCK_R]
        if FROM_TABLE:
            latent_block = tl.load(
                embed_table_ptr + tok_id * stride_et_v + offs_r * stride_et_r,
                mask=mask_r & tok_in_range,
                other=0.0,
            ).to(tl.float32)
        else:
            latent_block = tl.load(
                latent_ptr + pid_b * stride_latent_b + offs_r * stride_latent_r,
                mask=mask_r,
                other=0.0,
            ).to(tl.float32)

        # Load fc2 slice [BLOCK_V, BLOCK_R]
        fc2_ptrs = (
            fc2_ptr
            + offs_v[:, None] * stride_f2_v
            + offs_r[None, :] * stride_f2_r
        )
        fc2_block = tl.load(
            fc2_ptrs,
            mask=mask_v[:, None] & mask_r[None, :],
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)

        acc += tl.sum(fc2_block * latent_block[None, :], axis=1)

    # 对齐 eager 的原生精度边界：linear 将 fp32 累加结果写回输入精度，
    # 随后的 logits 加法再次写回该精度。
    input_dtype = base_ptr.dtype.element_ty
    projected = acc.to(input_dtype)
    base_block = tl.load(
        base_ptr + pid_b * stride_base_b + offs_v * stride_base_v,
        mask=mask_v,
        other=-float("inf"),
        cache_modifier=".cg",
    )
    logits = (projected + base_block).to(input_dtype).to(tl.float32)
    logits = tl.where(mask_v, logits, -float("inf"))

    # 字典序 argmax：值较大者优先，值相同时词表索引较小者优先。
    local_max_val = tl.max(logits, axis=0)
    local_max_idx = tl.min(
        tl.where(mask_v & (logits == local_max_val), offs_v, V), axis=0
    )

    out_val_offs = pid_b * stride_outv_b + pid_v * stride_outv_n
    out_idx_offs = pid_b * stride_outi_b + pid_v * stride_outi_n
    tl.store(out_val_ptr + out_val_offs, local_max_val)
    tl.store(out_idx_ptr + out_idx_offs, local_max_idx)


# ---------------------------------------------------------------------------
# Kernel 1b: Reduce across tiles to find global argmax
# ---------------------------------------------------------------------------


@triton.jit
def _dspark_reduce_argmax_kernel(
    out_val_ptr, out_idx_ptr, final_token_ptr,
    N,
    stride_val_b, stride_val_n,
    stride_idx_b, stride_idx_n,
    stride_final_b,
    BLOCK_N: tl.constexpr,
):
    """Pass 2: reduce per-tile (val, idx) pairs to a single global argmax."""
    pid_b = tl.program_id(0)
    best_val = -float("inf")
    best_idx = 0

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        vals = tl.load(
            out_val_ptr + pid_b * stride_val_b + offs_n * stride_val_n,
            mask=mask_n, other=-float("inf"),
        )
        indices = tl.load(
            out_idx_ptr + pid_b * stride_idx_b + offs_n * stride_idx_n,
            mask=mask_n,
            other=2147483647,
        )
        local_val = tl.max(vals, axis=0)
        local_idx = tl.min(
            tl.where(mask_n & (vals == local_val), indices, 2147483647), axis=0
        )

        take = (local_val > best_val) | (
            (local_val == best_val) & (local_idx < best_idx)
        )
        best_val = tl.where(take, local_val, best_val)
        best_idx = tl.where(take, local_idx, best_idx)

    tl.store(final_token_ptr + pid_b * stride_final_b, best_idx)


# ---------------------------------------------------------------------------
# Kernel 2: Fused RNN cell (gate computation + state update)
# ---------------------------------------------------------------------------


@triton.jit
def _dspark_fused_rnn_cell_kernel(
    # proj_out: [B, 3*R] — output of joint_proj matmul
    proj_out_ptr,
    # state: [B, R] — current hidden state
    state_ptr,
    # outputs
    new_state_ptr,  # [B, R]
    output_emb_ptr,  # [B, R] — tanh(output_raw), to be fed to markov_w2
    # Dimensions
    R,
    # Strides
    stride_proj_b, stride_proj_r,
    stride_state_b, stride_state_r,
    stride_ns_b, stride_ns_r,
    stride_oe_b, stride_oe_r,
    # Compile-time
    BLOCK_R: tl.constexpr,
):
    """Fuses: chunk(proj_out,3) -> sigmoid(gate) * state + (1-gate) * tanh(cand) -> tanh(output)."""
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offs_r < R

    # Load state
    h_state = tl.load(
        state_ptr + pid_b * stride_state_b + offs_r * stride_state_r,
        mask=mask_r, other=0.0,
    ).to(tl.float32)

    # Load the three chunks from proj_out: gate_raw, candidate_raw, output_raw
    # proj_out layout: [B, 3*R] with chunks at [0..R), [R..2R), [2R..3R)
    gate_raw = tl.load(
        proj_out_ptr + pid_b * stride_proj_b + offs_r * stride_proj_r,
        mask=mask_r, other=0.0,
    ).to(tl.float32)
    candidate_raw = tl.load(
        proj_out_ptr + pid_b * stride_proj_b + (R + offs_r) * stride_proj_r,
        mask=mask_r, other=0.0,
    ).to(tl.float32)
    output_raw = tl.load(
        proj_out_ptr + pid_b * stride_proj_b + (2 * R + offs_r) * stride_proj_r,
        mask=mask_r, other=0.0,
    ).to(tl.float32)

    # GRU-like update — all arithmetic at input dtype to match eager fp16 path
    input_dtype = state_ptr.dtype.element_ty
    gate = tl.sigmoid(gate_raw).to(input_dtype)
    exp_2c = tl.exp(2.0 * candidate_raw)
    candidate = (1.0 - 2.0 / (exp_2c + 1.0)).to(input_dtype)
    h_state_native = h_state.to(input_dtype)
    one_minus_gate = (1.0 - gate.to(tl.float32)).to(input_dtype)
    new_state = gate * h_state_native + one_minus_gate * candidate
    exp_2o = tl.exp(2.0 * output_raw)
    output_emb = (1.0 - 2.0 / (exp_2o + 1.0)).to(input_dtype)

    # Store new_state and output_emb (already in input_dtype)
    tl.store(
        new_state_ptr + pid_b * stride_ns_b + offs_r * stride_ns_r,
        new_state,
        mask=mask_r,
    )
    tl.store(
        output_emb_ptr + pid_b * stride_oe_b + offs_r * stride_oe_r,
        output_emb,
        mask=mask_r,
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def fused_markov_bias_argmax(
    *,
    prev_tokens: torch.Tensor,       # [B] int64
    embed_table: torch.Tensor,       # [V_emb, R] markov_w1.weight
    fc2_weight: torch.Tensor,        # [V, R] markov_w2.weight
    base_logits: torch.Tensor,       # [B, V]
    out_val: torch.Tensor,           # [B, num_v_blocks] fp32 scratch
    out_idx: torch.Tensor,           # [B, num_v_blocks] int32 scratch
    final_token: torch.Tensor,       # [B] int64 destination
    latent: torch.Tensor | None = None,  # [B, R] pre-computed (RNN output_emb)
    block_v: int = 512,
    block_r: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    """Fused: embed_table[prev_token] @ fc2_weight.T + base_logits -> argmax.

    When ``latent`` is provided, uses it directly instead of the table lookup
    (for RNN heads where the latent is tanh(output_raw) from the RNN cell).
    """
    B = base_logits.shape[0]
    V = base_logits.shape[1]
    R = fc2_weight.shape[1]
    V_EMB = embed_table.shape[0]
    num_v_blocks = (V + block_v - 1) // block_v
    from_table = latent is None

    # Placeholder pointers for unused paths
    if from_table:
        latent_arg = out_val  # unused, just need a valid ptr
        latent_stride_b = 0
        latent_stride_r = 0
    else:
        latent_arg = latent
        latent_stride_b = latent.stride(0)
        latent_stride_r = latent.stride(1)

    grid_main = (num_v_blocks, B)
    _dspark_fused_markov_bias_argmax_kernel[grid_main](
        embed_table, prev_tokens, latent_arg,
        fc2_weight, base_logits,
        out_val, out_idx,
        R, V, V_EMB,
        embed_table.stride(0), embed_table.stride(1),
        latent_stride_b, latent_stride_r,
        fc2_weight.stride(0), fc2_weight.stride(1),
        base_logits.stride(0), base_logits.stride(1),
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        BLOCK_V=block_v,
        BLOCK_R=block_r,
        FROM_TABLE=from_table,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _dspark_reduce_argmax_kernel[(B,)](
        out_val, out_idx, final_token,
        num_v_blocks,
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        final_token.stride(0),
        BLOCK_N=512,
        num_warps=1,
    )


def fused_dspark_rnn_cell(
    *,
    proj_out: torch.Tensor,       # [B, 3*R] — output of joint_proj linear
    state: torch.Tensor,          # [B, R] — current RNN state
    new_state: torch.Tensor,      # [B, R] — output: updated state
    output_emb: torch.Tensor,     # [B, R] — output: tanh(output_raw)
    block_r: int = 256,
) -> None:
    """Fused RNN cell: splits proj_out into 3 chunks, applies gate logic.

    new_state = sigmoid(gate_raw) * state + (1 - sigmoid(gate_raw)) * tanh(candidate_raw)
    output_emb = tanh(output_raw)
    """
    B = state.shape[0]
    R = state.shape[1]
    grid = (B, (R + block_r - 1) // block_r)
    _dspark_fused_rnn_cell_kernel[grid](
        proj_out, state, new_state, output_emb,
        R,
        proj_out.stride(0), proj_out.stride(1),
        state.stride(0), state.stride(1),
        new_state.stride(0), new_state.stride(1),
        output_emb.stride(0), output_emb.stride(1),
        BLOCK_R=block_r,
    )


# ---------------------------------------------------------------------------
# Scratch buffer container
# ---------------------------------------------------------------------------


class MarkovFusedScratch:
    """Pre-allocated scratch buffers for fused markov sampling kernels."""

    def __init__(self, *, max_bs: int, vocab_size: int, markov_rank: int,
                 hidden_size: int, device: torch.device, dtype: torch.dtype,
                 is_rnn: bool = False) -> None:
        num_v_blocks = (vocab_size + 512 - 1) // 512
        self.out_val = torch.empty((max_bs, num_v_blocks), dtype=torch.float32, device=device)
        self.out_idx = torch.empty((max_bs, num_v_blocks), dtype=torch.int32, device=device)
        self.tok_buf = torch.empty((max_bs,), dtype=torch.int64, device=device)
        if is_rnn:
            self.rnn_state_buf = torch.zeros((max_bs, markov_rank), dtype=dtype, device=device)
            self.rnn_new_state_buf = torch.empty((max_bs, markov_rank), dtype=dtype, device=device)
            self.output_emb_buf = torch.empty((max_bs, markov_rank), dtype=dtype, device=device)
            self.rnn_proj_buf = torch.empty((max_bs, 3 * markov_rank), dtype=dtype, device=device)
        else:
            self.rnn_state_buf = None
            self.rnn_new_state_buf = None
            self.output_emb_buf = None
            self.rnn_proj_buf = None
