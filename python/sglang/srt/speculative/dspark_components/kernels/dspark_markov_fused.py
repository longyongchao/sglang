"""Fused Triton kernels for the DSpark Markov head sampling loop.

Reduces per-step kernel count for the greedy sampling path:
- Vanilla: from ~6 ops to 3 kernels (index_select + fused_markov_bias_argmax,
  two passes)
- RNN: from ~11 ops to 4 kernels (cuBLAS matmul + fused_rnn_cell +
  fused_markov_bias_argmax two passes)

Both kernels are CUDA-graph friendly (no host-device sync, fixed shapes).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fused latent @ markov_w2.T + base_logits -> argmax
# ---------------------------------------------------------------------------


MARKOV_BLOCK_V = 128


def _score_block_b(batch_size: int) -> int:
    """打分 kernel 的批分块大小：``tl.dot`` 要求各维 >= 16，上限取 64。

    分块越大，fc2 的重读次数 ``ceil(B / BLOCK_B)`` 越少；超过 64 后收益反转，
    因为累加器 ``[BLOCK_B, BLOCK_V]`` 的 SMEM 占用把每 SM 的 CTA 并发压到 1
    （寄存器未溢出）。
    """
    if batch_size <= 16:
        return 16
    if batch_size <= 32:
        return 32
    return 64


@triton.jit
def _dspark_fused_markov_bias_argmax_kernel(
    # Input: pre-computed latent (embedding row for vanilla, RNN output for rnn)
    latent_ptr,  # [B, R]
    # Projection weight
    fc2_ptr,  # [V, R] markov_w2.weight
    # Base logits to add
    base_ptr,  # [B, V] base_logits for current step
    # Output scratch
    out_val_ptr,  # [B, num_v_blocks] fp32
    out_idx_ptr,  # [B, num_v_blocks] int32
    # Dimensions
    B,  # batch size
    R,  # markov_rank (hidden dim of the latent)
    V,  # vocab_size
    # Strides
    stride_latent_b, stride_latent_r,
    stride_f2_v, stride_f2_r,
    stride_base_b, stride_base_v,
    stride_outv_b, stride_outv_n,
    stride_outi_b, stride_outi_n,
    # Compile-time constants
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Pass 1: per-tile matmul + base_logits + local argmax."""
    # 一个 CTA 同时算 BLOCK_B 个 request：fc2 的 [BLOCK_V, R] 切片进 SMEM 后被
    # BLOCK_B 行复用，整表读取次数从 B 降到 ceil(B / BLOCK_B)。原来的
    # 每 request 一个 CTA 只能指望 L2 命中，实测 fc2 仍被按 request 重读，
    # γ 步循环的时延随并发线性上升（78 MiB × B × 步数）。
    # 批维仍放在 grid 的最快变化轴上，让共享同一 v 分块的 CTA 相邻调度。
    pid_b = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    # Accumulate: latent[BLOCK_B, R] @ fc2_weight[BLOCK_V, R].T
    acc = tl.zeros([BLOCK_B, BLOCK_V], dtype=tl.float32)
    fc2_dtype = fc2_ptr.dtype.element_ty

    for r_start in range(0, R, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R

        # Load latent slice [BLOCK_B, BLOCK_R]
        latent_block = tl.load(
            latent_ptr + offs_b[:, None] * stride_latent_b
            + offs_r[None, :] * stride_latent_r,
            mask=mask_b[:, None] & mask_r[None, :],
            other=0.0,
        ).to(fc2_dtype)

        # fc2 是 [V, R] 行主序，按 [BLOCK_V, BLOCK_R] 读才沿 R 连续（向量化
        # 访存）；转置交给 tl.dot 的操作数布局吸收，不额外过一趟 SMEM。
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
        )

        acc = tl.dot(latent_block, tl.trans(fc2_block), acc)

    # 对齐 eager 的原生精度边界：linear 将 fp32 累加结果写回输入精度，
    # 随后的 logits 加法再次写回该精度。
    input_dtype = base_ptr.dtype.element_ty
    projected = acc.to(input_dtype)
    base_block = tl.load(
        base_ptr + offs_b[:, None] * stride_base_b + offs_v[None, :] * stride_base_v,
        mask=mask_b[:, None] & mask_v[None, :],
        other=-float("inf"),
        cache_modifier=".cg",
    )
    logits = (projected + base_block).to(input_dtype).to(tl.float32)
    logits = tl.where(mask_v[None, :], logits, -float("inf"))

    # 字典序 argmax：值较大者优先，值相同时词表索引较小者优先。
    local_max_val = tl.max(logits, axis=1)
    local_max_idx = tl.min(
        tl.where(
            mask_v[None, :] & (logits == local_max_val[:, None]), offs_v[None, :], V
        ),
        axis=1,
    )

    tl.store(
        out_val_ptr + offs_b * stride_outv_b + pid_v * stride_outv_n,
        local_max_val, mask=mask_b,
    )
    tl.store(
        out_idx_ptr + offs_b * stride_outi_b + pid_v * stride_outi_n,
        local_max_idx, mask=mask_b,
    )


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
# Kernel 1c: Fused latent @ markov_w2.T + base_logits -> write logits -> sample
# ---------------------------------------------------------------------------


@triton.jit
def _dspark_fused_markov_bias_sample_kernel(
    latent_ptr,  # [B, R]
    fc2_ptr,  # [V, R] markov_w2.weight
    base_ptr,  # [B, V] base_logits for current step
    logits_ptr,  # [B, V] destination: corrected logits (q) for the verifier
    temperatures_ptr,  # [B] fp32, already clamped
    greedy_mask_ptr,  # [B] int32, nonzero -> argmax row
    exp_noise_ptr,  # [B, V] fp32 Exp(1)
    out_val_ptr,  # [B, num_v_blocks] fp32
    out_idx_ptr,  # [B, num_v_blocks] int32
    B,
    R,
    V,
    stride_latent_b, stride_latent_r,
    stride_f2_v, stride_f2_r,
    stride_base_b, stride_base_v,
    stride_logits_b, stride_logits_v,
    stride_noise_b, stride_noise_v,
    stride_outv_b, stride_outv_n,
    stride_outi_b, stride_outi_n,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Pass 1 of the sampled path: same tiling as the argmax kernel, plus the
    logits store and the exponential-race key."""
    pid_b = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    mask_bv = mask_b[:, None] & mask_v[None, :]

    acc = tl.zeros([BLOCK_B, BLOCK_V], dtype=tl.float32)
    fc2_dtype = fc2_ptr.dtype.element_ty

    for r_start in range(0, R, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R
        latent_block = tl.load(
            latent_ptr + offs_b[:, None] * stride_latent_b
            + offs_r[None, :] * stride_latent_r,
            mask=mask_b[:, None] & mask_r[None, :],
            other=0.0,
        ).to(fc2_dtype)
        fc2_block = tl.load(
            fc2_ptr
            + offs_v[:, None] * stride_f2_v
            + offs_r[None, :] * stride_f2_r,
            mask=mask_v[:, None] & mask_r[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        acc = tl.dot(latent_block, tl.trans(fc2_block), acc)

    # 精度边界与 argmax kernel 一致（对齐 eager 的 linear + add）。
    input_dtype = base_ptr.dtype.element_ty
    projected = acc.to(input_dtype)
    base_block = tl.load(
        base_ptr + offs_b[:, None] * stride_base_b + offs_v[None, :] * stride_base_v,
        mask=mask_bv,
        other=-float("inf"),
        cache_modifier=".cg",
    )
    logits_native = (projected + base_block).to(input_dtype)
    # 验证端对这份 logits 做 softmax 得到 q，采样必须基于同一份存储精度的数，
    # 否则被抽中的 token 与 q(x) 来自两份不同的分布。
    tl.store(
        logits_ptr
        + offs_b[:, None] * stride_logits_b
        + offs_v[None, :] * stride_logits_v,
        logits_native,
        mask=mask_bv,
    )
    logits = logits_native.to(tl.float32)

    # argmax_i p_i / E_i（E ~ Exp(1)）即按 Categorical(p) 抽样；取 log 后 softmax
    # 的归一化常数沿行恒定，退化为 argmax(logit / T - log E)。greedy 行直接
    # argmax 原 logits，与 eager 中 noise 置 1 的处理等价。
    temperatures = tl.load(temperatures_ptr + offs_b, mask=mask_b, other=1.0)
    greedy = tl.load(greedy_mask_ptr + offs_b, mask=mask_b, other=1) != 0
    noise = tl.load(
        exp_noise_ptr
        + offs_b[:, None] * stride_noise_b
        + offs_v[None, :] * stride_noise_v,
        mask=mask_bv,
        other=1.0,
        cache_modifier=".cg",
    )
    sampled_key = logits / temperatures[:, None] - tl.log(noise)
    key = tl.where(greedy[:, None], logits, sampled_key)
    key = tl.where(mask_v[None, :], key, -float("inf"))

    local_max_val = tl.max(key, axis=1)
    local_max_idx = tl.min(
        tl.where(mask_v[None, :] & (key == local_max_val[:, None]), offs_v[None, :], V),
        axis=1,
    )

    tl.store(
        out_val_ptr + offs_b * stride_outv_b + pid_v * stride_outv_n,
        local_max_val, mask=mask_b,
    )
    tl.store(
        out_idx_ptr + offs_b * stride_outi_b + pid_v * stride_outi_n,
        local_max_idx, mask=mask_b,
    )


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
    latent: torch.Tensor,            # [B, R] materialized latent
    fc2_weight: torch.Tensor,        # [V, R] markov_w2.weight
    base_logits: torch.Tensor,       # [B, V]
    out_val: torch.Tensor,           # [B, num_v_blocks] fp32 scratch
    out_idx: torch.Tensor,           # [B, num_v_blocks] int32 scratch
    final_token: torch.Tensor,       # [B] int64 destination
    block_v: int = MARKOV_BLOCK_V,
    block_r: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    """Fused: ``latent @ fc2_weight.T + base_logits`` -> argmax.

    ``latent`` must already be materialized as a contiguous ``[B, R]`` tensor
    (the embedding row for the vanilla head, ``tanh(output_raw)`` for the RNN
    head). Doing the embedding lookup inside the kernel instead would re-gather
    the same table row from every one of the ``B * ceil(V / BLOCK_V)`` CTAs of
    every step, which made draft latency scale steeply with batch size.
    """
    B = base_logits.shape[0]
    V = base_logits.shape[1]
    R = fc2_weight.shape[1]
    num_v_blocks = (V + block_v - 1) // block_v
    block_b = _score_block_b(B)

    # dim0 必须是批维，见 kernel 内的 L2 复用说明。
    grid_main = ((B + block_b - 1) // block_b, num_v_blocks)
    _dspark_fused_markov_bias_argmax_kernel[grid_main](
        latent,
        fc2_weight, base_logits,
        out_val, out_idx,
        B, R, V,
        latent.stride(0), latent.stride(1),
        fc2_weight.stride(0), fc2_weight.stride(1),
        base_logits.stride(0), base_logits.stride(1),
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        BLOCK_B=block_b,
        BLOCK_V=block_v,
        BLOCK_R=block_r,
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


def fused_markov_bias_sample(
    *,
    latent: torch.Tensor,            # [B, R] materialized latent
    fc2_weight: torch.Tensor,        # [V, R] markov_w2.weight
    base_logits: torch.Tensor,       # [B, V]
    logits_out: torch.Tensor,        # [B, V] destination, base_logits dtype
    temperatures: torch.Tensor,      # [B] fp32, clamped > 0
    greedy_mask: torch.Tensor,       # [B] int32, nonzero -> argmax row
    exp_noise: torch.Tensor,         # [B, V] fp32 Exp(1)
    out_val: torch.Tensor,           # [B, num_v_blocks] fp32 scratch
    out_idx: torch.Tensor,           # [B, num_v_blocks] int32 scratch
    final_token: torch.Tensor,       # [B] int64 destination
    block_v: int = MARKOV_BLOCK_V,
    block_r: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    """Fused: ``latent @ fc2_weight.T + base_logits`` -> store logits -> sample.

    Sampled counterpart of ``fused_markov_bias_argmax`` for the rejection
    sampling path: the corrected logit row is written to ``logits_out`` (the
    verifier needs q), and the token is drawn by exponential race against the
    same stored values, per-row temperature, with greedy rows argmax'd. The
    chain linear -> add -> softmax -> noise divide -> argmax collapses into one
    full-vocab pass plus the shared tile reduce.

    ``exp_noise`` is generated by the caller (``Tensor.exponential_``), same
    trade-off as ``domino_kernels.fused_silu_fc2_sample``. ``logits_out`` may
    alias ``base_logits``: every tile loads its base block before storing the
    corrected block to the same addresses.
    """
    B = base_logits.shape[0]
    V = base_logits.shape[1]
    R = fc2_weight.shape[1]
    if temperatures.dtype != torch.float32 or greedy_mask.dtype != torch.int32:
        raise ValueError(
            "fused_markov_bias_sample expects fp32 temperatures and int32 "
            f"greedy_mask, got {temperatures.dtype} / {greedy_mask.dtype}."
        )
    if exp_noise.dtype != torch.float32 or logits_out.dtype != base_logits.dtype:
        raise ValueError(
            "fused_markov_bias_sample expects fp32 exp_noise and logits_out in "
            f"the base_logits dtype, got {exp_noise.dtype} / {logits_out.dtype}."
        )
    num_v_blocks = (V + block_v - 1) // block_v
    block_b = _score_block_b(B)

    # dim0 必须是批维，见 argmax kernel 内的 L2 复用说明。
    grid_main = ((B + block_b - 1) // block_b, num_v_blocks)
    _dspark_fused_markov_bias_sample_kernel[grid_main](
        latent,
        fc2_weight, base_logits,
        logits_out,
        temperatures, greedy_mask, exp_noise,
        out_val, out_idx,
        B, R, V,
        latent.stride(0), latent.stride(1),
        fc2_weight.stride(0), fc2_weight.stride(1),
        base_logits.stride(0), base_logits.stride(1),
        logits_out.stride(0), logits_out.stride(1),
        exp_noise.stride(0), exp_noise.stride(1),
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        BLOCK_B=block_b,
        BLOCK_V=block_v,
        BLOCK_R=block_r,
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
        num_v_blocks = (vocab_size + MARKOV_BLOCK_V - 1) // MARKOV_BLOCK_V
        self.out_val = torch.empty((max_bs, num_v_blocks), dtype=torch.float32, device=device)
        self.out_idx = torch.empty((max_bs, num_v_blocks), dtype=torch.int32, device=device)
        self.tok_buf = torch.empty((max_bs,), dtype=torch.int64, device=device)
        # 两条路径都把 latent 物化到这里再交给 kernel，见
        # fused_markov_bias_argmax 的 docstring。
        self.latent_buf = torch.empty((max_bs, markov_rank), dtype=dtype, device=device)
        if is_rnn:
            self.rnn_state_buf = torch.zeros((max_bs, markov_rank), dtype=dtype, device=device)
            self.rnn_new_state_buf = torch.empty((max_bs, markov_rank), dtype=dtype, device=device)
            self.rnn_proj_buf = torch.empty((max_bs, 3 * markov_rank), dtype=dtype, device=device)
        else:
            self.rnn_state_buf = None
            self.rnn_new_state_buf = None
            self.rnn_proj_buf = None
