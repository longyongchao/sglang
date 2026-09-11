"""Triton kernels for the DFLASH Domino sequential rollout.

The per-step body of Domino rollout is dominated by tiny ops at bs=1. Even with a
CUDA graph capturing the loop, the per-step work is fundamentally limited by
fc2 ([B,256] @ [256, V=152K]) bandwidth and a handful of elementwise/reduce
launches. These kernels collapse the small ops:

  1. ``fused_silu_fc2_argmax`` — fuses ``SiLU(z+s) @ fc2.T + fc2_bias +
     base_logits`` into a single per-step matmul that emits per-block
     (val, idx) pairs, followed by a reduction kernel that produces the global
     argmax token. Avoids materializing the full [B, V] bias tensor.
  2. ``fused_gru_cell_from_table`` — fuses GRU input-table lookup and the
     sigmoid/tanh/gate update into one kernel.
  3. ``fused_silu_fc2_sample`` — the rejection-sampling variant of (1). The
     verifier needs the whole logit row, so this one does materialize [B, V],
     but it writes it exactly once and finishes the categorical draw in the same
     pass, instead of following the matmul with separate bias-add, softmax,
     noise and argmax passes.

All kernels are CUDA-graph friendly (no host-driven sync, fixed shapes).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _score_block_b(batch_size: int) -> int:
    """打分 kernel 的批分块大小：``tl.dot`` 要求各维 >= 16，上限取 64。

    分块越大，fc2 的重读次数 ``ceil(B / BLOCK_B)`` 越少；超过 64 后收益反转，
    因为累加器 ``[BLOCK_B, BLOCK_V]`` 的 SMEM 占用把每 SM 的 CTA 并发压到 1
    （实测 bs=128 下 BLOCK_B 64->128 是 0.079 -> 0.103 ms，寄存器未溢出）。
    """
    if batch_size <= 16:
        return 16
    if batch_size <= 32:
        return 32
    return 64


@triton.jit
def _domino_fused_silu_fc2_argmax_kernel(
    z_ptr, s_ptr,
    fc2_ptr, base_ptr, fc2_bias_ptr,
    out_val_ptr, out_idx_ptr,
    B, M, V,
    stride_z_b, stride_z_m,
    stride_s_b, stride_s_m,
    stride_f2_v, stride_f2_m,
    stride_base_b, stride_base_v,
    stride_outv_b, stride_outv_n,
    stride_outi_b, stride_outi_n,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_FC2_BIAS: tl.constexpr,
):
    # 一个 CTA 同时算 BLOCK_B 个 request：fc2 的 [BLOCK_V, M] 切片进 SMEM 后被
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

    acc = tl.zeros([BLOCK_B, BLOCK_V], dtype=tl.float32)
    fc2_dtype = fc2_ptr.dtype.element_ty

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        mask_bm = mask_b[:, None] & mask_m[None, :]

        z_block = tl.load(
            z_ptr + offs_b[:, None] * stride_z_b + offs_m[None, :] * stride_z_m,
            mask=mask_bm, other=0.0,
        ).to(tl.float32)
        s_block = tl.load(
            s_ptr + offs_b[:, None] * stride_s_b + offs_m[None, :] * stride_s_m,
            mask=mask_bm, other=0.0,
        ).to(tl.float32)
        preact = z_block + s_block
        # eager 参考实现是 ``F.linear(F.silu(z + s), fc2_w)``，即 mid 落回
        # 输入精度后交给 cuBLAS 做 bf16 MMA；这里保持同一个精度边界。
        mid_block = (preact * tl.sigmoid(preact)).to(fc2_dtype)

        # fc2 是 [V, M] 行主序，按 [BLOCK_V, BLOCK_M] 读才沿 M 连续（向量化
        # 访存）；转置交给 tl.dot 的操作数布局吸收，不额外过一趟 SMEM。
        fc2_ptrs = (
            fc2_ptr
            + offs_v[:, None] * stride_f2_v
            + offs_m[None, :] * stride_f2_m
        )
        fc2_block = tl.load(
            fc2_ptrs,
            mask=mask_v[:, None] & mask_m[None, :],
            other=0.0,
            cache_modifier=".cg",
        )

        acc = tl.dot(mid_block, tl.trans(fc2_block), acc)

    if HAS_FC2_BIAS:
        bias_block = tl.load(
            fc2_bias_ptr + offs_v, mask=mask_v, other=0.0
        ).to(tl.float32)
        acc += bias_block[None, :]

    base_block = tl.load(
        base_ptr + offs_b[:, None] * stride_base_b + offs_v[None, :] * stride_base_v,
        mask=mask_b[:, None] & mask_v[None, :], other=-float("inf"),
        cache_modifier=".cg",
    ).to(tl.float32)
    acc += base_block

    acc = tl.where(mask_v[None, :], acc, -float("inf"))

    local_max_val = tl.max(acc, axis=1)
    local_max_idx = tl.argmax(acc, axis=1)
    global_max_idx = pid_v * BLOCK_V + local_max_idx

    tl.store(
        out_val_ptr + offs_b * stride_outv_b + pid_v * stride_outv_n,
        local_max_val, mask=mask_b,
    )
    tl.store(
        out_idx_ptr + offs_b * stride_outi_b + pid_v * stride_outi_n,
        global_max_idx, mask=mask_b,
    )


@triton.jit
def _domino_reduce_argmax_kernel(
    out_val_ptr, out_idx_ptr, final_token_ptr,
    N,
    stride_val_b, stride_val_n,
    stride_idx_b, stride_idx_n,
    stride_final_b,
    BLOCK_N: tl.constexpr,
):
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
        local_pos = tl.argmax(vals, axis=0)
        local_val = tl.max(vals, axis=0)
        local_idx = tl.load(
            out_idx_ptr + pid_b * stride_idx_b
            + (start_n + local_pos) * stride_idx_n
        )

        take = local_val > best_val
        best_val = tl.where(take, local_val, best_val)
        best_idx = tl.where(take, local_idx, best_idx)

    tl.store(final_token_ptr + pid_b * stride_final_b, best_idx)


@triton.jit
def _domino_reduce_argmax_with_value_kernel(
    out_val_ptr, out_idx_ptr, final_token_ptr, final_value_ptr,
    N,
    stride_val_b, stride_val_n,
    stride_idx_b, stride_idx_n,
    stride_final_b,
    stride_final_value_b,
    BLOCK_N: tl.constexpr,
):
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
        local_pos = tl.argmax(vals, axis=0)
        local_val = tl.max(vals, axis=0)
        local_idx = tl.load(
            out_idx_ptr + pid_b * stride_idx_b
            + (start_n + local_pos) * stride_idx_n
        )

        take = local_val > best_val
        best_val = tl.where(take, local_val, best_val)
        best_idx = tl.where(take, local_idx, best_idx)

    tl.store(final_token_ptr + pid_b * stride_final_b, best_idx)
    tl.store(final_value_ptr + pid_b * stride_final_value_b, best_val)



@triton.jit
def _domino_fused_silu_fc2_candidate_argmax_kernel(
    z_ptr, s_ptr,
    fc2_ptr, base_ptr, fc2_bias_ptr, cand_ptr,
    out_val_ptr, out_idx_ptr,
    M, V, C,
    stride_z_b, stride_z_m,
    stride_s_b, stride_s_m,
    stride_f2_v, stride_f2_m,
    stride_base_b, stride_base_v,
    stride_cand_b, stride_cand_c,
    stride_outv_b, stride_outv_n,
    stride_outi_b, stride_outi_n,
    BLOCK_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_FC2_BIAS: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    cand_idx = tl.load(
        cand_ptr + pid_b * stride_cand_b + offs_c * stride_cand_c,
        mask=mask_c, other=0,
    )
    cand_valid = mask_c & (cand_idx >= 0) & (cand_idx < V)

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        z_block = tl.load(
            z_ptr + pid_b * stride_z_b + offs_m * stride_z_m,
            mask=mask_m, other=0.0,
        ).to(tl.float32)
        s_block = tl.load(
            s_ptr + pid_b * stride_s_b + offs_m * stride_s_m,
            mask=mask_m, other=0.0,
        ).to(tl.float32)
        preact = z_block + s_block
        mid_block = preact * tl.sigmoid(preact)

        fc2_ptrs = (
            fc2_ptr
            + cand_idx[:, None] * stride_f2_v
            + offs_m[None, :] * stride_f2_m
        )
        fc2_block = tl.load(
            fc2_ptrs,
            mask=cand_valid[:, None] & mask_m[None, :],
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)

        acc += tl.sum(fc2_block * mid_block[None, :], axis=1)

    if HAS_FC2_BIAS:
        bias_block = tl.load(
            fc2_bias_ptr + cand_idx, mask=cand_valid, other=0.0
        ).to(tl.float32)
        acc += bias_block

    base_block = tl.load(
        base_ptr + pid_b * stride_base_b + cand_idx * stride_base_v,
        mask=cand_valid, other=-float("inf"),
        cache_modifier=".cg",
    ).to(tl.float32)
    acc += base_block

    acc = tl.where(cand_valid, acc, -float("inf"))

    local_max_val = tl.max(acc, axis=0)
    local_max_pos = tl.argmax(acc, axis=0)
    local_max_idx = tl.load(
        cand_ptr
        + pid_b * stride_cand_b
        + (pid_c * BLOCK_C + local_max_pos) * stride_cand_c
    )

    out_val_offs = pid_b * stride_outv_b + pid_c * stride_outv_n
    out_idx_offs = pid_b * stride_outi_b + pid_c * stride_outi_n
    tl.store(out_val_ptr + out_val_offs, local_max_val)
    tl.store(out_idx_ptr + out_idx_offs, local_max_idx)


def fused_silu_fc2_argmax(
    *,
    z_proj: torch.Tensor,        # [B, M]
    s_proj: torch.Tensor,        # [B, M]
    fc2_weight: torch.Tensor,    # [V, M]
    fc2_bias: torch.Tensor | None,  # [V] or None
    base_logits: torch.Tensor,   # [B, V]
    out_val: torch.Tensor,       # [B, num_v_blocks] fp32 scratch
    out_idx: torch.Tensor,       # [B, num_v_blocks] int32 scratch
    final_token: torch.Tensor,   # [B] int64 destination
    block_v: int = 128,
    block_m: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    """Single-step fused: SiLU(z+s) @ fc2.T + bias + base_logits -> argmax.

    Writes the per-batch global argmax into ``final_token`` in-place.
    """
    B, M = z_proj.shape
    V = base_logits.shape[1]
    num_v_blocks = (V + block_v - 1) // block_v
    block_b = _score_block_b(B)
    has_fc2_bias = fc2_bias is not None
    fc2_bias_arg = fc2_bias if has_fc2_bias else out_val  # placeholder ptr

    # dim0 必须是批维，见 kernel 内的 L2 复用说明。
    grid_main = ((B + block_b - 1) // block_b, num_v_blocks)
    _domino_fused_silu_fc2_argmax_kernel[grid_main](
        z_proj, s_proj,
        fc2_weight, base_logits, fc2_bias_arg,
        out_val, out_idx,
        B, M, V,
        z_proj.stride(0), z_proj.stride(1),
        s_proj.stride(0), s_proj.stride(1),
        fc2_weight.stride(0), fc2_weight.stride(1),
        base_logits.stride(0), base_logits.stride(1),
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        BLOCK_B=block_b,
        BLOCK_V=block_v,
        BLOCK_M=block_m,
        HAS_FC2_BIAS=has_fc2_bias,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _domino_reduce_argmax_kernel[(B,)](
        out_val, out_idx, final_token,
        num_v_blocks,
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        final_token.stride(0),
        BLOCK_N=512,
        num_warps=1,
    )


@triton.jit
def _domino_fused_silu_fc2_sample_kernel(
    z_ptr, s_ptr,
    fc2_ptr, base_ptr, fc2_bias_ptr,
    logits_ptr, exp_noise_ptr,
    out_val_ptr, out_idx_ptr,
    B, M, V,
    stride_z_b, stride_z_m,
    stride_s_b, stride_s_m,
    stride_f2_v, stride_f2_m,
    stride_base_b, stride_base_v,
    stride_logits_b, stride_logits_v,
    stride_noise_b, stride_noise_v,
    stride_outv_b, stride_outv_n,
    stride_outi_b, stride_outi_n,
    inv_temperature,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_FC2_BIAS: tl.constexpr,
    HAS_NOISE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    acc = tl.zeros([BLOCK_B, BLOCK_V], dtype=tl.float32)
    fc2_dtype = fc2_ptr.dtype.element_ty

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        mask_bm = mask_b[:, None] & mask_m[None, :]

        z_block = tl.load(
            z_ptr + offs_b[:, None] * stride_z_b + offs_m[None, :] * stride_z_m,
            mask=mask_bm, other=0.0,
        ).to(tl.float32)
        s_block = tl.load(
            s_ptr + offs_b[:, None] * stride_s_b + offs_m[None, :] * stride_s_m,
            mask=mask_bm, other=0.0,
        ).to(tl.float32)
        preact = z_block + s_block
        mid_block = (preact * tl.sigmoid(preact)).to(fc2_dtype)

        fc2_ptrs = (
            fc2_ptr
            + offs_v[:, None] * stride_f2_v
            + offs_m[None, :] * stride_f2_m
        )
        fc2_block = tl.load(
            fc2_ptrs,
            mask=mask_v[:, None] & mask_m[None, :],
            other=0.0,
            cache_modifier=".cg",
        )

        acc = tl.dot(mid_block, tl.trans(fc2_block), acc)

    if HAS_FC2_BIAS:
        bias_block = tl.load(
            fc2_bias_ptr + offs_v, mask=mask_v, other=0.0
        ).to(tl.float32)
        acc += bias_block[None, :]

    base_block = tl.load(
        base_ptr + offs_b[:, None] * stride_base_b + offs_v[None, :] * stride_base_v,
        mask=mask_b[:, None] & mask_v[None, :], other=-float("inf"),
        cache_modifier=".cg",
    ).to(tl.float32)
    acc += base_block

    # 验证阶段读的是这份 logits，所以采样也必须在同一个精度边界上进行：先落回
    # 存储精度，再拿它算 key，避免 q(x) 与被抽中的 token 来自两份不同的数。
    logits_block = acc.to(logits_ptr.dtype.element_ty)
    tl.store(
        logits_ptr
        + offs_b[:, None] * stride_logits_b
        + offs_v[None, :] * stride_logits_v,
        logits_block,
        mask=mask_b[:, None] & mask_v[None, :],
    )

    key = logits_block.to(tl.float32)
    if HAS_NOISE:
        # E ~ Exp(1) 时 argmax_i p_i / E_i 就是从 Categorical(p) 抽样；取 log 之后
        # softmax 的归一化常数沿行恒定可以丢掉，于是退化成一次普通 argmax。噪声由
        # curand 在 kernel 外批量生成：kernel 内的 philox 每个元素要跑满 10 轮，
        # 实测 bs=64 时比读一份预生成的噪声贵 4 倍。
        noise = tl.load(
            exp_noise_ptr
            + offs_b[:, None] * stride_noise_b
            + offs_v[None, :] * stride_noise_v,
            mask=mask_b[:, None] & mask_v[None, :],
            other=1.0,
            cache_modifier=".cg",
        )
        key = key * inv_temperature - tl.log(noise)
    key = tl.where(mask_v[None, :], key, -float("inf"))

    local_max_val = tl.max(key, axis=1)
    local_max_idx = tl.argmax(key, axis=1)
    global_max_idx = pid_v * BLOCK_V + local_max_idx

    tl.store(
        out_val_ptr + offs_b * stride_outv_b + pid_v * stride_outv_n,
        local_max_val, mask=mask_b,
    )
    tl.store(
        out_idx_ptr + offs_b * stride_outi_b + pid_v * stride_outi_n,
        global_max_idx, mask=mask_b,
    )


def fused_silu_fc2_sample(
    *,
    z_proj: torch.Tensor,        # [B, M]
    s_proj: torch.Tensor,        # [B, M]
    fc2_weight: torch.Tensor,    # [V, M]
    fc2_bias: torch.Tensor | None,  # [V] or None
    base_logits: torch.Tensor,   # [B, V]
    logits_out: torch.Tensor,    # [B, V] destination, base_logits dtype
    exp_noise: torch.Tensor | None,  # [B, V] fp32 Exp(1), required when sampling
    inv_temperature: float,      # 1 / T, <= 0 means greedy
    out_val: torch.Tensor,       # [B, num_v_blocks] fp32 scratch
    out_idx: torch.Tensor,       # [B, num_v_blocks] int32 scratch
    final_token: torch.Tensor,   # [B] int64 destination, local vocab ids
    block_v: int = 128,
    block_m: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    """Fused scoring + categorical draw for the rejection-sampling rollout.

    Same matmul as ``fused_silu_fc2_argmax``, but the logit row is written to
    ``logits_out`` (the verifier needs q) and the token is drawn instead of
    argmax'd. Both happen while the block is still in registers, so the chain
    fc2 -> bias add -> softmax -> noise divide -> argmax collapses from five
    full-vocab passes to one.

    ``exp_noise`` is generated by the caller (``Tensor.exponential_``) rather
    than inside the kernel; see the noise comment in the kernel.
    """
    B, M = z_proj.shape
    V = base_logits.shape[1]
    num_v_blocks = (V + block_v - 1) // block_v
    block_b = _score_block_b(B)
    has_fc2_bias = fc2_bias is not None
    fc2_bias_arg = fc2_bias if has_fc2_bias else out_val  # placeholder ptr
    has_noise = inv_temperature > 0
    if has_noise and exp_noise is None:
        raise ValueError("fused_silu_fc2_sample needs exp_noise when sampling")
    noise_arg = exp_noise if has_noise else out_val  # placeholder ptr

    # dim0 必须是批维，见 kernel 内的 L2 复用说明。
    grid_main = ((B + block_b - 1) // block_b, num_v_blocks)
    _domino_fused_silu_fc2_sample_kernel[grid_main](
        z_proj, s_proj,
        fc2_weight, base_logits, fc2_bias_arg,
        logits_out, noise_arg,
        out_val, out_idx,
        B, M, V,
        z_proj.stride(0), z_proj.stride(1),
        s_proj.stride(0), s_proj.stride(1),
        fc2_weight.stride(0), fc2_weight.stride(1),
        base_logits.stride(0), base_logits.stride(1),
        logits_out.stride(0), logits_out.stride(1),
        noise_arg.stride(0), noise_arg.stride(1),
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        float(inv_temperature),
        BLOCK_B=block_b,
        BLOCK_V=block_v,
        BLOCK_M=block_m,
        HAS_FC2_BIAS=has_fc2_bias,
        HAS_NOISE=has_noise,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _domino_reduce_argmax_kernel[(B,)](
        out_val, out_idx, final_token,
        num_v_blocks,
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        final_token.stride(0),
        BLOCK_N=512,
        num_warps=1,
    )


def fused_silu_fc2_argmax_with_value(
    *,
    z_proj: torch.Tensor,        # [B, M]
    s_proj: torch.Tensor,        # [B, M]
    fc2_weight: torch.Tensor,    # [V, M]
    fc2_bias: torch.Tensor | None,  # [V] or None
    base_logits: torch.Tensor,   # [B, V]
    out_val: torch.Tensor,       # [B, num_v_blocks] fp32 scratch
    out_idx: torch.Tensor,       # [B, num_v_blocks] int32 scratch
    final_token: torch.Tensor,   # [B] int64 destination, local vocab ids
    final_value: torch.Tensor,   # [B] fp32 destination, local max values
    block_v: int = 128,
    block_m: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    """Fused full-shard scoring that also returns the local max score."""
    B, M = z_proj.shape
    V = base_logits.shape[1]
    num_v_blocks = (V + block_v - 1) // block_v
    block_b = _score_block_b(B)
    has_fc2_bias = fc2_bias is not None
    fc2_bias_arg = fc2_bias if has_fc2_bias else out_val

    # dim0 必须是批维，见 kernel 内的 L2 复用说明。
    grid_main = ((B + block_b - 1) // block_b, num_v_blocks)
    _domino_fused_silu_fc2_argmax_kernel[grid_main](
        z_proj, s_proj,
        fc2_weight, base_logits, fc2_bias_arg,
        out_val, out_idx,
        B, M, V,
        z_proj.stride(0), z_proj.stride(1),
        s_proj.stride(0), s_proj.stride(1),
        fc2_weight.stride(0), fc2_weight.stride(1),
        base_logits.stride(0), base_logits.stride(1),
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        BLOCK_B=block_b,
        BLOCK_V=block_v,
        BLOCK_M=block_m,
        HAS_FC2_BIAS=has_fc2_bias,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _domino_reduce_argmax_with_value_kernel[(B,)](
        out_val, out_idx, final_token, final_value,
        num_v_blocks,
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        final_token.stride(0),
        final_value.stride(0),
        BLOCK_N=512,
        num_warps=1,
    )


def fused_silu_fc2_candidate_argmax(
    *,
    z_proj: torch.Tensor,        # [B, M]
    s_proj: torch.Tensor,        # [B, M]
    fc2_weight: torch.Tensor,    # [V, M]
    fc2_bias: torch.Tensor | None,  # [V] or None
    base_logits: torch.Tensor,   # [B, V]
    candidate_ids: torch.Tensor, # [B, C] local vocab ids into fc2/base_logits
    out_val: torch.Tensor,       # [B, num_candidate_blocks] fp32 scratch
    out_idx: torch.Tensor,       # [B, num_candidate_blocks] int32 scratch
    final_token: torch.Tensor,   # [B] int64 destination, local vocab ids
    block_c: int = 512,
    block_m: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    """Candidate-restricted Domino scoring over a shared per-block token pool.

    ``candidate_ids`` contains local vocab ids for the current TP shard. The
    output token ids are local ids, matching ``fused_silu_fc2_argmax`` semantics.
    """
    B, M = z_proj.shape
    V = base_logits.shape[1]
    C = candidate_ids.shape[1]
    num_c_blocks = (C + block_c - 1) // block_c
    has_fc2_bias = fc2_bias is not None
    fc2_bias_arg = fc2_bias if has_fc2_bias else out_val  # placeholder ptr

    grid_main = (num_c_blocks, B)
    _domino_fused_silu_fc2_candidate_argmax_kernel[grid_main](
        z_proj, s_proj,
        fc2_weight, base_logits, fc2_bias_arg, candidate_ids,
        out_val, out_idx,
        M, V, C,
        z_proj.stride(0), z_proj.stride(1),
        s_proj.stride(0), s_proj.stride(1),
        fc2_weight.stride(0), fc2_weight.stride(1),
        base_logits.stride(0), base_logits.stride(1),
        candidate_ids.stride(0), candidate_ids.stride(1),
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        BLOCK_C=block_c,
        BLOCK_M=block_m,
        HAS_FC2_BIAS=has_fc2_bias,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _domino_reduce_argmax_kernel[(B,)](
        out_val, out_idx, final_token,
        num_c_blocks,
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        final_token.stride(0),
        BLOCK_N=512,
        num_warps=1,
    )


def fused_silu_fc2_candidate_argmax_with_value(
    *,
    z_proj: torch.Tensor,        # [B, M]
    s_proj: torch.Tensor,        # [B, M]
    fc2_weight: torch.Tensor,    # [V, M]
    fc2_bias: torch.Tensor | None,  # [V] or None
    base_logits: torch.Tensor,   # [B, V]
    candidate_ids: torch.Tensor, # [B, C] local vocab ids into fc2/base_logits
    out_val: torch.Tensor,       # [B, num_candidate_blocks] fp32 scratch
    out_idx: torch.Tensor,       # [B, num_candidate_blocks] int32 scratch
    final_token: torch.Tensor,   # [B] int64 destination, local vocab ids
    final_value: torch.Tensor,   # [B] fp32 destination, local max values
    block_c: int = 512,
    block_m: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    """Candidate-restricted fused scoring that also returns the local max score."""
    B, M = z_proj.shape
    V = base_logits.shape[1]
    C = candidate_ids.shape[1]
    num_c_blocks = (C + block_c - 1) // block_c
    has_fc2_bias = fc2_bias is not None
    fc2_bias_arg = fc2_bias if has_fc2_bias else out_val

    grid_main = (num_c_blocks, B)
    _domino_fused_silu_fc2_candidate_argmax_kernel[grid_main](
        z_proj, s_proj,
        fc2_weight, base_logits, fc2_bias_arg, candidate_ids,
        out_val, out_idx,
        M, V, C,
        z_proj.stride(0), z_proj.stride(1),
        s_proj.stride(0), s_proj.stride(1),
        fc2_weight.stride(0), fc2_weight.stride(1),
        base_logits.stride(0), base_logits.stride(1),
        candidate_ids.stride(0), candidate_ids.stride(1),
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        BLOCK_C=block_c,
        BLOCK_M=block_m,
        HAS_FC2_BIAS=has_fc2_bias,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _domino_reduce_argmax_with_value_kernel[(B,)](
        out_val, out_idx, final_token, final_value,
        num_c_blocks,
        out_val.stride(0), out_val.stride(1),
        out_idx.stride(0), out_idx.stride(1),
        final_token.stride(0),
        final_value.stride(0),
        BLOCK_N=512,
        num_warps=1,
    )


@triton.jit
def _domino_fused_gru_cell_from_table_kernel(
    tok_ptr, table_ptr, gh_ptr, gh_bias_ptr, h_ptr, h_out_ptr,
    G, V,
    stride_table_v, stride_table_g,
    stride_gh_b, stride_gh_g,
    stride_gh_bias_g,
    stride_h_b, stride_h_g,
    stride_hout_b, stride_hout_g,
    BLOCK_G: tl.constexpr,
    HAS_GH_BIAS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    mask_g = offs_g < G

    tok_id_raw = tl.load(tok_ptr + pid_b)
    tok_in_range = (tok_id_raw >= 0) & (tok_id_raw < V)
    tok_id = tl.where(tok_in_range, tok_id_raw, 0)
    mask_tok = mask_g & tok_in_range

    h_state = tl.load(
        h_ptr + pid_b * stride_h_b + offs_g * stride_h_g,
        mask=mask_g, other=0.0,
    ).to(tl.float32)

    row_base = tok_id * stride_table_v
    gi_r = tl.load(
        table_ptr + row_base + offs_g * stride_table_g,
        mask=mask_tok, other=0.0,
    ).to(tl.float32)
    gi_z = tl.load(
        table_ptr + row_base + (G + offs_g) * stride_table_g,
        mask=mask_tok, other=0.0,
    ).to(tl.float32)
    gi_n = tl.load(
        table_ptr + row_base + (2 * G + offs_g) * stride_table_g,
        mask=mask_tok, other=0.0,
    ).to(tl.float32)

    gh_r = tl.load(
        gh_ptr + pid_b * stride_gh_b + offs_g * stride_gh_g,
        mask=mask_g, other=0.0,
    ).to(tl.float32)
    gh_z = tl.load(
        gh_ptr + pid_b * stride_gh_b + (G + offs_g) * stride_gh_g,
        mask=mask_g, other=0.0,
    ).to(tl.float32)
    gh_n = tl.load(
        gh_ptr + pid_b * stride_gh_b + (2 * G + offs_g) * stride_gh_g,
        mask=mask_g, other=0.0,
    ).to(tl.float32)

    if HAS_GH_BIAS:
        gh_r = (
            gh_r
            + tl.load(
                gh_bias_ptr + offs_g * stride_gh_bias_g,
                mask=mask_g, other=0.0,
            ).to(tl.float32)
        ).to(gh_ptr.dtype.element_ty).to(tl.float32)
        gh_z = (
            gh_z
            + tl.load(
                gh_bias_ptr + (G + offs_g) * stride_gh_bias_g,
                mask=mask_g, other=0.0,
            ).to(tl.float32)
        ).to(gh_ptr.dtype.element_ty).to(tl.float32)
        gh_n = (
            gh_n
            + tl.load(
                gh_bias_ptr + (2 * G + offs_g) * stride_gh_bias_g,
                mask=mask_g, other=0.0,
            ).to(tl.float32)
        ).to(gh_ptr.dtype.element_ty).to(tl.float32)

    r = tl.sigmoid(gi_r + gh_r)
    z = tl.sigmoid(gi_z + gh_z)
    n = 2.0 * tl.sigmoid(2.0 * (gi_n + r * gh_n)) - 1.0
    h_new = (1.0 - z) * n + z * h_state

    tl.store(
        h_out_ptr + pid_b * stride_hout_b + offs_g * stride_hout_g,
        h_new.to(h_ptr.dtype.element_ty),
        mask=mask_g,
    )


def fused_gru_cell_from_table(
    *,
    tok_full: torch.Tensor,       # [B] int64 token ids (already shifted by org_vocab_start)
    gru_input_table: torch.Tensor,  # [V, 3*G] precomputed embed @ W_ih.T + b_ih
    gh: torch.Tensor,             # [B, 3*G] h_state @ W_hh.T (+ b_hh)
    gh_bias: torch.Tensor | None = None,  # [3*G] or None
    h_state: torch.Tensor,        # [B, G]
    h_out: torch.Tensor,          # [B, G] destination
    block_g: int = 256,
) -> None:
    """Fused: gi = gru_input_table[tok_full]; then GRU cell update.

    Replaces the index_select + fused_gru_cell pair with a single kernel that
    reads gi directly from the table by tok id, skipping the gi_buf materialize.
    """
    B = tok_full.shape[0]
    G = h_state.shape[1]
    V = gru_input_table.shape[0]
    has_gh_bias = gh_bias is not None
    gh_bias_arg = gh_bias if has_gh_bias else gh  # placeholder ptr
    gh_bias_stride = gh_bias_arg.stride(0) if has_gh_bias else 0
    grid = (B, (G + block_g - 1) // block_g)
    _domino_fused_gru_cell_from_table_kernel[grid](
        tok_full, gru_input_table, gh, gh_bias_arg, h_state, h_out,
        G, V,
        gru_input_table.stride(0), gru_input_table.stride(1),
        gh.stride(0), gh.stride(1),
        gh_bias_stride,
        h_state.stride(0), h_state.stride(1),
        h_out.stride(0), h_out.stride(1),
        BLOCK_G=block_g,
        HAS_GH_BIAS=has_gh_bias,
    )
