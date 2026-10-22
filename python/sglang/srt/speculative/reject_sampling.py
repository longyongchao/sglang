from typing import Optional, Union

import torch
import triton
import triton.language as tl


@triton.jit
def _row_lse_kernel(
    Logits,
    Lse,
    Temperatures,
    stride_l_b,
    stride_l_s,
    rows_per_b,
    n_cols,
    temperature,
    BLOCK: tl.constexpr,
    PER_REQ_TEMP: tl.constexpr,
):
    """``Lse[b, s] = logsumexp(Logits[b, s, :] / T_b)``，fp32 累加，单趟在线算法。"""
    row = tl.program_id(0)
    b = row // rows_per_b
    s = row % rows_per_b
    if PER_REQ_TEMP:
        temp = tl.load(Temperatures + b).to(tl.float32)
    else:
        temp = temperature
    base = Logits + b.to(tl.int64) * stride_l_b + s * stride_l_s
    run_max = -float("inf")
    run_sum = 0.0
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(base + offs, mask=offs < n_cols, other=-float("inf"))
        x = x.to(tl.float32) / temp
        new_max = tl.maximum(run_max, tl.max(x, axis=0))
        # 整块 -inf 时 new_max 仍是 -inf，exp(-inf - -inf) 会出 nan，单独置零
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        run_sum = run_sum * tl.exp(run_max - safe_max) + tl.sum(
            tl.exp(x - safe_max), axis=0
        )
        run_max = new_max
    tl.store(Lse + row, run_max + tl.log(run_sum))


@triton.jit
def _load_prob(
    Ptr,
    Lse,
    Temperatures,
    pid,
    row,
    v_offsets,
    v_mask,
    stride_b,
    stride_s,
    stride_v,
    stride_lse_b,
    stride_lse_s,
    temperature,
    vocab_start,
    vocab_len,
    num_rows,
    FROM_LOGITS: tl.constexpr,
    PER_REQ_TEMP: tl.constexpr,
):
    """读一份分布在 ``[pid, row, v_offsets]`` 处的概率。

    ``FROM_LOGITS`` 时 ``Ptr`` 是 logits ``[bs, rows, n]``，第 ``j`` 列对应词表下标
    ``vocab_start + j``；概率现算为 ``exp(l / T - lse)``，词表区间外和
    ``row >= num_rows`` 都视作 0。否则 ``Ptr`` 就是稠密概率。
    """
    if FROM_LOGITS:
        local = v_offsets - vocab_start
        in_range = v_mask & (local >= 0) & (local < vocab_len) & (row < num_rows)
        safe_row = tl.minimum(row, num_rows - 1)
        l_val = tl.load(
            Ptr + pid.to(tl.int64) * stride_b + safe_row * stride_s + local * stride_v,
            mask=in_range,
            other=0.0,
        ).to(tl.float32)
        lse = tl.load(Lse + pid * stride_lse_b + safe_row * stride_lse_s)
        if PER_REQ_TEMP:
            temp = tl.load(Temperatures + pid).to(tl.float32)
        else:
            temp = temperature
        return tl.where(in_range, tl.exp(l_val / temp - lse), 0.0)
    else:
        return tl.load(
            Ptr + pid * stride_b + row * stride_s + v_offsets * stride_v,
            mask=v_mask,
            other=0.0,
        )


@triton.jit
def speculative_sampling_classic_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    # P_FROM_LOGITS 时 TargetProbs 是 target logits，以下描述其 lse / 温度
    TargetLse,
    stride_tlse_b,
    stride_tlse_s,
    TargetTemps,
    target_temperature,
    # Q_FROM_LOGITS 时 DraftProbs 是草稿 logits，以下描述其 lse / 温度 / 词表区间
    DraftLse,
    stride_dlse_b,
    stride_dlse_s,
    DraftTemps,
    draft_temperature,
    q_vocab_start,
    q_vocab_len,
    num_q_rows,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    P_FROM_LOGITS: tl.constexpr = False,
    P_PER_REQ_TEMP: tl.constexpr = False,
    Q_FROM_LOGITS: tl.constexpr = False,
    Q_PER_REQ_TEMP: tl.constexpr = False,
):
    pid = tl.program_id(0)
    cur_prob_row = 0

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0

    # Verification Loop
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
        tok_valid = draft_token >= 0

        p = _load_prob(
            TargetProbs, TargetLse, TargetTemps, pid, cur_prob_row,
            draft_token, tok_valid, stride_tp_b, stride_tp_s, stride_tp_v,
            stride_tlse_b, stride_tlse_s, target_temperature,
            0, VOCAB_SIZE, NUM_SLOTS, P_FROM_LOGITS, P_PER_REQ_TEMP,
        )  # fmt: skip
        q = _load_prob(
            DraftProbs, DraftLse, DraftTemps, pid, cur_prob_row,
            draft_token, tok_valid, stride_dp_b, stride_dp_s, stride_dp_v,
            stride_dlse_b, stride_dlse_s, draft_temperature,
            q_vocab_start, q_vocab_len, num_q_rows, Q_FROM_LOGITS, Q_PER_REQ_TEMP,
        )  # fmt: skip

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        if coin * q < p:
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)

            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = continue_verifying
    coin_final = tl.load(UniformSamplesFinal + pid)
    norm_sum = 0.0

    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # reads q; on rejection cur_prob_row <= num_steps - 1.

    # Pass 1: Sum
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE

        p_val = _load_prob(
            TargetProbs, TargetLse, TargetTemps, pid, cur_prob_row,
            v_offsets, mask, stride_tp_b, stride_tp_s, stride_tp_v,
            stride_tlse_b, stride_tlse_s, target_temperature,
            0, VOCAB_SIZE, NUM_SLOTS, P_FROM_LOGITS, P_PER_REQ_TEMP,
        )  # fmt: skip

        if all_drafts_accepted:
            val = p_val
        else:
            q_val = _load_prob(
                DraftProbs, DraftLse, DraftTemps, pid, cur_prob_row,
                v_offsets, mask, stride_dp_b, stride_dp_s, stride_dp_v,
                stride_dlse_b, stride_dlse_s, draft_temperature,
                q_vocab_start, q_vocab_len, num_q_rows, Q_FROM_LOGITS, Q_PER_REQ_TEMP,
            )  # fmt: skip
            diff = p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)

        norm_sum += tl.sum(val)

    # Pass 2: CDF. Degenerate residual (norm_sum == 0, i.e. p == q everywhere on
    # rejection) leaves the cumsum at 0 <= target_u, so final_token falls back to
    # VOCAB_SIZE - 1; acceptable since this case is numerically near-impossible.
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_val = _load_prob(
                TargetProbs, TargetLse, TargetTemps, pid, cur_prob_row,
                v_offsets, mask, stride_tp_b, stride_tp_s, stride_tp_v,
                stride_tlse_b, stride_tlse_s, target_temperature,
                0, VOCAB_SIZE, NUM_SLOTS, P_FROM_LOGITS, P_PER_REQ_TEMP,
            )  # fmt: skip

            if all_drafts_accepted:
                val = p_val
            else:
                q_val = _load_prob(
                    DraftProbs, DraftLse, DraftTemps, pid, cur_prob_row,
                    v_offsets, mask, stride_dp_b, stride_dp_s, stride_dp_v,
                    stride_dlse_b, stride_dlse_s, draft_temperature,
                    q_vocab_start, q_vocab_len, num_q_rows, Q_FROM_LOGITS, Q_PER_REQ_TEMP,
                )  # fmt: skip
                diff = p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = total_cumsum > target_u
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    tl.store(Predicts + last_accepted_global_idx, final_token)


_BLOCK_V = 4096
TemperatureArg = Union[float, torch.Tensor]


def _as_rows_last_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


def _temperature_args(temperature: TemperatureArg, bs: int, dummy: torch.Tensor):
    """标量温度走 constexpr 分支；张量温度视作 per-request ``[bs]``。"""
    if isinstance(temperature, torch.Tensor):
        temps = temperature.reshape(bs).to(torch.float32).contiguous()
        return temps, 1.0, True
    return dummy, float(temperature), False


def row_logsumexp(logits: torch.Tensor, temperature: TemperatureArg = 1.0):
    """``logsumexp(logits / T, -1)``，``logits`` 为 ``[bs, rows, n]``，直接读原 dtype
    不落 fp32 副本。``temperature`` 为标量或 per-request ``[bs]``。返回 ``[bs, rows]``。"""
    logits = _as_rows_last_contiguous(logits)
    bs, rows, n_cols = logits.shape
    lse = torch.empty((bs, rows), dtype=torch.float32, device=logits.device)
    temps, temp_scalar, per_req = _temperature_args(temperature, bs, lse)
    _row_lse_kernel[(bs * rows,)](
        logits,
        lse,
        temps,
        logits.stride(0),
        logits.stride(1),
        rows,
        n_cols,
        temp_scalar,
        BLOCK=_BLOCK_V,
        PER_REQ_TEMP=per_req,
    )
    return lse


def chain_speculative_sampling_fused_triton(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: Optional[torch.Tensor] = None,
    target_logits: Optional[torch.Tensor] = None,
    target_temperature: TemperatureArg = 1.0,
    draft_probs: Optional[torch.Tensor] = None,
    draft_logits: Optional[torch.Tensor] = None,
    draft_vocab_start: int = 0,
    draft_temperature: TemperatureArg = 1.0,
):
    """chain 拒绝采样；p、q 各自既可给稠密概率，也可给 logits 由 kernel 现算。

    - ``target_logits`` ``[bs, num_slots, vocab]``：``p = softmax(l / T_target)``，
      省掉稠密 fp32 target probs 的写出和读回。仅适用于不带 top-k/top-p 截断的分布。
    - ``draft_logits`` ``[bs, rows, draft_vocab]``：
      ``q[b, r, draft_vocab_start + j] = softmax(l[b, r] / T_draft)[j]``，区间外为 0；
      ``rows`` 不足 ``num_slots - 1`` 时缺的行按 q=0 处理。
    温度可为标量或 per-request ``[bs]`` 张量。
    """
    if (target_probs is None) == (target_logits is None):
        raise ValueError("exactly one of target_probs / target_logits is required")
    if (draft_probs is None) == (draft_logits is None):
        raise ValueError("exactly one of draft_probs / draft_logits is required")
    batch_size, num_slots = candidates.shape
    p_src = target_probs if target_logits is None else _as_rows_last_contiguous(target_logits)
    q_src = draft_probs if draft_logits is None else _as_rows_last_contiguous(draft_logits)
    vocab_size = p_src.shape[-1]

    p_from_logits = target_logits is not None
    if p_from_logits:
        t_lse = row_logsumexp(p_src, target_temperature)
        t_temps, t_temp, t_per_req = _temperature_args(target_temperature, batch_size, t_lse)
    else:
        t_lse, t_temps, t_temp, t_per_req = p_src, p_src, 1.0, False

    q_from_logits = draft_logits is not None
    if q_from_logits:
        num_q_rows, draft_vocab = q_src.shape[1], q_src.shape[2]
        q_vocab_len = max(
            0, min(draft_vocab_start + draft_vocab, vocab_size) - draft_vocab_start
        )
        d_lse = row_logsumexp(q_src, draft_temperature)
        d_temps, d_temp, d_per_req = _temperature_args(draft_temperature, batch_size, d_lse)
    else:
        num_q_rows, q_vocab_len = 0, 0
        d_lse, d_temps, d_temp, d_per_req = q_src, q_src, 1.0, False

    speculative_sampling_classic_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        p_src,
        q_src,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        p_src.stride(0),
        p_src.stride(1),
        p_src.stride(2),
        q_src.stride(0),
        q_src.stride(1),
        q_src.stride(2),
        t_lse,
        t_lse.stride(0) if p_from_logits else 0,
        t_lse.stride(1) if p_from_logits else 0,
        t_temps,
        t_temp,
        d_lse,
        d_lse.stride(0) if q_from_logits else 0,
        d_lse.stride(1) if q_from_logits else 0,
        d_temps,
        d_temp,
        int(draft_vocab_start),
        int(q_vocab_len),
        int(num_q_rows),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=_BLOCK_V,
        P_FROM_LOGITS=p_from_logits,
        P_PER_REQ_TEMP=t_per_req,
        Q_FROM_LOGITS=q_from_logits,
        Q_PER_REQ_TEMP=d_per_req,
    )


def chain_speculative_sampling_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,  # not used in chain verification
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,  # not used
):
    chain_speculative_sampling_fused_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
    )
