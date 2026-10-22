import torch
import triton
import triton.language as tl


@triton.jit
def _draft_row_lse_kernel(
    Logits,
    Lse,
    stride_l_b,
    stride_l_s,
    num_rows_per_b,
    n_cols,
    inv_temperature,
    BLOCK: tl.constexpr,
):
    """``Lse[b, s] = logsumexp(Logits[b, s, :] * inv_temperature)``，fp32 累加。"""
    row = tl.program_id(0)
    b = row // num_rows_per_b
    s = row % num_rows_per_b
    base = Logits + b * stride_l_b + s * stride_l_s
    run_max = -float("inf")
    run_sum = 0.0
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(base + offs, mask=offs < n_cols, other=-float("inf"))
        x = x.to(tl.float32) * inv_temperature
        new_max = tl.maximum(run_max, tl.max(x, axis=0))
        # 整块 -inf 时 new_max 仍是 -inf，exp(-inf - -inf) 会出 nan，单独置零
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        run_sum = run_sum * tl.exp(run_max - safe_max) + tl.sum(
            tl.exp(x - safe_max), axis=0
        )
        run_max = new_max
    tl.store(Lse + row, run_max + tl.log(run_sum))


@triton.jit
def _load_draft_q(
    DraftProbs,
    DraftLse,
    pid,
    row,
    v_offsets,
    v_mask,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    stride_lse_b,
    stride_lse_s,
    inv_temperature,
    q_vocab_start,
    q_vocab_len,
    num_q_rows,
    Q_FROM_LOGITS: tl.constexpr,
):
    """读 ``q[pid, row, v_offsets]``。

    ``Q_FROM_LOGITS`` 时 ``DraftProbs`` 实际是草稿 logits ``[bs, rows, draft_vocab]``，
    其第 ``j`` 列对应目标词表下标 ``q_vocab_start + j``；q 现算为
    ``exp(l * inv_T - lse)``，词表区间外和 ``row >= num_q_rows`` 都视作 0。
    """
    if Q_FROM_LOGITS:
        local = v_offsets - q_vocab_start
        in_range = v_mask & (local >= 0) & (local < q_vocab_len) & (row < num_q_rows)
        safe_row = tl.minimum(row, num_q_rows - 1)
        l_val = tl.load(
            DraftProbs + pid * stride_dp_b + safe_row * stride_dp_s + local * stride_dp_v,
            mask=in_range,
            other=0.0,
        ).to(tl.float32)
        lse = tl.load(DraftLse + pid * stride_lse_b + safe_row * stride_lse_s)
        return tl.where(in_range, tl.exp(l_val * inv_temperature - lse), 0.0)
    else:
        return tl.load(
            DraftProbs + pid * stride_dp_b + row * stride_dp_s + v_offsets * stride_dp_v,
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
    # Q_FROM_LOGITS 专用（否则忽略）
    DraftLse,
    stride_lse_b,
    stride_lse_s,
    inv_temperature,
    q_vocab_start,
    q_vocab_len,
    num_q_rows,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    Q_FROM_LOGITS: tl.constexpr = False,
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

        offset_prob = (
            (pid * stride_tp_b)
            + (cur_prob_row * stride_tp_s)
            + (draft_token * stride_tp_v)
        )

        p = tl.load(TargetProbs + offset_prob)
        q = _load_draft_q(
            DraftProbs,
            DraftLse,
            pid,
            cur_prob_row,
            draft_token,
            draft_token >= 0,
            stride_dp_b,
            stride_dp_s,
            stride_dp_v,
            stride_lse_b,
            stride_lse_s,
            inv_temperature,
            q_vocab_start,
            q_vocab_len,
            num_q_rows,
            Q_FROM_LOGITS,
        )

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

    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # reads q; on rejection cur_prob_row <= num_steps - 1.

    # Pass 1: Sum
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE

        p_ptr = tp_base_ptr + v_offsets * stride_tp_v
        p_val = tl.load(p_ptr, mask=mask, other=0.0)

        if all_drafts_accepted:
            val = p_val
        else:
            q_val = _load_draft_q(
                DraftProbs,
                DraftLse,
                pid,
                cur_prob_row,
                v_offsets,
                mask,
                stride_dp_b,
                stride_dp_s,
                stride_dp_v,
                stride_lse_b,
                stride_lse_s,
                inv_temperature,
                q_vocab_start,
                q_vocab_len,
                num_q_rows,
                Q_FROM_LOGITS,
            )
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

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                q_val = _load_draft_q(
                    DraftProbs,
                    DraftLse,
                    pid,
                    cur_prob_row,
                    v_offsets,
                    mask,
                    stride_dp_b,
                    stride_dp_s,
                    stride_dp_v,
                    stride_lse_b,
                    stride_lse_s,
                    inv_temperature,
                    q_vocab_start,
                    q_vocab_len,
                    num_q_rows,
                    Q_FROM_LOGITS,
                )
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
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]

    grid = (batch_size,)
    speculative_sampling_classic_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        draft_probs,  # DraftLse，非 Q_FROM_LOGITS 时不读
        0,
        0,
        1.0,
        0,
        0,
        0,
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
        Q_FROM_LOGITS=False,
    )


def draft_row_logsumexp(draft_logits: torch.Tensor, inv_temperature: float):
    """``logsumexp(draft_logits * inv_temperature, -1)``，直接读 bf16/fp16 logits，
    不在显存里落一份 fp32 副本。返回 ``[bs, rows]`` fp32。"""
    bs, rows, n_cols = draft_logits.shape
    if draft_logits.stride(2) != 1:
        draft_logits = draft_logits.contiguous()
    lse = torch.empty((bs, rows), dtype=torch.float32, device=draft_logits.device)
    _draft_row_lse_kernel[(bs * rows,)](
        draft_logits,
        lse,
        draft_logits.stride(0),
        draft_logits.stride(1),
        rows,
        n_cols,
        float(inv_temperature),
        BLOCK=4096,
    )
    return lse


def chain_speculative_sampling_from_draft_logits_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_logits,
    draft_vocab_start: int,
    draft_inv_temperature: float,
):
    """与 ``chain_speculative_sampling_triton`` 同一套 chain 拒绝采样，但 q 直接从
    草稿 logits 现算：``q[b, r, draft_vocab_start + j] = softmax(draft_logits[b, r] *
    inv_T)[j]``，区间外为 0。

    省掉的是 ``[bs, num_slots, vocab]`` 的 fp32 零张量、一份 fp32 logits 副本、
    softmax 输出以及 scatter；代价只是一个 ``[bs, rows]`` 的 logsumexp。
    ``draft_logits`` 是 ``[bs, rows, draft_vocab]``，第 ``r`` 行对应 verify 的
    ``cur_prob_row == r``，``rows`` 不足 ``num_slots - 1`` 时缺的行按 q=0 处理。
    """
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    _, num_q_rows, draft_vocab = draft_logits.shape
    if draft_logits.stride(2) != 1:
        draft_logits = draft_logits.contiguous()
    q_vocab_len = max(0, min(draft_vocab_start + draft_vocab, vocab_size) - draft_vocab_start)
    lse = draft_row_logsumexp(draft_logits, draft_inv_temperature)

    speculative_sampling_classic_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_logits,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_logits.stride(2),
        lse,
        lse.stride(0),
        lse.stride(1),
        float(draft_inv_temperature),
        int(draft_vocab_start),
        int(q_vocab_len),
        int(num_q_rows),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
        Q_FROM_LOGITS=True,
    )
