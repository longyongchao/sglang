"""Target-verify logits: the LM head GEMM writes fp32 straight into the graph
logits buffer instead of producing half-precision logits and cast-copying."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

DEVICE = "cuda"


class LMHeadStub(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(vocab, hidden, dtype=torch.bfloat16, device=DEVICE),
            requires_grad=False,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestVerifyLogitsFp32Gemm(CustomTestCase):
    VOCAB, PADDED_VOCAB, HIDDEN, ROWS = 1000, 1024, 256, 24

    def _processor(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        get_global_server_args().enable_dp_lm_head = False
        cfg = SimpleNamespace(vocab_size=self.VOCAB, final_logit_softcapping=None)
        return LogitsProcessor(cfg, skip_all_gather=True, logit_scale=None)

    def _run(self, forward_mode):
        torch.manual_seed(0)
        processor = self._processor()
        head = LMHeadStub(self.PADDED_VOCAB, self.HIDDEN)
        hidden = torch.randn(self.ROWS, self.HIDDEN, dtype=torch.bfloat16, device=DEVICE)
        buffer = torch.full((self.ROWS, self.VOCAB), float("nan"), device=DEVICE)
        meta = SimpleNamespace(
            next_token_logits_buffer=buffer, forward_mode=forward_mode
        )
        with patch.object(
            processor, "_compute_lm_head", wraps=processor._compute_lm_head
        ) as slow_path, torch.inference_mode():
            logits = processor._get_logits(hidden, head, meta)
        return logits, buffer, hidden, head, slow_path.call_count

    def test_verify_writes_fp32_gemm_into_buffer(self):
        logits, buffer, hidden, head, slow_calls = self._run(ForwardMode.TARGET_VERIFY)
        self.assertEqual(slow_calls, 0)
        self.assertIs(logits, buffer)
        reference = hidden.float() @ head.weight[: self.VOCAB].float().T
        torch.testing.assert_close(logits, reference, atol=1e-3, rtol=1e-4)

    def test_decode_keeps_half_precision_copy_path(self):
        logits, buffer, hidden, head, slow_calls = self._run(ForwardMode.DECODE)
        self.assertEqual(slow_calls, 1)
        self.assertIs(logits, buffer)
        reference = (hidden @ head.weight.T)[:, : self.VOCAB].float()
        torch.testing.assert_close(logits, reference, atol=0, rtol=0)

    def test_paths_agree_up_to_half_precision_rounding(self):
        fast = self._run(ForwardMode.TARGET_VERIFY)[0]
        slow = self._run(ForwardMode.DECODE)[0]
        # The only difference is the bf16 rounding of the slow path's output.
        torch.testing.assert_close(fast.bfloat16().float(), slow, atol=0.25, rtol=0)
        self.assertLessEqual(
            float(((fast - slow).abs() / slow.abs().clamp_min(1.0)).max()), 2**-7
        )


if __name__ == "__main__":
    unittest.main()
