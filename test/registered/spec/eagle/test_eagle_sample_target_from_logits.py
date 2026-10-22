"""eagle_sample 经典 chain RS：无 top-k/top-p 时 p 由 kernel 从 logits 现算，与先落
稠密 fp32 target probs 的路径在同一份 coin 下 accept 逐位一致（q 走 d2t 映射）。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.spec_utils import build_draft_vocab_map
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-a")

DEVICE = "cuda"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton kernels")
class TestEagleSampleTargetFromLogits(CustomTestCase):
    def test_matches_dense_target_probs(self):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", speculative_use_rejection_sampling=True)
        )
        bs, slots, vocab, dvocab = 32, 4, 6000, 1500
        g = torch.Generator(device=DEVICE).manual_seed(3)
        hot = torch.randperm(vocab, generator=g, device=DEVICE)[:dvocab]
        draft_logits = torch.randn(bs, slots - 1, dvocab, generator=g, device=DEVICE)
        q = torch.softmax(draft_logits * 3, -1)
        cands = torch.zeros(bs, slots, dtype=torch.int64, device=DEVICE)
        cands[:, 1:] = hot[
            torch.multinomial(q.view(-1, dvocab), 1, generator=g).view(bs, -1)
        ]
        logits = torch.randn(bs * slots, vocab, generator=g, device=DEVICE)
        logits.view(bs, slots, vocab)[:, : slots - 1, hot] += draft_logits * 3
        temps = torch.rand(bs, 1, generator=g, device=DEVICE) + 0.5

        def run(need_top_k):
            # top_k 取极大值时 renorm 是恒等变换，但会走稠密 probs 路径
            sampling_info = SimpleNamespace(
                acc_additive_penalties=None, acc_scaling_penalties=None,
                logit_bias=None, is_all_greedy=False, temperatures=temps,
                need_top_k_sampling=need_top_k, need_top_p_sampling=False,
                top_ks=torch.full((bs,), 1 << 30, dtype=torch.int32, device=DEVICE),
            )  # fmt: skip
            batch = SimpleNamespace(
                device=DEVICE, forward_mode=SimpleNamespace(is_idle=lambda: False),
                seq_lens=torch.ones(bs, device=DEVICE), sampling_info=sampling_info,
            )  # fmt: skip
            verify_input = SimpleNamespace(
                draft_token=cands.reshape(-1), draft_token_num=slots,
                max_tree_depth=slots,
                retrieve_index=torch.arange(bs * slots, device=DEVICE).view(bs, slots),
                retrieve_next_token=None, retrieve_next_sibling=None,
                use_eagle3_rejection_sampling=False, draft_probs=q,
                draft_vocab_map=build_draft_vocab_map(hot, vocab), grammar=None,
            )  # fmt: skip
            output = SimpleNamespace(next_token_logits=logits.clone())
            torch.manual_seed(7)  # 同一份 coin
            with patch(
                "sglang.srt.distributed.get_tp_group",
                lambda: SimpleNamespace(world_size=1),
            ), patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                lambda: False,
            ):
                outs = eagle_utils.eagle_sample(verify_input, batch, output)
            return [t.clone() for t in outs]

        fused, dense = run(False), run(True)
        torch.cuda.synchronize()
        for a, b in zip(fused, dense):
            self.assertTrue(torch.equal(a, b), (a, b))
        accept_lens = fused[1]
        self.assertTrue((accept_lens == slots).any() and (accept_lens < slots).any())


if __name__ == "__main__":
    unittest.main()
