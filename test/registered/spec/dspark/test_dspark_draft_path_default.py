import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _target_checkpoint_bundles_dspark_draft,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    sample_draft_block,
)
from sglang.srt.speculative.dspark_components.dspark_verify import accept_draft_tokens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BUNDLED_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
_PLAIN_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _bundled_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )


def _plain_hf_config() -> SimpleNamespace:
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


def _make_dspark_server_args(
    *, model_path: str, hf_config: SimpleNamespace
) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = model_path
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = None
    server_args.speculative_dspark_block_size = 5
    server_args.model_config = SimpleNamespace(hf_config=hf_config)
    return server_args


class TestTargetCheckpointBundlesDsparkDraft(CustomTestCase):
    def test_bundled_dsv4_config_is_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        self.assertTrue(_target_checkpoint_bundles_dspark_draft(server_args))

    def test_plain_target_config_is_not_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        self.assertFalse(_target_checkpoint_bundles_dspark_draft(server_args))


class TestDsparkDraftPathDefaulting(CustomTestCase):
    def test_bundled_checkpoint_defaults_draft_path_to_model_path(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _handle_dspark(server_args)
        self.assertEqual(server_args.speculative_draft_model_path, _BUNDLED_MODEL_PATH)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_plain_target_without_draft_path_raises(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        with self.assertRaises(ValueError):
            _handle_dspark(server_args)

    def test_explicit_draft_path_is_not_overwritten(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_draft_model_path = "deepseek-ai/some-other-dspark-draft"
        _handle_dspark(server_args)
        self.assertEqual(
            server_args.speculative_draft_model_path,
            "deepseek-ai/some-other-dspark-draft",
        )


class TestDsparkDraftTemperature(CustomTestCase):
    def test_zero_temperature_forces_deterministic_argmax(self):
        logits = torch.tensor([[[1.0, 3.0, 2.0], [4.0, 0.0, 1.0]]])

        class MarkovHead:
            def sample_block(self, base_logits, **kwargs):
                sampler = kwargs["sampler"]
                tokens = torch.stack(
                    [sampler(base_logits[:, step], step) for step in range(2)], dim=1
                )
                return tokens, base_logits

        sampling_info = SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.tensor([[1.0]]),
            top_ks=torch.tensor([[50]]),
        )
        result = sample_draft_block(
            base_logits=logits,
            anchor_tokens=torch.zeros(1, dtype=torch.long),
            draft_hidden=torch.zeros(1, 2, 1),
            sampling_info=sampling_info,
            markov_head=MarkovHead(),
            device=torch.device("cpu"),
            draft_temperature=0.0,
        )

        self.assertTrue(result.deterministic_draft)
        self.assertTrue(bool(result.greedy_mask.all()))
        torch.testing.assert_close(result.draft_tokens, torch.tensor([[1, 0]]))

    def test_deterministic_draft_uses_target_only_verifier(self):
        draft_block = DraftBlockResult(
            draft_tokens=torch.tensor([[1, 2]]),
            corrected_logits=None,
            greedy_mask=torch.ones(1, dtype=torch.bool),
            temperatures=torch.ones(1),
            deterministic_draft=True,
        )
        sampling_info = SimpleNamespace(is_all_greedy=False)
        expected = (torch.tensor([2]), torch.tensor([3]), torch.tensor([0]))

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_verify.AcceptSampling.execute",
            return_value=expected,
        ) as accept_sampling:
            actual = accept_draft_tokens(
                candidates=torch.tensor([[1, 2, 3]]),
                target_logits=torch.zeros(3, 4),
                draft_block=draft_block,
                sampling_info=sampling_info,
                draft_input=SimpleNamespace(),
                gamma=2,
                verify_num_draft_tokens=3,
            )

        self.assertEqual(actual, expected)
        self.assertIsNone(accept_sampling.call_args.kwargs["draft_probs"])


if __name__ == "__main__":
    unittest.main()
