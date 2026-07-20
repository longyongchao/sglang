import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.dflash_utils import (
    is_dflash_domino_projector,
    parse_dflash_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(projector_type=None, **dflash_fields):
    dflash_config = {
        "block_size": 16,
        "target_layer_ids": [0],
        "projector_type": projector_type,
        **dflash_fields,
    }
    return SimpleNamespace(
        hidden_size=8,
        num_hidden_layers=1,
        num_target_layers=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
        dflash_config=dflash_config,
    )


class _FakeLayer(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))


class TestDFlashDominoConfig(CustomTestCase):
    def test_projector_is_exactly_domino(self):
        self.assertTrue(is_dflash_domino_projector("domino"))
        for projector_type in (None, "dspark", "prism", "causal_v5"):
            self.assertFalse(is_dflash_domino_projector(projector_type))

    def test_parse_vanilla_and_dspark_without_domino_fields(self):
        for projector_type in (None, "dspark"):
            parsed = parse_dflash_draft_config(
                draft_hf_config=_config(projector_type)
            )
            self.assertEqual(parsed.projector_type, projector_type)
            self.assertIsNone(parsed.gru_hidden_dim)
            self.assertIsNone(parsed.emb_dim)

    def test_parse_domino_nested_fields(self):
        parsed = parse_dflash_draft_config(
            draft_hf_config=_config(
                "domino",
                pure_draft_prefix_len=1,
                gru_hidden_dim=1024,
                emb_dim=256,
            )
        )
        self.assertEqual(parsed.pure_draft_prefix_len, 1)
        self.assertEqual(parsed.gru_hidden_dim, 1024)
        self.assertEqual(parsed.emb_dim, 256)

    def test_domino_requires_dimensions_and_prefix(self):
        with self.assertRaises(ValueError):
            parse_dflash_draft_config(
                draft_hf_config=_config("domino", pure_draft_prefix_len=1)
            )
        with self.assertRaises(NotImplementedError):
            parse_dflash_draft_config(
                draft_hf_config=_config(
                    "domino", gru_hidden_dim=16, emb_dim=8
                )
            )

    def test_model_allocates_domino_modules_only_for_domino(self):
        with patch("sglang.srt.models.dflash.DFlashDecoderLayer", _FakeLayer):
            vanilla = DFlashDraftModel(_config())
            domino = DFlashDraftModel(
                _config(
                    "domino",
                    pure_draft_prefix_len=1,
                    gru_hidden_dim=16,
                    emb_dim=8,
                )
            )

        self.assertFalse(hasattr(vanilla, "prefix_gru"))
        self.assertFalse(hasattr(vanilla, "embed_proj"))
        self.assertTrue(hasattr(domino, "prefix_gru"))
        self.assertTrue(hasattr(domino, "embed_proj"))
        self.assertIn("prefix_gru.weight_ih_l0", domino.state_dict())
        self.assertIn("embed_proj.0.weight", domino.state_dict())
        self.assertIn("embed_proj.2.weight", domino.state_dict())


if __name__ == "__main__":
    unittest.main()
