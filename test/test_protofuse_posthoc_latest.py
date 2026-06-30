import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.protofuse import ProtoFuse


class CountingProtoFuse(ProtoFuse):
    hopc_calls = 0
    fallback_calls = 0

    def hopc_alpha(self, *args, **kwargs):
        type(self).hopc_calls += 1
        return super().hopc_alpha(*args, **kwargs)

    def select_fallback_candidate(self, *args, **kwargs):
        type(self).fallback_calls += 1
        return super().select_fallback_candidate(*args, **kwargs)


class ProtoFusePosthocLatestTest(unittest.TestCase):
    def setUp(self):
        CountingProtoFuse.hopc_calls = 0
        CountingProtoFuse.fallback_calls = 0

    def test_one_shot_reuses_centroid_mix_alpha_after_query_aggregation(self):
        text = F.normalize(torch.eye(3), dim=-1)
        support = text.clone()
        labels = torch.arange(3)
        query = F.normalize(
            torch.tensor([
                [0.9, 0.1, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.1, 0.9],
            ]),
            dim=-1,
        )

        selection = CountingProtoFuse.posthoc_fuse(
            text,
            support,
            labels,
            device="cpu",
            query_features=query,
        )

        self.assertEqual(CountingProtoFuse.hopc_calls, 1)
        self.assertEqual(CountingProtoFuse.fallback_calls, 0)
        self.assertEqual(selection["alpha"], selection["alpha_init"])
        self.assertEqual(selection["alpha_final"], selection["alpha_init"])
        self.assertEqual(selection["selected_candidate"], "sqs_adversarial_fixed")

    def test_one_shot_trainer_path_also_uses_a_single_alpha(self):
        text = F.normalize(torch.eye(3), dim=-1)
        support = text.clone()
        labels = torch.arange(3)
        query = F.normalize(text + 0.05, dim=-1)
        trainer = CountingProtoFuse.from_precomputed(text, device="cpu")

        metrics = trainer.fuse_and_evaluate(
            support,
            labels,
            query,
            labels,
            num_classes=3,
        )

        self.assertEqual(CountingProtoFuse.hopc_calls, 1)
        self.assertEqual(CountingProtoFuse.fallback_calls, 0)
        self.assertEqual(metrics["alpha"], metrics["alpha_init"])
        self.assertEqual(metrics["alpha_final"], metrics["alpha_init"])
        self.assertEqual(metrics["selected_candidate"], "sqs_adversarial_fixed")

    def test_query_features_enable_latest_qx_fallback(self):
        text = F.normalize(torch.eye(3), dim=-1)
        support = text.repeat_interleave(2, dim=0)
        labels = torch.arange(3).repeat_interleave(2)
        query = F.normalize(
            torch.tensor([
                [0.9, 0.1, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.1, 0.9],
            ]),
            dim=-1,
        )

        selection = ProtoFuse.posthoc_fuse(
            text,
            support,
            labels,
            device="cpu",
            query_features=query,
        )

        import math
        self.assertAlmostEqual(selection["rho"], 0.50 / math.sqrt(2))
        self.assertEqual(selection["selected_candidate"], "sqs_adversarial_fixed")
        self.assertEqual(
            set(selection["candidate_scores"]),
            {"sqs_adversarial_fixed"},
        )
        self.assertIn("alpha_init", selection)
        self.assertIn("alpha_final", selection)
        self.assertTrue(torch.isfinite(selection["fused_prototypes"]).all())

    def test_multi_shot_still_recalibrates_and_selects_fallback(self):
        text = F.normalize(torch.eye(3), dim=-1)
        support = text.repeat_interleave(2, dim=0)
        labels = torch.arange(3).repeat_interleave(2)
        query = text.clone()

        CountingProtoFuse.posthoc_fuse(
            text,
            support,
            labels,
            device="cpu",
            query_features=query,
        )

        self.assertEqual(CountingProtoFuse.hopc_calls, 1)
        self.assertEqual(CountingProtoFuse.fallback_calls, 0)


if __name__ == "__main__":
    unittest.main()
