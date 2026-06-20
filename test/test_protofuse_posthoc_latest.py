import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.protofuse import ProtoFuse


class ProtoFusePosthocLatestTest(unittest.TestCase):
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
            rho=0.75,
        )

        self.assertEqual(selection["rho"], 0.75)
        self.assertIn(
            selection["selected_candidate"],
            {"orig", "qx_fixed_alpha", "qx_recal_alpha"},
        )
        self.assertEqual(
            set(selection["candidate_scores"]),
            {"orig", "qx_fixed_alpha", "qx_recal_alpha"},
        )
        self.assertIn("alpha_init", selection)
        self.assertIn("alpha_final", selection)
        self.assertTrue(torch.isfinite(selection["fused_prototypes"]).all())


if __name__ == "__main__":
    unittest.main()
