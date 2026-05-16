import math
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    DEFAULT_ARG_SCHEMA,
    compute_metrics,
    create_argument_parser,
    load_config_file,
    logger,
    merge_configs,
    parse_override_arguments,
    process_parsed_args,
    setup_logging,
)
from src.pipelines.protofuse import ProtoFusePipeline


ARG_SCHEMA = DEFAULT_ARG_SCHEMA


class ProtoFuseAlphaSweepPipeline(ProtoFusePipeline):
    METHOD_NAME = "ProtoFuse Alpha Sweep"
    DEFAULT_OUTPUT_DIR = "outputs/protofuse_alpha"

    def _float(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.detach().cpu().item()
        return float(value)

    def _label_remap(self, train_labels, eval_labels):
        task_classes = sorted(set(train_labels.tolist()))
        remap = {class_idx: idx for idx, class_idx in enumerate(task_classes)}
        missing = sorted(set(eval_labels.tolist()) - set(remap.keys()))
        if missing:
            raise ValueError(f"Eval labels not present in train split: {missing[:10]}")
        remapped_train = torch.tensor([remap[label.item()] for label in train_labels], dtype=torch.long)
        remapped_eval = torch.tensor([remap[label.item()] for label in eval_labels], dtype=torch.long)
        return remapped_train, remapped_eval, len(task_classes)

    def _shots_per_class(self, labels, num_classes):
        counts = torch.bincount(labels, minlength=num_classes)
        return int(counts.min().item())

    def _logits_for_alpha(self, alpha, T, V, features, num_classes, one_shot_mode):
        features = F.normalize(features.to(self.device).float(), dim=-1)
        if one_shot_mode:
            logits_text = features @ T.T
            logits_visual = features @ V.T
            probs_visual = F.softmax(logits_visual / 0.05, dim=-1)
            entropy = -(probs_visual * torch.log(probs_visual + 1e-8)).sum(dim=-1)
            entropy_denom = max(math.log(num_classes), 1e-12)
            confidence = 1.0 - (entropy / entropy_denom).clamp(0.0, 1.0)
            alpha_x = alpha * (0.5 + 0.5 * confidence)
            return (1 - alpha_x).unsqueeze(-1) * logits_text + alpha_x.unsqueeze(-1) * logits_visual

        prototypes = F.normalize((1 - alpha) * T + alpha * V, dim=-1)
        return features @ prototypes.T

    def _alpha_table_title(self):
        return f"{self.config.data.dataset_name} x {self.kshot}-shot x seed {self.seed}"

    def _sweep_alpha(self, T, V, eval_features, eval_labels, num_classes, one_shot_mode):
        labels = eval_labels.to(self.device)
        labels_list = eval_labels.tolist()
        rows = []

        for alpha in self.trainer.alphas:
            alpha_value = self._float(alpha)
            logits = self._logits_for_alpha(alpha_value, T, V, eval_features, num_classes, one_shot_mode)
            preds = logits.argmax(dim=-1)
            metrics = {key: float(value) for key, value in compute_metrics(labels_list, preds.cpu().tolist()).items()}
            metrics["alpha"] = alpha_value
            metrics["correct"] = int(preds.eq(labels).sum().item())
            metrics["total"] = int(labels.numel())
            rows.append(metrics)

        return rows

    def _print_alpha_table(self, sweep):
        rows = [
            {
                "Alpha": f"{row['alpha']:.2f}",
                "Performance": f"{row['accuracy']:.2f}%",
            }
            for row in sweep
        ]
        logger.comparison_table(
            rows=rows,
            columns=["Alpha", "Performance"],
            title=self._alpha_table_title(),
        )

    def _train_epochs(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before alpha sweep.")

        train_features, train_labels = self._cached_train_features()
        eval_features, eval_labels = self._cached_val_features()
        remapped_train, remapped_eval, num_classes = self._label_remap(train_labels, eval_labels)
        shots_per_class = self._shots_per_class(remapped_train, num_classes)
        one_shot_mode = shots_per_class < 2

        logger.info(
            f"Running alpha sweep: steps={len(self.trainer.alphas)}, "
            f"classes={num_classes}, shots/class={shots_per_class}"
        )

        with torch.no_grad():
            V = self.trainer.build_visual_centroids(train_features, remapped_train, num_classes)
            T = self.trainer.text_prototypes
            sweep = self._sweep_alpha(T, V, eval_features, remapped_eval, num_classes, one_shot_mode)

        best = max(sweep, key=lambda row: (row.get("accuracy", 0.0), row.get("mca", 0.0)))
        best_alpha = float(best["alpha"])
        self.best_val_acc = float(best["accuracy"])
        self.trainer.best_alpha = best_alpha
        self.trainer.fused_prototypes = F.normalize((1 - best_alpha) * T + best_alpha * V, dim=-1)

        summary = {
            "alpha_steps": len(sweep),
            "best_alpha": best_alpha,
            "best_accuracy": float(best["accuracy"]),
            "best_mca": float(best.get("mca", 0.0)),
            "one_shot_mode": bool(one_shot_mode),
            "sweep": sweep,
        }
        self.metrics.append(summary)

        self._print_alpha_table(sweep)
        logger.info(
            f"Best alpha={best_alpha:.2f}, performance={best['accuracy']:.2f}%, "
            f"MCA={best.get('mca', 0.0):.2f}%"
        )


def parse_args():
    parsed, unknown = create_argument_parser("Run ProtoFuse alpha sweep", ARG_SCHEMA).parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, process_parsed_args(parsed, ARG_SCHEMA, overrides)


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, "debug", True), getattr(args, "disable_coloring", True))
    config = merge_configs(load_config_file(args.config), overrides)
    if getattr(args, "output_dir", None) is None:
        config.setdefault("logging", {})
        config["logging"]["output_dir"] = ProtoFuseAlphaSweepPipeline.DEFAULT_OUTPUT_DIR
    pipeline = ProtoFuseAlphaSweepPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
