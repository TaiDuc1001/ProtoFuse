import json
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
    log_experiment_metrics,
    logger,
    merge_configs,
    parse_override_arguments,
    process_parsed_args,
    setup_logging,
)
from src.pipelines.protofuse import ProtoFusePipeline


ARG_SCHEMA = DEFAULT_ARG_SCHEMA
ALPHA_STEP = 0.01


class ProtoFuseDiagnosisPipeline(ProtoFusePipeline):
    METHOD_NAME = "ProtoFuse Diagnosis"
    DEFAULT_OUTPUT_DIR = "outputs/protofuse_diagnosis"

    def _label_remap(self, train_labels, val_labels):
        task_classes = sorted(set(train_labels.tolist()))
        remap = {class_idx: idx for idx, class_idx in enumerate(task_classes)}

        missing = sorted(set(val_labels.tolist()) - set(remap.keys()))
        if missing:
            raise ValueError(f"Validation labels not present in train split: {missing[:10]}")

        remapped_train = torch.tensor(
            [remap[label.item()] for label in train_labels],
            dtype=torch.long,
        )
        remapped_val = torch.tensor(
            [remap[label.item()] for label in val_labels],
            dtype=torch.long,
        )
        return remapped_train, remapped_val, len(task_classes)

    def _shots_per_class(self, labels, num_classes):
        counts = torch.bincount(labels, minlength=num_classes)
        return int(counts.min().item())

    def _predict_for_alpha(self, alpha, T, V, eval_features, num_classes, one_shot_mode):
        eval_norm = F.normalize(eval_features.to(self.device), dim=-1)

        if one_shot_mode:
            tau = 0.05
            logits_text = eval_norm @ T.T
            logits_visual = eval_norm @ V.T
            probs_visual = F.softmax(logits_visual / tau, dim=-1)
            entropy = -(probs_visual * torch.log(probs_visual + 1e-8)).sum(dim=-1)
            confidence = 1.0 - (entropy / math.log(num_classes)).clamp(0.0, 1.0)
            alpha_x = alpha * (0.5 + 0.5 * confidence)
            logits = (1 - alpha_x).unsqueeze(-1) * logits_text + alpha_x.unsqueeze(-1) * logits_visual
        else:
            prototypes = F.normalize((1 - alpha) * T + alpha * V, dim=-1)
            logits = eval_norm @ prototypes.T

        return logits.argmax(dim=-1).cpu().tolist()

    def _sweep_alpha(self, T, V, eval_features, eval_labels, num_classes, one_shot_mode):
        labels = eval_labels.tolist()
        results = []

        with torch.no_grad():
            for alpha_idx in range(101):
                alpha = alpha_idx * ALPHA_STEP
                preds = self._predict_for_alpha(
                    alpha,
                    T,
                    V,
                    eval_features,
                    num_classes,
                    one_shot_mode,
                )
                metrics = {key: float(value) for key, value in compute_metrics(labels, preds).items()}
                metrics["alpha"] = alpha
                results.append(metrics)

        return results

    def _train_epochs(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before diagnosis.")

        logger.info("Extracting cached CLIP features with the regular ProtoFuse pipeline")
        train_features, train_labels = self._cached_train_features()
        val_features, val_labels = self._cached_val_features()

        remapped_train, remapped_val, num_classes = self._label_remap(train_labels, val_labels)
        shots_per_class = self._shots_per_class(remapped_train, num_classes)
        one_shot_mode = shots_per_class < 2

        logger.info(
            f"Running alpha diagnosis: alpha=0.00..1.00, step={ALPHA_STEP:.2f}, "
            f"classes={num_classes}, shots/class={shots_per_class}"
        )

        V = self.trainer.build_visual_centroids(train_features, remapped_train, num_classes)
        T = self.trainer.text_prototypes

        sweep = self._sweep_alpha(
            T,
            V,
            val_features,
            remapped_val,
            num_classes,
            one_shot_mode,
        )
        best = max(sweep, key=lambda row: (row.get("accuracy", 0.0), row.get("mca", 0.0)))

        best_alpha = float(best["alpha"])
        self.best_val_acc = float(best["accuracy"])
        self.trainer.best_alpha = best_alpha
        self.trainer.best_gamma = 0.0
        self.trainer.fused_prototypes = F.normalize((1 - best_alpha) * T + best_alpha * V, dim=-1)

        summary = {
            "alpha_step": ALPHA_STEP,
            "best_alpha": best_alpha,
            "best_accuracy": float(best["accuracy"]),
            "best_mca": float(best.get("mca", 0.0)),
            "best_metrics": best,
            "sweep": sweep,
        }
        self.metrics.append(summary)

        out_path = os.path.join(self.run_dir, "alpha_sweep.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        top_rows = []
        for row in sorted(sweep, key=lambda item: item.get("accuracy", 0.0), reverse=True)[:10]:
            top_rows.append({
                "alpha": f"{row['alpha']:.2f}",
                "accuracy": f"{row['accuracy']:.2f}%",
                "mca": f"{row.get('mca', 0.0):.2f}%",
            })

        logger.comparison_table(
            top_rows,
            columns=["alpha", "accuracy", "mca"],
            title="Top Alpha Values",
        )
        logger.info(
            f"Best alpha={best_alpha:.2f}, accuracy={best['accuracy']:.2f}%, "
            f"MCA={best.get('mca', 0.0):.2f}%"
        )
        logger.info(f"Alpha sweep saved to: {out_path}")
        log_experiment_metrics(best, title=self._metrics_title())


def parse_args():
    parsed, unknown = create_argument_parser("Run ProtoFuse alpha diagnosis", ARG_SCHEMA).parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, process_parsed_args(parsed, ARG_SCHEMA, overrides)


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, "debug", True), getattr(args, "disable_coloring", True))
    config = merge_configs(load_config_file(args.config), overrides)
    pipeline = ProtoFuseDiagnosisPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
