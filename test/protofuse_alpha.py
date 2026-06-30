import math
import os
import sys
import json
import logging
import random
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.patheffects as path_effects
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    DEFAULT_ARG_SCHEMA,
    compute_metrics,
    create_argument_parser,
    get_config_value,
    load_config_file,
    logger,
    merge_configs,
    parse_override_arguments,
    process_parsed_args,
    iter_dataset_configs,
    run_dataset_eda,
    set_global_seed,
    setup_logging,
    fast_image_folder,
)
from src.pipelines.protofuse import ProtoFusePipeline


ARG_SCHEMA = DEFAULT_ARG_SCHEMA
DEFAULT_KSHOTS = [1, 2, 4, 8, 16]
DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]
ALPHA_FIGURE_THEME = {
    "name": "viridian",
    "cmap": "viridis",
    "marker": "white",
    "edge": "black",
    "text": "white",
    "stroke": "black",
}


class ProtoFuseAlphaSweepPipeline(ProtoFusePipeline):
    METHOD_NAME = "ProtoFuse Alpha Sweep"
    DEFAULT_OUTPUT_DIR = "outputs/protofuse_alpha"
    ALPHA_BATCH_SIZE = 16

    def _train_only_root(self):
        train_path = os.path.join(self.dataset_root, "train")
        return train_path if os.path.isdir(train_path) else self.dataset_root

    def _load_dataset(self):
        transform = self._build_transforms()
        train_root = self._train_only_root()
        try:
            self.dataset = fast_image_folder(train_root, transform=transform)
        except Exception as exc:
            raise RuntimeError(f"Failed to load train dataset from {train_root}: {exc}")
        if self.run_eda:
            run_dataset_eda(self.dataset, self.eda_dir, sample_limit=512, seed=self.seed)

    def _split_dataset(self):
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before splitting.")

        samples_by_class_idx = {}
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            samples_by_class_idx.setdefault(class_idx, []).append(idx)

        rng = random.Random(self.seed)
        train_indices = []
        unlabeled_indices = []
        for class_idx in sorted(samples_by_class_idx):
            class_samples = sorted(samples_by_class_idx[class_idx])
            rng.shuffle(class_samples)

            if self.kshot > 0:
                labeled_part = class_samples[:self.kshot]
                leftover_part = class_samples[self.kshot:]
            else:
                labeled_part = class_samples
                leftover_part = []
            train_indices.extend(labeled_part)
            unlabeled_indices.extend(leftover_part)

        self.val_indices = []
        self.train_indices = train_indices
        self.labeled_indices = list(train_indices)
        self.unlabeled_indices = unlabeled_indices
        self.val_loader = None
        self.classnames = list(self.dataset.classes)

        stats = {
            "total_images": len(self.dataset),
            "val_count": 0,
            "train_count": len(self.train_indices),
            "labeled_count": len(self.train_indices),
            "unlabeled_count": len(self.unlabeled_indices),
            "train_pool_size": len(self.train_indices) + len(self.unlabeled_indices),
        }
        trainer_cfg = self._build_trainer_config(stats, 0.0)
        with open(self.config_path, "w") as f:
            json.dump(trainer_cfg.to_dict(), f, indent=4)

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

    def _sweep_alpha_fast(self, T, V, eval_features, eval_labels, num_classes, one_shot_mode):
        eval_features = F.normalize(eval_features.to(self.device).float(), dim=-1)
        labels = eval_labels.to(self.device).long()
        alphas = self.trainer.alphas.float()
        rows = []

        if one_shot_mode:
            logits_text = eval_features @ T.T
            logits_visual = eval_features @ V.T
            probs_visual = F.softmax(logits_visual / 0.05, dim=-1)
            entropy = -(probs_visual * torch.log(probs_visual + 1e-8)).sum(dim=-1)
            entropy_denom = max(math.log(num_classes), 1e-12)
            confidence = 1.0 - (entropy / entropy_denom).clamp(0.0, 1.0)
            confidence_scale = 0.5 + 0.5 * confidence

        for start in range(0, len(alphas), self.ALPHA_BATCH_SIZE):
            alpha_chunk = alphas[start:start + self.ALPHA_BATCH_SIZE]
            if one_shot_mode:
                alpha_x = alpha_chunk.view(-1, 1) * confidence_scale.view(1, -1)
                logits = (
                    (1.0 - alpha_x).unsqueeze(-1) * logits_text.unsqueeze(0)
                    + alpha_x.unsqueeze(-1) * logits_visual.unsqueeze(0)
                )
            else:
                prototypes = F.normalize(
                    (1.0 - alpha_chunk).view(-1, 1, 1) * T.unsqueeze(0)
                    + alpha_chunk.view(-1, 1, 1) * V.unsqueeze(0),
                    dim=-1,
                )
                logits = torch.einsum("nd,acd->anc", eval_features, prototypes)

            preds = logits.argmax(dim=-1)
            accuracies = preds.eq(labels.view(1, -1)).float().mean(dim=1) * 100.0
            for alpha_value, accuracy in zip(alpha_chunk.detach().cpu().tolist(), accuracies.detach().cpu().tolist()):
                rows.append({"alpha": float(alpha_value), "accuracy": float(accuracy)})

        return rows

    def _support_indices(self, kshot, seed):
        samples_by_class_idx = {}
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            samples_by_class_idx.setdefault(class_idx, []).append(idx)

        rng = random.Random(seed)
        train_indices = []
        for class_idx in sorted(samples_by_class_idx):
            class_samples = sorted(samples_by_class_idx[class_idx])
            rng.shuffle(class_samples)
            train_indices.extend(class_samples[:kshot] if kshot > 0 else class_samples)

        return train_indices

    def _alpha_output_for(self, train_features, train_labels, eval_features, eval_labels, kshot, seed):
        remapped_train, remapped_eval, num_classes = self._label_remap(train_labels, eval_labels)
        shots_per_class = self._shots_per_class(remapped_train, num_classes)
        one_shot_mode = shots_per_class < 2

        with torch.no_grad():
            V = self.trainer.build_visual_centroids(train_features, remapped_train, num_classes)
            T = self.trainer.text_prototypes
            sweep = self._sweep_alpha_fast(T, V, eval_features, remapped_eval, num_classes, one_shot_mode)

        best = max(sweep, key=lambda row: row.get("accuracy", 0.0))
        best_alpha = float(best["alpha"])
        return {
            "dataset": self.config.data.dataset_name,
            "split": "train",
            "kshot": int(kshot),
            "seed": int(seed),
            "alpha_best": best_alpha,
            "title": (
                f"{self.config.data.dataset_name} x {self.METHOD_NAME} x "
                f"{int(kshot)}-shot x seed {int(seed)} x alpha_best {best_alpha:.2f}"
            ),
            "rows": sweep,
        }

    def run_dataset_sweep(self, kshots, seeds):
        set_global_seed(self.seed)
        self._prepare_directories()
        self._load_dataset()
        self._split_dataset()
        self._initialize_trainer()

        train_payload = self._full_dataset_clip_features()
        train_features_all = train_payload["image_features"]
        train_labels_all = train_payload["labels"]

        outputs = []
        for kshot in kshots:
            for seed in seeds:
                set_global_seed(seed)
                train_idx = self._support_indices(int(kshot), int(seed))
                train_idx = torch.tensor(train_idx, dtype=torch.long)
                train_features = train_features_all[train_idx].contiguous()
                train_labels = train_labels_all[train_idx].contiguous()

                outputs.append(
                    self._alpha_output_for(
                        train_features,
                        train_labels,
                        train_features,
                        train_labels,
                        int(kshot),
                        int(seed),
                    )
                )
        return outputs

    def run(self):
        set_global_seed(self.seed)
        self._prepare_directories()
        self._load_dataset()
        self._split_dataset()
        self._initialize_trainer()
        self._train_epochs()
        self._finalize()
        return self.alpha_sweep_output

    def _train_epochs(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before alpha sweep.")

        train_features, train_labels = self._cached_train_features()
        eval_features, eval_labels = train_features, train_labels
        remapped_train, remapped_eval, num_classes = self._label_remap(train_labels, eval_labels)
        shots_per_class = self._shots_per_class(remapped_train, num_classes)
        one_shot_mode = shots_per_class < 2

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
            "eval_split": "train",
            "sweep": sweep,
        }
        self.metrics.append(summary)
        self.alpha_sweep_output = {
            "dataset": self.config.data.dataset_name,
            "split": "train",
            "kshot": self.kshot,
            "seed": self.seed,
            "alpha_best": best_alpha,
            "title": (
                f"{self.config.data.dataset_name} x {self.METHOD_NAME} x "
                f"{self.kshot}-shot x seed {self.seed} x alpha_best {best_alpha:.2f}"
            ),
            "rows": [
                {
                    "alpha": float(row["alpha"]),
                    "accuracy": float(row["accuracy"]),
                }
                for row in sweep
            ],
        }


def parse_args():
    parsed, unknown = create_argument_parser("Run ProtoFuse alpha sweep", ARG_SCHEMA).parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, process_parsed_args(parsed, ARG_SCHEMA, overrides)


def safe_filename(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value)).strip("_")


def parse_int_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def alpha_sweep_values(config):
    kshots = parse_int_list(get_config_value(config, "data.kshots", None), DEFAULT_KSHOTS)
    seeds = parse_int_list(get_config_value(config, "data.seeds", None), DEFAULT_SEEDS)
    return kshots, seeds


def alpha_outputs_frame(outputs):
    all_rows = []
    for output in outputs:
        for row in output["rows"]:
            all_rows.append(
                {
                    "dataset": output["dataset"],
                    "kshot": int(output["kshot"]),
                    "seed": int(output["seed"]),
                    "alpha": float(row["alpha"]),
                    "accuracy": float(row["accuracy"]),
                }
            )
    return pd.DataFrame(all_rows)


def aggregate_alpha_outputs(outputs):
    data = alpha_outputs_frame(outputs)
    if data.empty:
        return data, data

    grouped = (
        data.groupby(["kshot", "alpha"], as_index=False)["accuracy"]
        .agg(mean="mean", std="std")
        .sort_values(["kshot", "alpha"])
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return data, grouped


def build_alpha_summary_table(grouped):
    rows = []
    for kshot in sorted(grouped["kshot"].unique()):
        shot_rows = grouped[grouped["kshot"] == kshot]
        row = {"kshot": f"{int(kshot)}-shot"}
        for _, alpha_row in shot_rows.iterrows():
            alpha_key = f"{float(alpha_row['alpha']):.2f}"
            row[alpha_key] = f"{float(alpha_row['mean']):.2f} +/- {float(alpha_row['std']):.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def save_alpha_figure(outputs, output_dir, theme=None):
    data, grouped = aggregate_alpha_outputs(outputs)
    if data.empty:
        return None

    theme = theme or ALPHA_FIGURE_THEME
    dataset_name = str(data["dataset"].iloc[0])
    heatmap = grouped.pivot(index="kshot", columns="alpha", values="mean").sort_index()
    kshots = [int(value) for value in heatmap.index.tolist()]
    alphas = [float(value) for value in heatmap.columns.tolist()]
    values = heatmap.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=theme["cmap"], origin="lower")

    tick_positions = np.linspace(0, len(alphas) - 1, 6).round().astype(int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{alphas[idx]:.1f}" for idx in tick_positions], fontsize=12)
    ax.set_yticks(np.arange(len(kshots)))
    ax.set_yticklabels([f"{kshot}-shot" for kshot in kshots], fontsize=12)
    ax.set_xlabel(r"$\alpha$", fontsize=16)
    ax.set_ylabel("K-shot", fontsize=16)

    for row_idx, kshot in enumerate(kshots):
        best_col = int(np.nanargmax(values[row_idx]))
        best_alpha = alphas[best_col]

        ax.scatter(
            [best_col],
            [row_idx],
            marker="*",
            s=120,
            c=theme["marker"],
            edgecolors=theme["edge"],
            linewidths=0.8,
            zorder=3,
        )

        near_right = best_col >= 0.75 * (len(alphas) - 1)

        ax.annotate(
            rf"$\alpha^*={best_alpha:.2f}$",
            xy=(best_col, row_idx),
            xytext=(-10 if near_right else 10, 0),
            textcoords="offset points",
            color=theme["text"],
            fontsize=10,
            ha="right" if near_right else "left",
            va="center",
            path_effects=[path_effects.withStroke(linewidth=2, foreground=theme["stroke"])],
            zorder=4,
        )

    cbar = fig.colorbar(image, ax=ax, pad=0.015)
    cbar.set_label(r"$\mathrm{Acc}_{\mathrm{cal}}(\alpha)$ (%)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{safe_filename(dataset_name)}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, "debug", True), getattr(args, "disable_coloring", True))
    config = merge_configs(load_config_file(args.config), overrides)
    if getattr(args, "output_dir", None) is None:
        config.setdefault("logging", {})
        config["logging"]["output_dir"] = ProtoFuseAlphaSweepPipeline.DEFAULT_OUTPUT_DIR

    for dataset_config, _ in iter_dataset_configs(config):
        kshots, seeds = alpha_sweep_values(dataset_config)
        dataset_config.setdefault("data", {})
        dataset_config["data"]["kshot"] = int(max(kshots))
        dataset_config["data"]["seed"] = int(seeds[0])
        previous_level = logger._logger.level
        logger._logger.setLevel(logging.WARNING)
        try:
            dataset_outputs = ProtoFuseAlphaSweepPipeline(dataset_config).run_dataset_sweep(kshots, seeds)
        finally:
            logger._logger.setLevel(previous_level)

        data, grouped = aggregate_alpha_outputs(dataset_outputs)
        dataset_name = str(data["dataset"].iloc[0]) if not data.empty else str(dataset_config["data"]["dataset_name"])
        print(f"\n{dataset_name} x {ProtoFuseAlphaSweepPipeline.METHOD_NAME} x seed mean +/- std")
        print(build_alpha_summary_table(grouped).to_string(index=False), flush=True)
        save_alpha_figure(dataset_outputs, config["logging"]["output_dir"])


if __name__ == "__main__":
    main()
