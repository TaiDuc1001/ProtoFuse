import logging
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    DEFAULT_ARG_SCHEMA,
    create_argument_parser,
    get_config_value,
    iter_dataset_configs,
    load_config_file,
    logger,
    merge_configs,
    parse_override_arguments,
    process_parsed_args,
    set_global_seed,
    setup_logging,
)
from src.pipelines.protofuse import ProtoFusePipeline


ARG_SCHEMA = DEFAULT_ARG_SCHEMA
DEFAULT_KSHOTS = [1, 2, 4, 8, 16]
DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]


class ProtoFuseOracleAlphaPipeline(ProtoFusePipeline):
    METHOD_NAME = "ProtoFuse Oracle Alpha"
    DEFAULT_OUTPUT_DIR = "outputs/protofuse_oracle_alpha"
    ALPHA_BATCH_SIZE = 16

    def _float(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.detach().cpu().item()
        return float(value)

    def _split_indices_for(self, kshot, seed):
        samples_by_class_idx = defaultdict(list)
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            samples_by_class_idx[class_idx].append(idx)

        rng = random.Random(seed)
        train_indices = []
        val_indices = []

        for class_idx in sorted(samples_by_class_idx):
            class_samples = sorted(samples_by_class_idx[class_idx])
            rng.shuffle(class_samples)

            if self.val_fraction is None:
                train_candidates = class_samples
            else:
                val_count = int(math.floor(len(class_samples) * self.val_fraction))
                if self.val_fraction > 0 and val_count == 0 and class_samples:
                    val_count = 1
                val_indices.extend(class_samples[:val_count])
                train_candidates = class_samples[val_count:]

            if kshot > 0:
                train_indices.extend(train_candidates[:kshot])
            else:
                train_indices.extend(train_candidates)

        return train_indices, val_indices

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

    def _oracle_output_for(self, train_features, train_labels, eval_features, eval_labels, kshot, seed):
        remapped_train, remapped_eval, num_classes = self._label_remap(train_labels, eval_labels)
        shots_per_class = self._shots_per_class(remapped_train, num_classes)
        one_shot_mode = shots_per_class < 2

        with torch.inference_mode():
            train_features = train_features.to(self.device).float()
            V = self.trainer.build_visual_centroids(train_features, remapped_train.to(self.device), num_classes)
            T = self.trainer.text_prototypes
            sweep = self._sweep_alpha_fast(T, V, eval_features, remapped_eval, num_classes, one_shot_mode)

        best = max(sweep, key=lambda row: row["accuracy"])
        return {
            "dataset": self.config.data.dataset_name,
            "kshot": int(kshot),
            "seed": int(seed),
            "oracle_alpha": float(best["alpha"]),
            "oracle_accuracy": float(best["accuracy"]),
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

        test_payload = None
        if self.val_fraction is None:
            test_features_all, test_labels_all = self._cached_test_features()
            test_payload = {
                "image_features": test_features_all,
                "labels": test_labels_all,
            }

        outputs = []
        for kshot in kshots:
            for seed in seeds:
                set_global_seed(seed)
                train_indices, val_indices = self._split_indices_for(int(kshot), int(seed))
                train_idx = torch.tensor(train_indices, dtype=torch.long)
                train_features = train_features_all[train_idx].contiguous()
                train_labels = train_labels_all[train_idx].contiguous()

                if self.val_fraction is None:
                    eval_features = test_payload["image_features"]
                    eval_labels = test_payload["labels"]
                else:
                    eval_idx = torch.tensor(val_indices, dtype=torch.long)
                    eval_features = train_features_all[eval_idx].contiguous()
                    eval_labels = train_labels_all[eval_idx].contiguous()

                outputs.append(
                    self._oracle_output_for(
                        train_features,
                        train_labels,
                        eval_features,
                        eval_labels,
                        int(kshot),
                        int(seed),
                    )
                )

        return outputs


def parse_args():
    parsed, unknown = create_argument_parser("Run ProtoFuse oracle alpha", ARG_SCHEMA).parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, process_parsed_args(parsed, ARG_SCHEMA, overrides)


def parse_int_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def oracle_sweep_values(config):
    kshots = parse_int_list(get_config_value(config, "data.kshots", None), DEFAULT_KSHOTS)
    seeds = parse_int_list(get_config_value(config, "data.seeds", None), DEFAULT_SEEDS)
    return kshots, seeds


def mean_std(values):
    count = len(values)
    if count == 0:
        return 0.0, 0.0
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return mean, math.sqrt(variance)


def build_oracle_summary_rows(outputs):
    accuracy_groups = defaultdict(list)
    alpha_groups = defaultdict(list)
    for output in outputs:
        kshot = int(output["kshot"])
        accuracy_groups[kshot].append(float(output["oracle_accuracy"]))
        alpha_groups[kshot].append(float(output["oracle_alpha"]))

    rows = []
    for kshot in sorted(accuracy_groups):
        alpha_mean, alpha_std = mean_std(alpha_groups[kshot])
        accuracy_mean, accuracy_std = mean_std(accuracy_groups[kshot])
        rows.append(
            {
                "kshot": f"{kshot}-shot",
                "oracle alpha": f"{alpha_mean:.2f} +/- {alpha_std:.2f}",
                "oracle accuracy": f"{accuracy_mean:.2f} +/- {accuracy_std:.2f}",
            }
        )
    return rows


def format_table(rows):
    if not rows:
        return "kshot  oracle alpha  oracle accuracy"

    columns = ["kshot", "oracle alpha", "oracle accuracy"]
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    lines = [
        "  ".join(column.ljust(widths[column]) for column in columns),
        "  ".join("-" * widths[column] for column in columns),
    ]
    for row in rows:
        lines.append("  ".join(str(row[column]).ljust(widths[column]) for column in columns))
    return "\n".join(lines)


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, "debug", True), getattr(args, "disable_coloring", True))
    config = merge_configs(load_config_file(args.config), overrides)
    if getattr(args, "output_dir", None) is None:
        config.setdefault("logging", {})
        config["logging"]["output_dir"] = ProtoFuseOracleAlphaPipeline.DEFAULT_OUTPUT_DIR

    for dataset_config, _ in iter_dataset_configs(config):
        kshots, seeds = oracle_sweep_values(dataset_config)
        dataset_config.setdefault("data", {})
        dataset_config["data"]["kshot"] = int(max(kshots))
        dataset_config["data"]["seed"] = int(seeds[0])

        previous_level = logger._logger.level
        logger._logger.setLevel(logging.WARNING)
        try:
            dataset_outputs = ProtoFuseOracleAlphaPipeline(dataset_config).run_dataset_sweep(kshots, seeds)
        finally:
            logger._logger.setLevel(previous_level)

        dataset_name = str(dataset_config["data"]["dataset_name"])
        print(f"\n{dataset_name} x {ProtoFuseOracleAlphaPipeline.METHOD_NAME} x seed mean +/- std")
        print(format_table(build_oracle_summary_rows(dataset_outputs)), flush=True)


if __name__ == "__main__":
    main()
