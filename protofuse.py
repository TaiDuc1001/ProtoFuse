import copy
import logging
import math
import os
import random

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

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


def parse_args():
    parser = create_argument_parser("Run ProtoFuse batch sweep", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
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


def has_config_path(config, path):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def sweep_values(config, overrides):
    kshots = parse_int_list(
        get_config_value(config, "data.kshots", None),
        [get_config_value(config, "data.kshot")] if has_config_path(overrides, "data.kshot") else DEFAULT_KSHOTS,
    )
    seeds = parse_int_list(
        get_config_value(config, "data.seeds", None),
        [get_config_value(config, "data.seed")] if has_config_path(overrides, "data.seed") else DEFAULT_SEEDS,
    )
    return kshots, seeds


def split_indices_for(pipeline, kshot, seed):
    samples_by_class_idx = {}
    for idx, (_, class_idx) in enumerate(pipeline.dataset.samples):
        samples_by_class_idx.setdefault(class_idx, []).append(idx)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for class_idx in sorted(samples_by_class_idx):
        class_samples = sorted(samples_by_class_idx[class_idx])
        rng.shuffle(class_samples)

        if pipeline.val_fraction is None:
            train_candidates = class_samples
        else:
            val_count = int(math.floor(len(class_samples) * pipeline.val_fraction))
            if pipeline.val_fraction > 0 and val_count == 0 and class_samples:
                val_count = 1
            val_indices.extend(class_samples[:val_count])
            train_candidates = class_samples[val_count:]

        train_indices.extend(train_candidates[:kshot] if kshot > 0 else train_candidates)

    return train_indices, val_indices


def remap_labels(train_labels, eval_labels):
    task_classes = sorted(set(train_labels.tolist()))
    remap = {label: idx for idx, label in enumerate(task_classes)}
    missing = sorted(set(eval_labels.tolist()) - set(remap))
    if missing:
        raise ValueError(f"Eval labels not present in train split: {missing[:10]}")
    train = torch.tensor([remap[int(label)] for label in train_labels], dtype=torch.long)
    eval_ = torch.tensor([remap[int(label)] for label in eval_labels], dtype=torch.long)
    return train, eval_, len(task_classes)


def run_dataset_sweep(config, kshots, seeds):
    dataset_config = copy.deepcopy(config)
    data_cfg = dataset_config.setdefault("data", {})
    data_cfg["kshot"] = int(max(kshots))
    data_cfg["seed"] = int(seeds[0])

    set_global_seed(int(seeds[0]))
    pipeline = ProtoFusePipeline(dataset_config)
    pipeline._prepare_directories()
    pipeline._load_dataset()
    pipeline._split_dataset()
    pipeline._initialize_trainer()

    train_payload = pipeline._full_dataset_clip_features()
    train_features_all = train_payload["image_features"]
    train_labels_all = train_payload["labels"]

    if pipeline.val_fraction is None:
        eval_features_all, eval_labels_all = pipeline._cached_val_features()
    else:
        eval_features_all, eval_labels_all = train_features_all, train_labels_all

    results = {}
    for kshot in kshots:
        for seed in seeds:
            kshot = int(kshot)
            seed = int(seed)
            set_global_seed(seed)
            pipeline.kshot = kshot
            pipeline.seed = seed
            pipeline.config.data.kshot = kshot
            pipeline.config.data.seed = seed

            train_indices, val_indices = split_indices_for(pipeline, kshot, seed)
            train_idx = torch.tensor(train_indices, dtype=torch.long)
            train_features = train_features_all[train_idx].contiguous()
            train_labels = train_labels_all[train_idx].contiguous()

            if pipeline.val_fraction is None:
                eval_features = eval_features_all
                eval_labels = eval_labels_all
            else:
                eval_idx = torch.tensor(val_indices, dtype=torch.long)
                eval_features = eval_features_all[eval_idx].contiguous()
                eval_labels = eval_labels_all[eval_idx].contiguous()

            remapped_train, remapped_eval, num_classes = remap_labels(train_labels, eval_labels)
            metrics = pipeline.trainer.fuse_and_evaluate(
                train_features,
                remapped_train,
                eval_features,
                remapped_eval,
                num_classes,
            )
            results[(kshot, seed)] = metrics

    return results


def mean_std(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())


def fmt_mean_std(values, decimals=2, suffix=""):
    mean, std = mean_std(values)
    return f"{mean:.{decimals}f}{suffix} +/- {std:.{decimals}f}{suffix}"


def build_summary_rows(results, kshots, seeds):
    rows = []
    for kshot in kshots:
        members = [results[(int(kshot), int(seed))] for seed in seeds]
        rows.append(
            {
                "kshot": f"{int(kshot)}-shot",
                "runs": str(len(members)),
                "acc": fmt_mean_std([float(row.get("accuracy", 0.0)) for row in members], suffix="%"),
                "alpha": fmt_mean_std([float(row.get("alpha", 0.0)) for row in members], decimals=3),
            }
        )
    return rows


def format_table(rows, columns):
    if not rows:
        return " ".join(columns)
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


def print_dataset_summary(dataset_name, results, kshots, seeds):
    rows = build_summary_rows(results, kshots, seeds)
    print(f"\n{dataset_name} x ProtoFuse x seed mean +/- std")
    print(format_table(rows, ["kshot", "runs", "acc", "alpha"]), flush=True)


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, "debug", True), getattr(args, "disable_coloring", True))
    config = merge_configs(load_config_file(args.config), overrides)

    previous_level = logger._logger.level
    logger._logger.setLevel(logging.WARNING)
    try:
        dataset_configs = list(iter_dataset_configs(config))
    finally:
        logger._logger.setLevel(previous_level)

    for dataset_config, _ in dataset_configs:
        kshots, seeds = sweep_values(dataset_config, overrides)
        dataset_name = str(dataset_config["data"]["dataset_name"])

        previous_level = logger._logger.level
        logger._logger.setLevel(logging.WARNING)
        try:
            results = run_dataset_sweep(dataset_config, kshots, seeds)
        finally:
            logger._logger.setLevel(previous_level)

        print_dataset_summary(dataset_name, results, kshots, seeds)


if __name__ == "__main__":
    main()
