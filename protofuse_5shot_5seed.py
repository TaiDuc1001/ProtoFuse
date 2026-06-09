import os
import random

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

from utils import (
    DEFAULT_ARG_SCHEMA,
    create_argument_parser,
    logger,
    load_config_file,
    merge_configs,
    parse_override_arguments,
    process_parsed_args,
    setup_logging,
)

from src.pipelines.protofuse import ProtoFusePipeline


ARG_SCHEMA = DEFAULT_ARG_SCHEMA
KSHOTS = [1, 2, 4, 8, 16]
SEEDS = [1, 10, 100, 1000, 10000]
METRICS = [
    ("accuracy", "Acc", "{:.2f}%"),
    ("mca", "MCA", "{:.2f}%"),
    ("f1_macro", "F1-Mac", "{:.4f}"),
    ("f1_micro", "F1-Mic", "{:.4f}"),
    ("f1_weighted", "F1-Wei", "{:.4f}"),
    ("precision_macro", "P-Mac", "{:.4f}"),
    ("precision_micro", "P-Mic", "{:.4f}"),
    ("precision_weighted", "P-Wei", "{:.4f}"),
    ("recall_macro", "R-Mac", "{:.4f}"),
    ("recall_micro", "R-Mic", "{:.4f}"),
    ("recall_weighted", "R-Wei", "{:.4f}"),
]


def parse_args():
    parser = create_argument_parser("Run ProtoFuse 5-shot x 5-seed batch sweep", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = process_parsed_args(parsed, ARG_SCHEMA, parse_override_arguments(unknown))
    return parsed, overrides


def support_indices(dataset, kshot, seed):
    by_class = {}
    for idx, (_, class_idx) in enumerate(dataset.samples):
        by_class.setdefault(class_idx, []).append(idx)

    rng = random.Random(seed)
    indices = []
    for class_idx in sorted(by_class):
        class_samples = sorted(by_class[class_idx])
        rng.shuffle(class_samples)
        indices.extend(class_samples[:kshot] if kshot > 0 else class_samples)
    return indices


def remap_labels(train_labels, eval_labels):
    task_classes = sorted(set(train_labels.tolist()))
    remap = {label: idx for idx, label in enumerate(task_classes)}
    train = torch.tensor([remap[int(label)] for label in train_labels], dtype=torch.long)
    eval_ = torch.tensor([remap[int(label)] for label in eval_labels], dtype=torch.long)
    return train, eval_, len(task_classes)


def run_batch(config):
    data_cfg = config.setdefault("data", {})
    data_cfg["kshot"] = KSHOTS[0]
    data_cfg["seed"] = SEEDS[0]

    pipeline = ProtoFusePipeline(config)
    pipeline._prepare_directories()
    pipeline._load_dataset()
    pipeline._split_dataset()
    pipeline._initialize_trainer()

    train_payload = pipeline._full_dataset_clip_features()
    val_features, val_labels = pipeline._cached_val_features()

    results = {}
    for kshot in KSHOTS:
        for seed in SEEDS:
            indices = torch.tensor(support_indices(pipeline.dataset, kshot, seed), dtype=torch.long)
            train_features = train_payload["image_features"][indices]
            train_labels = train_payload["labels"][indices]
            remapped_train, remapped_val, num_classes = remap_labels(train_labels, val_labels)
            metrics = pipeline.trainer.fuse_and_evaluate(
                train_features,
                remapped_train,
                val_features,
                remapped_val,
                num_classes,
            )
            results[(kshot, seed)] = metrics
    return results


def print_summary_table(results, method_name, dataset_name):
    rows = []
    shot_columns = [f"{kshot}-shot" for kshot in KSHOTS]
    for metric_key, metric_label, fmt in METRICS:
        row = {"Metric": metric_label}
        for kshot, column in zip(KSHOTS, shot_columns):
            values = [
                float(results[(kshot, seed)].get(metric_key, 0.0))
                for seed in SEEDS
            ]
            mean = sum(values) / len(values)
            std = torch.tensor(values, dtype=torch.float32).std(unbiased=False).item()
            row[column] = f"{fmt.format(mean)} +/- {fmt.format(std)}"
        rows.append(row)

    logger.comparison_table(
        rows=rows,
        columns=["Metric", *shot_columns],
        title=f"{method_name} x {dataset_name}",
    )


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, "debug", True), getattr(args, "disable_coloring", True))
    config = merge_configs(load_config_file(args.config), overrides)
    results = run_batch(config)
    print_summary_table(results, "ProtoFuse", config["data"]["dataset_name"])


if __name__ == "__main__":
    main()
