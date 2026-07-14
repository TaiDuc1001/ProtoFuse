import copy
import logging
import math
import os
import random

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.nn.functional as F

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
ALPHA_BATCH_SIZE = 16
TOP_K_ALPHA = 5


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


def _support_calibration_curve(trainer, T, V, train_features, train_labels, num_classes):
    class_indices = [
        torch.nonzero(train_labels == class_idx, as_tuple=False).flatten()
        for class_idx in range(num_classes)
    ]
    shots_per_class = min(int(indices.numel()) for indices in class_indices)

    if shots_per_class < 2:
        best = None
        beta_values = trainer._centroid_mix_beta_values()
        for beta in beta_values:
            Q = trainer.generate_surrogate_mix(
                V,
                V,
                beta,
                top_k=1,
                samples_per_nb=1,
                std=0.0,
                filter_correct=False,
            )
            pseudo_features = Q.squeeze(1)

            labels = torch.arange(num_classes, device=trainer.device)
            net_scores = []
            for alpha in trainer.alphas:
                proto = F.normalize((1.0 - alpha) * T + alpha * V, dim=-1)
                preds = (pseudo_features @ proto.T).argmax(dim=-1)
                correct = preds.eq(labels).sum().float()
                net_scores.append(correct)

            curve = torch.stack(net_scores)
            alpha_idx = int(curve.argmax().item())
            score = float(curve[alpha_idx].item())
            alpha = float(trainer.alphas[alpha_idx].item())
            candidate = (score, -alpha, curve)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

        if best is None:
            return torch.zeros_like(trainer.alphas)
        return best[2].float()

    kshot = shots_per_class
    class_features = torch.stack(
        [
            train_features[class_indices[class_idx][:kshot]].to(trainer.device)
            for class_idx in range(num_classes)
        ]
    )
    scores = torch.zeros(len(trainer.alphas), device=trainer.device)
    targets = torch.arange(num_classes, device=trainer.device)

    for hold_idx in range(kshot):
        held = F.normalize(class_features[:, hold_idx, :], dim=-1)
        keep = torch.arange(kshot, device=trainer.device) != hold_idx
        V_minus = torch.stack(
            [
                trainer._visual_centroid(class_features[class_idx, keep])
                for class_idx in range(num_classes)
            ]
        )
        prototypes = F.normalize(
            (1.0 - trainer.alphas).view(-1, 1, 1) * T
            + trainer.alphas.view(-1, 1, 1) * V_minus,
            dim=-1,
        )
        predictions = torch.einsum("cd,akd->ack", held, prototypes).argmax(dim=-1)
        correct = predictions.eq(targets.view(1, -1))
        scores += correct.sum(dim=1).float()
    return scores


def _test_accuracy_curve(trainer, T, V, eval_features, eval_labels):
    features = F.normalize(eval_features.to(trainer.device).float(), dim=-1)
    labels = eval_labels.to(trainer.device).long()
    values = []
    for start in range(0, len(trainer.alphas), ALPHA_BATCH_SIZE):
        alpha = trainer.alphas[start:start + ALPHA_BATCH_SIZE].float()
        prototypes = F.normalize(
            (1.0 - alpha).view(-1, 1, 1) * T.unsqueeze(0)
            + alpha.view(-1, 1, 1) * V.unsqueeze(0),
            dim=-1,
        )
        logits = torch.einsum("nd,acd->anc", features, prototypes)
        accuracy = logits.argmax(dim=-1).eq(labels.view(1, -1)).float().mean(dim=1) * 100.0
        values.append(accuracy)
    return torch.cat(values)


def _pearson(x, y):
    x = torch.as_tensor(x, dtype=torch.float64).flatten()
    y = torch.as_tensor(y, dtype=torch.float64).flatten()
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.numel() < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if denom <= 1e-12:
        return None
    return float((x @ y / denom).item())


def _classification_margin(logits, labels):
    correct = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    other = logits.clone()
    other.scatter_(1, labels.view(-1, 1), -float("inf"))
    return correct - other.max(dim=1).values


def analyze_run(trainer, train_features, train_labels, eval_features, eval_labels, num_classes, metrics):
    with torch.inference_mode():
        device = trainer.device
        train_norm = F.normalize(train_features.to(device).float(), dim=-1)
        eval_norm = F.normalize(eval_features.to(device).float(), dim=-1)
        train_labels_device = train_labels.to(device).long()
        eval_labels_device = eval_labels.to(device).long()
        T = trainer.text_prototypes
        V = trainer.build_visual_centroids(train_norm, train_labels_device, num_classes)

        selected_alpha = float(metrics["alpha"])
        selected_proto = F.normalize((1.0 - selected_alpha) * T + selected_alpha * V, dim=-1)
        test_curve = _test_accuracy_curve(trainer, T, V, eval_features, eval_labels)
        oracle_idx = int(test_curve.argmax().item())
        oracle_alpha = float(trainer.alphas[oracle_idx].item())
        oracle_proto = F.normalize((1.0 - oracle_alpha) * T + oracle_alpha * V, dim=-1)
        support_curve = _support_calibration_curve(
            trainer, T, V, train_norm, train_labels_device, num_classes
        )

        text_logits = eval_norm @ T.T
        visual_logits = eval_norm @ V.T
        selected_logits = eval_norm @ selected_proto.T
        oracle_logits = eval_norm @ oracle_proto.T
        text_correct = text_logits.argmax(dim=-1).eq(eval_labels_device)
        visual_correct = visual_logits.argmax(dim=-1).eq(eval_labels_device)
        selected_correct = selected_logits.argmax(dim=-1).eq(eval_labels_device)
        oracle_correct = oracle_logits.argmax(dim=-1).eq(eval_labels_device)
        total = max(int(eval_labels_device.numel()), 1)

        def percentage(mask):
            return float(mask.sum().item()) * 100.0 / total

        selected_idx = int(
            torch.abs(trainer.alphas - selected_alpha).argmin().item()
        )
        kth_accuracy = torch.topk(
            test_curve, k=min(TOP_K_ALPHA, int(test_curve.numel()))
        ).values.min()

        if num_classes > 1:
            visual_similarity = V @ V.T
            text_similarity = T @ T.T
            diagonal = torch.eye(num_classes, dtype=torch.bool, device=device)
            visual_similarity = visual_similarity.masked_fill(diagonal, -float("inf"))
            text_similarity = text_similarity.masked_fill(diagonal, -float("inf"))
            centroid_separation = float(
                (1.0 - visual_similarity.max(dim=1).values).mean().item()
            )
            text_separation = float(
                (1.0 - text_similarity.max(dim=1).values).mean().item()
            )
            query_margin = float(
                _classification_margin(selected_logits, eval_labels_device).mean().item()
            )
        else:
            centroid_separation = 1.0
            text_separation = 1.0
            query_margin = 0.0

        text_accuracy = percentage(text_correct)
        selected_accuracy = percentage(selected_correct)
        oracle_accuracy = float(test_curve[oracle_idx].item())

        return {
            "failure": {
                "text_correct_visual_wrong": percentage(text_correct & ~visual_correct),
                "text_wrong_visual_correct": percentage(~text_correct & visual_correct),
                "text_wrong_visual_wrong": percentage(~text_correct & ~visual_correct),
                "ours_wrong_oracle_correct": percentage(~selected_correct & oracle_correct),
                "ours_wrong_oracle_wrong": percentage(~selected_correct & ~oracle_correct),
            },
            "calibration": {
                "selected_alpha": selected_alpha,
                "oracle_alpha": oracle_alpha,
                "regret": oracle_accuracy - selected_accuracy,
                "curve_corr": _pearson(support_curve, test_curve),
                "top_k_hit": float(test_curve[selected_idx] >= kth_accuracy),
            },
            "geometry": {
                "text_visual_alignment": float((T * V).sum(dim=-1).mean().item()),
                "support_compactness": float(
                    (train_norm * V[train_labels_device]).sum(dim=-1).mean().item()
                ),
                "centroid_separation": centroid_separation,
                "text_separation": text_separation,
                "query_margin": query_margin,
                "gain_over_text": selected_accuracy - text_accuracy,
                "oracle_regret": oracle_accuracy - selected_accuracy,
                "selected_alpha": selected_alpha,
            },
        }


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
            metrics["analysis"] = analyze_run(
                pipeline.trainer,
                train_features,
                remapped_train,
                eval_features,
                remapped_eval,
                num_classes,
                metrics,
            )
            results[(kshot, seed)] = metrics

    return results


def mean_std(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())


def fmt_mean_std(values, decimals=2, suffix=""):
    mean, std = mean_std(values)
    return f"{mean:.{decimals}f}{suffix} +/- {std:.{decimals}f}{suffix}"


def fmt_optional_mean_std(values, decimals=3, suffix=""):
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not valid:
        return "n/a"
    return fmt_mean_std(valid, decimals=decimals, suffix=suffix)


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


def build_failure_rows(results, kshots, seeds):
    columns = [
        ("text_correct_visual_wrong", "T ok / V wrong"),
        ("text_wrong_visual_correct", "T wrong / V ok"),
        ("text_wrong_visual_wrong", "T wrong / V wrong"),
        ("ours_wrong_oracle_correct", "ours wrong / oracle ok"),
        ("ours_wrong_oracle_wrong", "ours+oracle wrong"),
    ]
    rows = []
    for kshot in kshots:
        failures = [
            results[(int(kshot), int(seed))]["analysis"]["failure"]
            for seed in seeds
        ]
        row = {"kshot": f"{int(kshot)}-shot"}
        for key, label in columns:
            row[label] = fmt_mean_std(
                [member[key] for member in failures], decimals=2, suffix="%"
            )
        rows.append(row)
    return rows


def build_calibration_rows(results, kshots, seeds):
    rows = []
    for kshot in kshots:
        members = [
            results[(int(kshot), int(seed))]["analysis"]["calibration"]
            for seed in seeds
        ]
        rows.append(
            {
                "kshot": f"{int(kshot)}-shot",
                "selected alpha": fmt_mean_std(
                    [row["selected_alpha"] for row in members], decimals=3
                ),
                "oracle alpha": fmt_mean_std(
                    [row["oracle_alpha"] for row in members], decimals=3
                ),
                "regret": fmt_mean_std(
                    [row["regret"] for row in members], decimals=2, suffix="%"
                ),
                "curve corr": fmt_optional_mean_std(
                    [row["curve_corr"] for row in members], decimals=3
                ),
                f"top-{TOP_K_ALPHA} hit": fmt_mean_std(
                    [100.0 * row["top_k_hit"] for row in members],
                    decimals=1,
                    suffix="%",
                ),
            }
        )
    return rows


def build_geometry_rows(results, kshots, seeds):
    labels = [
        ("text_visual_alignment", "text-visual alignment"),
        ("support_compactness", "support compactness"),
        ("centroid_separation", "centroid separation"),
        ("text_separation", "text separation"),
        ("query_margin", "query margin"),
    ]
    members = [
        results[(int(kshot), int(seed))]["analysis"]["geometry"]
        for kshot in kshots
        for seed in seeds
    ]
    rows = []
    for key, label in labels:
        values = [row[key] for row in members]
        rows.append(
            {
                "geometry metric": label,
                "mean +/- std": fmt_mean_std(values, decimals=3),
                "corr(gain)": format_correlation(
                    _pearson(values, [row["gain_over_text"] for row in members])
                ),
                "corr(regret)": format_correlation(
                    _pearson(values, [row["oracle_regret"] for row in members])
                ),
                "corr(alpha)": format_correlation(
                    _pearson(values, [row["selected_alpha"] for row in members])
                ),
            }
        )
    return rows


def format_correlation(value):
    return "n/a" if value is None else f"{value:.3f}"


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
