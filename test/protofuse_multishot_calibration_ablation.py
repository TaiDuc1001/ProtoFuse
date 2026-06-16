import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protofuse_candidates import (
    DEFAULT_CONFIG,
    parse_int_list,
    load_datasets,
    load_model,
    extract_image_features,
    build_text_features,
    support_positions,
    split_positions_by_class,
    resolve_path,
)
from src.models.protofuse import ProtoFuse
from utils import get_config_value, load_config_file, merge_configs, parse_override_arguments, set_global_seed


DEFAULT_KSHOTS = [2, 4, 8, 16]
DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]
FIXED_ALPHAS = [0.25, 0.50, 0.75]


def add_dataset_args(parser):
    parser.add_argument("--data.root", "--data-root", dest="data_root", default=None, help="Dataset root directory.")
    parser.add_argument(
        "--data.dataset_name",
        "--dataset-name",
        "--dataset_name",
        dest="dataset_name",
        default=None,
        help="Dataset name used for prompt template and reporting.",
    )


def apply_dataset_args(args, overrides):
    if args.data_root is not None:
        overrides.setdefault("data", {})["root"] = args.data_root
    if args.dataset_name is not None:
        overrides.setdefault("data", {})["dataset_name"] = args.dataset_name
    return overrides


def alpha_grid(steps, device):
    return torch.linspace(0.0, 1.0, max(2, int(steps)), device=device)


def build_selector(text_features, support_features, support_labels, device, alpha_steps, beta_values, force_loo_accuracy=True):
    selector = ProtoFuse.__new__(ProtoFuse)
    selector.device = torch.device(device)
    selector.alpha_steps = max(2, int(alpha_steps))
    selector.force_loo_accuracy = ProtoFuse._coerce_bool(force_loo_accuracy, True)
    selector.centroid_mix_beta_values = ProtoFuse._coerce_float_list(beta_values, [])
    selector.alphas = alpha_grid(alpha_steps, selector.device)
    selector.text_prototypes = F.normalize(text_features.to(selector.device).float(), dim=-1)
    selector.embed_dim = selector.text_prototypes.shape[-1]
    support_features = F.normalize(support_features.to(selector.device).float(), dim=-1)
    support_labels = support_labels.to(selector.device).long()
    visual = selector.build_visual_centroids(support_features, support_labels, selector.text_prototypes.shape[0])
    return selector, selector.text_prototypes, visual, support_features, support_labels


def accuracy_for_alpha(alpha, text, visual, eval_features, eval_labels, device, normalize=True):
    eval_features = F.normalize(eval_features.to(device).float(), dim=-1)
    eval_labels = eval_labels.to(device).long()
    proto = (1.0 - alpha) * text + alpha * visual
    if normalize:
        proto = F.normalize(proto, dim=-1)
    preds = (eval_features @ proto.T).argmax(dim=-1)
    return preds.eq(eval_labels).float().mean().item() * 100.0


def oracle_alpha(text, visual, eval_features, eval_labels, alphas, device):
    best = {"alpha": 0.0, "accuracy": -float("inf")}
    for alpha in alphas:
        value = float(alpha.item())
        acc = accuracy_for_alpha(value, text, visual, eval_features, eval_labels, device)
        if acc > best["accuracy"]:
            best = {"alpha": value, "accuracy": acc}
    return best


def class_feature_tensor(features, labels, num_classes, kshot):
    by_class = []
    for class_idx in range(num_classes):
        idx = torch.nonzero(labels.eq(class_idx), as_tuple=False).flatten()
        if idx.numel() < kshot:
            raise RuntimeError(f"Class {class_idx} has {idx.numel()} support samples, expected {kshot}.")
        by_class.append(features[idx[:kshot]])
    return torch.stack(by_class, dim=0)


def loo_curves(selector, text, support_features, support_labels, kshot):
    num_classes = text.shape[0]
    class_features = class_feature_tensor(support_features, support_labels, num_classes, kshot)
    targets = torch.arange(num_classes, device=selector.device)
    correct = torch.zeros(len(selector.alphas), device=selector.device)
    rescue = torch.zeros(len(selector.alphas), device=selector.device)
    damage = torch.zeros(len(selector.alphas), device=selector.device)

    for hold_idx in range(kshot):
        held = F.normalize(class_features[:, hold_idx, :], dim=-1)
        keep = torch.arange(kshot, device=selector.device) != hold_idx
        visual_minus = torch.stack(
            [selector._weighted_visual_centroid(class_features[class_idx, keep], text[class_idx]) for class_idx in range(num_classes)]
        )
        text_correct = (held @ text.T).argmax(dim=-1).eq(targets)
        refined = F.normalize(
            (1.0 - selector.alphas).view(-1, 1, 1) * text + selector.alphas.view(-1, 1, 1) * visual_minus,
            dim=-1,
        )
        fused_preds = torch.einsum("cd,akd->ack", held, refined).argmax(dim=-1)
        fused_correct = fused_preds.eq(targets.view(1, -1))
        correct += fused_correct.sum(dim=1).float()
        rescue += ((~text_correct).view(1, -1) & fused_correct).sum(dim=1).float()
        damage += (text_correct.view(1, -1) & ~fused_correct).sum(dim=1).float()

    return {
        "accuracy_count": correct,
        "rescue": rescue,
        "damage": damage,
        "net": rescue - damage,
    }


def support_accuracy_curve(alphas, text, visual, support_features, support_labels):
    correct = []
    for alpha in alphas:
        proto = F.normalize((1.0 - alpha) * text + alpha * visual, dim=-1)
        preds = (support_features @ proto.T).argmax(dim=-1)
        correct.append(preds.eq(support_labels).sum().float())
    return torch.stack(correct)


def select_from_curve(alphas, curve):
    idx = int(curve.argmax().item())
    return float(alphas[idx].item())


def evaluate_run(
    text_features,
    support_features,
    support_labels,
    eval_features,
    eval_labels,
    device,
    alpha_steps,
    beta_values,
    kshot,
    force_loo_accuracy=True,
):
    selector, text, visual, support_features, support_labels = build_selector(
        text_features, support_features, support_labels, device, alpha_steps, beta_values, force_loo_accuracy
    )
    alphas = selector.alphas
    curves = loo_curves(selector, text, support_features, support_labels, kshot)
    support_curve = support_accuracy_curve(alphas, text, visual, support_features, support_labels)
    oracle = oracle_alpha(text, visual, eval_features, eval_labels, alphas, device)

    strategies = [
        ("Text only (alpha=0)", 0.0),
        ("Visual only (alpha=1)", 1.0),
        *[(f"Fixed alpha={alpha:.2f}", alpha) for alpha in FIXED_ALPHAS],
        ("Support accuracy, no LOO", select_from_curve(alphas, support_curve)),
        ("LOO force accuracy (ours)", select_from_curve(alphas, curves["accuracy_count"])),
        ("Oracle alpha on test", oracle["alpha"]),
    ]

    rows = []
    text_acc = accuracy_for_alpha(0.0, text, visual, eval_features, eval_labels, device)
    for strategy, selected_alpha in strategies:
        acc = accuracy_for_alpha(selected_alpha, text, visual, eval_features, eval_labels, device)
        rows.append(
            {
                "strategy": strategy,
                "alpha": float(selected_alpha),
                "accuracy": float(acc),
                "delta": float(acc - text_acc),
            }
        )
    return rows


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def format_mean_std(stats):
    if stats is None:
        return "-"
    return f"{stats['mean']:.2f} +/- {stats['std']:.2f}"


def result_table_rows(aggregate, kshots):
    rows = []
    for row in aggregate:
        out = {"Calibration strategy": row["strategy"]}
        for kshot in kshots:
            out[f"{kshot}-shot"] = format_mean_std(row["accuracy_by_kshot"].get(str(kshot)))
        out["avg acc"] = format_mean_std({"mean": row["accuracy_mean"], "std": row["accuracy_std"]})
        out["Delta"] = f"{row['delta_mean']:+.2f}"
        out["alpha"] = f"{row['alpha_mean']:.2f} +/- {row['alpha_std']:.2f}"
        out["runs"] = str(row["runs"])
        rows.append(out)
    return rows


def build_table(aggregate, dataset_name, kshots):
    title = f"{dataset_name} multi-shot calibration ablation"
    rows = result_table_rows(aggregate, kshots)
    table = pd.DataFrame(rows).to_string(index=False)
    return f"{title}\n{table}"


def latex_rows(aggregate, kshots, include_delta=False):
    rows = []
    for row in aggregate:
        values = [
            f"{row['accuracy_by_kshot'][str(kshot)]['mean']:.2f}"
            if str(kshot) in row["accuracy_by_kshot"]
            else "-"
            for kshot in kshots
        ]
        value = f"{row['accuracy_mean']:.2f}"
        if include_delta:
            value = f"{value} ({row['delta_mean']:+.2f})"
        rows.append(f"{row['strategy']} & {' & '.join(values)} & {value} \\\\")
    return "\n".join(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Table A: ProtoFuse multi-shot calibration ablation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--kshots", default=",".join(str(v) for v in DEFAULT_KSHOTS))
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--latex", action="store_true", help="Print compact LaTeX rows for the current dataset.")
    parser.add_argument("--disable-coloring", action="store_true", help="Deprecated no-op; output is plain text.")
    add_dataset_args(parser)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, apply_dataset_args(parsed, overrides)


def main():
    args, overrides = parse_args()
    config = merge_configs(load_config_file(args.config), overrides)

    kshots = parse_int_list(args.kshots)
    seeds = parse_int_list(args.seeds)
    alpha_steps = int(get_config_value(config, "model.alpha_steps", 101))
    beta_values = get_config_value(config, "model.centroid_mix.beta_values", None)
    force_loo_accuracy = ProtoFuse._coerce_bool(get_config_value(config, "model.force_loo_accuracy", True), True)
    device_name = str(get_config_value(config, "training.device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    batch_size = int(get_config_value(config, "training.batch_size", 128))
    num_workers = int(get_config_value(config, "data.num_workers", 4))
    dataset_name = str(get_config_value(config, "data.dataset_name", "DTD"))

    set_global_seed(1)
    train_dataset, eval_dataset, val_fraction, dataset_root = load_datasets(config)
    classnames = list(train_dataset.classes)
    print(
        f"Dataset={dataset_name}, root={dataset_root}, classes={len(classnames)}, "
        f"kshots={kshots}, seeds={seeds}, force_loo_accuracy={force_loo_accuracy}, device={device}"
    )

    model = load_model(config, device)
    train_features_all, train_labels_all = extract_image_features(model, train_dataset, device, batch_size, num_workers)
    if eval_dataset is None:
        eval_features_all, eval_labels_all = train_features_all, train_labels_all
    else:
        eval_features_all, eval_labels_all = extract_image_features(model, eval_dataset, device, batch_size, num_workers)
    text_features = build_text_features(model, classnames, dataset_name, device)
    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    raw = []
    total = len(kshots) * len(seeds)
    completed = 0
    for kshot in kshots:
        for seed in seeds:
            completed += 1
            print(f"Running {kshot}-shot | seed {seed} ({completed}/{total})", flush=True)
            set_global_seed(seed)
            if val_fraction is None:
                train_idx = support_positions(train_labels_all, kshot, seed)
                support_features = train_features_all[train_idx].contiguous()
                support_labels = train_labels_all[train_idx].contiguous()
                eval_features = eval_features_all
                eval_labels = eval_labels_all
            else:
                train_idx, val_idx = split_positions_by_class(train_labels_all, kshot, seed, val_fraction)
                support_features = train_features_all[train_idx].contiguous()
                support_labels = train_labels_all[train_idx].contiguous()
                eval_features = train_features_all[val_idx].contiguous()
                eval_labels = train_labels_all[val_idx].contiguous()

            rows = evaluate_run(
                text_features,
                support_features,
                support_labels,
                eval_features,
                eval_labels,
                device,
                alpha_steps,
                beta_values,
                kshot,
                force_loo_accuracy,
            )
            for row in rows:
                raw.append({"dataset": dataset_name, "kshot": int(kshot), "seed": int(seed), **row})

    order = []
    by_strategy = {}
    for row in raw:
        by_strategy.setdefault(row["strategy"], []).append(row)
        if row["strategy"] not in order:
            order.append(row["strategy"])

    aggregate = []
    for strategy in order:
        members = by_strategy[strategy]
        acc_mean, acc_std = mean_std([row["accuracy"] for row in members])
        delta_mean, delta_std = mean_std([row["delta"] for row in members])
        alpha_mean, alpha_std = mean_std([row["alpha"] for row in members])
        accuracy_by_kshot = {}
        for kshot in kshots:
            shot_members = [row for row in members if row["kshot"] == int(kshot)]
            if not shot_members:
                continue
            shot_mean, shot_std = mean_std([row["accuracy"] for row in shot_members])
            accuracy_by_kshot[str(kshot)] = {
                "mean": shot_mean,
                "std": shot_std,
                "runs": len(shot_members),
            }
        aggregate.append(
            {
                "strategy": strategy,
                "runs": len(members),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "accuracy_by_kshot": accuracy_by_kshot,
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_std,
            }
        )

    print()
    print(build_table(aggregate, dataset_name, kshots))
    if args.latex:
        print("\nLaTeX rows")
        print(latex_rows(aggregate, kshots, include_delta=True))

    if args.output_json:
        out_path = resolve_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "ablation": "multishot_calibration",
                    "dataset": dataset_name,
                    "dataset_root": str(dataset_root),
                    "kshots": kshots,
                    "seeds": seeds,
                    "force_loo_accuracy": force_loo_accuracy,
                    "raw": raw,
                    "aggregate": aggregate,
                },
                f,
                indent=2,
            )
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
