import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protofuse_candidates import (
    DEFAULT_CONFIG,
    parse_int_list,
    parse_float_list,
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


DEFAULT_KSHOTS = [1, 2, 4, 8, 16]
DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]
DEFAULT_BETAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


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


def build_selector(text_features, support_features, support_labels, device, alpha_steps, beta_values):
    selector = ProtoFuse.__new__(ProtoFuse)
    selector.device = torch.device(device)
    selector.alpha_steps = max(2, int(alpha_steps))
    selector.centroid_mix_beta_values = ProtoFuse._coerce_float_list(beta_values, DEFAULT_BETAS)
    selector.alphas = alpha_grid(alpha_steps, selector.device)
    selector.text_prototypes = F.normalize(text_features.to(selector.device).float(), dim=-1)
    selector.embed_dim = selector.text_prototypes.shape[-1]
    support_features = F.normalize(support_features.to(selector.device).float(), dim=-1)
    support_labels = support_labels.to(selector.device).long()
    visual = selector.build_visual_centroids(support_features, support_labels, selector.text_prototypes.shape[0])
    return selector, selector.text_prototypes, visual, support_features, support_labels


def accuracy_for_proto(proto, eval_features, eval_labels, device):
    eval_features = F.normalize(eval_features.to(device).float(), dim=-1)
    eval_labels = eval_labels.to(device).long()
    preds = (eval_features @ proto.T).argmax(dim=-1)
    return preds.eq(eval_labels).float().mean().item() * 100.0


def fused_proto(text, visual, alpha, normalize=True):
    proto = (1.0 - alpha) * text + alpha * visual
    return F.normalize(proto, dim=-1) if normalize else proto


def calibrated_alpha(selector, text, visual, support_features, support_labels, num_classes):
    _, alpha = selector.hopc_alpha(text, visual, support_features, support_labels, num_classes)
    return float(alpha)


def evaluate_run(text_features, support_features, support_labels, eval_features, eval_labels, device, alpha_steps, beta_values):
    selector, text, visual, support_features, support_labels = build_selector(
        text_features, support_features, support_labels, device, alpha_steps, beta_values
    )
    num_classes = text.shape[0]
    alpha = calibrated_alpha(selector, text, visual, support_features, support_labels, num_classes)
    strategies = [
        ("Text prototype only", text),
        ("Visual centroid only", visual),
        ("Unnormalized text-visual fusion", fused_proto(text, visual, 0.5, normalize=False)),
        ("Normalized text-visual fusion, fixed alpha=0.5", fused_proto(text, visual, 0.5, normalize=True)),
        ("Normalized text-visual fusion, calibrated alpha (ours)", fused_proto(text, visual, alpha, normalize=True)),
    ]
    text_acc = accuracy_for_proto(text, eval_features, eval_labels, device)
    rows = []
    for strategy, proto in strategies:
        acc = accuracy_for_proto(proto, eval_features, eval_labels, device)
        rows.append(
            {
                "strategy": strategy,
                "alpha": 0.0 if strategy == "Text prototype only" else 1.0 if strategy == "Visual centroid only" else 0.5 if "fixed" in strategy or "Unnormalized" in strategy else alpha,
                "accuracy": float(acc),
                "delta": float(acc - text_acc),
            }
        )
    return rows


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def build_table(aggregate, dataset_name):
    table = Table(title=f"{dataset_name} prototype construction ablation")
    table.add_column("Prototype construction")
    table.add_column("accuracy (%)", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("alpha", justify="right")
    table.add_column("runs", justify="right")
    for row in aggregate:
        table.add_row(
            row["strategy"],
            f"{row['accuracy_mean']:.2f} +/- {row['accuracy_std']:.2f}",
            f"{row['delta_mean']:+.2f}",
            f"{row['alpha_mean']:.2f} +/- {row['alpha_std']:.2f}",
            str(row["runs"]),
        )
    return table


def parse_args():
    parser = argparse.ArgumentParser(description="Table C: ProtoFuse prototype construction ablation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--kshots", default=",".join(str(v) for v in DEFAULT_KSHOTS))
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--betas", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--disable-coloring", action="store_true")
    add_dataset_args(parser)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, apply_dataset_args(parsed, overrides)


def main():
    args, overrides = parse_args()
    config = merge_configs(load_config_file(args.config), overrides)
    console = Console(no_color=args.disable_coloring)

    kshots = parse_int_list(args.kshots)
    seeds = parse_int_list(args.seeds)
    alpha_steps = int(get_config_value(config, "model.alpha_steps", 101))
    beta_values = parse_float_list(args.betas, get_config_value(config, "model.centroid_mix.beta_values", DEFAULT_BETAS))
    device_name = str(get_config_value(config, "training.device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    batch_size = int(get_config_value(config, "training.batch_size", 128))
    num_workers = int(get_config_value(config, "data.num_workers", 4))
    dataset_name = str(get_config_value(config, "data.dataset_name", "DTD"))

    set_global_seed(1)
    train_dataset, eval_dataset, val_fraction, dataset_root = load_datasets(config)
    classnames = list(train_dataset.classes)
    console.print(
        f"Dataset={dataset_name}, root={dataset_root}, classes={len(classnames)}, "
        f"kshots={kshots}, seeds={seeds}, device={device}"
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
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("running", total=total)
        for kshot in kshots:
            for seed in seeds:
                progress.update(task, description=f"{kshot}-shot | seed {seed}")
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
                )
                for row in rows:
                    raw.append({"dataset": dataset_name, "kshot": int(kshot), "seed": int(seed), **row})
                progress.advance(task)

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
        aggregate.append(
            {
                "strategy": strategy,
                "runs": len(members),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_std,
            }
        )

    console.print()
    console.print(build_table(aggregate, dataset_name))
    if args.latex:
        console.print("\n[bold]LaTeX rows[/bold]")
        for row in aggregate:
            console.print(f"{row['strategy']} & {row['accuracy_mean']:.2f} \\\\")

    if args.output_json:
        out_path = resolve_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "ablation": "prototype_construction",
                    "dataset": dataset_name,
                    "dataset_root": str(dataset_root),
                    "kshots": kshots,
                    "seeds": seeds,
                    "beta_values": beta_values,
                    "raw": raw,
                    "aggregate": aggregate,
                },
                f,
                indent=2,
            )
        console.print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
