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


DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]
DEFAULT_BETAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
FIXED_ALPHAS = [0.25, 0.50, 0.75]
MODE_LABELS = {
    "vv": "Centroid-mix, visual neighbor only",
    "tt": "Centroid-mix, text neighbor only",
    "hybrid": "Centroid-mix, hybrid neighbor only",
}


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
    selector.centroid_mix_beta_values = ProtoFuse._coerce_float_list(beta_values, DEFAULT_BETAS)
    selector.alphas = alpha_grid(alpha_steps, selector.device)
    selector.text_prototypes = F.normalize(text_features.to(selector.device).float(), dim=-1)
    selector.embed_dim = selector.text_prototypes.shape[-1]
    support_features = F.normalize(support_features.to(selector.device).float(), dim=-1)
    support_labels = support_labels.to(selector.device).long()
    visual = selector.build_visual_centroids(support_features, support_labels, selector.text_prototypes.shape[0])
    return selector, selector.text_prototypes, visual


def accuracy_for_alpha(alpha, text, visual, eval_features, eval_labels, device):
    eval_features = F.normalize(eval_features.to(device).float(), dim=-1)
    eval_labels = eval_labels.to(device).long()
    proto = F.normalize((1.0 - alpha) * text + alpha * visual, dim=-1)
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


def curve_knee(values, device):
    values = values.float()
    span = values.max() - values.min()
    if span <= 1e-12:
        return None, 0.0, 0.0
    y = (values - values.min()) / (span + 1e-12)
    x = torch.linspace(0.0, 1.0, len(values), device=device)
    knee_scores = y - x
    idx = int(knee_scores.argmax().item())
    return idx, float(knee_scores[idx].item()), float(span.item())


def centroid_mix_neighbors(text, visual, mode):
    if mode == "vv":
        similarity = visual @ visual.T
    elif mode == "tt":
        similarity = text @ text.T
    elif mode == "hybrid":
        similarity = 0.5 * (visual @ visual.T) + 0.5 * (text @ text.T)
    else:
        raise ValueError(f"Unknown neighbor mode: {mode}")
    similarity = similarity.clone()
    similarity.fill_diagonal_(-float("inf"))
    return similarity.argmax(dim=1)


def centroid_mix_net_curve(alphas, text, visual, neighbors, beta, device, force_loo_accuracy=True):
    num_classes = text.shape[0]
    labels = torch.arange(num_classes, device=device)
    pseudo = F.normalize((1.0 - beta) * visual + beta * visual[neighbors], dim=-1)
    text_correct = None if force_loo_accuracy else (pseudo @ text.T).argmax(dim=-1).eq(labels)
    scores = []
    for alpha in alphas:
        proto = F.normalize((1.0 - alpha) * text + alpha * visual, dim=-1)
        fused_correct = (pseudo @ proto.T).argmax(dim=-1).eq(labels)
        if force_loo_accuracy:
            scores.append(fused_correct.sum().float())
        else:
            rescue = ((~text_correct) & fused_correct).sum().float()
            damage = (text_correct & ~fused_correct).sum().float()
            scores.append(rescue - damage)
    return torch.stack(scores)


def candidate_curves(selector, text, visual, modes, beta_values):
    rows = []
    beta_values = sorted({round(float(beta), 6) for beta in beta_values if 0.0 < float(beta) < 0.5})
    if 0.45 not in beta_values:
        beta_values.append(0.45)
        beta_values.sort()
    if text.shape[0] < 2:
        return rows
    for mode in modes:
        neighbors = centroid_mix_neighbors(text, visual, mode)
        for beta in beta_values:
            curve = centroid_mix_net_curve(
                selector.alphas,
                text,
                visual,
                neighbors,
                beta,
                selector.device,
                force_loo_accuracy=getattr(selector, "force_loo_accuracy", True),
            )
            knee_idx, knee_strength, signal_span = curve_knee(curve, selector.device)
            max_idx = int(curve.argmax().item())
            rows.append(
                {
                    "mode": mode,
                    "beta": float(beta),
                    "curve": curve,
                    "knee_idx": knee_idx,
                    "knee_strength": knee_strength,
                    "signal_span": signal_span,
                    "quality": knee_strength * signal_span / max(1, text.shape[0]),
                    "max_idx": max_idx,
                    "max_score": float(curve[max_idx].item()),
                }
            )
    return rows


def select_mode_knee(selector, curves, fallback=True):
    valid = [row for row in curves if row["knee_idx"] is not None]
    if not valid:
        return 0.0
    best = max(valid, key=lambda row: (row["quality"], -float(selector.alphas[row["knee_idx"]].item())))
    if fallback and best["quality"] <= 0.0:
        return 0.0
    return float(selector.alphas[best["knee_idx"]].item())


def select_all_argmax(selector, curves):
    if not curves:
        return 0.0
    best = max(curves, key=lambda row: (row["max_score"], -float(selector.alphas[row["max_idx"]].item())))
    return float(selector.alphas[best["max_idx"]].item())


def evaluate_run(
    text_features,
    support_features,
    support_labels,
    eval_features,
    eval_labels,
    device,
    alpha_steps,
    beta_values,
    force_loo_accuracy=True,
):
    selector, text, visual = build_selector(
        text_features, support_features, support_labels, device, alpha_steps, beta_values, force_loo_accuracy
    )
    all_curves = candidate_curves(selector, text, visual, ["vv", "tt", "hybrid"], selector.centroid_mix_beta_values)
    oracle = oracle_alpha(text, visual, eval_features, eval_labels, selector.alphas, device)

    strategies = [
        ("Text only (alpha=0)", 0.0),
        ("Visual only (alpha=1)", 1.0),
        *[(f"Fixed alpha={alpha:.2f}", alpha) for alpha in FIXED_ALPHAS],
    ]
    for mode, label in MODE_LABELS.items():
        mode_curves = [row for row in all_curves if row["mode"] == mode]
        strategies.append((label, select_mode_knee(selector, mode_curves, fallback=True)))
    strategies.extend(
        [
            ("All modes, argmax score", select_all_argmax(selector, all_curves)),
            ("All modes, knee selection w/o fallback", select_mode_knee(selector, all_curves, fallback=False)),
            ("All modes, knee selection + fallback (ours)", select_mode_knee(selector, all_curves, fallback=True)),
            ("Oracle alpha on test", oracle["alpha"]),
        ]
    )

    text_acc = accuracy_for_alpha(0.0, text, visual, eval_features, eval_labels, device)
    rows = []
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


def build_table(aggregate, dataset_name):
    table = Table(title=f"{dataset_name} one-shot centroid-mix ablation")
    table.add_column("One-shot calibration strategy")
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
    parser = argparse.ArgumentParser(description="Table B: ProtoFuse one-shot centroid-mix ablation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
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

    seeds = parse_int_list(args.seeds)
    alpha_steps = int(get_config_value(config, "model.alpha_steps", 101))
    beta_values = parse_float_list(args.betas, get_config_value(config, "model.centroid_mix.beta_values", DEFAULT_BETAS))
    force_loo_accuracy = ProtoFuse._coerce_bool(get_config_value(config, "model.force_loo_accuracy", True), True)
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
        f"seeds={seeds}, force_loo_accuracy={force_loo_accuracy}, device={device}"
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
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("running", total=len(seeds))
        for seed in seeds:
            progress.update(task, description=f"1-shot | seed {seed}")
            set_global_seed(seed)
            if val_fraction is None:
                train_idx = support_positions(train_labels_all, 1, seed)
                support_features = train_features_all[train_idx].contiguous()
                support_labels = train_labels_all[train_idx].contiguous()
                eval_features = eval_features_all
                eval_labels = eval_labels_all
            else:
                train_idx, val_idx = split_positions_by_class(train_labels_all, 1, seed, val_fraction)
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
                force_loo_accuracy,
            )
            for row in rows:
                raw.append({"dataset": dataset_name, "kshot": 1, "seed": int(seed), **row})
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
            console.print(f"{row['strategy']} & {row['accuracy_mean']:.2f} $\\pm$ {row['accuracy_std']:.2f} \\\\")

    if args.output_json:
        out_path = resolve_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "ablation": "oneshot_centroid_mix",
                    "dataset": dataset_name,
                    "dataset_root": str(dataset_root),
                    "seeds": seeds,
                    "beta_values": beta_values,
                    "force_loo_accuracy": force_loo_accuracy,
                    "raw": raw,
                    "aggregate": aggregate,
                },
                f,
                indent=2,
            )
        console.print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
