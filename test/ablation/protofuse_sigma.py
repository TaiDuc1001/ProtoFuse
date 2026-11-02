import argparse
import copy
import gc
import json
import math
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

from protofuse import split_indices_for
from protofuse_ablation_tables import (
    ForkRNG,
    batched_variant_accuracies,
    table_dataset_configs,
)
from protofuse_candidates import (
    DEFAULT_CONFIG,
    parse_float_list,
    parse_int_list,
    resolve_path,
)
from src.models.protofuse import ProtoFuse
from src.pipelines.protofuse import ProtoFusePipeline
from utils import (
    get_config_value,
    load_config_file,
    merge_configs,
    parse_override_arguments,
    set_global_seed,
)


DEFAULT_SIGMAS = [0.0, 0.01, 0.02, 0.03, 0.05]
DEFAULT_KSHOTS = [1, 2, 4, 8, 16]
DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]
DATASET_DISPLAY_NAMES = {
    "FGVC-Aircraft": "Aircraft",
    "Stanford Cars": "Cars",
    "CUB-200-2011": "CUB-200",
    "Flowers102": "Flowers",
    "Food-101": "Food-101",
    "OxfordPets": "Oxford Pets",
}


def custom_posthoc_fuse(
    text_prototypes,
    train_features,
    train_labels,
    device,
    alpha_steps=101,
    beta_values=None,
    l0=5,
    r=2,
    sigma=0.02,
    gamma_max=0.45,
    lambda_gamma=0.30,
    lambda_rho=0.50,
    visual_centroids=None,
    fused_prototypes_init=None,
    alpha_init=None,
):
    selector = ProtoFuse.from_precomputed(
        text_prototypes,
        device,
        alpha_steps=alpha_steps,
        beta_values=beta_values,
    )

    train_features = F.normalize(train_features.to(selector.device).float(), dim=-1)
    train_labels = train_labels.to(selector.device).long()
    num_classes = selector.text_prototypes.shape[0]
    class_counts = torch.bincount(train_labels, minlength=num_classes)
    is_one_shot = bool(class_counts.numel() == num_classes and class_counts.eq(1).all().item())

    if visual_centroids is None:
        visual_centroids = selector.build_visual_centroids(train_features, train_labels, num_classes)
    if fused_prototypes_init is None or alpha_init is None:
        fused_prototypes_init, alpha_init = selector.hopc_alpha(
            selector.text_prototypes,
            visual_centroids,
            train_features,
            train_labels,
            num_classes,
        )
    fused_prototypes = fused_prototypes_init
    kshot = int(class_counts.min().item())
    if kshot < 1:
        kshot = 1
    gamma_k = min(gamma_max, lambda_gamma / math.sqrt(kshot))
    rho_k = min(1.0, lambda_rho / math.sqrt(kshot))
    selector.rho = rho_k

    Q = selector.generate_surrogate_mix(
        visual_centroids,
        fused_prototypes,
        gamma_k,
        l0=l0,
        samples_per_nb=r,
        sigma=sigma,
        filter_correct=True,
    )
    query_centroids = selector.pseudo_label_aggregation(
        Q.view(-1, Q.shape[-1]),
        selector.text_prototypes,
        visual_centroids,
        alpha_init,
    )
    expanded_centroids = F.normalize((1.0 - rho_k) * visual_centroids + rho_k * query_centroids, dim=-1)

    if is_one_shot:
        alpha = alpha_init
        alpha_final = alpha_init
        selected_candidate = "cgse_fixed"
    else:
        _, alpha_final = selector.hopc_alpha(
            selector.text_prototypes,
            expanded_centroids,
            train_features,
            train_labels,
            num_classes,
            query_centroids=query_centroids,
        )
        selected, scored = selector.select_fallback_candidate(
            selector.text_prototypes,
            visual_centroids,
            expanded_centroids,
            alpha_init,
            alpha_final,
            train_features,
            train_labels,
            num_classes,
            query_centroids,
        )
        selected_candidate, expanded_centroids, alpha, _ = selected

    fused_prototypes = F.normalize(
        (1.0 - alpha) * selector.text_prototypes
        + alpha * expanded_centroids,
        dim=-1,
    )

    centroid_mask = torch.zeros(num_classes, device=selector.device, dtype=torch.bool)
    valid_labels = train_labels[(train_labels >= 0) & (train_labels < num_classes)]
    if valid_labels.numel() > 0:
        centroid_mask[valid_labels.unique()] = True
    fused_for_inference = fused_prototypes.clone()
    fused_for_inference[~centroid_mask] = selector.text_prototypes[~centroid_mask]

    return {
        "fused_prototypes": fused_for_inference,
        "raw_fused_prototypes": fused_prototypes,
        "alpha": alpha,
        "selected_candidate": selected_candidate,
    }


def evaluate_sigmas(
    sigmas,
    text_features,
    support_features,
    support_labels,
    eval_features,
    eval_labels,
    device,
    alpha_steps,
    beta_values,
    eval_batch_size,
):
    selector = ProtoFuse.from_precomputed(
        text_features,
        device,
        alpha_steps=alpha_steps,
        beta_values=beta_values,
    )
    support_features = F.normalize(support_features.to(device).float(), dim=-1)
    support_labels = support_labels.to(device).long()
    num_classes = selector.text_prototypes.shape[0]
    base_visual = selector.build_visual_centroids(support_features, support_labels, num_classes)
    base_fused, base_alpha_init = selector.hopc_alpha(
        selector.text_prototypes,
        base_visual,
        support_features,
        support_labels,
        num_classes,
    )

    variants = []
    metadata = {}
    rng_fork = ForkRNG(device=device)
    for sigma in sigmas:
        rng_fork.restore()
        selection = custom_posthoc_fuse(
            text_features,
            support_features,
            support_labels,
            device,
            alpha_steps=alpha_steps,
            beta_values=beta_values,
            sigma=sigma,
            visual_centroids=base_visual,
            fused_prototypes_init=base_fused,
            alpha_init=base_alpha_init,
        )
        name = f"sigma={sigma}"
        variants.append((name, selection["raw_fused_prototypes"]))
        metadata[name] = {
            "sigma": float(sigma),
            "alpha": float(selection["alpha"]),
            "selected_candidate": selection["selected_candidate"],
        }

    accuracies = batched_variant_accuracies(
        variants,
        eval_features,
        eval_labels,
        device,
        eval_batch_size,
    )
    return [
        {
            **metadata[name],
            "accuracy": float(accuracies[name]),
        }
        for name, _ in variants
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="ProtoFuse sigma sensitivity.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--sigmas",
        default=",".join(str(value) for value in DEFAULT_SIGMAS),
    )
    parser.add_argument(
        "--kshots",
        default=",".join(str(value) for value in DEFAULT_KSHOTS),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_SEEDS),
    )
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--output-figure",
        default="outputs/protofuse/protofuse_sigma.png",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--disable-coloring", action="store_true")
    parsed, unknown = parser.parse_known_args()
    return parsed, parse_override_arguments(unknown)


def format_table(rows, columns):
    if not rows:
        return " ".join(columns)
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    lines = [
        "  ".join(column.ljust(widths[column]) for column in columns),
        "  ".join("-" * widths[column] for column in columns),
    ]
    for row in rows:
        lines.append("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))
    return "\n".join(lines)


def run_dataset(args, table_name, config):
    sigmas = parse_float_list(args.sigmas, DEFAULT_SIGMAS)
    seeds = parse_int_list(args.seeds)
    kshots = parse_int_list(args.kshots)

    alpha_steps = int(get_config_value(config, "model.alpha_steps", 101))
    beta_values = get_config_value(config, "model.centroid_mix.beta_values", None)
    batch_size = int(get_config_value(config, "training.batch_size", 128))
    eval_batch_size = args.eval_batch_size or max(batch_size, 1024)

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

    dataset_root = pipeline.dataset_root
    classnames = pipeline.classnames
    device = pipeline.device
    print(
        f"\n{table_name}: root={dataset_root}, classes={len(classnames)}, "
        f"kshots={kshots}, sigma={sigmas}, seeds={seeds}, device={device}",
        flush=True,
    )

    train_payload = pipeline._full_dataset_clip_features()
    train_features_all = train_payload["image_features"]
    train_labels_all = train_payload["labels"]
    text_features = train_payload["text_features"]

    if pipeline.val_fraction is None:
        eval_features_all, eval_labels_all = pipeline._cached_val_features()
    else:
        eval_features_all, eval_labels_all = train_features_all, train_labels_all

    if hasattr(pipeline.trainer, "clip_model"):
        del pipeline.trainer.clip_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw = []
    print(flush=True)
    for kshot in kshots:
        for seed in seeds:
            set_global_seed(seed)
            train_indices, val_indices = split_indices_for(pipeline, kshot, seed)
            train_idx = torch.tensor(train_indices, dtype=torch.long)

            if pipeline.val_fraction is None:
                eval_features = eval_features_all
                eval_labels = eval_labels_all
            else:
                val_idx = torch.tensor(val_indices, dtype=torch.long)
                eval_features = eval_features_all[val_idx].contiguous()
                eval_labels = eval_labels_all[val_idx].contiguous()

            rows = evaluate_sigmas(
                sigmas,
                text_features,
                train_features_all[train_idx].contiguous(),
                train_labels_all[train_idx].contiguous(),
                eval_features,
                eval_labels,
                device,
                alpha_steps,
                beta_values,
                eval_batch_size,
            )
            for row in rows:
                sigma_val = float(row["sigma"])
                acc = float(row["accuracy"])
                print(
                    f"{table_name} x ProtoFuse x seed {seed} x {kshot}-shot x sigma={sigma_val} x {acc:.2f}%",
                    flush=True,
                )
            raw.extend(
                {
                    "dataset": table_name,
                    "kshot": int(kshot),
                    "seed": int(seed),
                    **row,
                }
                for row in rows
            )

    dataset_rows = [r for r in raw if r["dataset"] == table_name]
    summary_by_sigma = []
    shot_columns = [f"{k}-shot" for k in kshots]
    for sigma_val in sigmas:
        val_rows = [r for r in dataset_rows if np.isclose(r["sigma"], sigma_val)]
        row_dict = {"sigma": str(sigma_val)}
        shot_means = []
        for k in kshots:
            k_accs = [r["accuracy"] for r in val_rows if r["kshot"] == k]
            k_mean = float(np.mean(k_accs)) if k_accs else 0.0
            k_std = float(np.std(k_accs)) if k_accs else 0.0
            row_dict[f"{k}-shot"] = f"{k_mean:.2f} ± {k_std:.2f}"
            shot_means.append(k_mean)
        row_dict["Avg."] = f"{np.mean(shot_means):.2f}" if shot_means else "N/A"
        summary_by_sigma.append(row_dict)

    print(f"\n{table_name} x ProtoFuse (sigma) x seed mean +/- std", flush=True)
    table_cols = ["sigma", *shot_columns]
    if len(kshots) > 1:
        table_cols.append("Avg.")
    print(format_table(summary_by_sigma, table_cols), flush=True)

    return raw, str(dataset_root)


def aggregate_results(raw):
    frame = pd.DataFrame(raw)
    seed_macros = (
        frame.groupby(["sigma", "dataset", "seed"], as_index=False)["accuracy"]
        .mean()
    )
    summary = (
        seed_macros.groupby(["sigma", "dataset"], as_index=False)["accuracy"]
        .agg(mean="mean", std="std", runs="count")
        .sort_values(["sigma", "dataset"])
    )
    summary["std"] = summary["std"].fillna(0.0)
    return frame, summary


def build_final_table(summary, sigmas, datasets):
    rows = []
    for sigma in sigmas:
        row = {"sigma": f"{sigma}"}
        sigma_rows = summary[np.isclose(summary["sigma"], sigma)]
        means = []
        for dataset in datasets:
            matching = sigma_rows[sigma_rows["dataset"] == dataset]
            if len(matching) > 0:
                result = matching.iloc[0]
                row[dataset] = f"{result['mean']:.2f} ± {result['std']:.2f}"
                means.append(float(result["mean"]))
            else:
                row[dataset] = "N/A"
        row["Avg."] = f"{np.mean(means):.2f}" if means else "N/A"
        rows.append(row)
    return pd.DataFrame(rows, columns=["sigma", *datasets, "Avg."])


def save_bar_chart(summary, sigmas, datasets, output_path):
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.25,
        rc={
            "axes.edgecolor": "0.2",
            "axes.linewidth": 0.9,
            "grid.color": "0.88",
            "grid.linewidth": 0.8,
        },
    )
    display_names = [
        DATASET_DISPLAY_NAMES.get(dataset, dataset)
        for dataset in datasets
    ]
    sigma_labels = [rf"$\sigma={sigma}$" for sigma in sigmas]
    plot_data = summary.copy()
    plot_data["Dataset"] = plot_data["dataset"].map(
        dict(zip(datasets, display_names))
    )
    plot_data["Sigma"] = plot_data["sigma"].map(
        lambda s: next(label for val, label in zip(sigmas, sigma_labels) if np.isclose(s, val))
    )

    figure, axis = plt.subplots(
        figsize=(max(9.5, 1.55 * len(datasets)), 4.8),
    )
    sns.barplot(
        data=plot_data,
        x="Dataset",
        y="mean",
        hue="Sigma",
        order=display_names,
        hue_order=sigma_labels,
        palette=sns.color_palette("colorblind", n_colors=len(sigmas)),
        errorbar=None,
        saturation=0.9,
        edgecolor="white",
        linewidth=0.5,
        ax=axis,
    )

    for container, sigma in zip(axis.containers, sigmas):
        sigma_rows = summary[np.isclose(summary["sigma"], sigma)].set_index("dataset")
        stds = [float(sigma_rows.loc[dataset, "std"]) if dataset in sigma_rows.index else 0.0 for dataset in datasets]
        centers = [
            bar.get_x() + bar.get_width() / 2.0
            for bar in container
        ]
        heights = [bar.get_height() for bar in container]
        axis.errorbar(
            centers,
            heights,
            yerr=stds,
            fmt="none",
            ecolor="0.15",
            elinewidth=0.9,
            capsize=2,
            capthick=0.9,
        )

    axis.set_xlabel("")
    axis.set_ylabel("Top-1 accuracy (%)", labelpad=8)
    axis.set_ylim(0, 100)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.tick_params(axis="x", rotation=0, pad=6)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    axis.legend(
        title=None,
        ncol=len(sigmas),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        columnspacing=1.6,
        handlelength=1.8,
    )
    sns.despine(ax=axis)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    args, overrides = parse_args()
    config = merge_configs(load_config_file(args.config), overrides)
    sigmas = parse_float_list(args.sigmas, DEFAULT_SIGMAS)
    kshots = parse_int_list(args.kshots)

    raw = []
    dataset_roots = {}
    dataset_configs = table_dataset_configs(config)
    dataset_names = [table_name for table_name, _, _ in dataset_configs]
    for table_name, dataset_config, _ in dataset_configs:
        dataset_raw, dataset_root = run_dataset(
            args,
            table_name,
            dataset_config,
        )
        raw.extend(dataset_raw)
        dataset_roots[table_name] = dataset_root

    _, summary = aggregate_results(raw)
    final_table = build_final_table(summary, sigmas, dataset_names)
    shot_desc = f"macro-avg across {len(kshots)} shots: {','.join(str(k) for k in kshots)}" if len(kshots) > 1 else f"{kshots[0]}-shot"
    print(f"\nProtoFuse sigma sensitivity: seed mean ± std ({shot_desc})", flush=True)
    print(final_table.to_string(index=False), flush=True)

    figure_path = resolve_path(args.output_figure)
    save_bar_chart(summary, sigmas, dataset_names, figure_path)
    print(f"\nSaved figure to {figure_path}", flush=True)

    if args.output_json:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(
                {
                    "experiment": "protofuse_sigma",
                    "datasets": dataset_roots,
                    "kshots": kshots,
                    "sigmas": sigmas,
                    "seeds": parse_int_list(args.seeds),
                    "raw": raw,
                    "summary": summary.to_dict(orient="records"),
                },
                handle,
                indent=2,
            )
        print(f"Saved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
