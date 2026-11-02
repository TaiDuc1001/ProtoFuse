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


DEFAULT_RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
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


def evaluate_rhos(
    rhos,
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
    for rho in rhos:
        rng_fork.restore()
        selection = ProtoFuse.posthoc_fuse(
            text_features,
            support_features,
            support_labels,
            device,
            alpha_steps=alpha_steps,
            beta_values=beta_values,
            rho=rho,
            visual_centroids=base_visual,
            fused_prototypes_init=base_fused,
            alpha_init=base_alpha_init,
        )
        name = f"rho={rho:.2f}"
        variants.append((name, selection["raw_fused_prototypes"]))
        metadata[name] = {
            "rho": float(rho),
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
    parser = argparse.ArgumentParser(description="ProtoFuse rho sensitivity.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--rhos",
        default=",".join(str(value) for value in DEFAULT_RHOS),
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
        default="outputs/protofuse/protofuse_rho.png",
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
    rhos = parse_float_list(args.rhos, DEFAULT_RHOS)
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
        f"kshots={kshots}, rhos={rhos}, seeds={seeds}, device={device}",
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

            rows = evaluate_rhos(
                rhos,
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
                rho_val = float(row["rho"])
                acc = float(row["accuracy"])
                print(
                    f"{table_name} x ProtoFuse x seed {seed} x {kshot}-shot x rho={rho_val:.2f} x {acc:.2f}%",
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
    summary_by_rho = []
    shot_columns = [f"{k}-shot" for k in kshots]
    for rho_val in rhos:
        val_rows = [r for r in dataset_rows if np.isclose(r["rho"], rho_val)]
        row_dict = {"rho": f"{rho_val:.2f}"}
        shot_means = []
        for k in kshots:
            k_accs = [r["accuracy"] for r in val_rows if r["kshot"] == k]
            k_mean = float(np.mean(k_accs)) if k_accs else 0.0
            k_std = float(np.std(k_accs)) if k_accs else 0.0
            row_dict[f"{k}-shot"] = f"{k_mean:.2f} ± {k_std:.2f}"
            shot_means.append(k_mean)
        row_dict["Avg."] = f"{np.mean(shot_means):.2f}" if shot_means else "N/A"
        summary_by_rho.append(row_dict)

    print(f"\n{table_name} x ProtoFuse (rho) x seed mean +/- std", flush=True)
    table_cols = ["rho", *shot_columns]
    if len(kshots) > 1:
        table_cols.append("Avg.")
    print(format_table(summary_by_rho, table_cols), flush=True)

    return raw, str(dataset_root)


def aggregate_results(raw):
    frame = pd.DataFrame(raw)
    seed_macros = (
        frame.groupby(["rho", "dataset", "seed"], as_index=False)["accuracy"]
        .mean()
    )
    summary = (
        seed_macros.groupby(["rho", "dataset"], as_index=False)["accuracy"]
        .agg(mean="mean", std="std", runs="count")
        .sort_values(["rho", "dataset"])
    )
    summary["std"] = summary["std"].fillna(0.0)
    return frame, summary


def build_final_table(summary, rhos, datasets):
    rows = []
    for rho in rhos:
        row = {"rho": f"{rho:.2f}"}
        rho_rows = summary[np.isclose(summary["rho"], rho)]
        means = []
        for dataset in datasets:
            matching = rho_rows[rho_rows["dataset"] == dataset]
            if len(matching) > 0:
                result = matching.iloc[0]
                row[dataset] = f"{result['mean']:.2f} ± {result['std']:.2f}"
                means.append(float(result["mean"]))
            else:
                row[dataset] = "N/A"
        row["Avg."] = f"{np.mean(means):.2f}" if means else "N/A"
        rows.append(row)
    return pd.DataFrame(rows, columns=["rho", *datasets, "Avg."])


def save_bar_chart(summary, rhos, datasets, output_path):
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
    rho_labels = [rf"$\rho={rho:.2f}$" for rho in rhos]
    plot_data = summary.copy()
    plot_data["Dataset"] = plot_data["dataset"].map(
        dict(zip(datasets, display_names))
    )
    plot_data["Rho"] = plot_data["rho"].map(
        dict(zip(rhos, rho_labels))
    )

    figure, axis = plt.subplots(
        figsize=(max(9.5, 1.55 * len(datasets)), 4.8),
    )
    sns.barplot(
        data=plot_data,
        x="Dataset",
        y="mean",
        hue="Rho",
        order=display_names,
        hue_order=rho_labels,
        palette=sns.color_palette("colorblind", n_colors=len(rhos)),
        errorbar=None,
        saturation=0.9,
        edgecolor="white",
        linewidth=0.5,
        ax=axis,
    )

    for container, rho in zip(axis.containers, rhos):
        rho_rows = summary[np.isclose(summary["rho"], rho)].set_index("dataset")
        stds = [float(rho_rows.loc[dataset, "std"]) for dataset in datasets]
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
        ncol=len(rhos),
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
    rhos = parse_float_list(args.rhos, DEFAULT_RHOS)
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
    final_table = build_final_table(summary, rhos, dataset_names)
    shot_desc = f"macro-avg across {len(kshots)} shots: {','.join(str(k) for k in kshots)}" if len(kshots) > 1 else f"{kshots[0]}-shot"
    print(f"\nProtoFuse rho sensitivity: seed mean ± std ({shot_desc})", flush=True)
    print(final_table.to_string(index=False), flush=True)

    figure_path = resolve_path(args.output_figure)
    save_bar_chart(summary, rhos, dataset_names, figure_path)
    print(f"\nSaved figure to {figure_path}", flush=True)

    if args.output_json:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(
                {
                    "experiment": "protofuse_rho",
                    "datasets": dataset_roots,
                    "kshots": kshots,
                    "rhos": rhos,
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
