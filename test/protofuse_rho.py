import argparse
import gc
import json
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
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protofuse_ablation_tables import (
    batched_variant_accuracies,
    table_dataset_configs,
)
from protofuse_candidates import (
    DEFAULT_CONFIG,
    build_text_features,
    extract_image_features,
    load_datasets,
    load_model,
    parse_float_list,
    parse_int_list,
    resolve_path,
    split_positions_by_class,
    support_positions,
)
from src.models.protofuse import ProtoFuse
from utils import (
    get_config_value,
    load_config_file,
    merge_configs,
    parse_override_arguments,
    set_global_seed,
)


DEFAULT_RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
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
    variants = []
    metadata = {}
    for rho in rhos:
        selection = ProtoFuse.posthoc_fuse(
            text_features,
            support_features,
            support_labels,
            device,
            alpha_steps=alpha_steps,
            beta_values=beta_values,
            rho=rho,
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


def run_dataset(args, table_name, config, console):
    rhos = parse_float_list(args.rhos, DEFAULT_RHOS)
    seeds = parse_int_list(args.seeds)
    if len(rhos) != 5:
        raise ValueError(f"Expected exactly 5 rho values, found {len(rhos)}.")
    if len(seeds) != 5:
        raise ValueError(f"Expected exactly 5 seeds, found {len(seeds)}.")

    kshot = int(get_config_value(config, "data.kshot", 16))
    alpha_steps = int(get_config_value(config, "model.alpha_steps", 101))
    beta_values = get_config_value(config, "model.centroid_mix.beta_values", None)
    device_name = str(get_config_value(config, "training.device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    batch_size = int(get_config_value(config, "training.batch_size", 128))
    eval_batch_size = args.eval_batch_size or max(batch_size, 1024)
    num_workers = int(get_config_value(config, "data.num_workers", 4))
    dataset_name = str(get_config_value(config, "data.dataset_name", table_name))

    set_global_seed(1)
    train_dataset, eval_dataset, val_fraction, dataset_root = load_datasets(config)
    classnames = list(train_dataset.classes)
    console.print(
        f"{table_name}: root={dataset_root}, classes={len(classnames)}, "
        f"kshot={kshot}, rhos={rhos}, seeds={seeds}, device={device}"
    )

    model = load_model(config, device)
    train_features_all, train_labels_all = extract_image_features(
        model,
        train_dataset,
        device,
        batch_size,
        num_workers,
    )
    if eval_dataset is None:
        eval_features_all, eval_labels_all = train_features_all, train_labels_all
    else:
        eval_features_all, eval_labels_all = extract_image_features(
            model,
            eval_dataset,
            device,
            batch_size,
            num_workers,
        )
    text_features = build_text_features(model, classnames, dataset_name, device)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw = []
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ]
    with Progress(*progress_columns, console=console) as progress:
        task = progress.add_task(f"{table_name}: rho sweep", total=len(seeds))
        for seed in seeds:
            progress.update(
                task,
                description=f"{table_name}: {kshot}-shot, seed {seed}",
            )
            set_global_seed(seed)
            if val_fraction is None:
                train_idx = support_positions(train_labels_all, kshot, seed)
                eval_features = eval_features_all
                eval_labels = eval_labels_all
            else:
                train_idx, val_idx = split_positions_by_class(
                    train_labels_all,
                    kshot,
                    seed,
                    val_fraction,
                )
                eval_features = train_features_all[val_idx].contiguous()
                eval_labels = train_labels_all[val_idx].contiguous()

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
            raw.extend(
                {
                    "dataset": table_name,
                    "kshot": kshot,
                    "seed": int(seed),
                    **row,
                }
                for row in rows
            )
            progress.advance(task)
    return raw, str(dataset_root)


def aggregate_results(raw):
    frame = pd.DataFrame(raw)
    summary = (
        frame.groupby(["rho", "dataset"], as_index=False)["accuracy"]
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
            result = rho_rows[rho_rows["dataset"] == dataset].iloc[0]
            row[dataset] = f"{result['mean']:.2f} ± {result['std']:.2f}"
            means.append(float(result["mean"]))
        row["Avg."] = f"{np.mean(means):.2f}"
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
    console = Console(no_color=args.disable_coloring)
    rhos = parse_float_list(args.rhos, DEFAULT_RHOS)

    raw = []
    dataset_roots = {}
    dataset_configs = table_dataset_configs(config)
    dataset_names = [table_name for table_name, _, _ in dataset_configs]
    for table_name, dataset_config, _ in dataset_configs:
        dataset_raw, dataset_root = run_dataset(
            args,
            table_name,
            dataset_config,
            console,
        )
        raw.extend(dataset_raw)
        dataset_roots[table_name] = dataset_root

    _, summary = aggregate_results(raw)
    final_table = build_final_table(summary, rhos, dataset_names)
    console.print()
    console.print("ProtoFuse rho sensitivity: seed mean ± std")
    console.print(final_table.to_string(index=False))

    figure_path = resolve_path(args.output_figure)
    save_bar_chart(summary, rhos, dataset_names, figure_path)
    console.print(f"Saved figure to {figure_path}")

    if args.output_json:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(
                {
                    "experiment": "protofuse_rho",
                    "datasets": dataset_roots,
                    "kshot": int(get_config_value(config, "data.kshot", 16)),
                    "rhos": rhos,
                    "seeds": parse_int_list(args.seeds),
                    "raw": raw,
                    "summary": summary.to_dict(orient="records"),
                },
                handle,
                indent=2,
            )
        console.print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
