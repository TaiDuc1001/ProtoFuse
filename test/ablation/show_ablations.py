#!/usr/bin/env python3
"""
Render hyperparameter sensitivity figure for ProtoFuse.
3 rows x 2 columns line charts showing accuracy across six datasets and five shot settings.

Supports:
- Relative accuracy changes in percentage points (--metric delta, default)
- Absolute accuracy in percentage (--metric absolute)
- Shared Y-axis across all 6 subplots (--shared-y) or individual auto-scaling
- Legend placement (--legend-loc {upper_left, bottom, none})
- Optional shaded standard deviation bands across random seeds (--show-std)
- Export individual 300 DPI figures (--export-individual)
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "test" / "assets" / "hyperparam_ablation_results.txt"

# Hyperparameter configurations: (key, filename_stem, title, default_val, is_categorical, values, ticklabels, xlabel, pos)
CONFIGS = [
    {
        "key": "L_0",
        "stem": "l0",
        "title": r"(a) $L_0$",
        "default": 5.0,
        "categorical": True,
        "values": [1.0, 3.0, 5.0, 7.0, 10.0],
        "ticklabels": ["1", "3", "5", "7", "10"],
        "xlabel": r"Number of Neighbors ($L_0$)",
        "pos": (0, 0),
    },
    {
        "key": "R",
        "stem": "r",
        "title": r"(b) $R$",
        "default": 2.0,
        "categorical": True,
        "values": [1.0, 2.0, 4.0, 8.0, 16.0],
        "ticklabels": ["1", "2", "4", "8", "16"],
        "xlabel": r"Candidates per Neighbor ($R$)",
        "pos": (0, 1),
    },
    {
        "key": "sigma",
        "stem": "sigma",
        "title": r"(c) $\sigma$",
        "default": 0.02,
        "categorical": False,
        "values": [0.0, 0.01, 0.02, 0.03, 0.05],
        "ticklabels": ["0.00", "0.01", "0.02", "0.03", "0.05"],
        "xlabel": r"Noise Scale ($\sigma$)",
        "pos": (1, 0),
    },
    {
        "key": "gamma_max",
        "stem": "gamma_max",
        "title": r"(d) $\gamma_{\max}$",
        "default": 0.45,
        "categorical": False,
        "values": [0.05, 0.10, 0.15, 0.20, 0.30, 0.45],
        "ticklabels": ["0.05", "0.10", "0.15", "0.20", "0.30", "0.45"],
        "xlabel": r"Maximum Perturbation Bound ($\gamma_{\max}$)",
        "pos": (1, 1),
    },
    {
        "key": "lambda_gamma",
        "stem": "lambda_gamma",
        "title": r"(e) $\lambda_\gamma$",
        "default": 0.30,
        "categorical": False,
        "values": [0.10, 0.20, 0.30, 0.40, 0.50],
        "ticklabels": ["0.10", "0.20", "0.30", "0.40", "0.50"],
        "xlabel": r"Perturbation Scaling Factor ($\lambda_\gamma$)",
        "pos": (2, 0),
    },
    {
        "key": "lambda_rho",
        "stem": "lambda_rho",
        "title": r"(f) $\lambda_\rho$",
        "default": 0.50,
        "categorical": False,
        "values": [0.10, 0.25, 0.50, 0.75, 1.00],
        "ticklabels": ["0.10", "0.25", "0.50", "0.75", "1.00"],
        "xlabel": r"Centroid Mixing Factor ($\lambda_\rho$)",
        "pos": (2, 1),
    },
]

SHOTS = [1, 2, 4, 8, 16]
SHOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
SHOT_MARKERS = ["o", "s", "^", "D", "v"]


def parse_results(file_path: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"^(?P<dataset>.+?)\s+x\s+ProtoFuse\s+x\s+seed\s+(?P<seed>\d+)\s+x\s+(?P<kshot>\d+)-shot\s+x\s+(?P<param>[^=]+)=(?P<val>[^\s]+)\s+x\s+(?P<acc>[0-9.]+)%"
    )
    records = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line.strip())
            if match:
                records.append(
                    {
                        "dataset": match.group("dataset").strip(),
                        "seed": int(match.group("seed")),
                        "kshot": int(match.group("kshot")),
                        "param": match.group("param").strip(),
                        "val": float(match.group("val")),
                        "acc": float(match.group("acc")),
                    }
                )
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid records found in {file_path}")
    return df


def compute_statistics(df: pd.DataFrame, cfg: dict, metric: str = "delta") -> pd.DataFrame:
    """
    Computes mean and std across the 5 random seeds for either 'delta' or 'absolute'.
    """
    param_key = cfg["key"]
    def_val = cfg["default"]

    sub = df[df["param"] == param_key].copy()
    if sub.empty:
        raise ValueError(f"No records found for param {param_key}")

    if metric == "delta":
        base = sub[np.isclose(sub["val"], def_val)][
            ["dataset", "kshot", "seed", "acc"]
        ].rename(columns={"acc": "base_acc"})

        merged = pd.merge(sub, base, on=["dataset", "kshot", "seed"])
        merged["target_metric"] = merged["acc"] - merged["base_acc"]
    else:
        merged = sub.copy()
        merged["target_metric"] = merged["acc"]

    # Macro-average across the 6 datasets for each seed
    seed_macro = (
        merged.groupby(["kshot", "seed", "val"])["target_metric"]
        .mean()
        .reset_index()
    )

    # Compute mean and sample std across the 5 seeds
    stats = (
        seed_macro.groupby(["kshot", "val"])["target_metric"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)
    return stats


def determine_global_y_limits(
    df: pd.DataFrame,
    metric: str = "delta",
    show_std: bool = False,
    symmetric: bool = True,
    step: float = None,
):
    all_vals = []
    for cfg in CONFIGS:
        stats = compute_statistics(df, cfg, metric=metric)
        if show_std:
            all_vals.extend((stats["mean"] - stats["std"]).tolist())
            all_vals.extend((stats["mean"] + stats["std"]).tolist())
        else:
            all_vals.extend(stats["mean"].tolist())

    data_min = min(all_vals)
    data_max = max(all_vals)

    if metric == "delta":
        step = step or 0.5
        if symmetric:
            max_abs = max(abs(data_min), abs(data_max))
            bound = math.ceil(max_abs / step) * step
            if bound - max_abs < 0.05 * step:
                bound += step
            y_min, y_max = -bound, bound
        else:
            y_min = math.floor(data_min / step) * step
            y_max = math.ceil(data_max / step) * step
    else:
        # absolute metric
        step = step or 2.0
        y_min = math.floor((data_min - 0.5) / step) * step
        y_max = math.ceil((data_max + 0.5) / step) * step

    n_ticks = int(round((y_max - y_min) / step)) + 1
    ticks = np.linspace(y_min, y_max, n_ticks).tolist()
    return (y_min - (0.25 if metric == "absolute" else 0.0), y_max + (0.25 if metric == "absolute" else 0.0)), ticks


def plot_sensitivity(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
    metric: str = "delta",
    show_std: bool = False,
    std_alpha: float = 0.12,
    shared_y: bool = False,
    legend_loc: str = "upper_left",
    y_min: float = None,
    y_max: float = None,
    y_step: float = None,
):
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "grid.alpha": 0.55,
            "grid.linestyle": ":",
        }
    )

    fig_height = 11.5 if legend_loc == "upper_left" else 12.5
    fig, axes = plt.subplots(3, 2, figsize=(10.0, fig_height), dpi=dpi)

    handles = []
    labels = []

    if shared_y:
        if y_min is not None and y_max is not None:
            global_ylim = (y_min, y_max)
            step = y_step or (0.5 if metric == "delta" else 2.0)
            n_ticks = int(round((y_max - y_min) / step)) + 1
            global_yticks = np.linspace(y_min, y_max, n_ticks).tolist()
        else:
            global_ylim, global_yticks = determine_global_y_limits(
                df,
                metric=metric,
                show_std=show_std,
                symmetric=(metric == "delta"),
                step=y_step,
            )

    for cfg in CONFIGS:
        row, col = cfg["pos"]
        ax = axes[row, col]
        stats_df = compute_statistics(df, cfg, metric=metric)

        # Reference horizontal line at y=0 (only relevant for delta metric)
        if metric == "delta":
            ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.1, alpha=0.75, zorder=1)

        # Reference vertical dashed line at default hyperparameter value
        if cfg["categorical"]:
            def_idx = cfg["values"].index(cfg["default"])
            ax.axvline(def_idx, color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.85, zorder=1)
        else:
            ax.axvline(cfg["default"], color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.85, zorder=1)

        for s_idx, kshot in enumerate(SHOTS):
            shot_data = stats_df[stats_df["kshot"] == kshot].sort_values("val")

            if cfg["categorical"]:
                # Categorical equidistant x coordinates
                val_to_pos = {val: i for i, val in enumerate(cfg["values"])}
                x_coords = [val_to_pos[v] for v in shot_data["val"]]
            else:
                x_coords = shot_data["val"].tolist()

            y_coords = shot_data["mean"].values
            std_coords = shot_data["std"].values

            line, = ax.plot(
                x_coords,
                y_coords,
                color=SHOT_COLORS[s_idx],
                marker=SHOT_MARKERS[s_idx],
                markersize=6.5,
                linewidth=1.9,
                label=f"{kshot}-shot",
                zorder=3,
            )

            if show_std:
                ax.fill_between(
                    x_coords,
                    y_coords - std_coords,
                    y_coords + std_coords,
                    color=SHOT_COLORS[s_idx],
                    alpha=std_alpha,
                    zorder=2,
                )

            if len(handles) < len(SHOTS):
                handles.append(line)
                labels.append(f"{kshot}-shot")

        # Titles and axis labels
        ax.set_title(cfg["title"], fontweight="bold", pad=8)
        ax.set_xlabel(cfg["xlabel"], labelpad=7)
        y_label_str = r"$\Delta$ Top-1 Accuracy (pp)" if metric == "delta" else "Top-1 Accuracy (%)"
        ax.set_ylabel(y_label_str, labelpad=7)

        # Set ticks and bounds
        if cfg["categorical"]:
            ax.set_xticks(range(len(cfg["values"])))
            ax.set_xticklabels(cfg["ticklabels"])
            ax.set_xlim(-0.4, len(cfg["values"]) - 0.6)
        else:
            ax.set_xticks(cfg["values"])
            ax.set_xticklabels(cfg["ticklabels"])
            span = max(cfg["values"]) - min(cfg["values"])
            ax.set_xlim(min(cfg["values"]) - 0.05 * span, max(cfg["values"]) + 0.05 * span)

        if shared_y:
            ax.set_ylim(global_ylim)
            ax.set_yticks(global_yticks)
        else:
            # Padding for individual y-axis
            y_min_curr, y_max_curr = ax.get_ylim()
            y_margin = max(0.04, (y_max_curr - y_min_curr) * 0.12)
            ax.set_ylim(y_min_curr - y_margin, y_max_curr + y_margin)

        # If individual legend in upper left of each subfigure
        if legend_loc == "upper_left":
            ax.legend(
                loc="upper left",
                fontsize=9,
                frameon=True,
                facecolor="white",
                edgecolor="#cccccc",
                framealpha=0.92,
                handletextpad=0.4,
                labelspacing=0.25,
                borderpad=0.35,
            )

    # Shared bottom legend if requested
    if legend_loc == "bottom":
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=5,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            framealpha=0.95,
            columnspacing=2.5,
            handletextpad=0.6,
            fontsize=12,
        )

    # Adjust layout
    bottom_margin = 0.065 if legend_loc != "bottom" else 0.105
    plt.subplots_adjust(
        left=0.095,
        right=0.965,
        top=0.965,
        bottom=bottom_margin,
        hspace=0.36 if legend_loc != "bottom" else 0.38,
        wspace=0.28,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved figure to {output_path} (dpi={dpi}, metric={metric}, shared_y={shared_y}, legend_loc={legend_loc})")


def export_individual_figures(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
    metric: str = "delta",
    show_std: bool = False,
    std_alpha: float = 0.12,
    shared_y: bool = True,
    y_min: float = None,
    y_max: float = None,
    y_step: float = None,
):
    """
    Export each of the 6 subplots as a standalone 300 DPI PNG file with name format like gamma_max_ablation.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "grid.alpha": 0.55,
            "grid.linestyle": ":",
        }
    )

    if shared_y:
        if y_min is not None and y_max is not None:
            global_ylim = (y_min, y_max)
            step = y_step or (0.5 if metric == "delta" else 2.0)
            n_ticks = int(round((y_max - y_min) / step)) + 1
            global_yticks = np.linspace(y_min, y_max, n_ticks).tolist()
        else:
            global_ylim, global_yticks = determine_global_y_limits(
                df,
                metric=metric,
                show_std=show_std,
                symmetric=(metric == "delta"),
                step=y_step,
            )

    saved_paths = []
    for cfg in CONFIGS:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=dpi)
        stats_df = compute_statistics(df, cfg, metric=metric)

        # Horizontal reference line at y=0 (delta)
        if metric == "delta":
            ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.1, alpha=0.75, zorder=1)

        # Vertical reference dashed line at default value
        if cfg["categorical"]:
            def_idx = cfg["values"].index(cfg["default"])
            ax.axvline(def_idx, color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.85, zorder=1)
        else:
            ax.axvline(cfg["default"], color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.85, zorder=1)

        for s_idx, kshot in enumerate(SHOTS):
            shot_data = stats_df[stats_df["kshot"] == kshot].sort_values("val")

            if cfg["categorical"]:
                val_to_pos = {val: i for i, val in enumerate(cfg["values"])}
                x_coords = [val_to_pos[v] for v in shot_data["val"]]
            else:
                x_coords = shot_data["val"].tolist()

            y_coords = shot_data["mean"].values
            std_coords = shot_data["std"].values

            ax.plot(
                x_coords,
                y_coords,
                color=SHOT_COLORS[s_idx],
                marker=SHOT_MARKERS[s_idx],
                markersize=6.0,
                linewidth=1.8,
                label=f"{kshot}-shot",
                zorder=3,
            )

            if show_std:
                ax.fill_between(
                    x_coords,
                    y_coords - std_coords,
                    y_coords + std_coords,
                    color=SHOT_COLORS[s_idx],
                    alpha=std_alpha,
                    zorder=2,
                )

        clean_title = re.sub(r"^\([a-z]\)\s*", "", cfg["title"])
        ax.set_title(clean_title, fontweight="bold", pad=8)
        # Omit x-axis label for individual figures
        y_label_str = r"$\Delta$ Top-1 Accuracy (pp)" if metric == "delta" else "Top-1 Accuracy (%)"
        ax.set_ylabel(y_label_str, labelpad=7)

        if cfg["categorical"]:
            ax.set_xticks(range(len(cfg["values"])))
            ax.set_xticklabels(cfg["ticklabels"])
            ax.set_xlim(-0.4, len(cfg["values"]) - 0.6)
        else:
            ax.set_xticks(cfg["values"])
            ax.set_xticklabels(cfg["ticklabels"])
            span = max(cfg["values"]) - min(cfg["values"])
            ax.set_xlim(min(cfg["values"]) - 0.05 * span, max(cfg["values"]) + 0.05 * span)

        if shared_y:
            ax.set_ylim(global_ylim)
            ax.set_yticks(global_yticks)
        else:
            y_min_curr, y_max_curr = ax.get_ylim()
            y_margin = max(0.04, (y_max_curr - y_min_curr) * 0.12)
            ax.set_ylim(y_min_curr - y_margin, y_max_curr + y_margin)

        ax.legend(
            loc="upper left",
            fontsize=9,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            framealpha=0.92,
            handletextpad=0.4,
            labelspacing=0.25,
            borderpad=0.35,
        )

        plt.tight_layout()
        file_name = f"{cfg['stem']}_ablation.png"
        out_file = output_dir / file_name
        plt.savefig(out_file, dpi=dpi)
        plt.close(fig)
        saved_paths.append(out_file)
        print(f"Saved individual figure: {out_file} (dpi={dpi})")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Render ProtoFuse hyperparameter sensitivity figure.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to ablation results text file")
    parser.add_argument("--output", default=None, help="Path to output PNG image")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "test" / "assets"), help="Output directory for individual figures")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output PNG image")
    parser.add_argument(
        "--metric",
        choices=["delta", "absolute"],
        default="delta",
        help="Metric to plot: 'delta' (accuracy change in pp) or 'absolute' (Top-1 Accuracy in %)",
    )
    parser.add_argument(
        "--shared-y",
        action="store_true",
        help="Use unified identical Y-axis scale across all 6 subplots",
    )
    parser.add_argument(
        "--legend-loc",
        choices=["upper_left", "bottom", "none"],
        default=None,
        help="Legend location: 'upper_left' (in each subplot), 'bottom' (shared at bottom), or 'none'",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Manual lower bound for shared Y-axis",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Manual upper bound for shared Y-axis",
    )
    parser.add_argument(
        "--y-step",
        type=float,
        default=None,
        help="Step size for ticks on shared Y-axis (default: 0.5 for delta, 2.0 for absolute)",
    )
    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Display shaded standard deviation bands (std across random seeds) around lines",
    )
    parser.add_argument(
        "--std-alpha",
        type=float,
        default=0.12,
        help="Transparency alpha for shaded std bands (default: 0.12)",
    )
    parser.add_argument(
        "--export-individual",
        action="store_true",
        help="Export all 6 subplots as standalone PNG files (e.g. gamma_max_ablation.png)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print(f"Loading results from {input_path}...")
    df = parse_results(input_path)
    print(f"Parsed {len(df)} records across {df['dataset'].nunique()} datasets.")

    shared_y = args.shared_y or (args.metric == "absolute")
    if args.legend_loc:
        legend_loc = args.legend_loc
    else:
        legend_loc = "upper_left" if shared_y else "bottom"

    if args.export_individual:
        print(f"Exporting 6 individual figures to {output_dir}...")
        export_individual_figures(
            df,
            output_dir,
            dpi=args.dpi,
            metric=args.metric,
            show_std=args.show_std,
            std_alpha=args.std_alpha,
            shared_y=shared_y,
            y_min=args.y_min,
            y_max=args.y_max,
            y_step=args.y_step,
        )
    else:
        if args.output:
            output_path = Path(args.output)
        else:
            if args.metric == "absolute":
                output_path = output_dir / "hyperparameter_sensitivity_absolute.png"
            elif shared_y:
                output_path = output_dir / "hyperparameter_sensitivity_shared_y.png"
            else:
                output_path = output_dir / "hyperparameter_sensitivity.png"

        print(
            f"Generating 6-panel sensitivity figure (metric={args.metric}, shared_y={shared_y}, legend_loc={legend_loc})..."
        )
        plot_sensitivity(
            df,
            output_path,
            dpi=args.dpi,
            metric=args.metric,
            show_std=args.show_std,
            std_alpha=args.std_alpha,
            shared_y=shared_y,
            legend_loc=legend_loc,
            y_min=args.y_min,
            y_max=args.y_max,
            y_step=args.y_step,
        )


if __name__ == "__main__":
    main()
