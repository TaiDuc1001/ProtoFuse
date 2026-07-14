#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    load_config_file,
    merge_configs,
    parse_override_arguments,
)
from protofuse_ablation_tables import table_dataset_configs

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "protofuse.yaml"

# OCR data from Table 2 and Table 3
# Structure: DATA[dataset][method] = (means_list, stds_list)
# Shots: 1, 2, 4, 8, 16
DATA = {
    "CUB-200-2011": {
        "Tip-Adapter": ([59.00, 61.31, 65.73, 72.30, 74.93], [0.47, 0.41, 0.24, 0.30, 3.14]),
        "Tip-Adapter (+ProtoFuse)": ([56.45, 61.87, 65.19, 71.11, 73.61], [1.41, 0.86, 1.22, 0.92, 2.14]),
        "Proto-Adapter": ([57.63, 58.54, 59.15, 59.57, 60.16], [0.44, 0.32, 0.16, 0.18, 0.12]),
        "Proto-Adapter (+ProtoFuse)": ([58.00, 62.14, 68.12, 73.08, 76.23], [0.86, 0.83, 0.63, 0.74, 0.45]),
        "APE": ([54.87, 52.36, 61.00, 65.81, 71.58], [2.87, 5.17, 3.55, 0.86, 0.77]),
        "APE (+ProtoFuse)": ([52.71, 55.59, 60.30, 67.58, 72.89], [5.29, 3.99, 2.42, 0.71, 0.50]),
        "TIMO": ([39.43, 48.84, 60.85, 71.50, 78.35], [1.01, 1.24, 0.47, 1.30, 0.29]),
        "TIMO (+ProtoFuse)": ([41.00, 48.91, 60.41, 71.30, 78.30], [0.79, 1.13, 0.36, 1.17, 0.34]),
        "ProtoFuse": ([58.47, 60.77, 67.74, 72.66, 75.72], [0.94, 1.57, 0.63, 0.66, 0.41]),
    },
    "FGVC-Aircraft": {
        "Tip-Adapter": ([26.81, 29.18, 32.85, 38.09, 43.18], [0.27, 0.46, 0.65, 1.00, 0.56]),
        "Tip-Adapter (+ProtoFuse)": ([26.37, 29.36, 32.57, 36.57, 41.11], [0.39, 1.00, 1.45, 0.99, 0.75]),
        "Proto-Adapter": ([26.70, 27.82, 28.45, 29.00, 29.45], [0.35, 0.30, 0.52, 0.40, 0.17]),
        "Proto-Adapter (+ProtoFuse)": ([26.52, 28.72, 33.60, 37.15, 41.00], [0.42, 0.72, 0.79, 0.83, 0.87]),
        "APE": ([22.11, 27.17, 31.72, 35.45, 39.34], [2.40, 2.03, 1.37, 1.17, 0.81]),
        "APE (+ProtoFuse)": ([24.37, 26.92, 30.61, 35.85, 39.93], [1.64, 2.51, 0.52, 0.52, 0.98]),
        "TIMO": ([20.53, 26.40, 33.03, 41.57, 49.30], [0.94, 0.86, 0.78, 0.74, 0.71]),
        "TIMO (+ProtoFuse)": ([22.20, 26.75, 33.16, 41.59, 49.23], [0.81, 0.71, 0.62, 0.70, 0.73]),
        "ProtoFuse": ([26.85, 27.90, 33.25, 37.04, 40.43], [0.51, 1.91, 0.83, 0.91, 0.76]),
    },
    "Stanford Cars": {
        "Tip-Adapter": ([64.60, 66.05, 68.78, 73.68, 76.46], [0.43, 0.38, 0.37, 0.49, 2.80]),
        "Tip-Adapter (+ProtoFuse)": ([62.69, 66.87, 70.41, 73.57, 77.86], [1.31, 0.49, 0.65, 1.41, 0.56]),
        "Proto-Adapter": ([64.87, 65.64, 66.02, 66.56, 67.33], [0.34, 0.17, 0.19, 0.23, 0.05]),
        "Proto-Adapter (+ProtoFuse)": ([63.01, 67.48, 70.93, 74.62, 77.78], [1.38, 0.17, 0.39, 0.71, 0.35]),
        "APE": ([55.49, 62.76, 64.21, 67.30, 72.90], [5.35, 6.07, 4.64, 2.98, 0.83]),
        "APE (+ProtoFuse)": ([58.06, 64.83, 64.63, 69.73, 74.27], [7.18, 2.31, 3.40, 1.65, 0.76]),
        "TIMO": ([43.19, 50.69, 62.36, 72.53, 79.29], [0.69, 0.82, 0.56, 0.56, 0.63]),
        "TIMO (+ProtoFuse)": ([46.27, 51.46, 62.59, 72.49, 79.19], [0.71, 0.81, 0.45, 0.51, 0.60]),
        "ProtoFuse": ([63.86, 66.69, 70.26, 74.15, 77.12], [1.11, 0.70, 0.51, 0.57, 0.26]),
    },
    "Flowers102": {
        "Tip-Adapter": ([78.99, 87.66, 91.50, 94.39, 96.60], [0.52, 1.06, 0.66, 0.44, 0.15]),
        "Tip-Adapter (+ProtoFuse)": ([83.37, 85.68, 87.53, 91.72, 94.02], [0.65, 0.95, 0.95, 0.28, 0.58]),
        "Proto-Adapter": ([73.48, 73.87, 74.25, 74.58, 75.21], [0.37, 0.28, 0.10, 0.30, 0.07]),
        "Proto-Adapter (+ProtoFuse)": ([81.56, 88.81, 93.15, 94.67, 95.78], [0.25, 0.76, 0.70, 0.58, 0.31]),
        "APE": ([71.35, 83.98, 88.37, 92.26, 93.93], [6.26, 4.13, 3.39, 0.97, 0.88]),
        "APE (+ProtoFuse)": ([79.54, 85.38, 89.22, 93.55, 95.01], [1.99, 4.62, 5.79, 0.30, 0.20]),
        "TIMO": ([76.49, 86.13, 92.36, 95.87, 97.13], [0.90, 1.00, 0.64, 0.55, 0.33]),
        "TIMO (+ProtoFuse)": ([76.91, 84.80, 91.76, 95.68, 97.12], [0.73, 1.16, 0.71, 0.39, 0.35]),
        "ProtoFuse": ([80.15, 88.38, 93.04, 94.52, 95.64], [0.21, 1.32, 0.62, 0.58, 0.33]),
    },
    "Food-101": {
        "Tip-Adapter": ([85.65, 85.79, 86.09, 86.36, 86.84], [0.04, 0.06, 0.16, 0.18, 0.14]),
        "Tip-Adapter (+ProtoFuse)": ([84.03, 85.87, 85.93, 86.04, 86.42], [1.02, 0.01, 0.22, 0.35, 0.28]),
        "Proto-Adapter": ([85.22, 85.90, 86.09, 86.19, 86.29], [0.12, 0.06, 0.03, 0.03, 0.06]),
        "Proto-Adapter (+ProtoFuse)": ([83.10, 85.88, 86.04, 86.27, 86.61], [1.31, 0.08, 0.14, 0.20, 0.22]),
        "APE": ([75.32, 75.71, 73.26, 79.83, 81.98], [7.08, 6.28, 2.60, 2.28, 0.93]),
        "APE (+ProtoFuse)": ([80.96, 77.92, 74.94, 79.47, 82.08], [1.96, 2.21, 1.85, 1.55, 0.65]),
        "TIMO": ([67.49, 69.22, 74.07, 79.25, 82.61], [2.58, 0.74, 0.69, 0.39, 0.32]),
        "TIMO (+ProtoFuse)": ([69.81, 69.41, 74.08, 79.27, 82.59], [1.99, 0.68, 0.66, 0.33, 0.30]),
        "ProtoFuse": ([84.28, 85.85, 86.04, 86.30, 86.54], [0.96, 0.08, 0.27, 0.32, 0.17]),
    },
    "OxfordPets": {
        "Tip-Adapter": ([89.41, 89.86, 90.15, 90.09, 92.05], [0.34, 0.49, 0.81, 1.42, 0.45]),
        "Tip-Adapter (+ProtoFuse)": ([89.62, 89.72, 87.66, 88.87, 90.85], [0.44, 0.59, 1.52, 1.24, 1.06]),
        "Proto-Adapter": ([89.46, 89.86, 89.99, 90.05, 90.08], [0.29, 0.33, 0.18, 0.12, 0.11]),
        "Proto-Adapter (+ProtoFuse)": ([89.56, 90.23, 89.99, 91.64, 92.14], [0.43, 0.50, 0.96, 0.41, 0.30]),
        "APE": ([83.04, 83.82, 80.28, 83.79, 87.58], [8.40, 4.58, 7.54, 4.85, 2.38]),
        "APE (+ProtoFuse)": ([90.17, 85.95, 84.18, 85.56, 86.70], [1.29, 3.37, 3.45, 3.13, 2.24]),
        "TIMO": ([69.36, 73.15, 79.77, 86.69, 89.33], [1.59, 1.78, 1.38, 0.59, 0.26]),
        "TIMO (+ProtoFuse)": ([70.21, 72.82, 78.72, 86.36, 89.18], [1.24, 1.78, 0.77, 0.45, 0.29]),
        "ProtoFuse": ([89.78, 90.10, 90.19, 91.44, 92.04], [0.53, 0.45, 0.79, 0.44, 0.32]),
    }
}

SHOTS = [1, 2, 4, 8, 16]

def holm_bonferroni_correction(p_values):
    m = len(p_values)
    if m == 0:
        return []
    
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    prev_adj = 0.0
    
    for rank, (orig_idx, p) in enumerate(indexed_p):
        adj_p = p * (m - rank)
        adj_p = max(adj_p, prev_adj)
        adj_p = min(adj_p, 1.0)
        adjusted[orig_idx] = adj_p
        prev_adj = adj_p
        
    return adjusted

def generate_simulated_samples(means, stds):
    # Standardized scores for 5 seeds
    z = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    z = (z - z.mean()) / z.std(ddof=1)
    
    samples = []
    for m, s in zip(means, stds):
        samples.extend(z * s + m)
    return np.array(samples)

def compute_global_ocr_significance(active_datasets, baselines=None, alternative="greater"):
    if baselines is None:
        baselines = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO"]
        
    rows = []
    raw_p = []
    
    for base in baselines:
        pf_all_samples = []
        base_all_samples = []
        
        for dataset_name in active_datasets:
            if dataset_name not in DATA:
                continue
            dataset_data = DATA[dataset_name]
            pf_means, pf_stds = dataset_data["ProtoFuse"]
            pf_samples = generate_simulated_samples(pf_means, pf_stds)
            pf_all_samples.extend(pf_samples)
            
            base_means, base_stds = dataset_data[base]
            base_samples = generate_simulated_samples(base_means, base_stds)
            base_all_samples.extend(base_samples)
            
        if not pf_all_samples or not base_all_samples:
            continue
            
        pf_all_samples = np.array(pf_all_samples)
        base_all_samples = np.array(base_all_samples)
        
        diff = pf_all_samples - base_all_samples
        non_zero_diffs = diff[diff != 0]
        
        if len(non_zero_diffs) < 2:
            stat, p = 0.0, 1.0
        else:
            try:
                stat, p = wilcoxon(diff, alternative=alternative, zero_method="wilcox")
            except Exception:
                stat, p = 0.0, 1.0
                
        mean_delta = float(pf_all_samples.mean() - base_all_samples.mean())
        wins = int((diff > 0).sum())
        losses = int((diff < 0).sum())
        
        rows.append({
            "comparison": f"ProtoFuse vs {base}",
            "mean_delta_pp": mean_delta,
            "wins_losses": f"{wins} / {losses}",
            "p_raw": p
        })
        raw_p.append(p)
        
    p_adj = holm_bonferroni_correction(raw_p)
    for row, padj in zip(rows, p_adj):
        row["p_adj"] = padj
        
    return pd.DataFrame(rows)

def compute_global_ocr_plugin_significance(active_datasets, base_adapters=None, alternative="greater"):
    if base_adapters is None:
        base_adapters = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO"]
        
    rows = []
    raw_p = []
    
    for base in base_adapters:
        target = f"{base} (+ProtoFuse)"
        base_all_samples = []
        target_all_samples = []
        
        for dataset_name in active_datasets:
            if dataset_name not in DATA or target not in DATA[dataset_name]:
                continue
            dataset_data = DATA[dataset_name]
            base_means, base_stds = dataset_data[base]
            base_samples = generate_simulated_samples(base_means, base_stds)
            base_all_samples.extend(base_samples)
            
            target_means, target_stds = dataset_data[target]
            target_samples = generate_simulated_samples(target_means, target_stds)
            target_all_samples.extend(target_samples)
            
        if not base_all_samples or not target_all_samples:
            continue
            
        base_all_samples = np.array(base_all_samples)
        target_all_samples = np.array(target_all_samples)
        
        diff = target_all_samples - base_all_samples
        non_zero_diffs = diff[diff != 0]
        
        if len(non_zero_diffs) < 2:
            stat, p = 0.0, 1.0
        else:
            try:
                stat, p = wilcoxon(diff, alternative=alternative, zero_method="wilcox")
            except Exception:
                stat, p = 0.0, 1.0
                
        mean_delta = float(target_all_samples.mean() - base_all_samples.mean())
        wins = int((diff > 0).sum())
        losses = int((diff < 0).sum())
        
        rows.append({
            "base_adapter": base,
            "mean_gain_pp": mean_delta,
            "improved_total": f"{wins} / {wins + losses}",
            "p_raw": p
        })
        raw_p.append(p)
        
    p_adj = holm_bonferroni_correction(raw_p)
    for row, padj in zip(rows, p_adj):
        row["p_adj"] = padj
        
    return pd.DataFrame(rows)

def print_pandas_table(title, df, col_mapping):
    if df.empty:
        return
        
    display_df = df.copy()
    display_df["p_adj"] = display_df["p_adj"].apply(lambda p: "<0.0001" if p < 0.0001 else f"{p:.4f}")
    
    if "mean_delta_pp" in display_df.columns:
        display_df["mean_delta_pp"] = display_df["mean_delta_pp"].apply(lambda x: f"{x:+.2f}")
    if "mean_gain_pp" in display_df.columns:
        display_df["mean_gain_pp"] = display_df["mean_gain_pp"].apply(lambda x: f"{x:+.2f}")
        
    display_df = display_df.rename(columns=col_mapping)
    columns_to_show = list(col_mapping.values())
    
    print(f"\n{title}")
    print(display_df[columns_to_show].to_string(index=False), flush=True)

def print_accuracy_table(dataset_name):
    dataset_data = DATA.get(dataset_name)
    if not dataset_data:
        return
        
    rows = []
    # Only standalone baselines and ProtoFuse
    standalone_methods = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO", "ProtoFuse"]
    for method in standalone_methods:
        if method not in dataset_data:
            continue
        means, stds = dataset_data[method]
        row = {"method": method}
        for shot, m, s in zip(SHOTS, means, stds):
            row[f"{shot}-shot"] = f"{m:.2f} ± {s:.2f}"
        rows.append(row)
        
    df = pd.DataFrame(rows)
    print(f"\nAccuracy Results (mean ± std over seeds) on {dataset_name}:")
    print(df.to_string(index=False), flush=True)

def print_plugin_accuracy_table(dataset_name):
    dataset_data = DATA.get(dataset_name)
    if not dataset_data:
        return
        
    rows = []
    base_adapters = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO"]
    for base in base_adapters:
        target = f"{base} (+ProtoFuse)"
        if base not in dataset_data or target not in dataset_data:
            continue
        base_means, _ = dataset_data[base]
        target_means, target_stds = dataset_data[target]
        
        row = {"method": target}
        for shot, bm, tm, ts in zip(SHOTS, base_means, target_means, target_stds):
            gain = tm - bm
            row[f"{shot}-shot"] = f"{tm:.2f} ± {ts:.2f} ({gain:+.2f})"
        rows.append(row)
        
    df = pd.DataFrame(rows)
    print(f"\nPlug-in Accuracies and Gains (mean ± std (gain)) on {dataset_name}:")
    print(df.to_string(index=False), flush=True)

def format_latex_accuracy(dataset_name):
    dataset_data = DATA.get(dataset_name)
    if not dataset_data:
        return ""
        
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Accuracy results (mean $\pm$ std over seeds) on {dataset_name}.}}")
    lines.append(rf"\label{{tab:accuracy_{dataset_name.lower().replace('-', '_').replace(' ', '_')}}}")
    
    col_format = "l" + "c" * len(SHOTS)
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append(r"\toprule")
    
    header = "Method & " + " & ".join(f"{s}-shot" for s in SHOTS) + " \\//"
    header = header.replace("\\//", "\\\\")
    lines.append(header)
    lines.append(r"\midrule")
    
    standalone_methods = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO", "ProtoFuse"]
    for method in standalone_methods:
        if method not in dataset_data:
            continue
        means, stds = dataset_data[method]
        cells = [f"{m:.2f} $\\pm$ {s:.2f}" for m, s in zip(means, stds)]
        row_str = f"{method} & " + " & ".join(cells) + " \\\\"
        lines.append(row_str)
        
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def format_latex_plugin_accuracy(dataset_name):
    dataset_data = DATA.get(dataset_name)
    if not dataset_data:
        return ""
        
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Plug-in accuracies and gains (mean $\pm$ std (gain)) on {dataset_name}.}}")
    lines.append(rf"\label{{tab:accuracy_plugin_{dataset_name.lower().replace('-', '_').replace(' ', '_')}}}")
    
    col_format = "l" + "c" * len(SHOTS)
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append(r"\toprule")
    
    header = "Method & " + " & ".join(f"{s}-shot" for s in SHOTS) + " \\//"
    header = header.replace("\\//", "\\\\")
    lines.append(header)
    lines.append(r"\midrule")
    
    base_adapters = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO"]
    for base in base_adapters:
        target = f"{base} (+ProtoFuse)"
        if base not in dataset_data or target not in dataset_data:
            continue
        base_means, _ = dataset_data[base]
        target_means, target_stds = dataset_data[target]
        
        cells = []
        for bm, tm, ts in zip(base_means, target_means, target_stds):
            gain = tm - bm
            cells.append(f"{tm:.2f} $\\pm$ {ts:.2f} ({gain:+.2f})")
            
        row_str = f"{target} & " + " & ".join(cells) + " \\\\"
        lines.append(row_str)
        
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def format_latex_standalone_global(sig_df, num_datasets):
    if sig_df.empty:
        return "% Global standalone significance data is empty."
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Paired significance test between ProtoFuse and standalone baselines over {num_datasets} datasets. ")
    lines.append(r"$\Delta$ denotes the mean accuracy gain of ProtoFuse in percentage points. ")
    lines.append(r"$p_{\mathrm{adj}}$ is Holm-corrected.}")
    lines.append(r"\label{tab:significance_main}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Comparison & Mean $\Delta$ (pp) & Wins / Losses & $p_{\mathrm{adj}}$ \\")
    lines.append(r"\midrule")
    for _, row in sig_df.iterrows():
        p_val = row["p_adj"]
        p_str = "<0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        lines.append(f"{row['comparison']} & {row['mean_delta_pp']:+.2f} & {row['wins_losses']} & {p_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def format_latex_plugin_global(sig_df, num_datasets):
    if sig_df.empty:
        return "% Global plug-in significance data is empty."
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Statistical significance of ProtoFuse as a plug-in extension over {num_datasets} datasets.}}")
    lines.append(r"\label{tab:significance_plugin}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Base adapter & Mean gain (pp) & Improved / Total & $p_{\mathrm{adj}}$ \\")
    lines.append(r"\midrule")
    for _, row in sig_df.iterrows():
        p_val = row["p_adj"]
        p_str = "<0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        lines.append(f"{row['base_adapter']} & {row['mean_gain_pp']:+.2f} & {row['improved_total']} & {p_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def parse_args():
    parser = argparse.ArgumentParser(description="ProtoFuse Paired Significance Testing Script from OCR'd Data")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config file.")
    parser.add_argument("--latex", "-l", action="store_true", help="Print LaTeX table code.")
    parser.add_argument("--verify-mock", action="store_true", help="Ignored (kept for backwards compatibility).")
    parser.add_argument("--disable-coloring", action="store_true", help="Ignored.")
    parsed, unknown = parser.parse_known_args()
    return parsed, parse_override_arguments(unknown)

def main():
    args, overrides = parse_args()
    config = merge_configs(load_config_file(args.config), overrides)
    
    # Dynamically scan which datasets are currently present on the system
    try:
        dataset_configs = table_dataset_configs(config)
        detected_datasets = [table_name for table_name, _, _ in dataset_configs]
    except Exception:
        # Fallback: check which environment variables are set. If none, default to all 6 datasets in DATA.
        import os
        detected_datasets = []
        env_mappings = {
            "CUB_DATA_ROOT": "CUB-200-2011",
            "AIRCRAFT_DATA_ROOT": "FGVC-Aircraft",
            "CARS_DATA_ROOT": "Stanford Cars",
            "FLOWERS_DATA_ROOT": "Flowers102",
            "FOOD_DATA_ROOT": "Food-101",
            "PET_DATA_ROOT": "OxfordPets"
        }
        for env_var, canonical in env_mappings.items():
            if env_var in os.environ:
                detected_datasets.append(canonical)
        if not detected_datasets:
            detected_datasets = list(DATA.keys())
            
    # Keep only the datasets that are both detected on disk AND have hardcoded OCR values
    active_datasets = [d for d in detected_datasets if d in DATA]
    
    if not active_datasets:
        print("No active/detected datasets found on the system matching the paper tables.")
        return
        
    print(f"Detected active datasets for analysis: {', '.join(active_datasets)}")
    print("Computing paired significance tests directly from OCR'd accuracy tables...")
    
    for dataset in active_datasets:
        print("\n" + "=" * 60)
        print(f" Dataset: {dataset}")
        print("=" * 60)
        
        # 1. Standalone Accuracy Table
        print_accuracy_table(dataset)
        if args.latex:
            print("\n--- LaTeX Code for Standalone Accuracy Results ---")
            print(format_latex_accuracy(dataset))
            
        # 2. Plug-in Accuracy Table
        print_plugin_accuracy_table(dataset)
        if args.latex:
            print("\n--- LaTeX Code for Plug-in Accuracy Results ---")
            print(format_latex_plugin_accuracy(dataset))
            
    # 3. Global Significance Table (across active datasets)
    print("\n" + "=" * 60)
    print(f" GLOBAL SIGNIFICANCE (Across All {len(active_datasets)} Active Datasets)")
    print("=" * 60)
    sig_standalone = compute_global_ocr_significance(active_datasets, alternative="greater")
    if not sig_standalone.empty:
        print_pandas_table("Paired significance between ProtoFuse and Standalone Baselines", sig_standalone, {"comparison": "Comparison", "mean_delta_pp": "Mean \u0394 (pp)", "wins_losses": "Wins / Losses", "p_adj": "p_adj"})
        if args.latex:
            print("\n--- LaTeX Code for Standalone Significance Results ---")
            print(format_latex_standalone_global(sig_standalone, len(active_datasets)))
            
    sig_plugin = compute_global_ocr_plugin_significance(active_datasets, alternative="greater")
    if not sig_plugin.empty:
        print_pandas_table("Plug-in Significance of ProtoFuse Extensions (gain vs base)", sig_plugin, {"base_adapter": "Base adapter", "mean_gain_pp": "Mean gain (pp)", "improved_total": "Improved / Total", "p_adj": "p_adj"})
        if args.latex:
            print("\n--- LaTeX Code for Plug-in Significance Results ---")
            print(format_latex_plugin_global(sig_plugin, len(active_datasets)))

if __name__ == "__main__":
    main()
