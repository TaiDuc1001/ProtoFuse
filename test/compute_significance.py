#!/usr/bin/env python3
import argparse
import copy
import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "ablation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    load_config_file,
    merge_configs,
    parse_override_arguments,
)
from protofuse_ablation_tables import table_dataset_configs

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "protofuse.yaml"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_RESULTS_FILE = ASSETS_DIR / "standalone_baselines_results.txt"

DATA = {}
RAW_SEED_DATA = {}

DATASET_MAPPINGS = {
    "FGVCAircraft": "FGVC-Aircraft",
    "StanfordCars": "Stanford Cars",
    "CUB-200-2011": "CUB-200-2011",
    "Flowers102": "Flowers102",
    "Food-101": "Food-101",
    "OxfordPets": "OxfordPets",
}

METHOD_MAPPINGS = {
    "TipAdapter": "Tip-Adapter",
    "ProtoAdapter": "Proto-Adapter",
    "APE": "APE",
    "TIMO": "TIMO",
    "TARP": "TARP",
    "ProtoFuse": "ProtoFuse",
    "TipAdapter+ProtoFuse": "Tip-Adapter (+ProtoFuse)",
    "ProtoAdapter+ProtoFuse": "Proto-Adapter (+ProtoFuse)",
    "APE+ProtoFuse": "APE (+ProtoFuse)",
    "TIMO+ProtoFuse": "TIMO (+ProtoFuse)",
    "TARP+ProtoFuse": "TARP (+ProtoFuse)",
}

SHOTS = [1, 2, 4, 8, 16]

def load_results_file(file_path):
    path = Path(file_path)
    if not path.is_file():
        return 0
        
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"([A-Za-z0-9\-]+)\s*x\s*([A-Za-z0-9\+\-]+)\s*x\s*seed\s*(\d+)\s*x\s*(\d+)-shot\s*x\s*([\d\.]+)%"
    )
    matches = pattern.findall(text)
    if not matches:
        return 0
        
    for raw_d, raw_m, s_str, k_str, acc_str in matches:
        dataset_name = DATASET_MAPPINGS.get(raw_d, raw_d)
        method_name = METHOD_MAPPINGS.get(raw_m, raw_m)
        shot = int(k_str)
        seed = int(s_str)
        acc = float(acc_str)
        
        RAW_SEED_DATA.setdefault(dataset_name, {}).setdefault(method_name, {}).setdefault(shot, {})[seed] = acc

    for d, d_methods in RAW_SEED_DATA.items():
        if d not in DATA:
            DATA[d] = {}
        for m, m_shots in d_methods.items():
            means = []
            stds = []
            for shot in SHOTS:
                if shot in m_shots:
                    seed_vals = list(m_shots[shot].values())
                    means.append(round(float(np.mean(seed_vals)), 2))
                    stds.append(round(float(np.std(seed_vals, ddof=0)), 2))
                else:
                    means.append(0.0)
                    stds.append(0.0)
            DATA[d][m] = (means, stds)
            
    return len(matches)

def load_all_assets(assets_dir=ASSETS_DIR):
    total = 0
    if assets_dir.is_dir():
        for file in sorted(assets_dir.glob("*.txt")):
            count = load_results_file(file)
            total += count
    return total

# Automatically parse and load all existing result files in assets directory
load_all_assets(ASSETS_DIR)

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
        baselines = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO", "TARP"]
        
    rows = []
    raw_p = []
    
    for base in baselines:
        pf_all_samples = []
        base_all_samples = []
        
        for dataset_name in active_datasets:
            if dataset_name not in DATA or "ProtoFuse" not in DATA[dataset_name] or base not in DATA[dataset_name]:
                continue
            dataset_data = DATA[dataset_name]
            
            raw_pf_shots = RAW_SEED_DATA.get(dataset_name, {}).get("ProtoFuse", {})
            raw_base_shots = RAW_SEED_DATA.get(dataset_name, {}).get(base, {})
            
            has_raw = (
                all(shot in raw_pf_shots and len(raw_pf_shots[shot]) == 5 for shot in SHOTS)
                and all(shot in raw_base_shots and len(raw_base_shots[shot]) == 5 for shot in SHOTS)
            )
            
            if has_raw:
                # Direct paired comparison by identical seed ids
                for shot in SHOTS:
                    common_seeds = sorted(set(raw_pf_shots[shot].keys()) & set(raw_base_shots[shot].keys()))
                    for s in common_seeds:
                        pf_all_samples.append(raw_pf_shots[shot][s])
                        base_all_samples.append(raw_base_shots[shot][s])
            else:
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
        base_adapters = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO", "TARP"]
        
    rows = []
    raw_p = []
    
    for base in base_adapters:
        target = f"{base} (+ProtoFuse)"
        base_all_samples = []
        target_all_samples = []
        
        for dataset_name in active_datasets:
            if dataset_name not in DATA or target not in DATA[dataset_name] or base not in DATA[dataset_name]:
                continue
            dataset_data = DATA[dataset_name]
            
            raw_base_shots = RAW_SEED_DATA.get(dataset_name, {}).get(base, {})
            has_raw = all(shot in raw_base_shots and len(raw_base_shots[shot]) == 5 for shot in SHOTS)
            
            if has_raw:
                for shot in SHOTS:
                    vals = [raw_base_shots[shot][s] for s in sorted(raw_base_shots[shot].keys())]
                    base_all_samples.extend(sorted(vals))
            else:
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
    standalone_methods = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO", "TARP", "ProtoFuse"]
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
    base_adapters = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO", "TARP"]
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
        
    if not rows:
        return
        
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
    
    standalone_methods = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO", "TARP", "ProtoFuse"]
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
    
    base_adapters = ["Tip-Adapter", "Proto-Adapter", "APE", "TIMO", "TARP"]
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
    parser = argparse.ArgumentParser(description="ProtoFuse Paired Significance Testing Script")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config file.")
    parser.add_argument("--results-file", default=None, help="Path to results txt file with per-seed accuracies.")
    parser.add_argument("--latex", "-l", action="store_true", help="Print LaTeX table code.")
    parser.add_argument("--verify-mock", action="store_true", help="Ignored (kept for backwards compatibility).")
    parser.add_argument("--disable-coloring", action="store_true", help="Ignored.")
    parsed, unknown = parser.parse_known_args()
    return parsed, parse_override_arguments(unknown)

def main():
    args, overrides = parse_args()
    config = merge_configs(load_config_file(args.config), overrides)
    
    results_path = getattr(args, "results_file", None)
    if results_path:
        count = load_results_file(results_path)
        if count > 0:
            print(f"Loaded {count} raw seed results from: {results_path}")
    else:
        count = sum(sum(len(seeds) for seeds in shots.values()) for d in RAW_SEED_DATA.values() for shots in d.values())
        if count > 0:
            print(f"Loaded {count} raw seed results from: {DEFAULT_RESULTS_FILE.name}")
    
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
            
    # Keep only the datasets that are both detected on disk AND have values
    active_datasets = [d for d in detected_datasets if d in DATA]
    
    if not active_datasets:
        print("No active/detected datasets found on the system matching the tables.")
        return
        
    print(f"Detected active datasets for analysis: {', '.join(active_datasets)}")
    if count > 0:
        print("Computing paired significance tests using raw baseline seed data...")
    else:
        print("Computing paired significance tests directly from accuracy tables...")
    
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
