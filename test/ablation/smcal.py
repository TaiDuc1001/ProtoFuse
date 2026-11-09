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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from protofuse_ablation_tables import (
    build_selector,
    fused_prototypes,
    table_dataset_configs,
)
from protofuse_candidates import (
    DEFAULT_CONFIG,
    parse_int_list,
    resolve_path,
    split_positions_by_class,
    support_positions,
)
from src.pipelines.protofuse import ProtoFusePipeline
from utils import (
    get_config_value,
    load_config_file,
    merge_configs,
    parse_override_arguments,
    set_global_seed,
)


DEFAULT_EXP1_SEEDS = list(range(1, 51))
DEFAULT_EXP2_SEEDS = [1, 10, 100, 1000, 10000]
DEFAULT_BETAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

EXP2_VARIANTS = (
    "Text-only",
    "Fixed (α=0.25)",
    "SM-Cal only",
    "SM-Exp (Fixed α=0.25)",
    "Full ProtoFuse",
)


def fast_surrogatemix_calibration(T, V, num_classes, alphas, beta_values):
    """
    Fast vectorized evaluation of centroid-mix calibration for 1-shot.
    T: (C, D)
    V: (C, D)
    alphas: (A,)
    beta_values: list of float
    """
    sim = V @ V.T
    sim.fill_diagonal_(-float("inf"))
    nbs = sim.argmax(dim=-1)
    labels = torch.arange(num_classes, device=T.device)
    cos_phi = (T * V).sum(dim=-1)  # (C,)

    a = alphas.view(-1, 1, 1)  # (A, 1, 1)
    om_a = 1.0 - a
    denom = torch.sqrt((om_a ** 2 + a ** 2 + 2.0 * a * om_a * cos_phi.unsqueeze(0)).clamp(min=1e-12))  # (A, 1, C)

    best = {"score": -float("inf"), "alpha": 0.0}
    for beta in beta_values:
        Q = F.normalize((1.0 - beta) * V + beta * V[nbs], dim=-1)  # (C, D)
        S_T = Q @ T.T  # (C, C)
        S_V = Q @ V.T  # (C, C)

        logits = (om_a * S_T.unsqueeze(0) + a * S_V.unsqueeze(0)) / denom  # (A, C, C)
        preds = logits.argmax(dim=-1)  # (A, C)
        curve = preds.eq(labels.unsqueeze(0)).sum(dim=-1).float()  # (A,)

        score = float(curve.max().item())
        alpha = float(alphas[int(curve.argmax().item())].item())
        # Lexicographic tie-breaking: prefer higher score, tie-break by smaller alpha
        if score > best["score"] or (score == best["score"] and alpha < best["alpha"]):
            best = {"score": score, "alpha": alpha}
    return best["alpha"]


REPO_ROOT = Path(__file__).resolve().parents[2]


def compute_class_atypicality_rankings(train_features_all, train_labels_all, num_classes):
    """
    Vectorized computation of per-sample atypicality score:
    t_{c, i} = z_{c, i}^T Norm(sum_{j != i} z_{c, j})
    Returns: dict mapping class_idx -> {
      'all_indices': tensor of indices in train_features_all,
      'n_c': number of samples in class c,
      'atypical_pool': bottom 20% tensor indices in train_features_all,
      'typical_pool': top 20% tensor indices in train_features_all,
      'atypical_indices': list of bottom 20% indices sorted by similarity ascending,
      'atypical_scores': list of float similarity scores sorted ascending,
      'typical_indices': list of top 20% indices sorted by similarity descending,
      'typical_scores': list of float similarity scores sorted descending,
    }
    """
    rankings = {}
    for c in range(num_classes):
        mask = train_labels_all.eq(c)
        class_indices = torch.nonzero(mask, as_tuple=False).flatten()
        n_c = int(class_indices.numel())
        if n_c == 0:
            continue
        X_c = train_features_all[class_indices]  # (N_c, D), normalized

        if n_c <= 2:
            rankings[c] = {
                "n_c": n_c,
                "all_indices": class_indices,
                "atypical_pool": class_indices,
                "typical_pool": class_indices,
                "atypical_indices": class_indices.tolist(),
                "atypical_scores": [0.0] * n_c,
                "typical_indices": class_indices.tolist(),
                "typical_scores": [0.0] * n_c,
            }
            continue

        S_c = X_c.sum(dim=0)  # (D,)
        S_norm_sq = (S_c * S_c).sum()
        d_c = X_c @ S_c  # (N_c,)
        denom = torch.sqrt((S_norm_sq - 2.0 * d_c + 1.0).clamp(min=1e-12))
        t_c = (d_c - 1.0) / denom  # (N_c,)

        sorted_local = torch.argsort(t_c)  # ascending: lowest similarity to centroid first
        k_20 = max(1, int(math.ceil(0.20 * n_c)))
        atypical_local = sorted_local[:k_20]  # lowest similarity first (most atypical)
        typical_local = sorted_local.flip(0)[:k_20]  # highest similarity first (most typical)

        rankings[c] = {
            "n_c": n_c,
            "all_indices": class_indices,
            "atypical_pool": class_indices[atypical_local],
            "typical_pool": class_indices[typical_local],
            "atypical_indices": class_indices[atypical_local].tolist(),
            "atypical_scores": t_c[atypical_local].tolist(),
            "typical_indices": class_indices[typical_local].tolist(),
            "typical_scores": t_c[typical_local].tolist(),
        }
    return rankings


def export_typical_atypical_json(
    table_name,
    rankings,
    classnames,
    image_paths,
    output_dir,
    base_dir=None,
):
    """
    Exports typical.json and atypical.json for the dataset into output_dir/<table_name>/.
    Stores all images in the top/bottom 20% per class with relative paths.
    - typical.json: sorted by similarity descending (highest similarity first)
    - atypical.json: sorted by similarity ascending (lowest similarity first)
    """
    clean_name = table_name.replace(" ", "")
    dataset_out_dir = Path(output_dir) / clean_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    if base_dir is None:
        base_dir = REPO_ROOT

    typical_classes = {}
    atypical_classes = {}

    for c, info in rankings.items():
        n_c = info["n_c"]
        c_name = classnames[c] if c < len(classnames) else str(c)
        c_key = str(c)

        # 1. Atypical: sorted ascending by similarity (lowest similarity first)
        atypical_list = []
        for rank, (global_idx, score) in enumerate(zip(info["atypical_indices"], info["atypical_scores"]), start=1):
            raw_path = image_paths[global_idx] if image_paths and global_idx < len(image_paths) else f"image_{global_idx}"
            try:
                rel_path = str(Path(raw_path).resolve().relative_to(Path(base_dir).resolve()))
            except Exception:
                try:
                    rel_path = os.path.relpath(raw_path, base_dir)
                except Exception:
                    rel_path = str(raw_path)

            atypical_list.append({
                "image_path": rel_path,
                "class_name": c_name,
                "atypicality_score": float(round(score, 6)),
                "rank_in_class": rank,
                "percentile_in_class": float(round(rank / max(1, n_c), 4)),
            })
        atypical_classes[c_key] = atypical_list

        # 2. Typical: sorted descending by similarity (highest similarity first)
        typical_list = []
        for rank, (global_idx, score) in enumerate(zip(info["typical_indices"], info["typical_scores"]), start=1):
            raw_path = image_paths[global_idx] if image_paths and global_idx < len(image_paths) else f"image_{global_idx}"
            try:
                rel_path = str(Path(raw_path).resolve().relative_to(Path(base_dir).resolve()))
            except Exception:
                try:
                    rel_path = os.path.relpath(raw_path, base_dir)
                except Exception:
                    rel_path = str(raw_path)

            typical_list.append({
                "image_path": rel_path,
                "class_name": c_name,
                "atypicality_score": float(round(score, 6)),
                "rank_in_class": rank,
                "percentile_in_class": float(round(rank / max(1, n_c), 4)),
            })
        typical_classes[c_key] = typical_list

    atypical_payload = {
        "dataset": table_name,
        "group": "atypical",
        "classes": atypical_classes,
    }
    typical_payload = {
        "dataset": table_name,
        "group": "typical",
        "classes": typical_classes,
    }

    atypical_path = dataset_out_dir / "atypical.json"
    typical_path = dataset_out_dir / "typical.json"

    with open(atypical_path, "w") as f:
        json.dump(atypical_payload, f, indent=2)

    with open(typical_path, "w") as f:
        json.dump(typical_payload, f, indent=2)

    print(f"  [{table_name}] Exported typical/atypical JSONs to {dataset_out_dir}", flush=True)


def sample_stress_support_indices(rankings, num_classes, atypical_ratio, condition, seed):
    """
    Samples 1-shot support indices for all classes under atypicality stress.
    atypical_ratio: 0.0 (Random), 0.25 (Mild), 0.50 (Moderate), 1.0 (Severe)
    condition: 'atypical' or 'typical' or 'random'
    """
    rng = np.random.RandomState(seed)
    num_target_classes = int(round(num_classes * atypical_ratio))
    chosen_classes = set(rng.choice(num_classes, size=num_target_classes, replace=False))

    support_idx = []
    for c in range(num_classes):
        info = rankings[c]
        if c in chosen_classes:
            pool = info["atypical_pool"] if condition == "atypical" else info["typical_pool"]
        else:
            pool = info["all_indices"]
        selected = pool[rng.randint(len(pool))].item()
        support_idx.append(selected)
    return torch.tensor(support_idx, dtype=torch.long)


def evaluate_exp2_variants_batched(
    stacked_prototypes,
    eval_features,
    eval_labels,
    device,
    batch_size=2048,
):
    """
    Evaluates 5 variants (Text-only, Fixed 0.25, SM-Cal, SM-Exp Fixed, Full ProtoFuse)
    in a single batched GEMM pass.
    stacked_prototypes: (5, C, D)
    """
    V, C, D = stacked_prototypes.shape
    total = int(eval_labels.numel())
    batch_size = max(1, int(batch_size))
    stacked_flat = stacked_prototypes.view(V * C, D)

    correct_counts = torch.zeros(V, device=device, dtype=torch.long)
    eval_labels_dev = eval_labels.to(device).long()

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            features = eval_features[start:end].to(device, non_blocking=True).float()
            labels = eval_labels_dev[start:end]
            curr_b = int(labels.shape[0])

            batch_logits = (features @ stacked_flat.T).view(curr_b, V, C).permute(1, 0, 2)
            preds = batch_logits.argmax(dim=-1)  # (V, B)
            correct_counts += preds.eq(labels.unsqueeze(0)).sum(dim=1)

    accuracies = correct_counts.float().mul_(100.0 / max(1, total)).cpu().tolist()
    return accuracies


def run_experiment_1_dataset(
    dataset_name,
    text_features,
    train_features_all,
    train_labels_all,
    eval_features_all,
    eval_labels_all,
    val_fraction,
    device,
    alphas,
    beta_values,
    seeds,
    eval_batch_size,
):
    """
    Experiment 1: Runs 1-shot across 50 seeds, records alpha_0 and top-1 accuracy.
    """
    num_classes = text_features.shape[0]
    alpha_records = []
    acc_records = []

    for seed in seeds:
        set_global_seed(seed)
        if val_fraction is None:
            train_idx = support_positions(train_labels_all, 1, seed)
            eval_features = eval_features_all
            eval_labels = eval_labels_all
        else:
            train_idx, val_idx = split_positions_by_class(
                train_labels_all, 1, seed, val_fraction
            )
            eval_features = train_features_all[val_idx].contiguous()
            eval_labels = train_labels_all[val_idx].contiguous()

        support_features = train_features_all[train_idx].to(device).float()
        support_labels = train_labels_all[train_idx].to(device).long()

        # Build visual centroids for 1-shot (V is just support_features indexed by class)
        counts = torch.bincount(support_labels, minlength=num_classes).float().unsqueeze(-1)
        sums = torch.zeros(num_classes, support_features.shape[-1], device=device, dtype=support_features.dtype)
        sums.index_add_(0, support_labels, support_features)
        valid = counts.squeeze(-1) > 0
        V = torch.zeros_like(sums)
        V[valid] = F.normalize(sums[valid] / counts[valid], dim=-1)

        alpha_0 = fast_surrogatemix_calibration(text_features, V, num_classes, alphas, beta_values)
        alpha_records.append(alpha_0)

        # Full 1-shot prototype evaluation
        gamma_1 = 0.30
        rho_1 = 0.50
        selector, _, _, _, _ = build_selector(
            text_features, support_features, support_labels, device, len(alphas), beta_values, rho_1
        )
        proto_before = F.normalize((1.0 - alpha_0) * text_features + alpha_0 * V, dim=-1)
        expanded_V, _ = selector.surrogatemix_expansion(
            V, proto_before, gamma_1, text_features, alpha_0, rho_1
        )
        proto_full = F.normalize((1.0 - alpha_0) * text_features + alpha_0 * expanded_V, dim=-1)

        acc = evaluate_exp2_variants_batched(
            proto_full.unsqueeze(0), eval_features, eval_labels, device, eval_batch_size
        )[0]
        acc_records.append(acc)

    alphas_np = np.array(alpha_records)
    accs_np = np.array(acc_records)

    q25, q50, q75 = np.percentile(alphas_np, [25, 50, 75])
    iqr = q75 - q25
    rate_le_025 = float(np.mean(alphas_np <= 0.25) * 100.0)

    # Pearson correlation
    if np.std(alphas_np) > 1e-9 and np.std(accs_np) > 1e-9:
        corr = float(np.corrcoef(alphas_np, accs_np)[0, 1])
    else:
        corr = 0.0

    stats = {
        "dataset": dataset_name,
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(iqr),
        "min": float(alphas_np.min()),
        "max": float(alphas_np.max()),
        "rate_le_025": rate_le_025,
        "corr": corr,
        "mean_acc": float(np.mean(accs_np)),
        "alphas_raw": alpha_records,
        "accs_raw": acc_records,
    }
    return stats


def run_experiment_2_dataset(
    dataset_name,
    text_features,
    train_features_all,
    train_labels_all,
    eval_features_all,
    eval_labels_all,
    val_fraction,
    device,
    alphas,
    beta_values,
    seeds,
    eval_batch_size,
):
    """
    Experiment 2: One-shot Support Atypicality Stress Test across 4 tiers:
    Random (0%), Mild (25%), Moderate (50%), Severe (100%).
    """
    num_classes = text_features.shape[0]

    if val_fraction is None:
        support_pool_features = train_features_all
        support_pool_labels = train_labels_all
        eval_features = eval_features_all
        eval_labels = eval_labels_all
    else:
        # Avoid data leakage when eval_dataset is None: split candidate pool from evaluation pool
        pool_idx, val_idx = split_positions_by_class(train_labels_all, 1, 1, val_fraction)
        support_pool_features = train_features_all[pool_idx].contiguous()
        support_pool_labels = train_labels_all[pool_idx].contiguous()
        eval_features = train_features_all[val_idx].contiguous()
        eval_labels = train_labels_all[val_idx].contiguous()

    rankings = compute_class_atypicality_rankings(support_pool_features, support_pool_labels, num_classes)

    tiers = [
        ("Random (0%)", 0.0, ["baseline"]),
        ("Mild (25%)", 0.25, ["typical", "atypical"]),
        ("Moderate (50%)", 0.50, ["typical", "atypical"]),
        ("Severe (100%)", 1.0, ["typical", "atypical"]),
    ]

    tier_results = []
    gamma_1 = 0.30
    rho_1 = 0.50

    for tier_name, ratio, conditions in tiers:
        for cond in conditions:
            seed_alphas = []
            seed_variant_accs = {v: [] for v in EXP2_VARIANTS}

            for seed in seeds:
                # Controlled random seed across all runs to guarantee deterministic perturbations
                set_global_seed(seed)

                train_idx = sample_stress_support_indices(
                    rankings, num_classes, ratio, "random" if cond == "baseline" else cond, seed
                )
                support_features = support_pool_features[train_idx].to(device).float()
                support_labels = torch.arange(num_classes, device=device, dtype=torch.long)
                V = support_features

                alpha_0 = fast_surrogatemix_calibration(text_features, V, num_classes, alphas, beta_values)
                seed_alphas.append(alpha_0)

                # Selector for surrogatemix expansion
                selector, _, _, _, _ = build_selector(
                    text_features, support_features, support_labels, device, len(alphas), beta_values, rho_1
                )

                # 1. SM-Exp with Fixed alpha=0.25 (anchor, expansion, and fusion all strictly use 0.25)
                # Synchronize random state so both variants receive identical Gaussian perturbations
                proto_before_fixed = F.normalize((1.0 - 0.25) * text_features + 0.25 * V, dim=-1)
                torch.manual_seed(seed + 999)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + 999)
                expanded_V_fixed, _ = selector.surrogatemix_expansion(
                    V, proto_before_fixed, gamma_1, text_features, 0.25, rho_1
                )

                # 2. Full ProtoFuse (anchor, expansion, and fusion all use alpha_0 from SM-Cal)
                proto_before_full = F.normalize((1.0 - alpha_0) * text_features + alpha_0 * V, dim=-1)
                torch.manual_seed(seed + 999)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + 999)
                expanded_V_full, _ = selector.surrogatemix_expansion(
                    V, proto_before_full, gamma_1, text_features, alpha_0, rho_1
                )

                # 5 Variants:
                # 0: Text-only (alpha=0)
                # 1: Fixed (alpha=0.25)
                # 2: SM-Cal only (alpha=alpha_0, visual=V)
                # 3: SM-Exp (Fixed alpha=0.25) (alpha=0.25, visual=expanded_V_fixed)
                # 4: Full ProtoFuse (alpha=alpha_0, visual=expanded_V_full)
                p_text = text_features
                p_fixed = F.normalize((1.0 - 0.25) * text_features + 0.25 * V, dim=-1)
                p_smcal = F.normalize((1.0 - alpha_0) * text_features + alpha_0 * V, dim=-1)
                p_smexp_fixed = F.normalize((1.0 - 0.25) * text_features + 0.25 * expanded_V_fixed, dim=-1)
                p_full = F.normalize((1.0 - alpha_0) * text_features + alpha_0 * expanded_V_full, dim=-1)

                stacked = torch.stack([p_text, p_fixed, p_smcal, p_smexp_fixed, p_full], dim=0)
                accs = evaluate_exp2_variants_batched(
                    stacked, eval_features, eval_labels, device, eval_batch_size
                )
                for v_idx, v_name in enumerate(EXP2_VARIANTS):
                    seed_variant_accs[v_name].append(accs[v_idx])

            mean_alpha = float(np.mean(seed_alphas))
            std_alpha = float(np.std(seed_alphas))
            mean_accs = {v: float(np.mean(seed_variant_accs[v])) for v in EXP2_VARIANTS}
            std_accs = {v: float(np.std(seed_variant_accs[v])) for v in EXP2_VARIANTS}
            delta_cal = mean_accs["SM-Cal only"] - mean_accs["Fixed (α=0.25)"]
            delta_full = mean_accs["Full ProtoFuse"] - mean_accs["SM-Exp (Fixed α=0.25)"]

            tier_results.append({
                "dataset": dataset_name,
                "tier": tier_name,
                "condition": cond.capitalize(),
                "mean_alpha": mean_alpha,
                "std_alpha": std_alpha,
                "seed_alphas": list(seed_alphas),
                **mean_accs,
                "std_accs": std_accs,
                "seed_variant_accs": {v: list(seed_variant_accs[v]) for v in EXP2_VARIANTS},
                "delta_cal": delta_cal,
                "delta_full": delta_full,
            })
    return tier_results


def build_exp1_summary_table(exp1_results):
    rows = []
    all_alphas_raw = []
    all_accs_raw = []
    for res in exp1_results:
        all_alphas_raw.extend(res.get("alphas_raw", []))
        all_accs_raw.extend(res.get("accs_raw", []))
        rows.append({
            "Dataset": res["dataset"],
            "Median α0": f"{res['median']:.2f}",
            "IQR [Q25, Q75]": f"[{res['q25']:.2f}, {res['q75']:.2f}]",
            "Min - Max": f"{res['min']:.2f} - {res['max']:.2f}",
            "Rate (α0 <= 0.25)": f"{res['rate_le_025']:.1f}%",
            "Corr(α0, Acc)": f"{res['corr']:+.2f}",
            "Mean 1-shot Acc": f"{res['mean_acc']:.2f}%",
        })

    # Overall summary computed directly from the pooled alpha_0 distribution
    if all_alphas_raw:
        alphas_np = np.array(all_alphas_raw)
        accs_np = np.array(all_accs_raw)
        q25, q50, q75 = np.percentile(alphas_np, [25, 50, 75])
        iqr = q75 - q25
        rate_le_025 = float(np.mean(alphas_np <= 0.25) * 100.0)
        if np.std(alphas_np) > 1e-9 and np.std(accs_np) > 1e-9:
            corr = float(np.corrcoef(alphas_np, accs_np)[0, 1])
        else:
            corr = 0.0
        mean_acc = float(np.mean(accs_np))
        min_alpha = float(alphas_np.min())
        max_alpha = float(alphas_np.max())
    else:
        q50, q25, q75, iqr, min_alpha, max_alpha, rate_le_025, corr, mean_acc = (0.0,) * 9

    rows.append({
        "Dataset": "Overall Pooled",
        "Median α0": f"{q50:.2f}",
        "IQR [Q25, Q75]": f"[{q25:.2f}, {q75:.2f}]",
        "Min - Max": f"{min_alpha:.2f} - {max_alpha:.2f}",
        "Rate (α0 <= 0.25)": f"{rate_le_025:.1f}%",
        "Corr(α0, Acc)": f"{corr:+.2f}",
        "Mean 1-shot Acc": f"{mean_acc:.2f}%",
    })

    df = pd.DataFrame(rows)
    return (
        f"Experiment 1: SM-Cal Coefficient Selection Analysis (One-shot across 50 seeds)\n"
        f"{df.to_string(index=False)}"
    )


def build_exp2_summary_table(exp2_results, datasets):
    """
    Summarizes Experiment 2: Stress test macro-averaged across datasets.
    """
    tier_cond_order = [
        ("Random (0%)", "Baseline"),
        ("Mild (25%)", "Typical"),
        ("Mild (25%)", "Atypical"),
        ("Moderate (50%)", "Typical"),
        ("Moderate (50%)", "Atypical"),
        ("Severe (100%)", "Typical"),
        ("Severe (100%)", "Atypical"),
    ]

    rows = []
    for tier, cond in tier_cond_order:
        matches = [
            r for r in exp2_results
            if r["tier"] == tier and r["condition"] == cond
        ]
        mean_alpha = float(np.mean([m["mean_alpha"] for m in matches]))
        fixed_acc = float(np.mean([m["Fixed (α=0.25)"] for m in matches]))
        smcal_acc = float(np.mean([m["SM-Cal only"] for m in matches]))
        delta_cal = float(np.mean([m["delta_cal"] for m in matches]))
        smexp_fixed = float(np.mean([m["SM-Exp (Fixed α=0.25)"] for m in matches]))
        full_acc = float(np.mean([m["Full ProtoFuse"] for m in matches]))
        delta_full = float(np.mean([m["delta_full"] for m in matches]))

        rows.append({
            "Stress Tier": tier,
            "Condition": cond,
            "Mean α0": f"{mean_alpha:.2f}",
            "Fixed(0.25)": f"{fixed_acc:.2f}%",
            "SM-Cal only": f"{smcal_acc:.2f}%",
            "Δ_Cal": f"{delta_cal:+.2f} pp",
            "SM-Exp(0.25)": f"{smexp_fixed:.2f}%",
            "Full ProtoFuse": f"{full_acc:.2f}%",
            "Δ_Full": f"{delta_full:+.2f} pp",
        })

    df = pd.DataFrame(rows)
    return (
        f"Experiment 2: One-shot Support Atypicality Stress Test (Macro Avg across {len(datasets)} Datasets)\n"
        f"{df.to_string(index=False)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="SM-Cal Analysis: Coefficient Selection & Support Atypicality Stress Test"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--exp1-seeds", default=",".join(str(s) for s in DEFAULT_EXP1_SEEDS))
    parser.add_argument("--exp2-seeds", default=",".join(str(s) for s in DEFAULT_EXP2_SEEDS))
    parser.add_argument("--betas", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--export-json-dir", default=str(REPO_ROOT / "outputs" / "ablation" / "smcal"), help="Directory to export typical.json and atypical.json")
    parser.add_argument("--skip-exp1", action="store_true", help="Skip Experiment 1 (coefficient analysis)")
    parser.add_argument("--skip-exp2", action="store_true", help="Skip Experiment 2 (atypicality stress test)")
    parser.add_argument("--output-json", default=None, help="Optional path to save JSON if needed")
    parsed, unknown = parser.parse_known_args()
    return parsed, parse_override_arguments(unknown)


def main():
    args, overrides = parse_args()
    config = merge_configs(load_config_file(args.config), overrides)
    exp1_seeds = parse_int_list(args.exp1_seeds)
    exp2_seeds = parse_int_list(args.exp2_seeds)
    alpha_steps = int(get_config_value(config, "model.alpha_steps", 101))
    if args.betas:
        beta_values = [float(b.strip()) for b in args.betas.split(",") if b.strip()]
    else:
        beta_values = get_config_value(config, "model.centroid_mix.beta_values", DEFAULT_BETAS)
    device_name = str(get_config_value(config, "training.device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    batch_size = int(get_config_value(config, "training.batch_size", 128))
    eval_batch_size = args.eval_batch_size or max(batch_size, 2048)
    num_workers = int(get_config_value(config, "data.num_workers", 4))

    alphas = torch.linspace(0, 1, alpha_steps, device=device)

    dataset_configs = table_dataset_configs(config)
    dataset_names = [table_name for table_name, _, _ in dataset_configs]

    print(f"\nRunning SM-Cal Analysis on {len(dataset_names)} dataset(s)...\n", flush=True)

    all_exp1_results = []
    all_exp2_results = []

    for table_name, dataset_config, _ in dataset_configs:
        dataset_cfg = copy.deepcopy(dataset_config)
        data_cfg = dataset_cfg.setdefault("data", {})
        data_cfg["kshot"] = 1
        data_cfg["seed"] = int(exp1_seeds[0] if exp1_seeds else 1)

        set_global_seed(int(data_cfg["seed"]))
        pipeline = ProtoFusePipeline(dataset_cfg)
        pipeline._prepare_directories()
        pipeline._load_dataset()
        pipeline._split_dataset()
        pipeline._initialize_trainer()

        dataset_root = pipeline.dataset_root
        classnames = pipeline.classnames
        val_fraction = pipeline.val_fraction
        print(
            f"{table_name}: root={dataset_root}, classes={len(classnames)}, device={device}",
            flush=True,
        )

        train_payload = pipeline._full_dataset_clip_features()
        train_features_all = train_payload["image_features"]
        train_labels_all = train_payload["labels"]
        text_features = train_payload["text_features"].to(device).float()

        if pipeline.val_fraction is None:
            eval_features_all, eval_labels_all = pipeline._cached_val_features()
        else:
            eval_features_all, eval_labels_all = train_features_all, train_labels_all

        if hasattr(pipeline.trainer, "clip_model"):
            del pipeline.trainer.clip_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Export typical.json and atypical.json (all samples in top/bottom 20% per class)
        if args.export_json_dir:
            image_paths = train_payload.get("paths", [p for p, _ in getattr(pipeline.dataset, "samples", [])])
            if pipeline.val_fraction is not None:
                pool_idx, _ = split_positions_by_class(train_labels_all, 1, 1, pipeline.val_fraction)
                support_pool_feats = train_features_all[pool_idx].contiguous()
                support_pool_lbls = train_labels_all[pool_idx].contiguous()
                pool_paths = [image_paths[i] for i in pool_idx.tolist()] if image_paths else None
            else:
                support_pool_feats = train_features_all
                support_pool_lbls = train_labels_all
                pool_paths = image_paths

            dataset_rankings = compute_class_atypicality_rankings(
                support_pool_feats, support_pool_lbls, text_features.shape[0]
            )
            export_typical_atypical_json(
                table_name,
                dataset_rankings,
                classnames,
                pool_paths,
                args.export_json_dir,
            )

        # Run Experiment 1
        if not args.skip_exp1:
            print(f"  [{table_name}] Running Exp 1: Coefficient Selection (50 seeds)...", flush=True)
            exp1_stat = run_experiment_1_dataset(
                table_name,
                text_features,
                train_features_all,
                train_labels_all,
                eval_features_all,
                eval_labels_all,
                val_fraction,
                device,
                alphas,
                beta_values,
                exp1_seeds,
                eval_batch_size,
            )
            all_exp1_results.append(exp1_stat)
            print(
                f"  [{table_name}] Exp 1 Done: Median α0 = {exp1_stat['median']:.2f}, "
                f"Rate(<=0.25) = {exp1_stat['rate_le_025']:.1f}%, Mean Acc = {exp1_stat['mean_acc']:.2f}%",
                flush=True,
            )

        # Run Experiment 2
        if not args.skip_exp2:
            print(f"  [{table_name}] Running Exp 2: Atypicality Stress Test ({len(exp2_seeds)} seeds)...", flush=True)
            exp2_stats = run_experiment_2_dataset(
                table_name,
                text_features,
                train_features_all,
                train_labels_all,
                eval_features_all,
                eval_labels_all,
                val_fraction,
                device,
                alphas,
                beta_values,
                exp2_seeds,
                eval_batch_size,
            )
            all_exp2_results.extend(exp2_stats)
            print(f"  [{table_name}] Exp 2 Done.", flush=True)

    if not args.skip_exp1 and all_exp1_results:
        print("\n" + "=" * 95, flush=True)
        print(build_exp1_summary_table(all_exp1_results), flush=True)
        print("=" * 95 + "\n", flush=True)

    if not args.skip_exp2 and all_exp2_results:
        print("\n" + "=" * 95, flush=True)
        print(build_exp2_summary_table(all_exp2_results, dataset_names), flush=True)
        print("=" * 95 + "\n", flush=True)

    if args.output_json:
        out_path = resolve_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "ablation": "smcal_analysis",
                "exp1_seeds": exp1_seeds,
                "exp2_seeds": exp2_seeds,
                "exp1_results": all_exp1_results,
                "exp2_results": all_exp2_results,
            }, f, indent=2)
        print(f"Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

