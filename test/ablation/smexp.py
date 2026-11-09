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

from protofuse import split_indices_for
from protofuse_ablation_tables import (
    build_selector,
    fused_prototypes,
    table_dataset_configs,
)
from protofuse_candidates import (
    DEFAULT_CONFIG,
    parse_int_list,
    resolve_path,
)
from src.pipelines.protofuse import ProtoFusePipeline
from utils import (
    get_config_value,
    load_config_file,
    merge_configs,
    parse_override_arguments,
    set_global_seed,
)


DEFAULT_KSHOTS = [2, 4, 8, 16]
DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]
VARIANTS = (
    "A. LOO only",
    "B. LOO + SM-Exp",
    "C. LOO + Random Expansion",
    "D. Full ProtoFuse",
    "E. Random Expansion + Recalibration",
)


def class_feature_tensor(features, labels, num_classes, kshot):
    by_class = []
    for class_idx in range(num_classes):
        indices = torch.nonzero(labels.eq(class_idx), as_tuple=False).flatten()
        if indices.numel() < kshot:
            raise RuntimeError(
                f"Class {class_idx} has {indices.numel()} support samples, expected {kshot}."
            )
        by_class.append(features[indices[:kshot]])
    return torch.stack(by_class, dim=0)


def generate_matched_random_expansion_vectorized(visual, expanded_visual, num_trials=5, generator=None):
    """
    Constructs unit random vectors with the exact same angular displacement
    relative to visual as expanded_visual has, vectorized across all trials and classes.
    visual: (C, D)
    expanded_visual: (C, D)
    Returns: (num_trials, C, D) tensor
    """
    cos_theta = (visual * expanded_visual).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=0.0))

    C, D = visual.shape
    if generator is not None:
        u = torch.randn((num_trials, C, D), device=visual.device, dtype=visual.dtype, generator=generator)
    else:
        u = torch.randn((num_trials, C, D), device=visual.device, dtype=visual.dtype)

    vis_expanded = visual.unsqueeze(0)  # (1, C, D)
    u_proj = (u * vis_expanded).sum(dim=-1, keepdim=True) * vis_expanded
    u_perp = u - u_proj
    e_perp = F.normalize(u_perp, dim=-1, eps=1e-12)
    v_rand = cos_theta.unsqueeze(0) * vis_expanded + sin_theta.unsqueeze(0) * e_perp
    return F.normalize(v_rand, dim=-1)


def fast_loo_curve_from_precomputed(alphas, S_T, S_V, cos_phi, targets):
    """
    Algebraically fast evaluation of LOO accuracy curve over alpha candidates.
    S_T: (K, C, C) -> held dot text
    S_V: (K, C, C) -> held dot visual_minus
    cos_phi: (K, C) -> text dot visual_minus
    alphas: (A,)
    """
    a = alphas.view(-1, 1, 1, 1)
    om_a = 1.0 - a

    # Denominator: ||(1-a)*T + a*V_minus||
    denom = torch.sqrt((om_a ** 2 + a ** 2 + 2.0 * a * om_a * cos_phi.unsqueeze(0).unsqueeze(2)).clamp(min=1e-12))
    logits = (om_a * S_T.unsqueeze(0) + a * S_V.unsqueeze(0)) / denom
    preds = logits.argmax(dim=-1)
    return preds.eq(targets.view(1, 1, -1)).sum(dim=(1, 2)).float()


def evaluate_all_variants_batched(
    stacked_prototypes,
    eval_features,
    eval_labels,
    device,
    batch_size=2048,
):
    """
    Evaluates ALL 13 variants (A, B, D, 5x C, 5x E) in a single batched pass over eval_features.
    stacked_prototypes: (V, C, D) where V = 3 + 2*M
    Returns:
      accs: (V,) tensor of accuracies in percentage
      margins_A: (N,)
      margins_B: (N,)
      margins_D: (N,)
      preds_A: (N,)
      preds_B: (N,)
      preds_D: (N,)
    """
    V, C, D = stacked_prototypes.shape
    total = int(eval_labels.numel())
    batch_size = max(1, int(batch_size))
    stacked_flat = stacked_prototypes.view(V * C, D)

    correct_counts = torch.zeros(V, device=device, dtype=torch.long)
    all_preds_A, all_preds_B, all_preds_D = [], [], []
    all_margins_A, all_margins_B, all_margins_D = [], [], []

    eval_labels_dev = eval_labels.to(device).long()

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            features = eval_features[start:end].to(device, non_blocking=True).float()
            labels = eval_labels_dev[start:end]
            curr_b = int(labels.shape[0])

            # Single GEMM for all V variants: (B, V*C) -> (B, V, C) -> (V, B, C)
            batch_logits = (features @ stacked_flat.T).view(curr_b, V, C).permute(1, 0, 2)
            preds = batch_logits.argmax(dim=-1)  # (V, B)
            correct_counts += preds.eq(labels.unsqueeze(0)).sum(dim=1)

            # Store predictions for transitions
            all_preds_A.append(preds[0].cpu())
            all_preds_B.append(preds[1].cpu())
            all_preds_D.append(preds[2].cpu())

            # Compute margins for A, B, D in parallel: m(z) = logit_y - max_{j!=y} logit_j
            batch_idx = torch.arange(curr_b, device=device)
            for v_idx, m_list in [(0, all_margins_A), (1, all_margins_B), (2, all_margins_D)]:
                l_v = batch_logits[v_idx]
                true_l = l_v[batch_idx, labels]
                comp_l = l_v.clone()
                comp_l[batch_idx, labels] = -float("inf")
                margin_v = true_l - comp_l.max(dim=-1).values
                m_list.append(margin_v.cpu())

    accuracies = correct_counts.float().mul_(100.0 / max(1, total)).cpu()
    preds_A = torch.cat(all_preds_A, dim=0)
    preds_B = torch.cat(all_preds_B, dim=0)
    preds_D = torch.cat(all_preds_D, dim=0)
    margins_A = torch.cat(all_margins_A, dim=0)
    margins_B = torch.cat(all_margins_B, dim=0)
    margins_D = torch.cat(all_margins_D, dim=0)

    return accuracies, margins_A, margins_B, margins_D, preds_A, preds_B, preds_D


def compute_transitions(labels, preds_A, preds_B, preds_D):
    labels = labels.cpu()
    correct_A = preds_A.eq(labels)
    correct_B = preds_B.eq(labels)
    correct_D = preds_D.eq(labels)
    n = float(labels.numel())

    corr_AB = float((~correct_A & correct_B).sum().item()) / n * 100.0
    deg_AB = float((correct_A & ~correct_B).sum().item()) / n * 100.0

    corr_BD = float((~correct_B & correct_D).sum().item()) / n * 100.0
    deg_BD = float((correct_B & ~correct_D).sum().item()) / n * 100.0

    corr_AD = float((~correct_A & correct_D).sum().item()) / n * 100.0
    deg_AD = float((correct_A & ~correct_D).sum().item()) / n * 100.0

    return {
        "A_to_B": {"corrected": corr_AB, "degraded": deg_AB, "net": corr_AB - deg_AB},
        "B_to_D": {"corrected": corr_BD, "degraded": deg_BD, "net": corr_BD - deg_BD},
        "A_to_D": {"corrected": corr_AD, "degraded": deg_AD, "net": corr_AD - deg_AD},
    }


def compute_margin_stats(labels, margins_A, margins_B, margins_D, preds_A, preds_B):
    labels = labels.cpu()
    m_A = float(margins_A.mean().item())
    m_B = float(margins_B.mean().item())
    m_D = float(margins_D.mean().item())

    delta_AB = float((margins_B - margins_A).mean().item())
    delta_BD = float((margins_D - margins_B).mean().item())
    delta_AD = float((margins_D - margins_A).mean().item())

    correct_A = preds_A.eq(labels)
    correct_B = preds_B.eq(labels)
    degraded_mask = correct_A & ~correct_B
    deg_count = int(degraded_mask.sum().item())

    if deg_count > 0:
        deg_m_A = float(margins_A[degraded_mask].mean().item())
        deg_m_B = float(margins_B[degraded_mask].mean().item())
        deg_m_D = float(margins_D[degraded_mask].mean().item())
        deg_delta_BD = deg_m_D - deg_m_B
    else:
        deg_m_A = float("nan")
        deg_m_B = float("nan")
        deg_m_D = float("nan")
        deg_delta_BD = float("nan")

    return {
        "mean_m_A": m_A,
        "mean_m_B": m_B,
        "mean_m_D": m_D,
        "delta_m_AB": delta_AB,
        "delta_m_BD": delta_BD,
        "delta_m_AD": delta_AD,
        "degraded_subset": {
            "count": deg_count,
            "mean_m_A": deg_m_A,
            "mean_m_B": deg_m_B,
            "mean_m_D": deg_m_D,
            "delta_m_BD": deg_delta_BD,
        },
    }


def evaluate_single_run_fast(
    text_features,
    support_features,
    support_labels,
    eval_features,
    eval_labels,
    device,
    alpha_steps,
    beta_values,
    rho_k,
    gamma_k,
    kshot,
    eval_batch_size,
    num_random_trials=5,
    seed=1,
):
    selector, text, visual, support_features, support_labels = build_selector(
        text_features,
        support_features,
        support_labels,
        device,
        alpha_steps,
        beta_values,
        rho_k,
    )
    class_features = class_feature_tensor(
        support_features,
        support_labels,
        text.shape[0],
        kshot,
    )

    num_classes, _, D = class_features.shape
    targets = torch.arange(num_classes, device=device)
    class_sums = class_features.sum(dim=1)

    # Precompute fold holdout tensors
    held = F.normalize(class_features.permute(1, 0, 2), dim=-1)  # (K, C, D)
    base_v_minus = (class_sums.unsqueeze(0) - held) / float(kshot - 1)  # (K, C, D)
    v_minus_norm = F.normalize(base_v_minus, dim=-1)

    # Precompute S_T once for ALL alphas, ALL variants, ALL random trials!
    # S_T: (K, C, C) -> held[k, c] dot text[j]
    S_T = torch.einsum("kcd,jd->kcj", held, text)

    # 1. Initial LOO curve (LOO-only -> alpha_init)
    S_V_init = torch.einsum("kcd,kjd->kcj", held, v_minus_norm)
    cos_phi_init = (text.unsqueeze(0) * v_minus_norm).sum(dim=-1)
    loo_curve = fast_loo_curve_from_precomputed(selector.alphas, S_T, S_V_init, cos_phi_init, targets)
    alpha_init = float(selector.alphas[int(loo_curve.argmax().item())].item())

    # 2. SM-Exp expansion on full support set (for evaluating Variants B & D on eval set)
    proto_before = F.normalize((1.0 - alpha_init) * text + alpha_init * visual, dim=-1)
    expanded_visual, _ = selector.surrogatemix_expansion(
        visual,
        proto_before,
        gamma_k,
        text,
        alpha_init,
        rho_k,
    )

    # 3. Recalibration via strict leave-one-out (LOO):
    # SM-Exp must run independently for each fold r using only the other K-1 samples (v_minus_norm[r])
    # and a fold-specific anchor prototype, ensuring zero leakage of held[r].
    v_minus_smexp_list = []
    for r in range(kshot):
        v_r = v_minus_norm[r]
        proto_before_r = F.normalize((1.0 - alpha_init) * text + alpha_init * v_r, dim=-1)
        expanded_v_r, _ = selector.surrogatemix_expansion(
            v_r,
            proto_before_r,
            gamma_k,
            text,
            alpha_init,
            rho_k,
        )
        v_minus_smexp_list.append(expanded_v_r)
    v_minus_smexp = torch.stack(v_minus_smexp_list, dim=0)  # (K, C, D)

    S_V_smexp = torch.einsum("kcd,kjd->kcj", held, v_minus_smexp)
    cos_phi_smexp = (text.unsqueeze(0) * v_minus_smexp).sum(dim=-1)
    recalibrated_curve = fast_loo_curve_from_precomputed(selector.alphas, S_T, S_V_smexp, cos_phi_smexp, targets)
    alpha_final = float(selector.alphas[int(recalibrated_curve.argmax().item())].item())

    # 4. Generate random expansions matching SM-Exp angular displacement for full support set
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 42)
    random_visuals = generate_matched_random_expansion_vectorized(
        visual,
        expanded_visual,
        num_trials=num_random_trials,
        generator=gen,
    )  # (M, C, D)

    # 5. Strict LOO Recalibration for all M random trials:
    # For each trial m, match angular displacement fold-by-fold from fold-specific SM-Exp (v_minus_smexp),
    # ensuring zero data leakage and exact structural equivalence to Full ProtoFuse.
    alphas_rand = []
    for m in range(num_random_trials):
        v_rand_folds = generate_matched_random_expansion_vectorized(
            v_minus_norm.reshape(-1, D),
            v_minus_smexp.reshape(-1, D),
            num_trials=1,
            generator=gen,
        )  # (1, K*C, D)
        v_minus_rand_m = v_rand_folds.squeeze(0).reshape(kshot, num_classes, D)

        S_V_rand = torch.einsum("kcd,kjd->kcj", held, v_minus_rand_m)
        cos_phi_rand = (text.unsqueeze(0) * v_minus_rand_m).sum(dim=-1)
        curve_rand = fast_loo_curve_from_precomputed(selector.alphas, S_T, S_V_rand, cos_phi_rand, targets)
        alphas_rand.append(float(selector.alphas[int(curve_rand.argmax().item())].item()))

    # Build all prototypes for the 13 variants:
    # 0: A (LOO only)
    # 1: B (LOO + SM-Exp)
    # 2: D (Full ProtoFuse)
    # 3..3+M-1: C (LOO + Random Exp, alpha_init)
    # 3+M..3+2M-1: E (Random Exp + Recal, alpha_rand)
    proto_A = fused_prototypes(text, visual, alpha_init)
    proto_B = fused_prototypes(text, expanded_visual, alpha_init)
    proto_D = fused_prototypes(text, expanded_visual, alpha_final)

    protos_C = [fused_prototypes(text, random_visuals[m], alpha_init) for m in range(num_random_trials)]
    protos_E = [fused_prototypes(text, random_visuals[m], alphas_rand[m]) for m in range(num_random_trials)]

    stacked_prototypes = torch.stack([proto_A, proto_B, proto_D, *protos_C, *protos_E], dim=0)

    # 6. Evaluate all variants in a SINGLE batched GEMM pass over eval_features
    all_accs, margins_A, margins_B, margins_D, preds_A, preds_B, preds_D = evaluate_all_variants_batched(
        stacked_prototypes,
        eval_features,
        eval_labels,
        device,
        batch_size=eval_batch_size,
    )

    acc_A = float(all_accs[0].item())
    acc_B = float(all_accs[1].item())
    acc_D = float(all_accs[2].item())
    acc_C_trials = [float(all_accs[3 + m].item()) for m in range(num_random_trials)]
    acc_E_trials = [float(all_accs[3 + num_random_trials + m].item()) for m in range(num_random_trials)]
    acc_C = float(np.mean(acc_C_trials))
    acc_C_std = float(np.std(acc_C_trials))
    acc_E = float(np.mean(acc_E_trials))
    acc_E_std = float(np.std(acc_E_trials))

    transitions = compute_transitions(eval_labels, preds_A, preds_B, preds_D)
    margin_stats = compute_margin_stats(eval_labels, margins_A, margins_B, margins_D, preds_A, preds_B)

    accuracies = {
        VARIANTS[0]: acc_A,
        VARIANTS[1]: acc_B,
        VARIANTS[2]: acc_C,
        VARIANTS[3]: acc_D,
        VARIANTS[4]: acc_E,
    }
    alphas_summary = {
        VARIANTS[0]: alpha_init,
        VARIANTS[1]: alpha_init,
        VARIANTS[2]: alpha_init,
        VARIANTS[3]: alpha_final,
        VARIANTS[4]: float(np.mean(alphas_rand)),
    }

    return {
        "accuracies": accuracies,
        "alphas": alphas_summary,
        "random_control_details": {
            "C_trials": acc_C_trials,
            "C_std": acc_C_std,
            "E_trials": acc_E_trials,
            "E_std": acc_E_std,
            "alphas_rand": alphas_rand,
        },
        "transitions": transitions,
        "margins": margin_stats,
    }


def format_stat(mean_val, std_val):
    return f"{mean_val:.2f} ± {std_val:.2f}"


def build_controlled_ablation_table(title, summary, datasets):
    rows = []
    for variant in VARIANTS:
        row = {"Variant": variant}
        for d in datasets:
            row[d] = format_stat(summary[variant][d]["mean"], summary[variant][d]["std"])
        row["Avg."] = format_stat(summary[variant]["Avg."]["mean"], summary[variant]["Avg."]["std"])
        rows.append(row)

    diff_rows = [
        ("Δ_direction (D - E)", "D", "E"),
        ("Δ_recal (D - B)", "D", "B"),
        ("Δ_total (D - A)", "D", "A"),
    ]
    for label, var1, var2 in diff_rows:
        row = {"Variant": label}
        for d in datasets:
            val = summary[VARIANTS[ord(var1) - ord('A')]][d]["mean"] - summary[VARIANTS[ord(var2) - ord('A')]][d]["mean"]
            row[d] = f"{val:+.2f}"
        val_avg = summary[VARIANTS[ord(var1) - ord('A')]]["Avg."]["mean"] - summary[VARIANTS[ord(var2) - ord('A')]]["Avg."]["mean"]
        row["Avg."] = f"{val_avg:+.2f}"
        rows.append(row)

    df = pd.DataFrame(rows, columns=["Variant", *datasets, "Avg."])
    return f"{title}\n{df.to_string(index=False)}"


def build_transitions_table(trans_summary, datasets):
    rows = []
    for d in [*datasets, "Avg."]:
        rows.append({
            "Dataset": d,
            "A->B Corr(%)": f"{trans_summary[d]['A_to_B']['corrected']['mean']:.2f}",
            "A->B Deg(%)": f"{trans_summary[d]['A_to_B']['degraded']['mean']:.2f}",
            "A->B Net(%)": f"{trans_summary[d]['A_to_B']['net']['mean']:+.2f}",
            "B->D Corr(%)": f"{trans_summary[d]['B_to_D']['corrected']['mean']:.2f}",
            "B->D Deg(%)": f"{trans_summary[d]['B_to_D']['degraded']['mean']:.2f}",
            "B->D Net(%)": f"{trans_summary[d]['B_to_D']['net']['mean']:+.2f}",
            "A->D Corr(%)": f"{trans_summary[d]['A_to_D']['corrected']['mean']:.2f}",
            "A->D Deg(%)": f"{trans_summary[d]['A_to_D']['degraded']['mean']:.2f}",
            "A->D Net(%)": f"{trans_summary[d]['A_to_D']['net']['mean']:+.2f}",
        })
    df = pd.DataFrame(rows)
    return (
        f"Experiment 2A: Error Transition Rates (Multi-shot Macro Avg %)\n"
        f"{df.to_string(index=False)}"
    )


def build_margins_table(margin_summary, datasets):
    rows = []
    for d in [*datasets, "Avg."]:
        deg_val = margin_summary[d]["degraded_delta_BD"]["mean"]
        deg_str = f"{deg_val:+.4f}" if not math.isnan(deg_val) else "N/A"
        rows.append({
            "Dataset": d,
            "Mean m(A)": f"{margin_summary[d]['mean_m_A']['mean']:.4f}",
            "Mean m(B)": f"{margin_summary[d]['mean_m_B']['mean']:.4f}",
            "Mean m(D)": f"{margin_summary[d]['mean_m_D']['mean']:.4f}",
            "Δm (A->B)": f"{margin_summary[d]['delta_m_AB']['mean']:+.4f}",
            "Δm (B->D)": f"{margin_summary[d]['delta_m_BD']['mean']:+.4f}",
            "Δm (A->D)": f"{margin_summary[d]['delta_m_AD']['mean']:+.4f}",
            "Degraded Δm(B->D)": deg_str,
        })
    df = pd.DataFrame(rows)
    return (
        f"Experiment 2B: Classification Margin Analysis (m(z) = z^T p_y - max_{{j!=y}} z^T p_j)\n"
        f"{df.to_string(index=False)}"
    )


def run_dataset(args, table_name, config):
    kshots = parse_int_list(args.kshots)
    seeds = parse_int_list(args.seeds)
    alpha_steps = int(get_config_value(config, "model.alpha_steps", 101))
    beta_values = get_config_value(config, "model.centroid_mix.beta_values", None)
    rho_cfg = get_config_value(config, "model.rho", None)
    lambda_rho = float(get_config_value(config, "model.lambda_rho", 0.50))
    gamma_max = float(get_config_value(config, "model.gamma_max", 0.45))
    lambda_gamma = float(get_config_value(config, "model.lambda_gamma", 0.30))
    eval_batch_size = args.eval_batch_size or max(int(get_config_value(config, "training.batch_size", 128)), 2048)

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
        f"{table_name}: root={dataset_root}, classes={len(classnames)}, "
        f"kshots={kshots}, seeds={seeds}, device={device}",
        flush=True,
    )

    # Load precomputed or cached CLIP features
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

    raw_results = []
    for kshot in kshots:
        # Compute shot-specific expansion hyperparameters according to manuscript specification:
        # rho_K = min(1.0, lambda_rho / sqrt(K)), gamma_K = min(gamma_max, lambda_gamma / sqrt(K))
        if rho_cfg is not None:
            rho_k = float(rho_cfg)
        else:
            rho_k = min(1.0, lambda_rho / math.sqrt(kshot))
        gamma_k = min(gamma_max, lambda_gamma / math.sqrt(kshot))

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

            result = evaluate_single_run_fast(
                text_features,
                train_features_all[train_idx].contiguous(),
                train_labels_all[train_idx].contiguous(),
                eval_features,
                eval_labels,
                device,
                alpha_steps,
                beta_values,
                rho_k,
                gamma_k,
                kshot,
                eval_batch_size,
                num_random_trials=args.num_random_trials,
                seed=seed,
            )
            raw_results.append({
                "dataset": table_name,
                "kshot": int(kshot),
                "seed": int(seed),
                "rho_k": float(rho_k),
                "gamma_k": float(gamma_k),
                **result,
            })
            accs = result["accuracies"]
            del_dir = accs[VARIANTS[3]] - accs[VARIANTS[4]]
            del_rec = accs[VARIANTS[3]] - accs[VARIANTS[1]]
            print(
                f"  [{table_name} k={kshot:2d}, s={seed:5d} (rho={rho_k:.3f})] "
                f"A: {accs[VARIANTS[0]]:.2f}% | "
                f"B: {accs[VARIANTS[1]]:.2f}% | "
                f"C: {accs[VARIANTS[2]]:.2f}% | "
                f"D: {accs[VARIANTS[3]]:.2f}% | "
                f"E: {accs[VARIANTS[4]]:.2f}% | "
                f"Δ_dir: {del_dir:+.2f} | Δ_recal: {del_rec:+.2f}",
                flush=True,
            )
    return raw_results, str(dataset_root)


def aggregate_all_results(raw_all, datasets, kshots, seeds):
    summary_ablation = {v: {} for v in VARIANTS}
    trans_summary = {}
    margin_summary = {}

    kshot_set = set(kshots)

    # 1. Ablation Summary
    for variant in VARIANTS:
        all_dataset_overall_seed_means = {s: [] for s in seeds}
        for d in datasets:
            seed_means = []
            for s in seeds:
                vals = [
                    r["accuracies"][variant]
                    for r in raw_all
                    if r["dataset"] == d and r["kshot"] in kshot_set and r["seed"] == s
                ]
                mean_s = float(np.mean(vals))
                seed_means.append(mean_s)
                all_dataset_overall_seed_means[s].append(mean_s)
            summary_ablation[variant][d] = {
                "mean": float(np.mean(seed_means)),
                "std": float(np.std(seed_means)),
            }
        overall_seed_macros = [float(np.mean(all_dataset_overall_seed_means[s])) for s in seeds]
        summary_ablation[variant]["Avg."] = {
            "mean": float(np.mean(overall_seed_macros)),
            "std": float(np.std(overall_seed_macros)),
        }

    # 2. Transition Summary
    for d in datasets:
        d_rows = [r for r in raw_all if r["dataset"] == d]
        trans_summary[d] = {}
        for tr in ["A_to_B", "B_to_D", "A_to_D"]:
            trans_summary[d][tr] = {}
            for met in ["corrected", "degraded", "net"]:
                vals = [r["transitions"][tr][met] for r in d_rows]
                trans_summary[d][tr][met] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    # Macro average across datasets for transitions
    trans_summary["Avg."] = {}
    for tr in ["A_to_B", "B_to_D", "A_to_D"]:
        trans_summary["Avg."][tr] = {}
        for met in ["corrected", "degraded", "net"]:
            vals = [trans_summary[d][tr][met]["mean"] for d in datasets]
            trans_summary["Avg."][tr][met] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    # 3. Margin Summary
    for d in datasets:
        d_rows = [r for r in raw_all if r["dataset"] == d]
        # Sample-weighted aggregation for degraded subset margin delta
        deg_counts = [r["margins"]["degraded_subset"]["count"] for r in d_rows]
        deg_deltas = [r["margins"]["degraded_subset"]["delta_m_BD"] for r in d_rows]
        total_deg = sum(deg_counts)
        if total_deg > 0:
            valid_weighted_deltas = [
                c * val for c, val in zip(deg_counts, deg_deltas)
                if not math.isnan(val)
            ]
            weighted_deg_delta = float(sum(valid_weighted_deltas) / total_deg)
        else:
            weighted_deg_delta = float("nan")

        margin_summary[d] = {
            "mean_m_A": {"mean": float(np.mean([r["margins"]["mean_m_A"] for r in d_rows]))},
            "mean_m_B": {"mean": float(np.mean([r["margins"]["mean_m_B"] for r in d_rows]))},
            "mean_m_D": {"mean": float(np.mean([r["margins"]["mean_m_D"] for r in d_rows]))},
            "delta_m_AB": {"mean": float(np.mean([r["margins"]["delta_m_AB"] for r in d_rows]))},
            "delta_m_BD": {"mean": float(np.mean([r["margins"]["delta_m_BD"] for r in d_rows]))},
            "delta_m_AD": {"mean": float(np.mean([r["margins"]["delta_m_AD"] for r in d_rows]))},
            "degraded_delta_BD": {"mean": weighted_deg_delta},
        }

    # Macro average across datasets for margins (using nanmean to avoid biasing by datasets with 0 degraded samples)
    margin_summary["Avg."] = {
        key: {
            "mean": float(
                np.nanmean([margin_summary[d][key]["mean"] for d in datasets])
            )
        }
        for key in [
            "mean_m_A",
            "mean_m_B",
            "mean_m_D",
            "delta_m_AB",
            "delta_m_BD",
            "delta_m_AD",
            "degraded_delta_BD",
        ]
    }

    return summary_ablation, trans_summary, margin_summary

    return summary_ablation, trans_summary, margin_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled SM-Exp Ablation & Error / Classification Margin Analysis"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--kshots", default=",".join(str(v) for v in DEFAULT_KSHOTS))
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--num-random-trials", type=int, default=5, help="Number of random directions per seed")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--output-json", default=None, help="Optional path to save JSON if needed")
    parsed, unknown = parser.parse_known_args()
    return parsed, parse_override_arguments(unknown)


def main():
    args, overrides = parse_args()
    config = merge_configs(load_config_file(args.config), overrides)
    kshots = parse_int_list(args.kshots)
    seeds = parse_int_list(args.seeds)

    raw_all = []
    dataset_roots = {}
    dataset_configs = table_dataset_configs(config)
    dataset_names = [table_name for table_name, _, _ in dataset_configs]

    print(f"\nRunning Controlled SM-Exp Ablation on {len(dataset_names)} dataset(s)...\n", flush=True)

    for table_name, dataset_config, _ in dataset_configs:
        dataset_raw, dataset_root = run_dataset(args, table_name, dataset_config)
        raw_all.extend(dataset_raw)
        dataset_roots[table_name] = dataset_root

    summary_ablation, trans_summary, margin_summary = aggregate_all_results(
        raw_all, dataset_names, kshots, seeds
    )

    print("\n" + "=" * 95, flush=True)
    print(
        build_controlled_ablation_table(
            f"Table: Controlled SM-Exp Ablation (Macro Avg across K={{{','.join(str(k) for k in kshots)}}})",
            summary_ablation,
            dataset_names,
        ),
        flush=True,
    )
    print("=" * 95 + "\n", flush=True)

    print("=" * 95, flush=True)
    print(build_transitions_table(trans_summary, dataset_names), flush=True)
    print("=" * 95 + "\n", flush=True)

    print("=" * 95, flush=True)
    print(build_margins_table(margin_summary, dataset_names), flush=True)
    print("=" * 95 + "\n", flush=True)

    if args.output_json:
        out_path = resolve_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "ablation": "controlled_smexp",
                "datasets": dataset_roots,
                "kshots": kshots,
                "seeds": seeds,
                "summary_ablation": summary_ablation,
                "transitions_summary": trans_summary,
                "margin_summary": margin_summary,
                "raw": raw_all,
            }, f, indent=2)
        print(f"Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
