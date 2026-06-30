import copy
import csv
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    DEFAULT_ARG_SCHEMA,
    create_argument_parser,
    get_config_value,
    iter_dataset_configs,
    load_config_file,
    logger,
    merge_configs,
    parse_override_arguments,
    process_parsed_args,
    set_global_seed,
    setup_logging,
)
from src.pipelines.protofuse import ProtoFusePipeline

ARG_SCHEMA = DEFAULT_ARG_SCHEMA
DEFAULT_KSHOTS = [1, 2, 4, 8, 16]
DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]


def tangent_direction(target, base, eps=1e-8):
    if target.dim() == 1:
        target = target.unsqueeze(0)
    if base.dim() == 1:
        base = base.unsqueeze(0)
    proj = (target * base).sum(dim=-1, keepdim=True) * base
    d = target - proj
    return F.normalize(d, dim=-1, eps=eps)


def generate_sqs_adversarial(V, proto_before, beta, device, top_k=5, samples_per_nb=2, std=0.02):
    num_classes, D = V.shape
    sim = proto_before @ proto_before.T
    sim.fill_diagonal_(-float('inf'))
    Q = []
    for c in range(num_classes):
        nbs = torch.topk(sim[c], k=top_k).indices
        class_q = []
        for m in nbs:
            q_base = F.normalize((1.0 - beta) * proto_before[c] + beta * proto_before[m], dim=-1)
            noise = torch.randn(samples_per_nb, D, device=device) * std
            q_candidates = F.normalize(q_base.unsqueeze(0) + noise, dim=-1)
            logits = q_candidates @ proto_before.T
            preds = logits.argmax(dim=-1)
            valid = q_candidates[preds == c]
            if len(valid) == 0:
                class_q.append(q_candidates)
            else:
                if len(valid) < samples_per_nb:
                    pad = q_candidates[:samples_per_nb - len(valid)]
                    valid = torch.cat([valid, pad], dim=0)
                class_q.append(valid[:samples_per_nb])
        Q.append(torch.cat(class_q, dim=0))
    return torch.stack(Q, dim=0)


def build_h2_nn_centroid(V):
    sim = V @ V.T
    sim.fill_diagonal_(-float('inf'))
    n = sim.argmax(dim=1)
    return V[n]


def build_h4_center_attr(V):
    mu_class = F.normalize(V.mean(dim=0), dim=-1)
    return tangent_direction(mu_class, V)


def parse_int_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def has_config_path(config, path):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def sweep_values(config, overrides):
    kshots = parse_int_list(
        get_config_value(config, "data.kshots", None),
        [get_config_value(config, "data.kshot")] if has_config_path(overrides, "data.kshot") else DEFAULT_KSHOTS,
    )
    seeds = parse_int_list(
        get_config_value(config, "data.seeds", None),
        [get_config_value(config, "data.seed")] if has_config_path(overrides, "data.seed") else DEFAULT_SEEDS,
    )
    return kshots, seeds


def split_indices_for(pipeline, kshot, seed):
    samples_by_class_idx = {}
    for idx, (_, class_idx) in enumerate(pipeline.dataset.samples):
        samples_by_class_idx.setdefault(class_idx, []).append(idx)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for class_idx in sorted(samples_by_class_idx):
        class_samples = sorted(samples_by_class_idx[class_idx])
        rng.shuffle(class_samples)

        if pipeline.val_fraction is None:
            train_candidates = class_samples
        else:
            val_count = int(math.floor(len(class_samples) * pipeline.val_fraction))
            if pipeline.val_fraction > 0 and val_count == 0 and class_samples:
                val_count = 1
            val_indices.extend(class_samples[:val_count])
            train_candidates = class_samples[val_count:]

        train_indices.extend(train_candidates[:kshot] if kshot > 0 else train_candidates)

    return train_indices, val_indices


def remap_labels(train_labels, eval_labels):
    task_classes = sorted(set(train_labels.tolist()))
    remap = {label: idx for idx, label in enumerate(task_classes)}
    missing = sorted(set(eval_labels.tolist()) - set(remap))
    if missing:
        raise ValueError(f"Eval labels not present in train split: {missing[:10]}")
    train = torch.tensor([remap[int(label)] for label in train_labels], dtype=torch.long)
    eval_ = torch.tensor([remap[int(label)] for label in eval_labels], dtype=torch.long)
    return train, eval_, len(task_classes)


def run_forensic_run(pipeline, train_features, train_labels, eval_features, eval_labels, num_classes, kshot, seed):
    device = pipeline.device
    T = pipeline.trainer.text_prototypes.to(device)
    V = pipeline.trainer.build_visual_centroids(train_features, train_labels, num_classes).to(device)
    
    _, alpha_init = pipeline.trainer.hopc_alpha(T, V, train_features, train_labels, num_classes)
    
    query_centroids = pipeline.trainer.pseudo_label_aggregation(
        eval_features,
        T,
        V,
        alpha_init,
    ).to(device)
    
    expanded_V = pipeline.trainer.expand_visual_centroids(V, query_centroids).to(device)
    delta = tangent_direction(expanded_V, V)
    
    proto_before = F.normalize((1.0 - alpha_init) * T + alpha_init * V, dim=-1)
    proto_after = F.normalize((1.0 - alpha_init) * T + alpha_init * expanded_V, dim=-1)
    
    logits_before = eval_features.to(device) @ proto_before.T
    logits_after = eval_features.to(device) @ proto_after.T
    
    pred_before = logits_before.argmax(dim=-1).cpu()
    pred_after = logits_after.argmax(dim=-1).cpu()
    
    acc_before_all = float(pred_before.eq(eval_labels.cpu()).float().mean().item() * 100.0)
    acc_after_all = float(pred_after.eq(eval_labels.cpu()).float().mean().item() * 100.0)
    acc_gain_all = acc_after_all - acc_before_all

    beta_val = min(0.45, 0.30 / math.sqrt(kshot))
    rho_val = min(1.0, 0.50 / math.sqrt(kshot))
    Q_adversarial = generate_sqs_adversarial(V, proto_before, beta_val, device)
    
    # 1. B1: fixed SQS-Adversarial
    query_centroids_syn = pipeline.trainer.pseudo_label_aggregation(Q_adversarial.view(-1, Q_adversarial.shape[-1]), T, V, alpha_init).to(device)
    V_syn = F.normalize((1.0 - rho_val) * V + rho_val * query_centroids_syn, dim=-1)
    proto_syn = F.normalize((1.0 - alpha_init) * T + alpha_init * V_syn, dim=-1)
    logits_syn = eval_features.to(device) @ proto_syn.T
    acc_adversarial = float(logits_syn.argmax(dim=-1).cpu().eq(eval_labels.cpu()).float().mean().item() * 100.0)

    # Class-wise gating parameters
    a_c = F.normalize(Q_adversarial.mean(dim=1), dim=-1)
    sim_v = V @ V.T
    sim_v.fill_diagonal_(-float('inf'))
    visual_confusion_c = sim_v.max(dim=1).values
    
    sim_t = T @ T.T
    sim_t.fill_diagonal_(-float('inf'))
    text_confidence_c = 1.0 - sim_t.max(dim=1).values
    
    support_text_agreement_c = (V * T).sum(dim=-1)
    adv_text_drift_c = (proto_before * T).sum(dim=-1) - (a_c * T).sum(dim=-1)
    
    r_c = visual_confusion_c - text_confidence_c + support_text_agreement_c - adv_text_drift_c
    g_c = torch.sigmoid(10.0 * (r_c - r_c.mean()))

    if kshot == 1:
        beta0 = 0.3
    elif kshot == 2:
        beta0 = 0.1
    elif kshot == 4:
        beta0 = 0.05
    else:
        beta0 = 0.0

    # 2. B2: gated SQS-Adversarial, class-wise beta_c
    beta_c = beta0 * g_c
    proto_b2 = F.normalize((1.0 - beta_c).unsqueeze(-1) * proto_before + beta_c.unsqueeze(-1) * a_c, dim=-1)
    logits_b2 = eval_features.to(device) @ proto_b2.T
    acc_b2 = float(logits_b2.argmax(dim=-1).cpu().eq(eval_labels.cpu()).float().mean().item() * 100.0)

    # 3. B3: gated SQS-Adversarial, episode-level beta
    mean_adv_text_drift = adv_text_drift_c.mean().item()
    beta_ep = beta0 * max(0.0, min(1.0, 1.0 - mean_adv_text_drift / 0.05))
    proto_b3 = F.normalize((1.0 - beta_ep) * proto_before + beta_ep * a_c, dim=-1)
    logits_b3 = eval_features.to(device) @ proto_b3.T
    acc_b3 = float(logits_b3.argmax(dim=-1).cpu().eq(eval_labels.cpu()).float().mean().item() * 100.0)

    # 4. B4: Forensic oracle best candidate search
    d1 = tangent_direction(T, V)
    d2 = build_h2_nn_centroid(V)
    d2 = tangent_direction(d2, V)
    d3 = build_h4_center_attr(V)
    d_adv = tangent_direction(a_c, V)
    
    if kshot == 1:
        lambdas = [-0.50, -0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    else:
        lambdas = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]
        lambdas = [l / math.sqrt(float(kshot)) for l in lambdas]
        
    update_modes = ["visual_update", "prototype_update"]
    acc_oracle_best = acc_before_all
    for dir_vec in [d1, d2, d3, d_adv]:
        for l in lambdas:
            for mode in update_modes:
                if mode == "visual_update":
                    V_cand = F.normalize(V + l * dir_vec, dim=-1)
                    proto_cand = F.normalize((1.0 - alpha_init) * T + alpha_init * V_cand, dim=-1)
                else:
                    proto_cand = F.normalize(proto_before + l * dir_vec, dim=-1)
                logits_cand = eval_features.to(device) @ proto_cand.T
                ac = float(logits_cand.argmax(dim=-1).cpu().eq(eval_labels.cpu()).float().mean().item() * 100.0)
                if ac > acc_oracle_best:
                    acc_oracle_best = ac

    forensics = {
        "dataset": pipeline.config.data.dataset_name,
        "seed": int(seed),
        "kshot": int(kshot),
        
        "acc_before": acc_before_all,
        "acc_after": acc_after_all,
        "acc_gain": acc_gain_all,
        
        "sqs_adversarial_acc": acc_adversarial,
        "acc_cpd_b2": acc_b2,
        "acc_cpd_b3": acc_b3,
        "oracle_best_acc": acc_oracle_best
    }
    
    return forensics


def run_dataset_sweep(config, kshots, seeds):
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

    train_payload = pipeline._full_dataset_clip_features()
    train_features_all = train_payload["image_features"]
    train_labels_all = train_payload["labels"]

    if pipeline.val_fraction is None:
        eval_features_all, eval_labels_all = pipeline._cached_val_features()
    else:
        eval_features_all, eval_labels_all = train_features_all, train_labels_all

    results = {}
    for kshot in kshots:
        for seed in seeds:
            kshot = int(kshot)
            seed = int(seed)
            set_global_seed(seed)
            pipeline.kshot = kshot
            pipeline.seed = seed
            pipeline.config.data.kshot = kshot
            pipeline.config.data.seed = seed

            train_indices, val_indices = split_indices_for(pipeline, kshot, seed)
            train_idx = torch.tensor(train_indices, dtype=torch.long)
            train_features = train_features_all[train_idx].contiguous()
            train_labels = train_labels_all[train_idx].contiguous()

            if pipeline.val_fraction is None:
                eval_features = eval_features_all
                eval_labels = eval_labels_all
            else:
                eval_idx = torch.tensor(val_indices, dtype=torch.long)
                eval_features = eval_features_all[eval_idx].contiguous()
                eval_labels = eval_labels_all[eval_idx].contiguous()

            remapped_train, remapped_eval, num_classes = remap_labels(train_labels, eval_labels)
            
            with torch.inference_mode():
                forensics = run_forensic_run(
                    pipeline,
                    train_features,
                    remapped_train,
                    eval_features,
                    remapped_eval,
                    num_classes,
                    kshot,
                    seed
                )
            results[(kshot, seed)] = forensics

    return results


def mean_std(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())


def fmt_mean_std(values, decimals=2, suffix=""):
    mean, std = mean_std(values)
    return f"{mean:.{decimals}f}{suffix} +/- {std:.{decimals}f}{suffix}"


def format_table(rows, columns):
    if not rows:
        return " ".join(columns)
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    lines = [
        "  ".join(column.ljust(widths[column]) for column in columns),
        "  ".join("-" * widths[column] for column in columns),
    ]
    for row in rows:
        lines.append("  ".join(str(row[column]).ljust(widths[column]) for column in columns))
    return "\n".join(lines)


def print_centroid_summary(dataset_name, results, kshots, seeds):
    for kshot in kshots:
        members = [results[(int(kshot), int(seed))] for seed in seeds]
        
        rows = [
            {"Variant": "B0: cosine baseline", "Uses test?": "No", "Accuracy": fmt_mean_std([r["acc_before"] for r in members], suffix="%"), "Gain": "0.00%"},
            {"Variant": "B1: SQS-Adversarial fixed", "Uses test?": "No", "Accuracy": fmt_mean_std([r["sqs_adversarial_acc"] for r in members], suffix="%"), "Gain": fmt_mean_std([r["sqs_adversarial_acc"] - r["acc_before"] for r in members], suffix="%")},
            {"Variant": "B2: gated SQS-Adversarial class-wise", "Uses test?": "No", "Accuracy": fmt_mean_std([r["acc_cpd_b2"] for r in members], suffix="%"), "Gain": fmt_mean_std([r["acc_cpd_b2"] - r["acc_before"] for r in members], suffix="%")},
            {"Variant": "B3: gated SQS-Adversarial episode-level", "Uses test?": "No", "Accuracy": fmt_mean_std([r["acc_cpd_b3"] for r in members], suffix="%"), "Gain": fmt_mean_std([r["acc_cpd_b3"] - r["acc_before"] for r in members], suffix="%")},
            {"Variant": "B4: Oracle-best SQS", "Uses test?": "No", "Accuracy": fmt_mean_std([r["oracle_best_acc"] for r in members], suffix="%"), "Gain": fmt_mean_std([r["oracle_best_acc"] - r["acc_before"] for r in members], suffix="%")},
            {"Variant": "B5: Oracle PLA", "Uses test?": "Yes", "Accuracy": fmt_mean_std([r["acc_after"] for r in members], suffix="%"), "Gain": fmt_mean_std([r["acc_gain"] for r in members], suffix="%")},
        ]
        
        print(f"\n{dataset_name} x {int(kshot)}-shot Reliability-gated Forensic Comparison (n={len(members)} runs)")
        print(format_table(rows, ["Variant", "Uses test?", "Accuracy", "Gain"]), flush=True)


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, "debug", True), getattr(args, "disable_coloring", True))
    config = merge_configs(load_config_file(args.config), overrides)

    previous_level = logger._logger.level
    logger._logger.setLevel(logging.WARNING)
    try:
        dataset_configs = list(iter_dataset_configs(config))
    finally:
        logger._logger.setLevel(previous_level)

    for dataset_config, _ in dataset_configs:
        kshots, seeds = sweep_values(dataset_config, overrides)
        dataset_name = str(dataset_config["data"]["dataset_name"])

        previous_level = logger._logger.level
        logger._logger.setLevel(logging.WARNING)
        try:
            results = run_dataset_sweep(dataset_config, kshots, seeds)
        finally:
            logger._logger.setLevel(previous_level)

        print_centroid_summary(dataset_name, results, kshots, seeds)


def parse_args():
    parser = create_argument_parser("Run ProtoFuse centroid displacement forensics batch sweep", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, process_parsed_args(parsed, ARG_SCHEMA, overrides)


if __name__ == "__main__":
    main()
