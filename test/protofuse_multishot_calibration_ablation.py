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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protofuse_ablation_tables import (
    batched_variant_accuracies,
    build_selector,
    build_table,
    fused_prototypes,
    latex_rows,
    summarize_table,
    table_dataset_configs,
)
from protofuse_candidates import (
    DEFAULT_CONFIG,
    build_text_features,
    extract_image_features,
    load_datasets,
    load_model,
    parse_int_list,
    resolve_path,
    split_positions_by_class,
    support_positions,
)
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
    "Text-only",
    "Fixed fusion, α=0.25",
    "Fixed fusion, α=0.50",
    "Fixed fusion, α=0.75",
    "Support accuracy without LOO",
    "LOO support-only",
    "LOO + standard PLA, no recalibration",
    "Full ProtoFuse",
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


def support_accuracy_curve(alphas, text, visual, support_features, support_labels, alpha_batch_size=16):
    counts = []
    alpha_batch_size = max(1, int(alpha_batch_size))
    for start in range(0, len(alphas), alpha_batch_size):
        alpha_batch = alphas[start:start + alpha_batch_size]
        prototypes = F.normalize(
            (1.0 - alpha_batch).view(-1, 1, 1) * text
            + alpha_batch.view(-1, 1, 1) * visual,
            dim=-1,
        )
        predictions = torch.einsum(
            "nd,acd->anc", support_features, prototypes
        ).argmax(dim=-1)
        counts.append(
            predictions.eq(support_labels.view(1, -1)).sum(dim=-1).float()
        )
    return torch.cat(counts, dim=0)


def loo_accuracy_curve(
    alphas,
    text,
    class_features,
    rho,
    query_centroids=None,
    holdout_batch_size=2,
):
    num_classes, kshot, _ = class_features.shape
    targets = torch.arange(num_classes, device=class_features.device)
    counts = torch.zeros(len(alphas), device=class_features.device)
    class_sums = class_features.sum(dim=1)
    holdout_batch_size = max(1, int(holdout_batch_size))

    for start in range(0, kshot, holdout_batch_size):
        held = F.normalize(
            class_features[:, start:start + holdout_batch_size, :].permute(1, 0, 2),
            dim=-1,
        )
        visual_minus = F.normalize(
            (
                class_sums.unsqueeze(0)
                - class_features[:, start:start + holdout_batch_size, :].permute(1, 0, 2)
            )
            / float(kshot - 1),
            dim=-1,
        )
        if query_centroids is not None:
            visual_minus = F.normalize(
                (1.0 - rho) * visual_minus + rho * query_centroids.unsqueeze(0),
                dim=-1,
            )
        prototypes = F.normalize(
            (1.0 - alphas).view(1, -1, 1, 1) * text.view(1, 1, num_classes, -1)
            + alphas.view(1, -1, 1, 1) * visual_minus.unsqueeze(1),
            dim=-1,
        )
        predictions = torch.einsum("hcd,hakd->hack", held, prototypes).argmax(dim=-1)
        counts += predictions.eq(targets.view(1, 1, -1)).sum(dim=(0, 2)).float()
    return counts


def select_from_curve(alphas, curve):
    return float(alphas[int(curve.argmax().item())].item())


def evaluate_run(
    text_features,
    support_features,
    support_labels,
    eval_features,
    eval_labels,
    device,
    alpha_steps,
    beta_values,
    rho,
    kshot,
    eval_batch_size,
):
    selector, text, visual, support_features, support_labels = build_selector(
        text_features,
        support_features,
        support_labels,
        device,
        alpha_steps,
        beta_values,
        rho,
    )
    class_features = class_feature_tensor(
        support_features,
        support_labels,
        text.shape[0],
        kshot,
    )
    support_curve = support_accuracy_curve(
        selector.alphas,
        text,
        visual,
        support_features,
        support_labels,
    )
    loo_curve = loo_accuracy_curve(
        selector.alphas,
        text,
        class_features,
        rho,
    )
    alpha_support = select_from_curve(selector.alphas, support_curve)
    alpha_init = select_from_curve(selector.alphas, loo_curve)

    query_centroids = selector.pseudo_label_aggregation(
        eval_features,
        text,
        visual,
        alpha_init,
        batch_size=eval_batch_size,
    )
    expanded_visual = selector.expand_visual_centroids(visual, query_centroids)

    recalibrated_curve = loo_accuracy_curve(
        selector.alphas,
        text,
        class_features,
        rho,
        query_centroids=query_centroids,
    )
    alpha_final = select_from_curve(selector.alphas, recalibrated_curve)

    variants = [
        (VARIANTS[0], fused_prototypes(text, visual, 0.0)),
        (VARIANTS[1], fused_prototypes(text, visual, 0.25)),
        (VARIANTS[2], fused_prototypes(text, visual, 0.50)),
        (VARIANTS[3], fused_prototypes(text, visual, 0.75)),
        (VARIANTS[4], fused_prototypes(text, visual, alpha_support)),
        (VARIANTS[5], fused_prototypes(text, visual, alpha_init)),
        (VARIANTS[6], fused_prototypes(text, expanded_visual, alpha_init)),
        (VARIANTS[7], fused_prototypes(text, expanded_visual, alpha_final)),
    ]
    accuracies = batched_variant_accuracies(
        variants,
        eval_features,
        eval_labels,
        device,
        eval_batch_size,
    )
    alphas = {
        VARIANTS[0]: 0.0,
        VARIANTS[1]: 0.25,
        VARIANTS[2]: 0.50,
        VARIANTS[3]: 0.75,
        VARIANTS[4]: alpha_support,
        VARIANTS[5]: alpha_init,
        VARIANTS[6]: alpha_init,
        VARIANTS[7]: alpha_final,
    }
    return [
        {
            "variant": variant,
            "accuracy": float(accuracies[variant]),
            "alpha": float(alphas[variant]),
            "selected_candidate": "qx_recal_alpha" if variant == VARIANTS[7] else None,
        }
        for variant in VARIANTS
    ]


def shotwise_delta_diagnostics(raw, kshots, datasets):
    diagnostics = []
    for current_idx in range(1, len(VARIANTS)):
        previous = VARIANTS[current_idx - 1]
        current = VARIANTS[current_idx]
        for dataset in datasets:
            deltas = {}
            for kshot in kshots:
                previous_values = [
                    row["accuracy"]
                    for row in raw
                    if row["variant"] == previous
                    and row["dataset"] == dataset
                    and row["kshot"] == kshot
                ]
                current_values = [
                    row["accuracy"]
                    for row in raw
                    if row["variant"] == current
                    and row["dataset"] == dataset
                    and row["kshot"] == kshot
                ]
                if previous_values and current_values:
                    deltas[str(kshot)] = float(
                        np.mean(current_values) - np.mean(previous_values)
                    )
            signs = {int(np.sign(value)) for value in deltas.values() if abs(value) > 1e-9}
            diagnostics.append(
                {
                    "from": previous,
                    "to": current,
                    "dataset": dataset,
                    "delta_by_kshot": deltas,
                    "direction_consistent": len(signs) <= 1,
                }
            )
    return diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description="Table 6: ProtoFuse multi-shot component ablation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--kshots", default=",".join(str(value) for value in DEFAULT_KSHOTS))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--disable-coloring", action="store_true")
    parsed, unknown = parser.parse_known_args()
    return parsed, parse_override_arguments(unknown)


def run_dataset(args, table_name, config):
    kshots = parse_int_list(args.kshots)
    seeds = parse_int_list(args.seeds)
    alpha_steps = int(get_config_value(config, "model.alpha_steps", 101))
    beta_values = get_config_value(config, "model.centroid_mix.beta_values", None)
    rho = float(get_config_value(config, "model.rho", 0.5))
    device_name = str(get_config_value(config, "training.device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    batch_size = int(get_config_value(config, "training.batch_size", 128))
    eval_batch_size = args.eval_batch_size or max(batch_size, 1024)
    num_workers = int(get_config_value(config, "data.num_workers", 4))
    dataset_name = str(get_config_value(config, "data.dataset_name", table_name))

    set_global_seed(1)
    train_dataset, eval_dataset, val_fraction, dataset_root = load_datasets(config)
    classnames = list(train_dataset.classes)
    print(
        f"{table_name}: root={dataset_root}, classes={len(classnames)}, "
        f"kshots={kshots}, seeds={seeds}, device={device}",
        flush=True,
    )

    model = load_model(config, device)
    train_features_all, train_labels_all = extract_image_features(
        model, train_dataset, device, batch_size, num_workers
    )
    if eval_dataset is None:
        eval_features_all, eval_labels_all = train_features_all, train_labels_all
    else:
        eval_features_all, eval_labels_all = extract_image_features(
            model, eval_dataset, device, batch_size, num_workers
        )
    text_features = build_text_features(model, classnames, dataset_name, device)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw = []
    for kshot in kshots:
        for seed in seeds:
            set_global_seed(seed)
            if val_fraction is None:
                train_idx = support_positions(train_labels_all, kshot, seed)
                eval_features = eval_features_all
                eval_labels = eval_labels_all
            else:
                train_idx, val_idx = split_positions_by_class(
                    train_labels_all, kshot, seed, val_fraction
                )
                eval_features = train_features_all[val_idx].contiguous()
                eval_labels = train_labels_all[val_idx].contiguous()
            rows = evaluate_run(
                text_features,
                train_features_all[train_idx].contiguous(),
                train_labels_all[train_idx].contiguous(),
                eval_features,
                eval_labels,
                device,
                alpha_steps,
                beta_values,
                rho,
                kshot,
                eval_batch_size,
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
    return raw, str(dataset_root)


def main():
    args, overrides = parse_args()
    config = merge_configs(load_config_file(args.config), overrides)
    kshots = parse_int_list(args.kshots)
    console = Console(no_color=args.disable_coloring)

    raw = []
    dataset_roots = {}
    dataset_configs = table_dataset_configs(config)
    dataset_names = [table_name for table_name, _, _ in dataset_configs]
    for table_name, dataset_config, _ in dataset_configs:
        dataset_raw, dataset_root = run_dataset(args, table_name, dataset_config)
        raw.extend(dataset_raw)
        dataset_roots[table_name] = dataset_root

    summary = summarize_table(raw, VARIANTS, kshots=kshots, datasets=dataset_names)
    console.print()
    console.print(build_table("Table 6: Multi-shot ablation", summary, dataset_names))
    if args.latex:
        console.print("\nLaTeX rows")
        console.print(latex_rows(summary, dataset_names))

    diagnostics = shotwise_delta_diagnostics(raw, kshots, dataset_names)
    inconsistent = [row for row in diagnostics if not row["direction_consistent"]]
    if inconsistent:
        print(
            f"\nShot-wise check: {len(inconsistent)} dataset/component transitions "
            f"change direction across K={{{','.join(str(value) for value in kshots)}}}; "
            "details are stored in JSON.",
            flush=True,
        )
    else:
        print("\nShot-wise check: all component deltas have a consistent direction.", flush=True)

    if args.output_json:
        out_path = resolve_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(
                {
                    "ablation": "table_6_multishot",
                    "datasets": dataset_roots,
                    "kshots": kshots,
                    "seeds": parse_int_list(args.seeds),
                    "raw": raw,
                    "summary": summary,
                    "shotwise_delta_diagnostics": diagnostics,
                },
                handle,
                indent=2,
            )
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
