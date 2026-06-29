import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protofuse_ablation_tables import (
    batched_variant_accuracies,
    build_table,
    fused_prototypes,
    latex_rows,
    summarize_table,
    table_dataset_configs,
)
from src.models.protofuse import ProtoFuse
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
from utils import (
    get_config_value,
    load_config_file,
    merge_configs,
    parse_override_arguments,
    set_global_seed,
)


DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]
DEFAULT_BETAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
VARIANTS = (
    "Text-only, α=0",
    "Fixed fusion, α=0.25",
    "Fixed fusion, α=0.50",
    "Fixed fusion, α=0.75",
    "Centroid-mix calibration, support-only",
    "Full ProtoFuse",
)


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
    eval_batch_size,
):
    selection = ProtoFuse.posthoc_fuse(
        text_features,
        support_features,
        support_labels,
        device,
        alpha_steps=alpha_steps,
        beta_values=beta_values,
        query_features=eval_features,
        rho=rho,
        query_batch_size=eval_batch_size,
    )
    text = selection["text_prototypes"]
    visual = selection["support_visual_centroids"]
    alpha_main = float(selection["alpha"])
    variants = [
        (VARIANTS[0], fused_prototypes(text, visual, 0.0)),
        (VARIANTS[1], fused_prototypes(text, visual, 0.25)),
        (VARIANTS[2], fused_prototypes(text, visual, 0.50)),
        (VARIANTS[3], fused_prototypes(text, visual, 0.75)),
        (VARIANTS[4], fused_prototypes(text, visual, alpha_main)),
        (
            VARIANTS[5],
            selection["raw_fused_prototypes"],
        ),
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
        VARIANTS[4]: alpha_main,
        VARIANTS[5]: alpha_main,
    }
    return [
        {
            "variant": variant,
            "accuracy": float(accuracies[variant]),
            "alpha": float(alphas[variant]),
            "selected_candidate": (
                selection["selected_candidate"] if variant == VARIANTS[5] else None
            ),
        }
        for variant in VARIANTS
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Table 5: ProtoFuse one-shot component ablation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--betas", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--disable-coloring", action="store_true")
    parsed, unknown = parser.parse_known_args()
    return parsed, parse_override_arguments(unknown)


def run_dataset(args, table_name, config, console):
    seeds = parse_int_list(args.seeds)
    alpha_steps = int(get_config_value(config, "model.alpha_steps", 101))
    beta_values = parse_float_list(
        args.betas,
        get_config_value(config, "model.centroid_mix.beta_values", DEFAULT_BETAS),
    )
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
    console.print(
        f"{table_name}: root={dataset_root}, classes={len(classnames)}, "
        f"seeds={seeds}, device={device}"
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
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ]
    with Progress(*progress_columns, console=console) as progress:
        task = progress.add_task(f"{table_name}: one-shot", total=len(seeds))
        for seed in seeds:
            progress.update(task, description=f"{table_name}: 1-shot, seed {seed}")
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
                eval_batch_size,
            )
            raw.extend(
                {
                    "dataset": table_name,
                    "kshot": 1,
                    "seed": int(seed),
                    **row,
                }
                for row in rows
            )
            progress.advance(task)
    return raw, str(dataset_root)


def main():
    args, overrides = parse_args()
    config = merge_configs(load_config_file(args.config), overrides)
    console = Console(no_color=args.disable_coloring)

    raw = []
    dataset_roots = {}
    dataset_configs = table_dataset_configs(config)
    dataset_names = [table_name for table_name, _, _ in dataset_configs]
    for table_name, dataset_config, _ in dataset_configs:
        dataset_raw, dataset_root = run_dataset(args, table_name, dataset_config, console)
        raw.extend(dataset_raw)
        dataset_roots[table_name] = dataset_root

    summary = summarize_table(raw, VARIANTS, kshots=(1,), datasets=dataset_names)
    console.print()
    console.print(build_table("Table 5: One-shot ablation", summary, dataset_names))
    if args.latex:
        console.print("\nLaTeX rows")
        console.print(latex_rows(summary, dataset_names))

    if args.output_json:
        out_path = resolve_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(
                {
                    "ablation": "table_5_oneshot",
                    "datasets": dataset_roots,
                    "seeds": parse_int_list(args.seeds),
                    "raw": raw,
                    "summary": summary,
                },
                handle,
                indent=2,
            )
        console.print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
