import argparse
import gc
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clip import clip
from src.models.apt import CUSTOM_TEMPLATES
from src.models.protofuse import ProtoFuse
from utils import get_config_value, load_clip_to_cpu, load_config_file, merge_configs, parse_override_arguments, set_global_seed


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "protofuse.yaml"
DEFAULT_ALPHA_STEPS = [11, 51, 101, 151, 201]
DEFAULT_KSHOTS = [1, 2, 4, 8, 16]
DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]
DEFAULT_INFER_FEATURES = 10_000
DEFAULT_INFER_WARMUP = 5
DEFAULT_INFER_ITERS = 30
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def parse_int_list(raw):
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def parse_float_list(raw, default):
    if raw is None:
        return list(default)
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    return [float(part.strip()) for part in str(raw).split(",") if part.strip()]


def resolve_path(raw):
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def get_transform(config):
    mean = get_config_value(config, "data.clip_mean", CLIP_MEAN)
    std = get_config_value(config, "data.clip_std", CLIP_STD)
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_datasets(config):
    root = resolve_path(get_config_value(config, "data.root", REPO_ROOT / "datasets" / "DTD"))
    transform = get_transform(config)
    val_size = get_config_value(config, "data.val_size", None)

    train_root = root / "train"
    test_root = root / "test"
    if val_size is None and train_root.exists() and test_root.exists():
        train_dataset = ImageFolder(str(train_root), transform=transform)
        eval_dataset = ImageFolder(str(test_root), transform=transform)
        if train_dataset.classes != eval_dataset.classes:
            raise ValueError("Train/test class folders do not match.")
        return train_dataset, eval_dataset, None, root

    if val_size is None:
        raise ValueError(
            f"{root} does not contain train/test folders. Set data.val_size or pass --data.val_size for a split."
        )

    val_fraction = float(val_size)
    if val_fraction > 1.0:
        val_fraction = val_fraction / 100.0
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("data.val_size must be in (0, 1) or 0-100 range when expressed as a percentage.")

    dataset = ImageFolder(str(root), transform=transform)
    return dataset, None, val_fraction, root


def load_model(config, device):
    backbone = str(get_config_value(config, "model.backbone", "ViT-B/16"))
    precision = str(get_config_value(config, "training.precision", "fp32"))
    model = load_clip_to_cpu(backbone)
    if precision in {"fp32", "amp"}:
        model.float()
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def extract_image_features(model, dataset, device, batch_size, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    features = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            batch_features = model.encode_image(images).float()
            batch_features = F.normalize(batch_features, dim=-1)
            features.append(batch_features.cpu())
            labels.append(batch_labels.cpu().long())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def build_text_features(model, classnames, dataset_name, device):
    template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")
    prompts = [template.format(name.replace("_", " ")) for name in classnames]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens).float()
    return F.normalize(text_features, dim=-1).detach().cpu()


def support_positions(labels, kshot, seed):
    by_class = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        by_class[int(label)].append(idx)

    rng = random.Random(seed)
    selected = []
    for class_idx in sorted(by_class):
        positions = list(by_class[class_idx])
        positions.sort()
        rng.shuffle(positions)
        if len(positions) < kshot:
            raise RuntimeError(f"Class {class_idx} has {len(positions)} samples, cannot sample {kshot}-shot.")
        selected.extend(positions[:kshot])
    return torch.tensor(selected, dtype=torch.long)


def split_positions_by_class(labels, kshot, seed, val_fraction):
    by_class = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        by_class[int(label)].append(idx)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    for class_idx in sorted(by_class):
        positions = list(by_class[class_idx])
        positions.sort()
        rng.shuffle(positions)

        val_count = int(math.floor(len(positions) * val_fraction))
        if val_count == 0 and positions:
            val_count = 1
        val_part = positions[:val_count]
        train_candidates = positions[val_count:]
        if len(train_candidates) < kshot:
            raise RuntimeError(
                f"Class {class_idx} has {len(train_candidates)} train candidates after split, cannot sample {kshot}-shot."
            )
        train_indices.extend(train_candidates[:kshot])
        val_indices.extend(val_part)

    if not val_indices:
        raise RuntimeError("Validation split is empty.")
    return torch.tensor(train_indices, dtype=torch.long), torch.tensor(val_indices, dtype=torch.long)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def prepare_infer_features(features, count, device):
    if features.shape[0] <= 0:
        raise RuntimeError("Cannot benchmark inference with an empty feature tensor.")
    repeats = math.ceil(count / features.shape[0])
    return features.repeat((repeats, 1))[:count].contiguous().to(device)


def benchmark_infer_logits(device, logits_fn, features, warmup, iters):
    sync(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = logits_fn(features)
        sync(device)

        if str(device).startswith("cuda"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _ = logits_fn(features)
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / max(1, iters)

        start_time = time.perf_counter()
        for _ in range(iters):
            _ = logits_fn(features)
        return 1000.0 * (time.perf_counter() - start_time) / max(1, iters)


def evaluate_protofuse(
    text_features,
    train_features,
    train_labels,
    eval_features,
    eval_labels,
    infer_features,
    device,
    alpha_steps,
    beta_values,
    force_loo_accuracy,
    infer_warmup,
    infer_iters,
):
    selection = ProtoFuse.posthoc_fuse(
        text_features,
        train_features,
        train_labels,
        device,
        alpha_steps=alpha_steps,
        beta_values=beta_values,
        force_loo_accuracy=force_loo_accuracy,
    )
    fused = selection["fused_prototypes"]
    logits = eval_features.to(device).float() @ fused.t()
    preds = logits.argmax(dim=-1)
    labels = eval_labels.to(device).long()
    accuracy = preds.eq(labels).float().mean().item() * 100.0

    def logits_fn(features):
        return features.to(device).float() @ fused.t()

    infer_time_us = 1000.0 * benchmark_infer_logits(device, logits_fn, infer_features, infer_warmup, infer_iters)
    return accuracy, float(selection["alpha"]), float(infer_time_us)


def format_mean_std(values, decimals=2):
    arr = np.asarray(values, dtype=np.float64)
    return f"{arr.mean():.{decimals}f} +/- {arr.std():.{decimals}f}"


def build_table(rows, dataset_name):
    table = Table(title=str(dataset_name))
    table.add_column("alpha_steps", justify="right")
    table.add_column("runs", justify="right")
    table.add_column("accuracy (%)", justify="right")
    table.add_column("infer time (us/10k)", justify="right")

    for row in rows:
        table.add_row(
            str(row["alpha_steps"]),
            str(row["runs"]),
            format_mean_std(row["accuracies"], decimals=2),
            format_mean_std(row["infer_times"], decimals=2),
        )
    return table


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep ProtoFuse alpha candidate counts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML configuration file.")
    parser.add_argument("--alpha-steps", default=",".join(str(v) for v in DEFAULT_ALPHA_STEPS))
    parser.add_argument("--kshots", default=",".join(str(v) for v in DEFAULT_KSHOTS))
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--infer-features", type=int, default=DEFAULT_INFER_FEATURES)
    parser.add_argument("--infer-warmup", type=int, default=DEFAULT_INFER_WARMUP)
    parser.add_argument("--infer-iters", type=int, default=DEFAULT_INFER_ITERS)
    parser.add_argument("--output-json", default=None, help="Optional path to write raw and aggregate results.")
    parser.add_argument("--disable-coloring", action="store_true")
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    config = merge_configs(load_config_file(args.config), overrides)
    console = Console(no_color=args.disable_coloring)

    alpha_steps_values = parse_int_list(args.alpha_steps)
    kshots = parse_int_list(args.kshots)
    seeds = parse_int_list(args.seeds)
    beta_values = parse_float_list(
        get_config_value(config, "model.centroid_mix.beta_values", None),
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    )
    force_loo_accuracy = ProtoFuse._coerce_bool(get_config_value(config, "model.force_loo_accuracy", False), False)

    device_name = str(get_config_value(config, "training.device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    batch_size = int(get_config_value(config, "training.batch_size", 128))
    num_workers = int(get_config_value(config, "data.num_workers", 4))
    dataset_name = str(get_config_value(config, "data.dataset_name", "DTD"))

    set_global_seed(1)
    train_dataset, eval_dataset, val_fraction, dataset_root = load_datasets(config)
    classnames = list(train_dataset.classes)

    console.print(
        f"Dataset={dataset_name}, root={dataset_root}, classes={len(classnames)}, "
        f"kshots={kshots}, seeds={seeds}, alpha_steps={alpha_steps_values}, device={device}"
    )
    console.print("Extracting CLIP features once; ProtoFuse candidate grids are swept on cached tensors.")
    console.print(
        "Infer time follows test/computational_cost.py: logits on cached image features only, "
        f"{args.infer_features:,} features, warmup={args.infer_warmup}, iters={args.infer_iters}."
    )

    model = load_model(config, device)
    train_features_all, train_labels_all = extract_image_features(model, train_dataset, device, batch_size, num_workers)
    if eval_dataset is None:
        eval_features_all, eval_labels_all = train_features_all, train_labels_all
    else:
        eval_features_all, eval_labels_all = extract_image_features(model, eval_dataset, device, batch_size, num_workers)
    text_features = build_text_features(model, classnames, dataset_name, device)
    infer_features_all = None
    if val_fraction is None:
        infer_features_all = prepare_infer_features(eval_features_all, args.infer_features, device)
    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    raw_results = []
    total = len(alpha_steps_values) * len(kshots) * len(seeds)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("sweeping", total=total)
        for alpha_steps in alpha_steps_values:
            for kshot in kshots:
                for seed in seeds:
                    progress.update(task, description=f"steps={alpha_steps} | {kshot}-shot | seed {seed}")
                    set_global_seed(seed)
                    if val_fraction is None:
                        train_idx = support_positions(train_labels_all, kshot, seed)
                        support_features = train_features_all[train_idx].contiguous()
                        support_labels = train_labels_all[train_idx].contiguous()
                        eval_features = eval_features_all
                        eval_labels = eval_labels_all
                        infer_features = infer_features_all
                    else:
                        train_idx, val_idx = split_positions_by_class(train_labels_all, kshot, seed, val_fraction)
                        support_features = train_features_all[train_idx].contiguous()
                        support_labels = train_labels_all[train_idx].contiguous()
                        eval_features = train_features_all[val_idx].contiguous()
                        eval_labels = train_labels_all[val_idx].contiguous()
                        infer_features = prepare_infer_features(eval_features, args.infer_features, device)

                    accuracy, alpha, infer_time_us = evaluate_protofuse(
                        text_features,
                        support_features,
                        support_labels,
                        eval_features,
                        eval_labels,
                        infer_features,
                        device,
                        alpha_steps,
                        beta_values,
                        force_loo_accuracy,
                        args.infer_warmup,
                        args.infer_iters,
                    )
                    raw_results.append(
                        {
                            "alpha_steps": int(alpha_steps),
                            "kshot": int(kshot),
                            "seed": int(seed),
                            "accuracy": float(accuracy),
                            "alpha": float(alpha),
                            "infer_time_us_10k": float(infer_time_us),
                        }
                    )
                    progress.advance(task)

    aggregate_rows = []
    for alpha_steps in alpha_steps_values:
        members = [row for row in raw_results if row["alpha_steps"] == alpha_steps]
        aggregate_rows.append(
            {
                "alpha_steps": int(alpha_steps),
                "runs": len(members),
                "accuracies": [row["accuracy"] for row in members],
                "alphas": [row["alpha"] for row in members],
                "infer_times": [row["infer_time_us_10k"] for row in members],
            }
        )

    console.print()
    console.print(build_table(aggregate_rows, dataset_name))

    if args.output_json:
        out_path = resolve_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": dataset_name,
            "dataset_root": str(dataset_root),
            "kshots": kshots,
            "seeds": seeds,
            "alpha_steps": alpha_steps_values,
            "infer_features": args.infer_features,
            "infer_warmup": args.infer_warmup,
            "infer_iters": args.infer_iters,
            "raw": raw_results,
            "aggregate": [
                {
                    "alpha_steps": row["alpha_steps"],
                    "runs": row["runs"],
                    "accuracy_mean": float(np.mean(row["accuracies"])),
                    "accuracy_std": float(np.std(row["accuracies"])),
                    "infer_time_us_10k_mean": float(np.mean(row["infer_times"])),
                    "infer_time_us_10k_std": float(np.std(row["infer_times"])),
                }
                for row in aggregate_rows
            ],
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        console.print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
