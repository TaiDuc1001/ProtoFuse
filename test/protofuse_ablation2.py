import sys
import argparse
import math
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from clip import clip
import compare_methods as cm
from src.models.apt import CUSTOM_TEMPLATES
from utils import compute_metrics


REPO_ROOT = Path(__file__).parent.parent
CLIP_MODEL_PATH = REPO_ROOT / "models" / "ViT-B-16.pt"
DEFAULT_DEVICE = "cuda:0"

DEVICE = DEFAULT_DEVICE
BATCH_SIZE = 128
VAL_SIZE = 0.7
SEEDS = [1, 10, 100]
KSHOTS = [1, 2, 4, 8, 16]
ALPHA_STEPS = 11
GRID_SIZES = [11, 21, 51, 101]
GRID_KSHOT = 4
NUM_WORKERS = 4
COMP_NUM_CLASSES = 200

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CACHE_DIR = REPO_ROOT / "checkpoints" / "protofuse_ablation_cache"
OUTPUT_DIR = REPO_ROOT / "outputs" / "protofuse_ablation2"
PLOT_PALETTE = [
    "#115e59",
    "#1e3a8a",
    "#9a3412",
    "#9f1239",
    "#6d28d9",
]

DATASET_SPECS = [
    {
        "key": "cub",
        "display_name": "CUB-200-2011",
        "template_name": "CUB-200-2011",
        "path": Path("/state/partition1/tri.pm/APT/datasets/cub-200-2011-renamed"),
    },
    {
        "key": "flowers",
        "display_name": "Flowers102",
        "template_name": "Flowers102",
        "path": Path("/state/partition1/tri.pm/APT/datasets/flowers102"),
    },
    {
        "key": "aircraft",
        "display_name": "FGVC-Aircraft",
        "template_name": "FGVCAircraft",
        "path": Path("/state/partition1/tri.pm/APT/datasets/fgvc_aircraft"),
    },
    {
        "key": "cars",
        "display_name": "Stanford Cars",
        "template_name": "StanfordCars",
        "path": Path("datasets/stanford_cars"),
    },
]


def load_clip():
    model = torch.jit.load(str(CLIP_MODEL_PATH), map_location="cpu").eval()
    state_dict = model.state_dict()
    model = clip.build_model(state_dict)
    model = model.to(DEVICE).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def extract_and_cache_features(clip_model, dataset, cache_name):
    cache_path = CACHE_DIR / f"{cache_name}.pt"
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["features"], data["labels"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Extracting features", leave=False):
            images = images.to(DEVICE)
            features = clip_model.encode_image(images).float()
            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({"features": features, "labels": labels}, cache_path)
    return features, labels


def split_by_class(dataset, val_size, kshot, seed):
    samples_by_class = defaultdict(list)
    for idx, (_, class_idx) in enumerate(dataset.samples):
        samples_by_class[class_idx].append(idx)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for class_idx in sorted(samples_by_class.keys()):
        class_samples = list(samples_by_class[class_idx])
        class_samples.sort()
        rng.shuffle(class_samples)

        val_count = int(math.floor(len(class_samples) * val_size))
        if val_size > 0 and val_count == 0 and len(class_samples) > 0:
            val_count = 1

        val_part = class_samples[:val_count]
        train_candidates = class_samples[val_count:]

        val_indices.extend(val_part)
        if kshot > 0:
            train_indices.extend(train_candidates[:kshot])
        else:
            train_indices.extend(train_candidates)

    return train_indices, val_indices


def count_samples_per_class(dataset, indices=None):
    counts_by_class = defaultdict(int)
    if indices is None:
        iterator = dataset.samples
    else:
        iterator = (dataset.samples[idx] for idx in indices)

    for _, class_idx in iterator:
        counts_by_class[class_idx] += 1

    return [counts_by_class[class_idx] for class_idx in sorted(counts_by_class.keys())]


def summarize_eval_split_balance(dataset, val_size, seed=SEEDS[0]):
    _, val_indices = split_by_class(dataset, val_size, kshot=1, seed=seed)
    full_counts = count_samples_per_class(dataset)
    val_counts = count_samples_per_class(dataset, val_indices)

    return {
        "num_classes": len(full_counts),
        "dataset_min": min(full_counts),
        "dataset_max": max(full_counts),
        "val_min": min(val_counts),
        "val_max": max(val_counts),
        "val_is_balanced": len(set(val_counts)) == 1,
    }


def print_eval_split_balance_note(console, dataset_name, balance_summary):
    console.print(
        "[bold blue]Class balance:[/bold blue] "
        f"dataset={balance_summary['dataset_min']}-{balance_summary['dataset_max']} img/class, "
        f"eval={balance_summary['val_min']}-{balance_summary['val_max']} img/class"
    )

    if balance_summary["val_is_balanced"]:
        console.print(
            "[yellow]Metric note:[/yellow] "
            f"{dataset_name} uses a perfectly class-balanced eval split, so top-1 accuracy and MCA "
            "are identical by definition on this split."
        )


def infer_dataset_name(dataset_path):
    path_name = Path(dataset_path).name.lower()
    if "cub" in path_name:
        return "CUB-200-2011"
    if "flower" in path_name:
        return "Flowers102"
    if "aircraft" in path_name:
        return "FGVCAircraft"
    if "car" in path_name:
        return "StanfordCars"
    if "dog" in path_name:
        return "OxfordPets"
    if "food" in path_name:
        return "Food101"
    return Path(dataset_path).name


def get_task_text_features(clip_model, classnames, task_classes, dataset_name):
    sorted_classes = sorted(task_classes)
    template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")
    prompts = [template.format(classnames[c].replace("_", " ")) for c in sorted_classes]

    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(DEVICE)
        text_features = clip_model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)

    class_remap = {c: i for i, c in enumerate(sorted_classes)}
    return text_features, class_remap


def build_visual_centroids(train_features, remapped_train_labels, num_classes, embed_dim):
    device = train_features.device
    dtype = train_features.dtype
    visual_centroids = torch.zeros(num_classes, embed_dim, device=device, dtype=dtype)
    counts = torch.zeros(num_classes, 1, device=device, dtype=dtype)
    ones = torch.ones(remapped_train_labels.shape[0], 1, device=device, dtype=dtype)

    visual_centroids.index_add_(0, remapped_train_labels, train_features)
    counts.index_add_(0, remapped_train_labels, ones)

    valid = counts.squeeze(1) > 0
    visual_centroids[valid] = F.normalize(visual_centroids[valid] / counts[valid], dim=-1)
    return visual_centroids


def remap_labels(labels, class_remap, device=None):
    remapped = torch.tensor([class_remap[label.item()] for label in labels], dtype=torch.long)
    if device is not None:
        remapped = remapped.to(device, non_blocking=True)
    return remapped


def prepare_split_tensors(all_features, all_labels, train_indices, val_indices, class_remap):
    train_features = all_features[train_indices].to(DEVICE, non_blocking=True)
    val_features = all_features[val_indices].to(DEVICE, non_blocking=True)
    train_labels_raw = all_labels[train_indices]
    val_labels_raw = all_labels[val_indices]
    remapped_train = remap_labels(train_labels_raw, class_remap, device=DEVICE)
    remapped_val = remap_labels(val_labels_raw, class_remap, device=DEVICE)
    return train_features, val_features, remapped_train, remapped_val


def evaluate_discriminative_accuracy(T, V_all, alpha_value, eval_features, eval_labels, num_classes, one_shot_mode):
    eval_norm = F.normalize(eval_features, dim=-1)
    labels = eval_labels

    if one_shot_mode:
        tau = 0.05
        logits_text = eval_norm @ T.T
        logits_visual = eval_norm @ V_all.T
        probs_visual = F.softmax(logits_visual / tau, dim=-1)
        entropy = -(probs_visual * torch.log(probs_visual + 1e-8)).sum(dim=-1)
        confidence = 1.0 - torch.clamp(entropy / math.log(num_classes), 0.0, 1.0)
        alpha_x = alpha_value * (0.5 + 0.5 * confidence)
        logits_final = (1 - alpha_x).unsqueeze(-1) * logits_text + alpha_x.unsqueeze(-1) * logits_visual
        preds = logits_final.argmax(dim=-1)
        return float(preds.eq(labels).float().mean().item() * 100.0)

    refined = F.normalize((1 - alpha_value) * T + alpha_value * V_all, dim=-1)
    logits = eval_norm @ refined.T
    preds = logits.argmax(dim=-1)
    return float(preds.eq(labels).float().mean().item() * 100.0)


def opt_base_eval(T, V_all, alpha, val_features, remapped_val_labels):
    val_norm = F.normalize(val_features, dim=-1)
    refined = F.normalize((1 - alpha) * T + alpha * V_all, dim=-1)
    logits = val_norm @ refined.T
    preds = logits.argmax(dim=-1).cpu().numpy()
    remapped_labels = remapped_val_labels.cpu().numpy()
    return compute_metrics(remapped_labels.tolist(), preds.tolist())


def opt_base_loo_cv_alpha(T, V_all, train_features, remapped_train_labels, num_classes, alphas):
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(remapped_train_labels.tolist()):
        class_indices[label].append(idx)

    shots_per_class = min(len(indices) for indices in class_indices)
    if shots_per_class < 2:
        train_norm = F.normalize(train_features, dim=-1)
        refined = F.normalize((1 - alphas).view(-1, 1, 1) * T + alphas.view(-1, 1, 1) * V_all, dim=-1)
        logits = torch.einsum("qd,apd->aqp", train_norm, refined)
        preds = logits.argmax(dim=-1)
        scores = (preds == remapped_train_labels).float().mean(dim=-1)
        return float(alphas[scores.argmax()].item())

    k = shots_per_class
    class_feat = torch.stack([train_features[class_indices[c][:k]] for c in range(num_classes)])
    class_sums = class_feat.sum(dim=1)
    loo_scores = torch.zeros(len(alphas), device=DEVICE)

    for fold in range(k):
        held = F.normalize(class_feat[:, fold, :], dim=-1)
        v_loo = F.normalize((class_sums - class_feat[:, fold, :]) / (k - 1), dim=-1)
        refined = F.normalize((1 - alphas).view(-1, 1, 1) * T + alphas.view(-1, 1, 1) * v_loo, dim=-1)
        logits = torch.einsum("qd,apd->aqp", held, refined)
        preds = logits.argmax(dim=-1)
        loo_scores += (preds == torch.arange(num_classes, device=DEVICE)).float().mean(dim=-1)

    return float(alphas[loo_scores.argmax()].item())


def opt_entropy_eval(T, V_all, alpha_base, val_features, remapped_val_labels, num_classes, tau=0.05):
    val_norm = F.normalize(val_features, dim=-1)
    logits_text = val_norm @ T.T
    logits_visual = val_norm @ V_all.T

    probs_visual = F.softmax(logits_visual / tau, dim=-1)
    entropy = -(probs_visual * torch.log(probs_visual + 1e-8)).sum(dim=-1)
    confidence = 1.0 - torch.clamp(entropy / math.log(num_classes), 0.0, 1.0)
    alpha_x = alpha_base * (0.5 + 0.5 * confidence)

    logits_final = (1 - alpha_x).unsqueeze(-1) * logits_text + alpha_x.unsqueeze(-1) * logits_visual
    preds = logits_final.argmax(dim=-1).cpu().numpy()
    remapped_labels = remapped_val_labels.cpu().numpy()
    return compute_metrics(remapped_labels.tolist(), preds.tolist())


def sweep_alpha_loo_scores(T, V_all, alpha_grid, train_features, train_labels, num_classes):
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(train_labels.tolist()):
        class_indices[label].append(idx)

    shots_per_class = min(len(indices) for indices in class_indices)
    if shots_per_class < 2:
        train_norm = F.normalize(train_features, dim=-1)
        refined = F.normalize((1 - alpha_grid).view(-1, 1, 1) * T + alpha_grid.view(-1, 1, 1) * V_all, dim=-1)
        logits = torch.einsum("qd,apd->aqp", train_norm, refined)
        preds = logits.argmax(dim=-1)
        scores = (preds == train_labels).float().mean(dim=-1)
        return (scores.cpu().numpy() * 100.0).astype(np.float32)

    k = shots_per_class
    class_feat = torch.stack([train_features[class_indices[c][:k]] for c in range(num_classes)])
    class_sums = class_feat.sum(dim=1)
    loo_scores = torch.zeros(len(alpha_grid), device=DEVICE)

    for fold in range(k):
        held = F.normalize(class_feat[:, fold, :], dim=-1)
        v_loo = F.normalize((class_sums - class_feat[:, fold, :]) / (k - 1), dim=-1)
        refined = F.normalize((1 - alpha_grid).view(-1, 1, 1) * T + alpha_grid.view(-1, 1, 1) * v_loo, dim=-1)
        logits = torch.einsum("qd,apd->aqp", held, refined)
        preds = logits.argmax(dim=-1)
        loo_scores += (preds == torch.arange(num_classes, device=DEVICE)).float().mean(dim=-1)

    loo_scores = loo_scores / k
    return (loo_scores.cpu().numpy() * 100.0).astype(np.float32)


def sweep_alpha_resub_accuracy(T, V_all, alpha_grid, support_features, support_labels):
    support_norm = F.normalize(support_features, dim=-1)
    refined = F.normalize((1 - alpha_grid).view(-1, 1, 1) * T + alpha_grid.view(-1, 1, 1) * V_all, dim=-1)
    logits = torch.einsum("qd,apd->aqp", support_norm, refined)
    preds = logits.argmax(dim=-1)
    labels = support_labels.view(1, -1)
    accuracies = preds.eq(labels).float().mean(dim=-1).cpu().numpy() * 100.0
    return accuracies.astype(np.float32)


def select_alpha_loo(T, V_all, alpha_grid, train_features, train_labels, num_classes):
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(train_labels.tolist()):
        class_indices[label].append(idx)

    shots_per_class = min(len(indices) for indices in class_indices)
    if shots_per_class < 2:
        train_norm = F.normalize(train_features, dim=-1)
        refined = F.normalize((1 - alpha_grid).view(-1, 1, 1) * T + alpha_grid.view(-1, 1, 1) * V_all, dim=-1)
        logits = torch.einsum("qd,apd->aqp", train_norm, refined)
        preds = logits.argmax(dim=-1)
        scores = (preds == train_labels).float().mean(dim=-1)
        return float(alpha_grid[scores.argmax()].item())

    k = shots_per_class
    class_feat = torch.stack([train_features[class_indices[c][:k]] for c in range(num_classes)])
    class_sums = class_feat.sum(dim=1)
    loo_scores = torch.zeros(len(alpha_grid), device=DEVICE)

    for fold in range(k):
        held = F.normalize(class_feat[:, fold, :], dim=-1)
        v_loo = F.normalize((class_sums - class_feat[:, fold, :]) / (k - 1), dim=-1)
        refined = F.normalize((1 - alpha_grid).view(-1, 1, 1) * T + alpha_grid.view(-1, 1, 1) * v_loo, dim=-1)
        logits = torch.einsum("qd,apd->aqp", held, refined)
        preds = logits.argmax(dim=-1)
        loo_scores += (preds == torch.arange(num_classes, device=DEVICE)).float().mean(dim=-1)

    return float(alpha_grid[loo_scores.argmax()].item())


def evaluate_fixed_alpha_accuracy(T, V_all, alpha_value, eval_features, eval_labels):
    return evaluate_discriminative_accuracy(
        T,
        V_all,
        alpha_value,
        eval_features,
        eval_labels,
        num_classes=0,
        one_shot_mode=False,
    )


def compute_one_shot_scores(T, V_all, eval_features, num_classes, alpha_base=1.0, tau=1.0):
    eval_norm = F.normalize(eval_features, dim=-1)
    logits_text = eval_norm @ T.T
    logits_visual = eval_norm @ V_all.T
    probs_visual = F.softmax(logits_visual / tau, dim=-1)
    entropy = -(probs_visual * torch.log(probs_visual + 1e-8)).sum(dim=-1)
    confidence = 1.0 - torch.clamp(entropy / math.log(num_classes), 0.0, 1.0)
    alpha_q = alpha_base * (0.5 + 0.5 * confidence)
    logits_full = (1 - alpha_q).unsqueeze(-1) * logits_text + alpha_q.unsqueeze(-1) * logits_visual
    return logits_text, logits_visual, logits_full


def select_alpha_resub(T, V_all, alpha_grid, support_features, support_labels):
    resub_accs = sweep_alpha_resub_accuracy(T, V_all, alpha_grid, support_features, support_labels)
    best_idx = int(np.argmax(resub_accs))
    return float(alpha_grid[best_idx].item())


def evaluate_one_shot_metrics(T, V_all, alpha_grid, support_features, support_labels, eval_features, eval_labels, num_classes):
    alpha_resub = select_alpha_resub(T, V_all, alpha_grid, support_features, support_labels)
    _, _, logits_full = compute_one_shot_scores(T, V_all, eval_features, num_classes, alpha_base=alpha_resub)
    preds = logits_full.argmax(dim=-1).cpu().numpy()
    remapped_labels = eval_labels.cpu().numpy()
    return compute_metrics(remapped_labels.tolist(), preds.tolist())


def evaluate_one_shot_variants(T, V_all, alpha_grid, support_features, support_labels, eval_features, eval_labels, num_classes):
    labels = eval_labels
    logits_text, logits_visual, _ = compute_one_shot_scores(T, V_all, eval_features, num_classes, alpha_base=1.0)
    preds_text = logits_text.argmax(dim=-1)
    preds_visual = logits_visual.argmax(dim=-1)
    acc_text = preds_text.eq(labels).float().mean().item() * 100.0
    acc_visual = preds_visual.eq(labels).float().mean().item() * 100.0

    alpha_resub = select_alpha_resub(T, V_all, alpha_grid, support_features, support_labels)
    _, _, logits_full = compute_one_shot_scores(T, V_all, eval_features, num_classes, alpha_base=alpha_resub)
    preds_full = logits_full.argmax(dim=-1)
    acc_full = preds_full.eq(labels).float().mean().item() * 100.0

    return {
        "text_only": acc_text,
        "visual_only": acc_visual,
        "protofuse_full": acc_full,
    }


def run_ablation_1_alpha(all_features, all_labels, dataset, class_remap, text_features, alpha_grid, alpha_values, num_classes, embed_dim):
    curves_by_kshot = {}

    with torch.inference_mode():
        for kshot in (k for k in KSHOTS if k != 1):
            seed_curves = []

            for seed in SEEDS:
                train_indices, val_indices = split_by_class(dataset, VAL_SIZE, kshot, seed)
                train_features, _, remapped_train, _ = prepare_split_tensors(
                    all_features,
                    all_labels,
                    train_indices,
                    val_indices,
                    class_remap,
                )

                visual_centroids = build_visual_centroids(train_features, remapped_train, num_classes, embed_dim)
                mean_accuracy_curve = sweep_alpha_loo_scores(
                    text_features,
                    visual_centroids,
                    alpha_grid,
                    train_features,
                    remapped_train,
                    num_classes,
                )
                seed_curves.append(mean_accuracy_curve)

            curves_by_kshot[kshot] = aggregate_curve_stats(seed_curves, alpha_values)

    return curves_by_kshot


def run_ablation_2_grid_size(all_features, all_labels, dataset, class_remap, text_features, num_classes, embed_dim):
    grid_metrics = {grid_size: {"accuracy": [], "selection_time_ms": []} for grid_size in GRID_SIZES}

    with torch.inference_mode():
        for grid_size in GRID_SIZES:
            alpha_grid = torch.linspace(0, 1, grid_size, device=DEVICE)

            for seed in SEEDS:
                train_indices, val_indices = split_by_class(dataset, VAL_SIZE, GRID_KSHOT, seed)
                train_features, val_features, remapped_train, remapped_val = prepare_split_tensors(
                    all_features,
                    all_labels,
                    train_indices,
                    val_indices,
                    class_remap,
                )

                visual_centroids = build_visual_centroids(train_features, remapped_train, num_classes, embed_dim)

                start_time = time.perf_counter()
                alpha_loo = select_alpha_loo(
                    text_features,
                    visual_centroids,
                    alpha_grid,
                    train_features,
                    remapped_train,
                    num_classes,
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                grid_metrics[grid_size]["accuracy"].append(
                    evaluate_fixed_alpha_accuracy(
                        text_features,
                        visual_centroids,
                        alpha_loo,
                        val_features,
                        remapped_val,
                    )
                )
                grid_metrics[grid_size]["selection_time_ms"].append(elapsed_ms)

    return grid_metrics


def run_ablation_3_one_shot(all_features, all_labels, dataset, class_remap, text_features, num_classes, embed_dim):
    one_shot_variant_metrics = {
        "text_only": [],
        "visual_only": [],
        "protofuse_full": [],
    }

    with torch.inference_mode():
        for seed in SEEDS:
            train_indices, val_indices = split_by_class(dataset, VAL_SIZE, 1, seed)
            train_features, val_features, remapped_train, remapped_val = prepare_split_tensors(
                all_features,
                all_labels,
                train_indices,
                val_indices,
                class_remap,
            )

            visual_centroids = build_visual_centroids(train_features, remapped_train, num_classes, embed_dim)
            variant_results = evaluate_one_shot_variants(
                text_features,
                visual_centroids,
                torch.linspace(0, 1, ALPHA_STEPS, device=DEVICE),
                train_features,
                remapped_train,
                val_features,
                remapped_val,
                num_classes,
            )

            for key, value in variant_results.items():
                one_shot_variant_metrics[key].append(value)

    return one_shot_variant_metrics


def run_ablation_4_fixed_alpha_vs_loo(all_features, all_labels, dataset, class_remap, text_features, alpha_grid, num_classes, embed_dim):
    ablation4_kshots = [2, 4, 8, 16]
    metrics_by_kshot = {
        kshot: {"alpha_0": [], "alpha_1": [], "alpha_05": [], "alpha_loo": []}
        for kshot in ablation4_kshots
    }

    with torch.inference_mode():
        for kshot in ablation4_kshots:
            for seed in SEEDS:
                train_indices, val_indices = split_by_class(dataset, VAL_SIZE, kshot, seed)
                train_features, val_features, remapped_train, remapped_val = prepare_split_tensors(
                    all_features,
                    all_labels,
                    train_indices,
                    val_indices,
                    class_remap,
                )

                visual_centroids = build_visual_centroids(train_features, remapped_train, num_classes, embed_dim)

                metrics_by_kshot[kshot]["alpha_0"].append(
                    evaluate_fixed_alpha_accuracy(text_features, visual_centroids, 0.0, val_features, remapped_val)
                )
                metrics_by_kshot[kshot]["alpha_1"].append(
                    evaluate_fixed_alpha_accuracy(text_features, visual_centroids, 1.0, val_features, remapped_val)
                )
                metrics_by_kshot[kshot]["alpha_05"].append(
                    evaluate_fixed_alpha_accuracy(text_features, visual_centroids, 0.5, val_features, remapped_val)
                )

                alpha_loo = select_alpha_loo(
                    text_features,
                    visual_centroids,
                    alpha_grid,
                    train_features,
                    remapped_train,
                    num_classes,
                )
                metrics_by_kshot[kshot]["alpha_loo"].append(
                    evaluate_fixed_alpha_accuracy(text_features, visual_centroids, alpha_loo, val_features, remapped_val)
                )

    return metrics_by_kshot


def evaluate_fewshot_protofuse(all_features, all_labels, dataset, class_remap, text_features, num_classes, embed_dim):
    alpha_grid = torch.linspace(0, 1, ALPHA_STEPS, device=DEVICE)
    fewshot_metrics = {}

    with torch.inference_mode():
        for kshot in KSHOTS:
            values_by_metric = defaultdict(list)

            for seed in SEEDS:
                train_indices, val_indices = split_by_class(dataset, VAL_SIZE, kshot, seed)
                train_features, val_features, remapped_train, remapped_val = prepare_split_tensors(
                    all_features,
                    all_labels,
                    train_indices,
                    val_indices,
                    class_remap,
                )
                visual_centroids = build_visual_centroids(train_features, remapped_train, num_classes, embed_dim)
                alpha_loo = opt_base_loo_cv_alpha(
                    text_features,
                    visual_centroids,
                    train_features,
                    remapped_train,
                    num_classes,
                    alpha_grid,
                )

                if kshot < 2:
                    metrics = evaluate_one_shot_metrics(
                        text_features,
                        visual_centroids,
                        alpha_grid,
                        train_features,
                        remapped_train,
                        val_features,
                        remapped_val,
                        num_classes,
                    )
                else:
                    metrics = opt_base_eval(
                        text_features,
                        visual_centroids,
                        alpha_loo,
                        val_features,
                        remapped_val,
                    )

                values_by_metric["accuracy"].append(metrics["accuracy"])
                values_by_metric["mca"].append(metrics["mca"])

            fewshot_metrics[kshot] = {
                "accuracy": (float(np.mean(values_by_metric["accuracy"])), float(np.std(values_by_metric["accuracy"]))),
                "mca": (float(np.mean(values_by_metric["mca"])), float(np.std(values_by_metric["mca"]))),
            }

    return fewshot_metrics


def configure_plot_theme():
    sns.set_theme(
        style="ticks",
        context="talk",
        rc={
            "axes.facecolor": "#f6f1e8",
            "figure.facecolor": "#fffdf8",
            "axes.edgecolor": "#3d3a34",
            "grid.color": "#b8ad9c",
            "grid.linestyle": "--",
            "grid.alpha": 0.22,
            "axes.labelcolor": "#26231f",
            "xtick.color": "#3a3732",
            "ytick.color": "#3a3732",
        },
    )


def plot_curve_family(dataset_name, title, xlabel, series_map, output_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_theme()

    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    palette = sns.color_palette(PLOT_PALETTE[: len(series_map)])

    ax.set_facecolor("#f6f1e8")
    fig.patch.set_facecolor("#fffdf8")

    y_values = []
    for idx, (color, (series_label, curve_stats)) in enumerate(zip(palette, series_map.items())):
        alpha_values = curve_stats["alpha_values"]
        mean_curve = curve_stats["mean"]
        std_curve = curve_stats["std"]
        y_values.extend(mean_curve.tolist())
        y_values.extend((mean_curve - std_curve).tolist())
        y_values.extend((mean_curve + std_curve).tolist())

        best_idx = int(np.argmax(mean_curve))
        best_alpha = float(alpha_values[best_idx])
        best_acc = float(mean_curve[best_idx])

        ax.fill_between(
            alpha_values,
            np.clip(mean_curve - std_curve, 0.0, 100.0),
            np.clip(mean_curve + std_curve, 0.0, 100.0),
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            alpha_values,
            mean_curve,
            label=series_label,
            color=color,
            linewidth=3.0,
            marker="o",
            markersize=4.0,
            markevery=max(1, len(alpha_values) // 10),
            alpha=0.98,
            zorder=3,
        )
        ax.scatter(
            [best_alpha],
            [best_acc],
            color=color,
            s=82,
            edgecolors="#fffdf8",
            linewidths=1.2,
            zorder=5,
        )

        x_offset = 0.014 if best_alpha < 0.88 else -0.12
        y_offset = 0.45 if idx % 2 == 0 else -0.7
        ax.text(
            best_alpha + x_offset,
            best_acc + y_offset,
            f"$a^*={best_alpha:.2f}$\n$Acc_{{LOO}}={best_acc:.2f}$",
            color=color,
            fontsize=10.2,
            weight="semibold",
            ha="left" if x_offset >= 0 else "right",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.28,rounding_size=0.18",
                "facecolor": "#fffdf8",
                "edgecolor": color,
                "linewidth": 0.9,
                "alpha": 0.95,
            },
            zorder=6,
        )

    y_min = max(0.0, min(y_values) - 2.0) if y_values else 0.0
    y_max = min(100.0, max(y_values) + 2.0) if y_values else 100.0

    ax.set_xlabel(xlabel, fontsize=15, labelpad=10)
    ax.set_ylabel(r"$Acc_{LOO}$ (%)", fontsize=15, labelpad=10)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.22)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.55, alpha=0.10)
    ax.minorticks_on()

    legend = ax.legend(loc="upper left", frameon=True, framealpha=0.95, fancybox=True, borderpad=0.8)
    legend.get_title().set_fontsize(12)
    legend.get_frame().set_facecolor("#fffaf1")
    legend.get_frame().set_edgecolor("#d6c7b2")

    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()

    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def sanitize_name(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def plot_alpha_ablation(dataset_name, curves_by_kshot):
    series_map = {f"{kshot}-shot": curve_stats for kshot, curve_stats in curves_by_kshot.items()}
    output_name = f"{sanitize_name(dataset_name)}_alpha_ablation.png"
    return plot_curve_family(
        dataset_name,
        r"ProtoFuse Ablation 1: Alpha vs $Acc_{LOO}$",
        "Alpha",
        series_map,
        output_name,
    )


def aggregate_curve_stats(seed_curves, alpha_values):
    seed_curves_np = np.stack(seed_curves, axis=0)
    return {
        "alpha_values": alpha_values.copy(),
        "mean": np.mean(seed_curves_np, axis=0),
        "std": np.std(seed_curves_np, axis=0),
    }


def print_curve_summary(console, title, series_map):
    console.print(f"[bold blue]{title}[/bold blue]")
    for label, curve_stats in series_map.items():
        alpha_values = curve_stats["alpha_values"]
        mean_curve = curve_stats["mean"]
        best_idx = int(np.argmax(mean_curve))
        console.print(
            f"{label:>8} -> a*={alpha_values[best_idx]:.2f}, "
            f"mean Acc_LOO={mean_curve[best_idx]:.2f}%"
        )
    console.print()


def format_mean_std(values):
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    return f"{mean_val:.2f} ± {std_val:.2f}"


def format_pair(pair):
    return f"{pair[0]:.2f} ± {pair[1]:.2f}"


def format_comp_params(num_params):
    if num_params >= 1e6:
        return f"{num_params / 1e6:.3f}"
    if num_params >= 1e3:
        return f"{num_params / 1e3:.3f}"
    return f"{float(num_params):.0f}"


def get_result(results_by_key, key):
    return results_by_key.get(key)


class ProtoFuseCostModel(nn.Module):
    def __init__(self, clip_model, classnames, template):
        super().__init__()
        self.clip_model = clip_model
        prompts = [template.format(name.replace("_", " ")) for name in classnames]
        clip_device = next(self.clip_model.parameters()).device
        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(clip_device)
            text_features = self.clip_model.encode_text(tokens).float()
        self.register_buffer("text_prototypes", F.normalize(text_features, dim=-1))

    def forward(self, images):
        image_features = self.clip_model.encode_image(images).float()
        image_features = F.normalize(image_features, dim=-1)
        return image_features @ self.text_prototypes.T


def compute_computational_cost_results():
    cm.DEVICE = DEVICE
    coop_cfg = cm.load_config("coop")
    backbone = coop_cfg.model.backbone

    clip_model = cm.load_clip_to_cpu(backbone)
    clip_model.float()
    classnames = [f"class_{idx}" for idx in range(COMP_NUM_CLASSES)]

    coop_model, coop_params, _, coop_cfg = cm.analyze_coop(clip_model, classnames)
    coop_gflops = cm.get_gflops(coop_model)

    maple_model, maple_params, _, maple_cfg = cm.analyze_maple(clip_model, classnames)
    maple_gflops = cm.get_gflops(maple_model)

    apt_model, apt_params, _, apt_cfg = cm.analyze_apt(clip_model, classnames)
    apt_gflops = cm.get_gflops(apt_model)

    protofuse_model = ProtoFuseCostModel(
        clip_model,
        classnames,
        CUSTOM_TEMPLATES["CUB-200-2011"],
    )
    protofuse_params = 0
    protofuse_gflops = cm.get_gflops(protofuse_model)

    clip_model_fps = cm.load_clip_to_cpu(backbone)
    clip_model_fps.float()

    coop_fps, coop_latency = cm.benchmark_fps(lambda: cm.CoOPCLIP(coop_cfg, classnames, clip_model_fps))
    maple_fps, maple_latency = cm.benchmark_fps(lambda: cm.MaPLeCLIP(maple_cfg, classnames, clip_model_fps))
    apt_fps, apt_latency = cm.benchmark_fps(lambda: cm.APTCLIP(apt_cfg, classnames, clip_model_fps, DEVICE))
    protofuse_fps, protofuse_latency = cm.benchmark_fps(
        lambda: ProtoFuseCostModel(
            clip_model_fps,
            classnames,
            CUSTOM_TEMPLATES["CUB-200-2011"],
        )
    )

    return [
        {
            "method": "CoOp",
            "params_m": format_comp_params(coop_params),
            "gflops": coop_gflops,
            "fps": coop_fps,
            "latency": coop_latency,
        },
        {
            "method": "MaPLe",
            "params_m": format_comp_params(maple_params),
            "gflops": maple_gflops,
            "fps": maple_fps,
            "latency": maple_latency,
        },
        {
            "method": "APT",
            "params_m": format_comp_params(apt_params),
            "gflops": apt_gflops,
            "fps": apt_fps,
            "latency": apt_latency,
        },
        {
            "method": "ProtoFuse",
            "params_m": format_comp_params(protofuse_params),
            "gflops": protofuse_gflops,
            "fps": protofuse_fps,
            "latency": protofuse_latency,
        },
    ]


def create_computational_cost_table(cost_rows):
    table = Table(title="Computational Cost Comparison (CUB-200-2011, CLIP ViT-B/16)")
    table.add_column("Method", justify="left", style="cyan")
    table.add_column("Params (M)", justify="center")
    table.add_column("GFLOPs", justify="center")
    table.add_column("FPS", justify="center")
    table.add_column("Latency (ms)", justify="center")

    for row in cost_rows:
        table.add_row(
            row["method"],
            row["params_m"],
            f"{row['gflops']:.2f}" if row["gflops"] is not None else "N/A",
            f"{row['fps']:.2f}",
            f"{row['latency']:.2f}",
        )

    return table


def collect_class_counts(dataset):
    return np.array(count_samples_per_class(dataset), dtype=np.int32)


def plot_dataset_class_balance(specs):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_theme()

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.6))
    axes = axes.flatten()

    for idx, (ax, spec) in enumerate(zip(axes, specs)):
        dataset_path = Path(spec["path"])
        if not dataset_path.exists():
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                f"Missing dataset\n{spec['display_name']}",
                ha="center",
                va="center",
                fontsize=13,
                color="#7c6f64",
            )
            continue

        dataset = ImageFolder(str(dataset_path))
        class_counts = np.sort(collect_class_counts(dataset))[::-1]
        x = np.arange(1, len(class_counts) + 1)
        color = PLOT_PALETTE[idx % len(PLOT_PALETTE)]

        ax.bar(
            x,
            class_counts,
            width=0.88,
            color=color,
            alpha=0.96,
            edgecolor="#fffaf1",
            linewidth=0.12,
            zorder=3,
        )

        tick_positions = sorted(set([1, max(1, len(class_counts) // 2), len(class_counts)]))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(pos) for pos in tick_positions])
        ax.set_xlim(0.0, len(class_counts) + 1)
        ax.set_ylim(0.0, max(class_counts) * 1.12)
        ax.set_xlabel("Class rank", fontsize=11)
        ax.set_ylabel("Images per class", fontsize=11)
        ax.set_title(spec["display_name"], fontsize=14, pad=10, weight="semibold")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.75, alpha=0.16, zorder=1)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    fig.tight_layout(pad=1.2, w_pad=1.4, h_pad=1.2)

    output_path = OUTPUT_DIR / "dataset_class_balance.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_fewshot_table(results_by_key, metric_name, title):
    table = Table(title=title)
    table.add_column("Dataset", justify="left", style="cyan")
    table.add_column("Method", justify="left")
    for kshot in KSHOTS:
        table.add_column(f"{kshot}-shot", justify="center")

    for spec in DATASET_SPECS:
        result = get_result(results_by_key, spec["key"])
        if result is None:
            values = ["-"] * len(KSHOTS)
        else:
            values = [format_pair(result["fewshot"][kshot][metric_name]) for kshot in KSHOTS]

        table.add_row(spec["display_name"], "ProtoFuse (Ours)", *values)

    return table


def create_grid_sensitivity_table(results_by_key):
    table = Table(title=f"Grid Sensitivity ({GRID_KSHOT}-shot)")
    table.add_column("Grid size M", justify="right", style="cyan")
    table.add_column("CUB Acc", justify="center")
    table.add_column("CUB Time (ms)", justify="center")
    table.add_column("FGVC Acc", justify="center")
    table.add_column("FGVC Time (ms)", justify="center")

    cub_result = get_result(results_by_key, "cub")
    aircraft_result = get_result(results_by_key, "aircraft")

    for grid_size in GRID_SIZES:
        cub_acc = "-"
        cub_time = "-"
        air_acc = "-"
        air_time = "-"

        if cub_result is not None:
            cub_acc = format_mean_std(cub_result["ablation_2"][grid_size]["accuracy"])
            cub_time = format_mean_std(cub_result["ablation_2"][grid_size]["selection_time_ms"])

        if aircraft_result is not None:
            air_acc = format_mean_std(aircraft_result["ablation_2"][grid_size]["accuracy"])
            air_time = format_mean_std(aircraft_result["ablation_2"][grid_size]["selection_time_ms"])

        table.add_row(str(grid_size), cub_acc, cub_time, air_acc, air_time)

    return table


def create_one_shot_table(results_by_key):
    table = Table(title="One-shot Variant Comparison")
    table.add_column("Method", justify="left", style="cyan")
    for spec in DATASET_SPECS:
        table.add_column(spec["display_name"], justify="center")

    methods = [
        ("Text-only", "text_only"),
        ("Visual-only", "visual_only"),
        ("ProtoFuse (full)", "protofuse_full"),
    ]

    for display_name, metric_key in methods:
        row = [display_name]
        for spec in DATASET_SPECS:
            result = get_result(results_by_key, spec["key"])
            if result is None:
                row.append("-")
            else:
                row.append(format_mean_std(result["ablation_3"][metric_key]))
        table.add_row(*row)

    return table


def create_multishot_table(results_by_key):
    table = Table(title="Multi-shot Prototype Fusion")
    table.add_column("Dataset", justify="left", style="cyan")
    table.add_column("Method", justify="left")
    for kshot in [2, 4, 8, 16]:
        table.add_column(f"{kshot}-shot", justify="center")

    rows = [
        ("Text-only (alpha=0)", "alpha_0"),
        ("Visual-only (alpha=1)", "alpha_1"),
        ("Fixed fusion (alpha=0.5)", "alpha_05"),
        ("ProtoFuse (LOO-selected a*)", "alpha_loo"),
    ]

    for dataset_key in ["cub", "aircraft"]:
        result = get_result(results_by_key, dataset_key)
        display_name = next(spec["display_name"] for spec in DATASET_SPECS if spec["key"] == dataset_key)

        if result is None:
            for idx, (method_name, _) in enumerate(rows):
                table.add_row(display_name if idx == 0 else "", method_name, "-", "-", "-", "-", end_section=(idx == len(rows) - 1))
            continue

        metrics_by_kshot = result["ablation_4"]
        for idx, (method_name, metric_key) in enumerate(rows):
            values = [format_mean_std(metrics_by_kshot[kshot][metric_key]) for kshot in [2, 4, 8, 16]]
            table.add_row(display_name if idx == 0 else "", method_name, *values, end_section=(idx == len(rows) - 1))

    return table


def evaluate_dataset(spec, clip_model, console):
    dataset_path = Path(spec["path"])
    if not dataset_path.exists():
        console.print(f"[yellow]Skipping {spec['display_name']}[/yellow]: missing path `{dataset_path}`")
        console.print()
        return None

    dataset_name = spec.get("template_name") or infer_dataset_name(dataset_path)
    transform = get_transform()
    dataset = ImageFolder(str(dataset_path), transform=transform)
    classnames = list(dataset.classes)
    task_classes = sorted(set(label for _, label in dataset.samples))
    num_classes = len(task_classes)

    all_features, all_labels = extract_and_cache_features(
        clip_model,
        dataset,
        cache_name=f"protofuse_ablation2_{spec['key']}_{dataset_path.name}",
    )

    text_features, class_remap = get_task_text_features(clip_model, classnames, task_classes, dataset_name)
    text_features = F.normalize(text_features, dim=-1)
    embed_dim = text_features.shape[-1]
    alpha_grid = torch.linspace(0, 1, ALPHA_STEPS, device=DEVICE)
    alpha_values = alpha_grid.cpu().numpy()

    console.print(f"[bold blue]Dataset:[/bold blue] {spec['display_name']}")
    console.print(f"[bold blue]Path:[/bold blue] {dataset_path}")
    print_eval_split_balance_note(
        console,
        spec["display_name"],
        summarize_eval_split_balance(dataset, VAL_SIZE),
    )

    with console.status(f"[bold green]Running ablations for {spec['display_name']}..."):
        ablation_1 = run_ablation_1_alpha(
            all_features,
            all_labels,
            dataset,
            class_remap,
            text_features,
            alpha_grid,
            alpha_values,
            num_classes,
            embed_dim,
        )
        ablation_2 = run_ablation_2_grid_size(
            all_features,
            all_labels,
            dataset,
            class_remap,
            text_features,
            num_classes,
            embed_dim,
        )
        ablation_3 = run_ablation_3_one_shot(
            all_features,
            all_labels,
            dataset,
            class_remap,
            text_features,
            num_classes,
            embed_dim,
        )
        ablation_4 = run_ablation_4_fixed_alpha_vs_loo(
            all_features,
            all_labels,
            dataset,
            class_remap,
            text_features,
            alpha_grid,
            num_classes,
            embed_dim,
        )
        fewshot = evaluate_fewshot_protofuse(
            all_features,
            all_labels,
            dataset,
            class_remap,
            text_features,
            num_classes,
            embed_dim,
        )

    alpha_output_path = plot_alpha_ablation(spec["display_name"], ablation_1)
    console.print(f"[bold green]Saved figure:[/bold green] {alpha_output_path}")
    print_curve_summary(
        console,
        f"Ablation 1 Summary ({spec['display_name']})",
        {f"{kshot}-shot": curve_stats for kshot, curve_stats in ablation_1.items()},
    )

    return {
        "display_name": spec["display_name"],
        "dataset_name": dataset_name,
        "path": dataset_path,
        "fewshot": fewshot,
        "ablation_1_curves": ablation_1,
        "ablation_1_plot_path": alpha_output_path,
        "ablation_2": ablation_2,
        "ablation_3": ablation_3,
        "ablation_4": ablation_4,
    }


def run():
    global DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()
    DEVICE = args.device

    console = Console()
    console.print("[bold blue]PROTOFUSE ABLATION 2[/bold blue]")
    console.print(f"[bold blue]Device:[/bold blue] {DEVICE}")
    console.print(f"[bold blue]Seeds:[/bold blue] {SEEDS}")
    console.print(f"[bold blue]Shots:[/bold blue] {KSHOTS}")
    console.print()
    class_balance_plot_path = plot_dataset_class_balance(DATASET_SPECS)
    console.print(f"[bold green]Saved figure:[/bold green] {class_balance_plot_path}")
    console.print()

    clip_model = load_clip()
    results_by_key = {}

    for spec in DATASET_SPECS:
        result = evaluate_dataset(spec, clip_model, console)
        if result is not None:
            results_by_key[spec["key"]] = result

    # with console.status("[bold green]Benchmarking computational cost..."):
    #     computational_cost_rows = compute_computational_cost_results()

    console.print(create_fewshot_table(results_by_key, "accuracy", "Few-shot Classification Accuracy"))
    console.print()
    console.print(create_fewshot_table(results_by_key, "mca", "Few-shot Classification MCA"))
    console.print()
    # console.print(create_computational_cost_table(computational_cost_rows))
    # console.print()
    console.print(create_grid_sensitivity_table(results_by_key))
    console.print()
    console.print(create_one_shot_table(results_by_key))
    console.print()
    console.print(create_multishot_table(results_by_key))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run()
