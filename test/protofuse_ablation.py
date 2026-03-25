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
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from clip import clip
from src.models.apt import CUSTOM_TEMPLATES
CLIP_MODEL_PATH = Path(__file__).parent.parent / "models" / "ViT-B-16.pt"
DEFAULT_DATASET = Path(__file__).parent.parent / "datasets" / "cub-200-2011-renamed"
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
RUN_ABLATION_1 = True
RUN_ABLATION_2 = True
RUN_ABLATION_3 = True
RUN_ABLATION_4 = True

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CACHE_DIR = Path(__file__).parent.parent / "checkpoints" / "protofuse_ablation_cache"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "protofuse_ablation"
PLOT_PALETTE = [
    "#115e59",
    "#1e3a8a",
    "#9a3412",
    "#9f1239",
    "#6d28d9",
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


def get_task_text_features(
    clip_model,
    classnames,
    task_classes,
    dataset_name,
):
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
    visual_centroids = torch.zeros(num_classes, embed_dim, device=DEVICE)
    for idx in range(num_classes):
        mask = remapped_train_labels == idx
        if mask.any():
            visual_centroids[idx] = F.normalize(train_features[mask].to(DEVICE).mean(0), dim=-1)
    return visual_centroids


def remap_labels(labels, class_remap):
    return torch.tensor(
        [class_remap[label.item()] for label in labels],
        dtype=torch.long,
    )


def evaluate_discriminative_accuracy(
    T,
    V_all,
    alpha_value,
    eval_features,
    eval_labels,
    num_classes,
    one_shot_mode,
):
    eval_norm = F.normalize(eval_features.to(DEVICE), dim=-1)
    labels = eval_labels.to(DEVICE)

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


def sweep_alpha_accuracy(T, V_all, alpha_grid, eval_features, eval_labels, num_classes, one_shot_mode):
    accuracies = [
        evaluate_discriminative_accuracy(
            T,
            V_all,
            float(alpha.item()),
            eval_features,
            eval_labels,
            num_classes,
            one_shot_mode,
        )
        for alpha in alpha_grid
    ]
    return np.asarray(accuracies, dtype=np.float32)


def sweep_alpha_loo_scores(T, V_all, alpha_grid, train_features, train_labels, num_classes):
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(train_labels.tolist()):
        class_indices[label].append(idx)

    shots_per_class = min(len(indices) for indices in class_indices)
    if shots_per_class < 2:
        train_norm = F.normalize(train_features.to(DEVICE), dim=-1)
        refined = F.normalize(
            (1 - alpha_grid).view(-1, 1, 1) * T + alpha_grid.view(-1, 1, 1) * V_all,
            dim=-1,
        )
        logits = torch.einsum("qd,apd->aqp", train_norm, refined)
        preds = logits.argmax(dim=-1)
        scores = (preds == train_labels.to(DEVICE)).float().mean(dim=-1)
        return (scores.cpu().numpy() * 100.0).astype(np.float32)

    k = shots_per_class
    class_feat = torch.stack(
        [train_features[class_indices[c][:k]].to(DEVICE) for c in range(num_classes)]
    )
    class_sums = class_feat.sum(dim=1)
    loo_scores = torch.zeros(len(alpha_grid), device=DEVICE)

    for fold in range(k):
        held = F.normalize(class_feat[:, fold, :], dim=-1)
        v_loo = F.normalize((class_sums - class_feat[:, fold, :]) / (k - 1), dim=-1)
        refined = F.normalize(
            (1 - alpha_grid).view(-1, 1, 1) * T + alpha_grid.view(-1, 1, 1) * v_loo,
            dim=-1,
        )
        logits = torch.einsum("qd,apd->aqp", held, refined)
        preds = logits.argmax(dim=-1)
        loo_scores += (preds == torch.arange(num_classes, device=DEVICE)).float().mean(dim=-1)

    loo_scores = loo_scores / k
    return (loo_scores.cpu().numpy() * 100.0).astype(np.float32)


def sweep_alpha_resub_accuracy(T, V_all, alpha_grid, support_features, support_labels):
    support_norm = F.normalize(support_features.to(DEVICE), dim=-1)
    refined = F.normalize(
        (1 - alpha_grid).view(-1, 1, 1) * T + alpha_grid.view(-1, 1, 1) * V_all,
        dim=-1,
    )
    logits = torch.einsum("qd,apd->aqp", support_norm, refined)
    preds = logits.argmax(dim=-1)
    labels = support_labels.to(DEVICE).view(1, -1)
    accuracies = preds.eq(labels).float().mean(dim=-1).cpu().numpy() * 100.0
    return accuracies.astype(np.float32)


def select_alpha_loo(T, V_all, alpha_grid, train_features, train_labels, num_classes):
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(train_labels.tolist()):
        class_indices[label].append(idx)

    shots_per_class = min(len(indices) for indices in class_indices)
    if shots_per_class < 2:
        train_norm = F.normalize(train_features.to(DEVICE), dim=-1)
        refined = F.normalize(
            (1 - alpha_grid).view(-1, 1, 1) * T + alpha_grid.view(-1, 1, 1) * V_all,
            dim=-1,
        )
        logits = torch.einsum("qd,apd->aqp", train_norm, refined)
        preds = logits.argmax(dim=-1)
        scores = (preds == train_labels.to(DEVICE)).float().mean(dim=-1)
        return float(alpha_grid[scores.argmax()].item())

    k = shots_per_class
    class_feat = torch.stack(
        [train_features[class_indices[c][:k]].to(DEVICE) for c in range(num_classes)]
    )
    class_sums = class_feat.sum(dim=1)
    loo_scores = torch.zeros(len(alpha_grid), device=DEVICE)

    for fold in range(k):
        held = F.normalize(class_feat[:, fold, :], dim=-1)
        v_loo = F.normalize((class_sums - class_feat[:, fold, :]) / (k - 1), dim=-1)
        refined = F.normalize(
            (1 - alpha_grid).view(-1, 1, 1) * T + alpha_grid.view(-1, 1, 1) * v_loo,
            dim=-1,
        )
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


def evaluate_one_shot_variants(
    T,
    V_all,
    alpha_grid,
    support_features,
    support_labels,
    eval_features,
    eval_labels,
    num_classes,
):
    tau = 0.05
    eval_norm = F.normalize(eval_features.to(DEVICE), dim=-1)
    labels = eval_labels.to(DEVICE)

    logits_text = eval_norm @ T.T
    logits_visual = eval_norm @ V_all.T

    preds_text = logits_text.argmax(dim=-1)
    preds_visual = logits_visual.argmax(dim=-1)
    acc_text = preds_text.eq(labels).float().mean().item() * 100.0
    acc_visual = preds_visual.eq(labels).float().mean().item() * 100.0

    resub_accs = sweep_alpha_resub_accuracy(
        T,
        V_all,
        alpha_grid,
        support_features,
        support_labels,
    )
    best_idx = int(np.argmax(resub_accs))
    alpha_resub = float(alpha_grid[best_idx].item())

    logits_fused_resub = (1 - alpha_resub) * logits_text + alpha_resub * logits_visual
    preds_fused_resub = logits_fused_resub.argmax(dim=-1)
    acc_fused_resub = preds_fused_resub.eq(labels).float().mean().item() * 100.0

    probs_visual = F.softmax(logits_visual / tau, dim=-1)
    entropy = -(probs_visual * torch.log(probs_visual + 1e-8)).sum(dim=-1)
    confidence = 1.0 - torch.clamp(entropy / math.log(num_classes), 0.0, 1.0)
    alpha_q = alpha_resub * (0.5 + 0.5 * confidence)

    logits_full = (1 - alpha_q).unsqueeze(-1) * logits_text + alpha_q.unsqueeze(-1) * logits_visual
    preds_full = logits_full.argmax(dim=-1)
    acc_full = preds_full.eq(labels).float().mean().item() * 100.0

    return {
        "text_only": acc_text,
        "visual_only": acc_visual,
        "fused_resub": acc_fused_resub,
        "protofuse_full": acc_full,
    }


def run_ablation_1_alpha(
    all_features,
    all_labels,
    dataset,
    class_remap,
    text_features,
    alpha_grid,
    alpha_values,
    num_classes,
    embed_dim,
):
    curves_by_kshot = {}

    for kshot in (k for k in KSHOTS if k != 1):
        seed_curves = []

        for seed in SEEDS:
            train_indices, val_indices = split_by_class(dataset, VAL_SIZE, kshot, seed)
            train_features = all_features[train_indices]
            train_labels = all_labels[train_indices]
            remapped_train = remap_labels(train_labels, class_remap)

            visual_centroids = build_visual_centroids(
                train_features,
                remapped_train,
                num_classes,
                embed_dim,
            )
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


def run_ablation_2_grid_size(
    all_features,
    all_labels,
    dataset,
    class_remap,
    text_features,
    num_classes,
    embed_dim,
):
    grid_metrics = {
        grid_size: {"accuracy": [], "selection_time_ms": []}
        for grid_size in GRID_SIZES
    }

    for grid_size in GRID_SIZES:
        alpha_grid = torch.linspace(0, 1, grid_size, device=DEVICE)

        for seed in SEEDS:
            train_indices, val_indices = split_by_class(dataset, VAL_SIZE, GRID_KSHOT, seed)
            train_features = all_features[train_indices]
            train_labels = all_labels[train_indices]
            val_features = all_features[val_indices]
            val_labels = all_labels[val_indices]

            remapped_train = remap_labels(train_labels, class_remap)
            remapped_val = remap_labels(val_labels, class_remap)

            visual_centroids = build_visual_centroids(
                train_features,
                remapped_train,
                num_classes,
                embed_dim,
            )

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


def run_ablation_3_one_shot(
    all_features,
    all_labels,
    dataset,
    class_remap,
    text_features,
    alpha_grid_resub,
    num_classes,
    embed_dim,
):
    one_shot_variant_metrics = {
        "text_only": [],
        "visual_only": [],
        "fused_resub": [],
        "protofuse_full": [],
    }

    for seed in SEEDS:
        train_indices, val_indices = split_by_class(dataset, VAL_SIZE, 1, seed)
        train_features = all_features[train_indices]
        train_labels = all_labels[train_indices]
        val_features = all_features[val_indices]
        val_labels = all_labels[val_indices]

        remapped_train = remap_labels(train_labels, class_remap)
        remapped_val = remap_labels(val_labels, class_remap)

        visual_centroids = build_visual_centroids(
            train_features,
            remapped_train,
            num_classes,
            embed_dim,
        )
        variant_results = evaluate_one_shot_variants(
            text_features,
            visual_centroids,
            alpha_grid_resub,
            train_features,
            remapped_train,
            val_features,
            remapped_val,
            num_classes,
        )

        for key, value in variant_results.items():
            one_shot_variant_metrics[key].append(value)

    return one_shot_variant_metrics


def run_ablation_4_fixed_alpha_vs_loo(
    all_features,
    all_labels,
    dataset,
    class_remap,
    text_features,
    alpha_grid,
    num_classes,
    embed_dim,
):
    ablation4_kshots = [2, 4, 8, 16]
    metrics_by_kshot = {
        kshot: {"alpha_0": [], "alpha_1": [], "alpha_05": [], "alpha_loo": []}
        for kshot in ablation4_kshots
    }

    for kshot in ablation4_kshots:
        for seed in SEEDS:
            train_indices, val_indices = split_by_class(dataset, VAL_SIZE, kshot, seed)
            train_features = all_features[train_indices]
            train_labels = all_labels[train_indices]
            val_features = all_features[val_indices]
            val_labels = all_labels[val_indices]

            remapped_train = remap_labels(train_labels, class_remap)
            remapped_val = remap_labels(val_labels, class_remap)

            visual_centroids = build_visual_centroids(
                train_features,
                remapped_train,
                num_classes,
                embed_dim,
            )

            metrics_by_kshot[kshot]["alpha_0"].append(
                evaluate_fixed_alpha_accuracy(
                    text_features,
                    visual_centroids,
                    0.0,
                    val_features,
                    remapped_val,
                )
            )
            metrics_by_kshot[kshot]["alpha_1"].append(
                evaluate_fixed_alpha_accuracy(
                    text_features,
                    visual_centroids,
                    1.0,
                    val_features,
                    remapped_val,
                )
            )
            metrics_by_kshot[kshot]["alpha_05"].append(
                evaluate_fixed_alpha_accuracy(
                    text_features,
                    visual_centroids,
                    0.5,
                    val_features,
                    remapped_val,
                )
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
                evaluate_fixed_alpha_accuracy(
                    text_features,
                    visual_centroids,
                    alpha_loo,
                    val_features,
                    remapped_val,
                )
            )

    return metrics_by_kshot


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

    # ax.set_title(f"{title}\n{dataset_name}", fontsize=20, weight="bold", pad=18, color="#1f1c18")
    ax.set_xlabel(xlabel, fontsize=15, labelpad=10)
    ax.set_ylabel(r"$Acc_{LOO}$ (%)", fontsize=15, labelpad=10)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.22)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.55, alpha=0.10)
    ax.minorticks_on()

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        fancybox=True,
        borderpad=0.8,
    )
    legend.get_title().set_fontsize(12)
    legend.get_frame().set_facecolor("#fffaf1")
    legend.get_frame().set_edgecolor("#d6c7b2")

    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()

    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_alpha_ablation(dataset_name, curves_by_kshot):
    series_map = {
        f"{kshot}-shot": curve_stats for kshot, curve_stats in curves_by_kshot.items()
    }
    output_name = f"{Path(dataset_name).stem.lower().replace(' ', '_')}_alpha_ablation.png"
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
    return f"{mean_val:.2f} +- {std_val:.2f}"


def create_grid_size_table(grid_metrics):
    table = Table(title=f"ProtoFuse Ablation 2: Grid Size ({GRID_KSHOT}-shot)")
    table.add_column("M", justify="right", style="cyan", no_wrap=True)
    table.add_column("Accuracy", justify="right")
    table.add_column("Selection Time (ms)", justify="right")

    for grid_size in GRID_SIZES:
        metrics = grid_metrics[grid_size]
        table.add_row(
            str(grid_size),
            format_mean_std(metrics["accuracy"]),
            format_mean_std(metrics["selection_time_ms"]),
        )

    return table


def create_one_shot_variant_table(variant_metrics):
    table = Table(title="ProtoFuse Ablation 3: One-Shot Variant Comparison")
    table.add_column("Text-only", justify="center")
    table.add_column("Visual-only", justify="center")
    table.add_column("Fused (a*_{resub})", justify="center")
    table.add_column("ProtoFuse (full)", justify="center")

    table.add_row(
        format_mean_std(variant_metrics["text_only"]),
        format_mean_std(variant_metrics["visual_only"]),
        format_mean_std(variant_metrics["fused_resub"]),
        format_mean_std(variant_metrics["protofuse_full"]),
    )
    return table


def create_fixed_alpha_vs_loo_table(metrics_by_kshot):
    table = Table(title="ProtoFuse Ablation 4: Fixed Alpha vs LOO")
    table.add_column("K", justify="right", style="cyan", no_wrap=True)
    table.add_column("alpha = 0", justify="center")
    table.add_column("alpha = 1", justify="center")
    table.add_column("alpha = 0.5", justify="center")
    table.add_column("a* (LOO)", justify="center")

    for kshot in [2, 4, 8, 16]:
        metrics = metrics_by_kshot[kshot]
        table.add_row(
            str(kshot),
            format_mean_std(metrics["alpha_0"]),
            format_mean_std(metrics["alpha_1"]),
            format_mean_std(metrics["alpha_05"]),
            format_mean_std(metrics["alpha_loo"]),
        )

    return table


def run():
    global DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    DEVICE = args.device
    dataset_path = Path(args.dataset)
    dataset_name = infer_dataset_name(dataset_path)

    console = Console()
    console.print("[bold blue]PROTOFUSE ABLATIONS[/bold blue]")
    console.print(f"[bold blue]Dataset:[/bold blue] {dataset_name}")
    console.print(f"[bold blue]Path:[/bold blue] {dataset_path}")
    console.print(f"[bold blue]Device:[/bold blue] {DEVICE}")
    console.print(f"[bold blue]K-shot:[/bold blue] {KSHOTS}")
    console.print(f"[bold blue]Grid M:[/bold blue] {GRID_SIZES} (fixed {GRID_KSHOT}-shot)")
    console.print(f"[bold blue]Seeds:[/bold blue] {SEEDS}")
    console.print()

    clip_model = load_clip()
    transform = get_transform()
    dataset = ImageFolder(str(dataset_path), transform=transform)
    all_features, all_labels = extract_and_cache_features(
        clip_model,
        dataset,
        cache_name=f"protofuse_ablation_{dataset_path.name}",
    )

    classnames = list(dataset.classes)
    task_classes = sorted(set(label for _, label in dataset.samples))
    text_features, class_remap = get_task_text_features(
        clip_model,
        classnames,
        task_classes,
        dataset_name,
    )
    text_features = F.normalize(text_features, dim=-1)
    alpha_grid = torch.linspace(0, 1, ALPHA_STEPS, device=DEVICE)
    alpha_grid_resub = torch.linspace(0, 1, ALPHA_STEPS, device=DEVICE)
    alpha_values = alpha_grid.cpu().numpy()

    num_classes = len(task_classes)
    embed_dim = text_features.shape[-1]
    with console.status("[bold green]Sweeping alpha for ProtoFuse..."):
        ablation_1_results = None
        ablation_2_results = None
        ablation_3_results = None
        ablation_4_results = None

        if RUN_ABLATION_1:
            ablation_1_results = run_ablation_1_alpha(
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

        if RUN_ABLATION_2:
            ablation_2_results = run_ablation_2_grid_size(
                all_features,
                all_labels,
                dataset,
                class_remap,
                text_features,
                num_classes,
                embed_dim,
            )

        if RUN_ABLATION_3:
            ablation_3_results = run_ablation_3_one_shot(
                all_features,
                all_labels,
                dataset,
                class_remap,
                text_features,
                alpha_grid_resub,
                num_classes,
                embed_dim,
            )

        if RUN_ABLATION_4:
            ablation_4_results = run_ablation_4_fixed_alpha_vs_loo(
                all_features,
                all_labels,
                dataset,
                class_remap,
                text_features,
                alpha_grid,
                num_classes,
                embed_dim,
            )

    if ablation_1_results is not None:
        alpha_output_path = plot_alpha_ablation(dataset_name, ablation_1_results)
        console.print(f"[bold green]Saved figure:[/bold green] {alpha_output_path}")
        console.print()
        print_curve_summary(
            console,
            "Ablation 1 Summary",
            {f"{kshot}-shot": curve_stats for kshot, curve_stats in ablation_1_results.items()},
        )

    if ablation_2_results is not None:
        console.print(create_grid_size_table(ablation_2_results))
        console.print()

    if ablation_3_results is not None:
        console.print(create_one_shot_variant_table(ablation_3_results))

    if ablation_4_results is not None:
        console.print()
        console.print(create_fixed_alpha_vs_loo_table(ablation_4_results))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run()
