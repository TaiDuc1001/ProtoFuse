import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import cv2
import matplotlib

if matplotlib.get_backend().lower() != 'agg':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

import umap


def generate_confusion_matrix_plot(args):
    cm, row_idx, col_idx, start_row, start_col, end_row, end_col, epoch, cm_dir = args
    sub_cm = cm[start_row:end_row, start_col:end_col]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        sub_cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        cbar=True,
        xticklabels=[str(j) for j in range(start_col, end_col)],
        yticklabels=[str(j) for j in range(start_row, end_row)],
        annot_kws={"size": 6},
    )
    ax.set_title(
        f'Confusion Matrix - Epoch {epoch} (True: {start_row}-{end_row - 1}, '
        f'Pred: {start_col}-{end_col - 1})',
        fontsize=10,
    )
    ax.set_xlabel('Predicted Label', fontsize=8)
    ax.set_ylabel('True Label', fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(cm_dir, f'confusion_matrix_r{row_idx:02d}_c{col_idx:02d}.pdf'),
        dpi=100,
        bbox_inches='tight',
    )
    plt.close()


def save_confusion_artifacts(
    all_labels: List[int],
    all_preds: List[int],
    epoch: int,
    epoch_dir: str,
    log_file: Optional[str],
    chunk_size: int = 50,
    step: int = 50,
    max_processes: int = 8,
) -> None:
    if not all_labels or not all_preds:
        return

    cm_dir = os.path.join(epoch_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    num_classes = cm.shape[0]

    if num_classes > chunk_size:
        num_blocks_per_dim = max(1, (num_classes - chunk_size) // step + 1)
        plot_args = []
        for row_idx in range(num_blocks_per_dim):
            for col_idx in range(num_blocks_per_dim):
                start_row = row_idx * step
                end_row = min(start_row + chunk_size, num_classes)
                start_col = col_idx * step
                end_col = min(start_col + chunk_size, num_classes)
                plot_args.append(
                    (cm, row_idx, col_idx, start_row, start_col, end_row, end_col, epoch, cm_dir)
                )
        if plot_args:
            with mp.Pool(processes=min(mp.cpu_count(), max_processes)) as pool:
                pool.map(generate_confusion_matrix_plot, plot_args)
    else:
        fig, ax = plt.subplots(
            figsize=(max(16, num_classes // 2), max(16, num_classes // 2))
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            cbar=True,
            xticklabels=[str(i) for i in range(num_classes)],
            yticklabels=[str(i) for i in range(num_classes)],
            annot_kws={"size": 16},
        )
        ax.set_title(f'Confusion Matrix - Epoch {epoch}', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(cm_dir, 'confusion_matrix.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

    summary = f"Confusion matrices for epoch {epoch} saved to {cm_dir}"
    print(summary)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(summary + '\n')


def save_class_distribution_plot(
    all_labels: List[int],
    all_preds: List[int],
    epoch: int,
    epoch_dir: str,
    log_file: Optional[str],
    classnames: Optional[List[str]] = None,
) -> None:
    if not all_labels and not all_preds:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    gt_counts = Counter(all_labels)
    pred_counts = Counter(all_preds)

    classes = sorted(set(gt_counts.keys()) | set(pred_counts.keys()))
    if not classes:
        return

    labels = [classnames[c] if classnames and c < len(classnames) else str(c) for c in classes]
    gt_values = [gt_counts.get(cls, 0) for cls in classes]
    pred_values = [pred_counts.get(cls, 0) for cls in classes]

    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(x - width / 2, gt_values, width, label='Ground Truth', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width / 2, pred_values, width, label='Predictions', color='salmon', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Class Distribution - Epoch {epoch}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    max_height = max(gt_values + pred_values) if (gt_values or pred_values) else 0
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max_height * 0.01 if max_height > 0 else 0.5,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max_height * 0.01 if max_height > 0 else 0.5,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=8,
        )

    plt.tight_layout()
    output_path = os.path.join(epoch_dir, 'class_distribution.pdf')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    summary = f"Class distribution plot for epoch {epoch} saved to {output_path}"
    print(summary)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(summary + '\n')


def _flatten_score_values(score_map: Dict[int, List[Any]]) -> List[float]:
    values: List[float] = []
    for entries in score_map.values():
        for item in entries:
            if isinstance(item, (list, tuple)) and item:
                values.append(float(item[0]))
            elif isinstance(item, (int, float)):
                values.append(float(item))
            elif isinstance(item, str):
                try:
                    values.append(float(item))
                except ValueError:
                    continue
    return values


def _plot_score_distribution(
    values: List[float],
    title: str,
    xlabel: str,
    output_path: str,
    color: str,
    log_file: Optional[str],
) -> None:
    if not values:
        return

    directory = os.path.dirname(output_path) or '.'
    os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(10, 6))
    bins = min(50, max(10, len(values) // 5))
    sns.histplot(values, bins=bins, kde=True, color=color, alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    summary = f"{title} saved to {output_path}"
    print(summary)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(summary + '\n')


def plot_entropy_distribution(
    entropy_scores: Dict[int, List[Any]],
    round_idx: int,
    output_path: str,
    log_file: Optional[str],
) -> None:
    values = _flatten_score_values(entropy_scores)
    title = f"Entropy Score Distribution - Round {round_idx}"
    _plot_score_distribution(values, title, 'Entropy', output_path, '#3b7dd8', log_file)


def plot_conflict_distribution(
    conflict_scores: Dict[int, List[Any]],
    round_idx: int,
    output_path: str,
    log_file: Optional[str],
) -> None:
    values = _flatten_score_values(conflict_scores)
    title = f"Conflict Score Distribution - Round {round_idx}"
    _plot_score_distribution(values, title, 'KL-Divergence', output_path, '#d83b73', log_file)


def plot_bald_distribution(
    bald_scores: Dict[int, List[Any]],
    round_idx: int,
    output_path: str,
    log_file: Optional[str],
) -> None:
    values = _flatten_score_values(bald_scores)
    title = f"BALD Score Distribution - Round {round_idx}"
    _plot_score_distribution(values, title, 'BALD', output_path, '#8b5cf6', log_file)


def _prepare_coreset_embedding_matrix(
    embeddings: Dict[int, torch.Tensor],
    labeled_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    val_indices: Sequence[int],
    selected_indices: Sequence[int],
):
    if not embeddings:
        return None, None

    status_map: Dict[int, str] = {}
    for idx in val_indices:
        status_map[int(idx)] = 'val'
    for idx in unlabeled_indices:
        status_map[int(idx)] = 'unlabeled'
    for idx in labeled_indices:
        status_map[int(idx)] = 'labeled'
    for idx in selected_indices:
        status_map[int(idx)] = 'selected'

    vectors: List[np.ndarray] = []
    statuses: List[str] = []
    for idx, status in status_map.items():
        vec = embeddings.get(idx)
        if vec is None:
            continue
        if isinstance(vec, torch.Tensor):
            vec_np = vec.detach().cpu().numpy()
        else:
            vec_np = np.asarray(vec)
        if vec_np.ndim > 1:
            vec_np = vec_np.reshape(-1)
        vectors.append(vec_np.astype(np.float32))
        statuses.append(status)

    if len(vectors) < 2:
        return None, None

    matrix = np.stack(vectors)
    return matrix, np.array(statuses)


def _plot_embedding_projection(
    coords: np.ndarray,
    statuses: np.ndarray,
    method_name: str,
    round_idx: int,
    output_path: str,
    log_file: Optional[str],
) -> None:
    if coords.size == 0:
        return

    directory = os.path.dirname(output_path) or '.'
    os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(10, 8))
    order = ['val', 'unlabeled', 'labeled', 'selected']
    style_map = {
        'val': {'color': "#d1d14c", 'marker': 'o', 'size': 25, 'alpha': 0.3, 'edgecolors': 'black', 'linewidths': 0.5},
        'unlabeled': {'color': '#7f7f7f', 'marker': 'o', 'size': 25, 'alpha': 0.3, 'edgecolors': 'black', 'linewidths': 0.5},
        'labeled': {'color': "#464fb4", 'marker': 'o', 'size': 25, 'alpha': 0.3, 'edgecolors': 'black', 'linewidths': 0.5},
        'selected': {'color': "#ee7272", 'marker': 'o', 'size': 25, 'alpha': 0.3, 'edgecolors': 'black', 'linewidths': 0.5},
    }

    label_map = {
        'val': 'Validation',
        'unlabeled': 'Unlabeled',
        'labeled': 'Labeled',
        'selected': 'Selected',
    }

    for status in order:
        mask = statuses == status
        if not np.any(mask):
            continue
        style = style_map.get(status, style_map['unlabeled'])
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=style['size'],
            c=style['color'],
            alpha=style['alpha'],
            marker=style['marker'],
            label=label_map.get(status, status.capitalize()),
            edgecolors=style.get('edgecolors', 'none'),
            linewidths=style.get('linewidths', 0.5),
        )

    plt.title(f"Coreset Embeddings ({method_name}) - Round {round_idx}")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()

    summary = f"Coreset embedding plot ({method_name}) saved to {output_path}"
    print(summary)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(summary + '\n')


def plot_coreset_embedding_umap(
    embeddings: Dict[int, torch.Tensor],
    labeled_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    val_indices: Sequence[int],
    selected_indices: Sequence[int],
    round_idx: int,
    output_path: str,
    log_file: Optional[str],
    random_state: int = 42,
) -> None:
    if umap is None:
        msg = "UMAP is not installed; skipping UMAP plot."
        print(msg)
        if log_file is not None:
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
        return

    matrix, statuses = _prepare_coreset_embedding_matrix(
        embeddings, labeled_indices, unlabeled_indices, val_indices, selected_indices
    )
    if matrix is None or statuses is None:
        print("No embeddings available for UMAP plot; skipping.")
        return

    n_neighbors = max(2, min(7, matrix.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=None, n_jobs=-1)
    coords = reducer.fit_transform(matrix)
    _plot_embedding_projection(coords, statuses, 'UMAP', round_idx, output_path, log_file) # type: ignore


def plot_coreset_embedding_tsne(
    embeddings: Dict[int, torch.Tensor],
    labeled_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    val_indices: Sequence[int],
    selected_indices: Sequence[int],
    round_idx: int,
    output_path: str,
    log_file: Optional[str],
    random_state: int = 42,
) -> None:
    matrix, statuses = _prepare_coreset_embedding_matrix(
        embeddings, labeled_indices, unlabeled_indices, val_indices, selected_indices
    )
    if matrix is None or statuses is None:
        print("No embeddings available for t-SNE plot; skipping.")
        return

    if matrix.shape[0] < 3:
        msg = "Not enough samples for t-SNE projection; skipping plot."
        print(msg)
        if log_file is not None:
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
        return
    
    n_samples = matrix.shape[0]
    perplexity = max(5, min(50, n_samples // 100))
    # perplexity = max(5, min(30, matrix.shape[0] - 1))
    reducer = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=random_state, metric='cosine', n_jobs=-1, early_exaggeration=4.0)
    coords = reducer.fit_transform(matrix)
    _plot_embedding_projection(coords, statuses, 't-SNE', round_idx, output_path, log_file)


def visualize_attention_maps(
    trainer,
    dataset,
    sample_cache: Dict[str, Any],
    classnames: List[str],
    epoch: int,
    maps_dir: str,
    log_file: Optional[str],
) -> None:
    images = sample_cache.get('images') if sample_cache else None
    labels = sample_cache.get('labels') if sample_cache else None
    paths = sample_cache.get('paths', []) if sample_cache else []

    if trainer is None or images is None:
        return

    trainer.model.cfg['mode'] = 'map'

    if isinstance(images, torch.Tensor):
        vis_images = images.to(trainer.device)
    elif isinstance(images, (list, tuple)):
        vis_images = torch.stack([
            x.to(trainer.device) if isinstance(x, torch.Tensor) else torch.tensor(x).to(trainer.device)
            for x in images
        ])
    else:
        vis_images = torch.tensor(images).to(trainer.device)

    if labels is not None:
        if isinstance(labels, torch.Tensor):
            vis_labels = labels.to(trainer.device)
        else:
            vis_labels = torch.tensor(labels).to(trainer.device)
    else:
        vis_labels = None

    logits, attn_maps = trainer.model(vis_images)
    trainer.model.cfg['mode'] = trainer.cfg.get('mode', 'logits')

    if not attn_maps:
        return

    attn_map_to_vis = attn_maps[0]
    try:
        shape_info = getattr(attn_map_to_vis, 'shape', None)
        shape_msg = f"Epoch {epoch} attention map shape: {shape_info}"
        print(shape_msg)
        if log_file is not None:
            with open(log_file, 'a') as lf:
                lf.write(shape_msg + '\n')
    except Exception:
        pass

    for i in range(len(vis_images)):
        image_path = paths[i] if i < len(paths) else None
        if dataset is None or image_path is None or vis_labels is None:
            continue

        label = int(vis_labels[i].item())
        try:
            weights = attn_map_to_vis[i, label, :]
        except Exception:
            warn_msg = (
                f"Warning: unable to index attention map for image {i}, label {label}. Skipping visualization."
            )
            print(warn_msg)
            if log_file is not None:
                with open(log_file, 'a') as lf:
                    lf.write(warn_msg + '\n')
            continue

        if weights.dim() > 1:
            mean_weights = weights.mean(dim=0).detach().cpu().numpy()
        else:
            mean_weights = weights.detach().cpu().numpy()

        patch_weights = mean_weights[1:]
        num_patches = patch_weights.shape[0]
        h = w = int(np.sqrt(num_patches))
        if h * w != num_patches:
            warn_msg = (
                f"Warning: Cannot reshape {num_patches} patches into a square grid. Skipping visualization for image {i}."
            )
            print(warn_msg)
            if log_file is not None:
                with open(log_file, 'a') as lf:
                    lf.write(warn_msg + '\n')
            continue

        heatmap = patch_weights.reshape(h, w)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)

        original_img = cv2.imread(image_path)
        if original_img is None:
            continue
        original_img = cv2.resize(original_img, (224, 224))
        heatmap_img = cv2.applyColorMap(cv2.resize(heatmap, (224, 224)), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_img, 0.4, 0)

        class_name = classnames[label] if label < len(classnames) else f"Class_{label}"
        save_name = f"epoch_{epoch:03d}_img_{i}_class_{label}_{class_name}.jpg"
        save_path = os.path.join(maps_dir, save_name)
        cv2.imwrite(save_path, superimposed_img)

    log_str = f"Saved {len(vis_images)} attention visualizations to {maps_dir}"
    print(log_str)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')


def visualize_gradcam_maps(
    trainer,
    dataset,
    sample_cache: Dict[str, Optional[torch.Tensor]],
    classnames: List[str],
    epoch: int,
    maps_dir: str,
    log_file: Optional[str],
) -> None:
    images = sample_cache.get('images') if sample_cache else None
    labels = sample_cache.get('labels') if sample_cache else None
    paths = sample_cache.get('paths', []) if sample_cache else []
    if not isinstance(paths, list):
        paths = []

    if trainer is None or images is None or labels is None:
        return

    if isinstance(images, torch.Tensor):
        vis_images = images.to(trainer.device)
    elif isinstance(images, (list, tuple)):
        vis_images = torch.stack([
            x.to(trainer.device) if isinstance(x, torch.Tensor) else torch.tensor(x).to(trainer.device)
            for x in images  # type: ignore
        ])
    else:
        vis_images = torch.tensor(images).to(trainer.device)

    if isinstance(labels, torch.Tensor):
        vis_labels = labels.to(trainer.device)
    else:
        vis_labels = torch.tensor(labels).to(trainer.device)

    gradcams = trainer.generate_gradcam(vis_images, vis_labels)

    for i, gradcam in enumerate(gradcams):
        image_path: Optional[str] = paths[i] if i < len(paths) else None
        if dataset is None or image_path is None:
            continue
        label = int(vis_labels[i].item())

        heatmap = gradcam.astype(np.float32)
        heatmap = (heatmap * 255).astype(np.uint8)

        original_img = cv2.imread(image_path)
        if original_img is None:
            continue
        original_img = cv2.resize(original_img, (224, 224))
        heatmap_img = cv2.applyColorMap(cv2.resize(heatmap, (224, 224)), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_img, 0.4, 0)

        class_name = classnames[label] if label < len(classnames) else f"Class_{label}"
        save_name = f"gradcam_epoch_{epoch:03d}_img_{i}_class_{label}_{class_name}.jpg"
        save_path = os.path.join(maps_dir, save_name)
        cv2.imwrite(save_path, superimposed_img)

    log_str = f"Saved {len(gradcams)} GradCAM visualizations to {maps_dir}"
    print(log_str)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')


def run_dataset_eda(dataset, eda_dir: str, sample_limit: int = 512, seed: int = 42) -> None:
    if dataset is None or not hasattr(dataset, 'samples'):
        return

    os.makedirs(eda_dir, exist_ok=True)
    class_counts = Counter(label for _, label in dataset.samples)
    if not class_counts:
        return

    classes = list(range(len(dataset.classes)))
    labels = [dataset.classes[i] if i < len(dataset.classes) else str(i) for i in classes]
    counts = [class_counts.get(i, 0) for i in classes]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, counts, color='teal', alpha=0.85)
    ax.set_title('Dataset Class Balance Overview')
    ax.set_xlabel('Class')
    ax.set_ylabel('Image Count')
    ax.tick_params(axis='x', rotation=90)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    class_balance_path = os.path.join(eda_dir, 'eda_class_balance.png')
    plt.savefig(class_balance_path, dpi=200)
    plt.close()

    rng = random.Random(seed)
    sampled = list(dataset.samples)
    rng.shuffle(sampled)
    sampled = sampled[:sample_limit]

    widths, heights, brightness = [], [], []
    for path, _ in sampled:
        try:
            with Image.open(path) as img:
                widths.append(img.width)
                heights.append(img.height)
                brightness.append(float(np.array(img.convert('L')).mean()))
        except Exception:
            continue

    if widths and heights:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(widths, heights, alpha=0.3, color='purple')
        ax.set_title('Image Resolution Scatter (Width vs Height)')
        ax.set_xlabel('Width (px)')
        ax.set_ylabel('Height (px)')
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        scatter_path = os.path.join(eda_dir, 'eda_resolution_scatter.png')
        plt.savefig(scatter_path, dpi=200)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.kdeplot(widths, label='Width', fill=True, ax=ax)
        sns.kdeplot(heights, label='Height', fill=True, ax=ax)
        ax.set_title('Image Dimension Density')
        ax.legend()
        plt.tight_layout()
        density_path = os.path.join(eda_dir, 'eda_resolution_density.png')
        plt.savefig(density_path, dpi=200)
        plt.close()

    if brightness:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(brightness, bins=30, kde=True, color='goldenrod', ax=ax)
        ax.set_title('Average Image Brightness Distribution')
        ax.set_xlabel('Mean Pixel Intensity (0-255)')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        brightness_path = os.path.join(eda_dir, 'eda_brightness_hist.png')
        plt.savefig(brightness_path, dpi=200)
        plt.close()

    print(f"Dataset EDA artifacts saved to {eda_dir}")