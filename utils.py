import matplotlib

if matplotlib.get_backend().lower() != 'agg':
    matplotlib.use('Agg')

import os
import sys
import cv2
import csv
import copy
import json
import yaml
import umap
import torch
import random
import hashlib
import datetime
import argparse
import numpy as np
from clip import clip
from PIL import Image
import seaborn as sns
import multiprocessing as mp
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Any, Dict, List, Optional, Sequence
from sklearn.metrics import confusion_matrix

from logger import logger, setup_logging # type: ignore


class CheckpointCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.index_path = os.path.join(cache_dir, 'index.csv')

    def _get_key_settings(self, config) -> dict:
        if hasattr(config, 'to_dict'):
            config = config.to_dict()
        model_cfg = config.get('model', {})
        method_params = {k: v for k, v in model_cfg.items() 
                         if k not in ('backbone', 'dataset_name', 'use_cache')}
        return {
            'dataset_root': config.get('data', {}).get('root'),
            'kshot': config.get('data', {}).get('kshot'),
            'seed': config.get('data', {}).get('seed'),
            'epochs': config.get('training', {}).get('epochs'),
            'backbone': model_cfg.get('backbone'),
            'method_params': json.dumps(method_params, sort_keys=True),
        }

    def compute_checkpoint_id(self, config) -> str:
        key_settings = self._get_key_settings(config)
        key_str = json.dumps(key_settings, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def get_checkpoint_path(self, checkpoint_id: str) -> str:
        return os.path.join(self.cache_dir, f'{checkpoint_id}.pt')

    def exists(self, checkpoint_id: str) -> bool:
        return os.path.exists(self.get_checkpoint_path(checkpoint_id))

    def _update_index(self, checkpoint_id: str, key_settings: dict, path: str):
        rows = []
        fieldnames = ['checkpoint_id', 'file', 'dataset_root', 'kshot', 'seed',
                      'epochs', 'backbone', 'method_params', 'created_at']
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader if row.get('checkpoint_id') != checkpoint_id]
        row = {
            'checkpoint_id': checkpoint_id,
            'file': os.path.basename(path),
            'dataset_root': key_settings.get('dataset_root'),
            'kshot': key_settings.get('kshot'),
            'seed': key_settings.get('seed'),
            'epochs': key_settings.get('epochs'),
            'backbone': key_settings.get('backbone'),
            'method_params': key_settings.get('method_params'),
            'created_at': datetime.datetime.now().isoformat(),
        }
        rows.append(row)
        with open(self.index_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save(self, checkpoint_id, model_state, optimizer_state, scheduler_state,
             labeled_indices, unlabeled_indices, metrics, config):
        key_settings = self._get_key_settings(config)
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'labeled_indices': labeled_indices,
            'unlabeled_indices': unlabeled_indices,
            'metrics': metrics,
            'config_snapshot': config.to_dict() if hasattr(config, 'to_dict') else dict(config),
            'timestamp': datetime.datetime.now().isoformat(),
        }
        path = self.get_checkpoint_path(checkpoint_id)
        torch.save(checkpoint, path)
        self._update_index(checkpoint_id, key_settings, path)
        return path

    def load(self, checkpoint_id):
        path = self.get_checkpoint_path(checkpoint_id)
        if not os.path.exists(path):
            return None
        return torch.load(path, map_location='cpu')


class ConfigNode(dict):
    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        super().__init__()
        if initial:
            self.update(initial)

    def _convert(self, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            return ConfigNode(value)
        if isinstance(value, list):
            return [self._convert(item) for item in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(f"Config key '{item}' not found") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._convert(value)

    def update(self, *args, **kwargs) -> None:
        for key, value in dict(*args, **kwargs).items():
            super().__setitem__(key, self._convert(value))

    def copy(self) -> "ConfigNode":
        return ConfigNode(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, ConfigNode) else item for item in value]
            else:
                result[key] = value
        return result


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deep_merge_dicts(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge_dicts(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def build_config_namespace(base_config: Dict[str, Any], extra_values: Optional[Dict[str, Any]] = None) -> ConfigNode:
    config_copy = copy.deepcopy(base_config)
    if extra_values:
        meta = config_copy.setdefault('meta', {})
        deep_merge_dicts(meta, extra_values)
    return ConfigNode(config_copy)


def load_config_file(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping.")
    return data


def get_config_value(config, path, default=None):
    current = config
    for key in path.split('.'):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def merge_configs(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_configs(base[key], value)
        else:
            base[key] = value
    return base


def set_nested_value(config, keys, value):
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def coerce_to_str(value, default, key=None):
    if value is None:
        return str(default)
    if isinstance(value, (list, dict)):
        raise ValueError(f"Configuration value for {key or 'unknown'} must be a string.")
    return str(value)


def coerce_to_int(value, default, key=None):
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError as exc:
                raise ValueError(f"Configuration value for {key or 'unknown'} must be numeric.") from exc
    raise ValueError(f"Configuration value for {key or 'unknown'} must be numeric.")


def coerce_to_float(value, default, key=None):
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Configuration value for {key or 'unknown'} must be a float.") from exc
    raise ValueError(f"Configuration value for {key or 'unknown'} must be a float.")


def create_argument_parser(description, arg_schema):
    parser = argparse.ArgumentParser(description=description)
    for arg_name, spec in arg_schema.items():
        kwargs = {
            'type': spec['type'],
            'help': spec['help']
        }
        if spec.get('required'):
            kwargs['required'] = True
        else:
            kwargs['default'] = None
        parser.add_argument(f'--{arg_name}', **kwargs)
    return parser


def process_parsed_args(parsed_args, arg_schema, overrides):
    for arg_name, spec in arg_schema.items():
        value = getattr(parsed_args, arg_name)
        if value is not None and 'config_path' in spec:
            keys = spec['config_path'].split('.')
            set_nested_value(overrides, keys, value)
    return overrides


def infer_override_value(raw):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None
    try:
        if raw.startswith(("0x", "-0x", "0X", "-0X")):
            return int(raw, 16)
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_override_arguments(tokens):
    overrides = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        key_token = token[2:]
        if "=" in key_token:
            key_part, raw_value = key_token.split("=", 1)
            value = infer_override_value(raw_value)
        else:
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                value = infer_override_value(tokens[i + 1])
                i += 1
            else:
                value = True
            key_part = key_token
        if not key_part:
            i += 1
            continue
        keys = key_part.split(".")
        set_nested_value(overrides, keys, value)
        i += 1
    return overrides


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='./models')

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model




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
    chunk_size: int = 50,
    step: int = 50,
    max_processes: int = 8,
) -> None:
    if not all_labels or not all_preds:
        logger.debug("save_confusion_artifacts: empty labels/preds")
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

    logger.debug(f"Confusion matrices for epoch {epoch} saved to {cm_dir}")


def save_class_distribution_plot(
    all_labels: List[int],
    all_preds: List[int],
    epoch: int,
    epoch_dir: str,
    classnames: Optional[List[str]] = None,
) -> None:
    if not all_labels and not all_preds:
        logger.debug("save_class_distribution_plot: empty labels/preds")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    gt_counts = Counter(all_labels)
    pred_counts = Counter(all_preds)

    classes = sorted(set(gt_counts.keys()) | set(pred_counts.keys()))
    if not classes:
        logger.debug("save_class_distribution_plot: no classes")
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

    logger.debug(f"Class distribution plot for epoch {epoch} saved")


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
) -> None:
    if not values:
        logger.debug("_plot_score_distribution: no values")
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

    logger.debug(f"{title} saved")


def plot_entropy_distribution(
    entropy_scores: Dict[int, List[Any]],
    round_idx: int,
    output_path: str,
) -> None:
    values = _flatten_score_values(entropy_scores)
    title = f"Entropy Score Distribution - Round {round_idx}"
    _plot_score_distribution(values, title, 'Entropy', output_path, '#3b7dd8')


def plot_conflict_distribution(
    conflict_scores: Dict[int, List[Any]],
    round_idx: int,
    output_path: str,
) -> None:
    values = _flatten_score_values(conflict_scores)
    title = f"Conflict Score Distribution - Round {round_idx}"
    _plot_score_distribution(values, title, 'KL-Divergence', output_path, '#d83b73')


def plot_bald_distribution(
    bald_scores: Dict[int, List[Any]],
    round_idx: int,
    output_path: str,
) -> None:
    values = _flatten_score_values(bald_scores)
    title = f"BALD Score Distribution - Round {round_idx}"
    _plot_score_distribution(values, title, 'BALD', output_path, '#8b5cf6')


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
) -> None:
    if coords.size == 0:
        logger.debug("_plot_embedding_projection: empty coords")
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

    logger.debug(f"Coreset embedding plot ({method_name}) saved")


def plot_coreset_embedding_umap(
    embeddings: Dict[int, torch.Tensor],
    labeled_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    val_indices: Sequence[int],
    selected_indices: Sequence[int],
    round_idx: int,
    output_path: str,
    random_state: int = 42,
) -> None:
    if umap is None:
        logger.warning("UMAP is not installed; skipping UMAP plot")
        return

    matrix, statuses = _prepare_coreset_embedding_matrix(
        embeddings, labeled_indices, unlabeled_indices, val_indices, selected_indices
    )
    if matrix is None or statuses is None:
        logger.debug("No embeddings available for UMAP plot; skipping")
        return

    n_neighbors = max(2, min(7, matrix.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=None, n_jobs=-1)
    coords = np.asarray(reducer.fit_transform(matrix))
    _plot_embedding_projection(coords, statuses, 'UMAP', round_idx, output_path)


def plot_coreset_embedding_tsne(
    embeddings: Dict[int, torch.Tensor],
    labeled_indices: Sequence[int],
    unlabeled_indices: Sequence[int],
    val_indices: Sequence[int],
    selected_indices: Sequence[int],
    round_idx: int,
    output_path: str,
    random_state: int = 42,
) -> None:
    matrix, statuses = _prepare_coreset_embedding_matrix(
        embeddings, labeled_indices, unlabeled_indices, val_indices, selected_indices
    )
    if matrix is None or statuses is None:
        logger.debug("No embeddings available for t-SNE plot; skipping")
        return

    if matrix.shape[0] < 3:
        logger.warning("Not enough samples for t-SNE projection; skipping plot")
        return
    
    n_samples = matrix.shape[0]
    perplexity = max(5, min(50, n_samples // 100))
    reducer = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=random_state, metric='cosine', n_jobs=-1, early_exaggeration=4.0)
    coords = reducer.fit_transform(matrix)
    _plot_embedding_projection(coords, statuses, 't-SNE', round_idx, output_path)


def visualize_attention_maps(
    trainer,
    dataset,
    sample_cache: Dict[str, Any],
    classnames: List[str],
    epoch: int,
    maps_dir: str,
) -> None:
    images = sample_cache.get('images') if sample_cache else None
    labels = sample_cache.get('labels') if sample_cache else None
    paths = sample_cache.get('paths', []) if sample_cache else []

    if trainer is None or images is None:
        logger.debug("visualize_attention_maps: missing trainer/images")
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
        logger.debug("visualize_attention_maps: no attn_maps")
        return

    attn_map_to_vis = attn_maps[0]
    try:
        shape_info = getattr(attn_map_to_vis, 'shape', None)
        logger.debug(f"Epoch {epoch} attention map shape: {shape_info}")
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
            logger.warning(f"Unable to index attention map for image {i}, label {label}")
            continue

        if weights.dim() > 1:
            mean_weights = weights.mean(dim=0).detach().cpu().numpy()
        else:
            mean_weights = weights.detach().cpu().numpy()

        patch_weights = mean_weights[1:]
        num_patches = patch_weights.shape[0]
        h = w = int(np.sqrt(num_patches))
        if h * w != num_patches:
            logger.warning(f"Cannot reshape {num_patches} patches into square grid for image {i}")
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

    logger.debug(f"Saved {len(vis_images)} attention visualizations")


def visualize_gradcam_maps(
    trainer,
    dataset,
    sample_cache: Dict[str, Optional[torch.Tensor]],
    classnames: List[str],
    epoch: int,
    maps_dir: str,
) -> None:
    images = sample_cache.get('images') if sample_cache else None
    labels = sample_cache.get('labels') if sample_cache else None
    paths = sample_cache.get('paths', []) if sample_cache else []
    if not isinstance(paths, list):
        paths = []

    if trainer is None or images is None or labels is None:
        logger.debug("visualize_gradcam_maps: missing trainer/images/labels")
        return

    if isinstance(images, torch.Tensor):
        vis_images = images.to(trainer.device)
    elif isinstance(images, (list, tuple)):
        image_list: List[Any] = list(images)
        vis_images = torch.stack([
            x.to(trainer.device) if isinstance(x, torch.Tensor) else torch.tensor(x).to(trainer.device)
            for x in image_list
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

    logger.debug(f"Saved {len(gradcams)} GradCAM visualizations")


def run_dataset_eda(dataset, eda_dir: str, sample_limit: int = 512, seed: int = 42) -> None:
    if dataset is None or not hasattr(dataset, 'samples'):
        logger.debug("run_dataset_eda: invalid dataset")
        return

    os.makedirs(eda_dir, exist_ok=True)
    class_counts = Counter(label for _, label in dataset.samples)
    if not class_counts:
        logger.debug("run_dataset_eda: no class counts")
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

    logger.debug(f"Dataset EDA artifacts saved to {eda_dir}")


def log_experiment_start(method_name: str, dataset_name: str, kshot: int, seed: int) -> None:
    logger.info(f"{'='*60}")
    logger.info(f"Method:   {method_name}")
    logger.info(f"Dataset:  {dataset_name}")
    logger.info(f"K-shot:   {kshot}")
    logger.info(f"Seed:     {seed}")
    logger.info(f"{'='*60}")


def log_experiment_accuracy(accuracy: float) -> None:
    logger.info(f"{'='*60}")
    logger.info(f"Final Accuracy: {accuracy:.2f}%")
    logger.info(f"{'='*60}")