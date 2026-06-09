import os

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg", force=True)

import sys
import cv2
import csv
import copy
import json
import math
import time
import yaml
import umap
import torch
import random
import hashlib
import datetime
import argparse
import numpy as np
import torch.nn as nn
from clip import clip
from PIL import Image
import seaborn as sns
import torch.nn.functional as F
import multiprocessing as mp
from torchvision import transforms
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from typing import Any, Dict, List, Optional, Sequence
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from logger import logger, setup_logging


class CheckpointCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.index_path = os.path.join(cache_dir, 'index.csv')

    def _get_key_settings(self, config) -> dict:
        if hasattr(config, 'to_dict'):
            config = config.to_dict()
        model_cfg = config.get('model', {})
        training_cfg = config.get('training', {})
        method_params = {k: v for k, v in model_cfg.items() 
                         if k not in ('backbone',)}
        training_params = {
            'learning_rate': training_cfg.get('learning_rate'),
            'weight_decay': training_cfg.get('weight_decay'),
            'optimizer': training_cfg.get('optimizer'),
        }
        return {
            'dataset_root': config.get('data', {}).get('root'),
            'kshot': config.get('data', {}).get('kshot'),
            'seed': config.get('data', {}).get('seed'),
            'epochs': training_cfg.get('epochs'),
            'batch_size': training_cfg.get('batch_size'),
            'backbone': model_cfg.get('backbone'),
            'method_params': json.dumps(method_params, sort_keys=True),
            'training_params': json.dumps(training_params, sort_keys=True),
            'base_novel': json.dumps(config.get('data', {}).get('base_novel', {}), sort_keys=True),
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
                      'epochs', 'batch_size', 'backbone', 'method_params', 'training_params', 'created_at']
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
            'batch_size': key_settings.get('batch_size'),
            'backbone': key_settings.get('backbone'),
            'method_params': key_settings.get('method_params'),
            'training_params': key_settings.get('training_params'),
            'created_at': datetime.datetime.now().isoformat(),
        }
        rows.append(row)
        with open(self.index_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save(self, checkpoint_id, model_state, optimizer_state, scheduler_state,
             labeled_indices, unlabeled_indices, metrics, config, apt_predictions=None):
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
        if apt_predictions is not None:
            checkpoint['apt_predictions'] = apt_predictions
        path = self.get_checkpoint_path(checkpoint_id)
        torch.save(checkpoint, path)
        self._update_index(checkpoint_id, key_settings, path)
        return path

    def load(self, checkpoint_id):
        path = self.get_checkpoint_path(checkpoint_id)
        if not os.path.exists(path):
            return None
        return torch.load(path, map_location='cpu')


class CLIPFeatureCache:
    VERSION = 1
    DEFAULT_CACHE_DIR = "checkpoints/clip_features"

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        os.makedirs(self.cache_dir, exist_ok=True)

    def _dataset_signature(self, dataset, dataset_id, classnames, template, backbone,
                           precision, clip_mean, clip_std, transform_spec):
        paths = [path for path, _ in dataset.samples]
        labels = [int(label) for _, label in dataset.samples]
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        return {
            "version": self.VERSION,
            "dataset_id": dataset_id,
            "dataset_root": getattr(dataset, "root", ""),
            "backbone": backbone,
            "precision": precision,
            "classnames": list(classnames),
            "template": template,
            "prompts": prompts,
            "paths": paths,
            "labels": labels,
            "clip_mean": list(clip_mean),
            "clip_std": list(clip_std),
            "transform_spec": transform_spec,
        }

    def compute_cache_key(self, dataset, dataset_id, classnames, template, backbone,
                          precision, clip_mean, clip_std, transform_spec):
        signature = self._dataset_signature(
            dataset, dataset_id, classnames, template, backbone,
            precision, clip_mean, clip_std, transform_spec
        )
        key_str = json.dumps(signature, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:16], signature

    def _cache_path(self, cache_key):
        return os.path.join(self.cache_dir, f"{cache_key}.pt")

    def load_or_compute(self, dataset, dataset_id, clip_model, classnames, template,
                        backbone, precision, batch_size, num_workers, device,
                        clip_mean, clip_std, transform_spec):
        cache_key, signature = self.compute_cache_key(
            dataset, dataset_id, classnames, template, backbone,
            precision, clip_mean, clip_std, transform_spec
        )
        path = self._cache_path(cache_key)
        if self.enabled and os.path.exists(path):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded CLIP feature cache ({dataset_id}) from {path}")
            return payload

        if self.enabled:
            logger.info(f"Computing CLIP feature cache ({dataset_id}); cache miss at {path}")
        else:
            logger.info(f"Computing CLIP feature cache ({dataset_id}); cache disabled")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        image_features = []
        labels = []
        clip_model.eval()
        with torch.no_grad():
            for images, batch_labels in loader:
                images = images.to(device)
                feats = clip_model.encode_image(images).float()
                feats = F.normalize(feats, dim=-1)
                image_features.append(feats.cpu())
                labels.append(batch_labels.cpu().long())

            prompts = signature["prompts"]
            tokens = clip.tokenize(prompts).to(device)
            text_features = clip_model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1).cpu()

        payload = {
            "image_features": torch.cat(image_features, dim=0).float(),
            "labels": torch.cat(labels, dim=0).long(),
            "paths": signature["paths"],
            "text_features": text_features.float(),
            "prompts": signature["prompts"],
            "classnames": signature["classnames"],
            "metadata": signature,
            "cache_key": cache_key,
        }

        if self.enabled:
            tmp_path = f"{path}.tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            logger.info(f"Saved CLIP feature cache ({dataset_id}) to {path}")
        return payload


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
            kwargs['default'] = spec.get('default')
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



def compute_metrics(true_labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, float]:
    if not true_labels or not predictions:
        return {}

    y_true = np.array(true_labels)
    y_pred = np.array(predictions)

    metrics = {}

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

        metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
        metrics['mca'] = balanced_accuracy_score(y_true, y_pred) * 100
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    return metrics


def log_experiment_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics") -> None:
    _metric_order = [
        ("accuracy",           "Acc",    "{:.2f}%"),
        ("mca",                "MCA",    "{:.2f}%"),
        ("f1_macro",           "F1-Mac", "{:.4f}"),
        ("f1_micro",           "F1-Mic", "{:.4f}"),
        ("f1_weighted",        "F1-Wei", "{:.4f}"),
        ("precision_macro",    "P-Mac",  "{:.4f}"),
        ("precision_micro",    "P-Mic",  "{:.4f}"),
        ("precision_weighted", "P-Wei",  "{:.4f}"),
        ("recall_macro",       "R-Mac",  "{:.4f}"),
        ("recall_micro",       "R-Mic",  "{:.4f}"),
        ("recall_weighted",    "R-Wei",  "{:.4f}"),
    ]
    row = {}
    columns = []
    for key, label, fmt in _metric_order:
        if key in metrics and metrics[key] is not None:
            row[label] = fmt.format(float(metrics[key]))
            columns.append(label)
    if row:
        logger.comparison_table(rows=[row], columns=columns, title=title)


def log_experiment_accuracy(accuracy: float) -> None:
    log_experiment_metrics({'accuracy': accuracy})


DEFAULT_ARG_SCHEMA = {
    'config': {'type': str, 'required': True, 'help': 'Path to YAML configuration file'},
    'output_dir': {'type': str, 'help': 'Override logging.output_dir from config', 'config_path': 'logging.output_dir'},
    'debug': {'type': bool, 'help': 'Enable debug logging', 'default': True},
    'disable_coloring': {'type': bool, 'help': 'Disable colored output for log files', 'default': True},
}


def format_params(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


class PromptTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class BaseTrainer:
    DEFAULT_LR = 0.002
    DEFAULT_TRAINING_EPOCHS = 100

    def __init__(self, cfg, classnames, device="cuda"):
        if not isinstance(cfg, ConfigNode):
            cfg = ConfigNode(cfg)
        self.cfg = cfg
        self.training_cfg = self.cfg.get('training', ConfigNode())
        self.model_cfg = self.cfg.get('model', ConfigNode())
        self.data_cfg = self.cfg.get('data', ConfigNode())
        self.classnames = classnames
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.build_model()
        self.setup_optimizer()
        precision_mode = self._cfg_str('fp32', 'training.precision', 'precision')
        self.scaler = GradScaler() if precision_mode == 'amp' else None

    def _cfg_value(self, *paths, default=None):
        sentinel = object()
        for path in paths:
            value = get_config_value(self.cfg, path, sentinel)
            if value is not sentinel:
                return value
        return default

    def _cfg_float(self, default, *paths):
        value = self._cfg_value(*paths, default=default)
        return coerce_to_float(value, default)

    def _cfg_int(self, default, *paths):
        value = self._cfg_value(*paths, default=default)
        return coerce_to_int(value, default)

    def _cfg_str(self, default, *paths):
        value = self._cfg_value(*paths, default=default)
        return coerce_to_str(value, default)

    def build_model(self):
        raise NotImplementedError

    def setup_optimizer(self):
        lr = self._cfg_float(self.DEFAULT_LR, 'training.learning_rate')
        weight_decay = self._cfg_float(0.0005, 'training.weight_decay')
        optimizer_type = self._cfg_str('SGD', 'training.optimizer')
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9)

        num_epochs = self._cfg_int(self.DEFAULT_TRAINING_EPOCHS, 'training.epochs')
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

    def reset_optimizer_scheduler(self):
        self.setup_optimizer()

    def reset_model(self):
        if hasattr(self, 'initial_model_state'):
            self.model.load_state_dict(self.initial_model_state)
            self.model.to(self.device)

    def train_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.model.train()

        precision = self._cfg_str('fp32', 'training.precision', 'precision')

        if precision == 'amp':
            with autocast():
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
        else:
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        return {"loss": loss.item(), "accuracy": accuracy}

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        steps = 0
        all_preds = []
        all_labels_list = []

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)
                running_loss += loss.item()
                steps += 1

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        metrics = compute_metrics(all_labels_list, all_preds)
        avg_loss = running_loss / max(1, steps)
        metrics['loss'] = avg_loss
        metrics['predictions'] = all_preds
        metrics['true_labels'] = all_labels_list
        return metrics

    def save_model(self, path):
        checkpoint = {
            'prompt_learner_state_dict': self.model.prompt_learner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'cfg': self.cfg
        }
        torch.save(checkpoint, path)
        # logger.info(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['prompt_learner_state_dict']
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]
        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]
        self.model.prompt_learner.load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # logger.info(f"Model loaded from {path}")


class BaseTrainingPipeline:
    METHOD_NAME = "Base"
    DEFAULT_OUTPUT_DIR = "outputs/base"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/base"
    TRAINER_CLASS = None
    SAVE_BEST_LAST = True
    _EXTRA_PIPELINE_CLASSES = []

    @classmethod
    def _all_subclasses(cls):
        result = []
        for sub in cls.__subclasses__():
            result.append(sub)
            result.extend(sub._all_subclasses())
        return result

    @classmethod
    def register_extra_pipeline(cls, pipeline_cls):
        if pipeline_cls not in cls._EXTRA_PIPELINE_CLASSES:
            cls._EXTRA_PIPELINE_CLASSES.append(pipeline_cls)

    @classmethod
    def get_pipeline_by_name(cls, name):
        name_lower = name.lower()
        for sub in cls._all_subclasses():
            if getattr(sub, 'METHOD_NAME', '').lower() == name_lower:
                return sub
        for sub in cls._EXTRA_PIPELINE_CLASSES:
            if getattr(sub, 'METHOD_NAME', '').lower() == name_lower:
                return sub
            for child in sub.__subclasses__():
                if getattr(child, 'METHOD_NAME', '').lower() == name_lower:
                    return child
        available = [getattr(s, 'METHOD_NAME', '?') for s in cls._all_subclasses() + cls._EXTRA_PIPELINE_CLASSES]
        raise ValueError(f"No pipeline with METHOD_NAME='{name}'. Available: {available}")

    def __init__(self, config):
        if not isinstance(config, ConfigNode):
            config = ConfigNode(config)
        self.config = config
        self.model_cfg = self.config.get('model', ConfigNode())
        self.training_cfg = self.config.get('training', ConfigNode())
        self.data_cfg = self.config.get('data', ConfigNode())
        self.logging_cfg = self.config.get('logging', ConfigNode())
        self.feature_cache_cfg = self.config.get('feature_cache', ConfigNode())
        save_best_last_value = get_config_value(self.training_cfg, "save_best_last", self.SAVE_BEST_LAST)
        self.save_best_last = bool(self.SAVE_BEST_LAST if save_best_last_value is None else save_best_last_value)

        device_value = self.training_cfg.get("device", None)
        device_name = coerce_to_str(device_value, "cuda:0", key="training.device")
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")

        batch_value = self.training_cfg.get("batch_size", None)
        self.batch_size = coerce_to_int(batch_value, 32, key="training.batch_size")

        workers_value = self.data_cfg.get("num_workers", None)
        self.num_workers = coerce_to_int(workers_value, 4, key="data.num_workers")

        val_value = self.data_cfg.get("val_size", None)
        if val_value is not None:
            self.val_fraction = coerce_to_float(val_value, 0.7, key="data.val_size")
            if self.val_fraction > 1.0:
                self.val_fraction = self.val_fraction / 100.0
            if self.val_fraction < 0 or self.val_fraction >= 1.0:
                raise ValueError("data.val_size must be in [0, 1) or 0-100 range when expressed as percentage.")
        else:
            self.val_fraction = None

        dataset_root_value = self.data_cfg.get("root", "./datasets/cub-200-2011-renamed")
        self.dataset_root = coerce_to_str(dataset_root_value, "./datasets/cub-200-2011-renamed", key="data.root")

        seed_value = self.data_cfg.get("seed", None)
        self.seed = coerce_to_int(seed_value, 42, key="data.seed")

        kshot_value = self.data_cfg.get("kshot", None)
        self.kshot = coerce_to_int(kshot_value, -1, key="data.kshot")

        run_eda_value = get_config_value(self.data_cfg, "run_eda", False)
        self.run_eda = bool(False if run_eda_value is None else run_eda_value)

        class_dist_value = get_config_value(self.training_cfg, "class_distribution", False)
        self.class_distribution_enabled = bool(False if class_dist_value is None else class_dist_value)
        interval_value = get_config_value(self.training_cfg, "log_interval", None)
        if interval_value is None:
            interval_value = get_config_value(self.logging_cfg, "epoch_log_interval", None)
        self.epoch_log_interval = max(1, coerce_to_int(interval_value, 10, key="training.log_interval"))

        base_output_value = self.logging_cfg.get("output_dir", self.DEFAULT_OUTPUT_DIR)
        base_output = coerce_to_str(base_output_value, self.DEFAULT_OUTPUT_DIR, key="logging.output_dir")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(base_output, timestamp)
        logger.info(f"Run directory: {self.run_dir}")
        self.config_path = os.path.join(self.run_dir, 'config.json')
        self.metrics_path = os.path.join(self.run_dir, 'metrics.json')
        self.best_model_path = os.path.join(self.run_dir, 'best.pt')
        self.last_model_path = os.path.join(self.run_dir, 'last.pt')
        self.eda_dir = os.path.join(self.run_dir, 'eda')

        self.clip_mean = get_config_value(self.data_cfg, "clip_mean", [0.48145466, 0.4578275, 0.40821073])
        self.clip_std = get_config_value(self.data_cfg, "clip_std", [0.26862954, 0.26130258, 0.27577711])

        self.dataset: Optional[ImageFolder] = None
        self.val_loader: Optional[DataLoader] = None
        self.classnames: List[str] = []
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.labeled_indices: List[int] = []
        self.unlabeled_indices: List[int] = []
        self.metrics: List[Dict[str, Any]] = []
        self.best_val_acc = -float('inf')
        self.global_epoch = 0

        self.trainer = None
        self.trainer_cfg: ConfigNode = ConfigNode({})

        self.checkpoint_cache: Optional[CheckpointCache] = None
        self.checkpoint_id: Optional[str] = None
        feature_cache_enabled = bool(self.feature_cache_cfg.get('enabled', True))
        feature_cache_dir = self.feature_cache_cfg.get('cache_dir', CLIPFeatureCache.DEFAULT_CACHE_DIR)
        self.clip_feature_cache = CLIPFeatureCache(feature_cache_dir, enabled=feature_cache_enabled)
        self._init_checkpoint_cache()

        base_novel_cfg = self.data_cfg.get('base_novel', ConfigNode())
        self.base_novel_enabled = bool(base_novel_cfg.get('enabled', False))
        self.base_novel_split_ratio = coerce_to_float(base_novel_cfg.get('split_ratio', 0.5), 0.5)
        self.base_class_indices: List[int] = []
        self.novel_class_indices: List[int] = []

    def _get_training_epochs(self):
        epochs_value = None
        if isinstance(self.training_cfg, dict):
            epochs_value = self.training_cfg.get('epochs', None)
        return coerce_to_int(epochs_value, 100, key='training.epochs')

    def _should_log_epoch(self, epoch_idx: int, epochs_total: int) -> bool:
        return (epoch_idx % self.epoch_log_interval == 0) or (epoch_idx == epochs_total)

    def _init_checkpoint_cache(self):
        checkpoint_cfg = self.config.get('checkpoint', ConfigNode())
        if bool(checkpoint_cfg.get('enabled', False)):
            cache_dir = checkpoint_cfg.get('cache_dir', self.DEFAULT_CHECKPOINT_DIR)
            self.checkpoint_cache = CheckpointCache(cache_dir)
            self.checkpoint_id = self.checkpoint_cache.compute_checkpoint_id(self.config)
            logger.info(f"Checkpoint cache enabled. ID: {self.checkpoint_id}")

    def _try_load_checkpoint(self) -> bool:
        if self.checkpoint_cache is None or self.checkpoint_id is None:
            return False
        if not self.checkpoint_cache.exists(self.checkpoint_id):
            return False
        ckpt = self.checkpoint_cache.load(self.checkpoint_id)
        if ckpt is None:
            return False
        if self.trainer is None:
            return False

        model_state = ckpt['model_state_dict']
        if 'ctx' in model_state:
            self.trainer.model.prompt_learner.ctx.data = model_state['ctx']

        self.trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.labeled_indices = ckpt['labeled_indices']
        self.unlabeled_indices = ckpt['unlabeled_indices']
        self.metrics = ckpt['metrics']
        self.global_epoch = len(self.metrics)
        logger.info(f"Loaded checkpoint: {self.checkpoint_id} (epoch {self.global_epoch})")
        return True

    def _save_checkpoint(self):
        if self.checkpoint_cache is None or self.checkpoint_id is None:
            return
        if self.trainer is None:
            return
        model_state = {
            'ctx': self.trainer.model.prompt_learner.ctx.data,
        }
        path = self.checkpoint_cache.save(
            self.checkpoint_id,
            model_state,
            self.trainer.optimizer.state_dict(),
            self.trainer.scheduler.state_dict(),
            self.labeled_indices,
            self.unlabeled_indices,
            self.metrics,
            self.config
        )
        logger.debug(f"Saved checkpoint to: {path}")

    def run(self):
        set_global_seed(self.seed)

        logger.section("Initialization", "config")
        self._prepare_directories()
        self._load_dataset()
        self._split_dataset()
        self._initialize_trainer()

        dataset_name = self.config.data.dataset_name
        log_experiment_start(self.METHOD_NAME, dataset_name, self.kshot, self.seed)

        logger.section(f"{self.METHOD_NAME} Training", "train")
        self._train_epochs()

        logger.section("Finalization", "save")
        self._finalize()

    def _prepare_directories(self):
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.eda_dir, exist_ok=True)

    def _build_transforms(self):
        base_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.clip_mean, std=self.clip_std),
        ]
        return transforms.Compose(base_transforms)

    def _clip_feature_transform_spec(self):
        return {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": list(self.clip_mean),
            "normalize_std": list(self.clip_std),
        }

    def _apply_cached_text_features(self, payload):
        if self.trainer is None:
            return
        text_features = payload["text_features"].to(self.device).float()
        if hasattr(self.trainer, "text_features"):
            self.trainer.text_features = text_features
        if hasattr(self.trainer, "clip_weights"):
            self.trainer.clip_weights = text_features.t().contiguous()
        if hasattr(self.trainer, "text_prototypes"):
            self.trainer.text_prototypes = text_features

    def _load_clip_feature_payload(self, dataset, dataset_id):
        if self.trainer is None:
            raise RuntimeError("Trainer must be initialized before loading CLIP feature cache.")
        if dataset is None:
            raise RuntimeError(f"Cannot build CLIP feature cache for missing dataset: {dataset_id}")

        backbone = coerce_to_str(self.model_cfg.get("backbone", "ViT-B/16"), "ViT-B/16", key="model.backbone")
        precision = coerce_to_str(self.training_cfg.get("precision", "fp32"), "fp32", key="training.precision")
        template = getattr(self.trainer, "template", "a photo of a {}.")
        payload = self.clip_feature_cache.load_or_compute(
            dataset=dataset,
            dataset_id=dataset_id,
            clip_model=self.trainer.clip_model,
            classnames=self.classnames,
            template=template,
            backbone=backbone,
            precision=precision,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
            clip_mean=self.clip_mean,
            clip_std=self.clip_std,
            transform_spec=self._clip_feature_transform_spec(),
        )
        self._apply_cached_text_features(payload)
        return payload

    def _full_dataset_clip_features(self):
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before loading CLIP feature cache.")
        if not hasattr(self, '_clip_feature_payload') or self._clip_feature_payload is None:
            self._clip_feature_payload = self._load_clip_feature_payload(self.dataset, "train_or_full")
        return self._clip_feature_payload

    def _cached_train_features(self):
        payload = self._full_dataset_clip_features()
        indices = torch.tensor(self.train_indices, dtype=torch.long)
        return payload["image_features"][indices], payload["labels"][indices]

    def _cached_val_features(self):
        if self.val_fraction is not None:
            payload = self._full_dataset_clip_features()
            indices = torch.tensor(self.val_indices, dtype=torch.long)
            return payload["image_features"][indices], payload["labels"][indices]
        if not hasattr(self, '_clip_val_payload') or self._clip_val_payload is None:
            self._clip_val_payload = self._load_clip_feature_payload(self._val_dataset, "val_or_test")
        return self._clip_val_payload["image_features"], self._clip_val_payload["labels"]

    def _cached_test_features(self):
        if hasattr(self, '_val_dataset') and self._val_dataset is not None:
            if not hasattr(self, '_clip_val_payload') or self._clip_val_payload is None:
                self._clip_val_payload = self._load_clip_feature_payload(self._val_dataset, "val_or_test")
            return self._clip_val_payload["image_features"], self._clip_val_payload["labels"]
        return self._cached_val_features()

    def _load_dataset(self):
        transform = self._build_transforms()
        if self.val_fraction is not None:
            try:
                self.dataset = ImageFolder(self.dataset_root, transform=transform)
            except Exception as exc:
                raise RuntimeError(f"Failed to load dataset from {self.dataset_root}: {exc}")
            if self.run_eda:
                run_dataset_eda(self.dataset, self.eda_dir, sample_limit=512, seed=self.seed)
        else:
            train_path = os.path.join(self.dataset_root, 'train')
            test_path = os.path.join(self.dataset_root, 'test')
            try:
                self.dataset = ImageFolder(train_path, transform=transform)
            except Exception as exc:
                raise RuntimeError(f"Failed to load train dataset from {train_path}: {exc}")
            try:
                self._val_dataset = ImageFolder(test_path, transform=transform)
            except Exception as exc:
                raise RuntimeError(f"Failed to load test dataset from {test_path}: {exc}")
            if self.run_eda:
                run_dataset_eda(self.dataset, self.eda_dir, sample_limit=512, seed=self.seed)

    def _split_dataset(self):
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before splitting.")
        samples_by_class_idx = defaultdict(list)
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            samples_by_class_idx[class_idx].append(idx)

        rng = random.Random(self.seed)
        val_indices = []
        train_indices = []
        unlabeled_indices = []

        if self.val_fraction is not None:
            for class_idx in sorted(samples_by_class_idx.keys()):
                class_samples = list(samples_by_class_idx[class_idx])
                class_samples.sort()
                rng.shuffle(class_samples)

                val_count = int(math.floor(len(class_samples) * self.val_fraction))
                if self.val_fraction > 0 and val_count == 0 and len(class_samples) > 0:
                    val_count = 1

                val_part = class_samples[:val_count]
                train_candidates = class_samples[val_count:]
                if self.kshot > 0:
                    labeled_part = train_candidates[:self.kshot]
                    leftover_part = train_candidates[self.kshot:]
                else:
                    labeled_part = train_candidates
                    leftover_part = []

                val_indices.extend(val_part)
                train_indices.extend(labeled_part)
                unlabeled_indices.extend(leftover_part)
        else:
            for class_idx in sorted(samples_by_class_idx.keys()):
                class_samples = list(samples_by_class_idx[class_idx])
                class_samples.sort()
                rng.shuffle(class_samples)

                if self.kshot > 0:
                    labeled_part = class_samples[:self.kshot]
                    leftover_part = class_samples[self.kshot:]
                else:
                    labeled_part = class_samples
                    leftover_part = []
                train_indices.extend(labeled_part)
                unlabeled_indices.extend(leftover_part)

        self.val_indices = val_indices
        self.train_indices = train_indices
        self.labeled_indices = list(train_indices)
        self.unlabeled_indices = unlabeled_indices

        if self.val_fraction is not None:
            if len(self.val_indices) > 0:
                val_ds = Subset(self.dataset, self.val_indices)
                self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            else:
                logger.warning("Validation split is empty; skipping validation metrics")
        else:
            self.val_loader = DataLoader(self._val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        self.classnames = list(self.dataset.classes)

        if self.base_novel_enabled:
            num_classes = len(self.classnames)
            num_base = int(num_classes * self.base_novel_split_ratio)
            all_class_indices = list(range(num_classes))
            rng.shuffle(all_class_indices)
            self.base_class_indices = sorted(all_class_indices[:num_base])
            self.novel_class_indices = sorted(all_class_indices[num_base:])
            base_set = set(self.base_class_indices)
            self.train_indices = [i for i in self.train_indices if self.dataset.samples[i][1] in base_set]
            self.labeled_indices = list(self.train_indices)
            logger.info(f"Base-to-Novel: {len(self.base_class_indices)} base, {len(self.novel_class_indices)} novel classes")

        if self.val_fraction is not None:
            total_images = len(self.dataset)
            val_count = len(self.val_indices)
        else:
            total_images = len(self.dataset) + len(self._val_dataset)
            val_count = len(self._val_dataset)

        stats = {
            'total_images': total_images,
            'val_count': val_count,
            'train_count': len(self.train_indices),
            'labeled_count': len(self.train_indices),
            'unlabeled_count': len(self.unlabeled_indices),
            'train_pool_size': len(self.train_indices) + len(self.unlabeled_indices)
        }
        logger.info(f"Dataset loaded: {stats['total_images']} total images")
        val_percentage = (stats['val_count'] / stats['total_images'] * 100.0) if stats['total_images'] > 0 else 0.0
        logger.info(f"Validation: {stats['val_count']} ({val_percentage:.2f}%), Train: {stats['train_count']}, Unlabeled: {stats['unlabeled_count']}")

        trainer_cfg = self._build_trainer_config(stats, val_percentage)
        with open(self.config_path, 'w') as f:
            json.dump(trainer_cfg.to_dict(), f, indent=4)

    def _build_trainer_config(self, stats, val_percentage):
        extra_values = {
            'dataset_root': self.dataset_root,
            'val_size': self.val_fraction,
            'classnames': self.classnames,
            'num_classes': len(self.classnames),
            'train_size': stats.get('labeled_count', stats['train_count']),
            'val_size_count': stats['val_count'],
            'train_pool_size': stats.get('train_pool_size', stats['train_count'] + stats.get('unlabeled_count', 0)),
            'unlabeled_pool_size': stats.get('unlabeled_count', 0),
            'val_percentage_actual': val_percentage,
        }

        trainer_cfg = build_config_namespace(self.config, extra_values)
        self.trainer_cfg = trainer_cfg
        return trainer_cfg

    def _metrics_title(self, method_name=None):
        method = method_name or self.METHOD_NAME
        dataset = get_config_value(self.data_cfg, "dataset_name", "unknown-dataset")
        return f"{method} x {dataset}"

    def _initialize_trainer(self):
        if not self.classnames:
            raise RuntimeError("Class names unavailable before trainer initialization.")
        if self.TRAINER_CLASS is None:
            raise NotImplementedError("Subclass must set TRAINER_CLASS")
        self.trainer = self.TRAINER_CLASS(self.trainer_cfg, self.classnames, device=str(self.device))

    def _train_epochs(self):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before training.")
        if not self.train_indices:
            raise RuntimeError("No training samples available.")

        if self._try_load_checkpoint():
            logger.info("Skipping training (loaded from checkpoint)")
            return

        train_subset = Subset(self.dataset, list(self.train_indices))
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        epochs_total = self._get_training_epochs()

        for epoch_idx in range(1, epochs_total + 1):
            self._run_epoch(epoch_idx, epochs_total, train_loader, self.run_dir)

        self._save_checkpoint()

    def _run_epoch(self, epoch_idx, epochs_total, train_loader, run_dir):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before epoch run.")
        self.global_epoch += 1
        start_time = time.time()
        self.trainer.model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        steps = 0

        for batch in train_loader:
            loss_dict = self.trainer.train_step(batch)
            running_loss += loss_dict['loss']
            running_accuracy += loss_dict['accuracy']
            steps += 1

        avg_loss = running_loss / max(1, steps)
        avg_acc = running_accuracy / max(1, steps)

        if self.val_loader is not None:
            results = self.trainer.evaluate(self.val_loader)
            val_acc = results['accuracy']
            val_loss = results['loss']
            all_preds = results['predictions']
            all_labels = results['true_labels']
        else:
            val_acc = 0.0
            val_loss = 0.0
            all_preds = []
            all_labels = []

        epoch_dir = os.path.join(run_dir, f'epoch_{epoch_idx:03d}')
        os.makedirs(epoch_dir, exist_ok=True)

        if bool(get_config_value(self.training_cfg, 'confusion_matrix', False)) and all_labels:
            save_confusion_artifacts(all_labels, all_preds, self.global_epoch, epoch_dir)

        if self.class_distribution_enabled and all_labels:
            save_class_distribution_plot(
                all_labels,
                all_preds,
                self.global_epoch,
                epoch_dir,
                self.classnames,
            )

        epoch_time = time.time() - start_time

        base_val_acc = None
        novel_val_acc = None
        harmonic_mean = None
        if self.base_novel_enabled and all_labels:
            base_set = set(self.base_class_indices)
            novel_set = set(self.novel_class_indices)
            base_correct = sum(1 for p, l in zip(all_preds, all_labels) if l in base_set and p == l)
            base_total = sum(1 for l in all_labels if l in base_set)
            novel_correct = sum(1 for p, l in zip(all_preds, all_labels) if l in novel_set and p == l)
            novel_total = sum(1 for l in all_labels if l in novel_set)
            if base_total > 0:
                base_val_acc = 100 * base_correct / base_total
            if novel_total > 0:
                novel_val_acc = 100 * novel_correct / novel_total
            if base_val_acc is not None and novel_val_acc is not None and (base_val_acc + novel_val_acc) > 0:
                harmonic_mean = 2 * base_val_acc * novel_val_acc / (base_val_acc + novel_val_acc)

        epoch_result = {
            'epoch': epoch_idx,
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'base_val_acc': base_val_acc,
            'novel_val_acc': novel_val_acc,
            'harmonic_mean': harmonic_mean,
            'time': epoch_time,
            'accuracy': results.get('accuracy', val_acc) if self.val_loader else 0.0,
            'mca': results.get('mca', 0.0) if self.val_loader else 0.0,
            'f1_macro': results.get('f1_macro', 0.0) if self.val_loader else 0.0,
            'f1_micro': results.get('f1_micro', 0.0) if self.val_loader else 0.0,
            'f1_weighted': results.get('f1_weighted', 0.0) if self.val_loader else 0.0,
            'precision_macro': results.get('precision_macro', 0.0) if self.val_loader else 0.0,
            'precision_micro': results.get('precision_micro', 0.0) if self.val_loader else 0.0,
            'precision_weighted': results.get('precision_weighted', 0.0) if self.val_loader else 0.0,
            'recall_macro': results.get('recall_macro', 0.0) if self.val_loader else 0.0,
            'recall_micro': results.get('recall_micro', 0.0) if self.val_loader else 0.0,
            'recall_weighted': results.get('recall_weighted', 0.0) if self.val_loader else 0.0,
        }
        with open(os.path.join(epoch_dir, 'result.json'), 'w') as f:
            json.dump(epoch_result, f, indent=2)

        self.metrics.append(epoch_result)

        if self.save_best_last and self.val_loader is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.trainer.save_model(self.best_model_path)

        if self._should_log_epoch(epoch_idx, epochs_total):
            if self.base_novel_enabled and base_val_acc is not None:
                logger.info(f"{self.METHOD_NAME} Epoch {epoch_idx} - loss={avg_loss:.4f} - acc={avg_acc:.2f}% - val_acc={val_acc:.2f}% - base={base_val_acc:.2f}% - novel={novel_val_acc:.2f}% - H={harmonic_mean:.2f}% - {epoch_time:.2f}s")
            else:
                val_acc_display = f"{val_acc:.2f}%" if self.val_loader is not None else "N/A"
                logger.info(f"{self.METHOD_NAME} Epoch {epoch_idx} - loss={avg_loss:.4f} - acc={avg_acc:.2f}% - val_acc={val_acc_display} - {epoch_time:.2f}s")

        if self.trainer.scheduler is not None:
            self.trainer.scheduler.step()

    def _finalize(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before finalization.")

        with open(self.config_path, 'w') as f:
            json.dump(self.trainer_cfg.to_dict(), f, indent=4)

        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        if self.save_best_last:
            self.trainer.save_model(self.last_model_path)

        logger.info(f"Training completed. Results written to {self.run_dir}")

        final_metrics = self.metrics[-1] if self.metrics else {}
        final_method = final_metrics.get('method') if isinstance(final_metrics, dict) else None
        log_experiment_metrics(final_metrics, title=self._metrics_title(final_method))
