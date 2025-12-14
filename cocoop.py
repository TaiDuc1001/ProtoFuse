import os
import sys
import time
import math
import copy
import json
import torch
import random
import hashlib
import argparse
import datetime
import numpy as np
from collections import OrderedDict
from clip import clip
import torch.nn as nn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import transforms
from collections import defaultdict
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from typing import Any, Dict, List, Optional
from utils import (
    logger,
    setup_logging,
    run_dataset_eda,
    save_class_distribution_plot,
    save_confusion_artifacts,
)

_tokenizer = _Tokenizer()

ARG_SCHEMA = {
    'config': {'type': str, 'required': True, 'help': 'Path to YAML configuration file'},
    'output_dir': {'type': str, 'help': 'Override logging.output_dir from config', 'config_path': 'logging.output_dir'},
    'device': {'type': str, 'help': 'Override training.device from config', 'config_path': 'training.device'},
    'debug': {'type': bool, 'help': 'Enable debug output', 'default': False},
}


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEFAULT_TRAINING_EPOCHS = 100
DEFAULT_CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_INDEX_FILE = 'checkpoint_index.csv'


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


def set_nested_value(config, keys, value):
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


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


def merge_configs(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_configs(base[key], value)
        else:
            base[key] = value
    return base


def load_config_file(path):
    import yaml
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


class CheckpointCache:
    def __init__(self, cache_dir: str = DEFAULT_CHECKPOINT_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.index_path = os.path.join(cache_dir, CHECKPOINT_INDEX_FILE)

    def _get_key_settings(self, config) -> dict:
        if hasattr(config, 'to_dict'):
            config = config.to_dict()
        return {
            'method': 'cocoop',
            'dataset_root': config.get('data', {}).get('root'),
            'kshot': config.get('data', {}).get('kshot'),
            'seed': config.get('data', {}).get('seed'),
            'epochs': config.get('training', {}).get('epochs'),
            'backbone': config.get('model', {}).get('backbone'),
            'n_ctx': config.get('model', {}).get('n_ctx'),
        }

    def compute_checkpoint_id(self, config) -> str:
        key_settings = self._get_key_settings(config)
        key_str = json.dumps(key_settings, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get_checkpoint_path(self, checkpoint_id: str) -> str:
        return os.path.join(self.cache_dir, f"cocoop_ckpt_{checkpoint_id}.pt")

    def exists(self, checkpoint_id: str) -> bool:
        return os.path.exists(self.get_checkpoint_path(checkpoint_id))

    def _update_index(self, checkpoint_id: str, key_settings: dict, path: str):
        import csv
        rows = []
        fieldnames = ['checkpoint_id', 'file', 'method', 'dataset_root', 'kshot', 'seed',
                      'epochs', 'backbone', 'n_ctx', 'created_at']
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader if row.get('checkpoint_id') != checkpoint_id]
        row = {
            'checkpoint_id': checkpoint_id,
            'file': os.path.basename(path),
            'method': key_settings.get('method'),
            'dataset_root': key_settings.get('dataset_root'),
            'kshot': key_settings.get('kshot'),
            'seed': key_settings.get('seed'),
            'epochs': key_settings.get('epochs'),
            'backbone': key_settings.get('backbone'),
            'n_ctx': key_settings.get('n_ctx'),
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


class TextEncoder(nn.Module):
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


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        if not isinstance(cfg, ConfigNode):
            cfg = ConfigNode(cfg)
        model_cfg = cfg.get('model', ConfigNode())
        
        n_cls = len(classnames)
        n_ctx = coerce_to_int(model_cfg.get('n_ctx', 4), 4)
        ctx_init = coerce_to_str(model_cfg.get('ctx_init', ''), '')
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts


class CoCoOPCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        return logits


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


class CoCoOP:
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
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')
        logger.info(f"Loading CLIP (backbone: {backbone_name})")
        
        clip_model = load_clip_to_cpu(backbone_name)
        
        if self._cfg_str('fp32', 'training.precision', 'precision') in ['fp32', 'amp']:
            clip_model.float()

        self.model = CoCoOPCLIP(self.cfg, self.classnames, clip_model)

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        def format_params(num):
            if num >= 1e9:
                return f"{num/1e9:.2f}B"
            elif num >= 1e6:
                return f"{num/1e6:.2f}M"
            elif num >= 1e3:
                return f"{num/1e3:.2f}K"
            else:
                return str(num)
        
        n_ctx = self._cfg_int(4, 'model.n_ctx')
        logger.info(f"CoCoOP: n_ctx={n_ctx}, meta_net enabled")
        logger.info(f"Learnable parameters: {format_params(trainable_params)} / Total: {format_params(total_params)}")
        
        self.model.to(self.device)
        self.initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def setup_optimizer(self):
        lr = self._cfg_float(0.002, 'training.learning_rate')
        weight_decay = self._cfg_float(0.0005, 'training.weight_decay')
        optimizer_type = self._cfg_str('SGD', 'training.optimizer')
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        
        num_epochs = self._cfg_int(DEFAULT_TRAINING_EPOCHS, 'training.epochs')
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
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
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
        
        accuracy = 100 * correct / total
        avg_loss = running_loss / max(1, steps)
        return {"accuracy": accuracy, "loss": avg_loss, "predictions": all_preds, "true_labels": all_labels_list}
    
    def save_model(self, path):
        checkpoint = {
            'prompt_learner_state_dict': self.model.prompt_learner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'cfg': self.cfg
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
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
        logger.info(f"Model loaded from {path}")


class CoCoOPTrainingPipeline:
    def __init__(self, config):
        if not isinstance(config, ConfigNode):
            config = ConfigNode(config)
        self.config = config
        self.model_cfg = self.config.get('model', ConfigNode())
        self.training_cfg = self.config.get('training', ConfigNode())
        self.data_cfg = self.config.get('data', ConfigNode())
        self.logging_cfg = self.config.get('logging', ConfigNode())

        device_value = self.training_cfg.get("device", None)
        device_name = coerce_to_str(device_value, "cuda:0", key="training.device")
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")

        batch_value = self.training_cfg.get("batch_size", None)
        self.batch_size = coerce_to_int(batch_value, 32, key="training.batch_size")

        workers_value = self.data_cfg.get("num_workers", None)
        self.num_workers = coerce_to_int(workers_value, 4, key="data.num_workers")

        val_value = self.data_cfg.get("val_size", None)
        self.val_fraction = coerce_to_float(val_value, 0.2, key="data.val_size")
        if self.val_fraction > 1.0:
            self.val_fraction = self.val_fraction / 100.0
        if self.val_fraction < 0 or self.val_fraction >= 1.0:
            raise ValueError("data.val_size must be in [0, 1) or 0-100 range when expressed as percentage.")

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

        base_output_value = self.logging_cfg.get("output_dir", "outputs/cocoop")
        base_output = coerce_to_str(base_output_value, "outputs/cocoop", key="logging.output_dir")
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

        self.trainer: Optional[CoCoOP] = None
        self.trainer_cfg: ConfigNode = ConfigNode({})

        self.checkpoint_cache: Optional[CheckpointCache] = None
        self.checkpoint_id: Optional[str] = None
        self._init_checkpoint_cache()

    def _get_training_epochs(self):
        epochs_value = None
        if isinstance(self.training_cfg, dict):
            epochs_value = self.training_cfg.get('epochs', None)
        return coerce_to_int(epochs_value, DEFAULT_TRAINING_EPOCHS, key='training.epochs')

    def _init_checkpoint_cache(self):
        checkpoint_cfg = self.config.get('checkpoint', ConfigNode())
        if bool(checkpoint_cfg.get('enabled', False)):
            cache_dir = checkpoint_cfg.get('cache_dir', DEFAULT_CHECKPOINT_DIR)
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
        
        logger.section("CoCoOP Training", "train")
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

    def _load_dataset(self):
        transform = self._build_transforms()
        try:
            self.dataset = ImageFolder(self.dataset_root, transform=transform)
        except Exception as exc:
            raise RuntimeError(f"Failed to load dataset from {self.dataset_root}: {exc}")
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

        self.val_indices = val_indices
        self.train_indices = train_indices
        self.labeled_indices = list(train_indices)
        self.unlabeled_indices = unlabeled_indices

        if len(self.val_indices) > 0:
            val_ds = Subset(self.dataset, self.val_indices)
            self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        else:
            logger.warning("Validation split is empty; skipping validation metrics")

        self.classnames = list(self.dataset.classes)

        stats = {
            'total_images': len(self.dataset),
            'val_count': len(self.val_indices),
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

    def _initialize_trainer(self):
        if not self.classnames:
            raise RuntimeError("Class names unavailable before trainer initialization.")
        self.trainer = CoCoOP(self.trainer_cfg, self.classnames, device=str(self.device))

    def _train_epochs(self):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before training.")
        if not self.train_indices:
            raise RuntimeError("No training samples available.")

        if self._try_load_checkpoint():
            logger.info("Skipping training (loaded from checkpoint)")
            return

        cocoop_dir = os.path.join(self.run_dir, 'cocoop')
        os.makedirs(cocoop_dir, exist_ok=True)
        train_subset = Subset(self.dataset, list(self.train_indices))
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        epochs_total = self._get_training_epochs()

        for epoch_idx in range(1, epochs_total + 1):
            self._run_epoch(epoch_idx, epochs_total, train_loader, cocoop_dir)

        self._save_checkpoint()

    def _run_epoch(self, epoch_idx, epochs_total, train_loader, cocoop_dir):
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

        epoch_dir = os.path.join(cocoop_dir, f'epoch_{epoch_idx:03d}')
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
        epoch_result = {
            'epoch': epoch_idx,
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': epoch_time
        }
        with open(os.path.join(epoch_dir, 'result.json'), 'w') as f:
            json.dump(epoch_result, f, indent=2)

        self.metrics.append(epoch_result)

        if self.val_loader is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.trainer.save_model(self.best_model_path)

        val_acc_display = f"{val_acc:.2f}%" if self.val_loader is not None else "N/A"
        logger.info(f"CoCoOP Epoch {epoch_idx} - loss={avg_loss:.4f} - acc={avg_acc:.2f}% - val_acc={val_acc_display} - {epoch_time:.2f}s")

        if self.trainer.scheduler is not None:
            self.trainer.scheduler.step()

    def _finalize(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before finalization.")
        
        with open(self.config_path, 'w') as f:
            json.dump(self.trainer_cfg.to_dict(), f, indent=4)

        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        self.trainer.save_model(self.last_model_path)

        logger.info(f"Training completed. Results written to {self.run_dir}")


def parse_args():
    parser = create_argument_parser("Train CoCoOP model", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True))
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = CoCoOPTrainingPipeline(merged)
    pipeline.run()


if __name__ == "__main__":
    main()
