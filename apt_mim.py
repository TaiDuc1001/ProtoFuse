import os
import time
import math
import copy
import json
import torch
import random
import argparse
import datetime
import numpy as np
from clip import clip
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from typing import Any, Dict, List, Optional


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "CUBirds": "a photo of a {}, a type of bird.",
    "V1922_13": "a photo of a {}, a type of military vehicle."
}


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


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        visual = clip_model.visual
        self.conv1 = visual.conv1
        self.class_embedding = visual.class_embedding
        self.positional_embedding = visual.positional_embedding
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer
        self.ln_post = visual.ln_post
        self.proj = visual.proj

    def forward(self, x):
        x = x.type(self.conv1.weight.dtype)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls_tokens = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        for block in self.transformer.resblocks:
            x = block(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        cls_feature = x[:, 0, :]
        patch_features = x[:, 1:, :]
        return patch_features, cls_feature


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.text_projection = clip_model.text_projection

    def encode_text_tokens(self, text):
        x = self.clip_model.token_embedding(text).type(self.dtype)
        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.dtype)
        return x

    def cls_from_tokens(self, tokens, text):
        eos = text.argmax(dim=-1)
        cls = tokens[torch.arange(tokens.shape[0]), eos]
        return cls @ self.text_projection

    def encode_text(self, text):
        tokens = self.encode_text_tokens(text)
        return self.cls_from_tokens(tokens, text)


class MIMModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_encoder_layers, num_decoder_layers, num_heads, dropout):
        super().__init__()
        self.feature_dim = feature_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, patch_tokens, mask=None):
        B, N, D = patch_tokens.shape
        if mask is not None:
            mask_tokens = self.mask_token.expand(B, N, -1)
            encoder_input = torch.where(mask.unsqueeze(-1), mask_tokens, patch_tokens)
        else:
            encoder_input = patch_tokens
        memory = self.encoder(encoder_input)
        decoder_input = self.mask_token.expand(B, N, -1)
        reconstructed = self.decoder(decoder_input, memory)
        reconstructed = self.output_proj(reconstructed)
        return reconstructed

    def encode(self, patch_tokens):
        return self.encoder(patch_tokens)


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Linear(feature_dim, feature_dim)

    def forward(self, unpooled, text_features):
        out, attn_weights = self.cross_attn(text_features, unpooled, unpooled)
        text_features = self.norm1(self.dropout(text_features + out))
        ff = self.feed_forward(text_features)
        text_features = self.norm2(self.dropout(text_features + ff))
        return text_features, attn_weights


class APTMIM(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.model_cfg = cfg.get('model', ConfigNode())
        self.clip_model = clip_model.to(device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.image_encoder = ImageEncoder(self.clip_model)
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        feature_dim = self.clip_model.text_projection.shape[1]
        hidden_dim = self.model_cfg.get('mim_hidden_dim', 512)
        num_encoder_layers = self.model_cfg.get('mim_encoder_layers', 3)
        num_decoder_layers = self.model_cfg.get('mim_decoder_layers', 3)
        mim_num_heads = self.model_cfg.get('mim_num_heads', 8)
        mim_dropout = self.model_cfg.get('mim_dropout', 0.1)
        self.mask_ratio = self.model_cfg.get('mask_ratio', 0.4)
        self.mim_module = MIMModule(feature_dim, hidden_dim, num_encoder_layers, num_decoder_layers, mim_num_heads, mim_dropout)
        num_heads = self.model_cfg.get('num_heads', 8)
        dropout = self.model_cfg.get('dropout', 0.1)
        num_layers = self.model_cfg.get('num_layers', 1)
        prompt_layers = []
        for _ in range(num_layers):
            prompt_layers.append(CrossAttention(feature_dim=feature_dim, num_heads=num_heads, dropout=dropout))
        self.prompt_learner = nn.ModuleList(prompt_layers)
        self.logit_scale = clip_model.logit_scale
        dataset_name = self.model_cfg.get('dataset_name', 'ImageNet')
        template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")
        self.classnames = classnames
        self.text_features, self.prompts = self._init_text_feats(template, classnames)

    def _init_text_feats(self, template, classnames):
        myencoder = TextEncoder(self.clip_model).to(self.device)
        prompts = [template.format(c.replace('_', ' ')) for c in classnames]
        prompts_tokens = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        tokens = myencoder.encode_text_tokens(prompts_tokens)
        text_features = myencoder.cls_from_tokens(tokens, prompts_tokens)
        return text_features, prompts_tokens

    def generate_mask(self, batch_size, num_patches):
        num_masked = int(num_patches * self.mask_ratio)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            indices = torch.randperm(num_patches, device=self.device)[:num_masked]
            mask[i, indices] = True
        return mask

    def forward_mim(self, images):
        with torch.no_grad():
            patch_tokens, cls_feature = self.image_encoder(images)
        batch_size, num_patches, _ = patch_tokens.shape
        mask = self.generate_mask(batch_size, num_patches)
        reconstructed = self.mim_module(patch_tokens, mask)
        loss = F.mse_loss(reconstructed[mask], patch_tokens[mask])
        return loss, reconstructed, patch_tokens, mask

    def forward_apt(self, images, label=None):
        with torch.no_grad():
            patch_tokens, cls_feature = self.image_encoder(images)
            mim_encoded = self.mim_module.encode(patch_tokens)
        unpooled_with_cls = torch.cat([cls_feature.unsqueeze(1), mim_encoded], dim=1)
        unpooled_images = unpooled_with_cls.permute(1, 0, 2)
        base_text_features = self.text_features.clone()
        text_features = base_text_features.unsqueeze(1).expand(-1, unpooled_images.shape[1], -1)
        for layer in self.prompt_learner:
            text_features, _ = layer(unpooled_images, text_features)
        text_features = text_features.permute(1, 0, 2)
        text_features = F.normalize(text_features, dim=-1)
        logit_scale = self.logit_scale.exp()
        image_features = F.normalize(cls_feature, dim=-1)
        image_features = image_features.unsqueeze(1)
        logits = logit_scale * F.cosine_similarity(image_features, text_features, dim=-1)
        if self.training and label is not None:
            loss = F.cross_entropy(logits, label)
            return loss, logits
        return logits

    def forward(self, images, label=None):
        return self.forward_apt(images, label)

    def mim_trainable_parameters(self):
        return self.mim_module.parameters()

    def apt_trainable_parameters(self):
        return self.prompt_learner.parameters()

    def freeze_mim(self):
        for param in self.mim_module.parameters():
            param.requires_grad = False

    def unfreeze_mim(self):
        for param in self.mim_module.parameters():
            param.requires_grad = True


class APTMIMTrainer:
    def __init__(self, cfg, classnames, device, log_file=None):
        self.cfg = cfg
        self.training_cfg = cfg.get('training', ConfigNode())
        self.model_cfg = cfg.get('model', ConfigNode())
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.log_file = log_file
        backbone_name = self.model_cfg.get('backbone', 'ViT-B/16')
        print(f"Loading CLIP (backbone: {backbone_name})")
        clip_model = load_clip_to_cpu(backbone_name)
        precision = self.training_cfg.get('precision', 'fp32')
        if precision in ['fp32', 'amp']:
            clip_model.float()
        self.model = APTMIM(cfg, classnames, clip_model, self.device)
        self.model.to(self.device)
        self.mim_optimizer = None
        self.mim_scheduler = None
        self.apt_optimizer = None
        self.apt_scheduler = None
        self.setup_mim_optimizer()

    def setup_mim_optimizer(self):
        lr = coerce_to_float(self.training_cfg.get('mim_learning_rate', 0.0001), 0.0001)
        weight_decay = coerce_to_float(self.training_cfg.get('weight_decay', 0.01), 0.01)
        optimizer_type = self.training_cfg.get('optimizer', 'AdamW')
        trainable_params = list(self.model.mim_trainable_parameters())
        if optimizer_type == 'AdamW':
            self.mim_optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            self.mim_optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.mim_optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        num_epochs = coerce_to_int(self.training_cfg.get('mim_epochs', 20), 20)
        self.mim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.mim_optimizer, T_max=num_epochs)

    def setup_apt_optimizer(self):
        lr = coerce_to_float(self.training_cfg.get('apt_learning_rate', 0.002), 0.002)
        weight_decay = coerce_to_float(self.training_cfg.get('weight_decay', 0.01), 0.01)
        optimizer_type = self.training_cfg.get('optimizer', 'AdamW')
        trainable_params = list(self.model.apt_trainable_parameters())
        if optimizer_type == 'AdamW':
            self.apt_optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            self.apt_optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.apt_optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        num_epochs = coerce_to_int(self.training_cfg.get('apt_epochs', 20), 20)
        self.apt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.apt_optimizer, T_max=num_epochs)

    def train_mim_step(self, images):
        self.model.mim_module.train()
        images = images.to(self.device)
        loss, _, _, _ = self.model.forward_mim(images)
        self.mim_optimizer.zero_grad() # type: ignore
        loss.backward()
        self.mim_optimizer.step() # type: ignore
        return {"loss": loss.item()}

    def train_apt_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        self.model.train()
        loss, logits = self.model.forward_apt(images, labels)
        self.apt_optimizer.zero_grad() # type: ignore
        loss.backward()
        self.apt_optimizer.step() # type: ignore
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
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model.forward_apt(images)
                loss = F.cross_entropy(logits, labels) # type: ignore
                running_loss += loss.item()
                steps += 1
                _, predicted = torch.max(logits, 1) # type: ignore
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = 100 * correct / total if total > 0 else 0.0
        avg_loss = running_loss / max(1, steps)
        return {"accuracy": accuracy, "loss": avg_loss, "predictions": all_preds, "true_labels": all_labels}

    def save_model(self, path):
        checkpoint = {
            'mim_module_state_dict': self.model.mim_module.state_dict(),
            'prompt_learner_state_dict': self.model.prompt_learner.state_dict(),
        }
        if self.mim_optimizer is not None:
            checkpoint['mim_optimizer_state_dict'] = self.mim_optimizer.state_dict()
        if self.mim_scheduler is not None:
            checkpoint['mim_scheduler_state_dict'] = self.mim_scheduler.state_dict()
        if self.apt_optimizer is not None:
            checkpoint['apt_optimizer_state_dict'] = self.apt_optimizer.state_dict()
        if self.apt_scheduler is not None:
            checkpoint['apt_scheduler_state_dict'] = self.apt_scheduler.state_dict()
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.mim_module.load_state_dict(checkpoint['mim_module_state_dict'])
        if 'prompt_learner_state_dict' in checkpoint:
            self.model.prompt_learner.load_state_dict(checkpoint['prompt_learner_state_dict'])
        print(f"Model loaded from {path}")


class APTMIMPipeline:
    def __init__(self, config):
        if not isinstance(config, ConfigNode):
            config = ConfigNode(config)
        self.config = config
        self.model_cfg = self.config.get('model', ConfigNode())
        self.training_cfg = self.config.get('training', ConfigNode())
        self.data_cfg = self.config.get('data', ConfigNode())
        self.logging_cfg = self.config.get('logging', ConfigNode())
        device_value = self.training_cfg.get("device", "cuda:0")
        device_name = coerce_to_str(device_value, "cuda:0")
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.batch_size = coerce_to_int(self.training_cfg.get("batch_size", 16), 16)
        self.num_workers = coerce_to_int(self.data_cfg.get("num_workers", 4), 4)
        self.val_fraction = coerce_to_float(self.data_cfg.get("val_size", 0.7), 0.7)
        if self.val_fraction > 1.0:
            self.val_fraction = self.val_fraction / 100.0
        self.dataset_root = coerce_to_str(self.data_cfg.get("root", "./datasets/cub-200-2011-renamed"), "./datasets/cub-200-2011-renamed")
        self.seed = coerce_to_int(self.data_cfg.get("seed", 42), 42)
        self.kshot = coerce_to_int(self.data_cfg.get("kshot", 0), 0)
        base_output = coerce_to_str(self.logging_cfg.get("output_dir", "outputs"), "outputs")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(base_output, timestamp)
        print(f"Run directory: {self.run_dir}")
        self.config_path = os.path.join(self.run_dir, 'config.json')
        self.metrics_path = os.path.join(self.run_dir, 'metrics.json')
        self.mim_model_path = os.path.join(self.run_dir, 'mim_trained.pt')
        self.best_model_path = os.path.join(self.run_dir, 'best.pt')
        self.last_model_path = os.path.join(self.run_dir, 'last.pt')
        self.log_file = os.path.join(self.run_dir, 'training.log')
        self.clip_mean = get_config_value(self.data_cfg, "clip_mean", [0.48145466, 0.4578275, 0.40821073])
        self.clip_std = get_config_value(self.data_cfg, "clip_std", [0.26862954, 0.26130258, 0.27577711])
        self.dataset: Optional[ImageFolder] = None
        self.val_loader: Optional[DataLoader] = None
        self.train_loader: Optional[DataLoader] = None
        self.unlabeled_loader: Optional[DataLoader] = None
        self.classnames: List[str] = []
        self.val_indices: List[int] = []
        self.labeled_indices: List[int] = []
        self.unlabeled_indices: List[int] = []
        self.metrics: List[Dict[str, Any]] = []
        self.best_val_acc = -float('inf')
        self.trainer: Optional[APTMIMTrainer] = None

    def run(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self._prepare_directories()
        self._load_dataset()
        self._split_dataset()
        self._initialize_trainer()
        self._train_mim()
        self._train_apt()
        self._finalize()

    def _prepare_directories(self):
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.log_file, 'w') as f:
            f.write('')

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

    def _split_dataset(self):
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before splitting.")
        samples_by_class = defaultdict(list)
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            samples_by_class[class_idx].append(idx)
        rng = random.Random(self.seed)
        val_indices = []
        labeled_indices = []
        unlabeled_indices = []
        for class_idx in sorted(samples_by_class.keys()):
            class_samples = list(samples_by_class[class_idx])
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
            labeled_indices.extend(labeled_part)
            unlabeled_indices.extend(leftover_part)
        self.val_indices = val_indices
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        if len(self.val_indices) > 0:
            val_ds = Subset(self.dataset, self.val_indices)
            self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        if len(self.labeled_indices) > 0:
            train_ds = Subset(self.dataset, self.labeled_indices)
            self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        all_train_indices = self.labeled_indices + self.unlabeled_indices
        if len(all_train_indices) > 0:
            unlabeled_ds = Subset(self.dataset, all_train_indices)
            self.unlabeled_loader = DataLoader(unlabeled_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.classnames = list(self.dataset.classes)
        print(f"Dataset loaded: {len(self.dataset)} total images.")
        print(f"Validation split: {len(self.val_indices)} images ({len(self.val_indices)/len(self.dataset)*100:.2f}%).")
        print(f"Labeled (kshot={self.kshot}): {len(self.labeled_indices)} images.")
        print(f"Unlabeled pool: {len(self.unlabeled_indices)} images.")
        with open(self.config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=4)

    def _initialize_trainer(self):
        if not self.classnames:
            raise RuntimeError("Class names unavailable before trainer initialization.")
        self.trainer = APTMIMTrainer(self.config, self.classnames, device=str(self.device), log_file=self.log_file)

    def _train_mim(self):
        if self.trainer is None or self.unlabeled_loader is None:
            raise RuntimeError("Trainer or unlabeled loader not initialized.")
        mim_epochs = coerce_to_int(self.training_cfg.get('mim_epochs', 20), 20)
        print(f"\n{'='*50}")
        print(f"Phase 1: MIM Training for {mim_epochs} epochs")
        print(f"{'='*50}")
        with open(self.log_file, 'a') as f:
            f.write(f"\nPhase 1: MIM Training for {mim_epochs} epochs\n")
        self.trainer.model.unfreeze_mim()
        for epoch_idx in range(1, mim_epochs + 1):
            start_time = time.time()
            self.trainer.model.mim_module.train()
            running_loss = 0.0
            steps = 0
            for batch in self.unlabeled_loader:
                images, _ = batch
                loss_dict = self.trainer.train_mim_step(images)
                running_loss += loss_dict['loss']
                steps += 1
            avg_loss = running_loss / max(1, steps)
            val_acc = 0.0
            if self.val_loader is not None:
                results = self.trainer.evaluate(self.val_loader)
                val_acc = results['accuracy']
            epoch_time = time.time() - start_time
            self.metrics.append({
                'phase': 'mim',
                'epoch': epoch_idx,
                'mim_loss': avg_loss,
                'val_acc': val_acc,
                'time': epoch_time
            })
            epoch_str = f"MIM Epoch {epoch_idx}/{mim_epochs} - loss={avg_loss:.4f} - val_acc={val_acc:.2f}% - time={epoch_time:.2f}s"
            print(epoch_str)
            with open(self.log_file, 'a') as f:
                f.write(epoch_str + '\n')
            if self.trainer.mim_scheduler is not None:
                self.trainer.mim_scheduler.step()
        self.trainer.save_model(self.mim_model_path)
        print(f"MIM training completed. Model saved to {self.mim_model_path}")

    def _train_apt(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized.")
        if self.train_loader is None or len(self.labeled_indices) == 0:
            print("No labeled samples available for APT training. Skipping APT phase.")
            return
        apt_epochs = coerce_to_int(self.training_cfg.get('apt_epochs', 20), 20)
        print(f"\n{'='*50}")
        print(f"Phase 2: APT Training for {apt_epochs} epochs (MIM frozen)")
        print(f"{'='*50}")
        with open(self.log_file, 'a') as f:
            f.write(f"\nPhase 2: APT Training for {apt_epochs} epochs (MIM frozen)\n")
        self.trainer.model.freeze_mim()
        self.trainer.setup_apt_optimizer()
        for epoch_idx in range(1, apt_epochs + 1):
            start_time = time.time()
            self.trainer.model.prompt_learner.train()
            running_loss = 0.0
            running_accuracy = 0.0
            steps = 0
            for batch in self.train_loader:
                loss_dict = self.trainer.train_apt_step(batch)
                running_loss += loss_dict['loss']
                running_accuracy += loss_dict['accuracy']
                steps += 1
            avg_loss = running_loss / max(1, steps)
            avg_acc = running_accuracy / max(1, steps)
            val_acc = 0.0
            val_loss = 0.0
            if self.val_loader is not None:
                results = self.trainer.evaluate(self.val_loader)
                val_acc = results['accuracy']
                val_loss = results['loss']
            epoch_time = time.time() - start_time
            self.metrics.append({
                'phase': 'apt',
                'epoch': epoch_idx,
                'train_loss': avg_loss,
                'train_acc': avg_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'time': epoch_time
            })
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.trainer.save_model(self.best_model_path)
            epoch_str = f"APT Epoch {epoch_idx}/{apt_epochs} - loss={avg_loss:.4f} - train_acc={avg_acc:.2f}% - val_loss={val_loss:.4f} - val_acc={val_acc:.2f}% - time={epoch_time:.2f}s"
            print(epoch_str)
            with open(self.log_file, 'a') as f:
                f.write(epoch_str + '\n')
            if self.trainer.apt_scheduler is not None:
                self.trainer.apt_scheduler.step()

    def _finalize(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before finalization.")
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.trainer.save_model(self.last_model_path)
        print(f"\nTraining completed. Best val accuracy: {self.best_val_acc:.2f}%")
        print(f"Results written to {self.run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="APT with MIM")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    base_config = load_config_file(args.config)
    merged_config = merge_configs(base_config, overrides)
    pipeline = APTMIMPipeline(merged_config)
    pipeline.run()


if __name__ == "__main__":
    main()
