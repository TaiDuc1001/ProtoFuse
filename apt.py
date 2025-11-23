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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from decode import APTDecoder
from thop import profile
from typing import Any, Dict, List, Optional
from utils import (
    log_decoded_prompts,
    run_dataset_eda,
    save_class_distribution_plot,
    save_confusion_artifacts,
    visualize_attention_maps,
    visualize_gradcam_maps,
)

ARG_SCHEMA = {
    'config': {'type': str, 'required': True, 'help': 'Path to YAML configuration file'},
    'output_dir': {'type': str, 'help': 'Override logging.output_dir from config', 'config_path': 'logging.output_dir'},
    'device': {'type': str, 'help': 'Override training.device from config', 'config_path': 'training.device'},
}

DEFAULT_TRAINING_EPOCHS = 100

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

class CacheAdapter(nn.Module):
    def __init__(self, feature_dim, num_classes, alpha=1.0, beta=1.0, temperature=10.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        
        self.register_buffer('cache_keys', torch.zeros(0, feature_dim))
        self.register_buffer('cache_values', torch.zeros(0, num_classes))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        self.enabled = True

    def update_cache(self, keys, labels):
        keys = F.normalize(keys, dim=-1)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        self.cache_keys = keys.detach().clone()
        self.cache_values = one_hot.detach().clone()
        
        counts = torch.zeros(self.num_classes, device=keys.device)
        unique_labels, counts_per_label = labels.unique(return_counts=True)
        counts[unique_labels] = counts_per_label.float()
        self.class_counts = counts

    def get_cache_logits(self, image_features):
        if not self.enabled or self.cache_keys.shape[0] == 0:
            print("Cache is empty or disabled. Returning zero logits.")
            return torch.zeros(image_features.shape[0], self.num_classes, device=image_features.device)

        image_features = F.normalize(image_features, dim=-1)
        affinity = image_features @ self.cache_keys.t()
        
        cache_logits = ((-1) * (self.temperature - self.temperature * affinity)).exp() @ self.cache_values
        
        safe_counts = self.class_counts.clone()
        safe_counts[safe_counts == 0] = 1.0
        cache_logits = cache_logits / safe_counts.unsqueeze(0)
        
        return cache_logits

    def forward(self, image_features, apt_logits):
        if not self.enabled or self.cache_keys.shape[0] == 0:
            return apt_logits
        
        cache_logits = self.get_cache_logits(image_features)
        final_logits = self.beta * apt_logits + self.alpha * cache_logits
        return final_logits

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

        for idx, block in enumerate(self.transformer.resblocks):
            x = block(x)

        final_unpooled = x.permute(1, 0, 2)
        final_unpooled = self.ln_post(final_unpooled)

        if self.proj is not None:
            final_unpooled = final_unpooled @ self.proj

        global_feature = final_unpooled[:, 0, :]
        return final_unpooled, global_feature

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

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.clip_model = clip_model.to(device)
        if not isinstance(cfg, ConfigNode):
            cfg = ConfigNode(cfg)
        self.cfg = cfg
        self.model_cfg = getattr(self.cfg, 'model', ConfigNode())
        self.training_cfg = getattr(self.cfg, 'training', ConfigNode())
        self.device = device

        prompt_dim = self.clip_model.text_projection.shape[1]
        num_heads = self.model_cfg.get('num_heads', 8)
        dropout = self.model_cfg.get('dropout', 0.1)

        prompt_layers = []
        for _ in range(self.model_cfg.get('num_layers', 1)):
            prompt_layers.append(
                CrossAttention(
                    feature_dim=prompt_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
            )
        self.prompt_learner = nn.ModuleList(prompt_layers)

        if self.training_cfg.get('precision', 'fp32') == 'fp16':
            self.prompt_learner = self.prompt_learner.half()

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.vis_encoder = ImageEncoder(self.clip_model)
        self.logit_scale = clip_model.logit_scale

        self.text_features, self.prompts, self.text_tokens = self._init_text_feats(self.model_cfg, classnames)
        self.base_text_features = self.text_features.clone().detach()

    def _init_text_feats(self, cfg, classnames):
        dataset_name = cfg.get('dataset_name', 'ImageNet')
        temp = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")
        myencoder = TextEncoder(self.clip_model).to(self.device)
        prompts = [temp.format(c.replace('_', ' ')) for c in classnames]

        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        tokens = myencoder.encode_text_tokens(prompts)
        text_features = myencoder.cls_from_tokens(tokens, prompts)

        tokens = None
        return text_features, prompts, tokens

    def forward(self, image, label=None):
        with torch.no_grad():
            pass

        visual_output = self.vis_encoder(image)
        unpooled_levels, image_features = visual_output
        if not isinstance(unpooled_levels, list):
            unpooled_levels = [unpooled_levels]

        attn_maps = []
        base_text_features = self.text_features.clone()

        unpooled_images = unpooled_levels[0].permute(1, 0, 2)
        text_features = base_text_features.unsqueeze(1).expand(-1, unpooled_images.shape[1], -1)

        for layer in self.prompt_learner:
            text_features, attn_weights = layer(unpooled_images, text_features)
            attn_maps.append(attn_weights)

        text_features = text_features.permute(1, 0, 2)
        text_features = F.normalize(text_features, dim=-1)
        
        logit_scale = self.logit_scale.exp()
        image_features = F.normalize(image_features, dim=-1)
        image_features = image_features.unsqueeze(1)
        
        logits = logit_scale * F.cosine_similarity(image_features, text_features, dim=-1)

        mode = self.cfg.get('mode', self.training_cfg.get('mode', 'logits'))

        if self.training and label is not None:
            loss = F.cross_entropy(logits, label)
            return loss, logits
        elif mode == "logits":
            return logits
        elif mode == "map":
            return logits, attn_maps
        elif mode == "features":
            return logits, text_features

        return logits
    def _prompt_layers_iter(self):
        if isinstance(self.prompt_learner, nn.ModuleList):
            return self.prompt_learner
        return [self.prompt_learner]
    def _prepare_text_features(self):
        return self.text_features.clone()
    def trainable_parameters(self):
        return self.prompt_learner.parameters()

    def get_trainable_parameter_names(self):
        names = [f"prompt_learner.{name}" for name, _ in self.prompt_learner.named_parameters()]
        return names

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

class APT:
    def __init__(self, cfg, classnames, device="cuda", log_file=None):
        if not isinstance(cfg, ConfigNode):
            cfg = ConfigNode(cfg)
        self.cfg = cfg
        self.training_cfg = self.cfg.get('training', ConfigNode())
        self.model_cfg = self.cfg.get('model', ConfigNode())
        self.data_cfg = self.cfg.get('data', ConfigNode())
        self.cache_cfg = self.cfg.get('cache', ConfigNode())
        self.classnames = classnames
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.log_file = log_file
        
        self.gradients = None
        self.activations = None
        
        self.build_model()
        self.setup_optimizer()
        precision_mode = self._cfg_str('fp32', 'training.precision', 'precision')
        self.scaler = GradScaler() if precision_mode == 'amp' else None
        
        if bool(self._cfg_value('training.run_decoder', 'run_decoder', default=False)):
            self.decoder = APTDecoder(
                device=str(self.device),
                clip_model_name=self._cfg_str('ViT-B/32', 'model.backbone', 'backbone')
            )
        
        if bool(self._cfg_value('model.use_cache', 'cache.use_cache', 'training.use_cache', 'use_cache', default=False)):
            self.cache_adapter = CacheAdapter(
                feature_dim=self.model.vis_encoder.ln_post.normalized_shape[0],
                num_classes=len(classnames),
                alpha=self._cfg_float(1.0, 'cache.alpha', 'model.cache_alpha', 'cache_alpha'),
                beta=self._cfg_float(1.0, 'cache.beta', 'model.cache_beta', 'cache_beta'),
                temperature=self._cfg_float(5.0, 'cache.temperature', 'model.cache_temperature', 'cache_temperature')
            ).to(self.device)
        else:
            self.cache_adapter = None
    
    def _cfg_value(self, *paths, default=None):
        sentinel = object()
        for path in paths:
            value = get_config_value(self.cfg, path, sentinel)
            if value is not sentinel:
                return value
            print(f"Config path '{path}' not found. Using default value: {default}")
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
        backbone_name = self._cfg_str('ViT-B/32', 'model.backbone', 'backbone')
        msg = f"Loading CLIP (backbone: {backbone_name})"
        print(msg)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')
        
        clip_model = load_clip_to_cpu(backbone_name)
        
        if self._cfg_str('fp32', 'training.precision', 'precision') in ['fp32', 'amp']:
            clip_model.float()

        self.model = CustomCLIP(self.cfg, self.classnames, clip_model, self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        def format_params(num):
            if num >= 1e9:
                return f"{num/1e9:.2f}B"
            elif num >= 1e6:
                return f"{num/1e6:.2f}M"
            elif num >= 1e3:
                return f"{num/1e3:.2f}K"
            else:
                return str(num)
        
        flops_results = {}
        self.model.to(self.device)
        self.model.eval()
        
        for param in self.model.parameters():
            if param.device != self.device:
                param.data = param.data.to(self.device)
        
        input_tensor = torch.randn(1, 3, 224, 224, device=self.device, dtype=torch.float32)
        
        flops_results = {}
        with torch.no_grad():
            model_copy = copy.deepcopy(self.model)
            model_copy.to(self.device)
            result = profile(model_copy, inputs=(input_tensor,), verbose=False)
            if isinstance(result, (list, tuple)):
                macs = result[0] if len(result) > 0 else 0
            else:
                macs = result
            gflops_thop = macs / 1e9
            flops_results['thop'] = gflops_thop
            del model_copy
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        msg = f"Learnable parameters: {format_params(learnable_params)} / Total parameters: {format_params(total_params)} (FLOPs: {gflops_thop:.2f} GFLOPs)"
        print(msg)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')

        msg = "Turning off gradients in both the image and the text encoder"
        # print(msg)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')
        
        trainable_names = set(self.model.get_trainable_parameter_names())
        for name, param in self.model.named_parameters():
            param.requires_grad_(name in trainable_names)
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        msg = f"Parameters to be updated: {enabled}"
        # print(msg)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')

        self.model.to(self.device)
    
    def setup_optimizer(self):
        lr = self._cfg_float(0.002, 'training.learning_rate')
        weight_decay = self._cfg_float(0.0005, 'training.weight_decay')
        optimizer_type = self._cfg_str('SGD', 'training.optimizer')
        trainable_params = list(self.model.trainable_parameters())
        
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

    def update_cache_memory(self, dataset, labeled_indices):
        if self.cache_adapter is None:
            return

        print("Building Cache Memory from Labeled Set...")
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write("Building Cache Memory from Labeled Set...\n")

        self.model.eval()
        
        subset = Subset(dataset, labeled_indices)
        loader = DataLoader(
            subset,
            batch_size=32,
            shuffle=False,
            num_workers=self._cfg_int(4, 'data.num_workers', 'num_workers')
        )
        
        all_keys = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                visual_out = self.model.vis_encoder(images)
                img_feats = visual_out[1]
                img_feats = F.normalize(img_feats, dim=-1)
                
                all_keys.append(img_feats)
                all_labels.append(labels.to(self.device))
        
        if all_keys:
            keys = torch.cat(all_keys, dim=0)
            lbls = torch.cat(all_labels, dim=0)
            self.cache_adapter.update_cache(keys, lbls)
            msg = f"Cache updated: {keys.shape[0]} samples stored."
            print(msg)
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(msg + '\n')
    
    def train_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        self.model.train()
        self.model.prompt_learner.train()
        
        precision = self._cfg_str('fp32', 'training.precision', 'precision')
        
        if precision == 'amp':
            with autocast():
                loss, logits = self.model(images, labels)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward() # type: ignore
            self.scaler.step(self.optimizer) # type: ignore
            self.scaler.update() # type: ignore
        else:
            loss, logits = self.model(images, labels)
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

                # 1. APT Logits
                logits = self.model(images)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]

                # 2. Cache Logits
                if self.cache_adapter is not None:
                    visual_out = self.model.vis_encoder(images)
                    img_feats = visual_out[1]
                    logits = self.cache_adapter(img_feats, logits)

                loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
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
            'model_state_dict': self.model.state_dict(),
            'prompt_learner_state_dict': self.model.prompt_learner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'cfg': self.cfg
        }
        torch.save(checkpoint, path)
        msg = f"Model saved to {path}"
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        model_state = checkpoint.get('model_state_dict')
        if model_state is not None:
            self.model.load_state_dict(model_state, strict=False)
        if 'prompt_learner_state_dict' in checkpoint:
            self.model.prompt_learner.load_state_dict(checkpoint['prompt_learner_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        msg = f"Model loaded from {path}"
        print(msg)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')

    def decode_adapted_prompts(self, images, entry_length=30, temperature=1.0, batch_decode_size=32):
        self.model.eval()
        images = images.to(self.device)

        with torch.no_grad():
            encoder_output = self.model.vis_encoder(images)
            unpooled_images = encoder_output[0].permute(1, 0, 2)
            text_features = self.model._prepare_text_features()
            text_features = text_features.unsqueeze(1).expand(-1, unpooled_images.shape[1], -1)

            for layer in self.model._prompt_layers_iter():
                text_features, _ = layer(unpooled_images, text_features)

            adapted = text_features.permute(1, 0, 2)
            batch_size, num_classes, embedding_dim = adapted.shape
            all_embeddings = adapted.reshape(-1, embedding_dim)
            all_captions = []
            
            for i in range(0, all_embeddings.shape[0], batch_decode_size):
                batch_embeddings = all_embeddings[i:i+batch_decode_size]
                batch_captions = self.decoder.decode_from_apt_embedding_batch(
                    batch_embeddings, entry_length=entry_length, temperature=temperature
                )
                all_captions.extend(batch_captions)
            
            batch_out = []
            caption_idx = 0
            for b in range(batch_size):
                classes_out = []
                for c in range(num_classes):
                    class_name = self.classnames[c] if c < len(self.classnames) else f"Class_{c}"
                    classes_out.append({
                        'class_id': c, 'class_name': class_name, 'generated_caption': all_captions[caption_idx]
                    })
                    caption_idx += 1
                batch_out.append(classes_out)
        return batch_out

    def generate_gradcam(self, images, target_classes):
        original_mode = self.model.training
        self.model.train()
        
        for param in self.model.vis_encoder.parameters():
            param.requires_grad_(True)
            
        images = images.to(self.device)
        images.requires_grad_(True)
        
        encoder_output = self.model.vis_encoder(images)
        target_unpooled, _ = encoder_output

        target_unpooled.retain_grad()
        batch_size = images.shape[0]
        gradcams = []
        
        for i in range(batch_size):
            base_text = self.model._prepare_text_features()
            unpooled_single = target_unpooled[i:i+1].permute(1, 0, 2)
            text_features = base_text.unsqueeze(1).expand(-1, unpooled_single.shape[1], -1)
            for layer in self.model._prompt_layers_iter():
                text_features, _ = layer(unpooled_single, text_features)

            text_features = text_features.permute(1, 0, 2)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = target_unpooled[i:i+1, 0, :]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.unsqueeze(1)
            
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * F.cosine_similarity(image_features, text_features, dim=-1)
            score = logits[0, target_class]
            
            self.model.zero_grad()
            score.backward(retain_graph=True)
            
            if target_unpooled.grad is None:
                gradcams.append(np.zeros((8, 8)))
                print("Warning: Empty CAM encountered.")
                continue
            gradients = target_unpooled.grad[i]
            activations = target_unpooled[i]
            weights = torch.mean(gradients[1:], dim=0)
            cam = torch.sum(activations[1:] * weights, dim=-1)
            cam = F.relu(cam)
            cam_before = cam.detach().cpu().numpy()
            
            if cam_before.size > 0:
                cam = (cam_before - cam_before.min()) / (cam_before.max() - cam_before.min() + 1e-8)
                num_patches = cam.size
                grid_size = int(np.sqrt(num_patches))
                if grid_size * grid_size == num_patches:
                    cam = cam.reshape(grid_size, grid_size)
                else:
                    cam = np.pad(cam, (0, grid_size * grid_size - num_patches), mode='constant').reshape(grid_size, grid_size)
                    print("Warning: CAM size is not a perfect square, padding to make it square.")
            else:
                print("Warning: Empty CAM encountered.")
                cam = np.zeros((8, 8))
            
            gradcams.append(cam)
            if target_unpooled.grad is not None:
                target_unpooled.grad.zero_()
        
        for param in self.model.vis_encoder.parameters():
            param.requires_grad_(False)
        self.model.train(original_mode)
        return gradcams

class APTTrainingPipeline:
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
        self.batch_size = coerce_to_int(batch_value, 8, key="training.batch_size")

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

        run_eda_value = get_config_value(self.data_cfg, "run_eda", True)
        self.run_eda = bool(True if run_eda_value is None else run_eda_value)

        class_dist_value = get_config_value(self.training_cfg, "class_distribution", True)
        self.class_distribution_enabled = bool(True if class_dist_value is None else class_dist_value)

        base_output_value = self.logging_cfg.get("output_dir", "outputs")
        base_output = coerce_to_str(base_output_value, "outputs", key="logging.output_dir")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(base_output, timestamp)
        print(f"Run directory: {self.run_dir}")
        self.selection_log_path = os.path.join(self.run_dir, 'al_selected_paths.log')
        self.config_path = os.path.join(self.run_dir, 'config.json')
        self.metrics_path = os.path.join(self.run_dir, 'metrics.json')
        self.final_prompts_path = os.path.join(self.run_dir, 'final_prompts.json')
        self.best_model_path = os.path.join(self.run_dir, 'best.pt')
        self.last_model_path = os.path.join(self.run_dir, 'last.pt')
        self.log_file = os.path.join(self.run_dir, 'training.log')
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
        self.sample_cache = {
            'images': None, 'labels': None, 'paths': [], 'decoded_prompts': None
        }

        self.trainer: Optional[APT] = None
        self.trainer_cfg: ConfigNode = ConfigNode({})
        self.rounds = 1

    def _get_training_epochs(self):
        epochs_value = None
        if isinstance(self.training_cfg, dict):
            epochs_value = self.training_cfg.get('epochs', None)
        return coerce_to_int(epochs_value, DEFAULT_TRAINING_EPOCHS, key='training.epochs')

    def run(self):
        self._prepare_directories()
        self._load_dataset()
        self._split_dataset()
        self._initialize_trainer()
        self._train_epochs()
        self._finalize()

    def _prepare_directories(self):
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.eda_dir, exist_ok=True)
        with open(self.log_file, 'w') as f:
            f.write('')

    def _build_transforms(self):
        base_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.clip_mean, std=self.clip_std),
        ]
        if bool(get_config_value(self.training_cfg, "use_cutout", False)):
            base_transforms.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0))
            print("Using Cutout (RandomErasing) augmentation.")
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
            print("Warning: validation split is empty; skipping validation metrics.")

        self.classnames = list(self.dataset.classes)

        stats = {
            'total_images': len(self.dataset),
            'val_count': len(self.val_indices),
            'train_count': len(self.train_indices),
            'labeled_count': len(self.train_indices),
            'unlabeled_count': len(self.unlabeled_indices),
            'train_pool_size': len(self.train_indices) + len(self.unlabeled_indices)
        }
        print(f"Dataset loaded: {stats['total_images']} total images.")
        val_percentage = (stats['val_count'] / stats['total_images'] * 100.0) if stats['total_images'] > 0 else 0.0
        print(f"Validation split: {stats['val_count']} images ({val_percentage:.2f}%).")
        print(f"Train split size: {stats['train_count']} images.")
        if stats['unlabeled_count'] > 0:
            print(f"Unlabeled pool size: {stats['unlabeled_count']} images.")

        trainer_cfg = self._build_trainer_config(stats, val_percentage)
        with open(self.config_path, 'w') as f:
            json.dump(trainer_cfg.to_dict(), f, indent=4)

        with open(self.log_file, 'a') as f:
            f.write(f"Config: {trainer_cfg.to_dict()}\n\n")
            f.write('=' * 50 + '\n')

    def _build_trainer_config(self, stats, val_percentage):
        extra_values = {
            'dataset_root': self.dataset_root,
            'val_size': self.val_fraction,
            'rounds': self.rounds,
            'classnames': self.classnames,
            'num_classes': len(self.classnames),
            'train_size': stats.get('labeled_count', stats['train_count']),
            'val_size_count': stats['val_count'],
            'train_pool_size': stats.get('train_pool_size', stats['train_count'] + stats.get('unlabeled_count', 0)),
            'unlabeled_pool_size': stats.get('unlabeled_count', 0),
            'val_percentage_actual': val_percentage,
            'completed_rounds': 0,
        }

        trainer_cfg = build_config_namespace(self.config, extra_values)
        self.trainer_cfg = trainer_cfg
        return trainer_cfg

    def _initialize_trainer(self):
        if not self.classnames:
            raise RuntimeError("Class names unavailable before trainer initialization.")
        self.trainer = APT(self.trainer_cfg, self.classnames, device=str(self.device), log_file=self.log_file)

    def _train_epochs(self):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before training.")
        if not self.train_indices:
            raise RuntimeError("No training samples available.")

        round_dir = os.path.join(self.run_dir, 'round_01')
        os.makedirs(round_dir, exist_ok=True)
        train_subset = Subset(self.dataset, list(self.train_indices))
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        epochs_total = self._get_training_epochs()

        if self.trainer.cache_adapter is not None:
            self.trainer.update_cache_memory(self.dataset, self.train_indices)

        for epoch_idx in range(1, epochs_total + 1):
            self._run_epoch(1, epoch_idx, epochs_total, train_loader, round_dir)

        self.trainer_cfg.meta.completed_rounds = 1

    def _run_epoch(self, round_idx, epoch_in_round, epochs_this_round, train_loader, round_dir):
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

        epoch_dir = os.path.join(round_dir, f'epoch_{epoch_in_round:03d}')
        os.makedirs(epoch_dir, exist_ok=True)
        maps_dir = os.path.join(epoch_dir, 'maps')
        os.makedirs(maps_dir, exist_ok=True)

        if bool(get_config_value(self.training_cfg, 'confusion_matrix', False)) and all_labels:
            save_confusion_artifacts(all_labels, all_preds, self.global_epoch, epoch_dir, self.log_file)

        if self.class_distribution_enabled and all_labels:
            save_class_distribution_plot(
                all_labels,
                all_preds,
                self.global_epoch,
                epoch_dir,
                self.log_file,
                self.classnames,
            )

        self._refresh_sample_cache(all_labels)

        if bool(get_config_value(self.training_cfg, 'run_decoder', False)):
            self._decode_and_log_prompts()

        if bool(get_config_value(self.training_cfg, 'visualize_attention', False)):
            self._export_attention_overlays(maps_dir)

        if bool(get_config_value(self.training_cfg, 'visualize_gradcam', False)):
            self._export_gradcam_overlays(maps_dir)

        epoch_time = time.time() - start_time
        self.metrics.append({
            'round': round_idx,
            'epoch_in_round': epoch_in_round,
            'epoch_global': self.global_epoch,
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': epoch_time
        })

        if self.val_loader is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.trainer.save_model(self.best_model_path)

        val_loss_display = f"{val_loss:.4f}" if self.val_loader is not None else "N/A"
        val_acc_display = f"{val_acc:.2f}%" if self.val_loader is not None else "N/A"
        epoch_str = (
            f"Epoch {self.global_epoch} (round {round_idx}/{self.rounds}) - " # type: ignore
            f"loss={avg_loss:.4f} - train_acc={avg_acc:.2f}% - "
            f"val_loss={val_loss_display} - val_acc={val_acc_display} - time={epoch_time:.2f}s"
        )
        print(epoch_str)
        with open(self.log_file, 'a') as f:
            f.write(epoch_str + '\n')

        if self.trainer.scheduler is not None:
            self.trainer.scheduler.step()

    def _refresh_sample_cache(self, all_labels):
        if self.dataset is None:
            return
        if self.val_loader is None or len(self.val_indices) == 0:
            return

        num_display = min(10, len(self.classnames), len(self.val_indices))
        selected_indices = []
        seen_classes = set()
        for idx in self.val_indices:
            cls_idx = self.dataset.samples[idx][1]
            if cls_idx not in seen_classes:
                seen_classes.add(cls_idx)
                selected_indices.append(idx)
            if len(selected_indices) >= num_display:
                break

        if len(selected_indices) == 0:
            try:
                batch_data = next(iter(self.val_loader))
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    self.sample_cache['images'] = batch_data[0]
                    self.sample_cache['labels'] = batch_data[1]
                    batch_indices = self.val_indices[:len(batch_data[0])]
                    self.sample_cache['paths'] = [os.path.abspath(self.dataset.samples[idx][0]) for idx in batch_indices]
                else:
                    self.sample_cache['images'] = batch_data
                    self.sample_cache['labels'] = None
                    self.sample_cache['paths'] = []
            except StopIteration:
                self.sample_cache['images'] = None
                self.sample_cache['labels'] = None
                self.sample_cache['paths'] = []
        else:
            sample_images_list = []
            sample_labels_list = []
            sample_paths = []
            for idx in selected_indices:
                img, lbl = self.dataset[idx]
                sample_images_list.append(img)
                sample_labels_list.append(lbl)
                sample_paths.append(os.path.abspath(self.dataset.samples[idx][0]))

            self.sample_cache['images'] = torch.stack(sample_images_list)
            self.sample_cache['labels'] = torch.tensor(sample_labels_list)
            self.sample_cache['paths'] = sample_paths

    def _decode_and_log_prompts(self):
        if self.trainer is None:
            return
        images = self.sample_cache['images']
        if images is None:
            return

        decoded_prompts = self.trainer.decode_adapted_prompts(images, entry_length=30, temperature=1.0)
        self.sample_cache['decoded_prompts'] = decoded_prompts
        log_decoded_prompts(decoded_prompts, self.sample_cache, self.log_file, self.global_epoch)

    def _export_attention_overlays(self, maps_dir):
        if self.trainer is None:
            return
        visualize_attention_maps(
            self.trainer,
            self.dataset,
            self.sample_cache,
            self.classnames,
            self.global_epoch,
            maps_dir,
            self.log_file,
        )

    def _export_gradcam_overlays(self, maps_dir):
        if self.trainer is None:
            return
        visualize_gradcam_maps(
            self.trainer,
            self.dataset,
            self.sample_cache,
            self.classnames,
            self.global_epoch,
            maps_dir,
            self.log_file,
        )

    def _finalize(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before finalization.")
        
        with open(self.config_path, 'w') as f:
            json.dump(self.trainer_cfg.to_dict(), f, indent=4)

        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        self.trainer.save_model(self.last_model_path)

        completion_msg = f"Training completed. Results written to {self.run_dir}"
        print(completion_msg)
        with open(self.log_file, 'a') as f:
            f.write(completion_msg + '\n')

def parse_args():
    parser = create_argument_parser("Train APT model", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides

def main():
    args, overrides = parse_args()
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = APTTrainingPipeline(merged)
    pipeline.run()

if __name__ == "__main__":
    main()