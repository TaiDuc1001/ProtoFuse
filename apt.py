import time
import math
import torch
import random
import torch.nn as nn
from clip import clip
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from collections import defaultdict, Counter
import argparse
import datetime
import json
import os
from decode import APTDecoder
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import copy
from thop import profile
from typing import Any, Dict, List, Optional
from sklearn.metrics import confusion_matrix

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

def generate_confusion_matrix_plot(args):
    cm, row_idx, col_idx, start_row, start_col, end_row, end_col, epoch, cm_dir = args
    sub_cm = cm[start_row:end_row, start_col:end_col]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                xticklabels=[str(j) for j in range(start_col, end_col)], 
                yticklabels=[str(j) for j in range(start_row, end_row)],
                annot_kws={"size": 6})
    ax.set_title(f'Confusion Matrix - Epoch {epoch} (True: {start_row}-{end_row-1}, Pred: {start_col}-{end_col-1})', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=8)
    ax.set_ylabel('True Label', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, f'confusion_matrix_r{row_idx:02d}_c{col_idx:02d}.pdf'), dpi=100, bbox_inches='tight')
    plt.close()

def compute_conflict_scores_cache(trainer, dataset, indices, batch_size, num_workers):
    if not indices:
        return defaultdict(list)

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    conflict_per_class = defaultdict(list)
    position = 0
    trainer.model.eval()

    with torch.no_grad():
        for images, labels in loader:
            batch_size_local = images.size(0)
            batch_indices = indices[position:position + batch_size_local]
            position += batch_size_local
            images = images.to(trainer.device)

            # 1. APT Logits (Text Knowledge)
            # Note: Using original mode='logits'
            apt_logits = trainer.model(images)
            if isinstance(apt_logits, (list, tuple)): apt_logits = apt_logits[0]
            prob_apt = torch.softmax(apt_logits, dim=1)

            # 2. Cache Logits (Image Experience)
            # Extract frozen image features
            visual_out = trainer.model.vis_encoder(images)
            img_feats = visual_out[1]
            img_feats = F.normalize(img_feats, dim=-1)
            
            cache_logits_only = trainer.cache_adapter.get_cache_logits(img_feats)
            prob_cache = torch.softmax(cache_logits_only, dim=1)

            # KL Divergence: APT || Cache
            # High KL means APT disagrees with Cache -> Hard/Interesting Sample
            kl_div = F.kl_div(prob_apt.log(), prob_cache, reduction='none', log_target=False).sum(dim=1)

            for score, lbl, idx in zip(kl_div.cpu().tolist(), labels.cpu().tolist(), batch_indices):
                conflict_per_class[int(lbl)].append((float(score), int(idx)))
    
    return conflict_per_class

def select_global_topk_indices(conflict_per_class, nshot):
    if nshot <= 0: return []
    
    # Flatten all candidates into a single list (score, idx)
    all_candidates = []
    for class_id, scores in conflict_per_class.items():
        all_candidates.extend(scores)
    
    if not all_candidates:
        return []
        
    # Sort globally by score (descending) - Winner takes all strategy
    # We select total_k = nshot * num_classes to match the budget
    # If nshot is interpreted as 'per class budget', the total is nshot * num_classes
    # If nshot is 'total budget', then k = nshot.
    # Based on previous code, 'nshot' seemed to be per-class in random/entropy (iterating classes).
    # But for Global Top-K, we just pick the top N regardless of class.
    # Let's assume nshot is passed as the TOTAL budget for the round if strategy is global.
    # However, standard AL usually defines 'nshot' as 'samples to add'.
    # Let's calculate K based on the dictionary length to be safe, assuming nshot is per class average budget.
    # K = nshot * len(conflict_per_class.keys())
    
    k = nshot * len(conflict_per_class) # Total budget
    
    sorted_candidates = sorted(all_candidates, key=lambda item: item[0], reverse=True)
    selected = [idx for _, idx in sorted_candidates[:k]]
    
    return selected

def compute_entropy_scores(trainer, dataset, indices, batch_size, num_workers):
    if not indices:
        return defaultdict(list)

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    entropy_per_class = defaultdict(list)
    position = 0
    eps = 1e-12

    trainer.model.eval()

    with torch.no_grad():
        for images, labels in loader:
            batch_size_local = images.size(0)
            batch_indices = indices[position:position + batch_size_local]
            position += batch_size_local

            images = images.to(trainer.device)
            logits = trainer.model(images)

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + eps)).sum(dim=1)

            for ent, lbl, idx in zip(entropy.cpu().tolist(), labels.cpu().tolist(), batch_indices):
                entropy_per_class[int(lbl)].append((float(ent), int(idx)))

    return entropy_per_class

def select_high_entropy_indices(entropy_per_class, nshot):
    if nshot <= 0:
        return []

    selected = []
    for class_id in sorted(entropy_per_class.keys()):
        scores = entropy_per_class[class_id]
        if not scores:
            continue
        sorted_scores = sorted(scores, key=lambda item: item[0], reverse=True)
        selected.extend(idx for _, idx in sorted_scores[:nshot])

    return selected

def _group_indices_by_class(dataset, indices):
    grouped = defaultdict(list)
    for idx in indices:
        _, class_idx = dataset.samples[idx]
        grouped[int(class_idx)].append(int(idx))
    return grouped

def select_random_indices(dataset, indices, nshot, seed=None):
    if nshot <= 0 or not indices:
        return []

    rng = random.Random(seed)
    grouped = _group_indices_by_class(dataset, indices)
    selected = []

    for class_id in sorted(grouped.keys()):
        candidates = grouped[class_id]
        if not candidates:
            continue
        k = min(nshot, len(candidates))
        if k == len(candidates):
            chosen = list(candidates)
        else:
            chosen = rng.sample(candidates, k)
        selected.extend(chosen)

    return selected

def compute_coreset_embeddings(trainer, dataset, indices, batch_size, num_workers):
    if not indices:
        return {}

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    embeddings = {}
    position = 0

    original_training_state = trainer.model.training
    trainer.model.eval()

    old_mode = trainer.model.cfg.get('mode', None) if isinstance(trainer.model.cfg, dict) else None
    if isinstance(trainer.model.cfg, dict):
        trainer.model.cfg['mode'] = 'features'

    with torch.no_grad():
        for images, _ in loader:
            batch_size_local = images.size(0)
            batch_indices = indices[position:position + batch_size_local]
            position += batch_size_local

            images = images.to(trainer.device)

            outputs = trainer.model(images)
            if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                logits, text_features = outputs
            else:
                raise RuntimeError("Model in 'features' mode is expected to return (logits, text_features).")

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            predicted = torch.argmax(logits, dim=1)

            visual_output = trainer.model.vis_encoder(images)
            _, image_features = visual_output

            image_features = image_features.to(torch.float32)
            tuned_features = text_features[torch.arange(text_features.size(0)), predicted].to(torch.float32)

            image_features = F.normalize(image_features, dim=-1)
            tuned_features = F.normalize(tuned_features, dim=-1)

            combined = torch.cat([image_features, tuned_features], dim=-1)

            combined_cpu = combined.detach().cpu().to(torch.float32)

            for idx, vec in zip(batch_indices, combined_cpu):
                embeddings[int(idx)] = vec

    if isinstance(trainer.model.cfg, dict):
        if old_mode is None:
            trainer.model.cfg.pop('mode', None)
        else:
            trainer.model.cfg['mode'] = old_mode

    trainer.model.train(original_training_state)

    return embeddings

def _coreset_greedy_selection(candidates, centers, embeddings, k):
    if k <= 0 or not candidates:
        return []

    candidate_pool = list(candidates)
    selected = []

    center_vectors = [embeddings[idx] for idx in centers if idx in embeddings]

    if not center_vectors:
        candidate_matrix = torch.stack([embeddings[idx] for idx in candidate_pool]).to(torch.float32)
        norms = torch.norm(candidate_matrix, dim=1)
        first_choice = int(torch.argmax(norms).item())
        first_idx = candidate_pool.pop(first_choice)
        selected.append(first_idx)
        center_vectors = [embeddings[first_idx]]

    center_matrix = torch.stack(center_vectors).to(torch.float32)

    while candidate_pool and len(selected) < k:
        candidate_matrix = torch.stack([embeddings[idx] for idx in candidate_pool]).to(torch.float32)
        distances = torch.cdist(candidate_matrix, center_matrix)
        min_distances, _ = torch.min(distances, dim=1)
        next_choice = int(torch.argmax(min_distances).item())
        chosen_idx = candidate_pool.pop(next_choice)
        selected.append(chosen_idx)
        center_matrix = torch.cat([center_matrix, embeddings[chosen_idx].unsqueeze(0).to(torch.float32)], dim=0)

    return selected

def select_coreset_indices(trainer, dataset, labeled_indices, unlabeled_indices, nshot, batch_size, num_workers):
    if nshot <= 0 or not unlabeled_indices:
        return []

    grouped_unlabeled = _group_indices_by_class(dataset, unlabeled_indices)
    grouped_labeled = _group_indices_by_class(dataset, labeled_indices)

    all_needed_indices = set(unlabeled_indices) | set(labeled_indices)
    embeddings = compute_coreset_embeddings(trainer, dataset, list(all_needed_indices), batch_size, num_workers)

    selected = []

    for class_id in sorted(grouped_unlabeled.keys()):
        candidates = grouped_unlabeled[class_id]
        if not candidates:
            continue
        k = min(nshot, len(candidates))
        centers = grouped_labeled.get(class_id, [])
        chosen = _coreset_greedy_selection(candidates, centers, embeddings, k)
        selected.extend(chosen)

    return selected

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
        self.cfg = cfg
        self.device = device

        prompt_dim = self.clip_model.text_projection.shape[1]
        num_heads = cfg.get('num_heads', 8)
        dropout = cfg.get('dropout', 0.1)

        prompt_layers = []
        for _ in range(cfg.get('num_layers', 1)):
            prompt_layers.append(
                CrossAttention(
                    feature_dim=prompt_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
            )
        self.prompt_learner = nn.ModuleList(prompt_layers)

        if cfg.get('precision', 'fp32') == 'fp16':
            self.prompt_learner = self.prompt_learner.half()

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.vis_encoder = ImageEncoder(self.clip_model)
        self.logit_scale = clip_model.logit_scale

        self.text_features, self.prompts, self.text_tokens = self._init_text_feats(cfg, classnames)
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

        mode = self.cfg.get('mode', 'logits')

        if self.training and label is not None:
            loss = F.cross_entropy(logits, label)
            return loss, logits
        elif mode == "logits":
            return logits
        elif mode == "map":
            return logits, attn_maps

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
        self.cfg = cfg
        self.classnames = classnames
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.log_file = log_file
        
        self.gradients = None
        self.activations = None
        
        self.build_model()
        self.setup_optimizer()
        self.scaler = GradScaler() if cfg.get('precision', 'fp32') == 'amp' else None
        
        if cfg.get('run_decoder', False):
            self.decoder = APTDecoder(
                device=str(self.device),
                clip_model_name=cfg.get('backbone', 'ViT-B/32')
            )
        
        if cfg.get('use_cache', False):
            self.cache_adapter = CacheAdapter(
                feature_dim=self.model.vis_encoder.ln_post.normalized_shape[0],
                num_classes=len(classnames),
                alpha=cfg.get('cache_alpha', 1.0),
                beta=cfg.get('cache_beta', 1.0),
                temperature=cfg.get('cache_temperature', 5.0)
            ).to(self.device)
        else:
            self.cache_adapter = None
    
    def build_model(self):
        backbone_name = self.cfg.get('backbone', 'ViT-B/32')
        msg = f"Loading CLIP (backbone: {backbone_name})"
        print(msg)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')
        
        clip_model = load_clip_to_cpu(backbone_name)
        
        if self.cfg.get('precision', 'fp32') in ['fp32', 'amp']:
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
        lr = self.cfg.get('learning_rate', 0.002)
        weight_decay = self.cfg.get('weight_decay', 0.0005)
        optimizer_type = self.cfg.get('optimizer', 'SGD')
        trainable_params = list(self.model.trainable_parameters())
        
        if optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        
        num_epochs = self.cfg.get('num_epochs', 100)
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
        loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=self.cfg.get('num_workers', 4))
        
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
            msg = f"Cache Updated: {keys.shape[0]} samples stored."
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
        
        precision = self.cfg.get('precision', 'fp32')
        
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
            target_class = target_classes[i]
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
            else:
                cam = np.zeros((8, 8))
            
            gradcams.append(cam)
            if target_unpooled.grad is not None:
                target_unpooled.grad.zero_()
        
        for param in self.model.vis_encoder.parameters():
            param.requires_grad_(False)
        self.model.train(original_mode)
        return gradcams

class ActiveLearningPipeline:
    def __init__(self, config):
        self.config = config
        self.model_cfg = get_config_value(config, "model", {}) or {}
        self.training_cfg = get_config_value(config, "training", {}) or {}
        self.data_cfg = get_config_value(config, "data", {}) or {}
        self.active_cfg = get_config_value(config, "active_learning", {}) or {}
        self.logging_cfg = get_config_value(config, "logging", {}) or {}

        device_value = self.training_cfg.get("device", None)
        device_name = coerce_to_str(device_value, "cuda:0", key="training.device")
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")

        rounds_value = self.active_cfg.get("rounds", None)
        self.rounds = max(1, coerce_to_int(rounds_value, 1, key="active_learning.rounds"))

        incr_value = self.training_cfg.get("increment_epochs", None)
        self.incr_epochs = coerce_to_int(incr_value, 0, key="training.increment_epochs")

        batch_value = self.training_cfg.get("batch_size", None)
        self.batch_size = coerce_to_int(batch_value, 8, key="training.batch_size")

        workers_value = self.data_cfg.get("num_workers", None)
        self.num_workers = coerce_to_int(workers_value, 4, key="data.num_workers")

        kshot_value = self.data_cfg.get("kshot", None)
        self.initial_kshot = coerce_to_int(kshot_value, 16, key="data.kshot")

        val_value = self.data_cfg.get("val_size", None)
        self.val_fraction = coerce_to_float(val_value, 0.2, key="data.val_size")
        if self.val_fraction > 1.0:
            self.val_fraction = self.val_fraction / 100.0
        if self.val_fraction < 0 or self.val_fraction >= 1.0:
            raise ValueError("data.val_size must be in [0, 1) or 0-100 range when expressed as percentage.")

        strategy_value = self.active_cfg.get("strategy", None)
        if strategy_value is not None and not isinstance(strategy_value, str):
            raise ValueError("active_learning.strategy must be a string or null.")
        self.strategy = strategy_value
        if self.strategy not in (None, "entropy", "random", "coreset", "conflict"):
            raise ValueError("active_learning.strategy must be one of null, 'entropy', 'random', 'coreset', 'conflict'.")
        nshot_value = self.active_cfg.get("nshot", None)
        self.nshot = coerce_to_int(nshot_value, 0, key="active_learning.nshot")

        dataset_root_value = self.data_cfg.get("root", "./datasets/cub-200-2011-renamed")
        self.dataset_root = coerce_to_str(dataset_root_value, "./datasets/cub-200-2011-renamed", key="data.root")

        seed_value = self.data_cfg.get("seed", None)
        self.seed = coerce_to_int(seed_value, 42, key="data.seed")

        base_output_value = self.logging_cfg.get("output_dir", "outputs")
        base_output = coerce_to_str(base_output_value, "outputs", key="logging.output_dir")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(base_output, timestamp)
        self.selection_log_path = os.path.join(self.run_dir, 'al_selected_paths.log')
        self.config_path = os.path.join(self.run_dir, 'config.json')
        self.metrics_path = os.path.join(self.run_dir, 'metrics.json')
        self.final_prompts_path = os.path.join(self.run_dir, 'final_prompts.json')
        self.best_model_path = os.path.join(self.run_dir, 'best.pt')
        self.last_model_path = os.path.join(self.run_dir, 'last.pt')
        self.log_file = os.path.join(self.run_dir, 'training.log')

        self.clip_mean = get_config_value(self.data_cfg, "clip_mean", [0.48145466, 0.4578275, 0.40821073])
        self.clip_std = get_config_value(self.data_cfg, "clip_std", [0.26862954, 0.26130258, 0.27577711])

        self.dataset: Optional[ImageFolder] = None
        self.val_loader: Optional[DataLoader] = None
        self.classnames: List[str] = []
        self.labeled_indices: List[int] = []
        self.unlabeled_indices: List[int] = []
        self.val_indices: List[int] = []
        self.metrics: List[Dict[str, Any]] = []
        self.best_val_acc = -float('inf')
        self.global_epoch = 0
        self.sample_cache = {
            'images': None, 'labels': None, 'paths': [], 'decoded_prompts': None
        }

        self.trainer: Optional[APT] = None
        self.trainer_cfg: Dict[str, Any] = {}

    def run(self):
        self._prepare_directories()
        self._load_dataset()
        self._split_dataset()
        self._initialize_trainer()
        self._active_learning_loop()
        self._finalize()

    def _prepare_directories(self):
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.selection_log_path, 'w') as f:
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

    def _split_dataset(self):
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before splitting.")
        samples_by_class_idx = defaultdict(list)
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            samples_by_class_idx[class_idx].append(idx)

        rng = random.Random(self.seed)
        val_indices = []
        labeled_indices = []
        unlabeled_indices = []

        for class_idx in sorted(samples_by_class_idx.keys()):
            class_samples = list(samples_by_class_idx[class_idx])
            class_samples.sort()
            rng.shuffle(class_samples)

            val_count = int(math.floor(len(class_samples) * self.val_fraction))
            if self.val_fraction > 0 and val_count == 0 and len(class_samples) > 0:
                val_count = 1

            val_part = class_samples[:val_count]
            remaining = class_samples[val_count:]

            labeled_count = min(len(remaining), self.initial_kshot)
            labeled_part = remaining[:labeled_count]
            unlabeled_part = remaining[labeled_count:]

            val_indices.extend(val_part)
            labeled_indices.extend(labeled_part)
            unlabeled_indices.extend(unlabeled_part)

        self.val_indices = val_indices
        self.labeled_indices = labeled_indices
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
            'labeled_count': len(self.labeled_indices),
            'unlabeled_count': len(self.unlabeled_indices)
        }
        print(f"Dataset loaded: {stats['total_images']} total images.")
        val_percentage = (stats['val_count'] / stats['total_images'] * 100.0) if stats['total_images'] > 0 else 0.0
        print(f"Validation split: {stats['val_count']} images ({val_percentage:.2f}%).")
        print(
            f"Train pool size: {stats['labeled_count'] + stats['unlabeled_count']} images "
            f"({stats['labeled_count']} labeled, {stats['unlabeled_count']} unlabeled)."
        )

        trainer_cfg = self._build_trainer_config(stats, val_percentage)
        with open(self.config_path, 'w') as f:
            json.dump(trainer_cfg, f, indent=4)

        with open(self.log_file, 'w') as f:
            f.write(f"Config: {trainer_cfg}\n\n")
            f.write('=' * 50 + '\n')

    def _build_trainer_config(self, stats, val_percentage):
        optimizer_section = self.training_cfg.get('optimizer', {})
        if isinstance(optimizer_section, dict):
            optimizer_name = optimizer_section.get('name', 'SGD')
            lr_value = optimizer_section.get('learning_rate', self.training_cfg.get('learning_rate', 0.001))
            wd_value = optimizer_section.get('weight_decay', self.training_cfg.get('weight_decay', 0.0005))
        else:
            optimizer_name = optimizer_section if isinstance(optimizer_section, str) else 'SGD'
            lr_value = self.training_cfg.get('learning_rate', 0.001)
            wd_value = self.training_cfg.get('weight_decay', 0.0005)

        learning_rate = coerce_to_float(lr_value, 0.001, key='training.optimizer.learning_rate')
        weight_decay = coerce_to_float(wd_value, 0.0005, key='training.optimizer.weight_decay')
        precision_value = self.training_cfg.get('precision', 'fp32')
        precision = coerce_to_str(precision_value, 'fp32', key='training.precision')
        mode_value = self.training_cfg.get('mode', 'logits')
        mode = coerce_to_str(mode_value, 'logits', key='training.mode')
        epochs_value = self.training_cfg.get('epochs', 150)
        num_epochs = coerce_to_int(epochs_value, 150, key='training.epochs')

        backbone_value = get_config_value(self.model_cfg, 'backbone', 'ViT-B/32')
        backbone = coerce_to_str(backbone_value, 'ViT-B/32', key='model.backbone')
        dataset_name_value = get_config_value(self.model_cfg, 'dataset_name', 'CUBirds')
        dataset_name = coerce_to_str(dataset_name_value, 'CUBirds', key='model.dataset_name')
        num_heads_value = get_config_value(self.model_cfg, 'num_heads', 8)
        num_heads = coerce_to_int(num_heads_value, 8, key='model.num_heads')
        num_layers_value = get_config_value(self.model_cfg, 'num_layers', 1)
        num_layers = coerce_to_int(num_layers_value, 1, key='model.num_layers')
        dropout_value = get_config_value(self.model_cfg, 'dropout', 0.2)
        dropout = coerce_to_float(dropout_value, 0.2, key='model.dropout')

        trainer_cfg = {
            'backbone': backbone,
            'dataset_name': dataset_name,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
            'precision': precision,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'mode': mode,
            'run_decoder': bool(get_config_value(self.training_cfg, 'run_decoder', False)),
            'visualize_attention': bool(get_config_value(self.training_cfg, 'visualize_attention', False)),
            'visualize_gradcam': bool(get_config_value(self.training_cfg, 'visualize_gradcam', False)),
            'use_cutout': bool(get_config_value(self.training_cfg, 'use_cutout', False)),
            'generate_confusion_matrix': bool(get_config_value(self.training_cfg, 'confusion_matrix', False)),
            'optimizer': coerce_to_str(optimizer_name, 'SGD', key='training.optimizer.name'),
            'dataset_root': self.dataset_root,
            'active_learning': self.strategy,
            'nshot': self.nshot,
            'val_size': self.val_fraction,
            'rounds': self.rounds,
            'initial_kshot': self.initial_kshot,
            'classnames': self.classnames,
            'num_classes': len(self.classnames),
            'initial_labeled_size': stats['labeled_count'],
            'initial_unlabeled_size': stats['unlabeled_count'],
            'val_size_count': stats['val_count'],
            'train_pool_size': stats['labeled_count'] + stats['unlabeled_count'],
            'al_selection_log': self.selection_log_path,
            'val_percentage_actual': val_percentage,
            'use_cache': True,
            'cache_alpha': 1.0,
            'cache_beta': 1.0,
            'cache_temperature': 5.5,
            'reset_optimizer_per_round': True
        }
        self.trainer_cfg = trainer_cfg
        return trainer_cfg

    def _initialize_trainer(self):
        if not self.classnames:
            raise RuntimeError("Class names unavailable before trainer initialization.")
        self.trainer = APT(self.trainer_cfg, self.classnames, device=str(self.device), log_file=self.log_file)

    def _active_learning_loop(self):
        print('\n')
        print('=' * 50)

        self.trainer.update_cache_memory(self.dataset, self.labeled_indices) # type: ignore
        
        for round_idx in range(1, self.rounds + 1):
            self._run_round(round_idx)

    def _run_round(self, round_idx):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before running rounds.")
        round_dir = os.path.join(self.run_dir, f'round_{round_idx:02d}')
        os.makedirs(round_dir, exist_ok=True)

        if len(self.labeled_indices) == 0:
            msg = f"Round {round_idx}: no labeled samples available; stopping training."
            print(msg)
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')
            self.trainer_cfg['completed_rounds'] = round_idx - 1
            return

        train_subset = Subset(self.dataset, list(self.labeled_indices))
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        msg = (
            f"Starting round {round_idx}/{self.rounds}: {len(self.labeled_indices)} labeled | "
            f"{len(self.unlabeled_indices)} unlabeled"
        )
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')

        base_epochs_value = self.training_cfg.get('epochs', None)
        base_epochs = coerce_to_int(base_epochs_value, 150, key='training.epochs')
        epochs_this_round = base_epochs + (round_idx - 1) * self.incr_epochs
        
        with open(self.log_file, 'a') as f:
            f.write(f"  Epochs this round: {epochs_this_round} (base: {base_epochs})\n")

        for epoch_in_round in range(1, epochs_this_round + 1):
            self._run_epoch(round_idx, epoch_in_round, epochs_this_round, train_loader, round_dir)

        if self.strategy in ('entropy', 'random', 'coreset', 'conflict') and round_idx < self.rounds:
            self._perform_active_selection(round_idx)
            self.trainer.reset_optimizer_scheduler()
            self.trainer.update_cache_memory(self.dataset, self.labeled_indices)

        self.trainer_cfg['completed_rounds'] = round_idx

    def _perform_active_selection(self, round_idx):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before active selection.")

        strategy = self.strategy
        if strategy not in ('entropy', 'random', 'coreset', 'conflict'):
            return

        if not self.unlabeled_indices:
            skip_msg = (
                f"Active learning selection ({strategy}) (round {round_idx} -> {round_idx + 1}) skipped (no unlabeled samples)."
            )
            print(skip_msg)
            with open(self.log_file, 'a') as f:
                f.write(skip_msg + '\n')
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none" + '\n')
            return

        if self.nshot <= 0:
            no_shot_msg = (
                f"Active learning selection ({strategy}) (round {round_idx} -> {round_idx + 1}) skipped (nshot={self.nshot})."
            )
            print(no_shot_msg)
            with open(self.log_file, 'a') as f:
                f.write(no_shot_msg + '\n')
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none" + '\n')
            return

        raw_selected = []

        if strategy == 'entropy':
            entropy_scores = compute_entropy_scores(
                self.trainer, self.dataset, self.unlabeled_indices, self.batch_size, self.num_workers
            )
            raw_selected = select_high_entropy_indices(entropy_scores, self.nshot)
        elif strategy == 'random':
            seed = self.seed + round_idx
            raw_selected = select_random_indices(
                self.dataset, self.unlabeled_indices, self.nshot, seed=seed
            )
        elif strategy == 'coreset':
            raw_selected = select_coreset_indices(
                self.trainer,
                self.dataset,
                self.labeled_indices,
                self.unlabeled_indices,
                self.nshot,
                self.batch_size,
                self.num_workers
            )
        elif strategy == 'conflict':
            conflict_scores = compute_conflict_scores_cache(
                self.trainer, self.dataset, self.unlabeled_indices, self.batch_size, self.num_workers
            )
            raw_selected = select_global_topk_indices(conflict_scores, self.nshot)

        if not raw_selected:
            empty_msg = (
                f"Active learning selection ({strategy}) (round {round_idx} -> {round_idx + 1}) selected no samples."
            )
            print(empty_msg)
            with open(self.log_file, 'a') as f:
                f.write(empty_msg + '\n')
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none" + '\n')
            return

        unlabeled_set = set(self.unlabeled_indices)
        seen = set()
        selected_indices = []
        for idx in raw_selected:
            if idx in unlabeled_set and idx not in seen:
                seen.add(idx)
                selected_indices.append(idx)

        if not selected_indices:
            duplicate_msg = (
                f"Active learning selection ({strategy}) (round {round_idx} -> {round_idx + 1}): "
                "suggested samples were already labeled."
            )
            print(duplicate_msg)
            with open(self.log_file, 'a') as f:
                f.write(duplicate_msg + '\n')
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none" + '\n')
            return

        existing_labeled = set(self.labeled_indices)
        new_indices = [idx for idx in selected_indices if idx not in existing_labeled]

        if not new_indices:
            no_new_msg = (
                f"Active learning selection ({strategy}) (round {round_idx} -> {round_idx + 1}): "
                "all suggested samples were already labeled."
            )
            print(no_new_msg)
            with open(self.log_file, 'a') as f:
                f.write(no_new_msg + '\n')
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none" + '\n')
            return

        prev_labeled = len(self.labeled_indices)
        new_set = set(new_indices)
        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = [idx for idx in self.unlabeled_indices if idx not in new_set]
        after_labeled = len(self.labeled_indices)

        summary = f"Selected {len(new_indices)} new samples. Labeled: {after_labeled} (was {prev_labeled})."
        print(summary)
        with open(self.log_file, 'a') as f:
            f.write(summary + '\n')
        
        round_selected_paths = [os.path.abspath(self.dataset.samples[idx][0]) for idx in new_indices]
        with open(self.selection_log_path, 'a') as f:
            line = ';'.join(round_selected_paths) if round_selected_paths else f"round {round_idx}: none"
            f.write(line + '\n')

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
            self._generate_confusion_outputs(all_labels, all_preds, epoch_dir)

        if all_labels:
            self._generate_class_distribution(all_labels, all_preds, epoch_dir)

        self._maybe_prepare_samples(all_labels)

        if bool(get_config_value(self.training_cfg, 'run_decoder', False)):
            self._maybe_decode_prompts()

        if bool(get_config_value(self.training_cfg, 'visualize_attention', False)):
            self._maybe_visualize_attention(maps_dir)

        if bool(get_config_value(self.training_cfg, 'visualize_gradcam', False)):
            self._maybe_visualize_gradcam(maps_dir)

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
            f"Epoch {self.global_epoch} (round {round_idx}/{self.rounds}) - "
            f"loss={avg_loss:.4f} - train_acc={avg_acc:.2f}% - "
            f"val_loss={val_loss_display} - val_acc={val_acc_display} - time={epoch_time:.2f}s"
        )
        print(epoch_str)
        with open(self.log_file, 'a') as f:
            f.write(epoch_str + '\n')

        if self.trainer.scheduler is not None:
            self.trainer.scheduler.step()

    def _generate_confusion_outputs(self, all_labels, all_preds, epoch_dir):
        if self.dataset is None:
            raise RuntimeError("Dataset not available for confusion outputs.")
        cm_dir = os.path.join(epoch_dir, 'confusion_matrices')
        os.makedirs(cm_dir, exist_ok=True)

        cm = confusion_matrix(all_labels, all_preds)

        num_classes = cm.shape[0]
        if num_classes > 50:
            block_size = 50
            step = 50
            num_blocks_per_dim = (num_classes - block_size) // step + 1

            plot_args = []
            for row_idx in range(num_blocks_per_dim):
                for col_idx in range(num_blocks_per_dim):
                    start_row = row_idx * step
                    start_col = col_idx * step
                    end_row = start_row + block_size
                    end_col = start_col + block_size
                    plot_args.append((cm, row_idx, col_idx, start_row, start_col, end_row, end_col, self.global_epoch, cm_dir))

            with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
                pool.map(generate_confusion_matrix_plot, plot_args)
        else:
            fig, ax = plt.subplots(figsize=(max(16, num_classes // 2), max(16, num_classes // 2)))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax,
                cbar=True,
                xticklabels=[str(i) for i in range(num_classes)],
                yticklabels=[str(i) for i in range(num_classes)],
                annot_kws={"size": 16}
            )
            ax.set_title(f'Confusion Matrix - Epoch {self.global_epoch}', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(cm_dir, 'confusion_matrix.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

    def _generate_class_distribution(self, all_labels, all_preds, epoch_dir):
        fig, ax = plt.subplots(figsize=(12, 8))
        gt_counts = Counter(all_labels)
        pred_counts = Counter(all_preds)

        classes = sorted(set(gt_counts.keys()) | set(pred_counts.keys()))
        gt_values = [gt_counts.get(cls, 0) for cls in classes]
        pred_values = [pred_counts.get(cls, 0) for cls in classes]

        x = np.arange(len(classes))
        width = 0.35

        bars1 = ax.bar(x - width / 2, gt_values, width, label='Ground Truth', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width / 2, pred_values, width, label='Predictions', color='salmon', alpha=0.8)

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Class Distribution - Epoch {self.global_epoch}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([str(cls) for cls in classes], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        max_height = max(gt_values + pred_values) if (gt_values or pred_values) else 0
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + max_height * 0.01 if max_height > 0 else 0.5,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=8
            )
        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + max_height * 0.01 if max_height > 0 else 0.5,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=8
            )

        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, 'class_distribution.pdf'), dpi=150, bbox_inches='tight')
        plt.close()

    def _maybe_prepare_samples(self, all_labels):
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

    def _maybe_decode_prompts(self):
        if self.trainer is None:
            return
        images = self.sample_cache['images']
        if images is None:
            return

        decoded_prompts = self.trainer.decode_adapted_prompts(images, entry_length=30, temperature=1.0)
        self.sample_cache['decoded_prompts'] = decoded_prompts
        if decoded_prompts is None:
            return

        prompt_str = f"Generated captions from learned prompts for epoch {self.global_epoch}:"
        print(prompt_str)
        with open(self.log_file, 'a') as f:
            f.write(prompt_str + '\n')
            for i in range(len(decoded_prompts)):
                image_prompts = decoded_prompts[i]
                image_path = self.sample_cache['paths'][i] if i < len(self.sample_cache['paths']) else "Unknown path"
                label_tensor = self.sample_cache['labels']
                image_label = None
                if label_tensor is not None:
                    try:
                        image_label = int(label_tensor[i])
                    except Exception:
                        image_label = None

                image_str = f"Image ({image_path}):"
                print(image_str)
                f.write(image_str + '\n')

                selected_prompt = None
                if image_label is not None:
                    for prompt in image_prompts:
                        if prompt.get('class_id') == image_label:
                            selected_prompt = prompt
                            break

                if selected_prompt is None and len(image_prompts) > 0:
                    selected_prompt = image_prompts[0]

                class_id = selected_prompt.get('class_id', 'unknown') if selected_prompt else 'unknown'
                class_name = selected_prompt.get('class_name', f"Class_{class_id}") if selected_prompt else 'unknown'
                caption = selected_prompt.get('generated_caption', 'No caption generated') if selected_prompt else 'No caption generated'
                caption_line = f"  Class {class_id} ({class_name}): {caption}"
                print(caption_line)
                f.write(caption_line + '\n')
            f.write('\n')

    def _maybe_visualize_attention(self, maps_dir):
        if self.trainer is None:
            return
        images = self.sample_cache['images']
        if images is None:
            return

        labels = self.sample_cache['labels']
        self.trainer.model.cfg['mode'] = 'map'

        if isinstance(images, torch.Tensor):
            vis_images = images.to(self.trainer.device)
        elif isinstance(images, (list, tuple)):
            vis_images = torch.stack([
                x.to(self.trainer.device) if isinstance(x, torch.Tensor) else torch.tensor(x).to(self.trainer.device)
                for x in images
            ])
        else:
            vis_images = torch.tensor(images).to(self.trainer.device)

        if labels is not None:
            if isinstance(labels, torch.Tensor):
                vis_labels = labels.to(self.trainer.device)
            else:
                vis_labels = torch.tensor(labels).to(self.trainer.device)
        else:
            vis_labels = None

        logits, attn_maps = self.trainer.model(vis_images)
        self.trainer.model.cfg['mode'] = self.trainer_cfg.get('mode', 'logits')

        attn_map_to_vis = attn_maps[0]
        try:
            shape_info = getattr(attn_map_to_vis, 'shape', None)
            shape_msg = f"Epoch {self.global_epoch} attention map shape: {shape_info}"
            print(shape_msg)
            with open(self.log_file, 'a') as lf:
                lf.write(shape_msg + '\n')
        except Exception:
            pass

        for i in range(len(vis_images)):
            image_path = self.sample_cache['paths'][i]
            if self.dataset is None:
                continue
            if vis_labels is None:
                msg = f"No label for image {i}, skipping attention visualization."
                print(msg)
                with open(self.log_file, 'a') as lf:
                    lf.write(msg + '\n')
                continue

            label = int(vis_labels[i].item())
            try:
                weights = attn_map_to_vis[i, label, :]
            except Exception:
                warn_msg = (
                    f"Warning: unable to index attention map for image {i}, label {label} with expected layout."
                    " Skipping visualization."
                )
                print(warn_msg)
                with open(self.log_file, 'a') as lf:
                    lf.write(warn_msg + '\n')
                continue

            if weights is None:
                warn_msg = f"Warning: unable to index attention map for image {i}, label {label}. Skipping visualization."
                print(warn_msg)
                with open(self.log_file, 'a') as lf:
                    lf.write(warn_msg + '\n')
                continue

            if weights.dim() > 1:
                mean_weights = weights.mean(dim=0).detach().cpu().numpy()
            else:
                mean_weights = weights.detach().cpu().numpy()

            patch_weights = mean_weights[1:]

            stats_msg = (
                f"Epoch {self.global_epoch} image {i} label {label} attention stats: "
                f"mean={mean_weights.mean():.6f} min={mean_weights.min():.6f} max={mean_weights.max():.6f}"
            )
            print(stats_msg)
            with open(self.log_file, 'a') as lf:
                lf.write(stats_msg + '\n')

            num_patches = patch_weights.shape[0]
            h = w = int(np.sqrt(num_patches))
            if h * w != num_patches:
                warn_msg = (
                    f"Warning: Cannot reshape {num_patches} patches into a square grid. Skipping visualization for image {i}."
                )
                print(warn_msg)
                with open(self.log_file, 'a') as lf:
                    lf.write(warn_msg + '\n')
                continue

            heatmap = patch_weights.reshape(h, w)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = (heatmap * 255).astype(np.uint8)

            original_img = cv2.imread(image_path)
            if original_img is None:
                warn_read = f"Warning: unable to read image {image_path}, skipping attention visualization."
                print(warn_read)
                with open(self.log_file, 'a') as lf:
                    lf.write(warn_read + '\n')
                continue
            original_img = cv2.resize(original_img, (224, 224))
            heatmap_img = cv2.applyColorMap(cv2.resize(heatmap, (224, 224)), cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_img, 0.4, 0)

            class_name = self.classnames[label] if label < len(self.classnames) else f"Class_{label}"
            save_name = f"epoch_{self.global_epoch:03d}_img_{i}_class_{label}_{class_name}.jpg"
            save_path = os.path.join(maps_dir, save_name)
            cv2.imwrite(save_path, superimposed_img)

        log_str = f"Saved {len(vis_images)} attention visualizations to {maps_dir}"
        print(log_str)
        with open(self.log_file, 'a') as f:
            f.write(log_str + '\n')

    def _maybe_visualize_gradcam(self, maps_dir):
        if self.trainer is None:
            return
        images = self.sample_cache['images']
        labels = self.sample_cache['labels']
        if images is None or labels is None:
            return

        if isinstance(images, torch.Tensor):
            vis_images = images.to(self.trainer.device)
        elif isinstance(images, (list, tuple)):
            vis_images = torch.stack([
                x.to(self.trainer.device) if isinstance(x, torch.Tensor) else torch.tensor(x).to(self.trainer.device)
                for x in images
            ])
        else:
            vis_images = torch.tensor(images).to(self.trainer.device)

        if isinstance(labels, torch.Tensor):
            vis_labels = labels.to(self.trainer.device)
        else:
            vis_labels = torch.tensor(labels).to(self.trainer.device)

        gradcams = self.trainer.generate_gradcam(vis_images, vis_labels)

        for i, gradcam in enumerate(gradcams):
            image_path = self.sample_cache['paths'][i]
            if self.dataset is None:
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

            class_name = self.classnames[label] if label < len(self.classnames) else f"Class_{label}"
            save_name = f"gradcam_epoch_{self.global_epoch:03d}_img_{i}_class_{label}_{class_name}.jpg"
            save_path = os.path.join(maps_dir, save_name)
            cv2.imwrite(save_path, superimposed_img)

        log_str = f"Saved {len(gradcams)} GradCAM visualizations to {maps_dir}"
        with open(self.log_file, 'a') as f:
            f.write(log_str + '\n')

    def _finalize(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before finalization.")
        
        with open(self.config_path, 'w') as f:
            json.dump(self.trainer_cfg, f, indent=4)

        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        self.trainer.save_model(self.last_model_path)

        completion_msg = f"Training completed. Results written to {self.run_dir}"
        print(completion_msg)
        with open(self.log_file, 'a') as f:
            f.write(completion_msg + '\n')

def parse_args():
    parser = argparse.ArgumentParser(description="Train APT model")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--output_dir', type=str, default=None, help='Override logging.output_dir from config')
    parser.add_argument('--device', type=str, default=None, help='Override training.device from config')
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    if parsed.output_dir is not None:
        set_nested_value(overrides, ['logging', 'output_dir'], parsed.output_dir)
    if parsed.device is not None:
        set_nested_value(overrides, ['training', 'device'], parsed.device)
    return parsed, overrides

def main():
    args, overrides = parse_args()
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = ActiveLearningPipeline(merged)
    pipeline.run()

if __name__ == "__main__":
    main()