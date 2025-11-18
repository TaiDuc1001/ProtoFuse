import os
import json
import time
import math
import copy
import random
import datetime
import argparse
import itertools
import warnings
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD, Adam, AdamW
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
from thop import profile
from clip import clip
from decode import APTDecoder

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

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


def _group_indices_by_class(dataset, indices):
    grouped = defaultdict(list)
    for idx in indices:
        _, class_id = dataset.samples[idx]
        grouped[class_id].append(idx)
    return grouped


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
        for batch_data in loader:
            images, labels = batch_data
            images = images.to(trainer.device)
            
            logits = trainer.model(images)
            
            probs = torch.softmax(logits, dim=1) + eps
            entropy = -(probs * torch.log(probs)).sum(dim=1)
            
            for i, entropy_val in enumerate(entropy):
                original_idx = indices[position + i]
                class_id = labels[i].item()
                entropy_per_class[class_id].append((original_idx, entropy_val.item()))
            
            position += len(images)

    return entropy_per_class


def select_high_entropy_indices(entropy_per_class, nshot):
    if nshot <= 0:
        return []

    selected = []
    for class_id in sorted(entropy_per_class.keys()):
        class_entropies = entropy_per_class[class_id]
        class_entropies.sort(key=lambda x: x[1], reverse=True)
        class_selected = [idx for idx, _ in class_entropies[:nshot]]
        selected.extend(class_selected)

    return selected


def select_random_indices(dataset, indices, nshot, seed=None):
    if nshot <= 0 or not indices:
        return []

    rng = random.Random(seed)
    grouped = _group_indices_by_class(dataset, indices)
    selected = []

    for class_id in sorted(grouped.keys()):
        class_indices = grouped[class_id]
        
        if len(class_indices) <= nshot:
            selected.extend(class_indices)
        else:
            selected.extend(rng.sample(class_indices, nshot))

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
        trainer.model.cfg['mode'] = 'feature'

    with torch.no_grad():
        for batch_data in loader:
            images, labels = batch_data
            images = images.to(trainer.device)
            
            if hasattr(trainer.model, 'image_encoder'):
                features = trainer.model.image_encoder(images)
            else:
                features = trainer.model(images)
            
            if isinstance(features, tuple):
                features = features[0]
            
            features = features.view(features.size(0), -1)
            
            if torch.isnan(features).any() or torch.isinf(features).any():
                print(f"Warning: Invalid features detected in batch starting at position {position}")
                features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            features = F.normalize(features, p=2, dim=1)
            
            for i, feature_vector in enumerate(features):
                original_idx = indices[position + i]
                embeddings[original_idx] = feature_vector.cpu()
            
            position += len(images)

    if isinstance(trainer.model.cfg, dict):
        if old_mode is not None:
            trainer.model.cfg['mode'] = old_mode
        else:
            trainer.model.cfg.pop('mode', None)

    trainer.model.train(original_training_state)

    return embeddings


def _coreset_greedy_selection(candidates, centers, embeddings, k):
    if k <= 0 or not candidates:
        return []

    candidate_pool = list(candidates)
    selected = []

    center_vectors = [embeddings[idx] for idx in centers if idx in embeddings]

    if not center_vectors:
        if candidate_pool:
            selected.append(candidate_pool.pop(0))
            if len(selected) >= k:
                return selected
        center_vectors = [embeddings[selected[0]]]

    center_matrix = torch.stack(center_vectors).to(torch.float32)

    while candidate_pool and len(selected) < k:
        max_min_distance = -1
        best_candidate = None
        
        for candidate in candidate_pool:
            candidate_vec = embeddings[candidate].unsqueeze(0).to(torch.float32)
            distances = torch.cdist(candidate_vec, center_matrix).min()
            
            if distances > max_min_distance:
                max_min_distance = distances
                best_candidate = candidate
        
        if best_candidate is not None:
            selected.append(best_candidate)
            candidate_pool.remove(best_candidate)
            
            new_center = embeddings[best_candidate].unsqueeze(0).to(torch.float32)
            center_matrix = torch.cat([center_matrix, new_center], dim=0)

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
        unlabeled_class = grouped_unlabeled[class_id]
        labeled_class = grouped_labeled.get(class_id, [])
        
        class_selected = _coreset_greedy_selection(unlabeled_class, labeled_class, embeddings, nshot)
        selected.extend(class_selected)

    return selected


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, unpooled, text_features):
        text_features = text_features.unsqueeze(1).expand(-1, unpooled.size(1), -1)
        attn_output, _ = self.multihead_attn(text_features, unpooled, unpooled)
        attn_output = self.norm1(attn_output + text_features)
        output = self.norm2(attn_output + self.dropout(attn_output))
        return output.mean(dim=1)


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj
        
        self.input_resolution = clip_model.visual.input_resolution
        self.output_dim = clip_model.visual.output_dim

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def encode_text_tokens(self, text):
        x = text + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        
        return x

    def cls_from_tokens(self, tokens, text):
        x = tokens[torch.arange(tokens.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_text(self, text):
        tokens = self.encode_text_tokens(text)
        features = self.cls_from_tokens(tokens, text)
        return features


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.n_ctx = cfg.get('n_ctx', 16)
        self.dtype = clip_model.dtype
        self.device = device
        
        self.image_encoder = ImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.token_embedding = clip_model.token_embedding
        
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        if cfg.get('join_start_embeddings', False):
            self.join_attention = nn.MultiheadAttention(
                ctx_dim, cfg.get('join_num_heads', 8), 
                dropout=cfg.get('join_dropout', 0.1), 
                batch_first=True
            )
            self.join_norm = nn.LayerNorm(ctx_dim)
        else:
            self.join_attention = None
            self.join_norm = None
        
        ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        self.meta_net = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(ctx_dim // 2, ctx_dim)
        )
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(clip.tokenize(name)[0]) for name in classnames]
        
        prompts = [cfg.get('template', CUSTOM_TEMPLATES.get(cfg['dataset_name'], 'a photo of a {}')).format(name) for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])
        
        self.n_cls = len(classnames)
        self.n_ctx = self.n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = "end"
        
        self._init_text_feats(cfg, classnames)

    def _init_text_feats(self, cfg, classnames):
        temp = cfg.get('template', CUSTOM_TEMPLATES.get(cfg['dataset_name'], 'a photo of a {}'))
        
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        
        with torch.no_grad():
            text_features = self.text_encoder.encode_text(prompts)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def update_text_features(self, tuned_embeddings, add_to_start=False, join=None):
        if tuned_embeddings is None:
            return
        
        tuned_embeddings = tuned_embeddings.to(self.device)
        
        if add_to_start:
            self.text_features = self.text_features + tuned_embeddings
        elif join == 'attention' and self.join_attention is not None and self.join_norm is not None:
            combined_features = torch.stack([self.text_features, tuned_embeddings], dim=1)
            attended_features, _ = self.join_attention(combined_features, combined_features, combined_features)
            self.text_features = self.join_norm(attended_features.mean(dim=1))
        else:
            self.text_features = tuned_embeddings
        
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self._prepare_text_features()
        
        x = prompts + self.ctx.unsqueeze(0).expand(prompts.shape[0], -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        if not isinstance(prefix, torch.Tensor):
            prefix = torch.as_tensor(prefix)
        if not isinstance(suffix, torch.Tensor):
            suffix = torch.as_tensor(suffix)
        prefix = prefix.to(x.dtype).to(x.device)
        suffix = suffix.to(x.dtype).to(x.device)
        x = torch.cat((prefix, x, suffix), dim=1)

        text_features = self.text_encoder.encode_text_tokens(x)
        text_features = text_features[torch.arange(text_features.shape[0]), tokenized_prompts.argmax(dim=-1)]
        
        if hasattr(self, 'text_projection') and self.text_projection is not None:
            text_features = text_features @ self.text_projection

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()
        
        if self.training and label is not None:
            return logits
        else:
            return logits

    def _prompt_layers_iter(self):
        yield self.ctx

    def _prepare_text_features(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return ctx

    def _apply_join_attention(self, base_features, tuned_features):
        if self.join_attention is None or self.join_norm is None:
            return tuned_features
        
        combined = torch.stack([base_features, tuned_features], dim=1)
        attended, _ = self.join_attention(combined, combined, combined)
        return self.join_norm(attended.mean(dim=1))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_trainable_parameter_names(self):
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def _reshape_attn_map(self, attn_map, batch_size):
        num_patches = int((attn_map.size(-1) - 1) ** 0.5)
        attn_map = attn_map[:, :, 1:].reshape(batch_size, -1, num_patches, num_patches)
        return attn_map


class APT:
    def __init__(self, cfg, classnames, device="cuda", log_file=None):
        self.cfg = cfg
        self.classnames = classnames
        self.device = device
        self.log_file = log_file
        
        self.model: Optional[CustomCLIP] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.scaler: Optional[GradScaler] = None
        
        self.build_model()
        self.setup_optimizer()
        
        if cfg.get('precision') == 'fp16':
            self.scaler = GradScaler()
        
        self.activations = {}
        self.gradients = {}
    
    def build_model(self):
        print("Loading CLIP model")
        clip_model = load_clip_to_cpu(self.cfg['backbone'])
        clip_model.to(self.device)
        
        print("Building custom CLIP")
        self.model = CustomCLIP(self.cfg, self.classnames, clip_model, self.device)
        
        if self.model is None:
            raise RuntimeError("Failed to build model")
        
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name or "ctx" in name or "meta_net" in name:
                param.requires_grad_(True)
        
        if hasattr(self.model, 'join_attention') and self.model.join_attention is not None:
            for param in self.model.join_attention.parameters():
                param.requires_grad_(True)
            if self.model.join_norm is not None:
                for param in self.model.join_norm.parameters():
                    param.requires_grad_(True)
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        
        if self.cfg.get('precision') == 'fp16':
            self.model.half()
        
        self.model.to(self.device)
    
    def setup_optimizer(self):
        if self.model is None:
            raise RuntimeError("Model must be built before setting up optimizer")
            
        lr = self.cfg.get('learning_rate', 0.002)
        weight_decay = self.cfg.get('weight_decay', 0.0005)
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.cfg.get('optimizer', 'SGD') == 'SGD':
            self.optimizer = SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif self.cfg.get('optimizer') == 'Adam':
            self.optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        elif self.cfg.get('optimizer') == 'AdamW':
            self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        
        num_epochs = self.cfg.get('num_epochs', 50)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
    
    def reset_optimizer_scheduler(self):
        self.setup_optimizer()
    
    def train_step(self, batch):
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model and optimizer must be initialized")
            
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        
        if self.cfg.get('precision') == 'fp16' and self.scaler is not None:
            with autocast():
                logits = self.model(images, labels)
                loss = F.cross_entropy(logits, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.model(images, labels)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean()
        
        return loss.item(), acc.item()
    
    def evaluate(self, dataloader):
        if self.model is None:
            raise RuntimeError("Model must be initialized")
            
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)
                
                pred = logits.argmax(dim=1)
                correct = (pred == labels).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += labels.size(0)
                
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        self.model.train()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def predict(self, images, return_features=False):
        if self.model is None:
            raise RuntimeError("Model must be initialized")
            
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            logits = self.model(images)
            
            if return_features:
                features = self.model.image_encoder(images)
                return logits, features
            else:
                return logits

    def compute_average_text_embeddings(self, dataloader):
        if self.model is None:
            raise RuntimeError("Model must be initialized")
            
        self.model.eval()
        class_embeddings = defaultdict(list)
        
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)
                
                image_features = self.model.image_encoder(images.type(self.model.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                for i, label in enumerate(labels):
                    class_embeddings[label.item()].append(image_features[i])
        
        averaged_embeddings = torch.zeros(len(self.classnames), image_features.size(-1), device=self.device)
        
        for class_id, embeddings in class_embeddings.items():
            if embeddings:
                stacked = torch.stack(embeddings)
                averaged = stacked.mean(dim=0)
                averaged_embeddings[class_id] = averaged / averaged.norm()
        
        self.model.train()
        return averaged_embeddings
    
    def save_model(self, path):
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model and optimizer must be initialized")
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'cfg': self.cfg,
            'classnames': self.classnames
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_model(self, path):
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model and optimizer must be initialized")
            
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    @classmethod
    def from_checkpoint(cls, path, device="cuda"):
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint['cfg']
        classnames = checkpoint['classnames']
        
        trainer = cls(cfg, classnames, device)
        trainer.load_model(path)
        
        return trainer
    
    def forward_uq(self, images, num_samples=10):
        if self.model is None:
            raise RuntimeError("Model must be initialized")
            
        self.model.eval()
        predictions = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                logits = self.model(images)
                predictions.append(torch.softmax(logits, dim=1))
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=1)
        
        return mean_pred, uncertainty

    def save_activation(self, module, input, output):
        self.activations['last_conv'] = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients['last_conv'] = grad_output[0].detach()

    def generate_gradcam(self, images, target_classes):
        if self.model is None:
            raise RuntimeError("Model must be initialized")
            
        self.model.eval()
        
        if hasattr(self.model.image_encoder, 'transformer'):
            hook_layer = self.model.image_encoder.transformer.resblocks[-1]
        else:
            return None
        
        images = images.to(self.device)
        images.requires_grad_()
        
        handle_forward = hook_layer.register_forward_hook(self.save_activation)
        handle_backward = hook_layer.register_backward_hook(self.save_gradient)
        
        try:
            logits = self.model(images)
            
            gradcam_results = []
            
            for i, target_class in enumerate(target_classes):
                if self.model is not None:
                    self.model.zero_grad()
                
                class_logits = logits[i, target_class]
                class_logits.backward(retain_graph=True)
                
                if 'last_conv' in self.activations and 'last_conv' in self.gradients:
                    activations = self.activations['last_conv'][i:i+1]
                    gradients = self.gradients['last_conv'][i:i+1]
                    
                    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
                    gradcam = torch.sum(weights * activations, dim=1, keepdim=True)
                    gradcam = F.relu(gradcam)
                    gradcam = F.interpolate(gradcam, size=(224, 224), mode='bilinear', align_corners=False)
                    
                    gradcam = gradcam.squeeze().cpu().numpy()
                    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
                    
                    gradcam_results.append(gradcam)
                else:
                    gradcam_results.append(None)
        
        finally:
            handle_forward.remove()
            handle_backward.remove()
        
        return gradcam_results

    def decode_adapted_prompts(self, images, entry_length=30, temperature=1.0, batch_decode_size=32):
        if not self.cfg.get('run_decoder', False):
            return []
        
        if self.model is None:
            return []
            
        try:
            decoder = APTDecoder()
            
            with torch.no_grad():
                image_features = self.model.image_encoder(images)
                
                captions = []
                for i in range(0, len(image_features), batch_decode_size):
                    batch_features = image_features[i:i+batch_decode_size]
                    captions.extend([f"Feature_{j}" for j in range(len(batch_features))])
                
                return captions
        
        except Exception as e:
            print(f"Error in decoding: {e}")
            return []


def setup_data_splits(dataset, kshot, val_fraction, seed=42):
    samples_by_class_idx = defaultdict(list)
    for i, (path, class_idx) in enumerate(dataset.samples):
        samples_by_class_idx[class_idx].append(i)

    rng = random.Random(seed)
    val_indices = []
    labeled_indices = []
    unlabeled_indices = []

    for class_idx in sorted(samples_by_class_idx.keys()):
        class_samples = samples_by_class_idx[class_idx]
        rng.shuffle(class_samples)
        
        class_val_size = max(1, int(len(class_samples) * val_fraction))
        class_val_indices = class_samples[:class_val_size]
        remaining_samples = class_samples[class_val_size:]
        
        class_labeled_size = min(kshot, len(remaining_samples))
        class_labeled_indices = remaining_samples[:class_labeled_size]
        class_unlabeled_indices = remaining_samples[class_labeled_size:]
        
        val_indices.extend(class_val_indices)
        labeled_indices.extend(class_labeled_indices)
        unlabeled_indices.extend(class_unlabeled_indices)

    return labeled_indices, unlabeled_indices, val_indices


def run_active_learning_round(trainer, dataset, transform, labeled_indices, unlabeled_indices, 
                            val_indices, cfg, round_num, run_dir):
    batch_size = cfg.get('batch_size', 32)
    num_workers = 4
    num_epochs = cfg.get('num_epochs', 50)
    
    if cfg.get('incr_epochs', 0) > 0:
        num_epochs += round_num * cfg.get('incr_epochs', 0)
    
    train_subset = Subset(dataset, labeled_indices)
    val_subset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Round {round_num}: Training with {len(labeled_indices)} labeled samples")
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        trainer.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for batch in train_loader:
            loss, acc = trainer.train_step(batch)
            epoch_loss += loss
            epoch_acc += acc
        
        if trainer.scheduler:
            trainer.scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            val_results = trainer.evaluate(val_loader)
            val_acc = val_results['accuracy']
            
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, Val_Acc={val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                model_path = os.path.join(run_dir, f'best_model_round_{round_num}.pth')
                trainer.save_model(model_path)
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_val_acc


def select_new_samples(trainer, dataset, labeled_indices, unlabeled_indices, cfg):
    strategy = cfg.get('active_learning')
    nshot = cfg.get('nshot', 0)
    batch_size = cfg.get('batch_size', 32)
    num_workers = 4
    
    if strategy == 'entropy':
        return select_high_entropy_indices(
            compute_entropy_scores(trainer, dataset, unlabeled_indices, batch_size, num_workers),
            nshot
        )
    elif strategy == 'random':
        return select_random_indices(dataset, unlabeled_indices, nshot, seed=None)
    elif strategy == 'coreset':
        return select_coreset_indices(
            trainer, dataset, labeled_indices, unlabeled_indices, nshot, batch_size, num_workers
        )
    else:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset_root', type=str, default='./datasets/cub-200-2011-renamed')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--kshot', type=int, default=16)
    parser.add_argument('--backbone', type=str, default='ViT-B/32')
    parser.add_argument('--dataset_name', type=str, default='CUBirds')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--precision', type=str, default='fp32')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--mode', type=str, default='logits')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--run_decoder', action='store_true')
    parser.add_argument('--visualize_attention', action='store_true')
    parser.add_argument('--visualize_gradcam', action='store_true')
    parser.add_argument('--vis_dir', type=str, default='attention_maps')
    parser.add_argument('--use_cutout', action='store_true')
    parser.add_argument('--confusion_matrix', action='store_true')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--active_learning', type=str, default=None, choices=['entropy', 'random', 'coreset'])
    parser.add_argument('--nshot', type=int, default=0)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--init_per_round', action='store_true')
    parser.add_argument('--incr_epochs', type=int, default=0)
    parser.add_argument('--use_last_tuned_embeddings', action='store_true')
    parser.add_argument('--add_to_start_embeddings', action='store_true')
    parser.add_argument('--join_start_embeddings', action='store_true')
    parser.add_argument('--join_num_heads', type=int, default=8)
    parser.add_argument('--join_dropout', type=float, default=0.1)
    parser.add_argument('--reset_optimizer_per_round', action='store_true')

    args = parser.parse_args()

    if args.add_to_start_embeddings and args.join_start_embeddings:
        raise ValueError("Cannot use both add_to_start_embeddings and join_start_embeddings simultaneously")

    val_fraction = args.val_size
    if val_fraction > 1.0:
        val_fraction = val_fraction / 100.0
    if val_fraction < 0 or val_fraction >= 1.0:
        raise ValueError("val_size must be between 0 and 1 (or 0-100 if percentage)")

    cfg = {
        'backbone': args.backbone,
        'dataset_name': args.dataset_name,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'precision': args.precision,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'mode': args.mode,
        'run_decoder': args.run_decoder,
        'visualize_attention': args.visualize_attention,
        'visualize_gradcam': args.visualize_gradcam,
        'use_cutout': args.use_cutout,
        'generate_confusion_matrix': args.confusion_matrix,
        'optimizer': args.optimizer,
        'dataset_root': args.dataset_root,
        'active_learning': args.active_learning,
        'nshot': args.nshot,
        'val_size': val_fraction,
        'rounds': max(1, args.rounds),
        'initial_kshot': args.kshot,
        'init_per_round': args.init_per_round,
        'use_last_tuned_embeddings': args.use_last_tuned_embeddings,
        'add_to_start_embeddings': args.add_to_start_embeddings,
        'join_start_embeddings': args.join_start_embeddings,
        'join_num_heads': args.join_num_heads,
        'join_dropout': args.join_dropout,
        'reset_optimizer_per_round': args.reset_optimizer_per_round,
        'batch_size': args.batch_size,
    }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    run_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir)

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ]

    if cfg.get('use_cutout', False):
        base_transforms.insert(-1, transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)))

    transform = transforms.Compose(base_transforms)

    try:
        dataset = ImageFolder(args.dataset_root, transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    labeled_indices, unlabeled_indices, val_indices = setup_data_splits(
        dataset, args.kshot, val_fraction, seed=42
    )

    classnames = dataset.classes
    cfg['classnames'] = classnames
    cfg['num_classes'] = len(classnames)

    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    trainer = APT(cfg, classnames, device=args.device)
    
    tuned_embeddings = None
    
    for round_num in range(cfg['rounds']):
        print(f"\n=== Active Learning Round {round_num + 1}/{cfg['rounds']} ===")
        
        if round_num > 0:
            if cfg.get('init_per_round', False):
                trainer = APT(cfg, classnames, device=args.device)
            elif cfg.get('reset_optimizer_per_round', False):
                trainer.reset_optimizer_scheduler()
            
            if tuned_embeddings is not None and cfg.get('use_last_tuned_embeddings', False) and trainer.model is not None:
                if cfg.get('add_to_start_embeddings', False):
                    trainer.model.update_text_features(tuned_embeddings, add_to_start=True)
                elif cfg.get('join_start_embeddings', False):
                    trainer.model.update_text_features(tuned_embeddings, join='attention')
                else:
                    trainer.model.update_text_features(tuned_embeddings)
        
        best_acc = run_active_learning_round(
            trainer, dataset, transform, labeled_indices, unlabeled_indices, 
            val_indices, cfg, round_num + 1, run_dir
        )
        
        print(f"Round {round_num + 1} completed with best validation accuracy: {best_acc:.4f}")
        
        if round_num < cfg['rounds'] - 1 and cfg.get('active_learning'):
            if cfg.get('use_last_tuned_embeddings', False):
                val_subset = Subset(dataset, val_indices)
                val_loader = DataLoader(val_subset, batch_size=cfg.get('batch_size', 32), shuffle=False)
                tuned_embeddings = trainer.compute_average_text_embeddings(val_loader)
            
            new_samples = select_new_samples(trainer, dataset, labeled_indices, unlabeled_indices, cfg)
            
            if new_samples:
                labeled_indices.extend(new_samples)
                unlabeled_indices = [idx for idx in unlabeled_indices if idx not in new_samples]
                print(f"Added {len(new_samples)} new samples. Total labeled: {len(labeled_indices)}")
            else:
                print("No new samples selected. Stopping active learning.")
                break

    print(f"\nTraining completed! Results saved in: {run_dir}")


if __name__ == "__main__":
    main()