import os
import time
import json
import math
import copy
import hashlib
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
from clip import clip
from thop import profile
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import transforms
from collections import defaultdict
from torchvision.datasets import ImageFolder
from typing import Any, Dict, List, Optional
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from utils import (
    logger,
    ConfigNode,
    get_config_value,
    coerce_to_str,
    coerce_to_int,
    coerce_to_float,
    load_clip_to_cpu,
    compute_metrics,
)

DEFAULT_TRAINING_EPOCHS = 100
DEFAULT_CHECKPOINT_DIR = 'checkpoints/apt'


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
        
        self.patch_size = self.conv1.kernel_size[0]
        self._pos_embed_cache = {}

    def _interpolate_pos_embed(self, pos_embed, num_patches):
        expected_patches = pos_embed.shape[0] - 1
        
        if num_patches == expected_patches:
            return pos_embed
        
        cache_key = (num_patches, pos_embed.device)
        if cache_key in self._pos_embed_cache:
            return self._pos_embed_cache[cache_key]
        
        cls_embed = pos_embed[:1]
        patch_embed = pos_embed[1:]
        
        src_size = int(expected_patches ** 0.5)
        tgt_size = int(num_patches ** 0.5)
        
        embed_dim = patch_embed.shape[-1]
        patch_embed = patch_embed.reshape(1, src_size, src_size, embed_dim).permute(0, 3, 1, 2)
        patch_embed = F.interpolate(patch_embed, size=(tgt_size, tgt_size), mode='bicubic', align_corners=False)
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(num_patches, embed_dim)
        
        result = torch.cat([cls_embed, patch_embed], dim=0)
        self._pos_embed_cache[cache_key] = result
        return result

    def forward(self, x):
        x = x.type(self.conv1.weight.dtype)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls_tokens = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tokens, x], dim=1)
        
        num_patches = x.shape[1] - 1
        pos_embed = self._interpolate_pos_embed(self.positional_embedding, num_patches)
        x = x + pos_embed.to(x.dtype)

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
    "StanfordDogs": "a photo of a {}, a type of dog.",
    "Flowers102": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "FGVC-Aircraft": "a photo of a {}, a type of aircraft.",
    "FGVC Aircraft": "a photo of a {}, a type of aircraft.",
    "DTD": "{} texture.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food-101": "a photo of {}, a type of food.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "CUB-200-2011": "a photo of a {}, a type of bird.",
    "V1922_13": "a photo of a {}, a type of military vehicle.",
    "NEU-CLS": "a photo of a {}, a type of defect on steel surface.",
    "MVTEC-CLS": "a photo of a {}, a type of defect on steel strip surface.",
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
        self.posthoc_visual_centroids = None
        self.posthoc_centroid_mask = None
        self.posthoc_alpha = 0.0

    def _init_text_feats(self, cfg, classnames):
        data_cfg = getattr(self.cfg, 'data', ConfigNode())
        dataset_name = data_cfg.get('dataset_name', cfg.get('dataset_name', 'ImageNet'))
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

        if self.posthoc_visual_centroids is not None:
            visual_centroids = self.posthoc_visual_centroids.to(text_features.device, dtype=text_features.dtype)
            fused_text_features = F.normalize(
                (1.0 - self.posthoc_alpha) * text_features
                + self.posthoc_alpha * visual_centroids.unsqueeze(0),
                dim=-1,
            )
            if self.posthoc_centroid_mask is not None:
                centroid_mask = self.posthoc_centroid_mask.to(text_features.device).view(1, -1, 1)
                text_features = torch.where(centroid_mask, fused_text_features, text_features)
            else:
                text_features = fused_text_features
        
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

    def set_posthoc_protofuse(self, visual_centroids, alpha, centroid_mask=None):
        alpha = max(0.0, min(1.0, float(alpha)))
        self.posthoc_visual_centroids = F.normalize(
            visual_centroids.detach().to(self.device).float(),
            dim=-1,
        )
        self.posthoc_alpha = alpha
        if centroid_mask is None:
            centroid_mask = torch.ones(
                self.posthoc_visual_centroids.shape[0],
                device=self.device,
                dtype=torch.bool,
            )
        self.posthoc_centroid_mask = centroid_mask.detach().to(self.device).bool()

    def clear_posthoc_protofuse(self):
        self.posthoc_visual_centroids = None
        self.posthoc_centroid_mask = None
        self.posthoc_alpha = 0.0

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


class APT:
    def __init__(self, cfg, classnames, device="cuda"):
        if not isinstance(cfg, ConfigNode):
            cfg = ConfigNode(cfg)
        self.cfg = cfg
        self.training_cfg = self.cfg.get('training', ConfigNode())
        self.model_cfg = self.cfg.get('model', ConfigNode())
        self.data_cfg = self.cfg.get('data', ConfigNode())
        self.cache_cfg = self.cfg.get('cache', ConfigNode())
        self.classnames = classnames
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.gradients = None
        self.activations = None
        
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
            logger.debug(f"Config path '{path}' not found. Using default: {default}")
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
        # logger.info(f"Loading CLIP (backbone: {backbone_name})")
        
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
        
        # logger.info(f"Learnable parameters: {format_params(learnable_params)} / Total: {format_params(total_params)} (FLOPs: {gflops_thop:.2f} GFLOPs)")
        
        trainable_names = set(self.model.get_trainable_parameter_names())
        for name, param in self.model.named_parameters():
            param.requires_grad_(name in trainable_names)
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)


        self.model.to(self.device)
        self.initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.text_prototypes = None
        self.visual_centroids = None
        self.posthoc_centroid_mask = None
        self.posthoc_alpha = None
        self.posthoc_missing_classes = []
    
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

    def reset_model(self):
        if hasattr(self, 'initial_model_state'):
            self.model.load_state_dict(self.initial_model_state)
            self.model.to(self.device)

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def clear_posthoc_protofuse(self):
        self.visual_centroids = None
        self.posthoc_centroid_mask = None
        self.posthoc_alpha = None
        self.posthoc_missing_classes = []
        self.model.clear_posthoc_protofuse()

    def get_text_prototypes(self):
        with torch.no_grad():
            text_features = self.model._prepare_text_features().float()
        self.text_prototypes = F.normalize(text_features, dim=-1)
        return self.text_prototypes

    def support_text_prototypes(self, text_features, labels):
        text_features = text_features.to(self.device).float()
        labels = labels.to(self.device).long()
        base_text = self.get_text_prototypes()
        prototypes = base_text.clone()
        for class_idx in range(len(self.classnames)):
            mask = labels == class_idx
            if mask.any():
                prototypes[class_idx] = F.normalize(
                    text_features[mask, class_idx, :].mean(dim=0),
                    dim=-1,
                )
        self.text_prototypes = F.normalize(prototypes, dim=-1)
        return self.text_prototypes

    def _raw_image_text_features(self, images):
        visual_output = self.model.vis_encoder(images)
        unpooled_levels, image_features = visual_output
        if not isinstance(unpooled_levels, list):
            unpooled_levels = [unpooled_levels]

        base_text_features = self.model._prepare_text_features()
        unpooled_images = unpooled_levels[0].permute(1, 0, 2)
        text_features = base_text_features.unsqueeze(1).expand(-1, unpooled_images.shape[1], -1)

        for layer in self.model._prompt_layers_iter():
            text_features, _ = layer(unpooled_images, text_features)

        text_features = F.normalize(text_features.permute(1, 0, 2), dim=-1)
        image_features = F.normalize(image_features.float(), dim=-1)
        return image_features, text_features

    def extract_features(self, dataloader):
        all_features = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                image_features, _ = self._raw_image_text_features(images)
                all_features.append(image_features.cpu())
                all_labels.append(labels.cpu())
        if not all_features:
            raise RuntimeError("Cannot extract APT features from an empty dataloader.")
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def extract_posthoc_features(self, dataloader):
        all_image_features = []
        all_text_features = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                image_features, text_features = self._raw_image_text_features(images)
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
                all_labels.append(labels.cpu())
        if not all_image_features:
            raise RuntimeError("Cannot extract APT post-hoc features from an empty dataloader.")
        return (
            torch.cat(all_image_features, dim=0),
            torch.cat(all_text_features, dim=0),
            torch.cat(all_labels, dim=0),
        )

    def build_visual_centroids(self, features, labels, num_classes=None):
        if features.numel() == 0 or labels.numel() == 0:
            raise RuntimeError("Cannot build visual centroids without training samples.")

        num_classes = len(self.classnames) if num_classes is None else num_classes
        features = F.normalize(features.to(self.device).float(), dim=-1)
        labels = labels.to(self.device).long()

        centroids = torch.zeros(num_classes, features.shape[-1], device=self.device, dtype=features.dtype)
        counts = torch.zeros(num_classes, device=self.device, dtype=features.dtype)
        valid_mask = (labels >= 0) & (labels < num_classes)
        if valid_mask.any():
            centroids.index_add_(0, labels[valid_mask], features[valid_mask])
            counts.index_add_(0, labels[valid_mask], torch.ones_like(labels[valid_mask], dtype=features.dtype))

        present = counts > 0
        if present.any():
            centroids[present] = F.normalize(centroids[present] / counts[present].unsqueeze(1), dim=-1)

        self.visual_centroids = centroids
        self.posthoc_centroid_mask = present
        self.posthoc_missing_classes = torch.nonzero(~present, as_tuple=False).flatten().cpu().tolist()
        return centroids

    def _weighted_centroid_from_features(self, class_features, text_prototype):
        class_features = F.normalize(class_features.to(self.device).float(), dim=-1)
        text_prototype = F.normalize(text_prototype.to(self.device).float(), dim=-1)
        similarities = F.cosine_similarity(
            class_features,
            text_prototype.unsqueeze(0),
            dim=-1,
        ).clamp_min(0.0)
        sim_sum = similarities.sum()
        if sim_sum <= 1e-12:
            weights = torch.full_like(similarities, 1.0 / similarities.numel())
        else:
            weights = similarities / sim_sum
        return F.normalize((weights.unsqueeze(-1) * class_features).sum(dim=0), dim=-1)

    def select_posthoc_alpha(
        self,
        train_features,
        train_text_features,
        train_labels,
        alpha_steps=101,
        force_loo_accuracy=False,
    ):
        if self.text_prototypes is None:
            raise RuntimeError("Call support_text_prototypes before selecting APT post-hoc alpha.")

        train_features = F.normalize(train_features.to(self.device).float(), dim=-1)
        train_text_features = F.normalize(train_text_features.to(self.device).float(), dim=-1)
        train_labels = train_labels.to(self.device).long()
        num_classes = len(self.classnames)

        class_indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(train_labels.tolist()):
            if 0 <= label < num_classes:
                class_indices[label].append(idx)

        if any(len(indices) == 0 for indices in class_indices):
            return 0.0

        shots_per_class = min(len(indices) for indices in class_indices)
        if shots_per_class < 2:
            return None

        alpha_steps = max(2, int(alpha_steps))
        alphas = torch.linspace(0, 1, alpha_steps, device=self.device)
        net_scores = torch.zeros(alpha_steps, device=self.device)
        targets = torch.arange(num_classes, device=self.device)

        for hold_idx in range(shots_per_class):
            held_indices = torch.tensor(
                [class_indices[class_idx][hold_idx] for class_idx in range(num_classes)],
                device=self.device,
                dtype=torch.long,
            )
            held_images = train_features[held_indices]
            held_text = train_text_features[held_indices]

            visual_minus = []
            for class_idx in range(num_classes):
                keep = [idx for shot_idx, idx in enumerate(class_indices[class_idx][:shots_per_class]) if shot_idx != hold_idx]
                class_features = train_features[torch.tensor(keep, device=self.device, dtype=torch.long)]
                visual_minus.append(self._weighted_centroid_from_features(class_features, self.text_prototypes[class_idx]))
            visual_minus = torch.stack(visual_minus, dim=0)

            if force_loo_accuracy:
                baseline_correct = None
            else:
                baseline_logits = torch.einsum("cd,ckd->ck", held_images, held_text)
                baseline_correct = baseline_logits.argmax(dim=-1).eq(targets)

            for alpha_idx, alpha in enumerate(alphas):
                fused_text = F.normalize(
                    (1.0 - alpha) * held_text + alpha * visual_minus.unsqueeze(0),
                    dim=-1,
                )
                fused_correct = torch.einsum("cd,ckd->ck", held_images, fused_text).argmax(dim=-1).eq(targets)
                if force_loo_accuracy:
                    net_scores[alpha_idx] += fused_correct.sum().float()
                else:
                    rescue = (~baseline_correct) & fused_correct
                    damage = baseline_correct & ~fused_correct
                    net_scores[alpha_idx] += rescue.sum().float() - damage.sum().float()

        return alphas[int(net_scores.argmax().item())].item()

    def apply_posthoc_protofuse(self, alpha, visual_centroids, centroid_mask=None, missing_classes=None):
        self.freeze()
        alpha = max(0.0, min(1.0, alpha))

        if self.text_prototypes is None:
            self.get_text_prototypes()
        self.visual_centroids = F.normalize(visual_centroids.to(self.device).float(), dim=-1)
        if centroid_mask is None:
            centroid_mask = torch.ones(self.visual_centroids.shape[0], device=self.device, dtype=torch.bool)
        self.posthoc_centroid_mask = centroid_mask.to(self.device).bool()
        self.posthoc_missing_classes = list(missing_classes or [])
        self.posthoc_alpha = alpha
        self.model.set_posthoc_protofuse(
            self.visual_centroids,
            alpha,
            centroid_mask=self.posthoc_centroid_mask,
        )
        return self.visual_centroids

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
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
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

                logits = self.model(images)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]

                loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
                running_loss += loss.item()
                steps += 1

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().tolist())
                all_labels_list.extend(labels.cpu().tolist())
        
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
            'cfg': self.cfg,
            'posthoc_protofuse': {
                'text_prototypes': self.text_prototypes.detach().cpu() if self.text_prototypes is not None else None,
                'visual_centroids': self.visual_centroids.detach().cpu() if self.visual_centroids is not None else None,
                'centroid_mask': self.posthoc_centroid_mask.detach().cpu() if self.posthoc_centroid_mask is not None else None,
                'alpha': self.posthoc_alpha,
                'missing_classes': self.posthoc_missing_classes,
            },
        }
        torch.save(checkpoint, path)
        # logger.info(f"Model saved to {path}")

    def save_posthoc_protofuse(self, path):
        torch.save({
            'text_prototypes': self.text_prototypes.detach().cpu() if self.text_prototypes is not None else None,
            'visual_centroids': self.visual_centroids.detach().cpu() if self.visual_centroids is not None else None,
            'centroid_mask': self.posthoc_centroid_mask.detach().cpu() if self.posthoc_centroid_mask is not None else None,
            'alpha': self.posthoc_alpha,
            'missing_classes': self.posthoc_missing_classes,
            'classnames': self.classnames,
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if 'prompt_learner_state_dict' in checkpoint:
            self.model.prompt_learner.load_state_dict(checkpoint['prompt_learner_state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        posthoc = checkpoint.get('posthoc_protofuse') or {}
        self.text_prototypes = posthoc.get('text_prototypes')
        self.visual_centroids = posthoc.get('visual_centroids')
        self.posthoc_centroid_mask = posthoc.get('centroid_mask')
        self.posthoc_alpha = posthoc.get('alpha')
        self.posthoc_missing_classes = posthoc.get('missing_classes') or []
        if self.text_prototypes is not None:
            self.text_prototypes = self.text_prototypes.to(self.device)
        if self.visual_centroids is not None:
            self.visual_centroids = self.visual_centroids.to(self.device)
        if self.posthoc_centroid_mask is not None:
            self.posthoc_centroid_mask = self.posthoc_centroid_mask.to(self.device).bool()
        if self.visual_centroids is not None and self.posthoc_alpha is not None:
            self.model.set_posthoc_protofuse(
                self.visual_centroids,
                self.posthoc_alpha,
                centroid_mask=self.posthoc_centroid_mask,
            )
        # logger.info(f"Model loaded from {path}")

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
            score = logits[0, target_classes[i]]
            
            self.model.zero_grad()
            score.backward(retain_graph=True)
            
            if target_unpooled.grad is None:
                gradcams.append(np.zeros((8, 8)))
                logger.warning("Empty CAM encountered")
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
                    logger.warning("CAM size is not a perfect square, padding")
            else:
                logger.warning("Empty CAM encountered")
                cam = np.zeros((8, 8))
            
            gradcams.append(cam)
            if target_unpooled.grad is not None:
                target_unpooled.grad.zero_()
        
        for param in self.model.vis_encoder.parameters():
            param.requires_grad_(False)
        self.model.train(original_mode)
        return gradcams
