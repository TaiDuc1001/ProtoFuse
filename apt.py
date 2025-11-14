import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict, Counter
import argparse
import datetime
import json
import os
import itertools
from decode import APTDecoder
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import multiprocessing as mp
import copy
from thop import profile

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

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=False):
        super().__init__()
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.A = nn.Linear(in_features, r, bias=False)
            self.B = nn.Linear(r, out_features, bias=bias)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
        else:
            self.A = None
            self.B = None

    def forward(self, x):
        if self.A is None or self.B is None or self.r <= 0:
            return torch.zeros_like(x)
        return self.B(self.A(x)) * (self.alpha / self.r)


class FixedLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.base = base_layer
        self.alpha = alpha
        self.r = r
        in_features = base_layer.weight.shape[1]
        out_features = base_layer.weight.shape[0]

        if r > 0:
            self.A = nn.Parameter(torch.zeros(r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.A = None
            self.B = None

        for param in self.base.parameters():
            param.requires_grad = False

    def weight(self):
        if self.A is None or self.B is None or self.r <= 0:
            return self.base.weight
        delta = (self.B @ self.A) * (self.alpha / self.r)
        return self.base.weight + delta

    def forward(self, x):
        weight = self.weight().to(x.dtype)
        bias = self.base.bias
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(x.dtype)
        return F.linear(x, weight, bias)

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


class LoRACrossAttention(CrossAttention):
    def __init__(self, feature_dim, num_heads, dropout, lora_rank=4, lora_alpha=1.0):
        super().__init__(feature_dim, num_heads, dropout)
        self.use_lora = lora_rank > 0
        self.lora_q = LoRALinear(feature_dim, feature_dim, r=lora_rank, alpha=lora_alpha, bias=False)
        self.lora_k = LoRALinear(feature_dim, feature_dim, r=lora_rank, alpha=lora_alpha, bias=False)
        self.lora_v = LoRALinear(feature_dim, feature_dim, r=lora_rank, alpha=lora_alpha, bias=False)
        self.lora_ff = LoRALinear(feature_dim, feature_dim, r=lora_rank, alpha=lora_alpha, bias=True)

    def forward(self, unpooled, text_features):
        if not self.use_lora:
            return super().forward(unpooled, text_features)

        q = text_features + self.lora_q(text_features)
        k = unpooled + self.lora_k(unpooled)
        v = unpooled + self.lora_v(unpooled)
        out, attn_weights = self.cross_attn(q, k, v)
        text_features = self.norm1(self.dropout(text_features + out))
        ff = self.feed_forward(text_features) + self.lora_ff(text_features)
        text_features = self.norm2(self.dropout(text_features + ff))
        return text_features, attn_weights


class FixedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0, use_lora=False, lora_rank=4, lora_alpha=1.0):
        super().__init__()
        self.dim = dim
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.q_adapter = None
        self.k_adapter = None
        self.v_adapter = None

        if use_lora and lora_rank > 0:
            self.q_adapter = FixedLoRALinear(self.q_proj, r=lora_rank, alpha=lora_alpha)
            self.k_adapter = FixedLoRALinear(self.k_proj, r=lora_rank, alpha=lora_alpha)
            self.v_adapter = FixedLoRALinear(self.v_proj, r=lora_rank, alpha=lora_alpha)
            for module in (self.q_proj, self.k_proj, self.v_proj):
                for param in module.parameters():
                    param.requires_grad = False

    def _project(self, module, adapter, x):
        if adapter is not None:
            return adapter(x)
        return module(x)

    def forward(self, tokens, patches):
        q = self._project(self.q_proj, self.q_adapter, tokens)
        k = self._project(self.k_proj, self.k_adapter, patches)
        v = self._project(self.v_proj, self.v_adapter, patches)

        out, attn = self.attn(q, k, v)
        tokens = self.norm1(tokens + out)
        tokens = self.norm2(tokens + self.mlp(tokens))

        return tokens, attn


def fuse_levels_cascade(levels, text_tokens, attn_modules):
    out = text_tokens
    attn_maps = []
    modules = list(attn_modules)
    if not modules:
        raise ValueError("No attention modules provided for cascade fusion.")

    for idx, level in enumerate(levels):
        module = modules[idx] if idx < len(modules) else modules[-1]
        patches = level.permute(1, 0, 2)
        out, attn = module(out, patches)
        attn_maps.append(attn)

    return out, attn_maps

class ImageEncoder(nn.Module):
    def __init__(self, clip_model, selected_layers=None, use_fix_pack=False):
        super().__init__()
        visual = clip_model.visual

        self.conv1 = visual.conv1
        self.class_embedding = visual.class_embedding
        self.positional_embedding = visual.positional_embedding
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.use_fix_pack = use_fix_pack

        selected_layers = selected_layers or []
        self.selected_layers = sorted(set(selected_layers))
        self._selected_layers_set = set(self.selected_layers)

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

        collected_levels = []
        for idx, block in enumerate(self.transformer.resblocks):
            x = block(x)
            if self._selected_layers_set and idx in self._selected_layers_set:
                if self.use_fix_pack:
                    collected_levels.append(x.permute(1, 0, 2))
                else:
                    collected_levels.append(x.clone())

        final_unpooled = x.permute(1, 0, 2)
        final_unpooled = self.ln_post(final_unpooled)

        if self.proj is not None:
            final_unpooled = final_unpooled @ self.proj

        global_feature = final_unpooled[:, 0, :]

        if self._selected_layers_set:
            processed_levels = []
            for level in collected_levels:
                if self.use_fix_pack:
                    level = self.ln_post(level)
                    if self.proj is not None:
                        level = level @ self.proj
                else:
                    level = level.permute(1, 0, 2)
                    level = self.ln_post(level)
                    if self.proj is not None:
                        level = level @ self.proj
                processed_levels.append(level)

            if not processed_levels:
                processed_levels.append(final_unpooled)

            return processed_levels, global_feature

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

        self.use_fix_pack = cfg.get('use_fix_pack_ac', False)
        self.use_multilevel = cfg.get('use_multilevel', False)
        self.fusion_strategy = cfg.get('multilevel_fusion', 'mean')

        use_lora = cfg.get('use_lora', False)
        lora_rank = cfg.get('lora_rank', 4)
        lora_alpha = cfg.get('lora_alpha', 1.0)
        prompt_dim = self.clip_model.text_projection.shape[1]
        num_heads = cfg.get('num_heads', 8)
        dropout = cfg.get('dropout', 0.1)
        fix_mlp_ratio = cfg.get('fix_mlp_ratio', 4.0)

        if self.use_multilevel:
            if use_lora and lora_rank > 0:
                raise ValueError("LoRA and multi-level APT cannot be enabled simultaneously.")

            selected_layers = cfg.get('multilevel_layers', [])
            if isinstance(selected_layers, str):
                selected_layers = [s.strip() for s in selected_layers.split(',') if s.strip()]
            if selected_layers is None:
                selected_layers = []
            parsed_layers = []
            total_blocks = len(self.clip_model.visual.transformer.resblocks)
            for layer in selected_layers:
                try:
                    idx = int(layer)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < total_blocks:
                    parsed_layers.append(idx)
            if not parsed_layers:
                default_candidates = [2, 6, 10]
                parsed_layers = [idx for idx in default_candidates if idx < total_blocks]
            if not parsed_layers:
                parsed_layers = [max(0, total_blocks - 1)]
            last_idx = total_blocks - 1
            if last_idx not in parsed_layers:
                parsed_layers.append(last_idx)
            self.selected_layers = sorted(set(parsed_layers))
            self.cfg['multilevel_layers'] = self.selected_layers

            if self.use_fix_pack:
                self.prompt_learner = nn.ModuleList([
                    FixedCrossAttention(
                        dim=prompt_dim,
                        num_heads=num_heads,
                        mlp_ratio=fix_mlp_ratio,
                        dropout=dropout,
                        use_lora=False,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha
                    ) for _ in self.selected_layers
                ])
                self.fusion_logits = None
            else:
                self.prompt_learner = CrossAttention(
                    feature_dim=prompt_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )

                if self.fusion_strategy == 'weighted':
                    self.fusion_logits = nn.Parameter(torch.zeros(len(self.selected_layers)))
                else:
                    self.fusion_logits = None
        else:
            prompt_layers = []
            for _ in range(cfg.get('num_layers', 1)):
                if self.use_fix_pack:
                    prompt_layers.append(
                        FixedCrossAttention(
                            dim=prompt_dim,
                            num_heads=num_heads,
                            mlp_ratio=fix_mlp_ratio,
                            dropout=dropout,
                            use_lora=use_lora,
                            lora_rank=lora_rank,
                            lora_alpha=lora_alpha
                        )
                    )
                else:
                    if use_lora and lora_rank > 0:
                        prompt_layers.append(
                            LoRACrossAttention(
                                feature_dim=prompt_dim,
                                num_heads=num_heads,
                                dropout=dropout,
                                lora_rank=lora_rank,
                                lora_alpha=lora_alpha
                            )
                        )
                    else:
                        prompt_layers.append(
                            CrossAttention(
                                feature_dim=prompt_dim,
                                num_heads=num_heads,
                                dropout=dropout
                            )
                        )
            self.prompt_learner = nn.ModuleList(prompt_layers)
            self.selected_layers = []
            self.fusion_logits = None

        if cfg.get('precision', 'fp32') == 'fp16':
            self.prompt_learner = self.prompt_learner.half()

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.vis_encoder = ImageEncoder(
            self.clip_model,
            selected_layers=self.selected_layers if self.use_multilevel else None,
            use_fix_pack=self.use_fix_pack
        )
        self.logit_scale = clip_model.logit_scale

        self.text_features, self.prompts, self.text_tokens = self._init_text_feats(cfg, classnames)

    def _init_text_feats(self, cfg, classnames):
        dataset_name = cfg.get('dataset_name', 'ImageNet')
        temp = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")
        myencoder = TextEncoder(self.clip_model).to(self.device)
        prompts = [temp.format(c.replace('_', ' ')) for c in classnames]

        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        tokens = myencoder.encode_text_tokens(prompts)
        text_features = myencoder.cls_from_tokens(tokens, prompts)

        if not self.use_fix_pack:
            tokens = None

        return text_features, prompts, tokens

    def forward(self, image, label=None):
        with torch.no_grad():
            visual_output = self.vis_encoder(image)

        unpooled_levels, image_features = visual_output
        if not isinstance(unpooled_levels, list):
            unpooled_levels = [unpooled_levels]

        attn_maps = []

        if self.use_fix_pack:
            if self.text_tokens is None:
                raise RuntimeError("Token-level text features are required for fix pack mode.")

            batch_size = image.shape[0]
            token_vectors = self.text_tokens.mean(dim=1)
            token_vectors = token_vectors.unsqueeze(1).expand(-1, batch_size, -1)
            token_vectors = token_vectors.permute(0, 1, 2)
            target_dtype = unpooled_levels[0].dtype
            token_vectors = token_vectors.to(target_dtype)

            if self.use_multilevel:
                modules = list(self._prompt_layers_iter())
                adapted_tokens, attn_maps = fuse_levels_cascade(unpooled_levels, token_vectors, modules)
            else:
                patches = unpooled_levels[0].permute(1, 0, 2)
                adapted_tokens = token_vectors
                for module in self._prompt_layers_iter():
                    adapted_tokens, attn = module(adapted_tokens, patches)
                    attn_maps.append(attn)
            text_features = adapted_tokens.permute(1, 0, 2)
        else:
            base_text_features = self.text_features.clone()
            if self.use_multilevel:
                adapted_outputs = []
                for level in unpooled_levels:
                    unpooled_level = level.permute(1, 0, 2)
                    text_features = base_text_features.unsqueeze(1).expand(-1, unpooled_level.shape[1], -1)
                    text_features, attn_weights = self.prompt_learner(unpooled_level, text_features)
                    adapted_outputs.append(text_features)
                    attn_maps.append(attn_weights)

                if self.fusion_strategy == 'weighted' and self.fusion_logits is not None and len(adapted_outputs) > 1:
                    weights = F.softmax(self.fusion_logits[:len(adapted_outputs)], dim=0)
                    fused = torch.zeros_like(adapted_outputs[0])
                    for w, out in zip(weights, adapted_outputs):
                        fused = fused + w * out
                    text_features = fused
                elif self.fusion_strategy == 'weighted' and self.fusion_logits is not None:
                    text_features = adapted_outputs[0]
                else:
                    stacked = torch.stack(adapted_outputs, dim=0)
                    text_features = stacked.mean(dim=0)
            else:
                unpooled_images = unpooled_levels[0].permute(1, 0, 2)
                text_features = base_text_features.unsqueeze(1).expand(-1, unpooled_images.shape[1], -1)

                for layer in self._prompt_layers_iter():
                    text_features, attn_weights = layer(unpooled_images, text_features)
                    attn_maps.append(attn_weights)

            text_features = text_features.permute(1, 0, 2)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        image_features = image_features.unsqueeze(1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * F.cosine_similarity(image_features, text_features, dim=-1)

        mode = self.cfg.get('mode', 'logits')

        if self.training and label is not None:
            loss = F.cross_entropy(logits, label)
            return loss, logits
        elif mode == "logits":
            return logits
        elif mode == "features":
            return logits, text_features
        elif mode == "map":
            return logits, attn_maps

        return logits

    def _prompt_layers_iter(self):
        if isinstance(self.prompt_learner, nn.ModuleList):
            return self.prompt_learner
        return [self.prompt_learner]

    def trainable_parameters(self):
        if self.use_multilevel and self.fusion_strategy == 'weighted' and self.fusion_logits is not None:
            return itertools.chain(self.prompt_learner.parameters(), [self.fusion_logits])
        return self.prompt_learner.parameters()

    def get_trainable_parameter_names(self):
        names = [f"prompt_learner.{name}" for name, _ in self.prompt_learner.named_parameters()]
        if self.use_multilevel and self.fusion_strategy == 'weighted' and self.fusion_logits is not None:
            names.append('fusion_logits')
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

class StandaloneAPT:
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
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        
        num_epochs = self.cfg.get('num_epochs', 100)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs
        )
    
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
        old_mode = None
        if isinstance(self.model, nn.Module) and isinstance(self.model.cfg, dict):
            old_mode = self.model.cfg.get('mode', None)
            self.model.cfg['mode'] = 'logits'

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)

                if isinstance(logits, (list, tuple)):
                    logits = logits[0]

                loss = F.cross_entropy(logits, labels)
                running_loss += loss.item()
                steps += 1

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        if old_mode is not None:
            self.model.cfg['mode'] = old_mode
        
        accuracy = 100 * correct / total
        avg_loss = running_loss / max(1, steps)
        return {"accuracy": accuracy, "loss": avg_loss, "predictions": all_preds, "true_labels": all_labels_list}
    
    def predict(self, images, return_features=False):
        self.model.eval()
        images = images.to(self.device)
        
        with torch.no_grad():
            if return_features:
                result = self.model(images)
                return result
            else:
                logits = self.model(images)
                return logits
    
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
        print(msg)
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
    
    @classmethod
    def load_from_run_dir(cls, run_dir, device='cuda', checkpoint='best.pth'):
        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        classnames = cfg['classnames']
        trainer = cls(cfg, classnames, device=device)
        model_path = os.path.join(run_dir, checkpoint)
        trainer.load_model(model_path)
        return trainer
    
    def forward_uq(self, images, num_samples=10):
        self.model.train()
        images = images.to(self.device)
        
        all_logits = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.model(images)
                all_logits.append(logits.unsqueeze(0))
        
        all_logits = torch.cat(all_logits, dim=0)
        mean_logits = all_logits.mean(dim=0)
        std_logits = all_logits.std(dim=0)
        
        return mean_logits, std_logits

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def generate_gradcam(self, images, target_classes):
        original_mode = self.model.training
        self.model.train()
        
        for param in self.model.vis_encoder.parameters():
            param.requires_grad_(True)
            
        images = images.to(self.device)
        images.requires_grad_(True)
        
        encoder_output = self.model.vis_encoder(images)
        if self.model.use_multilevel:
            unpooled_levels, _ = encoder_output
            target_unpooled = unpooled_levels[-1]
        else:
            target_unpooled, _ = encoder_output
            unpooled_levels = [target_unpooled]

        target_unpooled.retain_grad()
        
        batch_size = images.shape[0]
        gradcams = []
        
        for i in range(batch_size):
            target_class = target_classes[i]
            
            base_text = self.model.text_features.clone()

            if self.model.use_multilevel:
                per_level_outputs = []
                for level in unpooled_levels:
                    unpooled_single = level[i:i+1].permute(1, 0, 2)
                    text_features = base_text.unsqueeze(1).expand(-1, unpooled_single.shape[1], -1)
                    text_features, _ = self.model.prompt_learner(unpooled_single, text_features)
                    per_level_outputs.append(text_features)

                if self.model.fusion_strategy == 'weighted' and self.model.fusion_logits is not None and len(per_level_outputs) > 1:
                    weights = F.softmax(self.model.fusion_logits[:len(per_level_outputs)], dim=0)
                    fused = torch.zeros_like(per_level_outputs[0])
                    for w, out in zip(weights, per_level_outputs):
                        fused = fused + w * out
                    text_features = fused
                elif self.model.fusion_strategy == 'weighted' and self.model.fusion_logits is not None:
                    text_features = per_level_outputs[0]
                else:
                    text_features = torch.stack(per_level_outputs, dim=0).mean(dim=0)
            else:
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
                print(f"Image {i}: No gradients captured!")
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

    def decode_adapted_prompts(self, images, entry_length=30, temperature=1.0, batch_decode_size=32):
        self.model.eval()
        images = images.to(self.device)

        with torch.no_grad():
            encoder_output = self.model.vis_encoder(images)
            if self.model.use_multilevel:
                unpooled_levels, _ = encoder_output
                base_text = self.model.text_features.clone()
                adapted_outputs = []
                for level in unpooled_levels:
                    level_unpooled = level.permute(1, 0, 2)
                    text_features = base_text.unsqueeze(1).expand(-1, level_unpooled.shape[1], -1)
                    text_features, _ = self.model.prompt_learner(level_unpooled, text_features)
                    adapted_outputs.append(text_features)

                if self.model.fusion_strategy == 'weighted' and self.model.fusion_logits is not None and len(adapted_outputs) > 1:
                    weights = F.softmax(self.model.fusion_logits[:len(adapted_outputs)], dim=0)
                    fused = torch.zeros_like(adapted_outputs[0])
                    for w, out in zip(weights, adapted_outputs):
                        fused = fused + w * out
                    text_features = fused
                elif self.model.fusion_strategy == 'weighted' and self.model.fusion_logits is not None:
                    text_features = adapted_outputs[0]
                else:
                    text_features = torch.stack(adapted_outputs, dim=0).mean(dim=0)
            else:
                unpooled_images = encoder_output[0].permute(1, 0, 2)
                text_features = self.model.text_features.clone()
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
                    batch_embeddings, 
                    entry_length=entry_length, 
                    temperature=temperature
                )
                all_captions.extend(batch_captions)
            
            batch_out = []
            caption_idx = 0
            for b in range(batch_size):
                classes_out = []
                for c in range(num_classes):
                    class_name = self.classnames[c] if c < len(self.classnames) else f"Class_{c}"
                    classes_out.append({
                        'class_id': c,
                        'class_name': class_name,
                        'generated_caption': all_captions[caption_idx]
                    })
                    caption_idx += 1
                batch_out.append(classes_out)
        
        return batch_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train APT model")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--kshot', type=int, default=16, help='K-shot (default: 16)')
    parser.add_argument('--backbone', type=str, default='ViT-B/32', help='CLIP backbone (default: ViT-B/32)')
    parser.add_argument('--dataset_name', type=str, default='CUBirds', help='Dataset name (default: CUBirds)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in cross attention (default: 8)')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of prompt learner layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--precision', type=str, default='fp32', help='Precision (default: fp32)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay (default: 0.0005)')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs (default: 150)')
    parser.add_argument('--mode', type=str, default='logits', help='Mode (default: logits)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory (default: outputs)')
    parser.add_argument('--run_decoder', action='store_true', help='Run the APTDecoder to generate captions')
    parser.add_argument('--visualize_attention', action='store_true', help='Generate and save attention heatmaps')
    parser.add_argument('--visualize_gradcam', action='store_true', help='Generate and save GradCAM heatmaps')
    parser.add_argument('--vis_dir', type=str, default='attention_maps', help='Directory to save attention maps (default: attention_maps)')
    parser.add_argument('--use_cutout', action='store_true', help='Use Cutout (RandomErasing) augmentation')
    parser.add_argument('--use_lora', action='store_true', help='Enable LoRA adaptation in prompt learner')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank (default: 4)')
    parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA alpha scaling (default: 1.0)')
    parser.add_argument('--use_multilevel', action='store_true', help='Enable multi-level prompt adaptation')
    parser.add_argument('--multilevel_layers', type=str, default='', help='Comma-separated ViT layer indices for multi-level adaptation')
    parser.add_argument('--multilevel_fusion', type=str, default='mean', choices=['mean', 'weighted'], help='Fusion strategy for multi-level outputs (default: mean)')
    parser.add_argument('--confusion_matrix', action='store_true', help='generate confusion matrices')
    parser.add_argument('--use_fix_pack_ac', action='store_true', help='Enable Fix Pack A+C corrected pipeline')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'], help='Optimizer type (default: SGD)')

    args = parser.parse_args()

    if args.multilevel_layers:
        multilevel_layers = []
        for item in args.multilevel_layers.split(','):
            item = item.strip()
            if not item:
                continue
            try:
                multilevel_layers.append(int(item))
            except ValueError:
                continue
    else:
        multilevel_layers = []

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
        'use_lora': args.use_lora,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'use_multilevel': args.use_multilevel,
        'multilevel_layers': multilevel_layers,
        'multilevel_fusion': args.multilevel_fusion,
        'generate_confusion_matrix': args.confusion_matrix,
        'optimizer': args.optimizer,
        'use_fix_pack_ac': args.use_fix_pack_ac
    }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    run_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir)
    
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)
    
    dataset_root = "./datasets/cub-200-2011-renamed"
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    K_SHOT = args.kshot
    num_workers = 4

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ]

    if cfg.get('use_cutout', False):
        base_transforms.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0))
        print("Using Cutout (RandomErasing) augmentation.")

    transform = transforms.Compose(base_transforms)

    try:
        dataset = ImageFolder(dataset_root, transform=transform)
    except Exception as e:
        print(f"Failed to load dataset from {dataset_root}: {e}")
        print("Please create a dataset at the given path or change dataset_root.")
        exit()

    samples_by_class_idx = defaultdict(list)
    for i, (path, class_idx) in enumerate(dataset.samples):
        samples_by_class_idx[class_idx].append(i)
    
    random.seed(42)
    train_indices = []
    val_indices = []

    for class_idx, indices in samples_by_class_idx.items():
        random.shuffle(indices)
        k = min(len(indices), K_SHOT)
        train_indices.extend(indices[:k])
        val_indices.extend(indices[k:])

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    print(f"Dataset loaded: {len(dataset)} total images.")
    print(f"Created non-episodic split: {len(train_ds)} train images ({K_SHOT}-shot), {len(val_ds)} val images.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classnames = dataset.classes
    cfg['classnames'] = classnames
    
    log_file = os.path.join(run_dir, 'training.log')
    trainer = StandaloneAPT(cfg, classnames, device=args.device, log_file=log_file)

    metrics = []
    best_val_acc = 0
    
    with open(log_file, 'w') as f:
        f.write(f"Config: {cfg}\n\n")
        f.write('='*50 + '\n')
        f.write(f"Dataset loaded: {len(dataset)} total images.\n")
        f.write(f"Created non-episodic split: {len(train_ds)} train images ({K_SHOT}-shot), {len(val_ds)} val images.\n")
        if cfg.get('use_cutout', False):
            f.write("Using Cutout (RandomErasing) augmentation.\n")
        f.write('\n')
        f.write('='*50 + '\n')

    print('\n')
    print('='*50)
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        trainer.model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            loss_dict = trainer.train_step(batch)
            running_loss += loss_dict['loss']
            running_accuracy += loss_dict['accuracy']
            steps += 1

        avg_loss = running_loss / max(1, steps)
        avg_acc = running_accuracy / max(1, steps)
        results = trainer.evaluate(val_loader)
        val_acc = results['accuracy']
        val_loss = results['loss']
        all_preds = results['predictions']
        all_labels = results['true_labels']
        
        epoch_dir = os.path.join(run_dir, f'epoch_{epoch:03d}')
        os.makedirs(epoch_dir, exist_ok=True)
        maps_dir = os.path.join(epoch_dir, 'maps')
        os.makedirs(maps_dir, exist_ok=True)
        
        if cfg.get('generate_confusion_matrix', True):
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
                        plot_args.append((cm, row_idx, col_idx, start_row, start_col, end_row, end_col, epoch, cm_dir))
                
                with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
                    pool.map(generate_confusion_matrix_plot, plot_args)
            else:
                fig, ax = plt.subplots(figsize=(max(16, num_classes // 2), max(16, num_classes // 2)))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                            xticklabels=[str(i) for i in range(num_classes)], yticklabels=[str(i) for i in range(num_classes)],
                            annot_kws={"size": 16})
                ax.set_title(f'Confusion Matrix - Epoch {epoch}', fontsize=12)
                ax.set_xlabel('Predicted Label', fontsize=12)
                ax.set_ylabel('True Label', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(cm_dir, 'confusion_matrix.pdf'), dpi=300, bbox_inches='tight')
                plt.close()
        
        gt_counts = Counter(all_labels)
        pred_counts = Counter(all_preds)
        
        classes = sorted(set(gt_counts.keys()) | set(pred_counts.keys()))
        gt_values = [gt_counts.get(cls, 0) for cls in classes]
        pred_values = [pred_counts.get(cls, 0) for cls in classes]
        
        fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.5), 8))
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gt_values, width, label='Ground Truth', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, pred_values, width, label='Predictions', color='salmon', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Class Distribution - Epoch {epoch}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([str(cls) for cls in classes], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(gt_values + pred_values) * 0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(gt_values + pred_values) * 0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, 'class_distribution.pdf'), dpi=150, bbox_inches='tight')
        plt.close()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trainer.save_model(os.path.join(run_dir, 'best.pt'))
        epoch_time = time.time() - start_time
        metrics.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': epoch_time
        })
        epoch_str = f"Epoch {epoch} - train_loss={avg_loss:.4f} - train_acc={avg_acc:.2f}% - val_loss={val_loss:.4f} - val_acc={val_acc:.2f}% - time={epoch_time:.2f}s"
        print(epoch_str)

        with open(log_file, 'a') as f:
            f.write(epoch_str + '\n')

        trainer.scheduler.step()

        num_display = min(10, len(classnames), len(val_indices))
        selected_indices = []
        seen_classes = set()
        for idx in val_indices:
            cls_idx = dataset.samples[idx][1]
            if cls_idx not in seen_classes:
                seen_classes.add(cls_idx)
                selected_indices.append(idx)
            if len(selected_indices) >= num_display:
                break

        if len(selected_indices) == 0:
            try:
                batch_data = next(iter(val_loader))
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    sample_images, sample_labels = batch_data[0], batch_data[1]
                    batch_indices = val_indices[:len(sample_images)]
                    sample_paths = [os.path.abspath(dataset.samples[idx][0]) for idx in batch_indices]
                else:
                    sample_images, sample_labels = batch_data, None
                    sample_paths = []
            except StopIteration:
                sample_images = None
                sample_labels = None
                sample_paths = []
                print("Validation loader is empty, skipping prompt decoding.")
                with open(log_file, 'a') as f:
                    f.write("Validation loader is empty, skipping prompt decoding.\n")
        else:
            sample_images_list = []
            sample_labels_list = []
            sample_paths = []
            for idx in selected_indices:
                img, lbl = dataset[idx]
                sample_images_list.append(img)
                sample_labels_list.append(lbl)
                sample_paths.append(os.path.abspath(dataset.samples[idx][0]))

            sample_images = torch.stack(sample_images_list)
            sample_labels = torch.tensor(sample_labels_list)

        if sample_images is not None:
            if args.run_decoder:
                decoded_prompts = trainer.decode_adapted_prompts(sample_images, entry_length=30, temperature=1.0)

                if decoded_prompts is not None:
                    prompt_str = f"Generated captions from learned prompts for epoch {epoch}:"
                    print(prompt_str)

                    with open(log_file, 'a') as f:
                        f.write(prompt_str + '\n')
                        for i in range(len(decoded_prompts)):
                            image_prompts = decoded_prompts[i]
                            image_path = sample_paths[i] if i < len(sample_paths) else "Unknown path"
                            image_label = None
                            if sample_labels is not None:
                                try:
                                    image_label = int(sample_labels[i])
                                except Exception:
                                    image_label = None

                            image_str = f"Image ({image_path}):"
                            print(image_str)
                            f.write(image_str + '\n')

                            selected_prompt = None
                            if image_label is not None:
                                for p in image_prompts:
                                    if p.get('class_id') == image_label:
                                        selected_prompt = p
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
            
            if args.visualize_attention:
                trainer.model.cfg['mode'] = 'map'

                device_for_model = trainer.device
                if isinstance(sample_images, torch.Tensor):
                    vis_images = sample_images.to(device_for_model)
                elif isinstance(sample_images, (list, tuple)):
                    vis_images = torch.stack([x.to(device_for_model) if isinstance(x, torch.Tensor) else torch.tensor(x).to(device_for_model) for x in sample_images])
                else:
                    vis_images = torch.tensor(sample_images).to(device_for_model)

                vis_labels = None
                if sample_labels is not None:
                    if isinstance(sample_labels, torch.Tensor):
                        vis_labels = sample_labels.to(device_for_model)
                    else:
                        vis_labels = torch.tensor(sample_labels).to(device_for_model)

                logits, attn_maps = trainer.model(vis_images)
                trainer.model.cfg['mode'] = cfg['mode']

                attn_map_to_vis = attn_maps[0]

                try:
                    shape_info = getattr(attn_map_to_vis, 'shape', None)
                    shape_msg = f"Epoch {epoch} attention map shape: {shape_info}"
                    print(shape_msg)
                    with open(log_file, 'a') as lf:
                        lf.write(shape_msg + '\n')
                except Exception:
                    pass

                for i in range(len(vis_images)):
                    image_path = sample_paths[i]
                    label = int(vis_labels[i].item()) if vis_labels is not None else None

                    if label is None:
                        print(f"No label for image {i}, skipping attention visualization.")
                        continue

                    try:
                        weights = attn_map_to_vis[i, label, :]
                    except Exception:
                        warn_msg = f"Warning: unable to index attention map for image {i}, label {label} with expected strategy a[i, label, :]. Skipping visualization."
                        print(warn_msg)
                        with open(log_file, 'a') as lf:
                            lf.write(warn_msg + '\n')
                        continue

                    if weights is None:
                        print(f"Warning: unable to index attention map for image {i}, label {label}. Skipping visualization.")
                        continue

                    if weights.dim() > 1:
                        mean_weights = weights.mean(dim=0).detach().cpu().numpy()
                    else:
                        mean_weights = weights.detach().cpu().numpy()

                    patch_weights = mean_weights[1:]

                    stats_msg = f"Epoch {epoch} image {i} label {label} attention stats: mean={mean_weights.mean():.6f} min={mean_weights.min():.6f} max={mean_weights.max():.6f}"
                    print(stats_msg)
                    with open(log_file, 'a') as lf:
                        lf.write(stats_msg + '\n')
                    num_patches = patch_weights.shape[0]
                    h = w = int(np.sqrt(num_patches))

                    if h * w != num_patches:
                        print(f"Warning: Cannot reshape {num_patches} patches into a square grid. Skipping visualization for image {i}.")
                        continue
                        
                    heatmap = patch_weights.reshape(h, w)
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                    heatmap = (heatmap * 255).astype(np.uint8)
                    
                    original_img = cv2.imread(image_path)
                    original_img = cv2.resize(original_img, (224, 224)) # type: ignore
                    
                    heatmap_img = cv2.applyColorMap(cv2.resize(heatmap, (224, 224)), cv2.COLORMAP_JET)
                    
                    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_img, 0.4, 0)
                    
                    save_name = f"epoch_{epoch:03d}_img_{i}_class_{label}_{classnames[int(label)]}.jpg"
                    save_path = os.path.join(maps_dir, save_name)
                    cv2.imwrite(save_path, superimposed_img)

                vis_log_str = f"Saved {len(vis_images)} attention visualizations to {maps_dir}"
                print(vis_log_str)
                with open(log_file, 'a') as f:
                    f.write(vis_log_str + '\n')
            
            if args.visualize_gradcam:
                device_for_model = trainer.device
                if isinstance(sample_images, torch.Tensor):
                    vis_images = sample_images.to(device_for_model)
                elif isinstance(sample_images, (list, tuple)):
                    vis_images = torch.stack([x.to(device_for_model) if isinstance(x, torch.Tensor) else torch.tensor(x).to(device_for_model) for x in sample_images])
                else:
                    vis_images = torch.tensor(sample_images).to(device_for_model)

                vis_labels = None
                if sample_labels is not None:
                    if isinstance(sample_labels, torch.Tensor):
                        vis_labels = sample_labels.to(device_for_model)
                    else:
                        vis_labels = torch.tensor(sample_labels).to(device_for_model)

                if vis_labels is not None:
                    gradcams = trainer.generate_gradcam(vis_images, vis_labels)

                    for i, gradcam in enumerate(gradcams):
                        image_path = sample_paths[i]
                        label = int(vis_labels[i].item())

                        heatmap = gradcam.astype(np.float32)
                        heatmap = (heatmap * 255).astype(np.uint8)

                        original_img = cv2.imread(image_path)
                        if original_img is not None:
                            original_img = cv2.resize(original_img, (224, 224)) # type: ignore

                            heatmap_img = cv2.applyColorMap(cv2.resize(heatmap, (224, 224)), cv2.COLORMAP_JET)

                            superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_img, 0.4, 0) # type: ignore

                            save_name = f"gradcam_epoch_{epoch:03d}_img_{i}_class_{label}_{classnames[int(label)]}.jpg"
                            save_path = os.path.join(maps_dir, save_name)
                            cv2.imwrite(save_path, superimposed_img)

                    gradcam_log_str = f"Saved {len(gradcams)} GradCAM visualizations to {maps_dir}"
                    print(gradcam_log_str)
                    with open(log_file, 'a') as f:
                        f.write(gradcam_log_str + '\n')
    
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    trainer.save_model(os.path.join(run_dir, 'last.pt'))
    
    if args.run_decoder and sample_images is not None and 'decoded_prompts' in locals():
        final_prompts = []
        for i in range(len(decoded_prompts)):
            image_prompts = decoded_prompts[i]
            image_path = sample_paths[i] if i < len(sample_paths) else "Unknown path"
            image_label = None
            if sample_labels is not None:
                try:
                    image_label = int(sample_labels[i])
                except Exception:
                    image_label = None

            selected_prompt = None
            if image_label is not None:
                for p in image_prompts:
                    if p.get('class_id') == image_label:
                        selected_prompt = p
                        break

            if selected_prompt is None and len(image_prompts) > 0:
                selected_prompt = image_prompts[0]

            if selected_prompt:
                final_prompts.append({
                    'image_path': image_path,
                    'image_idx': i,
                    'class_id': selected_prompt.get('class_id', 'unknown'),
                    'class_name': selected_prompt.get('class_name', 'unknown'),
                    'generated_caption': selected_prompt.get('generated_caption', '')
                })
        with open(os.path.join(run_dir, 'final_prompts.json'), 'w') as f:
            json.dump(final_prompts, f, indent=4)
    print(f"Training completed. Results written to {run_dir}")
    with open(log_file, 'a') as f:
        f.write(f"Training completed. Results written to {run_dir}\n")