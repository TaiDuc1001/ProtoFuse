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
    setup_logging,
    run_dataset_eda,
    save_class_distribution_plot,
    save_confusion_artifacts,
    visualize_attention_maps,
    visualize_gradcam_maps,
    ConfigNode,
    set_global_seed,
    build_config_namespace,
    create_argument_parser,
    process_parsed_args,
    parse_override_arguments,
    merge_configs,
    load_config_file,
    get_config_value,
    coerce_to_str,
    coerce_to_int,
    coerce_to_float,
    load_clip_to_cpu,
    CheckpointCache,
    BaseTrainingPipeline,
    log_experiment_start,
    log_experiment_metrics,
    compute_metrics,
)



ARG_SCHEMA = {
    'config': {'type': str, 'required': True, 'help': 'Path to YAML configuration file'},
    'output_dir': {'type': str, 'help': 'Override logging.output_dir from config', 'config_path': 'logging.output_dir'},
    'device': {'type': str, 'help': 'Override training.device from config', 'config_path': 'training.device'},
    'debug': {'type': bool, 'help': 'Enable debug output', 'default': False},
    'disable_coloring': {'type': bool, 'help': 'Disable colored output for log files', 'default': False},
}

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
        logger.info(f"Loading CLIP (backbone: {backbone_name})")
        
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
        
        logger.info(f"Learnable parameters: {format_params(learnable_params)} / Total: {format_params(total_params)} (FLOPs: {gflops_thop:.2f} GFLOPs)")
        
        trainable_names = set(self.model.get_trainable_parameter_names())
        for name, param in self.model.named_parameters():
            param.requires_grad_(name in trainable_names)
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)


        self.model.to(self.device)
        self.initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
    
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

                loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
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
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if 'prompt_learner_state_dict' in checkpoint:
            self.model.prompt_learner.load_state_dict(checkpoint['prompt_learner_state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Model loaded from {path}")

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

class APTTrainingPipeline:
    METHOD_NAME = "APT"

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

        run_eda_value = get_config_value(self.data_cfg, "run_eda", True)
        self.run_eda = bool(True if run_eda_value is None else run_eda_value)

        class_dist_value = get_config_value(self.training_cfg, "class_distribution", True)
        self.class_distribution_enabled = bool(True if class_dist_value is None else class_dist_value)

        base_output_value = self.logging_cfg.get("output_dir", "outputs")
        base_output = coerce_to_str(base_output_value, "outputs", key="logging.output_dir")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(base_output, timestamp)
        logger.info(f"Run directory: {self.run_dir}")
        self.selection_log_path = os.path.join(self.run_dir, 'al_selected_paths.log')
        self.config_path = os.path.join(self.run_dir, 'config.json')
        self.metrics_path = os.path.join(self.run_dir, 'metrics.json')
        self.final_prompts_path = os.path.join(self.run_dir, 'final_prompts.json')
        self.best_model_path = os.path.join(self.run_dir, 'best.pt')
        self.last_model_path = os.path.join(self.run_dir, 'last.pt')
        self.eda_dir = os.path.join(self.run_dir, 'eda')

        self.clip_mean = get_config_value(self.data_cfg, "clip_mean", [0.48145466, 0.4578275, 0.40821073])
        self.clip_std = get_config_value(self.data_cfg, "clip_std", [0.26862954, 0.26130258, 0.27577711])

        self.dataset: Optional[ImageFolder] = None
        self._val_dataset: Optional[ImageFolder] = None
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
        self.use_dn4 = bool(self.config.get('use_dn4', False))
        raw_dn4_cfg = self.config.get('dn4', ConfigNode())
        if not isinstance(raw_dn4_cfg, ConfigNode):
            raw_dn4_cfg = ConfigNode(raw_dn4_cfg)
        self.dn4_cfg = raw_dn4_cfg



        raw_base_novel_cfg = self.data_cfg.get('base_novel', ConfigNode())
        if not isinstance(raw_base_novel_cfg, ConfigNode):
            raw_base_novel_cfg = ConfigNode(raw_base_novel_cfg)
        self.base_novel_cfg = raw_base_novel_cfg
        self.base_novel_enabled = bool(self.base_novel_cfg.get('enabled', False))
        self.base_novel_split_ratio = coerce_to_float(self.base_novel_cfg.get('split_ratio', 0.5), 0.5)
        
        self.base_class_indices: List[int] = []
        self.novel_class_indices: List[int] = []
        self.base_val_loader: Optional[DataLoader] = None
        self.novel_val_loader: Optional[DataLoader] = None

        self.checkpoint_cache: Optional[CheckpointCache] = None
        self.checkpoint_id: Optional[str] = None
        self._init_checkpoint_cache()

    @property
    def val_dataset(self):
        if hasattr(self, '_val_dataset') and self._val_dataset is not None:
            return self._val_dataset
        return self.dataset

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
        first_key = next(iter(model_state.keys()), '')
        if first_key.startswith('0.'):
            self.trainer.model.prompt_learner.load_state_dict(model_state)
        else:
            prompt_state = {k.replace('prompt_learner.', ''): v for k, v in model_state.items() if k.startswith('prompt_learner.')}
            if prompt_state:
                self.trainer.model.prompt_learner.load_state_dict(prompt_state)
            else:
                self.trainer.model.load_state_dict(model_state, strict=False)
        self.trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.labeled_indices = ckpt['labeled_indices']
        self.unlabeled_indices = ckpt['unlabeled_indices']
        self.metrics = ckpt['metrics']
        self.global_epoch = len(self.metrics)
        self.cached_apt_predictions = ckpt.get('apt_predictions', None)
        logger.info(f"Loaded checkpoint: {self.checkpoint_id} (epoch {self.global_epoch})")
        return True

    def _compute_apt_predictions(self):
        if self.trainer is None or self.val_loader is None:
            return None
        self.trainer.model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                logits = self.trainer.model(images)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        return {
            'logits': torch.cat(all_logits, dim=0),
            'labels': torch.cat(all_labels, dim=0),
        }

    def _save_checkpoint(self):
        if self.checkpoint_cache is None or self.checkpoint_id is None:
            return
        if self.trainer is None:
            return
        apt_predictions = self._compute_apt_predictions()
        path = self.checkpoint_cache.save(
            self.checkpoint_id,
            self.trainer.model.prompt_learner.state_dict(),
            self.trainer.optimizer.state_dict(),
            self.trainer.scheduler.state_dict(),
            self.labeled_indices,
            self.unlabeled_indices,
            self.metrics,
            self.config,
            apt_predictions=apt_predictions
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
        method_name = "APT"
        log_experiment_start(method_name, dataset_name, self.kshot, self.seed)
        
        logger.section("APT Training", "train")
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
        if bool(get_config_value(self.training_cfg, "use_cutout", False)):
            base_transforms.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0))
            logger.debug("Using Cutout (RandomErasing) augmentation")
        return transforms.Compose(base_transforms)

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
            val_path = os.path.join(self.dataset_root, 'val')
            if not os.path.isdir(val_path):
                val_path = os.path.join(self.dataset_root, 'test')
            try:
                self.dataset = ImageFolder(train_path, transform=transform)
            except Exception as exc:
                raise RuntimeError(f"Failed to load train dataset from {train_path}: {exc}")
            try:
                self._val_dataset = ImageFolder(val_path, transform=transform)
            except Exception as exc:
                raise RuntimeError(f"Failed to load val dataset from {val_path}: {exc}")
            if self.run_eda:
                run_dataset_eda(self.dataset, self.eda_dir, sample_limit=512, seed=self.seed)

    def _split_dataset(self):
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before splitting.")
        samples_by_class_idx = defaultdict(list)
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            samples_by_class_idx[class_idx].append(idx)

        self.classnames = list(self.dataset.classes)
        all_class_indices = sorted(samples_by_class_idx.keys())

        if self.base_novel_enabled:
            self._compute_base_novel_classes(all_class_indices)
            logger.info(f"Base-to-Novel: {len(self.base_class_indices)} base classes, {len(self.novel_class_indices)} novel classes")

        rng = random.Random(self.seed)
        val_indices = []
        train_indices = []
        unlabeled_indices = []
        base_val_indices = []
        novel_val_indices = []

        if self.val_fraction is not None:
            for class_idx in all_class_indices:
                class_samples = list(samples_by_class_idx[class_idx])
                class_samples.sort()
                rng.shuffle(class_samples)

                val_count = int(math.floor(len(class_samples) * self.val_fraction))
                if self.val_fraction > 0 and val_count == 0 and len(class_samples) > 0:
                    val_count = 1

                val_part = class_samples[:val_count]
                train_candidates = class_samples[val_count:]

                if self.base_novel_enabled:
                    if class_idx in self.base_class_indices:
                        base_val_indices.extend(val_part)
                        if self.kshot > 0:
                            labeled_part = train_candidates[:self.kshot]
                            leftover_part = train_candidates[self.kshot:]
                        else:
                            labeled_part = train_candidates
                            leftover_part = []
                        train_indices.extend(labeled_part)
                        unlabeled_indices.extend(leftover_part)
                    else:
                        novel_val_indices.extend(val_part)
                else:
                    val_indices.extend(val_part)
                    if self.kshot > 0:
                        labeled_part = train_candidates[:self.kshot]
                        leftover_part = train_candidates[self.kshot:]
                    else:
                        labeled_part = train_candidates
                        leftover_part = []
                    train_indices.extend(labeled_part)
                    unlabeled_indices.extend(leftover_part)
        else:
            for class_idx in all_class_indices:
                class_samples = list(samples_by_class_idx[class_idx])
                class_samples.sort()
                rng.shuffle(class_samples)

                if self.base_novel_enabled:
                    if class_idx in self.base_class_indices:
                        if self.kshot > 0:
                            labeled_part = class_samples[:self.kshot]
                            leftover_part = class_samples[self.kshot:]
                        else:
                            labeled_part = class_samples
                            leftover_part = []
                        train_indices.extend(labeled_part)
                        unlabeled_indices.extend(leftover_part)
                else:
                    if self.kshot > 0:
                        labeled_part = class_samples[:self.kshot]
                        leftover_part = class_samples[self.kshot:]
                    else:
                        labeled_part = class_samples
                        leftover_part = []
                    train_indices.extend(labeled_part)
                    unlabeled_indices.extend(leftover_part)

        if self.base_novel_enabled:
            val_indices = base_val_indices + novel_val_indices

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

        if self.base_novel_enabled:
            if self.val_fraction is not None:
                base_val_src = self.dataset
                novel_val_src = self.dataset
            else:
                base_val_src = self._val_dataset
                novel_val_src = self._val_dataset
                base_set = set(self.base_class_indices)
                novel_set = set(self.novel_class_indices)
                base_val_indices = [i for i, (_, c) in enumerate(base_val_src.samples) if c in base_set]
                novel_val_indices = [i for i, (_, c) in enumerate(novel_val_src.samples) if c in novel_set]
            if len(base_val_indices) > 0:
                base_val_ds = Subset(base_val_src, base_val_indices)
                self.base_val_loader = DataLoader(base_val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                logger.info(f"Base validation: {len(base_val_indices)} samples")
            if len(novel_val_indices) > 0:
                novel_val_ds = Subset(novel_val_src, novel_val_indices)
                self.novel_val_loader = DataLoader(novel_val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                logger.info(f"Novel validation: {len(novel_val_indices)} samples")

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

    def _compute_base_novel_classes(self, all_class_indices):
        rng = random.Random(self.seed)
        shuffled_classes = list(all_class_indices)
        rng.shuffle(shuffled_classes)
        
        num_base = int(len(shuffled_classes) * self.base_novel_split_ratio)
        num_base = max(1, min(num_base, len(shuffled_classes) - 1))
        
        self.base_class_indices = sorted(shuffled_classes[:num_base])
        self.novel_class_indices = sorted(shuffled_classes[num_base:])

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
        self.trainer = APT(self.trainer_cfg, self.classnames, device=str(self.device))


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

        self.trainer_cfg.meta.completed_rounds = 1
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
            results = self._evaluate_with_ssl_fusion(self.val_loader)
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

        self._refresh_sample_cache(all_labels)

        if bool(get_config_value(self.training_cfg, 'visualize_attention', False)):
            attention_dir = os.path.join(epoch_dir, 'attention')
            os.makedirs(attention_dir, exist_ok=True)
            self._export_attention_overlays(attention_dir)

        if bool(get_config_value(self.training_cfg, 'visualize_gradcam', False)):
            gradcam_dir = os.path.join(epoch_dir, 'gradcam')
            os.makedirs(gradcam_dir, exist_ok=True)
            self._export_gradcam_overlays(gradcam_dir)

        epoch_time = time.time() - start_time
        epoch_result = {
            'epoch': epoch_idx,
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
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

        base_val_acc = None
        novel_val_acc = None
        harmonic_mean = None
        if self.base_novel_enabled:
            if self.base_val_loader is not None:
                base_results = self._evaluate_with_ssl_fusion(self.base_val_loader)
                base_val_acc = base_results['accuracy']
                epoch_result['base_val_acc'] = base_val_acc
            if self.novel_val_loader is not None:
                novel_results = self._evaluate_with_ssl_fusion(self.novel_val_loader)
                novel_val_acc = novel_results['accuracy']
                epoch_result['novel_val_acc'] = novel_val_acc
            if base_val_acc is not None and novel_val_acc is not None and (base_val_acc + novel_val_acc) > 0:
                harmonic_mean = 2 * base_val_acc * novel_val_acc / (base_val_acc + novel_val_acc)
                epoch_result['harmonic_mean'] = harmonic_mean

        with open(os.path.join(epoch_dir, 'result.json'), 'w') as f:
            json.dump(epoch_result, f, indent=2)

        self.metrics.append(epoch_result)

        if self.val_loader is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.trainer.save_model(self.best_model_path)

        val_acc_display = f"{val_acc:.2f}%" if self.val_loader is not None else "N/A"
        if self.base_novel_enabled and harmonic_mean is not None:
            logger.info(f"APT Epoch {epoch_idx} - loss={avg_loss:.4f} - acc={avg_acc:.2f}% - val_acc={val_acc_display} - base={base_val_acc:.2f}% - novel={novel_val_acc:.2f}% - H={harmonic_mean:.2f}% - {epoch_time:.2f}s")
        else:
            logger.info(f"APT Epoch {epoch_idx} - loss={avg_loss:.4f} - acc={avg_acc:.2f}% - val_acc={val_acc_display} - {epoch_time:.2f}s")

        if self.trainer.scheduler is not None:
            self.trainer.scheduler.step()

    def _refresh_sample_cache(self, all_labels):
        if self.val_dataset is None:
            return
        if self.val_loader is None or len(self.val_dataset) == 0:
            return

        num_display = min(10, len(self.classnames), len(self.val_dataset))
        selected_indices = []
        seen_classes = set()
        for idx in range(len(self.val_dataset)):
            cls_idx = self.val_dataset.samples[idx][1]
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
                    self.sample_cache['paths'] = [os.path.abspath(self.val_dataset.samples[idx][0]) for idx in range(len(batch_data[0]))]
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
                img, lbl = self.val_dataset[idx]
                sample_images_list.append(img)
                sample_labels_list.append(lbl)
                sample_paths.append(os.path.abspath(self.val_dataset.samples[idx][0]))

            self.sample_cache['images'] = torch.stack(sample_images_list)
            self.sample_cache['labels'] = torch.tensor(sample_labels_list)
            self.sample_cache['paths'] = sample_paths

    def _export_attention_overlays(self, maps_dir):
        if self.trainer is None:
            return
        visualize_attention_maps(
            self.trainer,
            self.val_dataset,
            self.sample_cache,
            self.classnames,
            self.global_epoch,
            maps_dir,
        )

    def _export_gradcam_overlays(self, maps_dir):
        if self.trainer is None:
            return
        visualize_gradcam_maps(
            self.trainer,
            self.val_dataset,
            self.sample_cache,
            self.classnames,
            self.global_epoch,
            maps_dir,
        )

    def _finalize(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before finalization.")
        
        with open(self.config_path, 'w') as f:
            json.dump(self.trainer_cfg.to_dict(), f, indent=4)

        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        self.trainer.save_model(self.last_model_path)

        logger.info(f"Training completed. Results written to {self.run_dir}")

        final_metrics = self.metrics[-1] if self.metrics else {}
        log_experiment_metrics(final_metrics)

BaseTrainingPipeline.register_extra_pipeline(APTTrainingPipeline)

def parse_args():
    parser = create_argument_parser("Train APT model", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides

def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True), getattr(args, 'disable_coloring', False))
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = APTTrainingPipeline(merged)
    pipeline.run()

if __name__ == "__main__":
    main()