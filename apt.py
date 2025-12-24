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
    log_experiment_start,
    log_experiment_accuracy,
)

from apt_ssl import (
    LinearClassifier,
    ImageSSLModel,
    DINOMultiCropTransform,
    FusionWeightLearner,
    create_teacher_from_student,
    update_teacher_ema,
    get_cosine_ema_momentum,
    update_center,
    dino_loss,
    visualize_dino_attention,
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

class DINODataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        from PIL import Image
        self.Image = Image

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.base_dataset.samples[real_idx]
        img = self.Image.open(path).convert('RGB')
        global_views, local_views = self.transform(img)
        return global_views, local_views, label


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

        raw_ssl_cfg = self.config.get('ssl', ConfigNode())
        if not isinstance(raw_ssl_cfg, ConfigNode):
            raw_ssl_cfg = ConfigNode(raw_ssl_cfg)
        self.ssl_cfg = raw_ssl_cfg
        self.use_ssl = bool(self.ssl_cfg.get('enabled', False))
        
        ssl_batch_value = self.ssl_cfg.get("ssl_batch_size", None)
        self.ssl_batch_size = coerce_to_int(ssl_batch_value, self.batch_size, key="ssl.ssl_batch_size")
        
        eval_batch_value = self.ssl_cfg.get("eval_batch_size", None)
        self.eval_batch_size = coerce_to_int(eval_batch_value, self.batch_size * 4, key="ssl.eval_batch_size")
        
        self.ssl_student = None
        self.ssl_teacher = None
        self.ssl_classifier = None
        self.ssl_center = None
        self.fusion_weights = None
        self.cached_apt_predictions = None

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

        dataset_name = self.config.model.dataset_name
        method_name = "ViFE" if self.use_ssl else "APT"
        log_experiment_start(method_name, dataset_name, self.kshot, self.seed)
        
        logger.section("APT Training", "train")
        self._train_epochs()
        
        if self.use_ssl:
            logger.section("SSL Stage 1: Self-Supervised Learning", "model")
            if self._try_load_ssl_stage1_checkpoint():
                logger.info("Skipping SSL Stage 1 training (loaded from checkpoint)")
            else:
                self._train_ssl_stage1()
            
            logger.section("SSL Stage 2: Linear Classifier Training", "train")
            self._train_ssl_stage2()
            
            if self.ssl_cfg.get('learn_fusion', False):
                logger.section("SSL Stage 3: Fusion Weight Learning", "train")
                self._train_ssl_stage3()
            
            logger.section("Dual-Branch Evaluation", "eval")
            eval_result = self._run_dual_branch_eval()
            self.learned_acc = eval_result.get('learned_acc') if eval_result else None
        
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

    def _train_ssl_stage1(self):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before SSL training.")
        use_labeled_for_ssl = bool(self.ssl_cfg.get('use_labeled_for_ssl', False))
        if not use_labeled_for_ssl and not self.unlabeled_indices:
            logger.warning("SSL Stage 1 skipped: no unlabeled data available")
            return

        feature_dim = self.trainer.model.vis_encoder.proj.shape[1]
        proj_dim = coerce_to_int(self.ssl_cfg.get('proj_dim', 256), 256)
        num_prototypes = coerce_to_int(self.ssl_cfg.get('num_prototypes', 4096), 4096)
        ssl_epochs = coerce_to_int(self.ssl_cfg.get('ssl_epochs', 20), 20)
        ssl_lr = coerce_to_float(self.ssl_cfg.get('ssl_lr', 0.0001), 0.0001)
        teacher_temp = coerce_to_float(self.ssl_cfg.get('teacher_temp', 0.04), 0.04)
        student_temp = coerce_to_float(self.ssl_cfg.get('student_temp', 0.1), 0.1)
        base_ema_momentum = coerce_to_float(self.ssl_cfg.get('ema_momentum', 0.996), 0.996)
        final_ema_momentum = coerce_to_float(self.ssl_cfg.get('final_ema_momentum', 1.0), 1.0)
        center_momentum = coerce_to_float(self.ssl_cfg.get('center_momentum', 0.9), 0.9)
        eval_freq = coerce_to_int(self.ssl_cfg.get('eval_freq', 5), 5)
        eval_linear_epochs = coerce_to_int(self.ssl_cfg.get('eval_linear_epochs', 5), 5)
        
        global_crop_size = coerce_to_int(self.ssl_cfg.get('global_crop_size', 224), 224)
        local_crop_size = coerce_to_int(self.ssl_cfg.get('local_crop_size', 96), 96)
        num_local_crops = coerce_to_int(self.ssl_cfg.get('num_local_crops', 6), 6)
        global_crop_scale_min = coerce_to_float(self.ssl_cfg.get('global_crop_scale_min', 0.4), 0.4)
        global_crop_scale_max = coerce_to_float(self.ssl_cfg.get('global_crop_scale_max', 1.0), 1.0)
        local_crop_scale_min = coerce_to_float(self.ssl_cfg.get('local_crop_scale_min', 0.05), 0.05)
        local_crop_scale_max = coerce_to_float(self.ssl_cfg.get('local_crop_scale_max', 0.4), 0.4)
        num_trans_layers = coerce_to_int(self.ssl_cfg.get('num_trans_layers', 1), 1)
        num_heads = coerce_to_int(self.ssl_cfg.get('num_heads', 8), 8)
        num_unlabeled = self.ssl_cfg.get('num_unlabeled', None)
        if num_unlabeled is not None:
            num_unlabeled = coerce_to_int(num_unlabeled, len(self.unlabeled_indices))
        num_plot = coerce_to_int(self.ssl_cfg.get('num_plot', 0), 0)

        logger.debug(f"SSL Stage 1 config: feature_dim={feature_dim}, proj_dim={proj_dim}, num_prototypes={num_prototypes}")
        logger.debug(f"SSL Stage 1 config: ssl_lr={ssl_lr}, teacher_temp={teacher_temp}, student_temp={student_temp}")
        logger.debug(f"SSL Stage 1 config: ema_momentum={base_ema_momentum}->{final_ema_momentum}, center_momentum={center_momentum}")
        logger.debug(f"SSL Stage 1 config: trans_layers={num_trans_layers}, heads={num_heads}, eval_freq={eval_freq}")
        logger.debug(f"Multi-crop config: global={global_crop_size}, local={local_crop_size}, num_local={num_local_crops}")

        self.ssl_student = ImageSSLModel(
            copy.deepcopy(self.trainer.model.vis_encoder),
            feature_dim,
            proj_dim,
            num_prototypes,
            num_trans_layers=num_trans_layers,
            num_heads=num_heads
        ).to(self.device)

        self.ssl_teacher = create_teacher_from_student(self.ssl_student)
        self.ssl_center = torch.zeros(num_prototypes, device=self.device)

        trainable_params = sum(p.numel() for p in self.ssl_student.parameters() if p.requires_grad)
        logger.debug(f"SSL student trainable params: {trainable_params:,}")
        logger.debug(f"SSL center shape: {self.ssl_center.shape}")

        ssl_optimizer = torch.optim.AdamW(
            [p for p in self.ssl_student.parameters() if p.requires_grad],
            lr=ssl_lr
        )

        dino_transform = DINOMultiCropTransform(
            self.clip_mean,
            self.clip_std,
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            global_crop_scale=(global_crop_scale_min, global_crop_scale_max),
            local_crop_scale=(local_crop_scale_min, local_crop_scale_max),
            num_local_crops=num_local_crops
        )

        def dino_collate_fn(batch):
            global_views_list = [[], []]
            local_views_list = [[] for _ in range(num_local_crops)]
            labels = []
            for global_views, local_views, label in batch:
                for i, gv in enumerate(global_views):
                    global_views_list[i].append(gv)
                for i, lv in enumerate(local_views):
                    local_views_list[i].append(lv)
                labels.append(label)
            global_views_batch = [torch.stack(gv_list) for gv_list in global_views_list]
            local_views_batch = [torch.stack(lv_list) for lv_list in local_views_list]
            return global_views_batch, local_views_batch, torch.tensor(labels)

        if use_labeled_for_ssl:
            ssl_data_indices = list(self.train_indices)
            logger.debug(f"Using labeled set for SSL: {len(ssl_data_indices)} samples")
        else:
            ssl_data_indices = self.unlabeled_indices
            if num_unlabeled is not None and num_unlabeled < len(ssl_data_indices):
                ssl_data_indices = ssl_data_indices[:num_unlabeled]
                logger.debug(f"Limited unlabeled samples to {num_unlabeled}")

        ssl_dataset = DINODataset(self.dataset, ssl_data_indices, dino_transform)
        ssl_loader = DataLoader(
            ssl_dataset,
            batch_size=self.ssl_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=dino_collate_fn
        )

        self.ssl_vis_images = None
        self.ssl_vis_paths = None
        if num_plot > 0:
            class_to_indices = defaultdict(list)
            for idx in ssl_data_indices:
                _, label = self.dataset.samples[idx]
                class_to_indices[label].append(idx)
            
            all_classes = list(class_to_indices.keys())
            num_classes_to_sample = min(num_plot, len(all_classes))
            
            rng = random.Random(self.seed)
            selected_classes = rng.sample(all_classes, num_classes_to_sample)
            
            vis_indices = []
            for cls in selected_classes:
                class_indices = class_to_indices[cls]
                vis_indices.append(rng.choice(class_indices))
            
            self.ssl_vis_images = []
            self.ssl_vis_paths = []
            vis_transform = transforms.Compose([
                transforms.Resize((global_crop_size, global_crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.clip_mean, std=self.clip_std),
            ])
            for idx in vis_indices:
                path, _ = self.dataset.samples[idx]
                img = PILImage.open(path).convert('RGB')
                self.ssl_vis_images.append(vis_transform(img))
                self.ssl_vis_paths.append(path)
            self.ssl_vis_images = torch.stack(self.ssl_vis_images)

        logger.info(f"SSL Stage 1: {ssl_epochs} epochs on {len(ssl_data_indices)} samples")
        for epoch in range(1, ssl_epochs + 1):
            if self.ssl_student is None:
                logger.warning("SSL student is None, skipping epoch")
                return
            self.ssl_student.train()
            running_loss = 0.0
            steps = 0
            start_time = time.time()
            
            ema_momentum = get_cosine_ema_momentum(epoch, ssl_epochs, base_ema_momentum, final_ema_momentum)
            
            for global_views_batch, local_views_batch, _ in ssl_loader:
                global_views_batch = [g.to(self.device) for g in global_views_batch]
                local_views_batch = [local_view.to(self.device) for local_view in local_views_batch]

                with torch.no_grad():
                    teacher_global_cat = torch.cat(global_views_batch, dim=0)
                    u_t_cat, _ = self.ssl_teacher(teacher_global_cat)
                    teacher_outputs = list(u_t_cat.chunk(len(global_views_batch), dim=0))

                student_global_cat = torch.cat(global_views_batch, dim=0)
                u_s_global, _ = self.ssl_student(student_global_cat)
                student_global_outputs = list(u_s_global.chunk(len(global_views_batch), dim=0))
                
                student_local_cat = torch.cat(local_views_batch, dim=0)
                u_s_local, _ = self.ssl_student(student_local_cat)
                student_local_outputs = list(u_s_local.chunk(len(local_views_batch), dim=0))
                
                student_outputs = student_global_outputs + student_local_outputs

                loss_value = dino_loss(teacher_outputs, student_outputs, teacher_temp, student_temp, self.ssl_center)
                if not isinstance(loss_value, torch.Tensor):
                    loss_tensor = torch.tensor(loss_value, device=self.device, dtype=torch.float32, requires_grad=True)
                else:
                    loss_tensor = loss_value

                ssl_optimizer.zero_grad()
                loss_tensor.backward()
                ssl_optimizer.step()

                with torch.no_grad():
                    batch_logits = torch.cat(teacher_outputs, dim=0)
                    self.ssl_center = update_center(self.ssl_center, batch_logits, center_momentum)

                update_teacher_ema(self.ssl_teacher, self.ssl_student, ema_momentum)

                loss_item = loss_tensor.item() if isinstance(loss_tensor, torch.Tensor) else float(loss_tensor)
                running_loss += loss_item
                steps += 1

            avg_loss = running_loss / max(1, steps)
            epoch_time = time.time() - start_time

            ssl1_epoch_dir = os.path.join(self.run_dir, 'ssl_stage1', f'epoch_{epoch:03d}')
            os.makedirs(ssl1_epoch_dir, exist_ok=True)
            epoch_result = {'epoch': epoch, 'loss': avg_loss, 'time': epoch_time}
            with open(os.path.join(ssl1_epoch_dir, 'result.json'), 'w') as f:
                json.dump(epoch_result, f, indent=2)

            logger.info(f"  SSL1 Epoch {epoch}/{ssl_epochs} - loss={avg_loss:.4f} - time={epoch_time:.2f}s")

            if epoch % eval_freq == 0 or epoch == ssl_epochs:
                eval_start = time.time()
                linear_acc = self._run_linear_eval(feature_dim, eval_linear_epochs)
                logger.info(f"  [EVAL] Linear acc: {linear_acc:.2f}% - time: {time.time() - eval_start:.2f}s")
                
                if self.ssl_vis_images is not None:
                    ssl_attn_dir = os.path.join(ssl1_epoch_dir, 'ssl_attention')
                    os.makedirs(ssl_attn_dir, exist_ok=True)
                    visualize_dino_attention(
                        self.ssl_student,
                        self.ssl_vis_images,
                        self.ssl_vis_paths,
                        epoch,
                        ssl_attn_dir,
                        self.clip_mean,
                        self.clip_std
                    )

        if self.ssl_cfg.get('save_stage1_checkpoint', False):
            self._save_ssl_stage1_checkpoint()

    def _get_ssl_stage1_checkpoint_path(self):
        ssl_key_settings = {
            'ssl_epochs': self.ssl_cfg.get('ssl_epochs'),
            'ssl_lr': self.ssl_cfg.get('ssl_lr'),
            'ssl_batch_size': self.ssl_cfg.get('ssl_batch_size'),
            'proj_dim': self.ssl_cfg.get('proj_dim'),
            'num_prototypes': self.ssl_cfg.get('num_prototypes'),
            'teacher_temp': self.ssl_cfg.get('teacher_temp'),
            'student_temp': self.ssl_cfg.get('student_temp'),
            'ema_momentum': self.ssl_cfg.get('ema_momentum'),
            'final_ema_momentum': self.ssl_cfg.get('final_ema_momentum'),
            'center_momentum': self.ssl_cfg.get('center_momentum'),
            'num_trans_layers': self.ssl_cfg.get('num_trans_layers'),
            'num_heads': self.ssl_cfg.get('num_heads'),
            'global_crop_size': self.ssl_cfg.get('global_crop_size'),
            'local_crop_size': self.ssl_cfg.get('local_crop_size'),
            'num_local_crops': self.ssl_cfg.get('num_local_crops'),
            'global_crop_scale_min': self.ssl_cfg.get('global_crop_scale_min'),
            'global_crop_scale_max': self.ssl_cfg.get('global_crop_scale_max'),
            'local_crop_scale_min': self.ssl_cfg.get('local_crop_scale_min'),
            'local_crop_scale_max': self.ssl_cfg.get('local_crop_scale_max'),
            'num_unlabeled': self.ssl_cfg.get('num_unlabeled'),
            'use_labeled_for_ssl': self.ssl_cfg.get('use_labeled_for_ssl'),
        }
        key_settings = {
            'dataset_root': self.data_cfg.get('root'),
            'kshot': self.data_cfg.get('kshot'),
            'seed': self.data_cfg.get('seed'),
            'backbone': self.model_cfg.get('backbone'),
            'ssl': json.dumps(ssl_key_settings, sort_keys=True),
        }
        key_str = json.dumps(key_settings, sort_keys=True)
        ssl_checkpoint_id = hashlib.md5(key_str.encode()).hexdigest()[:16]
        ssl1_dir = os.path.join('checkpoints/apt_ssl', ssl_checkpoint_id)
        os.makedirs(ssl1_dir, exist_ok=True)
        return os.path.join(ssl1_dir, 'checkpoint.pt')

    def _save_ssl_stage1_checkpoint(self):
        if self.ssl_student is None or self.ssl_teacher is None:
            return
        checkpoint_path = self._get_ssl_stage1_checkpoint_path()
        student_trainable = {
            'adapter': self.ssl_student.adapter.state_dict(),
            'ssl_head': self.ssl_student.ssl_head.state_dict(),
        }
        teacher_trainable = {
            'adapter': self.ssl_teacher.adapter.state_dict(),
            'ssl_head': self.ssl_teacher.ssl_head.state_dict(),
        }
        torch.save({
            'ssl_student_trainable': student_trainable,
            'ssl_teacher_trainable': teacher_trainable,
            'ssl_center': self.ssl_center,
        }, checkpoint_path)
        logger.info(f"Saved SSL Stage 1 checkpoint to: {checkpoint_path}")

    def _try_load_ssl_stage1_checkpoint(self):
        checkpoint_path = self._get_ssl_stage1_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            return False
        
        if self.trainer is None:
            return False
        
        feature_dim = self.trainer.model.vis_encoder.proj.shape[1]
        proj_dim = coerce_to_int(self.ssl_cfg.get('proj_dim', 256), 256)
        num_prototypes = coerce_to_int(self.ssl_cfg.get('num_prototypes', 4096), 4096)
        num_trans_layers = coerce_to_int(self.ssl_cfg.get('num_trans_layers', 1), 1)
        num_heads = coerce_to_int(self.ssl_cfg.get('num_heads', 8), 8)

        self.ssl_student = ImageSSLModel(
            copy.deepcopy(self.trainer.model.vis_encoder),
            feature_dim,
            proj_dim,
            num_prototypes,
            num_trans_layers=num_trans_layers,
            num_heads=num_heads
        ).to(self.device)
        self.ssl_teacher = create_teacher_from_student(self.ssl_student)
        self.ssl_center = torch.zeros(num_prototypes, device=self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if 'ssl_student_trainable' in ckpt:
            self.ssl_student.adapter.load_state_dict(ckpt['ssl_student_trainable']['adapter'])
            self.ssl_student.ssl_head.load_state_dict(ckpt['ssl_student_trainable']['ssl_head'])
            self.ssl_teacher.adapter.load_state_dict(ckpt['ssl_teacher_trainable']['adapter'])
            self.ssl_teacher.ssl_head.load_state_dict(ckpt['ssl_teacher_trainable']['ssl_head'])
        else:
            self.ssl_student.load_state_dict(ckpt['ssl_student_state_dict'])
            self.ssl_teacher.load_state_dict(ckpt['ssl_teacher_state_dict'])
        self.ssl_center = ckpt['ssl_center'].to(self.device)
        logger.info(f"Loaded SSL Stage 1 checkpoint from: {checkpoint_path}")
        return True

    def _run_linear_eval(self, feature_dim, num_epochs):
        if not self.train_indices or self.val_loader is None:
            return 0.0
        if self.ssl_student is None or self.dataset is None:
            logger.warning("Linear evaluation skipped: SSL student or dataset not available")
            return 0.0

        eval_classifier = LinearClassifier(feature_dim, len(self.classnames)).to(self.device)
        eval_optimizer = torch.optim.AdamW(eval_classifier.parameters(), lr=0.001)

        train_subset = Subset(self.dataset, list(self.train_indices))
        train_loader = DataLoader(train_subset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)

        self.ssl_student.encoder.eval()
        
        train_features, train_labels = [], []
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                _, cls_feat = self.ssl_student(images)
                train_features.append(cls_feat.cpu())
                train_labels.append(labels)
        train_features = torch.cat(train_features, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        
        val_features, val_labels = [], []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                _, cls_feat = self.ssl_student(images)
                val_features.append(cls_feat.cpu())
                val_labels.append(labels)
        val_features = torch.cat(val_features, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        cached_train_loader = DataLoader(train_dataset, batch_size=self.eval_batch_size, shuffle=True)
        
        for epoch_idx in range(num_epochs):
            eval_classifier.train()
            for feats, labels in cached_train_loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                logits = eval_classifier(feats)
                loss = F.cross_entropy(logits, labels)
                eval_optimizer.zero_grad()
                loss.backward()
                eval_optimizer.step()

        eval_classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            val_feats = val_features.to(self.device)
            val_lbls = val_labels.to(self.device)
            logits = eval_classifier(val_feats)
            _, predicted = torch.max(logits, 1)
            correct = (predicted == val_lbls).sum().item()
            total = val_lbls.size(0)
            
        return 100 * correct / total if total > 0 else 0.0


    def _train_ssl_stage2(self):
        if self.dataset is None or self.trainer is None or self.ssl_student is None:
            logger.warning("SSL Stage 2 skipped: Stage 1 not completed")
            return
        if not self.train_indices:
            logger.warning("SSL Stage 2 skipped: no labeled data available")
            return

        feature_dim = self.trainer.model.vis_encoder.proj.shape[1]
        linear_epochs = coerce_to_int(self.ssl_cfg.get('linear_epochs', 10), 10)
        linear_lr = coerce_to_float(self.ssl_cfg.get('linear_lr', 0.001), 0.001)

        logger.debug(f"SSL Stage 2 config: feature_dim={feature_dim}, epochs={linear_epochs}, lr={linear_lr}")

        if self.ssl_student is None:
            logger.warning("SSL student is None, skipping stage 2")
            return
        
        for param in self.ssl_student.encoder.parameters():
            param.requires_grad = False

        self.ssl_classifier = LinearClassifier(feature_dim, len(self.classnames)).to(self.device)
        linear_optimizer = torch.optim.AdamW(self.ssl_classifier.parameters(), lr=linear_lr)

        logger.debug(f"SSL classifier params: {sum(p.numel() for p in self.ssl_classifier.parameters()):,}")
        logger.debug(f"Encoder frozen: {not any(p.requires_grad for p in self.ssl_student.encoder.parameters())}")

        train_subset = Subset(self.dataset, list(self.train_indices))
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        logger.info(f"SSL Stage 2: {linear_epochs} epochs on {len(self.train_indices)} labeled samples")
        for epoch in range(1, linear_epochs + 1):
            self.ssl_student.encoder.eval()
            self.ssl_classifier.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    _, cls_feat = self.ssl_student(images)

                logits = self.ssl_classifier(cls_feat)
                loss = F.cross_entropy(logits, labels)

                linear_optimizer.zero_grad()
                loss.backward()
                linear_optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            avg_loss = running_loss / max(1, len(train_loader))
            acc = 100 * correct / total

            ssl2_epoch_dir = os.path.join(self.run_dir, 'ssl_stage2', f'epoch_{epoch:03d}')
            os.makedirs(ssl2_epoch_dir, exist_ok=True)
            epoch_result = {'epoch': epoch, 'loss': avg_loss, 'acc': acc}
            with open(os.path.join(ssl2_epoch_dir, 'result.json'), 'w') as f:
                json.dump(epoch_result, f, indent=2)

            logger.info(f"  SSL2 Epoch {epoch}/{linear_epochs} - loss={avg_loss:.4f} - acc={acc:.2f}%")

    def _train_ssl_stage3(self):
        if self.dataset is None or self.trainer is None:
            logger.warning("SSL Stage 3 skipped: pipeline not initialized")
            return
        if self.ssl_student is None or self.ssl_classifier is None:
            logger.warning("SSL Stage 3 skipped: Stage 2 not completed")
            return
        if not self.train_indices:
            logger.warning("SSL Stage 3 skipped: no labeled data")
            return

        fusion_max_iter = coerce_to_int(self.ssl_cfg.get('fusion_max_iter', 20), 20)
        fusion_lr = coerce_to_float(self.ssl_cfg.get('fusion_lr', 1.0), 1.0)

        logger.info(f"SSL Stage 3: Training fusion weights (max_iter={fusion_max_iter}, lr={fusion_lr})")

        self.fusion_weights = FusionWeightLearner().to(self.device)

        self.trainer.model.eval()
        self.ssl_student.encoder.eval()
        self.ssl_classifier.eval()

        train_subset = Subset(self.dataset, list(self.train_indices))
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        all_apt_logits = []
        all_img_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits_apt = self.trainer.model(images)
                if isinstance(logits_apt, (list, tuple)):
                    logits_apt = logits_apt[0]

                _, cls_feat = self.ssl_student(images)
                logits_img = self.ssl_classifier(cls_feat)

                all_apt_logits.append(logits_apt)
                all_img_logits.append(logits_img)
                all_labels.append(labels)

        all_apt_logits = torch.cat(all_apt_logits, dim=0)
        all_img_logits = torch.cat(all_img_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        fusion_optimizer = torch.optim.LBFGS(
            self.fusion_weights.parameters(),
            lr=fusion_lr,
            max_iter=fusion_max_iter,
            line_search_fn='strong_wolfe'
        )

        iteration = [0]

        def closure():
            if self.fusion_weights is None:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            fusion_optimizer.zero_grad()
            logits_fused = self.fusion_weights(all_apt_logits, all_img_logits)
            loss = F.cross_entropy(logits_fused, all_labels)
            loss.backward()
            iteration[0] += 1
            if iteration[0] % 5 == 0 or iteration[0] == 1:
                with torch.no_grad():
                    _, predicted = torch.max(logits_fused, 1)
                    acc = 100 * (predicted == all_labels).sum().item() / len(all_labels)
                    w1, w2 = self.fusion_weights.get_weights()
                    logger.info(f"  LBFGS iter {iteration[0]} - loss={loss.item():.4f} - acc={acc:.2f}% - w1={w1:.4f}, w2={w2:.4f}")
            return loss

        self.fusion_weights.train()
        fusion_optimizer.step(closure)

        w1, w2 = self.fusion_weights.get_weights()
        with torch.no_grad():
            logits_fused = self.fusion_weights(all_apt_logits, all_img_logits)
            _, predicted = torch.max(logits_fused, 1)
            train_acc = 100 * (predicted == all_labels).sum().item() / len(all_labels)

        ssl3_dir = os.path.join(self.run_dir, 'ssl_stage3')
        os.makedirs(ssl3_dir, exist_ok=True)
        stage3_result = {'w1': w1, 'w2': w2, 'train_acc': train_acc}
        with open(os.path.join(ssl3_dir, 'result.json'), 'w') as f:
            json.dump(stage3_result, f, indent=2)

        logger.info(f"SSL Stage 3: w1={w1:.4f}, w2={w2:.4f}, train_acc={train_acc:.2f}%")

    def _run_dual_branch_eval(self):
        if self.val_loader is None:
            logger.warning("Dual-branch evaluation skipped: no validation data")
            return
        if self.trainer is None:
            logger.warning("Dual-branch evaluation skipped: trainer not initialized")
            return

        self.trainer.model.eval()

        use_ssl_branch = self.ssl_student is not None and self.ssl_classifier is not None

        if use_ssl_branch and self.ssl_student is not None and self.ssl_classifier is not None:
            self.ssl_student.encoder.eval()
            self.ssl_classifier.eval()

        if self.cached_apt_predictions is not None:
            all_apt_logits = self.cached_apt_predictions['logits']
            all_labels = self.cached_apt_predictions['labels']
            logger.debug("Using cached APT predictions")
        else:
            all_apt_logits = []
            all_labels_list = []
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.to(self.device)
                    logits_apt = self.trainer.model(images)
                    if isinstance(logits_apt, (list, tuple)):
                        logits_apt = logits_apt[0]
                    all_apt_logits.append(logits_apt.cpu())
                    all_labels_list.append(labels)
            all_apt_logits = torch.cat(all_apt_logits, dim=0)
            all_labels = torch.cat(all_labels_list, dim=0)
            self.cached_apt_predictions = {'logits': all_apt_logits, 'labels': all_labels}
            self._save_checkpoint()
            logger.debug("Computed and saved APT predictions to checkpoint")

        all_img_logits = []
        if use_ssl_branch and self.ssl_student is not None and self.ssl_classifier is not None:
            with torch.no_grad():
                for images, _ in self.val_loader:
                    images = images.to(self.device)
                    _, cls_feat = self.ssl_student(images)
                    logits_img = self.ssl_classifier(cls_feat)
                    all_img_logits.append(logits_img.cpu())

        all_apt_probs = F.softmax(all_apt_logits, dim=-1)
        _, pred_apt = torch.max(all_apt_probs, 1)
        apt_acc = 100 * (pred_apt == all_labels).sum().item() / len(all_labels)
        logger.info(f"APT Branch Accuracy: {apt_acc:.2f}%")

        if use_ssl_branch:
            all_img_logits = torch.cat(all_img_logits, dim=0)

            all_img_probs = F.softmax(all_img_logits, dim=-1)
            _, pred_img = torch.max(all_img_probs, 1)
            img_acc = 100 * (pred_img == all_labels).sum().item() / len(all_labels)
            logger.info(f"Image Branch Accuracy: {img_acc:.2f}%")

            apt_correct = (pred_apt == all_labels)
            img_correct = (pred_img == all_labels)
            both_correct = (apt_correct & img_correct).sum().item()
            apt_only_correct = (apt_correct & ~img_correct).sum().item()
            img_only_correct = (~apt_correct & img_correct).sum().item()
            both_wrong = (~apt_correct & ~img_correct).sum().item()
            
            logger.debug(f"Both correct: {both_correct} ({100*both_correct/len(all_labels):.1f}%), APT only: {apt_only_correct}, Img only: {img_only_correct}, Both wrong: {both_wrong}")
            logger.debug(f"Disagreement rate: {100*(apt_correct != img_correct).float().mean():.2f}%")

            if self.fusion_weights is not None:
                self.fusion_weights.eval()
                with torch.no_grad():
                    logits_fused = self.fusion_weights(all_apt_logits, all_img_logits)
                prob_fused = F.softmax(logits_fused, dim=-1)
                _, pred_fused = torch.max(prob_fused, 1)
                pred_fused = pred_fused.cpu()
                learned_acc = 100 * (pred_fused == all_labels).sum().item() / len(all_labels)
                w1, w2 = self.fusion_weights.get_weights()
                logger.info(f"Learned Fusion Accuracy: {learned_acc:.2f}% (w1={w1:.4f}, w2={w2:.4f})")
            else:
                learned_acc = None

            logger.info("Fusion Weight Grid:")

            best_weight = 0.0
            best_fused_acc = apt_acc
            fusion_results = {}

            for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                logits_fused = (1 - w) * all_apt_logits + w * all_img_logits
                prob_fused = F.softmax(logits_fused, dim=-1)
                _, pred_fused = torch.max(prob_fused, 1)
                fused_acc = 100 * (pred_fused == all_labels).sum().item() / len(all_labels)
                fusion_results[w] = fused_acc
                logger.info(f"  w={w:.1f} (APT:{100*(1-w):.0f}% IMG:{100*w:.0f}%) → {fused_acc:.2f}%")

                if fused_acc > best_fused_acc:
                    best_fused_acc = fused_acc
                    best_weight = w

            if learned_acc is not None:
                logger.info(f"Learned (w1,w2): {learned_acc:.2f}% (+{learned_acc - apt_acc:.2f}%)")
            if best_weight > 0:
                logger.info(f"Best scalar: w={best_weight:.1f} → {best_fused_acc:.2f}% (+{best_fused_acc - apt_acc:.2f}%)")
            else:
                logger.info(f"Best: APT only → {apt_acc:.2f}%")
        else:
            logger.info("Image Branch: Not available (SSL not trained)")
            fusion_results = {}
            best_weight = None
            best_fused_acc = None
            learned_acc = None

        eval_result = {
            'apt_acc': apt_acc,
            'img_acc': img_acc if use_ssl_branch else None,
            'fusion_results': fusion_results,
            'best_weight': best_weight,
            'best_fused_acc': best_fused_acc,
            'learned_acc': learned_acc,
        }
        eval_dir = os.path.join(self.run_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, 'result.json'), 'w') as f:
            json.dump(eval_result, f, indent=2)

        return eval_result

    def _evaluate_with_ssl_fusion(self, dataloader):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized.")

        self.trainer.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        steps = 0
        all_preds = []
        all_labels_list = []

        use_ssl_fusion = self.use_ssl and self.ssl_student is not None and self.ssl_classifier is not None

        if use_ssl_fusion and self.ssl_student is not None and self.ssl_classifier is not None:
            self.ssl_student.encoder.eval()
            self.ssl_classifier.eval()

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits_apt = self.trainer.model(images)
                if isinstance(logits_apt, (list, tuple)):
                    logits_apt = logits_apt[0]

                if use_ssl_fusion and self.fusion_weights is not None and self.ssl_student is not None and self.ssl_classifier is not None:
                    _, cls_feat = self.ssl_student(images)
                    logits_img = self.ssl_classifier(cls_feat)
                    self.fusion_weights.eval()
                    logits = self.fusion_weights(logits_apt, logits_img)
                else:
                    logits = logits_apt

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
            'time': epoch_time
        }
        with open(os.path.join(epoch_dir, 'result.json'), 'w') as f:
            json.dump(epoch_result, f, indent=2)

        self.metrics.append(epoch_result)

        if self.val_loader is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.trainer.save_model(self.best_model_path)

        val_acc_display = f"{val_acc:.2f}%" if self.val_loader is not None else "N/A"
        logger.info(f"APT Epoch {epoch_idx} - loss={avg_loss:.4f} - acc={avg_acc:.2f}% - val_acc={val_acc_display} - {epoch_time:.2f}s")

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

        if self.use_ssl and hasattr(self, 'learned_acc') and self.learned_acc is not None:
            final_acc = self.learned_acc
        else:
            final_acc = self.metrics[-1]['val_acc'] if self.metrics else 0.0
        log_experiment_accuracy(final_acc)

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