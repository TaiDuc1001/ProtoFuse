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
from apt import parse_override_arguments, merge_configs


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


class CLIPZeroShotMIM(nn.Module):
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
        num_heads = self.model_cfg.get('mim_num_heads', 8)
        dropout = self.model_cfg.get('mim_dropout', 0.1)
        self.mask_ratio = self.model_cfg.get('mask_ratio', 0.4)
        self.mim_module = MIMModule(feature_dim, hidden_dim, num_encoder_layers, num_decoder_layers, num_heads, dropout)
        self.logit_scale = clip_model.logit_scale
        dataset_name = self.model_cfg.get('dataset_name', 'ImageNet')
        template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")
        self.classnames = classnames
        self.text_features = self._encode_text_features(template, classnames)

    def _encode_text_features(self, template, classnames):
        prompts = [template.format(c.replace('_', ' ')) for c in classnames]
        tokens = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

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

    def forward_zeroshot(self, images):
        with torch.no_grad():
            _, cls_feature = self.image_encoder(images)
            cls_feature = F.normalize(cls_feature, dim=-1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * cls_feature @ self.text_features.t()
        return logits

    def trainable_parameters(self):
        return self.mim_module.parameters()


class MIMTrainer:
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
        self.model = CLIPZeroShotMIM(cfg, classnames, clip_model, self.device)
        self.model.to(self.device)
        self.setup_optimizer()
        trainable_params = sum(p.numel() for p in self.model.mim_module.parameters())
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"MIM trainable parameters: {trainable_params:,} / Total parameters: {total_params:,}")

    def setup_optimizer(self):
        lr = coerce_to_float(self.training_cfg.get('learning_rate', 0.0001), 0.0001)
        weight_decay = coerce_to_float(self.training_cfg.get('weight_decay', 0.01), 0.01)
        optimizer_type = self.training_cfg.get('optimizer', 'AdamW')
        trainable_params = list(self.model.trainable_parameters())
        if optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        num_epochs = coerce_to_int(self.training_cfg.get('epochs', 20), 20)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

    def train_step(self, images):
        self.model.mim_module.train()
        images = images.to(self.device)
        loss, _, _, _ = self.model.forward_mim(images)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model.forward_zeroshot(images)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = 100 * correct / total if total > 0 else 0.0
        return {"accuracy": accuracy, "predictions": all_preds, "true_labels": all_labels}

    def save_model(self, path):
        checkpoint = {
            'mim_module_state_dict': self.model.mim_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.mim_module.load_state_dict(checkpoint['mim_module_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Model loaded from {path}")


class MIMTrainingPipeline:
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
        self.best_model_path = os.path.join(self.run_dir, 'best.pt')
        self.last_model_path = os.path.join(self.run_dir, 'last.pt')
        self.log_file = os.path.join(self.run_dir, 'training.log')
        self.clip_mean = get_config_value(self.data_cfg, "clip_mean", [0.48145466, 0.4578275, 0.40821073])
        self.clip_std = get_config_value(self.data_cfg, "clip_std", [0.26862954, 0.26130258, 0.27577711])
        self.dataset: Optional[ImageFolder] = None
        self.val_loader: Optional[DataLoader] = None
        self.unlabeled_loader: Optional[DataLoader] = None
        self.classnames: List[str] = []
        self.val_indices: List[int] = []
        self.labeled_indices: List[int] = []
        self.unlabeled_indices: List[int] = []
        self.metrics: List[Dict[str, Any]] = []
        self.best_val_acc = -float('inf')
        self.global_epoch = 0
        self.trainer: Optional[MIMTrainer] = None

    def run(self):
        self._prepare_directories()
        self._load_dataset()
        self._split_dataset()
        self._initialize_trainer()
        self._train_epochs()
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
                labeled_part = []
                leftover_part = train_candidates
            val_indices.extend(val_part)
            labeled_indices.extend(labeled_part)
            unlabeled_indices.extend(leftover_part)
        self.val_indices = val_indices
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        if len(self.val_indices) > 0:
            val_ds = Subset(self.dataset, self.val_indices)
            self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        if len(self.unlabeled_indices) > 0:
            unlabeled_ds = Subset(self.dataset, self.unlabeled_indices)
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
        self.trainer = MIMTrainer(self.config, self.classnames, device=str(self.device), log_file=self.log_file)

    def _train_epochs(self):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before training.")
        if not self.unlabeled_indices:
            raise RuntimeError("No unlabeled samples available for MIM training.")
        epochs_total = coerce_to_int(self.training_cfg.get('epochs', 20), 20)
        for epoch_idx in range(1, epochs_total + 1):
            self._run_epoch(epoch_idx, epochs_total)

    def _run_epoch(self, epoch_idx, epochs_total):
        if self.trainer is None or self.unlabeled_loader is None:
            raise RuntimeError("Trainer or unlabeled loader not initialized.")
        self.global_epoch += 1
        start_time = time.time()
        self.trainer.model.mim_module.train()
        running_loss = 0.0
        steps = 0
        for batch in self.unlabeled_loader:
            images, _ = batch
            loss_dict = self.trainer.train_step(images)
            running_loss += loss_dict['loss']
            steps += 1
        avg_loss = running_loss / max(1, steps)
        val_acc = 0.0
        if self.val_loader is not None:
            results = self.trainer.evaluate(self.val_loader)
            val_acc = results['accuracy']
        epoch_time = time.time() - start_time
        self.metrics.append({
            'epoch': self.global_epoch,
            'mim_loss': avg_loss,
            'val_acc': val_acc,
            'time': epoch_time
        })
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.trainer.save_model(self.best_model_path)
        epoch_str = f"Epoch {epoch_idx}/{epochs_total} - mim_loss={avg_loss:.4f} - val_acc={val_acc:.2f}% - time={epoch_time:.2f}s"
        print(epoch_str)
        with open(self.log_file, 'a') as f:
            f.write(epoch_str + '\n')
        if self.trainer.scheduler is not None:
            self.trainer.scheduler.step()

    def _finalize(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before finalization.")
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.trainer.save_model(self.last_model_path)
        print(f"Training completed. Best val accuracy: {self.best_val_acc:.2f}%")
        print(f"Results written to {self.run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Zero-Shot with MIM")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    base_config = load_config_file(args.config)
    merged_config = merge_configs(base_config, overrides)
    pipeline = MIMTrainingPipeline(merged_config)
    pipeline.run()


if __name__ == "__main__":
    main()
