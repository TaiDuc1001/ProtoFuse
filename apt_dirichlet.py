import os
import time
import math
import copy
import json
import torch
import random
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
from thop import profile
from typing import Any, Dict, List, Optional

from apt import (
    ConfigNode,
    load_config_file,
    merge_configs,
    parse_override_arguments,
    create_argument_parser,
    process_parsed_args,
    build_config_namespace,
    get_config_value,
    coerce_to_str,
    coerce_to_int,
    coerce_to_float,
    load_clip_to_cpu,
    CustomCLIP,
    DEFAULT_TRAINING_EPOCHS,
    set_global_seed,
)

ARG_SCHEMA = {
    'config': {'type': str, 'required': True, 'help': 'Path to YAML configuration file'},
    'output_dir': {'type': str, 'help': 'Override logging.output_dir from config', 'config_path': 'logging.output_dir'},
    'device': {'type': str, 'help': 'Override training.device from config', 'config_path': 'training.device'},
}


class DirichletHead(nn.Module):
    def __init__(self, num_classes: int, init_concentration: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.concentration_layer = nn.Linear(num_classes, num_classes)
        nn.init.eye_(self.concentration_layer.weight)
        nn.init.constant_(self.concentration_layer.bias, init_concentration)

    def forward(self, softmax_probs: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.concentration_layer(softmax_probs)) + 1e-6
        return alpha

    def expected_probability(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha_sum = alpha.sum(dim=-1, keepdim=True)
        return alpha / alpha_sum

    def evidence_loss(
        self, alpha: torch.Tensor, targets: torch.Tensor, kl_weight: float = 0.1
    ) -> torch.Tensor:
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        s_val = alpha.sum(dim=-1, keepdim=True)
        ce_loss = (one_hot * (torch.digamma(s_val) - torch.digamma(alpha))).sum(dim=-1).mean()
        alpha_tilde = one_hot + (1.0 - one_hot) * alpha
        kl_loss = self._kl_divergence(alpha_tilde)
        return ce_loss + kl_weight * kl_loss

    def _kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        prior = torch.ones_like(alpha)
        prior_0 = prior.sum(dim=-1, keepdim=True)
        kl = (
            torch.lgamma(alpha_0)
            - torch.lgamma(prior_0)
            - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
            + torch.lgamma(prior).sum(dim=-1, keepdim=True)
            + ((alpha - prior) * (torch.digamma(alpha) - torch.digamma(alpha_0))).sum(dim=-1, keepdim=True)
        )
        return kl.mean()


class CustomCLIPDirichlet(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.base_model = CustomCLIP(cfg, classnames, clip_model, device)
        self.dirichlet_head = DirichletHead(
            num_classes=len(classnames),
            init_concentration=cfg.get('dirichlet', {}).get('init_concentration', 1.0)
        )
        self.cfg = cfg
        self.device = device
        self.classnames = classnames
        self.kl_weight = cfg.get('dirichlet', {}).get('kl_weight', 0.1)

    def forward(self, image, label=None):
        if self.training and label is not None:
            loss, logits = self.base_model(image, label)
            softmax_probs = F.softmax(logits, dim=-1)
            alpha = self.dirichlet_head(softmax_probs)
            dirichlet_loss = self.dirichlet_head.evidence_loss(alpha, label, self.kl_weight)
            total_loss = loss + dirichlet_loss
            return total_loss, logits, alpha
        else:
            logits = self.base_model(image)
            if isinstance(logits, tuple):
                logits = logits[0]
            softmax_probs = F.softmax(logits, dim=-1)
            alpha = self.dirichlet_head(softmax_probs)
            return logits, alpha

    @property
    def prompt_learner(self):
        return self.base_model.prompt_learner

    @property
    def vis_encoder(self):
        return self.base_model.vis_encoder

    @property
    def logit_scale(self):
        return self.base_model.logit_scale

    def trainable_parameters(self):
        for param in self.base_model.trainable_parameters():
            yield param
        for param in self.dirichlet_head.parameters():
            yield param

    def get_trainable_parameter_names(self):
        base_names = [f"base_model.{name}" for name in self.base_model.get_trainable_parameter_names()]
        dirichlet_names = [f"dirichlet_head.{name}" for name, _ in self.dirichlet_head.named_parameters()]
        return base_names + dirichlet_names


class APTDirichlet:
    def __init__(self, cfg, classnames, device="cuda", log_file=None):
        if not isinstance(cfg, ConfigNode):
            cfg = ConfigNode(cfg)
        self.cfg = cfg
        self.training_cfg = self.cfg.get('training', ConfigNode())
        self.model_cfg = self.cfg.get('model', ConfigNode())
        self.data_cfg = self.cfg.get('data', ConfigNode())
        self.classnames = classnames
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.log_file = log_file

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
        backbone_name = self._cfg_str('ViT-B/32', 'model.backbone', 'backbone')
        msg = f"Loading CLIP (backbone: {backbone_name})"
        print(msg)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')

        clip_model = load_clip_to_cpu(backbone_name)

        if self._cfg_str('fp32', 'training.precision', 'precision') in ['fp32', 'amp']:
            clip_model.float()

        self.model = CustomCLIPDirichlet(self.cfg, self.classnames, clip_model, self.device)

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

        self.model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            if param.device != self.device:
                param.data = param.data.to(self.device)

        input_tensor = torch.randn(1, 3, 224, 224, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            model_copy = copy.deepcopy(self.model)
            model_copy.to(self.device)
            result = profile(model_copy, inputs=(input_tensor,), verbose=False)
            if isinstance(result, (list, tuple)):
                macs = result[0] if len(result) > 0 else 0
            else:
                macs = result
            gflops_thop = macs / 1e9
            del model_copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        msg = f"Learnable parameters: {format_params(learnable_params)} / Total parameters: {format_params(total_params)} (FLOPs: {gflops_thop:.2f} GFLOPs)"
        print(msg)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')

        trainable_names = set(self.model.get_trainable_parameter_names())
        for name, param in self.model.named_parameters():
            param.requires_grad_(name in trainable_names)

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

    def train_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.model.train()
        self.model.prompt_learner.train()

        precision = self._cfg_str('fp32', 'training.precision', 'precision')

        if precision == 'amp' and self.scaler is not None:
            with autocast():
                loss, logits, _ = self.model(images, labels)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, logits, _ = self.model(images, labels)
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

                logits, _ = self.model(images)
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
        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "predictions": all_preds,
            "true_labels": all_labels_list,
        }

    def save_model(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'prompt_learner_state_dict': self.model.prompt_learner.state_dict(),
            'dirichlet_head_state_dict': self.model.dirichlet_head.state_dict(),
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
        if 'dirichlet_head_state_dict' in checkpoint:
            self.model.dirichlet_head.load_state_dict(checkpoint['dirichlet_head_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        msg = f"Model loaded from {path}"
        print(msg)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')


class APTDirichletPipeline:
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

        base_output_value = self.logging_cfg.get("output_dir", "outputs")
        base_output = coerce_to_str(base_output_value, "outputs", key="logging.output_dir")
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
        self.classnames: List[str] = []
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.labeled_indices: List[int] = []
        self.unlabeled_indices: List[int] = []
        self.metrics: List[Dict[str, Any]] = []
        self.best_val_acc = -float('inf')
        self.global_epoch = 0

        self.trainer: Optional[APTDirichlet] = None
        self.trainer_cfg: ConfigNode = ConfigNode({})
        self.rounds = 1

    def _get_training_epochs(self):
        epochs_value = None
        if isinstance(self.training_cfg, dict):
            epochs_value = self.training_cfg.get('epochs', None)
        return coerce_to_int(epochs_value, DEFAULT_TRAINING_EPOCHS, key='training.epochs')

    def run(self):
        set_global_seed(self.seed)
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
        if bool(get_config_value(self.training_cfg, "use_cutout", False)):
            base_transforms.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0))
            print("Using Cutout (RandomErasing) augmentation.")
        return transforms.Compose(base_transforms)

    def _load_dataset(self):
        transform = self._build_transforms()
        try:
            self.dataset = ImageFolder(self.dataset_root, transform=transform)
        except Exception as exc:
            raise RuntimeError(f"Failed to load dataset from {self.dataset_root}: {exc}") from exc

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
        self.trainer = APTDirichlet(self.trainer_cfg, self.classnames, device=str(self.device), log_file=self.log_file)

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
        else:
            val_acc = 0.0
            val_loss = 0.0

        epoch_dir = os.path.join(round_dir, f'epoch_{epoch_in_round:03d}')
        os.makedirs(epoch_dir, exist_ok=True)

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
            f"val_loss={val_loss_display} - val_acc={val_acc_display} - "
            f"time={epoch_time:.2f}s"
        )
        print(epoch_str)
        with open(self.log_file, 'a') as f:
            f.write(epoch_str + '\n')

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

        completion_msg = f"Training completed. Results written to {self.run_dir}"
        print(completion_msg)
        with open(self.log_file, 'a') as f:
            f.write(completion_msg + '\n')


def parse_args():
    parser = create_argument_parser("Train APT Dirichlet model", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = APTDirichletPipeline(merged)
    pipeline.run()


if __name__ == "__main__":
    main()
