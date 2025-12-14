import os
import time
import math
import json
import torch
import random
import hashlib
import datetime
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from collections import defaultdict
from typing import Any, Dict, List, Optional
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from utils import (
    logger,
    setup_logging,
    run_dataset_eda,
    save_class_distribution_plot,
    save_confusion_artifacts,
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
)

ARG_SCHEMA = {
    'config': {'type': str, 'required': True, 'help': 'Path to YAML configuration file'},
    'output_dir': {'type': str, 'help': 'Override logging.output_dir from config', 'config_path': 'logging.output_dir'},
    'device': {'type': str, 'help': 'Override training.device from config', 'config_path': 'training.device'},
    'debug': {'type': bool, 'help': 'Enable debug output', 'default': False},
}

DEFAULT_TRAINING_EPOCHS = 100
DEFAULT_CHECKPOINT_DIR = 'checkpoints/vpt'


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
        self.hidden_size = visual.class_embedding.shape[-1]
        self.num_layers = len(visual.transformer.resblocks)

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
        return x

    def forward_transformer(self, x, prompts=None, deep_prompts=None, num_tokens=0):
        x = x.permute(1, 0, 2)
        
        for idx, block in enumerate(self.transformer.resblocks):
            if idx == 0 and prompts is not None:
                x = x.permute(1, 0, 2)
                B = x.shape[0]
                x = torch.cat([
                    x[:, :1, :],
                    prompts.expand(B, -1, -1),
                    x[:, 1:, :]
                ], dim=1)
                x = x.permute(1, 0, 2)
            elif idx > 0 and deep_prompts is not None and idx <= deep_prompts.shape[0]:
                x = x.permute(1, 0, 2)
                B = x.shape[0]
                deep_prompt = deep_prompts[idx - 1].unsqueeze(0).expand(B, -1, -1)
                x = torch.cat([
                    x[:, :1, :],
                    deep_prompt,
                    x[:, 1 + num_tokens:, :]
                ], dim=1)
                x = x.permute(1, 0, 2)
            
            x = block(x)

        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        
        cls_output = x[:, 0, :]
        if self.proj is not None:
            cls_output = cls_output @ self.proj
        
        return cls_output


class PromptedCLIP(nn.Module):
    def __init__(self, cfg, clip_model, num_classes, device):
        super().__init__()
        
        if not isinstance(cfg, ConfigNode):
            cfg = ConfigNode(cfg)
        self.cfg = cfg
        self.model_cfg = cfg.get('model', ConfigNode())
        self.device = device
        
        self.encoder = ImageEncoder(clip_model)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        hidden_size = self.encoder.hidden_size
        num_layers = self.encoder.num_layers
        patch_size = self.encoder.patch_size
        
        self.num_tokens = coerce_to_int(self.model_cfg.get('num_tokens', 10), 10)
        self.deep_prompting = bool(self.model_cfg.get('deep_prompting', True))
        prompt_dropout = coerce_to_float(self.model_cfg.get('prompt_dropout', 0.1), 0.1)
        prompt_project = coerce_to_int(self.model_cfg.get('prompt_project', -1), -1)
        
        if prompt_project > 0:
            prompt_dim = prompt_project
            self.prompt_proj = nn.Linear(prompt_dim, hidden_size)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = hidden_size
            self.prompt_proj = nn.Identity()
        
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        
        val = math.sqrt(6. / float(3 * patch_size * patch_size + prompt_dim))
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        
        if self.deep_prompting:
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(num_layers - 1, self.num_tokens, prompt_dim)
            )
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        else:
            self.deep_prompt_embeddings = None
        
        output_dim = self.encoder.proj.shape[1] if self.encoder.proj is not None else hidden_size
        self.head = nn.Linear(output_dim, num_classes)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def incorporate_prompt(self, x):
        B = x.shape[0]
        prompts = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings))
        return prompts

    def get_deep_prompts(self):
        if self.deep_prompt_embeddings is None:
            return None
        return self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings))

    def forward(self, x, return_features=False):
        embedded = self.encoder(x)
        prompts = self.incorporate_prompt(embedded)
        deep_prompts = self.get_deep_prompts()
        
        cls_features = self.encoder.forward_transformer(
            embedded, 
            prompts=prompts, 
            deep_prompts=deep_prompts,
            num_tokens=self.num_tokens
        )
        
        logits = self.head(cls_features)
        
        if return_features:
            return logits, cls_features
        return logits

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def trainable_parameters(self):
        yield from self.prompt_embeddings.unsqueeze(0).parameters() if False else [self.prompt_embeddings]
        if self.deep_prompt_embeddings is not None:
            yield self.deep_prompt_embeddings
        yield from self.prompt_proj.parameters()
        yield from self.head.parameters()

    def get_trainable_params_list(self):
        params = [self.prompt_embeddings]
        if self.deep_prompt_embeddings is not None:
            params.append(self.deep_prompt_embeddings)
        params.extend(list(self.prompt_proj.parameters()))
        params.extend(list(self.head.parameters()))
        return params







class VPT:
    def __init__(self, cfg, classnames, device="cuda"):
        if not isinstance(cfg, ConfigNode):
            cfg = ConfigNode(cfg)
        self.cfg = cfg
        self.training_cfg = self.cfg.get('training', ConfigNode())
        self.model_cfg = self.cfg.get('model', ConfigNode())
        self.data_cfg = self.cfg.get('data', ConfigNode())
        self.classnames = classnames
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
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
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')
        logger.info(f"Loading CLIP (backbone: {backbone_name})")
        
        clip_model = load_clip_to_cpu(backbone_name)
        
        if self._cfg_str('fp32', 'training.precision', 'precision') in ['fp32', 'amp']:
            clip_model.float()

        num_classes = len(self.classnames)
        self.model = PromptedCLIP(self.cfg, clip_model, num_classes, self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.get_trainable_params_list())
        
        def format_params(num):
            if num >= 1e9:
                return f"{num/1e9:.2f}B"
            elif num >= 1e6:
                return f"{num/1e6:.2f}M"
            elif num >= 1e3:
                return f"{num/1e3:.2f}K"
            else:
                return str(num)
        
        logger.info(f"VPT: num_tokens={self.model.num_tokens}, deep_prompting={self.model.deep_prompting}")
        logger.info(f"Learnable parameters: {format_params(trainable_params)} / Total: {format_params(total_params)}")
        
        self.model.to(self.device)
        self.initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def setup_optimizer(self):
        lr = self._cfg_float(0.001, 'training.learning_rate')
        weight_decay = self._cfg_float(0.0005, 'training.weight_decay')
        optimizer_type = self._cfg_str('SGD', 'training.optimizer')
        trainable_params = self.model.get_trainable_params_list()
        
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
        
        precision = self._cfg_str('fp32', 'training.precision', 'precision')
        
        if precision == 'amp':
            with autocast():
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
        else:
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
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
                loss = F.cross_entropy(logits, labels)
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
            'prompt_embeddings': self.model.prompt_embeddings.data,
            'deep_prompt_embeddings': self.model.deep_prompt_embeddings.data if self.model.deep_prompt_embeddings is not None else None,
            'prompt_proj_state_dict': self.model.prompt_proj.state_dict() if not isinstance(self.model.prompt_proj, nn.Identity) else None,
            'head_state_dict': self.model.head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'cfg': self.cfg
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.prompt_embeddings.data = checkpoint['prompt_embeddings']
        if checkpoint['deep_prompt_embeddings'] is not None and self.model.deep_prompt_embeddings is not None:
            self.model.deep_prompt_embeddings.data = checkpoint['deep_prompt_embeddings']
        if checkpoint['prompt_proj_state_dict'] is not None and not isinstance(self.model.prompt_proj, nn.Identity):
            self.model.prompt_proj.load_state_dict(checkpoint['prompt_proj_state_dict'])
        self.model.head.load_state_dict(checkpoint['head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Model loaded from {path}")


class VPTTrainingPipeline:
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
        self.batch_size = coerce_to_int(batch_value, 32, key="training.batch_size")

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

        run_eda_value = get_config_value(self.data_cfg, "run_eda", False)
        self.run_eda = bool(False if run_eda_value is None else run_eda_value)

        class_dist_value = get_config_value(self.training_cfg, "class_distribution", False)
        self.class_distribution_enabled = bool(False if class_dist_value is None else class_dist_value)

        base_output_value = self.logging_cfg.get("output_dir", "outputs/vpt")
        base_output = coerce_to_str(base_output_value, "outputs/vpt", key="logging.output_dir")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(base_output, timestamp)
        logger.info(f"Run directory: {self.run_dir}")
        self.config_path = os.path.join(self.run_dir, 'config.json')
        self.metrics_path = os.path.join(self.run_dir, 'metrics.json')
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

        self.trainer: Optional[VPT] = None
        self.trainer_cfg: ConfigNode = ConfigNode({})

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
        if 'prompt_embeddings' in model_state:
            self.trainer.model.prompt_embeddings.data = model_state['prompt_embeddings']
            if model_state.get('deep_prompt_embeddings') is not None and self.trainer.model.deep_prompt_embeddings is not None:
                self.trainer.model.deep_prompt_embeddings.data = model_state['deep_prompt_embeddings']
        
        self.trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.labeled_indices = ckpt['labeled_indices']
        self.unlabeled_indices = ckpt['unlabeled_indices']
        self.metrics = ckpt['metrics']
        self.global_epoch = len(self.metrics)
        logger.info(f"Loaded checkpoint: {self.checkpoint_id} (epoch {self.global_epoch})")
        return True

    def _save_checkpoint(self):
        if self.checkpoint_cache is None or self.checkpoint_id is None:
            return
        if self.trainer is None:
            return
        model_state = {
            'prompt_embeddings': self.trainer.model.prompt_embeddings.data,
            'deep_prompt_embeddings': self.trainer.model.deep_prompt_embeddings.data if self.trainer.model.deep_prompt_embeddings is not None else None,
        }
        path = self.checkpoint_cache.save(
            self.checkpoint_id,
            model_state,
            self.trainer.optimizer.state_dict(),
            self.trainer.scheduler.state_dict(),
            self.labeled_indices,
            self.unlabeled_indices,
            self.metrics,
            self.config
        )
        logger.debug(f"Saved checkpoint to: {path}")

    def run(self):
        set_global_seed(self.seed)
        
        logger.section("Initialization", "config")
        self._prepare_directories()
        self._load_dataset()
        self._split_dataset()
        self._initialize_trainer()
        
        logger.section("VPT Training", "train")
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
            'classnames': self.classnames,
            'num_classes': len(self.classnames),
            'train_size': stats.get('labeled_count', stats['train_count']),
            'val_size_count': stats['val_count'],
            'train_pool_size': stats.get('train_pool_size', stats['train_count'] + stats.get('unlabeled_count', 0)),
            'unlabeled_pool_size': stats.get('unlabeled_count', 0),
            'val_percentage_actual': val_percentage,
        }

        trainer_cfg = build_config_namespace(self.config, extra_values)
        self.trainer_cfg = trainer_cfg
        return trainer_cfg

    def _initialize_trainer(self):
        if not self.classnames:
            raise RuntimeError("Class names unavailable before trainer initialization.")
        self.trainer = VPT(self.trainer_cfg, self.classnames, device=str(self.device))

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
        logger.info(f"VPT Epoch {epoch_idx} - loss={avg_loss:.4f} - acc={avg_acc:.2f}% - val_acc={val_acc_display} - {epoch_time:.2f}s")

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

        logger.info(f"Training completed. Results written to {self.run_dir}")


def parse_args():
    parser = create_argument_parser("Train VPT model", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True))
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = VPTTrainingPipeline(merged)
    pipeline.run()


if __name__ == "__main__":
    main()
