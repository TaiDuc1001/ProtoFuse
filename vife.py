import os
import time
import json
import copy
import math
import torch
import random
import hashlib
import numpy as np
import torch.nn as nn
from scipy.ndimage import zoom
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

from apt import APTTrainingPipeline
from utils import (
    logger,
    setup_logging,
    set_global_seed,
    ConfigNode,
    BaseTrainingPipeline,
    DEFAULT_ARG_SCHEMA,
    create_argument_parser,
    process_parsed_args,
    parse_override_arguments,
    merge_configs,
    load_config_file,
    coerce_to_int,
    coerce_to_float,
    coerce_to_str,
    compute_metrics,
    log_experiment_start,
    log_experiment_metrics,
)

ARG_SCHEMA = DEFAULT_ARG_SCHEMA


class SSLHead(nn.Module):
    def __init__(self, feature_dim, proj_dim=256, num_prototypes=4096):
        super().__init__()
        self.proj = nn.Linear(feature_dim, proj_dim)
        self.proto = nn.Linear(proj_dim, num_prototypes, bias=False)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        z = self.proj(x)
        z = F.normalize(z, dim=-1)
        u = self.proto(z)
        return u


class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class FusionWeightLearner(nn.Module):
    def __init__(self, num_classes=None, init_w1=1.0, init_w2=1.0):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(init_w1))
        self.w2 = nn.Parameter(torch.tensor(init_w2))

    def forward(self, logits_apt, logits_img):
        device = self.w1.device
        logits_apt = logits_apt.to(device)
        logits_img = logits_img.to(device)
        apt_centered = logits_apt - logits_apt.mean(dim=-1, keepdim=True)
        img_centered = logits_img - logits_img.mean(dim=-1, keepdim=True)
        fused = self.w1 * apt_centered + self.w2 * img_centered
        return fused

    def get_weights(self):
        return self.w1.detach().item(), self.w2.detach().item()


class TransformerAdapter(nn.Module):
    def __init__(self, feature_dim, num_layers=1, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    'norm1': nn.LayerNorm(feature_dim),
                    'qkv': nn.Linear(feature_dim, feature_dim * 3),
                    'proj': nn.Linear(feature_dim, feature_dim),
                    'norm2': nn.LayerNorm(feature_dim),
                    'feed_forward': nn.Linear(feature_dim, feature_dim)
                })
            )
        self.norm = nn.LayerNorm(feature_dim)
        self.last_attn_weights = None

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        seq_len = x.shape[1]
        attn_weights_all = []
        for layer in self.layers:
            residual = x
            x = layer['norm1'](x)
            qkv = layer['qkv'](x).reshape(B, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn_weights_all.append(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, seq_len, -1)
            x = layer['proj'](x)
            x = residual + x
            x = layer['feed_forward'](layer['norm2'](residual))
        x = self.norm(x)
        if seq_len == 1:
            x = x.squeeze(1)
        if attn_weights_all:
            self.last_attn_weights = attn_weights_all[-1].detach()
        if return_attention:
            return x, attn_weights_all
        return x


class ImageSSLModel(nn.Module):
    def __init__(self, image_encoder, feature_dim, proj_dim=256, num_prototypes=4096,
                 num_trans_layers=1, num_heads=8):
        super().__init__()
        self.encoder = image_encoder
        self.num_heads = num_heads
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.adapter = TransformerAdapter(feature_dim, num_trans_layers, num_heads)
        self.ssl_head = SSLHead(feature_dim, proj_dim, num_prototypes)

    def forward(self, x, return_attention=False):
        with torch.no_grad():
            visual_output = self.encoder(x)
            if isinstance(visual_output, tuple):
                all_tokens, _ = visual_output
            else:
                all_tokens = visual_output
        encoder_cls = all_tokens[:, 0, :] if all_tokens.dim() == 3 else all_tokens
        if return_attention:
            adapted_tokens, attn_weights = self.adapter(all_tokens, return_attention=True)
            adapted_cls = adapted_tokens[:, 0, :] if adapted_tokens.dim() == 3 else adapted_tokens
            cls_feat = adapted_cls + encoder_cls
            u = self.ssl_head(cls_feat)
            return u, cls_feat, attn_weights
        adapted_tokens = self.adapter(all_tokens)
        adapted_cls = adapted_tokens[:, 0, :] if adapted_tokens.dim() == 3 else adapted_tokens
        cls_feat = adapted_cls + encoder_cls
        u = self.ssl_head(cls_feat)
        return u, cls_feat

    def get_attention_weights(self, x):
        with torch.no_grad():
            visual_output = self.encoder(x)
            if isinstance(visual_output, tuple):
                all_tokens, _ = visual_output
            else:
                all_tokens = visual_output
            _, attn_weights = self.adapter(all_tokens, return_attention=True)
        return attn_weights


def visualize_dino_attention(model, images, image_paths, epoch, output_dir, clip_mean, clip_std):

    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    
    if isinstance(images, list):
        images = torch.stack(images)
    images = images.to(device)
    
    with torch.no_grad():
        _, _, attn_weights_list = model(images, return_attention=True)
    
    attn_weights = attn_weights_list[-1]
    
    if attn_weights is None:
        logger.warning("Could not capture attention weights")
        return
    
    B, num_heads, seq_len, _ = attn_weights.shape
    num_patches = seq_len - 1
    grid_size = int(num_patches ** 0.5)
    
    cls_attn = attn_weights[:, :, 0, 1:]
    cls_attn = cls_attn.reshape(B, num_heads, grid_size, grid_size)
    
    clip_mean_arr = np.array(clip_mean)
    clip_std_arr = np.array(clip_std)
    
    img_size = images.shape[-1]
    scale_factor = img_size / grid_size
    
    for img_idx in range(min(len(images), 8)):
        img_tensor = images[img_idx].cpu()
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = img_np * clip_std_arr + clip_mean_arr
        img_np = np.clip(img_np, 0, 1)
        
        max_attn = cls_attn[img_idx].max(dim=0)[0].cpu().numpy()
        max_attn = (max_attn - max_attn.min()) / (max_attn.max() - max_attn.min() + 1e-8)
        max_attn_resized = zoom(max_attn, scale_factor, order=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        axes[0].imshow(img_np)
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis('off')
        
        axes[1].imshow(img_np)
        axes[1].imshow(max_attn_resized, cmap='inferno', alpha=0.6)
        axes[1].set_title("Attention", fontsize=11)
        axes[1].axis('off')
        
        plt.suptitle(f"Epoch {epoch}", fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'attn_epoch{epoch:03d}_img{img_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    logger.info(f"Saved DINO attention visualizations to: {output_dir}")
    model.train()


class DINOMultiCropTransform:
    def __init__(self, clip_mean, clip_std, global_crop_size=224, local_crop_size=96,
                 global_crop_scale=(0.4, 1.0), local_crop_scale=(0.05, 0.4), num_local_crops=6):
        self.num_local_crops = num_local_crops
        
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crop_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])
        
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crop_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])

    def __call__(self, img):
        global_views = [self.global_transform(img), self.global_transform(img)]
        local_views = [self.local_transform(img) for _ in range(self.num_local_crops)]
        return global_views, local_views

def create_teacher_from_student(student):
    teacher = copy.deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def update_teacher_ema(teacher, student, momentum=0.996):
    with torch.no_grad():
        for param_t, param_s in zip(teacher.parameters(), student.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)


def get_cosine_ema_momentum(epoch, total_epochs, base_momentum=0.996, final_momentum=1.0):
    return final_momentum - (final_momentum - base_momentum) * (math.cos(math.pi * epoch / total_epochs) + 1) / 2


def update_center(center, batch_logits, momentum=0.9):
    with torch.no_grad():
        batch_center = batch_logits.mean(dim=0)
        center.mul_(momentum).add_(batch_center, alpha=1 - momentum)
    return center


def ssl_loss(q_teacher, p_student, teacher_temp=0.04, student_temp=0.1, center=None):
    if center is not None:
        q_teacher = q_teacher - center
    q = F.softmax(q_teacher / teacher_temp, dim=-1)
    p = F.log_softmax(p_student / student_temp, dim=-1)
    loss = -(q * p).sum(dim=-1).mean()
    return loss


def dino_loss(teacher_global_outputs, student_all_outputs, teacher_temp, student_temp, center):
    total_loss = 0.0
    n_loss_terms = 0
    for t_idx, t_out in enumerate(teacher_global_outputs):
        for s_idx, s_out in enumerate(student_all_outputs):
            if s_idx == t_idx:
                continue
            loss = ssl_loss(t_out, s_out, teacher_temp, student_temp, center)
            total_loss += loss
            n_loss_terms += 1
    return total_loss / max(n_loss_terms, 1)

class DINODataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.Image = Image

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.base_dataset.samples[real_idx]
        img = self.Image.open(path).convert('RGB')
        global_views, local_views = self.transform(img)
        return global_views, local_views, label


class ViFETrainingPipeline(APTTrainingPipeline):
    METHOD_NAME = "ViFE"
    DEFAULT_OUTPUT_DIR = "outputs/vife"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/vife"

    def __init__(self, config):
        if not isinstance(config, ConfigNode):
            config = ConfigNode(config)
        ssl_cfg = config.get('ssl', ConfigNode())
        if not isinstance(ssl_cfg, ConfigNode):
            ssl_cfg = ConfigNode(ssl_cfg)
        ssl_cfg['enabled'] = True
        config['ssl'] = ssl_cfg
        super().__init__(config)

        self.ssl_cfg = ssl_cfg
        self.use_ssl = True

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

    def run(self):
        set_global_seed(self.seed)

        logger.section("Initialization", "config")
        self._prepare_directories()
        self._load_dataset()
        self._split_dataset()
        self._initialize_trainer()

        dataset_name = self.config.data.dataset_name
        log_experiment_start("ViFE", dataset_name, self.kshot, self.seed)

        logger.section("APT Training", "train")
        self._train_epochs()

        enable_stage1 = self.ssl_cfg.get('enable_stage1', True)
        enable_stage2 = self.ssl_cfg.get('enable_stage2', True)
        enable_stage3 = self.ssl_cfg.get('enable_stage3', True)

        if enable_stage1:
            logger.section("SSL Stage 1: Self-Supervised Learning", "model")
            if self.ssl_cfg.get('save_stage1_checkpoint', False) and self._try_load_ssl_stage1_checkpoint():
                logger.info("Skipping SSL Stage 1 training (loaded from checkpoint)")
            else:
                self._train_ssl_stage1()
        else:
            logger.info("Skipping SSL Stage 1 (disabled in config)")

        if enable_stage2:
            logger.section("SSL Stage 2: Linear Classifier Training", "train")
            self._train_ssl_stage2()
        else:
            logger.info("Skipping SSL Stage 2 (disabled in config)")

        if enable_stage3 and self.ssl_cfg.get('learn_fusion', False):
            logger.section("SSL Stage 3: Fusion Weight Learning", "train")
            self._train_ssl_stage3()
        elif not enable_stage3:
            logger.info("Skipping SSL Stage 3 (disabled in config)")

        logger.section("Dual-Branch Evaluation", "eval")
        eval_result = self._run_dual_branch_eval()
        self.dual_branch_eval_result = eval_result
        if enable_stage3:
            self.learned_acc = eval_result.get('learned_acc') if eval_result else None
        else:
            default_weight = coerce_to_float(self.ssl_cfg.get('default_fusion_weight', 0.5), 0.5)
            if eval_result and eval_result.get('fusion_results'):
                self.learned_acc = eval_result['fusion_results'].get(default_weight)
                logger.info(f"Using default fusion weight {default_weight} → {self.learned_acc:.2f}%")
            else:
                self.learned_acc = eval_result.get('apt_acc') if eval_result else None

        logger.section("Finalization", "save")
        self._finalize()
        self._finalize_vife()

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
                img = Image.open(path).convert('RGB')
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
        ssl1_dir = os.path.join('checkpoints/vife', ssl_checkpoint_id)
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

        base_acc = None
        novel_acc = None
        harmonic_mean = None
        if self.base_novel_enabled and self.base_class_indices and self.novel_class_indices:
            base_set = set(self.base_class_indices)
            novel_set = set(self.novel_class_indices)
            
            base_mask = torch.tensor([int(lbl.item()) in base_set for lbl in all_labels])
            novel_mask = torch.tensor([int(lbl.item()) in novel_set for lbl in all_labels])
            
            if base_mask.sum() > 0:
                base_correct = (pred_apt[base_mask] == all_labels[base_mask]).sum().item()
                base_acc = 100 * base_correct / base_mask.sum().item()
                logger.info(f"  Base Classes ({base_mask.sum().item()} samples): {base_acc:.2f}%")
            
            if novel_mask.sum() > 0:
                novel_correct = (pred_apt[novel_mask] == all_labels[novel_mask]).sum().item()
                novel_acc = 100 * novel_correct / novel_mask.sum().item()
                logger.info(f"  Novel Classes ({novel_mask.sum().item()} samples): {novel_acc:.2f}%")
            
            if base_acc is not None and novel_acc is not None and (base_acc + novel_acc) > 0:
                harmonic_mean = 2 * base_acc * novel_acc / (base_acc + novel_acc)
                logger.info(f"  Harmonic Mean (H): {harmonic_mean:.2f}%")

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

            output_dir = self.run_dir
            os.makedirs(output_dir, exist_ok=True)
            
            positive_conflicts = []
            negative_conflicts = []
            
            csv_lines = []
            for i in range(len(all_labels)):
                apt_pred_i = pred_apt[i].item()
                img_pred_i = pred_img[i].item()
                true_label_i = all_labels[i].item()
                
                img_path, _ = self.val_dataset.samples[i]
                apt_name = self.classnames[apt_pred_i]
                img_name = self.classnames[img_pred_i]
                
                csv_lines.append(f"{img_path},{img_name},{apt_name}")
                
                if apt_pred_i != img_pred_i:
                    true_name = self.classnames[true_label_i]
                    
                    line = f"{img_path}, {true_name}, {apt_name}, {img_name}"
                    
                    def start_point(size, pixel, height):
                        return (size[0] - pixel, size[1] - height)

                    def draw_text(img, text, is_correct):
                        draw = ImageDraw.Draw(img)
                        try:
                             font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
                        except IOError:
                             font = ImageFont.load_default()
                        
                        color = (0, 255, 0) if is_correct else (255, 0, 0)
                        
                        if hasattr(font, "getbbox"):
                             bbox = font.getbbox(text)
                             text_w = bbox[2] - bbox[0]
                             text_h = bbox[3] - bbox[1]
                        else:
                             text_w, text_h = draw.textsize(text, font)
                        
                        padding = 10
                        x, y = 20, 20
                        
                        draw.rectangle(
                            [(x, y), (x + text_w + 2*padding, y + text_h + 2*padding)],
                            fill=color,
                            outline=None
                        )
                        draw.text((x + padding, y + padding), text, fill="white", font=font)
                        return img


                    if img_pred_i == true_label_i and apt_pred_i != true_label_i:
                        positive_conflicts.append(line)
                        folder = os.path.join(output_dir, "positive_conflicts")
                        os.makedirs(folder, exist_ok=True)
                        fname = os.path.basename(img_path)
                        
                        img1 = Image.open(img_path).convert("RGB")
                        img1 = draw_text(img1, img_name, True)
                        # img1.save(os.path.join(folder, f"{fname.split('.')[0]}_correct.jpg"))
                        
                        img2 = Image.open(img_path).convert("RGB")
                        img2 = draw_text(img2, apt_name, False)
                        # img2.save(os.path.join(folder, f"{fname.split('.')[0]}_incorrect.jpg"))

                    elif apt_pred_i == true_label_i and img_pred_i != true_label_i:
                        negative_conflicts.append(line)
                        folder = os.path.join(output_dir, "negative_conflicts")
                        os.makedirs(folder, exist_ok=True)
                        fname = os.path.basename(img_path)
                        
                        img1 = Image.open(img_path).convert("RGB")
                        img1 = draw_text(img1, apt_name, True)
                        # img1.save(os.path.join(folder, f"{fname.split('.')[0]}_correct.jpg"))
                        
                        img2 = Image.open(img_path).convert("RGB")
                        img2 = draw_text(img2, img_name, False)
                        # img2.save(os.path.join(folder, f"{fname.split('.')[0]}_incorrect.jpg"))

            csv_path = os.path.join(output_dir, 'predictions.csv')
            with open(csv_path, 'w') as f:
                f.write('filename,vife,apt\n')
                f.write('\n'.join(csv_lines))
            logger.info(f"Saved predictions to {csv_path}")
            
            positive_path = os.path.join(output_dir, 'positive_conflict.txt')
            with open(positive_path, 'w') as f:
                f.write('\n'.join(positive_conflicts))
            logger.info(f"Positive conflicts (ViFE correct, APT wrong): {len(positive_conflicts)} → {positive_path}")
            
            negative_path = os.path.join(output_dir, 'negative_conflict.txt')
            with open(negative_path, 'w') as f:
                f.write('\n'.join(negative_conflicts))
            logger.info(f"Negative conflicts (APT correct, ViFE wrong): {len(negative_conflicts)} → {negative_path}")


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

        apt_preds = pred_apt.cpu().numpy().tolist()
        final_labels = all_labels.cpu().numpy().tolist()
        apt_metrics = compute_metrics(final_labels, apt_preds)
        
        fusion_metrics = {}
        if use_ssl_branch and self.fusion_weights is not None:
            with torch.no_grad():
                logits_fused = self.fusion_weights(all_apt_logits, all_img_logits)
            prob_fused = F.softmax(logits_fused, dim=-1)
            _, pred_learned_fused = torch.max(prob_fused, 1)
            fusion_preds = pred_learned_fused.cpu().numpy().tolist()
            fusion_metrics = compute_metrics(final_labels, fusion_preds)

        eval_result = {
            'apt_acc': apt_acc,
            'img_acc': img_acc if use_ssl_branch else None,
            'fusion_results': fusion_results,
            'best_weight': best_weight,
            'best_fused_acc': best_fused_acc,
            'learned_acc': learned_acc,
            'base_acc': base_acc,
            'novel_acc': novel_acc,
            'harmonic_mean': harmonic_mean,
            'apt_metrics': apt_metrics,
            'fusion_metrics': fusion_metrics
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

                if use_ssl_fusion and self.ssl_student is not None and self.ssl_classifier is not None:
                    _, cls_feat = self.ssl_student(images)
                    logits_img = self.ssl_classifier(cls_feat)
                    if self.fusion_weights is not None:
                        self.fusion_weights.eval()
                        logits = self.fusion_weights(logits_apt, logits_img)
                    else:
                        default_weight = coerce_to_float(self.ssl_cfg.get('default_fusion_weight', 0.5), 0.5)
                        logits = (1 - default_weight) * logits_apt + default_weight * logits_img
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

        metrics = compute_metrics(all_labels_list, all_preds)
        avg_loss = running_loss / max(1, steps)
        metrics['loss'] = avg_loss
        metrics['predictions'] = all_preds
        metrics['true_labels'] = all_labels_list
        return metrics

    def _finalize_vife(self):
        if hasattr(self, 'dual_branch_eval_result') and self.dual_branch_eval_result:
            apt_metrics = self.dual_branch_eval_result.get('apt_metrics', {})
            fusion_metrics = self.dual_branch_eval_result.get('fusion_metrics', {})

            logger.info("Orbit - APT Metrics:")
            log_experiment_metrics(apt_metrics)

            if fusion_metrics:
                logger.info("Orbit - ViFE (Fusion) Metrics:")
                log_experiment_metrics(fusion_metrics)


BaseTrainingPipeline.register_extra_pipeline(ViFETrainingPipeline)


def parse_args():
    parser = create_argument_parser("Train ViFE model", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True), getattr(args, 'disable_coloring', False))
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = ViFETrainingPipeline(merged)
    pipeline.run()


if __name__ == "__main__":
    main()
