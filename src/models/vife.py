import os
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
from scipy.ndimage import zoom
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from collections import defaultdict


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
    from utils import logger

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
