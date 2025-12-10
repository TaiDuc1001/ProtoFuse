import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


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


class TransformerAdapter(nn.Module):
    def __init__(self, feature_dim, num_layers=1, num_heads=8, mlp_ratio=4.0, dropout=0.0):
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
                    'mlp': nn.Sequential(
                        nn.Linear(feature_dim, int(feature_dim * mlp_ratio)),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(int(feature_dim * mlp_ratio), feature_dim),
                        nn.Dropout(dropout)
                    )
                })
            )
        self.norm = nn.LayerNorm(feature_dim)
        self.last_attn_weights = None

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        x = x.unsqueeze(1)
        attn_weights_all = []
        for layer in self.layers:
            residual = x
            x = layer['norm1'](x)
            qkv = layer['qkv'](x).reshape(B, 1, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn_weights_all.append(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, 1, -1)
            x = layer['proj'](x)
            x = residual + x
            x = x + layer['mlp'](layer['norm2'](x))
        x = self.norm(x).squeeze(1)
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
                _, cls_feat = visual_output
            else:
                cls_feat = visual_output
        if return_attention:
            adapted_feat, attn_weights = self.adapter(cls_feat, return_attention=True)
            u = self.ssl_head(adapted_feat)
            return u, adapted_feat, attn_weights
        adapted_feat = self.adapter(cls_feat)
        u = self.ssl_head(adapted_feat)
        return u, adapted_feat

    def get_attention_weights(self, x):
        with torch.no_grad():
            visual_output = self.encoder(x)
            if isinstance(visual_output, tuple):
                _, cls_feat = visual_output
            else:
                cls_feat = visual_output
            _, attn_weights = self.adapter(cls_feat, return_attention=True)
        return attn_weights


def visualize_dino_attention(model, images, image_paths, epoch, output_dir, clip_mean, clip_std):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    
    if isinstance(images, list):
        images = torch.stack(images)
    images = images.to(device)
    
    attn_weights_list = model.get_attention_weights(images)
    
    num_heads = model.num_heads
    head_colors = list(mcolors.TABLEAU_COLORS.values())[:num_heads]
    if len(head_colors) < num_heads:
        head_colors = plt.cm.tab20(np.linspace(0, 1, num_heads))
    
    clip_mean_arr = np.array(clip_mean)
    clip_std_arr = np.array(clip_std)
    
    for img_idx in range(min(len(images), 8)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        img_tensor = images[img_idx].cpu()
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = img_np * clip_std_arr + clip_mean_arr
        img_np = np.clip(img_np, 0, 1)
        
        axes[0].imshow(img_np)
        axes[0].set_title(f"Image {img_idx}")
        axes[0].axis('off')
        
        attn = attn_weights_list[-1][img_idx].cpu().numpy()
        
        head_strengths = attn.squeeze(-1).squeeze(-1)
        
        bars = axes[1].bar(range(num_heads), head_strengths, color=head_colors[:num_heads])
        axes[1].set_xlabel('Attention Head')
        axes[1].set_ylabel('Attention Weight')
        axes[1].set_title(f'Per-Head Attention (Epoch {epoch})')
        axes[1].set_xticks(range(num_heads))
        axes[1].set_xticklabels([f'H{i}' for i in range(num_heads)])
        
        for i, bar in enumerate(bars):
            bar.set_label(f'Head {i}')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'dino_attention_epoch{epoch:03d}_img{img_idx}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    all_attn = attn_weights_list[-1].cpu().numpy()
    mean_attn = all_attn.squeeze(-1).squeeze(-1).mean(axis=0)
    std_attn = all_attn.squeeze(-1).squeeze(-1).std(axis=0)
    
    bars = ax.bar(range(num_heads), mean_attn, yerr=std_attn, color=head_colors[:num_heads], capsize=3)
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title(f'Average Per-Head Attention (Epoch {epoch})')
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f'H{i}' for i in range(num_heads)])
    
    save_path = os.path.join(output_dir, f'dino_attention_epoch{epoch:03d}_summary.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  [VIS] Saved attention visualizations to {output_dir}")


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


class SSLTransform:
    def __init__(self, base_transform, clip_mean, clip_std):
        self.base_transform = base_transform
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])

    def __call__(self, img):
        v1 = self.aug(img)
        v2 = self.aug(img)
        return v1, v2


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


def ssl_loss_symmetric(u_t1, u_t2, u_s1, u_s2, teacher_temp, student_temp, center):
    loss_12 = ssl_loss(u_t1, u_s2, teacher_temp, student_temp, center)
    loss_21 = ssl_loss(u_t2, u_s1, teacher_temp, student_temp, center)
    return (loss_12 + loss_21) / 2
