import os
import math
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from apt import (
    ImageEncoder,
    CustomCLIP,
    APT,
    APTTrainingPipeline,
    ARG_SCHEMA,
    DEFAULT_TRAINING_EPOCHS,
)
from utils import (
    logger,
    setup_logging,
    ConfigNode,
    create_argument_parser,
    process_parsed_args,
    parse_override_arguments,
    merge_configs,
    load_config_file,
    load_clip_to_cpu,
    coerce_to_int,
    coerce_to_float,
)


class PatchGroupRouter(nn.Module):
    def __init__(self, feature_dim, num_groups):
        super().__init__()
        self.num_groups = num_groups
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_groups),
        )

    def forward(self, patches, tau, hard=True):
        logits = self.net(patches)
        if self.training:
            soft = F.gumbel_softmax(logits, tau=tau, hard=hard)
        else:
            idx = logits.argmax(dim=-1)
            soft = F.one_hot(idx, self.num_groups).float()
        return soft, logits


class PartDiscoveryAttention(nn.Module):
    def __init__(self, feature_dim, num_groups=4, num_heads=8,
                 dropout=0.1, warmup_epochs=10):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_groups = num_groups

        self.router = PatchGroupRouter(feature_dim, num_groups)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout,
            batch_first=False,
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Linear(feature_dim, feature_dim)

        self.warmup_epochs = warmup_epochs
        self.use_hard = warmup_epochs == 0
        self.current_tau = 1.0
        self.current_epoch = 0
        self.total_epochs = 1
        self._routing_history = []

    def set_epoch(self, epoch, total_epochs=None):
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
        self.use_hard = epoch >= self.warmup_epochs

    def clear_routing_history(self):
        self._routing_history = []

    def forward(self, unpooled, text_features):
        S, B, D = unpooled.shape
        C = text_features.size(0)
        G = self.num_groups

        assignments, logits = self.router(
            unpooled.reshape(S * B, D), self.current_tau, hard=self.use_hard
        )
        assignments = assignments.reshape(S, B, G)

        if self.training:
            self._routing_history.append(assignments.detach().mean(dim=0))

        parts = []
        for g in range(G):
            weights_g = assignments[:, :, g]
            weight_sum = weights_g.sum(dim=0, keepdim=True).clamp(min=1e-6)
            weighted_patches = (weights_g.unsqueeze(-1) * unpooled)
            part_g = weighted_patches.sum(dim=0) / weight_sum.transpose(0, 1)
            parts.append(part_g)

        part_tokens = torch.stack(parts, dim=0)

        out, attn_weights = self.cross_attn(text_features, part_tokens, part_tokens)
        text_features = self.norm1(self.dropout(text_features + out))
        ff = self.feed_forward(text_features)
        text_features = self.norm2(self.dropout(text_features + ff))

        return text_features, attn_weights

    def diversity_loss(self):
        if not self._routing_history:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        recent = torch.cat(self._routing_history[-32:], dim=0)
        avg = recent.mean(dim=0)
        entropy = -(avg * (avg + 1e-8).log()).sum()
        return -entropy

    def load_balance_loss(self):
        if not self._routing_history:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        recent = torch.cat(self._routing_history[-32:], dim=0)
        avg_prob = recent.mean(dim=0)
        fraction = (recent.argmax(dim=-1).unsqueeze(-1) == torch.arange(
            self.num_groups, device=recent.device
        ).unsqueeze(0)).float().mean(dim=0)
        loss = (fraction * avg_prob).sum() * self.num_groups
        return loss


class MoECustomCLIP(CustomCLIP):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__(cfg, classnames, clip_model, device)

        moe_cfg = self.model_cfg.get('moe', ConfigNode())
        if not isinstance(moe_cfg, ConfigNode):
            moe_cfg = ConfigNode(moe_cfg)

        prompt_dim = self.clip_model.text_projection.shape[1]
        num_groups = coerce_to_int(moe_cfg.get('num_groups', 4), 4)
        num_heads = coerce_to_int(self.model_cfg.get('num_heads', 8), 8)
        dropout = coerce_to_float(self.model_cfg.get('dropout', 0.1), 0.1)
        warmup_epochs = coerce_to_int(moe_cfg.get('warmup_epochs', 10), 10)

        self.prompt_learner = nn.ModuleList([
            PartDiscoveryAttention(
                feature_dim=prompt_dim,
                num_groups=num_groups,
                num_heads=num_heads,
                dropout=dropout,
                warmup_epochs=warmup_epochs,
            )
            for _ in range(self.model_cfg.get('num_layers', 1))
        ])

        self.diversity_weight = coerce_to_float(moe_cfg.get('diversity_weight', 0.1), 0.1)
        self.load_balance_weight = coerce_to_float(moe_cfg.get('load_balance_weight', 0.01), 0.01)

        if self.training_cfg.get('precision', 'fp32') == 'fp16':
            self.prompt_learner = self.prompt_learner.half()

    def forward(self, image, label=None):
        with torch.no_grad():
            pass

        visual_output = self.vis_encoder(image)
        unpooled_levels, image_features = visual_output
        if not isinstance(unpooled_levels, list):
            unpooled_levels = [unpooled_levels]

        base_text_features = self.text_features.clone()

        unpooled_images = unpooled_levels[0].permute(1, 0, 2)
        text_features = base_text_features.unsqueeze(1).expand(-1, unpooled_images.shape[1], -1)

        for layer in self.prompt_learner:
            text_features, _ = layer(unpooled_images, text_features)

        text_features = text_features.permute(1, 0, 2)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        image_features = F.normalize(image_features, dim=-1)
        image_features = image_features.unsqueeze(1)

        logits = logit_scale * F.cosine_similarity(image_features, text_features, dim=-1)

        mode = self.cfg.get('mode', self.training_cfg.get('mode', 'logits'))

        if self.training and label is not None:
            ce_loss = F.cross_entropy(logits, label)
            div_loss = sum(layer.diversity_loss() for layer in self.prompt_learner) / len(self.prompt_learner)
            lb_loss = sum(layer.load_balance_loss() for layer in self.prompt_learner) / len(self.prompt_learner)
            total_loss = (
                ce_loss
                + self.diversity_weight * div_loss
                + self.load_balance_weight * lb_loss
            )
            return total_loss, logits
        elif mode == "logits":
            return logits
        elif mode == "features":
            return logits, text_features

        return logits

    def trainable_parameters(self):
        return self.prompt_learner.parameters()

    def get_trainable_parameter_names(self):
        return [f"prompt_learner.{name}" for name, _ in self.prompt_learner.named_parameters()]


class MoEAPT(APT):
    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/32', 'model.backbone', 'backbone')
        logger.info(f"Loading CLIP (backbone: {backbone_name})")

        clip_model = load_clip_to_cpu(backbone_name)

        if self._cfg_str('fp32', 'training.precision', 'precision') in ['fp32', 'amp']:
            clip_model.float()

        self.model = MoECustomCLIP(self.cfg, self.classnames, clip_model, self.device)

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

        try:
            from thop import profile
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
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f"Learnable parameters: {format_params(learnable_params)} / Total: {format_params(total_params)} (FLOPs: {gflops_thop:.2f} GFLOPs)")
        except Exception:
            logger.info(f"Learnable parameters: {format_params(learnable_params)} / Total: {format_params(total_params)}")

        trainable_names = set(self.model.get_trainable_parameter_names())
        for name, param in self.model.named_parameters():
            param.requires_grad_(name in trainable_names)

        self.model.to(self.device)
        self.initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

    def train_step(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.model.train()

        precision = self._cfg_str('fp32', 'training.precision', 'precision')

        self.optimizer.zero_grad()

        if precision == 'amp':
            from torch.cuda.amp import autocast
            with autocast():
                loss, logits = self.model(images, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, logits = self.model(images, labels)
            loss.backward()
            self.optimizer.step()

        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        return {"loss": loss.item(), "accuracy": accuracy}

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
        elif 'moe_state_dict' in checkpoint:
            self.model.prompt_learner.load_state_dict(checkpoint['moe_state_dict'], strict=False)
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Model loaded from {path}")


class MoETrainingPipeline(APTTrainingPipeline):
    @property
    def val_dataset(self):
        if self.val_fraction is not None:
            from torch.utils.data import Subset
            if hasattr(self, 'val_indices') and self.dataset is not None:
                subset = Subset(self.dataset, self.val_indices)
                subset.samples = [self.dataset.samples[i] for i in self.val_indices]
                return subset
            return None
        if hasattr(self, '_val_dataset'):
            return self._val_dataset
        return None

    def _initialize_trainer(self):
        if not self.classnames:
            raise RuntimeError("Class names unavailable before trainer initialization.")
        self.trainer = MoEAPT(self.trainer_cfg, self.classnames, device=str(self.device))

    def _run_epoch(self, epoch_idx, epochs_total, train_loader, run_dir):
        if self.trainer is not None and hasattr(self.trainer, 'model'):
            for layer in self.trainer.model.prompt_learner:
                layer.set_epoch(epoch_idx, epochs_total)
                if epoch_idx == 0:
                    layer.clear_routing_history()
            layer0 = self.trainer.model.prompt_learner[0]
            phase = "soft" if not layer0.use_hard else "hard"
            logger.info(f"Epoch {epoch_idx} tau={layer0.current_tau:.4f} routing={phase}")
        return super()._run_epoch(epoch_idx, epochs_total, train_loader, run_dir)


MOE_ARG_SCHEMA = {
    'config': {'type': str, 'required': True, 'help': 'Path to YAML configuration file'},
    'output_dir': {'type': str, 'help': 'Override logging.output_dir from config', 'config_path': 'logging.output_dir'},
    'device': {'type': str, 'help': 'Override training.device from config', 'config_path': 'training.device'},
    'debug': {'type': bool, 'help': 'Enable debug output', 'default': False},
    'disable_coloring': {'type': bool, 'help': 'Disable colored output for log files', 'default': False},
}


def parse_args():
    parser = create_argument_parser("Train MoH-APT model", MOE_ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, MOE_ARG_SCHEMA, overrides)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True), getattr(args, 'disable_coloring', False))
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = MoETrainingPipeline(merged)
    pipeline.run()


if __name__ == "__main__":
    main()
