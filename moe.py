import os
import copy
import torch
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


class SemanticRouter(nn.Module):
    def __init__(self, feature_dim, num_experts, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feature_dim // 4
        self.temperature = nn.Parameter(torch.ones(1))
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, class_embeddings):
        scores = self.net(class_embeddings)
        weights = F.softmax(scores / self.temperature.clamp(min=0.1), dim=-1)
        return weights


class SemanticExpert(nn.Module):
    def __init__(self, feature_dim, expert_dim, dropout=0.1):
        super().__init__()
        self.expert_dim = expert_dim
        self.scale = expert_dim ** -0.5

        self.q_proj = nn.Linear(feature_dim, expert_dim)
        self.k_proj = nn.Linear(feature_dim, expert_dim)
        self.v_proj = nn.Linear(feature_dim, expert_dim)

        self.norm1 = nn.LayerNorm(expert_dim)
        self.ffn = nn.Sequential(
            nn.Linear(expert_dim, expert_dim),
            nn.GELU(),
        )
        self.norm2 = nn.LayerNorm(expert_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_features, image_features):
        q = self.q_proj(text_features)
        k = self.k_proj(image_features)
        v = self.v_proj(image_features)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = self.norm1(out)
        out = out + self.ffn(out)
        out = self.norm2(out)

        return out


class MixtureOfExperts(nn.Module):
    def __init__(self, feature_dim, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dim = feature_dim // num_experts

        self.router = SemanticRouter(feature_dim, num_experts)
        self.experts = nn.ModuleList([
            SemanticExpert(feature_dim, self.expert_dim, dropout)
            for _ in range(num_experts)
        ])
        self.output_proj = nn.Linear(self.expert_dim, feature_dim)
        self.output_norm = nn.LayerNorm(feature_dim)

        self.current_epoch = 0
        self.warmup_epochs = 5

        self._routing_history = []

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, text_features, image_features):
        B = image_features.size(0)
        C = text_features.size(0) if text_features.dim() == 2 else text_features.size(1)

        if text_features.dim() == 2:
            text_3d = text_features.unsqueeze(0).expand(B, -1, -1)
        elif text_features.size(0) == B:
            text_3d = text_features
        else:
            text_3d = text_features.expand(B, -1, -1)

        if text_features.dim() == 2:
            routing_input = text_features
        else:
            routing_input = text_3d[0]

        routing_weights = self.router(routing_input)

        if self.training:
            self._routing_history.append(routing_weights.detach())

        use_sparse = self.current_epoch >= self.warmup_epochs

        if use_sparse:
            top_values, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
            top_values = top_values / top_values.sum(dim=-1, keepdim=True)
            mask = torch.zeros_like(routing_weights)
            mask.scatter_(-1, top_indices, top_values)
            routing_weights = mask

        combined = torch.zeros(
            B, C, self.expert_dim,
            device=text_3d.device, dtype=text_3d.dtype,
        )

        for i, expert in enumerate(self.experts):
            w = routing_weights[:, i]
            if use_sparse and (w.abs().sum() < 1e-8):
                continue
            expert_out = expert(text_3d, image_features)
            combined = combined + w.unsqueeze(0).unsqueeze(-1) * expert_out

        output = self.output_proj(combined)
        output = self.output_norm(output)

        return output

    def diversity_loss(self):
        loss = 0.0
        n = 0
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                qi = self.experts[i].q_proj.weight
                qj = self.experts[j].q_proj.weight
                loss = loss + torch.norm(qi.T @ qj, p='fro') ** 2
                n += 1
        return loss / max(n, 1)

    def load_balance_loss(self):
        if not self._routing_history:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        recent = torch.cat(self._routing_history[-32:], dim=0)
        avg_prob = recent.mean(dim=0)
        fraction = (recent.argmax(dim=-1).unsqueeze(-1) == torch.arange(
            self.num_experts, device=recent.device
        ).unsqueeze(0)).float().mean(dim=0)
        loss = (fraction * avg_prob).sum() * self.num_experts
        return loss

    def clear_routing_history(self):
        self._routing_history = []


class MoEImageEncoder(ImageEncoder):
    def __init__(self, clip_model, intermediate_layers=None):
        super().__init__(clip_model)
        if intermediate_layers is None:
            intermediate_layers = [3, 6, 9, 12]
        self.intermediate_layers = intermediate_layers

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

        intermediate_outputs = []
        num_blocks = len(self.transformer.resblocks)
        target_set = set()
        for layer_idx in self.intermediate_layers:
            if layer_idx <= num_blocks:
                target_set.add(layer_idx - 1)

        for idx, block in enumerate(self.transformer.resblocks):
            x = block(x)
            if idx in target_set:
                feat = x.permute(1, 0, 2)
                feat = self.ln_post(feat)
                if self.proj is not None:
                    feat = feat @ self.proj
                intermediate_outputs.append(feat)

        final_unpooled = x.permute(1, 0, 2)
        final_unpooled = self.ln_post(final_unpooled)
        if self.proj is not None:
            final_unpooled = final_unpooled @ self.proj

        if not intermediate_outputs:
            intermediate_outputs = [final_unpooled]

        global_feature = final_unpooled[:, 0, :]
        return intermediate_outputs, global_feature


class MoECustomCLIP(CustomCLIP):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__(cfg, classnames, clip_model, device)

        moe_cfg = self.model_cfg.get('moe', ConfigNode())
        if not isinstance(moe_cfg, ConfigNode):
            moe_cfg = ConfigNode(moe_cfg)

        prompt_dim = self.clip_model.text_projection.shape[1]
        num_experts = moe_cfg.get('num_experts', 8)
        top_k = moe_cfg.get('top_k', 2)
        dropout = self.model_cfg.get('dropout', 0.1)

        self.moe = MixtureOfExperts(
            feature_dim=prompt_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )

        warmup = moe_cfg.get('warmup_epochs', 5)
        self.moe.warmup_epochs = warmup

        self.diversity_weight = coerce_to_float(moe_cfg.get('diversity_weight', 0.1), 0.1)
        self.load_balance_weight = coerce_to_float(moe_cfg.get('load_balance_weight', 0.01), 0.01)

        intermediate_layers = moe_cfg.get('intermediate_layers', [3, 6, 9, 12])
        self.vis_encoder = MoEImageEncoder(self.clip_model, intermediate_layers)

        if self.training_cfg.get('precision', 'fp32') == 'fp16':
            self.moe = self.moe.half()

    def forward(self, image, label=None):
        visual_output = self.vis_encoder(image)
        unpooled_levels, image_features = visual_output

        if not isinstance(unpooled_levels, list):
            unpooled_levels = [unpooled_levels]

        multi_scale = torch.cat(unpooled_levels, dim=1)

        base_text_features = self.text_features.clone()

        adapted_text = self.moe(base_text_features, multi_scale)

        adapted_text = F.normalize(adapted_text, dim=-1)

        logit_scale = self.logit_scale.exp()
        image_features = F.normalize(image_features, dim=-1)

        logits = logit_scale * torch.bmm(
            image_features.unsqueeze(1),
            adapted_text.transpose(1, 2),
        ).squeeze(1)

        mode = self.cfg.get('mode', self.training_cfg.get('mode', 'logits'))

        if self.training and label is not None:
            ce_loss = F.cross_entropy(logits, label)
            div_loss = self.moe.diversity_loss()
            lb_loss = self.moe.load_balance_loss()
            total_loss = (
                ce_loss
                + self.diversity_weight * div_loss
                + self.load_balance_weight * lb_loss
            )
            return total_loss, logits
        elif mode == "logits":
            return logits
        elif mode == "features":
            return logits, adapted_text

        return logits

    def trainable_parameters(self):
        return self.moe.parameters()

    def get_trainable_parameter_names(self):
        return [f"moe.{name}" for name, _ in self.moe.named_parameters()]


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
        self.model.moe.train()

        precision = self._cfg_str('fp32', 'training.precision', 'precision')

        if precision == 'amp':
            from torch.cuda.amp import autocast
            with autocast():
                loss, logits = self.model(images, labels)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
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

    def save_model(self, path):
        checkpoint = {
            'moe_state_dict': self.model.moe.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'cfg': self.cfg
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if 'moe_state_dict' in checkpoint:
            self.model.moe.load_state_dict(checkpoint['moe_state_dict'])
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
            if hasattr(self.trainer.model, 'moe'):
                self.trainer.model.moe.set_epoch(epoch_idx)
                if epoch_idx == 0:
                    self.trainer.model.moe.clear_routing_history()
        return super()._run_epoch(epoch_idx, epochs_total, train_loader, run_dir)


MOE_ARG_SCHEMA = {
    'config': {'type': str, 'required': True, 'help': 'Path to YAML configuration file'},
    'output_dir': {'type': str, 'help': 'Override logging.output_dir from config', 'config_path': 'logging.output_dir'},
    'device': {'type': str, 'help': 'Override training.device from config', 'config_path': 'training.device'},
    'debug': {'type': bool, 'help': 'Enable debug output', 'default': False},
    'disable_coloring': {'type': bool, 'help': 'Disable colored output for log files', 'default': False},
}


def parse_args():
    parser = create_argument_parser("Train MoSE-APT model", MOE_ARG_SCHEMA)
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
