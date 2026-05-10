import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip

from utils import (
    logger,
    ConfigNode,
    BaseTrainer,
    coerce_to_float,
    coerce_to_int,
    load_clip_to_cpu,
    compute_metrics,
)

from src.models.apt import CUSTOM_TEMPLATES


class ProtoAdapter(BaseTrainer):
    DEFAULT_LR = 4e-4

    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')
        self.alpha = self._cfg_float(1.0, 'model.alpha')

        data_cfg = self.cfg.get('data', ConfigNode())
        dataset_name = data_cfg.get('dataset_name', 'ImageNet')
        self.template = CUSTOM_TEMPLATES.get(dataset_name, 'a photo of a {}.')

        # logger.info(f'Loading CLIP (backbone: {backbone_name})')
        clip_model = load_clip_to_cpu(backbone_name)
        precision = self._cfg_str('fp32', 'training.precision', 'precision')
        if precision in ['fp32', 'amp']:
            clip_model.float()

        self.clip_model = clip_model.to(self.device).eval()
        for param in self.clip_model.parameters():
            param.requires_grad_(False)

        self.embed_dim = self.clip_model.text_projection.shape[1]
        self.num_classes = len(self.classnames)
        prompts = [self.template.format(c.replace('_', ' ')) for c in self.classnames]
        tokenized = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokenized).float()
        self.text_features = F.normalize(text_features, dim=-1)

        self.proto_weights = None
        self.adapter = None
        self.model = nn.Module()
        self.initial_model_state = {}

        # logger.info(f'Proto-Adapter: {self.num_classes} classes, alpha={self.alpha:.4f}')
        # logger.info(f'Template: "{self.template}"')
        # logger.info(f'Embed dim: {self.embed_dim}')

    def setup_optimizer(self):
        self.optimizer = None
        self.scheduler = None

    def _cfg_bool(self, default, *paths):
        value = self._cfg_value(*paths, default=default)
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    def extract_features(self, dataloader):
        all_features = []
        all_labels = []
        self.clip_model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                features = self.clip_model.encode_image(images).float()
                features = F.normalize(features, dim=-1)
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
        if not all_features:
            raise RuntimeError('Cannot extract Proto-Adapter features from an empty dataloader.')
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def build_prototypes(self, features, labels):
        if features.numel() == 0 or labels.numel() == 0:
            raise RuntimeError('Cannot build prototypes without training samples.')

        features = features.float()
        labels = labels.long()
        proto = torch.zeros(self.num_classes, self.embed_dim)
        counts = torch.zeros(self.num_classes)
        for i in range(len(labels)):
            c = labels[i].item()
            proto[c] += features[i].cpu()
            counts[c] += 1
        valid = counts > 0
        proto[valid] /= counts[valid].unsqueeze(1)

        proto = F.normalize(proto, dim=0)
        proto = F.normalize(proto, dim=1)

        self.proto_weights = proto.t().to(self.device)

        # logger.info(f'Built Proto-Adapter prototypes: shape={tuple(self.proto_weights.shape)}')

    def _clip_logits(self, image_features):
        logit_scale = self.clip_model.logit_scale.exp()
        return logit_scale * image_features @ self.text_features.t()

    def _proto_logits(self, image_features):
        if self.proto_weights is None:
            raise RuntimeError('Call build_prototypes before computing Proto-Adapter logits.')
        if self.adapter is not None:
            return self.adapter(image_features)
        return image_features @ self.proto_weights

    def logits_from_features(self, features, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        image_features = F.normalize(features.to(self.device).float(), dim=-1)
        clip_logits = self._clip_logits(image_features)
        proto_logits = self._proto_logits(image_features)
        return clip_logits + alpha * proto_logits

    def evaluate_features(self, features, labels, alpha=None):
        labels_device = labels.to(self.device).long()
        with torch.no_grad():
            logits = self.logits_from_features(features, alpha=alpha)
            loss = F.cross_entropy(logits, labels_device)
            preds = logits.argmax(dim=-1)

        labels_list = labels_device.cpu().tolist()
        preds_list = preds.cpu().tolist()
        metrics = compute_metrics(labels_list, preds_list)
        metrics['loss'] = loss.item()
        metrics['predictions'] = preds_list
        metrics['true_labels'] = labels_list
        metrics['alpha'] = self.alpha if alpha is None else alpha
        return metrics

    def tune_alpha(self, features, labels):
        search_cfg = self.model_cfg.get('search', ConfigNode())
        if not bool(search_cfg.get('enabled', False)):
            return self.alpha, None

        raw = search_cfg.get('alpha_values', [self.alpha])
        if isinstance(raw, str):
            alpha_values = [coerce_to_float(v.strip(), 0.0) for v in raw.split(',') if v.strip()]
        elif isinstance(raw, (list, tuple)):
            alpha_values = [coerce_to_float(v, 0.0) for v in raw]
        else:
            alpha_values = [coerce_to_float(raw, 0.0)]

        if not alpha_values:
            return self.alpha, None

        best = {'accuracy': -1.0, 'loss': float('inf'), 'alpha': self.alpha}
        # logger.info(f'Searching Proto-Adapter alpha over {len(alpha_values)} values')
        for alpha in alpha_values:
            metrics = self.evaluate_features(features, labels, alpha=alpha)
            acc = metrics.get('accuracy', 0.0)
            loss = metrics.get('loss', float('inf'))
            if acc > best['accuracy'] or (acc == best['accuracy'] and loss < best['loss']):
                best = {'accuracy': acc, 'loss': loss, 'alpha': alpha}

        self.alpha = best['alpha']
        # logger.info(f'Selected Proto-Adapter alpha={self.alpha:.4f} (acc={best["accuracy"]:.2f}%)')
        return self.alpha, best

    def finetune_adapter_from_features(self, features, labels):
        finetune_cfg = self.model_cfg.get('finetune', ConfigNode())
        if not bool(finetune_cfg.get('enabled', False)):
            return []
        if self.proto_weights is None:
            raise RuntimeError('Build prototypes before Proto-Adapter-F fine-tuning.')

        epochs = coerce_to_int(finetune_cfg.get('epochs', 20), 20)
        lr = coerce_to_float(finetune_cfg.get('lr', self.DEFAULT_LR), self.DEFAULT_LR)
        weight_decay = coerce_to_float(finetune_cfg.get('weight_decay', 0.0), 0.0)
        margin = coerce_to_float(finetune_cfg.get('margin', 0.2), 0.2)
        scale = coerce_to_float(finetune_cfg.get('scale', 64.0), 64.0)
        batch_size = self._cfg_int(256, 'training.batch_size')

        adapter = nn.Linear(self.embed_dim, self.num_classes, bias=False).to(self.device)
        with torch.no_grad():
            adapter.weight.copy_(self.proto_weights.t())
        self.adapter = adapter
        self.model = self.adapter

        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        self.adapter.weight.requires_grad_(True)

        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = epochs * max(1, math.ceil(features.shape[0] / batch_size))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr / 10.0
        )
        history = []
        # logger.info(f'Fine-tuning Proto-Adapter-F for {epochs} epochs (margin={margin})')

        for epoch_idx in range(1, epochs + 1):
            running_loss = 0.0
            running_acc = 0.0
            steps = 0
            self.adapter.train()
            order = torch.randperm(features.shape[0])

            for start in range(0, features.shape[0], batch_size):
                batch_idx = order[start:start + batch_size]
                image_features = F.normalize(features[batch_idx].to(self.device).float(), dim=-1)
                batch_labels = labels[batch_idx].to(self.device).long()

                W = F.normalize(self.adapter.weight, dim=1)
                cos_theta = image_features @ W.t()
                cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

                if margin > 0.0:
                    theta_yi = torch.acos(cos_theta.gather(1, batch_labels.unsqueeze(1)).squeeze(1))
                    cos_theta_m = torch.cos(theta_yi + margin)
                    one_hot = F.one_hot(batch_labels, num_classes=self.num_classes).float()
                    logits_arc = scale * (one_hot * cos_theta_m.unsqueeze(1) + (1 - one_hot) * cos_theta)
                else:
                    logits_arc = scale * cos_theta

                with torch.no_grad():
                    clip_logits = self._clip_logits(image_features)
                combined = clip_logits + self.alpha * logits_arc
                loss = F.cross_entropy(combined, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                preds = combined.argmax(dim=-1)
                running_loss += loss.item()
                running_acc += (preds == batch_labels).float().mean().item() * 100.0
                steps += 1

            epoch_result = {
                'epoch': epoch_idx,
                'loss': running_loss / max(1, steps),
                'accuracy': running_acc / max(1, steps),
            }
            history.append(epoch_result)
            # logger.info(f'Proto-Adapter-F Epoch {epoch_idx} - ' f'loss={epoch_result["loss"]:.4f} - acc={epoch_result["accuracy"]:.2f}%')

        self.adapter.eval()
        return history

    def train_step(self, batch):
        raise NotImplementedError('Proto-Adapter is prototype-based; use the pipeline to build/evaluate.')

    def evaluate(self, dataloader):
        features, labels = self.extract_features(dataloader)
        return self.evaluate_features(features, labels)

    def save_model(self, path):
        checkpoint = {
            'proto_weights': self.proto_weights.detach().cpu() if self.proto_weights is not None else None,
            'text_features': self.text_features.detach().cpu(),
            'alpha': self.alpha,
            'classnames': self.classnames,
            'template': self.template,
            'backbone': self._cfg_str('ViT-B/16', 'model.backbone', 'backbone'),
            'adapter_state_dict': self.adapter.state_dict() if self.adapter is not None else None,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, path)
        # logger.info(f'Proto-Adapter state saved to {path}')

    def load_model(self, path):
        data = torch.load(path, map_location=self.device)
        self.proto_weights = (
            data['proto_weights'].to(self.device) if data.get('proto_weights') is not None else None
        )
        self.text_features = data['text_features'].to(self.device)
        self.alpha = data['alpha']
        adapter_state = data.get('adapter_state_dict')
        if adapter_state is not None:
            self.adapter = nn.Linear(self.embed_dim, self.num_classes, bias=False).to(self.device)
            self.adapter.load_state_dict(adapter_state)
            self.adapter.eval()
            self.model = self.adapter
        # logger.info(f'Proto-Adapter state loaded from {path}')
