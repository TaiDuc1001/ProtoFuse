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


class TipAdapter(BaseTrainer):
    DEFAULT_LR = 0.001

    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')
        self.alpha = self._cfg_float(1.0, 'model.alpha')
        self.beta = self._cfg_float(5.5, 'model.beta')

        data_cfg = self.cfg.get('data', ConfigNode())
        dataset_name = data_cfg.get('dataset_name', 'ImageNet')
        self.template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")

        # logger.info(f"Loading CLIP (backbone: {backbone_name})")
        clip_model = load_clip_to_cpu(backbone_name)
        precision = self._cfg_str('fp32', 'training.precision', 'precision')
        if precision in ['fp32', 'amp']:
            clip_model.float()

        self.clip_model = clip_model.to(self.device).eval()
        for param in self.clip_model.parameters():
            param.requires_grad_(False)

        self.embed_dim = self.clip_model.text_projection.shape[1]
        self.num_classes = len(self.classnames)
        prompts = [self.template.format(c.replace("_", " ")) for c in self.classnames]
        tokenized = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokenized).float()
        self.text_features = F.normalize(text_features, dim=-1)

        self.cache_keys = None
        self.cache_values = None
        self.cache_labels = None
        self.adapter = None
        self.posthoc_fused_prototypes = None
        self.posthoc_visual_centroids = None
        self.posthoc_centroid_mask = None
        self.posthoc_alpha = None
        self.posthoc_missing_classes = []
        self.model = nn.Module()
        self.initial_model_state = {}

        # logger.info(f"Tip-Adapter: {self.num_classes} classes, alpha={self.alpha:.4f}, beta={self.beta:.4f}")
        # logger.info(f"Template: \"{self.template}\"")
        # logger.info(f"Embed dim: {self.embed_dim}")

    def setup_optimizer(self):
        self.optimizer = None
        self.scheduler = None

    def _cfg_bool(self, default, *paths):
        value = self._cfg_value(*paths, default=default)
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    def _cfg_float_list(self, default, *paths):
        value = self._cfg_value(*paths, default=default)
        if value is None:
            value = default
        if isinstance(value, str):
            return [coerce_to_float(item.strip(), 0.0) for item in value.split(',') if item.strip()]
        if isinstance(value, (list, tuple)):
            return [coerce_to_float(item, 0.0) for item in value]
        return [coerce_to_float(value, 0.0)]

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
            raise RuntimeError("Cannot extract Tip-Adapter features from an empty dataloader.")
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def build_cache(self, features, labels):
        if features.numel() == 0 or labels.numel() == 0:
            raise RuntimeError("Cannot build Tip-Adapter cache without training samples.")
        features = F.normalize(features.to(self.device).float(), dim=-1)
        labels = labels.to(self.device).long()

        self.cache_keys = features.t().contiguous()
        self.cache_values = F.one_hot(labels, num_classes=self.num_classes).float()
        self.cache_labels = labels.detach().clone()

        if self.adapter is not None:
            self.adapter = None

        # logger.info(f"Built Tip-Adapter cache: keys={tuple(self.cache_keys.shape)}, values={tuple(self.cache_values.shape)}")

    def clear_posthoc_protofuse(self):
        self.posthoc_fused_prototypes = None
        self.posthoc_visual_centroids = None
        self.posthoc_centroid_mask = None
        self.posthoc_alpha = None
        self.posthoc_missing_classes = []

    def get_text_prototypes(self):
        return self.text_features.detach().clone()

    def apply_posthoc_protofuse(self, alpha, fused_prototypes, visual_centroids=None, centroid_mask=None, missing_classes=None):
        self.posthoc_fused_prototypes = F.normalize(fused_prototypes.to(self.device).float(), dim=-1)
        self.posthoc_visual_centroids = (
            F.normalize(visual_centroids.to(self.device).float(), dim=-1)
            if visual_centroids is not None else None
        )
        self.posthoc_centroid_mask = centroid_mask.to(self.device).bool() if centroid_mask is not None else None
        self.posthoc_alpha = alpha
        self.posthoc_missing_classes = list(missing_classes or [])
        return self.posthoc_fused_prototypes

    def _clip_logits(self, image_features, text_features=None):
        text_features = self.text_features if text_features is None else text_features.to(self.device).float()
        logit_scale = self.clip_model.logit_scale.exp()
        return logit_scale * image_features @ text_features.t()

    def _cache_logits(self, image_features, beta=None, exclude_self=False):
        if self.cache_keys is None or self.cache_values is None:
            raise RuntimeError("Call build_cache before computing Tip-Adapter logits.")

        beta = self.beta if beta is None else beta
        if self.adapter is not None:
            affinity = self.adapter(image_features)
        else:
            affinity = image_features @ self.cache_keys
        if exclude_self:
            diag_count = min(affinity.shape[0], affinity.shape[1])
            diag_idx = torch.arange(diag_count, device=affinity.device)
            affinity[diag_idx, diag_idx] = -float('inf')
        return torch.exp(-beta + beta * affinity) @ self.cache_values

    def logits_from_features(self, features, alpha=None, beta=None, exclude_self=False, text_features=None):
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        image_features = F.normalize(features.to(self.device).float(), dim=-1)
        clip_logits = self._clip_logits(image_features, text_features=text_features)
        cache_logits = self._cache_logits(image_features, beta=beta, exclude_self=exclude_self)
        return clip_logits + alpha * cache_logits

    def evaluate_features(self, features, labels, alpha=None, beta=None, exclude_self=False, text_features=None):
        labels_device = labels.to(self.device).long()
        with torch.no_grad():
            logits = self.logits_from_features(
                features,
                alpha=alpha,
                beta=beta,
                exclude_self=exclude_self,
                text_features=text_features,
            )
            loss = F.cross_entropy(logits, labels_device)
            preds = logits.argmax(dim=-1)

        labels_list = labels_device.cpu().tolist()
        preds_list = preds.cpu().tolist()
        metrics = compute_metrics(labels_list, preds_list)
        metrics['loss'] = loss.item()
        metrics['predictions'] = preds_list
        metrics['true_labels'] = labels_list
        metrics['alpha'] = self.alpha if alpha is None else alpha
        metrics['beta'] = self.beta if beta is None else beta
        return metrics

    def tune_alpha_beta(self, features, labels):
        search_cfg = self.model_cfg.get('search', ConfigNode())
        if not bool(search_cfg.get('enabled', False)):
            return self.alpha, self.beta, None

        alpha_values = self._cfg_float_list([self.alpha], 'model.search.alpha_values')
        beta_values = self._cfg_float_list([self.beta], 'model.search.beta_values')
        if not alpha_values or not beta_values:
            return self.alpha, self.beta, None

        best = {
            'accuracy': -1.0,
            'loss': float('inf'),
            'alpha': self.alpha,
            'beta': self.beta,
        }

        # logger.info(f"Searching Tip-Adapter alpha/beta over {len(alpha_values)}x{len(beta_values)} grid")
        for alpha in alpha_values:
            for beta in beta_values:
                metrics = self.evaluate_features(
                    features,
                    labels,
                    alpha=alpha,
                    beta=beta,
                    exclude_self=True,
                )
                acc = metrics.get('accuracy', 0.0)
                loss = metrics.get('loss', float('inf'))
                if acc > best['accuracy'] or (acc == best['accuracy'] and loss < best['loss']):
                    best = {
                        'accuracy': acc,
                        'loss': loss,
                        'alpha': alpha,
                        'beta': beta,
                    }

        self.alpha = best['alpha']
        self.beta = best['beta']
        # logger.info(f"Selected Tip-Adapter alpha={self.alpha:.4f}, beta={self.beta:.4f} (LOO acc={best['accuracy']:.2f}%)")
        return self.alpha, self.beta, best

    def finetune_adapter(self, train_loader):
        finetune_cfg = self.model_cfg.get('finetune', ConfigNode())
        if not bool(finetune_cfg.get('enabled', False)):
            return []
        features, labels = self.extract_features(train_loader)
        return self.finetune_adapter_from_features(features, labels)

    def finetune_adapter_from_features(self, features, labels):
        finetune_cfg = self.model_cfg.get('finetune', ConfigNode())
        if not bool(finetune_cfg.get('enabled', False)):
            return []
        if self.cache_keys is None or self.cache_values is None:
            raise RuntimeError("Build cache before Tip-Adapter-F fine-tuning.")

        epochs = coerce_to_int(finetune_cfg.get('epochs', 20), 20)
        lr = coerce_to_float(finetune_cfg.get('lr', self.DEFAULT_LR), self.DEFAULT_LR)
        weight_decay = coerce_to_float(finetune_cfg.get('weight_decay', 0.0), 0.0)
        batch_size = self._cfg_int(128, 'training.batch_size')

        adapter = nn.Linear(self.embed_dim, self.cache_keys.shape[1], bias=False).to(self.device)
        with torch.no_grad():
            adapter.weight.copy_(self.cache_keys.t())
        self.adapter = adapter
        self.model = self.adapter

        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        self.adapter.weight.requires_grad_(True)

        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, weight_decay=weight_decay)
        history = []
        # logger.info(f"Fine-tuning Tip-Adapter-F for {epochs} epochs")

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
                with torch.no_grad():
                    clip_logits = self._clip_logits(image_features)

                affinity = self.adapter(image_features)
                cache_logits = torch.exp(-self.beta + self.beta * affinity) @ self.cache_values
                logits = clip_logits + self.alpha * cache_logits
                loss = F.cross_entropy(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=-1)
                running_loss += loss.item()
                running_acc += (preds == batch_labels).float().mean().item() * 100.0
                steps += 1

            epoch_result = {
                'epoch': epoch_idx,
                'loss': running_loss / max(1, steps),
                'accuracy': running_acc / max(1, steps),
            }
            history.append(epoch_result)
            # logger.info(f"Tip-Adapter-F Epoch {epoch_idx} - loss={epoch_result['loss']:.4f} - acc={epoch_result['accuracy']:.2f}%")

        self.adapter.eval()
        return history

    def train_step(self, batch):
        raise NotImplementedError("Tip-Adapter is cache-based; use the pipeline to build/evaluate.")

    def evaluate(self, dataloader):
        features, labels = self.extract_features(dataloader)
        return self.evaluate_features(features, labels)

    def save_model(self, path):
        checkpoint = {
            'cache_keys': self.cache_keys.detach().cpu() if self.cache_keys is not None else None,
            'cache_values': self.cache_values.detach().cpu() if self.cache_values is not None else None,
            'cache_labels': self.cache_labels.detach().cpu() if self.cache_labels is not None else None,
            'text_features': self.text_features.detach().cpu(),
            'alpha': self.alpha,
            'beta': self.beta,
            'classnames': self.classnames,
            'template': self.template,
            'backbone': self._cfg_str('ViT-B/16', 'model.backbone', 'backbone'),
            'adapter_state_dict': self.adapter.state_dict() if self.adapter is not None else None,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, path)
        # logger.info(f"Tip-Adapter state saved to {path}")

    def save_posthoc_protofuse(self, path):
        torch.save({
            'fused_prototypes': self.posthoc_fused_prototypes.detach().cpu()
            if self.posthoc_fused_prototypes is not None else None,
            'visual_centroids': self.posthoc_visual_centroids.detach().cpu()
            if self.posthoc_visual_centroids is not None else None,
            'centroid_mask': self.posthoc_centroid_mask.detach().cpu()
            if self.posthoc_centroid_mask is not None else None,
            'text_prototypes': self.text_features.detach().cpu(),
            'alpha': self.posthoc_alpha,
            'missing_classes': self.posthoc_missing_classes,
            'tip_alpha': self.alpha,
            'tip_beta': self.beta,
            'classnames': self.classnames,
            'template': self.template,
        }, path)

    def load_model(self, path):
        data = torch.load(path, map_location=self.device)
        self.cache_keys = data['cache_keys'].to(self.device) if data.get('cache_keys') is not None else None
        self.cache_values = data['cache_values'].to(self.device) if data.get('cache_values') is not None else None
        self.cache_labels = data.get('cache_labels')
        if self.cache_labels is not None:
            self.cache_labels = self.cache_labels.to(self.device)
        self.text_features = data['text_features'].to(self.device)
        self.alpha = data['alpha']
        self.beta = data['beta']
        adapter_state = data.get('adapter_state_dict')
        if adapter_state is not None and self.cache_keys is not None:
            self.adapter = nn.Linear(self.embed_dim, self.cache_keys.shape[1], bias=False).to(self.device)
            self.adapter.load_state_dict(adapter_state)
            self.adapter.eval()
            self.model = self.adapter
        # logger.info(f"Tip-Adapter state loaded from {path}")
