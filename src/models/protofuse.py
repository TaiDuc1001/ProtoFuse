import math
import torch
import torch.nn.functional as F
from clip import clip

from utils import (
    logger,
    ConfigNode,
    BaseTrainer,
    format_params,
    coerce_to_str,
    coerce_to_int,
    load_clip_to_cpu,
    compute_metrics,
)

from src.models.apt import CUSTOM_TEMPLATES


class ProtoFuse(BaseTrainer):
    DEFAULT_LR = 0.0

    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')
        self.alpha_steps = self._cfg_int(101, 'model.alpha_steps')

        data_cfg = self.cfg.get('data', ConfigNode())
        dataset_name = data_cfg.get('dataset_name', 'ImageNet')
        self.template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")

        # logger.info(f"Loading CLIP (backbone: {backbone_name})")
        clip_model = load_clip_to_cpu(backbone_name)

        precision = self._cfg_str('fp32', 'training.precision', 'precision')
        if precision in ['fp32', 'amp']:
            clip_model.float()

        self.clip_model = clip_model.to(self.device).eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        self.embed_dim = self.clip_model.text_projection.shape[1]

        prompts = [self.template.format(c.replace("_", " ")) for c in self.classnames]
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()
        self.text_prototypes = F.normalize(text_features, dim=-1)

        self.alphas = torch.linspace(0, 1, self.alpha_steps, device=self.device)
        self.fused_prototypes = None
        self.best_alpha = None

        # logger.info(f"ProtoFuse: {len(self.classnames)} classes, α steps={self.alpha_steps}")
        # logger.info(f"Template: \"{self.template}\"")
        # logger.info(f"Embed dim: {self.embed_dim}")

        self.model = None
        self.initial_model_state = {}

    def setup_optimizer(self):
        self.optimizer = None
        self.scheduler = None

    def extract_features(self, dataloader):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                features = self.clip_model.encode_image(images).float()
                all_features.append(features.cpu())
                all_labels.append(labels)
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def build_visual_centroids(self, features, labels, num_classes):
        centroids = torch.zeros(num_classes, self.embed_dim, device=self.device)
        for i in range(num_classes):
            mask = (labels == i)
            if mask.any():
                centroids[i] = F.normalize(features[mask].to(self.device).mean(0), dim=-1)
        return centroids

    def loo_cv_alpha(self, T, V_all, train_features, train_labels, num_classes):
        class_indices = [[] for _ in range(num_classes)]
        for idx, lbl in enumerate(train_labels.tolist()):
            class_indices[lbl].append(idx)

        shots_per_class = min(len(idxs) for idxs in class_indices)

        if shots_per_class < 2:
            train_norm = F.normalize(train_features.to(self.device), dim=-1)
            refined = F.normalize(
                (1 - self.alphas).view(-1, 1, 1) * T + self.alphas.view(-1, 1, 1) * V_all,
                dim=-1
            )
            logits = torch.einsum("qd,apd->aqp", train_norm, refined)
            preds = logits.argmax(dim=-1)
            scores = (preds == train_labels.to(self.device)).float().mean(dim=-1)
            best_alpha = self.alphas[scores.argmax()].item()
            best_proto = F.normalize((1 - best_alpha) * T + best_alpha * V_all, dim=-1)
        else:
            k = shots_per_class
            class_feat = torch.stack([
                train_features[class_indices[c][:k]].to(self.device) for c in range(num_classes)
            ])
            class_sums = class_feat.sum(dim=1)
            loo_scores = torch.zeros(len(self.alphas), device=self.device)

            for fold in range(k):
                held = F.normalize(class_feat[:, fold, :], dim=-1)
                V_loo = F.normalize((class_sums - class_feat[:, fold, :]) / (k - 1), dim=-1)
                refined = F.normalize(
                    (1 - self.alphas).view(-1, 1, 1) * T + self.alphas.view(-1, 1, 1) * V_loo,
                    dim=-1
                )
                logits = torch.einsum("qd,apd->aqp", held, refined)
                preds = logits.argmax(dim=-1)
                loo_scores += (preds == torch.arange(num_classes, device=self.device)).float().mean(dim=-1)

            best_alpha = self.alphas[loo_scores.argmax()].item()
            best_proto = F.normalize((1 - best_alpha) * T + best_alpha * V_all, dim=-1)

        return best_proto, best_alpha

    def fuse_and_evaluate(self, train_features, train_labels, eval_features, eval_labels, num_classes):
        V = self.build_visual_centroids(train_features, train_labels, num_classes)
        T = self.text_prototypes

        proto, alpha = self.loo_cv_alpha(T, V, train_features, train_labels, num_classes)
        self.fused_prototypes = proto
        self.best_alpha = alpha

        eval_norm = F.normalize(eval_features.to(self.device), dim=-1)

        class_indices = {}
        for idx, lbl in enumerate(train_labels.tolist()):
            class_indices.setdefault(lbl, []).append(idx)
        shots_per_class = min(len(v) for v in class_indices.values())

        if shots_per_class < 2:
            tau = 0.05
            L_T = eval_norm @ T.T
            L_V = eval_norm @ V.T
            p = F.softmax(L_V / tau, dim=-1)
            H_x = -(p * torch.log(p + 1e-8)).sum(dim=-1)
            w_x = 1.0 - (H_x / math.log(num_classes)).clamp(0.0, 1.0)
            alpha_x = alpha * (0.5 + 0.5 * w_x)
            L_final = (1 - alpha_x).unsqueeze(-1) * L_T + alpha_x.unsqueeze(-1) * L_V
            preds = L_final.argmax(dim=-1).cpu().tolist()
        else:
            logits = eval_norm @ proto.T
            preds = logits.argmax(dim=-1).cpu().tolist()

        labels_list = eval_labels.tolist()

        metrics = compute_metrics(labels_list, preds)
        metrics['alpha'] = alpha
        return metrics

    def train_step(self, batch):
        raise NotImplementedError("ProtoFuse is training-free")

    def evaluate(self, dataloader):
        if self.fused_prototypes is None:
            raise RuntimeError("Call fuse_and_evaluate first")

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                features = self.clip_model.encode_image(images).float()
                features = F.normalize(features, dim=-1)
                logits = features @ self.fused_prototypes.T
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        from utils import compute_metrics
        metrics = compute_metrics(all_labels, all_preds)
        metrics['alpha'] = self.best_alpha
        return metrics

    def save_model(self, path):
        torch.save({
            'fused_prototypes': self.fused_prototypes,
            'text_prototypes': self.text_prototypes,
            'best_alpha': self.best_alpha,
            'alpha_steps': self.alpha_steps,
            'classnames': self.classnames,
        }, path)
        # logger.info(f"ProtoFuse prototypes saved to {path}")

    def load_model(self, path):
        data = torch.load(path, map_location=self.device)
        self.fused_prototypes = data['fused_prototypes']
        self.text_prototypes = data['text_prototypes']
        self.best_alpha = data['best_alpha']
        # logger.info(f"ProtoFuse prototypes loaded from {path}")
