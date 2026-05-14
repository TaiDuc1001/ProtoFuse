import torch
import torch.nn.functional as F
from clip import clip

from utils import (
    ConfigNode,
    BaseTrainer,
    coerce_to_float,
    load_clip_to_cpu,
    compute_metrics,
)

from src.models.apt import CUSTOM_TEMPLATES


class ProtoFuse(BaseTrainer):
    DEFAULT_LR = 0.0

    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')
        self.alpha_steps = self._cfg_int(101, 'model.alpha_steps')
        self.centroid_mix_beta_values = self._cfg_float_list(
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
            'model.centroid_mix.beta_values',
        )

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

    def _cfg_float_list(self, default, *paths):
        raw = self._cfg_value(*paths, default=default)
        if isinstance(raw, str):
            values = [v.strip() for v in raw.split(',') if v.strip()]
        elif isinstance(raw, (list, tuple)):
            values = raw
        else:
            values = [raw]
        return [coerce_to_float(v, 0.0) for v in values]

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

    def _weighted_visual_centroid(self, class_features, text_prototype):
        class_features = class_features.to(self.device)
        text_prototype = text_prototype.to(self.device)
        similarities = F.cosine_similarity(
            F.normalize(class_features, dim=-1),
            F.normalize(text_prototype, dim=-1).unsqueeze(0),
            dim=-1,
        ).clamp_min(0.0)
        sim_sum = similarities.sum()
        if sim_sum <= 1e-12:
            weights = torch.full_like(similarities, 1.0 / similarities.numel())
        else:
            weights = similarities / sim_sum
        return F.normalize((weights.unsqueeze(-1) * class_features).sum(dim=0), dim=-1)

    def build_visual_centroids(self, features, labels, num_classes):
        centroids = torch.zeros(num_classes, self.embed_dim, device=self.device)
        for i in range(num_classes):
            mask = (labels == i)
            if mask.any():
                centroids[i] = self._weighted_visual_centroid(features[mask], self.text_prototypes[i])
        return centroids

    def _curve_knee(self, values):
        values = values.float()
        span = values.max() - values.min()
        if span <= 1e-12:
            return None, 0.0, 0.0
        y = (values - values.min()) / (span + 1e-12)
        x = torch.linspace(0.0, 1.0, len(values), device=self.device)
        knee_scores = y - x
        knee_idx = int(knee_scores.argmax().item())
        return knee_idx, knee_scores[knee_idx].item(), span.item()

    def _centroid_mix_neighbors(self, T, V_all, mode):
        if mode == 'vv':
            similarity = V_all @ V_all.T
        elif mode == 'tt':
            similarity = T @ T.T
        elif mode == 'hybrid':
            similarity = 0.5 * (V_all @ V_all.T) + 0.5 * (T @ T.T)
        else:
            raise ValueError(f"Unknown centroid-mix neighbor mode: {mode}")
        similarity = similarity.clone()
        similarity.fill_diagonal_(-float('inf'))
        return similarity.argmax(dim=1)

    def _centroid_mix_net_curve(self, T, V_all, neighbors, beta, num_classes):
        labels = torch.arange(num_classes, device=self.device)
        pseudo_features = F.normalize((1.0 - beta) * V_all + beta * V_all[neighbors], dim=-1)
        text_preds = (pseudo_features @ T.T).argmax(dim=-1)
        text_correct = text_preds.eq(labels)

        net_scores = []
        for alpha in self.alphas:
            proto = F.normalize((1.0 - alpha) * T + alpha * V_all, dim=-1)
            fused_preds = (pseudo_features @ proto.T).argmax(dim=-1)
            fused_correct = fused_preds.eq(labels)
            rescue = (~text_correct) & fused_correct
            damage = text_correct & ~fused_correct
            net_scores.append(rescue.sum().float() - damage.sum().float())
        return torch.stack(net_scores)

    def _centroid_mix_alpha(self, T, V_all, num_classes):
        beta_values = sorted({round(float(b), 6) for b in self.centroid_mix_beta_values if 0.0 < float(b) < 0.5})
        if 0.45 not in beta_values:
            beta_values.append(0.45)
            beta_values.sort()
        if num_classes < 2 or not beta_values:
            return 0.0

        best = {'score': -float('inf'), 'alpha': 0.0}
        for mode in ('vv', 'tt', 'hybrid'):
            neighbors = self._centroid_mix_neighbors(T, V_all, mode)
            for beta in beta_values:
                net_curve = self._centroid_mix_net_curve(T, V_all, neighbors, beta, num_classes)
                knee_idx, knee_strength, signal_span = self._curve_knee(net_curve)
                if knee_idx is None:
                    continue
                amplitude = signal_span / max(1, num_classes)
                quality = knee_strength * amplitude
                alpha = self.alphas[knee_idx].item()
                if quality > best['score'] or (quality == best['score'] and alpha < best['alpha']):
                    best = {'score': quality, 'alpha': alpha}

        return best['alpha'] if best['score'] > 0.0 else 0.0

    def hopc_alpha(self, T, V_all, train_features, train_labels, num_classes):
        class_indices = [[] for _ in range(num_classes)]
        for idx, lbl in enumerate(train_labels.tolist()):
            class_indices[lbl].append(idx)

        shots_per_class = min(len(idxs) for idxs in class_indices)

        if shots_per_class < 2:
            best_alpha = self._centroid_mix_alpha(T, V_all, num_classes)
            best_proto = F.normalize((1 - best_alpha) * T + best_alpha * V_all, dim=-1)
        else:
            k = shots_per_class
            class_feat = torch.stack([
                train_features[class_indices[c][:k]].to(self.device) for c in range(num_classes)
            ])
            net_scores = torch.zeros(len(self.alphas), device=self.device)
            targets = torch.arange(num_classes, device=self.device)

            for hold_idx in range(k):
                held = F.normalize(class_feat[:, hold_idx, :], dim=-1)
                V_minus = torch.stack([
                    self._weighted_visual_centroid(
                        class_feat[c, torch.arange(k, device=self.device) != hold_idx],
                        T[c],
                    )
                    for c in range(num_classes)
                ])

                text_preds = (held @ T.T).argmax(dim=-1)
                text_correct = text_preds.eq(targets)

                refined = F.normalize(
                    (1 - self.alphas).view(-1, 1, 1) * T + self.alphas.view(-1, 1, 1) * V_minus,
                    dim=-1
                )
                fused_preds = torch.einsum("cd,akd->ack", held, refined).argmax(dim=-1)
                fused_correct = fused_preds.eq(targets.view(1, -1))

                rescue = (~text_correct).view(1, -1) & fused_correct
                damage = text_correct.view(1, -1) & ~fused_correct
                net_scores += rescue.sum(dim=1).float() - damage.sum(dim=1).float()

            best_alpha = self.alphas[net_scores.argmax()].item()
            best_proto = F.normalize((1 - best_alpha) * T + best_alpha * V_all, dim=-1)

        return best_proto, best_alpha

    def fuse_and_evaluate(self, train_features, train_labels, eval_features, eval_labels, num_classes):
        V = self.build_visual_centroids(train_features, train_labels, num_classes)
        T = self.text_prototypes

        proto, alpha = self.hopc_alpha(T, V, train_features, train_labels, num_classes)
        self.fused_prototypes = proto
        self.best_alpha = alpha

        eval_norm = F.normalize(eval_features.to(self.device), dim=-1)
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
