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
    NUMERICAL_EPSILON = 1e-12

    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')
        self.alpha_steps = self._cfg_int(101, 'model.alpha_steps')
        self.rho = None
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
        self.alpha_init = None
        self.alpha_final = None
        self.selected_candidate = None
        self.candidate_scores = None
        self.support_visual_centroids = None
        self.query_centroids = None
        self.expanded_visual_centroids = None

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
        return self._coerce_float_list(raw, default)

    @staticmethod
    def _coerce_float_list(raw, default):
        if raw is None:
            return [coerce_to_float(v, 0.0) for v in default]
        if isinstance(raw, str):
            values = [v.strip() for v in raw.split(',') if v.strip()]
        elif isinstance(raw, (list, tuple)):
            values = raw
        else:
            values = [raw]
        return [coerce_to_float(v, 0.0) for v in values]

    @classmethod
    def from_precomputed(
        cls,
        text_prototypes,
        device,
        alpha_steps=101,
        beta_values=None,
        rho=None,
        classnames=None,
    ):
        trainer = cls.__new__(cls)
        trainer.device = torch.device(device)
        trainer.classnames = list(classnames or [])
        trainer.alpha_steps = max(2, int(alpha_steps))
        trainer.rho = None
        trainer.centroid_mix_beta_values = cls._coerce_float_list(
            beta_values,
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        )
        trainer.alphas = torch.linspace(0, 1, trainer.alpha_steps, device=trainer.device)
        trainer.text_prototypes = F.normalize(
            text_prototypes.to(trainer.device).float(),
            dim=-1,
        )
        trainer.embed_dim = trainer.text_prototypes.shape[-1]
        trainer.fused_prototypes = None
        trainer.best_alpha = None
        trainer.alpha_init = None
        trainer.alpha_final = None
        trainer.selected_candidate = None
        trainer.candidate_scores = None
        trainer.support_visual_centroids = None
        trainer.query_centroids = None
        trainer.expanded_visual_centroids = None
        return trainer

    @classmethod
    def posthoc_fuse(
        cls,
        text_prototypes,
        train_features,
        train_labels,
        device,
        alpha_steps=101,
        beta_values=None,
        query_features=None,
        rho=None,
        query_batch_size=None,
    ):
        selector = cls.from_precomputed(
            text_prototypes,
            device,
            alpha_steps=alpha_steps,
            beta_values=beta_values,
            rho=rho,
        )

        train_features = F.normalize(train_features.to(selector.device).float(), dim=-1)
        train_labels = train_labels.to(selector.device).long()
        num_classes = selector.text_prototypes.shape[0]
        class_counts = torch.bincount(train_labels, minlength=num_classes)
        is_one_shot = bool(class_counts.numel() == num_classes and class_counts.eq(1).all().item())

        visual_centroids = selector.build_visual_centroids(train_features, train_labels, num_classes)
        fused_prototypes, alpha_init = selector.hopc_alpha(
            selector.text_prototypes,
            visual_centroids,
            train_features,
            train_labels,
            num_classes,
        )
        import math
        kshot = int(class_counts.min().item())
        if kshot < 1:
            kshot = 1
        beta_val = min(0.45, 0.30 / math.sqrt(kshot))
        rho = min(1.0, 0.50 / math.sqrt(kshot))
        selector.rho = rho
        
        Q_adversarial = selector._generate_sqs_adversarial(visual_centroids, fused_prototypes, beta_val)
        query_centroids = selector.pseudo_label_aggregation(
            Q_adversarial.view(-1, Q_adversarial.shape[-1]),
            selector.text_prototypes,
            visual_centroids,
            alpha_init,
        )
        expanded_centroids = F.normalize((1.0 - rho) * visual_centroids + rho * query_centroids, dim=-1)
        alpha = alpha_init
        alpha_final = alpha_init
        selected_candidate = "sqs_adversarial_fixed"
        candidate_scores = {"sqs_adversarial_fixed": 0.0}
        
        fused_prototypes = F.normalize(
            (1.0 - alpha) * selector.text_prototypes
            + alpha * expanded_centroids,
            dim=-1,
        )

        centroid_mask = torch.zeros(num_classes, device=selector.device, dtype=torch.bool)
        valid_labels = train_labels[(train_labels >= 0) & (train_labels < num_classes)]
        if valid_labels.numel() > 0:
            centroid_mask[valid_labels.unique()] = True
        fused_for_inference = fused_prototypes.clone()
        fused_for_inference[~centroid_mask] = selector.text_prototypes[~centroid_mask]

        return {
            'fused_prototypes': fused_for_inference,
            'raw_fused_prototypes': fused_prototypes,
            'visual_centroids': expanded_centroids,
            'support_visual_centroids': visual_centroids,
            'query_centroids': query_centroids,
            'text_prototypes': selector.text_prototypes,
            'centroid_mask': centroid_mask,
            'missing_classes': torch.nonzero(~centroid_mask, as_tuple=False).flatten().cpu().tolist(),
            'alpha': alpha,
            'alpha_init': alpha_init,
            'alpha_final': alpha_final,
            'rho': selector.rho,
            'selected_candidate': selected_candidate,
            'candidate_scores': candidate_scores,
        }

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

    def _visual_centroid(self, class_features):
        class_features = class_features.to(self.device)
        return F.normalize(class_features.mean(dim=0), dim=-1)

    def build_visual_centroids(self, features, labels, num_classes):
        centroids = torch.zeros(num_classes, self.embed_dim, device=self.device)
        for i in range(num_classes):
            mask = (labels == i)
            if mask.any():
                centroids[i] = self._visual_centroid(features[mask])
        return centroids

    def _generate_sqs_adversarial(self, V, proto_before, beta, top_k=5, samples_per_nb=2, std=0.02):
        num_classes, D = V.shape
        top_k = min(top_k, num_classes - 1)
        sim = proto_before @ proto_before.T
        sim.fill_diagonal_(-float('inf'))
        Q = []
        for c in range(num_classes):
            nbs = torch.topk(sim[c], k=top_k).indices
            class_q = []
            for m in nbs:
                q_base = F.normalize((1.0 - beta) * proto_before[c] + beta * proto_before[m], dim=-1)
                noise = torch.randn(samples_per_nb, D, device=self.device) * std
                q_candidates = F.normalize(q_base.unsqueeze(0) + noise, dim=-1)
                logits = q_candidates @ proto_before.T
                preds = logits.argmax(dim=-1)
                valid = q_candidates[preds == c]
                if len(valid) == 0:
                    class_q.append(q_candidates)
                else:
                    if len(valid) < samples_per_nb:
                        pad = q_candidates[:samples_per_nb - len(valid)]
                        valid = torch.cat([valid, pad], dim=0)
                    class_q.append(valid[:samples_per_nb])
            Q.append(torch.cat(class_q, dim=0))
        return torch.stack(Q, dim=0)

    def pseudo_label_aggregation(
        self,
        query_features,
        T,
        V,
        alpha_init,
        batch_size=None,
    ):
        num_classes = T.shape[0]
        query_centroids = V.clone()

        if query_features.numel() == 0 or num_classes < 2:
            return query_centroids

        initial_prototypes = F.normalize(
            (1.0 - alpha_init) * T + alpha_init * V,
            dim=-1,
        )
        total_queries = int(query_features.shape[0])
        batch_size = total_queries if batch_size is None else max(1, int(batch_size))
        sample_counts = torch.zeros(num_classes, device=self.device, dtype=V.dtype)
        feature_sums = torch.zeros_like(V)

        for start in range(0, total_queries, batch_size):
            features = F.normalize(
                query_features[start:start + batch_size].to(self.device).float(),
                dim=-1,
            )
            similarities = features @ initial_prototypes.T
            pseudo_labels = similarities.argmax(dim=-1)
            sample_counts.index_add_(
                0,
                pseudo_labels,
                torch.ones_like(pseudo_labels, dtype=V.dtype),
            )
            feature_sums.index_add_(0, pseudo_labels, features)

        valid_mass = sample_counts > self.NUMERICAL_EPSILON
        if valid_mass.any():
            query_centroids[valid_mass] = F.normalize(
                feature_sums[valid_mass] / sample_counts[valid_mass].unsqueeze(-1),
                dim=-1,
            )
        return query_centroids

    def expand_visual_centroids(self, V, query_centroids):
        return F.normalize((1.0 - self.rho) * V + self.rho * query_centroids, dim=-1)

    def _centroid_mix_neighbors(self, V_all):
        similarity = V_all @ V_all.T
        similarity = similarity.clone()
        similarity.fill_diagonal_(-float('inf'))
        return similarity.argmax(dim=1)

    def _centroid_mix_net_curve(self, T, V_all, neighbors, beta, num_classes):
        labels = torch.arange(num_classes, device=self.device)
        pseudo_features = F.normalize((1.0 - beta) * V_all + beta * V_all[neighbors], dim=-1)
        net_scores = []
        for alpha in self.alphas:
            proto = F.normalize((1.0 - alpha) * T + alpha * V_all, dim=-1)
            fused_preds = (pseudo_features @ proto.T).argmax(dim=-1)
            fused_correct = fused_preds.eq(labels)
            net_scores.append(fused_correct.sum().float())
        return torch.stack(net_scores)

    def _centroid_mix_beta_values(self):
        beta_values = sorted({round(float(b), 6) for b in self.centroid_mix_beta_values if 0.0 < float(b) < 0.5})
        if 0.45 not in beta_values:
            beta_values.append(0.45)
            beta_values.sort()
        return beta_values

    def _centroid_mix_alpha(self, T, V_all, num_classes):
        beta_values = self._centroid_mix_beta_values()
        if num_classes < 2 or not beta_values:
            return 0.0

        best = {'score': -float('inf'), 'alpha': 0.0}
        neighbors = self._centroid_mix_neighbors(V_all)
        for beta in beta_values:
            curve = self._centroid_mix_net_curve(T, V_all, neighbors, beta, num_classes)
            score = float(curve.max().item())
            alpha = float(self.alphas[int(curve.argmax().item())].item())
            if score > best['score'] or (score == best['score'] and alpha < best['alpha']):
                best = {'score': score, 'alpha': alpha}
        return best['alpha']

    def support_calibration_score(
        self,
        T,
        V_all,
        alpha,
        train_features,
        train_labels,
        num_classes,
        query_centroids=None,
    ):
        class_indices = [
            torch.nonzero(train_labels == class_idx, as_tuple=False).flatten()
            for class_idx in range(num_classes)
        ]
        shots_per_class = min(int(indices.numel()) for indices in class_indices)

        if shots_per_class < 2:
            alpha_idx = int(torch.abs(self.alphas - float(alpha)).argmin().item())
            beta_values = sorted({
                round(float(beta), 6)
                for beta in self.centroid_mix_beta_values
                if 0.0 < float(beta) < 0.5
            })
            if 0.45 not in beta_values:
                beta_values.append(0.45)
                beta_values.sort()
            neighbors = self._centroid_mix_neighbors(V_all)
            scores = []
            for beta in beta_values:
                curve = self._centroid_mix_net_curve(
                    T,
                    V_all,
                    neighbors,
                    beta,
                    num_classes,
                )
                scores.append(curve[alpha_idx])
            if not scores:
                return 0.0
            return float(torch.stack(scores).max().item())

        k = shots_per_class
        class_features = torch.stack([
            train_features[class_indices[class_idx][:k]].to(self.device)
            for class_idx in range(num_classes)
        ])
        targets = torch.arange(num_classes, device=self.device)
        score = torch.zeros((), device=self.device)

        for hold_idx in range(k):
            held = F.normalize(class_features[:, hold_idx, :], dim=-1)
            keep = torch.arange(k, device=self.device) != hold_idx
            fold_centroids = torch.stack([
                self._visual_centroid(class_features[class_idx, keep])
                for class_idx in range(num_classes)
            ])
            if query_centroids is not None:
                fold_centroids = self.expand_visual_centroids(
                    fold_centroids,
                    query_centroids,
                )
            prototypes = F.normalize(
                (1.0 - float(alpha)) * T + float(alpha) * fold_centroids,
                dim=-1,
            )
            logits = held @ prototypes.T
            fused_correct = logits.argmax(dim=-1).eq(targets)
            score += fused_correct.sum().float()
        return float(score.item())

    def select_fallback_candidate(
        self,
        T,
        V,
        expanded_V,
        alpha_init,
        alpha_final,
        train_features,
        train_labels,
        num_classes,
        query_centroids,
    ):
        candidates = [
            ("orig", V, alpha_init, None),
            ("qx_fixed_alpha", expanded_V, alpha_init, query_centroids),
            ("qx_recal_alpha", expanded_V, alpha_final, query_centroids),
        ]
        scored = []
        for name, centroids, alpha, fold_query_centroids in candidates:
            score = self.support_calibration_score(
                T,
                centroids,
                alpha,
                train_features,
                train_labels,
                num_classes,
                query_centroids=fold_query_centroids,
            )
            scored.append((name, centroids, float(alpha), score))
        return max(scored, key=lambda candidate: candidate[3]), scored

    def logits_from_features(self, features):
        features = F.normalize(features.to(self.device).float(), dim=-1)
        return features @ self.fused_prototypes.T

    def hopc_alpha(
        self,
        T,
        V_all,
        train_features,
        train_labels,
        num_classes,
        query_centroids=None,
    ):
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
                    self._visual_centroid(
                        class_feat[c, torch.arange(k, device=self.device) != hold_idx]
                    )
                    for c in range(num_classes)
                ])
                if query_centroids is not None:
                    V_minus = self.expand_visual_centroids(
                        V_minus,
                        query_centroids,
                    )

                refined = F.normalize(
                    (1 - self.alphas).view(-1, 1, 1) * T + self.alphas.view(-1, 1, 1) * V_minus,
                    dim=-1
                )
                fused_preds = torch.einsum("cd,akd->ack", held, refined).argmax(dim=-1)
                fused_correct = fused_preds.eq(targets.view(1, -1))

                net_scores += fused_correct.sum(dim=1).float()

            best_alpha = self.alphas[net_scores.argmax()].item()
            best_proto = F.normalize((1 - best_alpha) * T + best_alpha * V_all, dim=-1)

        return best_proto, best_alpha

    def fuse_and_evaluate(self, train_features, train_labels, eval_features, eval_labels, num_classes):
        V = self.build_visual_centroids(train_features, train_labels, num_classes)
        T = self.text_prototypes
        class_counts = torch.bincount(
            train_labels.to(self.device).long(),
            minlength=num_classes,
        )
        is_one_shot = bool(class_counts.numel() == num_classes and class_counts.eq(1).all().item())

        _, alpha_init = self.hopc_alpha(T, V, train_features, train_labels, num_classes)
        proto_before = F.normalize((1.0 - alpha_init) * T + alpha_init * V, dim=-1)
        
        import math
        kshot = int(class_counts.min().item())
        if kshot < 1:
            kshot = 1
        beta_val = min(0.45, 0.30 / math.sqrt(kshot))
        rho = min(1.0, 0.50 / math.sqrt(kshot))
        self.rho = rho
        
        Q_adversarial = self._generate_sqs_adversarial(V, proto_before, beta_val)
        query_centroids = self.pseudo_label_aggregation(
            Q_adversarial.view(-1, Q_adversarial.shape[-1]),
            T,
            V,
            alpha_init,
        )
        selected_V = F.normalize((1.0 - rho) * V + rho * query_centroids, dim=-1)
        expanded_V = selected_V
        alpha = alpha_init
        alpha_final = alpha_init
        selected_candidate = "sqs_adversarial_fixed"
        candidate_scores = [("sqs_adversarial_fixed", selected_V, alpha, 0.0)]
        
        proto = F.normalize((1.0 - alpha) * T + alpha * selected_V, dim=-1)
        self.fused_prototypes = proto
        self.best_alpha = alpha
        self.alpha_init = alpha_init
        self.alpha_final = alpha_init
        self.selected_candidate = selected_candidate
        self.candidate_scores = {"sqs_adversarial_fixed": 0.0}
        self.support_visual_centroids = V
        self.query_centroids = query_centroids
        self.expanded_visual_centroids = selected_V
        shots_per_class = int(torch.bincount(
            train_labels.to(self.device).long(),
            minlength=num_classes,
        ).min().item())
        logged_shots = getattr(self, '_transductive_logged_shots', set())
        if shots_per_class not in logged_shots:
            centroid_shift = torch.linalg.vector_norm(expanded_V - V, dim=-1).mean().item()
            score_text = ",".join(
                f"{name}:{score:.3f}" for name, _, _, score in candidate_scores
            )
            logged_shots.add(shots_per_class)
            self._transductive_logged_shots = logged_shots

        logits = self.logits_from_features(eval_features)
        preds = logits.argmax(dim=-1).cpu().tolist()

        labels_list = eval_labels.tolist()

        metrics = compute_metrics(labels_list, preds)
        metrics['alpha'] = alpha
        metrics['alpha_init'] = alpha_init
        metrics['alpha_final'] = alpha_final
        metrics['rho'] = self.rho
        metrics['selected_candidate'] = selected_candidate
        metrics['candidate_scores'] = self.candidate_scores
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
                logits = self.logits_from_features(features)
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
            'alpha_init': getattr(self, 'alpha_init', None),
            'alpha_final': getattr(self, 'alpha_final', None),
            'selected_candidate': getattr(self, 'selected_candidate', None),
            'candidate_scores': getattr(self, 'candidate_scores', None),
            'rho': self.rho,
            'support_visual_centroids': getattr(self, 'support_visual_centroids', None),
            'query_centroids': getattr(self, 'query_centroids', None),
            'expanded_visual_centroids': getattr(self, 'expanded_visual_centroids', None),
            'alpha_steps': self.alpha_steps,
            'classnames': self.classnames,
        }, path)
        # logger.info(f"ProtoFuse prototypes saved to {path}")

    def load_model(self, path):
        data = torch.load(path, map_location=self.device)
        self.fused_prototypes = data['fused_prototypes']
        self.text_prototypes = data['text_prototypes']
        self.best_alpha = data['best_alpha']
        self.alpha_init = data.get('alpha_init')
        self.alpha_final = data.get('alpha_final')
        self.selected_candidate = data.get('selected_candidate')
        self.candidate_scores = data.get('candidate_scores')
        self.rho = data.get('rho', getattr(self, 'rho', 0.5))
        self.support_visual_centroids = data.get('support_visual_centroids')
        self.query_centroids = data.get('query_centroids')
        self.expanded_visual_centroids = data.get('expanded_visual_centroids')
        # logger.info(f"ProtoFuse prototypes loaded from {path}")
