import os
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


def compute_class_means_with_text_prior(
    feat,
    label,
    num_classes,
    text_prototypes,
    W_clip=None,
    text_prior_tau=4.0,
    normalize_proto=False,
):
    """Class means of image features fused with CLIP text prototypes.

    mu_c = (N_c / (N_c + tau)) * mu_img_c + (tau / (N_c + tau)) * mu_txt_c

    Args:
        feat:             [N, D] features
        label:            [N] integer class labels aligned with feat
        num_classes:      number of classes C
        text_prototypes:  [C, D] CLIP text weights (row = class)
        W_clip:           Optional [D, K] projection matrix
        text_prior_tau:   text-prior strength tau (default: 4.0)
        normalize_proto:  whether to L2-normalize class means
    """
    device = feat.device
    dtype = feat.dtype
    dim_feature = feat.shape[1]

    if W_clip is not None:
        W_clip = W_clip.to(device=device, dtype=torch.float32)
        feat_work = feat.to(device=device, dtype=torch.float32) @ W_clip
        dim_work = W_clip.shape[1]
    else:
        feat_work = feat
        dim_work = dim_feature

    class_means_img = []
    class_sizes_list = []
    for c in range(num_classes):
        mask = (label == c)
        n_c = mask.sum().item()
        class_sizes_list.append(n_c)
        if n_c > 0:
            class_means_img.append(feat_work[mask].mean(dim=0))
        else:
            class_means_img.append(torch.zeros(dim_work, device=device, dtype=dtype))

    class_means_img = torch.stack(class_means_img, dim=0)  # [C, dim_work]
    class_sizes = torch.tensor(class_sizes_list, device=device, dtype=torch.float32)  # [C]
    total_samples = class_sizes.sum()

    text_prototypes = text_prototypes.to(device=device, dtype=dtype)
    if W_clip is not None:
        txt_class_means = (text_prototypes.float() @ W_clip.float()).to(device=device, dtype=dtype)
    else:
        txt_class_means = text_prototypes

    tau = torch.tensor(float(text_prior_tau), device=device, dtype=torch.float32)
    img_ratio = class_sizes / (class_sizes + tau + 1e-12)
    txt_ratio = tau / (class_sizes + tau + 1e-12)

    class_means = (
        img_ratio.unsqueeze(1) * class_means_img
        + txt_ratio.unsqueeze(1) * txt_class_means
    )

    if normalize_proto:
        class_means_img = F.normalize(class_means_img, dim=1)
        class_means = F.normalize(class_means, dim=1)
        txt_class_means = F.normalize(txt_class_means, dim=1)

    return class_means_img, class_sizes, total_samples, class_means, txt_class_means


def fld_center(feat, label, num_classes, text_prototypes, text_prior_tau=4.0):
    """Re-centered Fisher Linear Discriminant (FLD) projection matrix.

    Args:
        feat:             [N, D] support/cache features
        label:            [N] integer class labels
        num_classes:      number of classes C
        text_prototypes:  [C, D] CLIP text weights
        text_prior_tau:   text-prior strength tau (default: 4.0)

    Returns:
        W: [D, K_proj] FLD projection matrix where K_proj = min(D, C - 1)
    """
    device = feat.device
    dim_feature = feat.shape[1]
    mean_total = torch.mean(feat.float(), dim=0)

    _, class_sizes, _, class_means, _ = compute_class_means_with_text_prior(
        feat.float(),
        label,
        num_classes,
        text_prototypes.float(),
        text_prior_tau=text_prior_tau,
    )

    mean_diff = feat.float() - class_means[label]
    mean_diff_total = class_means - mean_total

    # Between-class scatter S_B from re-centered class means:
    S_B = (mean_diff_total.T @ (mean_diff_total * class_sizes.unsqueeze(1))).float()

    # Within-class scatter under equal-covariance hypothesis:
    num_samples = mean_diff.shape[0]
    df = max(num_samples - 1, 1)
    cov = mean_diff.T.cov()
    cov_trace = torch.clamp(cov.trace(), min=1e-6)

    S_W_inv = num_classes * dim_feature * torch.linalg.pinv(
        df * cov + cov_trace * torch.eye(dim_feature, device=device)
    )

    eig_vals, eig_vecs = torch.linalg.eig(S_W_inv @ S_B)
    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real

    sorted_indices = torch.argsort(eig_vals, descending=True)
    num_proj = min(dim_feature, max(num_classes - 1, 1))
    W = eig_vecs[:, sorted_indices[:num_proj]]
    return W


class TARP(BaseTrainer):
    DEFAULT_LR = 0.0

    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')

        data_cfg = self.cfg.get('data', ConfigNode())
        dataset_name = data_cfg.get('dataset_name', 'ImageNet')
        self.template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")

        self.text_prior_tau = self._cfg_float(4.0, 'model.text_prior_tau')
        self.augment_epoch = self._cfg_int(10, 'model.augment_epoch')
        self.extra_text_shots = self._cfg_bool(True, 'model.extra_text_shots')

        scale_default = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
        alpha_default = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.scale_list = self._cfg_float_list(scale_default, 'model.scale_list')
        self.alpha_list = [int(a) for a in self._cfg_float_list(alpha_default, 'model.alpha_list')]

        checkpoint_cfg = self.cfg.get('checkpoint', ConfigNode())
        self.cache_dir = checkpoint_cfg.get('cache_dir', 'checkpoints/tarp')
        os.makedirs(self.cache_dir, exist_ok=True)

        clip_model = load_clip_to_cpu(backbone_name)
        precision = self._cfg_str('fp32', 'training.precision', 'precision')
        if precision in ['fp32', 'amp']:
            clip_model.float()

        self.clip_model = clip_model.to(self.device).eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        self.embed_dim = self.clip_model.text_projection.shape[1]
        self.num_classes = len(self.classnames)

        prompts = [self.template.format(c.replace('_', ' ')) for c in self.classnames]
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)

        self.text_features = text_features
        self.clip_weights = text_features.t().contiguous()

        self.train_vecs = None
        self.train_labels = None
        self.last_tarp_params = None

        self.posthoc_fused_prototypes = None
        self.posthoc_visual_centroids = None
        self.posthoc_centroid_mask = None
        self.posthoc_alpha = None
        self.posthoc_missing_classes = []

        self.model = nn.Module()
        self.initial_model_state = {}

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

    def clear_posthoc_protofuse(self):
        self.posthoc_fused_prototypes = None
        self.posthoc_visual_centroids = None
        self.posthoc_centroid_mask = None
        self.posthoc_alpha = None
        self.posthoc_missing_classes = []

    def get_text_prototypes(self):
        return self.clip_weights.t().detach().clone()

    def apply_posthoc_protofuse(self, alpha, fused_prototypes, visual_centroids=None, centroid_mask=None, missing_classes=None):
        self.posthoc_fused_prototypes = F.normalize(fused_prototypes.to(self.device).float(), dim=-1)
        self.posthoc_visual_centroids = (
            F.normalize(visual_centroids.to(self.device).float(), dim=-1)
            if visual_centroids is not None else None
        )
        self.posthoc_centroid_mask = centroid_mask.to(self.device).bool() if centroid_mask is not None else None
        self.posthoc_alpha = alpha
        self.posthoc_missing_classes = list(missing_classes or [])
        return self.posthoc_fused_prototypes.t().contiguous(), self.posthoc_fused_prototypes

    def build_augmented_cache(self, train_features, train_labels, text_prototypes=None):
        """Prepare support set with repeat and Text-as-Shot."""
        train_features = train_features.float().to(self.device)
        train_labels = train_labels.long().to(self.device)

        if self.augment_epoch > 1:
            aug_features = train_features.repeat((self.augment_epoch, 1))
            aug_labels = train_labels.repeat(self.augment_epoch)
        else:
            aug_features = train_features
            aug_labels = train_labels

        if self.extra_text_shots:
            if text_prototypes is None:
                text_prototypes = self.text_features.to(self.device).float()
            else:
                text_prototypes = text_prototypes.to(self.device).float()

            text_labels = torch.arange(self.num_classes, device=self.device, dtype=torch.long)
            aug_features = torch.cat([aug_features, text_prototypes], dim=0)
            aug_labels = torch.cat([aug_labels, text_labels], dim=0)

        return aug_features, aug_labels

    def compute_projection_and_prototypes(self, support_features, support_labels, text_prototypes=None):
        """Compute W_clip FLD matrix and normalized subspace class means."""
        if text_prototypes is None:
            text_prototypes = self.text_features.to(self.device).float()
        else:
            text_prototypes = text_prototypes.to(self.device).float()

        W_clip = fld_center(
            support_features,
            support_labels,
            self.num_classes,
            text_prototypes,
            text_prior_tau=self.text_prior_tau,
        )

        _, _, _, class_means, _ = compute_class_means_with_text_prior(
            support_features,
            support_labels,
            self.num_classes,
            text_prototypes,
            W_clip=W_clip,
            text_prior_tau=self.text_prior_tau,
        )
        class_means = F.normalize(class_means, dim=1)
        return W_clip, class_means

    def tune_hyperparameters(
        self,
        tune_features,
        tune_labels,
        W_clip,
        class_means,
        clip_weights=None,
    ):
        """Grid search over scale and (alpha1, alpha2) combinations."""
        tune_features = tune_features.float().to(self.device)
        tune_labels = tune_labels.long().to(self.device)
        if clip_weights is None:
            clip_weights = self.clip_weights.to(self.device).float()

        tune_fld = tune_features @ W_clip
        tune_dist = torch.cdist(tune_fld, class_means)
        tune_fld_logits = 1.0 / (tune_dist + 1e-6)
        tune_clip_logits = 100.0 * (tune_features @ clip_weights)

        best_acc = -1.0
        best_scale = self.scale_list[0] if self.scale_list else 1.0
        best_alpha1 = 1
        best_alpha2 = 1

        for scale in self.scale_list:
            for alpha1 in self.alpha_list:
                for alpha2 in self.alpha_list:
                    logits = tune_clip_logits + scale * (alpha2 / alpha1) * tune_fld_logits
                    preds = logits.argmax(dim=-1)
                    acc = (preds == tune_labels).float().mean().item() * 100.0
                    if acc >= best_acc:
                        best_acc = acc
                        best_scale = scale
                        best_alpha1 = alpha1
                        best_alpha2 = alpha2

        return best_scale, best_alpha1, best_alpha2, best_acc

    def evaluate_tarp(
        self,
        train_features,
        train_labels,
        test_features,
        test_labels,
        clip_weights=None,
        text_prototypes=None,
    ):
        """Full training-free TARP adaptation and evaluation."""
        if clip_weights is None:
            clip_weights = self.clip_weights.to(self.device).float()
        else:
            clip_weights = clip_weights.to(self.device).float()

        if text_prototypes is None:
            text_prototypes = self.text_features.to(self.device).float()
        else:
            text_prototypes = text_prototypes.to(self.device).float()

        self.train_vecs = train_features.cpu()
        self.train_labels = train_labels.cpu()

        # Build support set:
        support_features, support_labels = self.build_augmented_cache(
            train_features, train_labels, text_prototypes=text_prototypes
        )

        # Compute FLD projection and subspace class means:
        W_clip, class_means = self.compute_projection_and_prototypes(
            support_features, support_labels, text_prototypes=text_prototypes
        )

        # Tune hyperparameters on train features:
        best_scale, best_alpha1, best_alpha2, best_tune_acc = self.tune_hyperparameters(
            train_features, train_labels, W_clip, class_means, clip_weights=clip_weights
        )

        # Evaluate on test features:
        test_features_dev = test_features.float().to(self.device)
        test_labels_dev = test_labels.long().to(self.device)

        test_fld = test_features_dev @ W_clip
        test_dist = torch.cdist(test_fld, class_means)
        test_fld_logits = 1.0 / (test_dist + 1e-6)
        test_clip_logits = 100.0 * (test_features_dev @ clip_weights)

        gamma = best_scale * (best_alpha2 / best_alpha1)
        final_logits = test_clip_logits + gamma * test_fld_logits

        preds = final_logits.argmax(dim=-1).cpu().tolist()
        labels_list = test_labels_dev.cpu().tolist()

        metrics = compute_metrics(labels_list, preds)
        metrics['method'] = "TARP"
        metrics['scale'] = float(best_scale)
        metrics['alpha1'] = int(best_alpha1)
        metrics['alpha2'] = int(best_alpha2)
        metrics['gamma'] = float(gamma)
        metrics['val_acc'] = float(best_tune_acc)

        self.last_tarp_params = {
            'scale': float(best_scale),
            'alpha1': int(best_alpha1),
            'alpha2': int(best_alpha2),
            'gamma': float(gamma),
            'val_acc': float(best_tune_acc),
        }

        return metrics

    def train_step(self, batch):
        raise NotImplementedError("TARP is training-free; use the pipeline.")

    def evaluate(self, dataloader):
        raise NotImplementedError("TARP evaluation requires pre-extracted features; use the pipeline.")

    def save_model(self, path):
        torch.save({
            'clip_weights': self.clip_weights.cpu(),
            'text_features': self.text_features.cpu(),
            'train_vecs': self.train_vecs,
            'train_labels': self.train_labels,
            'classnames': self.classnames,
            'last_tarp_params': self.last_tarp_params,
        }, path)

    def save_posthoc_protofuse(self, path):
        torch.save({
            'fused_prototypes': self.posthoc_fused_prototypes.detach().cpu()
            if self.posthoc_fused_prototypes is not None else None,
            'visual_centroids': self.posthoc_visual_centroids.detach().cpu()
            if self.posthoc_visual_centroids is not None else None,
            'centroid_mask': self.posthoc_centroid_mask.detach().cpu()
            if self.posthoc_centroid_mask is not None else None,
            'text_prototypes': self.clip_weights.t().detach().cpu(),
            'alpha': self.posthoc_alpha,
            'missing_classes': self.posthoc_missing_classes,
            'tarp_params': self.last_tarp_params,
            'classnames': self.classnames,
        }, path)

    def load_model(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.clip_weights = data['clip_weights'].to(self.device)
        self.text_features = data.get('text_features', self.clip_weights.t()).to(self.device)
        self.train_vecs = data.get('train_vecs')
        self.train_labels = data.get('train_labels')
        self.last_tarp_params = data.get('last_tarp_params')

