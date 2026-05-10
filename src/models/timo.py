import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip

from utils import (
    logger,
    ConfigNode,
    BaseTrainer,
    load_clip_to_cpu,
    compute_metrics,
)

from src.models.apt import CUSTOM_TEMPLATES


def _gda(vecs, labels, clip_weights, val_features, val_labels, alpha_shift=False):
    num_classes = clip_weights.shape[1]
    device = clip_weights.device
    vecs = vecs.float().to(device)
    labels = labels.long().to(device)
    val_features = val_features.float().to(device)
    val_labels = val_labels.long().to(device)

    mus = torch.stack([vecs[labels == i].mean(dim=0) for i in range(num_classes)])
    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in range(num_classes)])
    n, d = center_vecs.shape
    cov = center_vecs.t().cov()
    cov_inv = d * torch.linalg.pinv((n - 1) * cov + cov.trace() * torch.eye(d, device=device))

    priors = torch.ones(num_classes, device=device) / num_classes
    W = torch.einsum('nd,dc->cn', mus, cov_inv)
    b = priors.log() - torch.einsum('nd,dc,nc->n', mus, cov_inv, mus) / 2.0

    best_acc, best_alpha = 0.0, 0.1
    for alpha in [10 ** i for i in range(-4, 5)]:
        if alpha_shift:
            val_logits = alpha * val_features @ clip_weights.float() + val_features @ W + b
        else:
            val_logits = 100.0 * val_features @ clip_weights.float() + alpha * (val_features @ W + b)
        acc = (val_logits.argmax(-1) == val_labels).float().mean().item() * 100.0
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # logger.info(f"TIMO GDA best_alpha={best_alpha}, best_val_acc={best_acc:.2f}%")
    return best_alpha, W, b, best_acc


def _image_guide_text(dataset_name, text_features, image_features, gamma=None):
    text_features = F.normalize(text_features, dim=-1)
    if image_features.dim() == 3:
        image_features = image_features.mean(dim=1)
    image_features = F.normalize(image_features, dim=-1).to(text_features.device, dtype=text_features.dtype)

    if gamma is None:
        dataset_key = dataset_name.lower()
        if dataset_key == "imagenet":
            gamma = 1
        elif dataset_key in {"oxford_flowers", "flowers102"}:
            gamma = 100
        else:
            gamma = 50

    matching_score = torch.einsum("cd,cpd->cp", image_features, text_features)
    weights = F.normalize(matching_score, dim=-1)
    weights = F.softmax(weights * gamma, dim=-1)
    guided_text = torch.einsum("cp,cpd->cd", weights, text_features)
    guided_text = F.normalize(guided_text, dim=-1)
    return guided_text, matching_score


def _image_guide_text_search(dataset_name, text_features_all, val_features, val_labels, image_weights):
    best_acc = -1.0
    best_gamma = 50
    best_guided = None
    best_matching = None
    val_features = val_features.float().to(text_features_all.device)
    val_labels = val_labels.long().to(text_features_all.device)

    for gamma in range(5, 101, 5):
        guided, matching = _image_guide_text(dataset_name, text_features_all, image_weights, gamma=gamma)
        logits = val_features @ guided.t()
        acc = (logits.argmax(-1) == val_labels).float().mean().item() * 100.0
        if acc > best_acc:
            best_acc = acc
            best_gamma = gamma
            best_guided = guided
            best_matching = matching

    # logger.info(f"TIMOS IGT best_gamma={best_gamma}, best_val_acc={best_acc:.2f}%")
    return best_guided, best_matching


def _vec_sort(text_features, matching_score):
    _, sorted_idx = torch.topk(matching_score, k=text_features.shape[1], dim=-1)
    sorted_text = torch.stack(
        [text_features[class_idx][sorted_idx[class_idx]].clone() for class_idx in range(text_features.shape[0])],
        dim=0,
    )
    sorted_weights = torch.gather(matching_score, 1, sorted_idx)
    return sorted_text, sorted_weights


class TIMO(BaseTrainer):
    DEFAULT_LR = 0.0

    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')

        data_cfg = self.cfg.get('data', ConfigNode())
        dataset_name = data_cfg.get('dataset_name', 'ImageNet')
        self.template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")

        self.shots = self._cfg_int(1, 'data.kshot')
        self.augment_epoch = self._cfg_int(1, 'model.augment_epoch')

        checkpoint_cfg = self.cfg.get('checkpoint', ConfigNode())
        self.cache_dir = checkpoint_cfg.get('cache_dir', 'checkpoints/timo')
        os.makedirs(self.cache_dir, exist_ok=True)

        # logger.info(f"Loading CLIP (backbone: {backbone_name})")
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
        self.clip_weights = text_features.t()
        self.text_features_all = text_features.unsqueeze(1)

        self.train_vecs = None
        self.train_labels = None
        self.model = nn.Module()
        self.initial_model_state = {}

        # logger.info(f"TIMO: {self.num_classes} classes, template=\"{self.template}\"")

    def setup_optimizer(self):
        self.optimizer = None
        self.scheduler = None

    def extract_train_features(self, dataloader):
        all_vecs = []
        all_labels = []
        self.clip_model.eval()
        with torch.no_grad():
            for _ in range(self.augment_epoch):
                for images, labels in dataloader:
                    images = images.to(self.device)
                    feats = self.clip_model.encode_image(images).float()
                    feats = F.normalize(feats, dim=-1)
                    all_vecs.append(feats.cpu())
                    all_labels.append(labels)
        self.train_vecs = torch.cat(all_vecs, dim=0)
        self.train_labels = torch.cat(all_labels, dim=0)

    def extract_val_test_features(self, dataloader):
        all_feats = []
        all_labels = []
        self.clip_model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                feats = self.clip_model.encode_image(images).float()
                feats = F.normalize(feats, dim=-1)
                all_feats.append(feats.cpu())
                all_labels.append(labels)
        return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

    def evaluate_timo(self, val_features, val_labels, test_features, test_labels):
        return self.evaluate_timo_variant(val_features, val_labels, test_features, test_labels, variant="TIMO")

    def evaluate_timos(self, val_features, val_labels, test_features, test_labels):
        return self.evaluate_timo_variant(val_features, val_labels, test_features, test_labels, variant="TIMOS")

    def evaluate_timo_variant(self, val_features, val_labels, test_features, test_labels, variant="TIMO"):
        if self.train_vecs is None:
            raise RuntimeError("Call extract_train_features before evaluate_timo.")

        train_vecs = self.train_vecs.float().to(self.device)
        train_labels = self.train_labels.long().to(self.device)
        val_features_dev = val_features.float().to(self.device)
        val_labels_dev = val_labels.long().to(self.device)
        test_features_dev = test_features.float().to(self.device)

        image_weights = torch.stack([
            train_vecs[train_labels == class_idx].mean(dim=0)
            for class_idx in range(self.num_classes)
        ])
        image_weights = F.normalize(image_weights, dim=-1)

        dataset_name = self.cfg.get('data', ConfigNode()).get('dataset_name', 'ImageNet')
        text_features_all = self.text_features_all.float().to(self.device)
        if variant.upper() == "TIMOS":
            clip_weights_igt, matching_score = _image_guide_text_search(
                dataset_name, text_features_all, val_features_dev, val_labels_dev, image_weights
            )
            grid_search = True
            n_quick_search = self._cfg_int(10, 'model.timos_quick_search')
            method_name = "TIMOS"
        else:
            clip_weights_igt, matching_score = _image_guide_text(
                dataset_name, text_features_all, image_weights
            )
            grid_search = False
            n_quick_search = -1
            method_name = "TIMO"

        clip_weights_igt = clip_weights_igt.t().contiguous()
        sorted_text, sorted_weights = _vec_sort(text_features_all, matching_score)
        prompt_num = sorted_text.shape[1]

        if grid_search:
            if n_quick_search > 0:
                beta_list = [int(t) for t in torch.linspace(1, prompt_num * 2, n_quick_search)]
            else:
                beta_list = range(1, prompt_num * 2)
        else:
            beta_list = [prompt_num]

        best_val_acc = -1.0
        best_alpha = 0.1
        best_beta = prompt_num
        best_weights = None

        for beta in beta_list:
            beta = beta + 1 if beta == 0 else beta
            sliced_text = sorted_text.repeat(1, 2, 1)[:, :beta, :]
            sliced_weights = sorted_weights.repeat(1, 2)[:, :beta]
            sliced_text = sliced_text * sliced_weights.unsqueeze(-1)
            text_vecs = sliced_text.reshape(self.num_classes * beta, -1)
            text_labels = torch.arange(self.num_classes, device=self.device).unsqueeze(1).repeat(1, beta).flatten()

            combined_vecs = torch.cat([text_vecs, train_vecs], dim=0)
            combined_labels = torch.cat([text_labels, train_labels], dim=0)

            alpha, W, b, val_acc = _gda(
                combined_vecs, combined_labels, clip_weights_igt, val_features_dev, val_labels_dev, alpha_shift=True
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_alpha = alpha
                best_beta = beta
                best_weights = (W.clone(), b.clone())

        W, b = best_weights
        test_features_dev = test_features.float().to(self.device)
        test_logits = best_alpha * test_features_dev @ clip_weights_igt.float() + (test_features_dev @ W + b)
        preds = test_logits.argmax(dim=-1).cpu().tolist()
        labels_list = test_labels.tolist()
        metrics = compute_metrics(labels_list, preds)
        metrics['method'] = method_name
        metrics['alpha'] = best_alpha
        metrics['beta'] = best_beta
        metrics['val_acc'] = best_val_acc
        # logger.info(f"{method_name} test accuracy: {metrics.get('accuracy', 0.0):.2f}%")
        return metrics

    def train_step(self, batch):
        raise NotImplementedError("TIMO is training-free; use the pipeline.")

    def evaluate(self, dataloader):
        raise NotImplementedError("TIMO evaluation requires pre-extracted features; use the pipeline.")

    def save_model(self, path):
        torch.save({
            'clip_weights': self.clip_weights.cpu(),
            'text_features_all': self.text_features_all.cpu(),
            'train_vecs': self.train_vecs,
            'train_labels': self.train_labels,
            'classnames': self.classnames,
            'shots': self.shots,
        }, path)
        # logger.info(f"TIMO state saved to {path}")

    def load_model(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.clip_weights = data['clip_weights'].to(self.device)
        self.text_features_all = data.get('text_features_all', self.clip_weights.t().unsqueeze(1)).to(self.device)
        self.train_vecs = data.get('train_vecs')
        self.train_labels = data.get('train_labels')
        # logger.info(f"TIMO state loaded from {path}")
