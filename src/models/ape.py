import os
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


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * (1.0 - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()


class APETrainingModule(nn.Module):
    def __init__(self, feat_num, cate_num, shots, indices):
        super().__init__()
        self.shots = shots
        self.feat_num = feat_num
        self.cate_num = cate_num
        self.indices = indices
        self.value_weights = nn.Parameter(torch.ones(cate_num * shots, 1).half(), requires_grad=True)
        self.res = nn.Parameter(torch.zeros(cate_num, feat_num).half(), requires_grad=True)

    def forward(self, cache_keys, clip_weights, cache_values):
        feat_dim = clip_weights.shape[0]
        res_keys = self.res.unsqueeze(1).repeat(1, self.shots, 1).reshape(-1, self.feat_num)
        new_cache_keys = cache_keys.clone().reshape(-1, feat_dim)
        new_cache_keys[:, self.indices] = new_cache_keys[:, self.indices] + res_keys

        res_text = self.res.t()
        new_clip_weights = clip_weights.clone()
        new_clip_weights[self.indices, :] = clip_weights[self.indices, :] + res_text
        new_cache_values = cache_values * self.value_weights
        return new_cache_keys.half(), new_clip_weights.half(), new_cache_values.half()


def _cal_criterion_vectorized(cfg, clip_weights, cache_keys, only_use_txt, training_free, cache_path):
    feat_dim, cate_num = clip_weights.shape
    text_feat = clip_weights.t().unsqueeze(1)

    if os.path.exists(cache_path):
        # logger.info("Loading APE criterion from cache...")
        sim = torch.load(cache_path, weights_only=False)
        if sim.device != clip_weights.device:
            sim = sim.to(clip_weights.device)
    elif only_use_txt:
        # logger.info("Computing APE criterion (text-only, vectorized)...")
        feats = text_feat.squeeze(1)
        S = feats.sum(dim=0)
        sim = S * S - (feats * feats).sum(dim=0)
        count = cate_num * (cate_num - 1)
        sim = sim / count
        torch.save(sim.cpu(), cache_path)
    else:
        # logger.info("Computing APE criterion (text+image, vectorized)...")
        shots = cfg.get('shots', 1)
        cache_feat = cache_keys.reshape(cate_num, shots, feat_dim)
        feats = torch.cat([text_feat, cache_feat], dim=1)
        samp_num = feats.shape[1]
        S = feats.reshape(-1, feat_dim).sum(dim=0)
        S_i = feats.reshape(cate_num, samp_num, feat_dim).sum(dim=1)
        sim = S * S - (S_i * S_i).sum(dim=0)
        count = cate_num * (cate_num - 1) * samp_num * samp_num
        sim = sim / count
        torch.save(sim.cpu(), cache_path)

    w = cfg.get('w', cfg.get('w_training_free', [0.5, 0.5]))
    criterion = (-1) * w[0] * sim + w[1] * torch.var(clip_weights, dim=1)

    feat_num_key = 'training_free_feat_num' if training_free else 'training_feat_num'
    k = cfg.get(feat_num_key, 800)
    feat_dim_actual = clip_weights.shape[0]
    ratio = 1024 / feat_dim_actual
    k = int(k // ratio)
    k = max(1, min(k, feat_dim_actual))
    _, indices = torch.topk(criterion, k=k)
    return indices


class APE(BaseTrainer):
    DEFAULT_LR = 0.001

    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')

        data_cfg = self.cfg.get('data', ConfigNode())
        dataset_name = data_cfg.get('dataset_name', 'ImageNet')
        self.template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.") 

        self.shots = self._cfg_int(1, 'data.kshot')
        self.init_alpha = self._cfg_float(1.0, 'model.init_alpha')
        self.init_beta = self._cfg_float(1.0, 'model.init_beta')
        self.init_gamma = self._cfg_float(0.1, 'model.init_gamma')

        search_cfg = self.cfg.get('model', ConfigNode()).get('search', ConfigNode())
        self.search_scale = list(search_cfg.get('scale', [7, 7, 1]))
        self.search_step = list(search_cfg.get('step', [200, 20, 20]))

        w_tf = list(self.cfg.get('model', ConfigNode()).get('w_training_free', [0.5, 0.5]))
        w_tr = list(self.cfg.get('model', ConfigNode()).get('w_training', [0.2, 0.8]))
        self.w_training_free = w_tf
        self.w_training = w_tr
        self.training_free_feat_num = self._cfg_int(800, 'model.training_free_feat_num')
        self.training_feat_num = self._cfg_int(900, 'model.training_feat_num')

        checkpoint_cfg = self.cfg.get('checkpoint', ConfigNode())
        self.cache_dir = checkpoint_cfg.get('cache_dir', 'checkpoints/ape')
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

        self.cache_keys = None
        self.cache_values = None
        self.ape_adapter = None
        self.model = nn.Module()
        self.initial_model_state = {}

        # logger.info(f"APE: {self.num_classes} classes, template=\"{self.template}\"")

    def setup_optimizer(self):
        self.optimizer = None
        self.scheduler = None

    def extract_features(self, dataloader, augment_epochs=1):
        all_keys = []
        all_values = []
        self.clip_model.eval()
        with torch.no_grad():
            for _ in range(augment_epochs):
                epoch_feats = []
                epoch_labels = []
                for images, labels in dataloader:
                    images = images.to(self.device)
                    feats = self.clip_model.encode_image(images).float()
                    epoch_feats.append(feats)
                    if len(all_values) == 0 and len(all_keys) == 0:
                        epoch_labels.append(labels)
                all_keys.append(torch.cat(epoch_feats, dim=0).unsqueeze(0))
                if epoch_labels:
                    all_values.extend(epoch_labels)

        cache_keys = torch.cat(all_keys, dim=0).mean(dim=0)
        cache_keys = F.normalize(cache_keys, dim=-1)
        cache_keys = cache_keys.t()
        cache_values = F.one_hot(torch.cat(all_values, dim=0).long(), num_classes=self.num_classes).half().to(self.device)
        return cache_keys, cache_values

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

    def build_cache(self, cache_keys, cache_values):
        self.cache_keys = cache_keys.to(self.device)
        self.cache_values = cache_values.to(self.device)

    def _get_criterion_cache_path(self, only_use_txt):
        suffix = 'txt' if only_use_txt else 'txtimg'
        backbone_safe = self._cfg_str('ViT-B/16', 'model.backbone').replace('/', '')
        return os.path.join(self.cache_dir, f'criterion_{backbone_safe}_{self.shots}shot_{suffix}.pt')

    def _build_cfg_dict(self, w_key):
        return {
            'shots': self.shots,
            'w': self.w_training_free if w_key == 'training_free' else self.w_training,
            'w_training_free': self.w_training_free,
            'w_training': self.w_training,
            'training_free_feat_num': self.training_free_feat_num,
            'training_feat_num': self.training_feat_num,
        }

    def evaluate_ape(self, val_features, val_labels, test_features, test_labels):
        clip_weights = self.clip_weights
        cache_keys = self.cache_keys
        cache_values = self.cache_values
        feat_dim, cate_num = clip_weights.shape

        reshaped_values = cache_values.reshape(cate_num, -1, cate_num).to(self.device)
        reshaped_keys = cache_keys.t().reshape(cate_num, self.shots, feat_dim).reshape(cate_num, -1, feat_dim).to(self.device)
        flat_keys = reshaped_keys.reshape(-1, feat_dim)
        flat_values = reshaped_values.reshape(-1, cate_num)

        cfg_dict = self._build_cfg_dict('training_free')
        cache_path = self._get_criterion_cache_path(only_use_txt=False)
        indices = _cal_criterion_vectorized(cfg_dict, clip_weights, flat_keys, only_use_txt=False,
                                            training_free=True, cache_path=cache_path)

        new_clip_weights = clip_weights[indices, :]
        new_cache_keys = flat_keys[:, indices]
        new_test_features = test_features.to(self.device)[:, indices]
        new_val_features = val_features.to(self.device)[:, indices]

        new_clip_weights = F.normalize(new_clip_weights, dim=0)
        new_cache_keys = F.normalize(new_cache_keys, dim=-1)
        new_test_features = F.normalize(new_test_features, dim=-1)
        new_val_features = F.normalize(new_val_features, dim=-1)

        key_logits = new_cache_keys @ new_clip_weights
        key_logits = key_logits.softmax(dim=1)
        cache_div = torch.sum(flat_values * torch.log2((flat_values + 1e-6) / (key_logits + 1e-6)), dim=1)[:, None]

        val_features_dev = val_features.to(self.device)
        test_features_dev = test_features.to(self.device)
        val_labels_dev = val_labels.to(self.device)
        test_labels_dev = test_labels.to(self.device)

        R_fF_val = new_val_features @ new_cache_keys.t()
        R_fW_val = 100.0 * val_features_dev @ clip_weights

        beta_list = [i * (self.search_scale[0] - 0.1) / self.search_step[0] + 0.1 for i in range(self.search_step[0])]
        alpha_list = [i * (self.search_scale[1] - 0.1) / self.search_step[1] + 0.1 for i in range(self.search_step[1])]
        gamma_list = [i * self.search_scale[2] / self.search_step[2] for i in range(self.search_step[2])]

        best_acc = 0.0
        best_alpha, best_beta, best_gamma = self.init_alpha, self.init_beta, self.init_gamma
        for beta in beta_list:
            for alpha in alpha_list:
                for gamma in gamma_list:
                    with torch.no_grad():
                        soft_vals = flat_values * (cache_div * gamma).exp()
                        cache_logits = ((-1) * (beta - beta * R_fF_val)).exp() @ soft_vals
                        logits = R_fW_val + cache_logits * alpha
                    acc = self._cls_acc(logits, val_labels_dev)
                    if acc > best_acc:
                        best_acc = acc
                        best_alpha, best_beta, best_gamma = alpha, beta, gamma

        # logger.info(f"APE val search best: alpha={best_alpha:.2f}, beta={best_beta:.2f}, gamma={best_gamma:.2f}, acc={best_acc:.2f}%")

        R_fF_test = new_test_features @ new_cache_keys.t()
        R_fW_test = 100.0 * test_features_dev @ clip_weights
        soft_vals = flat_values * (cache_div * best_gamma).exp()
        cache_logits = ((-1) * (best_beta - best_beta * R_fF_test)).exp() @ soft_vals
        logits = R_fW_test + cache_logits * best_alpha

        preds = logits.argmax(dim=-1).cpu().tolist()
        labels_list = test_labels.tolist()
        metrics = compute_metrics(labels_list, preds)
        metrics['alpha'] = best_alpha
        metrics['beta'] = best_beta
        metrics['gamma'] = best_gamma
        # logger.info(f"APE test accuracy: {metrics.get('accuracy', 0.0):.2f}%")
        return metrics

    def train_ape_t(self, train_loader, val_features, val_labels, test_features, test_labels,
                    train_features=None, train_labels=None):
        finetune_cfg = self.cfg.get('model', ConfigNode()).get('finetune', ConfigNode())
        epochs = self._cfg_int(30, 'model.finetune.epochs')
        lr = self._cfg_float(self.DEFAULT_LR, 'model.finetune.lr')
        eps = self._cfg_float(1e-3, 'model.finetune.eps')

        clip_weights = self.clip_weights
        cache_keys = self.cache_keys
        cache_values = self.cache_values
        feat_dim, cate_num = clip_weights.shape

        reshaped_values = cache_values.reshape(cate_num, -1, cate_num)
        reshaped_keys = cache_keys.t().reshape(cate_num, self.shots, feat_dim).reshape(cate_num, -1, feat_dim)
        flat_keys = reshaped_keys.reshape(-1, feat_dim)
        flat_values = reshaped_values.reshape(-1, cate_num)

        cfg_dict = self._build_cfg_dict('training')
        cfg_dict['w'] = self.w_training
        cache_path = self._get_criterion_cache_path(only_use_txt=False)
        indices = _cal_criterion_vectorized(cfg_dict, clip_weights, flat_keys, only_use_txt=False,
                                            training_free=False, cache_path=cache_path)

        adapter = APETrainingModule(
            feat_num=len(indices),
            cate_num=cate_num,
            shots=self.shots,
            indices=indices,
        ).to(self.device)

        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, eps=eps, weight_decay=1e-1)
        if train_features is not None:
            batch_size = self._cfg_int(128, 'training.batch_size')
            steps_per_epoch = max(1, math.ceil(train_features.shape[0] / batch_size))
        else:
            batch_size = None
            steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps_per_epoch)
        loss_fn = SmoothCrossEntropy()

        beta, alpha = self.init_beta, self.init_alpha
        best_acc = 0.0
        best_adapter_path = os.path.join(self.cache_dir, f'ape_t_{self.shots}shots.pt')

        for epoch in range(epochs):
            adapter.train()
            if train_features is not None:
                order = torch.randperm(train_features.shape[0])
                train_iter = []
                for start in range(0, train_features.shape[0], batch_size):
                    batch_idx = order[start:start + batch_size]
                    img_feats = F.normalize(train_features[batch_idx].to(self.device).float(), dim=-1)
                    target = train_labels[batch_idx].to(self.device).long()
                    train_iter.append((img_feats, target))
            else:
                train_iter = []
                for images, target in train_loader:
                    images, target = images.to(self.device), target.to(self.device).long()
                    with torch.no_grad():
                        img_feats = self.clip_model.encode_image(images).float()
                        img_feats = F.normalize(img_feats, dim=-1)
                    train_iter.append((img_feats, target))

            for img_feats, target in train_iter:

                new_cache_keys, new_clip_weights, R_FW = adapter(flat_keys, clip_weights, flat_values)
                R_fF = img_feats @ new_cache_keys.half().t()
                cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
                R_fW = 100.0 * img_feats @ new_clip_weights
                logits = R_fW + cache_logits * alpha
                loss = loss_fn(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            adapter.eval()
            with torch.no_grad():
                new_cache_keys, new_clip_weights, R_FW = adapter(flat_keys, clip_weights, flat_values)
                R_fF = val_features.to(self.device) @ new_cache_keys.half().t()
                cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
                R_fW = 100.0 * val_features.to(self.device) @ new_clip_weights
                logits = R_fW + cache_logits * alpha
            acc = self._cls_acc(logits, val_labels.to(self.device))
            # logger.info(f"APE-T Epoch {epoch+1}/{epochs} val_acc={acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(adapter.state_dict(), best_adapter_path)

        adapter.load_state_dict(torch.load(best_adapter_path, weights_only=True, map_location=self.device))
        # logger.info(f"APE-T best val acc: {best_acc:.2f}%")

        search_scale = self.search_scale[:2]
        search_step = self.search_step[:2]
        beta_list = [i * (search_scale[0] - 0.1) / search_step[0] + 0.1 for i in range(search_step[0])]
        alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]

        best_search_acc, best_alpha, best_beta = 0.0, alpha, beta
        adapter.eval()
        for beta_s in beta_list:
            for alpha_s in alpha_list:
                with torch.no_grad():
                    new_cache_keys, new_clip_weights, R_FW = adapter(flat_keys, clip_weights, flat_values)
                    R_fF = val_features.to(self.device) @ new_cache_keys.half().t()
                    cache_logits = ((-1) * (beta_s - beta_s * R_fF)).exp() @ R_FW
                    R_fW = 100.0 * val_features.to(self.device) @ new_clip_weights
                    logits = R_fW + cache_logits * alpha_s
                acc = self._cls_acc(logits, val_labels.to(self.device))
                if acc > best_search_acc:
                    best_search_acc = acc
                    best_alpha, best_beta = alpha_s, beta_s

        # logger.info(f"APE-T search best: alpha={best_alpha:.2f}, beta={best_beta:.2f}, val_acc={best_search_acc:.2f}%")

        with torch.no_grad():
            new_cache_keys, new_clip_weights, R_FW = adapter(flat_keys, clip_weights, flat_values)
            R_fF = test_features.to(self.device) @ new_cache_keys.half().t()
            cache_logits = ((-1) * (best_beta - best_beta * R_fF)).exp() @ R_FW
            R_fW = 100.0 * test_features.to(self.device) @ new_clip_weights
            logits = R_fW + cache_logits * best_alpha

        preds = logits.argmax(dim=-1).cpu().tolist()
        labels_list = test_labels.tolist()
        metrics = compute_metrics(labels_list, preds)
        metrics['alpha'] = best_alpha
        metrics['beta'] = best_beta
        # logger.info(f"APE-T test accuracy: {metrics.get('accuracy', 0.0):.2f}%")
        self.ape_adapter = adapter
        return metrics

    def _cls_acc(self, logits, labels):
        preds = logits.topk(1, dim=1)[1].squeeze(1)
        correct = preds.eq(labels).float().sum().item()
        return 100.0 * correct / labels.shape[0]

    def train_step(self, batch):
        raise NotImplementedError("APE is cache-based; use the pipeline.")

    def evaluate(self, dataloader):
        raise NotImplementedError("APE evaluation requires pre-extracted features; use the pipeline.")

    def save_model(self, path):
        torch.save({
            'cache_keys': self.cache_keys.cpu() if self.cache_keys is not None else None,
            'cache_values': self.cache_values.cpu() if self.cache_values is not None else None,
            'clip_weights': self.clip_weights.cpu(),
            'classnames': self.classnames,
            'shots': self.shots,
            'cfg': self.cfg,
        }, path)
        # logger.info(f"APE state saved to {path}")

    def load_model(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.cache_keys = data['cache_keys'].to(self.device) if data.get('cache_keys') is not None else None
        self.cache_values = data['cache_values'].to(self.device) if data.get('cache_values') is not None else None
        self.clip_weights = data['clip_weights'].to(self.device)
        # logger.info(f"APE state loaded from {path}")
