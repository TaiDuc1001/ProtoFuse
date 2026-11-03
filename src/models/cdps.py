import os
from collections import deque
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
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


def get_mask_bfs(
    attn_weights: torch.Tensor,
    num_patches: int = 14,
    tgt_area: int = 2,
    top_percent: float = 0.4,
    topk: int = 20,
) -> torch.Tensor:
    """Compute foreground connected-component binary masks from CLS attention weights.

    Args:
        attn_weights: [B, num_patches * num_patches] attention from CLS token to patches
        num_patches:  grid size (14 for ViT-B/16 at 224x224)
        tgt_area:     minimum number of connected components to include
        top_percent:  fraction of highest attention patches to consider
        topk:         minimum number of foreground patches to keep

    Returns:
        mask: [B, num_patches, num_patches] binary float mask
    """
    total_patches = num_patches * num_patches
    masks = []
    attn_weights = attn_weights.detach().cpu()

    for weight in attn_weights:
        labels = torch.zeros(total_patches, dtype=torch.float32)
        num_selected = max(1, int(total_patches * top_percent))
        top_indices = torch.topk(weight.reshape(-1), k=num_selected, sorted=False).indices
        labels[top_indices] = 1.0
        labels = labels.reshape(num_patches, num_patches)

        # BFS Connected Components
        visited = torch.zeros_like(labels)
        record_area = []
        record_sequence = []

        for i in range(num_patches):
            for j in range(num_patches):
                if labels[i, j] == 1.0 and visited[i, j] == 0.0:
                    queue = deque([(i, j)])
                    visited[i, j] = 1.0
                    sequence = [(i, j)]
                    area = 1
                    while queue:
                        x, y = queue.popleft()
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < num_patches
                                and 0 <= ny < num_patches
                                and labels[nx, ny] == 1.0
                                and visited[nx, ny] == 0.0
                            ):
                                visited[nx, ny] = 1.0
                                queue.append((nx, ny))
                                sequence.append((nx, ny))
                                area += 1
                    record_area.append(area)
                    record_sequence.append(sequence)

        mask = torch.zeros_like(labels)
        if record_area:
            record_area_tensor = torch.tensor(record_area, dtype=torch.long)
            num_area = 0
            while (num_area < tgt_area or mask.sum() < topk) and record_area_tensor.max() >= 0:
                num_area += 1
                max_id = torch.argmax(record_area_tensor).item()
                for x, y in record_sequence[max_id]:
                    mask[x, y] = 1.0
                record_area_tensor[max_id] = -1

        # Fallback if mask is too small: activate topk patches directly
        if mask.sum() < topk:
            top_indices = torch.topk(weight.reshape(-1), k=min(topk, total_patches), sorted=False).indices
            mask = mask.reshape(-1)
            mask[top_indices] = 1.0
            mask = mask.reshape(num_patches, num_patches)

        masks.append(mask)

    return torch.stack(masks, dim=0)


def filter_masked_patches(
    patch_features: torch.Tensor,
    masks: torch.Tensor,
) -> List[torch.Tensor]:
    """Extract foreground patch vectors using binary masks.

    Args:
        patch_features: [B, P, D] normalized patch tokens
        masks:          [B, 14, 14] binary masks

    Returns:
        List of [N_i, D] tensors where N_i is the number of active foreground patches
    """
    filtered = []
    masks_flat = masks.reshape(masks.shape[0], -1)
    for feat, mask in zip(patch_features, masks_flat):
        active_indices = torch.where(mask == 1.0)[0]
        if len(active_indices) == 0:
            active_indices = torch.arange(feat.shape[0], device=feat.device)
        filtered.append(feat[active_indices].detach().clone())
    return filtered


class ViTPatchHookExtractor:
    """Hooks the final Vision Transformer block of CLIP to extract patch tokens and attention weights."""

    def __init__(self, clip_model):
        self.model = clip_model
        if not hasattr(clip_model, "visual") or not hasattr(clip_model.visual, "transformer"):
            raise ValueError("CDPS requires a Vision Transformer (ViT) CLIP backbone.")
        self.last_block = clip_model.visual.transformer.resblocks[-1]
        self.orig_attention = self.last_block.attention
        self.captured_attn = None
        self.captured_tokens = None
        self.hook_handle = None

    def _custom_attention(self, x: torch.Tensor):
        attn_mask = self.last_block.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
        attn_output, attn_weights = self.last_block.attn(
            x, x, x, need_weights=True, attn_mask=attn_mask
        )
        # attn_weights: [B, L, L]; token 0 is [CLS], tokens 1: are spatial patches
        self.captured_attn = attn_weights[:, 0, 1:].detach()
        return attn_output

    def _block_hook(self, module, inp, output):
        # output is [L, B, D]
        self.captured_tokens = output.detach()

    def __enter__(self):
        self.last_block.attention = self._custom_attention
        self.hook_handle = self.last_block.register_forward_hook(self._block_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.last_block.attention = self.orig_attention
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def extract_vit_features_and_masks(
    clip_model,
    dataloader,
    device,
    tgt_area: int = 2,
    top_percent: float = 0.4,
    topk: int = 20,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Extract foreground patch features, global features, and labels using CLIP ViT forward hooks."""
    clip_model.eval()
    num_patches = 14
    all_masked_patches = []
    all_global_feats = []
    all_labels = []

    with ViTPatchHookExtractor(clip_model) as extractor:
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")

                images = images.to(device)
                global_feat = clip_model.encode_image(images).float()
                global_feat = F.normalize(global_feat, dim=-1)

                # Tokens from hooked block: [L, B, D] -> [B, L, D]
                tokens = extractor.captured_tokens.permute(1, 0, 2)
                patch_tokens = tokens[:, 1:, :]  # exclude CLS token: [B, 196, D]
                patch_tokens = clip_model.visual.ln_post(patch_tokens)
                if clip_model.visual.proj is not None:
                    patch_tokens = patch_tokens @ clip_model.visual.proj
                patch_tokens = F.normalize(patch_tokens.float(), dim=-1)

                # Attention weights: [B, 196]
                attn_weights = extractor.captured_attn
                masks = get_mask_bfs(
                    attn_weights,
                    num_patches=num_patches,
                    tgt_area=tgt_area,
                    top_percent=top_percent,
                    topk=topk,
                ).to(device)

                masked_patches = filter_masked_patches(patch_tokens, masks)
                all_masked_patches.extend([p.cpu() for p in masked_patches])
                all_global_feats.append(global_feat.cpu())
                all_labels.append(labels.cpu())

    return all_masked_patches, torch.cat(all_global_feats, dim=0), torch.cat(all_labels, dim=0)


def construct_cdps_feats(
    masked_feats: List[torch.Tensor],
    global_feats: torch.Tensor,
    num_classes: int,
    labels: torch.Tensor,
    omega: float = 0.1,
    num_confuse: int = 5,
    topk: int = 20,
    class_indices: Optional[List[np.ndarray]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Mine top-k discriminative patches for each class.

    Returns:
        cdps_feats: [C, K_shot * topk, D] tensor of mined discriminative patches
    """
    labels_np = labels.cpu().numpy()
    if class_indices is None:
        class_indices = [np.flatnonzero(labels_np == cls) for cls in range(num_classes)]

    # 1. Identify confusing classes for each class using global feature similarity
    confuse_ids = []
    global_feats_work = global_feats.float()
    for cls in range(num_classes):
        cls_inds = class_indices[cls]
        if len(cls_inds) == 0:
            confuse_ids.append(np.array([c for c in range(num_classes) if c != cls][:num_confuse]))
            continue

        cls_global = global_feats_work[cls_inds]
        other_cls_inds = np.where(labels_np != cls)[0]
        if len(other_cls_inds) == 0:
            confuse_ids.append(np.array([]))
            continue

        other_labels = labels_np[other_cls_inds]
        other_global = global_feats_work[other_cls_inds]
        similarity = cls_global @ other_global.t()
        sort_index = torch.argsort(similarity, dim=1, descending=True)
        ranked_labels = other_labels[sort_index.T.reshape(-1).cpu().numpy()]

        confuse_list = []
        seen = set()
        for cur_label in ranked_labels:
            cur_label = int(cur_label)
            if cur_label not in seen:
                seen.add(cur_label)
                confuse_list.append(cur_label)
                if len(confuse_list) >= num_confuse:
                    break
        confuse_ids.append(np.array(confuse_list))

    # 2. Score patches and select top-k per support image
    feats_list = []
    target_device = device or (masked_feats[0].device if masked_feats else torch.device("cpu"))

    for cls in range(num_classes):
        cls_inds = class_indices[cls]
        if len(cls_inds) == 0:
            feat_dim = masked_feats[0].shape[-1] if masked_feats else 512
            feats_list.append(torch.zeros(topk, feat_dim, device=target_device))
            continue

        concat_feats = []
        indexes = [0]
        for idx in cls_inds:
            c_feat = masked_feats[idx].to(target_device).float()
            concat_feats.append(c_feat)
            indexes.append(indexes[-1] + c_feat.shape[0])
        concat_feats = torch.cat(concat_feats, dim=0)

        # Intra-class patch similarity:
        sim = concat_feats @ concat_feats.t()
        scores = torch.zeros(sim.shape[0], device=target_device)
        for i in range(len(indexes) - 1):
            cur_sim = sim[:, indexes[i] : indexes[i + 1]]
            scores += cur_sim.max(dim=-1).values
        scores = scores / max(len(cls_inds), 1)

        # Inter-class patch similarity against confusing classes:
        confusing_classes = confuse_ids[cls]
        if len(confusing_classes) > 0:
            other_concat_feats = []
            other_indexes = [0]
            for other_cls in confusing_classes:
                for other_idx in class_indices[other_cls]:
                    c_feat = masked_feats[other_idx].to(target_device).float()
                    other_concat_feats.append(c_feat)
                    other_indexes.append(other_indexes[-1] + c_feat.shape[0])

            if len(other_concat_feats) > 0:
                other_concat_feats = torch.cat(other_concat_feats, dim=0)
                inter_sim = concat_feats @ other_concat_feats.t()
                inter_scores = torch.zeros(inter_sim.shape[0], device=target_device)
                for i in range(len(other_indexes) - 1):
                    cur_sim = inter_sim[:, other_indexes[i] : other_indexes[i + 1]]
                    inter_scores += cur_sim.max(dim=-1).values
                inter_scores = inter_scores / max(len(other_indexes) - 1, 1)

                denom = max(len(confusing_classes) - 1, 1)
                scores -= (omega / denom) * inter_scores

        # Top-k selection per support image:
        represent_feats = []
        for i in range(len(indexes) - 1):
            cur_feats = concat_feats[indexes[i] : indexes[i + 1]]
            cur_score = scores[indexes[i] : indexes[i + 1]]
            pick_k = min(topk, cur_feats.shape[0])
            pick_indexes = cur_score.argsort(descending=True)[:pick_k]
            selected = cur_feats[pick_indexes]
            # Pad if fewer than topk patches
            if selected.shape[0] < topk:
                pad_size = topk - selected.shape[0]
                selected = torch.cat([selected, selected[:pad_size]], dim=0)
            represent_feats.append(selected)

        feats_list.append(torch.cat(represent_feats, dim=0))

    return torch.stack(feats_list, dim=0)


def compute_patch_logits(
    masked_feats: List[torch.Tensor],
    cdps_feats: torch.Tensor,
    num_classes: int,
    device: torch.device,
    topk: int = 20,
    bsz: int = 32,
    is_train: bool = False,
    labels: Optional[torch.Tensor] = None,
    class_indices: Optional[List[np.ndarray]] = None,
) -> torch.Tensor:
    """Compute patch classification probabilities using max-match top-k pooling.

    Args:
        masked_feats:  List of query patch features [N_i, D]
        cdps_feats:    [C, K_shot * topk, D] discriminative patches
        num_classes:   number of classes C
        device:        execution device
        topk:          top-k pooling count
        bsz:           batch size
        is_train:      whether computing on training set (applies LOO self-exclusion)
        labels:        training labels (required if is_train=True)
        class_indices: cached indices per class

    Returns:
        probabilities: [N_query, C] normalized probabilities
    """
    if is_train:
        assert labels is not None
        if class_indices is None:
            labels_np = labels.cpu().numpy()
            class_indices = [np.flatnonzero(labels_np == cls) for cls in range(num_classes)]

    cdps_feats = cdps_feats.to(device).float()
    class_blocks = []
    for cls in range(0, num_classes, bsz):
        cls_end = min(cls + bsz, num_classes)
        sample_locations = None
        if is_train:
            sample_locations = {}
            for row_id, class_id in enumerate(range(cls, cls_end)):
                for col_id, sample_id in enumerate(class_indices[class_id]):
                    sample_locations[int(sample_id)] = (row_id, col_id)
        block_feats = cdps_feats[cls:cls_end]
        c_block, k_patches, d_dim = block_feats.shape
        block_feats_flat = block_feats.reshape(c_block * k_patches, d_dim)
        class_blocks.append((block_feats_flat, (c_block, k_patches), sample_locations))

    all_logits = []
    num_samples = len(masked_feats)

    for i in range(0, num_samples, bsz):
        batch_end = min(i + bsz, num_samples)
        concat_feats = []
        indexes = [0]
        for j in range(i, batch_end):
            feat = masked_feats[j].to(device).float()
            concat_feats.append(feat)
            indexes.append(indexes[-1] + feat.shape[0])

        concat_feats = torch.cat(concat_feats, dim=0)
        batch_logits = []

        for block_feats_flat, (c_block, k_patches), sample_locations in class_blocks:
            # Memory-efficient matrix multiplication:
            # [N_patches, D] @ [D, C_block * K_patches] -> [N_patches, C_block, K_patches]
            # Avoids allocating massive [N_patches, C_block, K_patches, D] intermediate tensor (~120GB -> ~200MB)
            sim = (concat_feats @ block_feats_flat.t()).view(concat_feats.shape[0], c_block, k_patches)

            if is_train:
                for c_iter, sample_idx in enumerate(range(i, batch_end)):
                    loc = sample_locations.get(sample_idx)
                    if loc is not None:
                        row_id, col_id = loc
                        sim[
                            indexes[c_iter] : indexes[c_iter + 1],
                            row_id,
                            col_id * topk : (col_id + 1) * topk,
                        ] = -1.0

            sim_list = []
            for idx in range(len(indexes) - 1):
                cur_sim = sim[indexes[idx] : indexes[idx + 1]]
                cur_sim = cur_sim.max(dim=-1).values
                cal_topk = min(topk, cur_sim.shape[0])
                cur_sim = torch.topk(cur_sim, cal_topk, dim=0).values.mean(dim=0)
                sim_list.append(cur_sim)

            batch_logits.append(torch.stack(sim_list, dim=0))

        all_logits.append(torch.cat(batch_logits, dim=1))

    logits = torch.cat(all_logits, dim=0)
    return F.softmax(logits, dim=-1)


def tune_lambda(
    patch_probs: torch.Tensor,
    text_probs: torch.Tensor,
    train_labels: torch.Tensor,
    lambda_list: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """Search lambda in [0, 1] to maximize training accuracy."""
    if lambda_list is None:
        lambda_list = [float(v) for v in np.arange(0.0, 1.05, 0.1)]

    best_lambda = 0.5
    best_acc = -1.0
    train_labels = train_labels.to(patch_probs.device)

    for lmbda in lambda_list:
        combined = lmbda * text_probs + (1.0 - lmbda) * patch_probs
        preds = combined.argmax(dim=-1)
        acc = (preds == train_labels).float().mean().item() * 100.0
        if acc > best_acc:
            best_acc = acc
            best_lambda = lmbda

    return best_lambda, best_acc


class CDPS(BaseTrainer):
    DEFAULT_LR = 0.0

    def build_model(self):
        backbone_name = self._cfg_str("ViT-B/16", "model.backbone", "backbone")
        if not backbone_name.startswith("ViT"):
            raise ValueError(
                f"CDPS relies on patch tokens from Vision Transformers. "
                f"Got backbone '{backbone_name}'. Please use ViT-B/16 or ViT-B/32."
            )

        data_cfg = self.cfg.get("data", ConfigNode())
        dataset_name = data_cfg.get("dataset_name", "ImageNet")
        self.template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")

        self.omega = self._cfg_float(0.1, "model.omega")
        self.num_confuse = self._cfg_int(5, "model.num_confuse")
        self.topk = self._cfg_int(20, "model.topk")
        self.top_percent = self._cfg_float(0.4, "model.top_percent")
        self.tgt_area = self._cfg_int(2, "model.tgt_area")
        self.batch_size = self._cfg_int(32, "training.batch_size", "model.batch_size")

        lambda_default = [round(float(v), 2) for v in np.arange(0.0, 1.05, 0.1)]
        self.lambda_list = self._cfg_float_list(lambda_default, "model.lambda_list")

        checkpoint_cfg = self.cfg.get("checkpoint", ConfigNode())
        self.cache_dir = checkpoint_cfg.get("cache_dir", "checkpoints/cdps")
        os.makedirs(self.cache_dir, exist_ok=True)

        clip_model = load_clip_to_cpu(backbone_name)
        precision = self._cfg_str("fp32", "training.precision", "precision")
        if precision in ["fp32", "amp"]:
            clip_model.float()

        self.clip_model = clip_model.to(self.device).eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        self.embed_dim = self.clip_model.text_projection.shape[1]
        self.num_classes = len(self.classnames)

        # 1-prompt generation using CUSTOM_TEMPLATES
        prompts = [self.template.format(c.replace("_", " ")) for c in self.classnames]
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)

        self.text_features = text_features
        self.clip_weights = text_features.t().contiguous()

        self.cdps_feats = None
        self.last_cdps_params = None

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
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _cfg_float_list(self, default, *paths):
        value = self._cfg_value(*paths, default=default)
        if value is None:
            value = default
        if isinstance(value, str):
            return [coerce_to_float(item.strip(), 0.0) for item in value.split(",") if item.strip()]
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

    def evaluate_cdps(
        self,
        train_masked_feats: List[torch.Tensor],
        train_global_feats: torch.Tensor,
        train_labels: torch.Tensor,
        test_masked_feats: List[torch.Tensor],
        test_global_feats: torch.Tensor,
        test_labels: torch.Tensor,
        clip_weights: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Execute full training-free CDPS discriminative patch mining and classification."""
        if clip_weights is None:
            clip_weights = self.clip_weights.to(self.device).float()
        else:
            clip_weights = clip_weights.to(self.device).float()

        labels_np = train_labels.cpu().numpy()
        class_indices = [np.flatnonzero(labels_np == cls) for cls in range(self.num_classes)]

        # 1. Mine discriminative patches from training support set
        cdps_feats = construct_cdps_feats(
            masked_feats=train_masked_feats,
            global_feats=train_global_feats,
            num_classes=self.num_classes,
            labels=train_labels,
            omega=self.omega,
            num_confuse=self.num_confuse,
            topk=self.topk,
            class_indices=class_indices,
            device=self.device,
        )
        self.cdps_feats = cdps_feats

        # 2. Compute LOO patch logits and zero-shot text probabilities on train set
        train_patch_probs = compute_patch_logits(
            masked_feats=train_masked_feats,
            cdps_feats=cdps_feats,
            num_classes=self.num_classes,
            device=self.device,
            topk=self.topk,
            bsz=self.batch_size,
            is_train=True,
            labels=train_labels,
            class_indices=class_indices,
        )

        train_global_dev = train_global_feats.to(self.device).float()
        train_text_probs = F.softmax(100.0 * (train_global_dev @ clip_weights), dim=-1)

        # 3. Grid search lambda on train split
        best_lambda, best_tune_acc = tune_lambda(
            train_patch_probs,
            train_text_probs,
            train_labels,
            lambda_list=self.lambda_list,
        )

        # 4. Evaluate on test split
        test_patch_probs = compute_patch_logits(
            masked_feats=test_masked_feats,
            cdps_feats=cdps_feats,
            num_classes=self.num_classes,
            device=self.device,
            topk=self.topk,
            bsz=self.batch_size,
            is_train=False,
        )

        test_global_dev = test_global_feats.to(self.device).float()
        test_text_probs = F.softmax(100.0 * (test_global_dev @ clip_weights), dim=-1)

        final_probs = best_lambda * test_text_probs + (1.0 - best_lambda) * test_patch_probs
        preds = final_probs.argmax(dim=-1).cpu().tolist()
        labels_list = test_labels.cpu().tolist()

        metrics = compute_metrics(labels_list, preds)
        metrics["method"] = "CDPS"
        metrics["best_lambda"] = float(best_lambda)
        metrics["val_acc"] = float(best_tune_acc)

        self.last_cdps_params = {
            "best_lambda": float(best_lambda),
            "val_acc": float(best_tune_acc),
        }

        return metrics

    def train_step(self, batch):
        raise NotImplementedError("CDPS is training-free; use the pipeline.")

    def evaluate(self, dataloader):
        raise NotImplementedError("CDPS evaluation requires pre-extracted patch features; use the pipeline.")

    def save_model(self, path):
        torch.save({
            "clip_weights": self.clip_weights.cpu(),
            "text_features": self.text_features.cpu(),
            "classnames": self.classnames,
            "last_cdps_params": self.last_cdps_params,
        }, path)

    def save_posthoc_protofuse(self, path):
        torch.save({
            "fused_prototypes": self.posthoc_fused_prototypes.detach().cpu()
            if self.posthoc_fused_prototypes is not None else None,
            "visual_centroids": self.posthoc_visual_centroids.detach().cpu()
            if self.posthoc_visual_centroids is not None else None,
            "centroid_mask": self.posthoc_centroid_mask.detach().cpu()
            if self.posthoc_centroid_mask is not None else None,
            "text_prototypes": self.clip_weights.t().detach().cpu(),
            "alpha": self.posthoc_alpha,
            "missing_classes": self.posthoc_missing_classes,
            "cdps_params": self.last_cdps_params,
            "classnames": self.classnames,
        }, path)

    def load_model(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.clip_weights = data["clip_weights"].to(self.device)
        self.text_features = data.get("text_features", self.clip_weights.t()).to(self.device)
        self.last_cdps_params = data.get("last_cdps_params")

