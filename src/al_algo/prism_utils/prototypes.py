from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

from utils import logger


def generate_pseudo_labels(
    models,
    weights: List[float],
    ulb_embs: List[torch.Tensor],
    ulb_indices: List[int],
    device: torch.device,
    threshold: float = 0.85,
    entropy_threshold: float = 0.5,
    method: str = "weighted_consensus",
    adaptive_threshold: bool = True,
    adaptive_beta: float = 0.95,
    num_classes: Optional[int] = None,
) -> Tuple[Dict[int, Tuple[int, float]], List[int]]:
    if method != "weighted_consensus":
        raise NotImplementedError(f"Pseudo-label method '{method}' not yet implemented")

    if not ulb_embs or len(ulb_embs[0]) == 0:
        return {}, []

    N = ulb_embs[0].size(0)
    batch_size = 512
    all_agg_probs = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_probs_list = []

        for m_idx, model in enumerate(models):
            emb = ulb_embs[m_idx][start:end].to(device)
            # Use prototypes if available via _temp_prototypes, else raw logits
            logits = model.classify_from_embeddings(
                emb, _get_proto_matrix(model, models, None)
            ) if hasattr(model, '_temp_prototypes') else model.get_logits_from_embeddings(emb)
            probs = F.softmax(logits, dim=-1).cpu()
            batch_probs_list.append(probs)

        max_classes = max(p.shape[-1] for p in batch_probs_list)
        agg = torch.zeros(end - start, max_classes)
        for m_idx, probs in enumerate(batch_probs_list):
            if probs.shape[-1] < max_classes:
                padded = torch.zeros(probs.shape[0], max_classes)
                padded[:, :probs.shape[-1]] = probs
                probs = padded
            agg += weights[m_idx] * probs

        all_agg_probs.append(agg)

    all_agg_probs = torch.cat(all_agg_probs, dim=0)

    if adaptive_threshold and num_classes is not None:
        pred_labels = all_agg_probs.argmax(dim=-1)
        class_counts = defaultdict(int)
        for lbl in pred_labels.tolist():
            class_counts[lbl] += 1
        max_count = max(class_counts.values()) if class_counts else 1
        class_thresholds = {}
        for c in range(num_classes):
            count_c = class_counts.get(c, 1)
            class_thresholds[c] = threshold * adaptive_beta * max_count / max(count_c, 1)
            class_thresholds[c] = min(class_thresholds[c], 0.99)
    else:
        class_thresholds = None

    max_probs, max_labels = all_agg_probs.max(dim=-1)

    log_probs = torch.log(all_agg_probs + 1e-10)
    entropies = -(all_agg_probs * log_probs).sum(dim=-1)

    pseudo_labels = {}
    conflicted_indices = []

    for i in range(N):
        idx = ulb_indices[i]
        conf = max_probs[i].item()
        label = max_labels[i].item()
        ent = entropies[i].item()

        thr = class_thresholds.get(label, threshold) if class_thresholds is not None else threshold

        if conf >= thr and ent <= entropy_threshold:
            pseudo_labels[idx] = (label, conf)
        elif ent > entropy_threshold:
            conflicted_indices.append(idx)

    logger.info(
        f"Pseudo-labels: {len(pseudo_labels)} accepted, "
        f"{len(conflicted_indices)} conflicted (priority candidates)"
    )

    all_predictions = {ulb_indices[i]: max_labels[i].item() for i in range(N)}
    return pseudo_labels, conflicted_indices, all_predictions


def compute_prototypes(
    models,
    labeled_indices: List[int],
    labeled_labels: Dict[int, int],
    pseudo_labeled: Dict[int, Tuple[int, float]],
    labeled_embs: List[torch.Tensor],
    ulb_embs: List[torch.Tensor],
    ulb_indices: List[int],
    device: torch.device,
) -> Tuple[Dict[int, Dict[int, torch.Tensor]], Dict[int, torch.Tensor]]:
    all_indices_by_class = defaultdict(list)
    for idx in labeled_indices:
        label = labeled_labels[idx]
        all_indices_by_class[label].append(idx)
    for idx, (label, _) in pseudo_labeled.items():
        all_indices_by_class[label].append(idx)

    all_unique = sorted(set(
        idx for indices in all_indices_by_class.values() for idx in indices
    ))

    if not all_unique:
        return {}, {}

    # We need to map `idx` to the pre-extracted embeddings.
    # We combine them into a single local lookup.
    emb_lookup = {m_idx: {} for m_idx in range(len(models))}
    
    # Process labeled
    if labeled_embs and len(labeled_embs[0]) > 0:
        for i, idx in enumerate(labeled_indices):
            for m_idx in range(len(models)):
                emb_lookup[m_idx][idx] = labeled_embs[m_idx][i]
                
    # Process unlabeled
    if ulb_embs and len(ulb_embs[0]) > 0:
        for i, idx in enumerate(ulb_indices):
            # Only store if actually needed (in all_unique) to save memory/time
            """ Wait, it's faster to just lookup when needed rather than building full lookup dict. """
            pass
            
    # Better mapping: list to dict for fast O(1) lookup
    labeled_idx_to_pos = {idx: pos for pos, idx in enumerate(labeled_indices)}
    ulb_idx_to_pos = {idx: pos for pos, idx in enumerate(ulb_indices)}

    prototypes = {}
    
    # Pre-allocate covariances tensor list
    class_embs_m0 = {}

    for m_idx in range(len(models)):
        prototypes[m_idx] = {}
        for cls, indices in all_indices_by_class.items():
            cls_embs_list = []
            for idx in indices:
                if idx in labeled_idx_to_pos:
                    cls_embs_list.append(labeled_embs[m_idx][labeled_idx_to_pos[idx]])
                elif idx in ulb_idx_to_pos:
                    cls_embs_list.append(ulb_embs[m_idx][ulb_idx_to_pos[idx]])
            
            if not cls_embs_list:
                continue
                
            cls_emb = torch.stack(cls_embs_list, dim=0)
            prototypes[m_idx][cls] = cls_emb.mean(dim=0)
            
            if m_idx == 0:
                class_embs_m0[cls] = cls_emb

    covariances = compute_class_covariances(class_embs_m0)

    return prototypes, covariances


def compute_class_covariances(
    class_embs: Dict[int, torch.Tensor],
    eps: float = 1e-4,
) -> Dict[int, torch.Tensor]:
    covariances = {}
    if not class_embs:
        return covariances
        
    first_cls = next(iter(class_embs.keys()))
    d = class_embs[first_cls].shape[-1]

    for cls, cls_emb in class_embs.items():
        if cls_emb.shape[0] < 2:
            covariances[cls] = torch.eye(d) * eps
            continue
            
        mean = cls_emb.mean(dim=0, keepdim=True)
        centered = cls_emb - mean
        cov = (centered.T @ centered) / (cls_emb.shape[0] - 1)
        cov = cov + eps * torch.eye(d)
        covariances[cls] = cov

    return covariances


def _get_proto_matrix(model, models, prototypes):
    return torch.zeros(1, model.embed_dim)
