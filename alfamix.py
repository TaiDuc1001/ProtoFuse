import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def compute_class_anchors(embeddings: Dict[int, torch.Tensor], labels: List[int]) -> Dict[int, torch.Tensor]:
    class_embeddings = defaultdict(list)
    for idx, label in zip(embeddings.keys(), labels):
        if idx in embeddings:
            class_embeddings[label].append(embeddings[idx])
    
    anchors = {}
    for class_id, emb_list in class_embeddings.items():
        if emb_list:
            stacked = torch.stack(emb_list)
            anchors[class_id] = torch.mean(stacked, dim=0)
    
    # print(f"[ALFAMIX DEBUG] Computed {len(anchors)} class anchors from {len(embeddings)} embeddings")
    # for cid, anc in anchors.items():
    #     print(f"  Class {cid}: anchor norm={torch.norm(anc).item():.4f}, has_nan={torch.isnan(anc).any().item()}")
    
    return anchors


def compute_closed_form_alpha(
    unlabeled_embeddings: Dict[int, torch.Tensor],
    class_anchors: Dict[int, torch.Tensor],
    classifier_weights: torch.Tensor,
    original_preds: Dict[int, int],
    alpha_cap: float = 0.5
) -> Dict[int, Tuple[float, int, int]]:
    results = {}
    num_classes = classifier_weights.shape[0]
    
    for idx, emb in unlabeled_embeddings.items():
        orig_pred = original_preds[idx]
        best_alpha = float('inf')
        best_target_class = -1
        
        for target_class in range(num_classes):
            if target_class == orig_pred or target_class not in class_anchors:
                continue
            
            anchor = class_anchors[target_class]
            diff = anchor - emb
            diff_norm_sq = torch.dot(diff, diff).item()
            
            if diff_norm_sq < 1e-10:
                continue
            
            w_orig = classifier_weights[orig_pred]
            w_target = classifier_weights[target_class]
            w_diff = w_target - w_orig
            
            numerator = torch.dot(w_diff, emb).item()
            denominator = torch.dot(w_diff, diff).item()
            
            if abs(denominator) < 1e-10:
                continue
            
            alpha = -numerator / denominator
            
            if 0 < alpha <= alpha_cap and alpha < best_alpha:
                best_alpha = alpha
                best_target_class = target_class
        
        if best_alpha <= alpha_cap:
            results[idx] = (best_alpha, orig_pred, best_target_class)
    
    return results


def find_candidate_samples(
    unlabeled_embeddings: Dict[int, torch.Tensor],
    labeled_embeddings: Dict[int, torch.Tensor],
    labeled_labels: List[int],
    predict_fn,
    alpha_cap: float = 0.5,
    device: torch.device = torch.device('cpu')
) -> Tuple[Dict[int, float], Dict[int, int]]:
    class_anchors = compute_class_anchors(labeled_embeddings, labeled_labels)
    
    if not class_anchors:
        print("[ALFAMIX DEBUG] No class anchors computed!")
        return {}, {}
    
    unlabeled_indices = list(unlabeled_embeddings.keys())
    if not unlabeled_indices:
        print("[ALFAMIX DEBUG] No unlabeled embeddings!")
        return {}, {}
    
    emb_matrix = torch.stack([unlabeled_embeddings[idx] for idx in unlabeled_indices]).to(device)
    # print(f"[ALFAMIX DEBUG] Unlabeled embedding matrix shape: {emb_matrix.shape}")
    # print(f"[ALFAMIX DEBUG] Embedding stats: min={emb_matrix.min().item():.4f}, max={emb_matrix.max().item():.4f}, mean={emb_matrix.mean().item():.4f}")
    # print(f"[ALFAMIX DEBUG] Any NaN in embeddings: {torch.isnan(emb_matrix).any().item()}")
    
    original_preds = predict_fn(emb_matrix)
    original_pred_dict = {idx: pred for idx, pred in zip(unlabeled_indices, original_preds.cpu().tolist())}
    
    pred_counts = defaultdict(int)
    for p in original_preds.cpu().tolist():
        pred_counts[p] += 1
    # print(f"[ALFAMIX DEBUG] Original prediction distribution: {dict(pred_counts)}")
    
    candidate_alphas = {}
    flip_targets = {}
    
    num_classes = len(class_anchors)
    anchor_ids = sorted(class_anchors.keys())
    
    for target_class in anchor_ids:
        anchor = class_anchors[target_class].to(device)
        
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if alpha > alpha_cap:
                break
            
            mixed = (1 - alpha) * emb_matrix + alpha * anchor.unsqueeze(0)
            mixed_preds = predict_fn(mixed)
            
            flips_at_alpha = 0
            for i, idx in enumerate(unlabeled_indices):
                if idx in candidate_alphas:
                    continue
                
                if mixed_preds[i].item() != original_pred_dict[idx]:
                    candidate_alphas[idx] = alpha
                    flip_targets[idx] = target_class
                    flips_at_alpha += 1
            
    #         if flips_at_alpha > 0:
    #             print(f"[ALFAMIX DEBUG] Target class {target_class}, alpha={alpha}: {flips_at_alpha} new flips")
    
    # print(f"[ALFAMIX DEBUG] Total candidates found: {len(candidate_alphas)} out of {len(unlabeled_indices)} unlabeled samples")
    
    return candidate_alphas, flip_targets


def compute_alfamix_scores(
    trainer,
    dataset,
    labeled_indices: List[int],
    unlabeled_indices: List[int],
    embeddings: Dict[int, torch.Tensor],
    alpha_cap: float = 0.5
) -> Dict[int, List[Tuple[float, int]]]:
    # print(f"[ALFAMIX DEBUG] ========== ALFAMIX SCORING START ==========")
    # print(f"[ALFAMIX DEBUG] Labeled samples: {len(labeled_indices)}, Unlabeled samples: {len(unlabeled_indices)}")
    # print(f"[ALFAMIX DEBUG] Total embeddings available: {len(embeddings)}")
    # print(f"[ALFAMIX DEBUG] Alpha cap: {alpha_cap}")
    
    # if not unlabeled_indices or not labeled_indices:
    #     print("[ALFAMIX DEBUG] Empty indices, returning empty scores")
    #     return defaultdict(list)
    
    device = trainer.device
    # print(f"[ALFAMIX DEBUG] Device: {device}")
    
    labeled_embeddings = {idx: embeddings[idx] for idx in labeled_indices if idx in embeddings}
    unlabeled_embeddings = {idx: embeddings[idx] for idx in unlabeled_indices if idx in embeddings}
    
    # print(f"[ALFAMIX DEBUG] Labeled embeddings with valid keys: {len(labeled_embeddings)}")
    # print(f"[ALFAMIX DEBUG] Unlabeled embeddings with valid keys: {len(unlabeled_embeddings)}")
    
    labeled_labels = [dataset.samples[idx][1] for idx in labeled_indices if idx in embeddings]
    label_dist = defaultdict(int)
    for lbl in labeled_labels:
        label_dist[lbl] += 1
    # print(f"[ALFAMIX DEBUG] Labeled class distribution: {dict(label_dist)}")
    
    if labeled_embeddings:
        sample_emb = list(labeled_embeddings.values())[0]
        # print(f"[ALFAMIX DEBUG] Sample embedding dim: {sample_emb.shape}, dtype: {sample_emb.dtype}")
    
    def predict_fn(emb_batch):
        emb_batch = F.normalize(emb_batch.to(device), dim=-1)
        text_features = trainer.model._prepare_text_features().to(device).to(emb_batch.dtype)
        text_features = F.normalize(text_features, dim=-1)
        logit_scale = trainer.model.logit_scale.exp()
        logits = logit_scale * emb_batch @ text_features.T
        return torch.argmax(logits, dim=1)
    
    candidate_alphas, flip_targets = find_candidate_samples(
        unlabeled_embeddings,
        labeled_embeddings,
        labeled_labels,
        predict_fn,
        alpha_cap=alpha_cap,
        device=device
    )
    
    scores_per_class = defaultdict(list)
    
    non_zero_scores = 0
    for idx in unlabeled_indices:
        if idx not in embeddings:
            continue
        
        _, class_idx = dataset.samples[idx]
        
        if idx in candidate_alphas:
            score = 1.0 / (candidate_alphas[idx] + 1e-8)
            non_zero_scores += 1
        else:
            score = 0.0
        
        scores_per_class[int(class_idx)].append((score, int(idx)))
    
    # print(f"[ALFAMIX DEBUG] Samples with non-zero scores: {non_zero_scores}")
    # print(f"[ALFAMIX DEBUG] Scores per class count: {[(k, len(v)) for k, v in sorted(scores_per_class.items())]}")
    # print(f"[ALFAMIX DEBUG] ========== ALFAMIX SCORING END ==========")
    
    return scores_per_class


def select_alfamix_indices(scores_per_class: Dict[int, List[Tuple[float, int]]], nshot: int) -> List[int]:
    if nshot <= 0:
        return []
    
    selected = []
    for class_id in sorted(scores_per_class.keys()):
        scores = scores_per_class[class_id]
        if not scores:
            continue
        sorted_scores = sorted(scores, key=lambda item: item[0], reverse=True)
        selected.extend(idx for _, idx in sorted_scores[:nshot])
    
    print(f"[ALFAMIX DEBUG] Selected {len(selected)} samples with nshot={nshot}")
    
    return selected
