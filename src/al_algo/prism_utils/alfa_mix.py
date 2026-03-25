import math
from typing import Dict, List, Tuple, Optional, Callable

import torch
import torch.nn.functional as F

from utils import logger


def calculate_optimum_alpha(
    eps: float,
    anchor_embedding: torch.Tensor,
    ulb_embedding: torch.Tensor,
    ulb_grads: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        z = anchor_embedding - ulb_embedding
        alpha = (
            (eps * z.norm(dim=1) / ulb_grads.norm(dim=1))
            .unsqueeze(1)
            .repeat(1, z.size(1))
            * ulb_grads
            / (z + 1e-8)
        )
    return alpha


def find_candidate_set(
    prototypes: Dict[int, torch.Tensor],
    ulb_embedding: torch.Tensor,
    pred_1: torch.Tensor,
    classify_fn: Callable[[torch.Tensor], torch.Tensor],
    alpha_cap: float,
    grads: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, D = ulb_embedding.size()
    pred_change = torch.zeros(N, dtype=torch.bool, device=ulb_embedding.device)
    min_alphas = torch.ones(N, dtype=torch.float32, device=ulb_embedding.device) * float('inf')

    # alpha_cap is adjusted by sqrt(D) as in reference implementation
    alpha_cap_scaled = alpha_cap / math.sqrt(D)
    num_classes = max(prototypes.keys()) + 1

    for c in range(num_classes):
        if c not in prototypes:
            continue
            
        anchor = prototypes[c].unsqueeze(0).repeat(N, 1).to(ulb_embedding.device)

        with torch.no_grad():
            alpha = calculate_optimum_alpha(alpha_cap_scaled, anchor, ulb_embedding, grads)
        
        alpha = torch.clamp(alpha, min=-alpha_cap, max=alpha_cap)

        # Mix the features (reference implementation convention: anchor is multiplied by alpha)
        z_tilde = (1 - alpha) * ulb_embedding + alpha * anchor
        
        with torch.no_grad():
            logits_2 = classify_fn(z_tilde)
            pred_2 = logits_2.argmax(dim=-1)

        changed = (pred_1 != pred_2)
        pred_change = pred_change | changed
        
        alpha_norm = alpha.norm(dim=1)
        update_mask = changed & (alpha_norm < min_alphas)
        min_alphas[update_mask] = alpha_norm[update_mask]

    return pred_change, min_alphas


def compute_inconsistency_scores(
    models,
    weights: List[float],
    prototypes: Dict[int, Dict[int, torch.Tensor]],
    ulb_embs: List[torch.Tensor],
    ulb_indices: List[int],
    device: torch.device,
    alpha_cap: float = 0.5,
    alpha_growth_step: float = 0.1,
) -> Tuple[Dict[int, float], Dict[int, Dict[int, int]], Dict[int, torch.Tensor]]:
    
    inconsistencies = {}
    preds_per_model = {}
    
    if not ulb_embs or len(ulb_embs[0]) == 0:
        return {}, {}, {}
        
    for idx in ulb_indices:
        inconsistencies[idx] = 0.0
        preds_per_model[idx] = {}

    N = ulb_embs[0].size(0)
    batch_size = 512

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        for m_idx, model in enumerate(models):
            emb = ulb_embs[m_idx][start:end].to(device)
            # Create a leaf tensor that requires grad
            emb_leaf = emb.clone().detach().requires_grad_(True)

            model_protos = prototypes.get(m_idx, {})
            if not model_protos:
                continue
                
            proto_matrix = _build_proto_matrix_from_dict(model_protos, emb.shape[-1]).to(device)

            def classify_fn(x):
                return model.classify_from_embeddings(x, proto_matrix)

            with torch.enable_grad():
                logits = classify_fn(emb_leaf)
                preds = logits.argmax(dim=-1)
                
                # CrossEntropy against self-prediction
                probs = F.softmax(logits, dim=-1)
                with torch.no_grad():
                    targets = probs.argmax(dim=-1)
                
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                grads = emb_leaf.grad.clone()

            # Record predictions for CrossDis
            for i in range(end - start):
                idx = ulb_indices[start + i]
                preds_per_model[idx][m_idx] = preds[i].item()

            pred_change, _ = find_candidate_set(
                model_protos,
                emb_leaf.detach(),
                preds,
                classify_fn,
                alpha_cap,
                grads
            )
            
            w = weights[m_idx]
            for i in range(end - start):
                if pred_change[i]:
                    idx = ulb_indices[start + i]
                    inconsistencies[idx] += w

    # Also return raw embeddings mapped to idx for fused novelty fallback
    raw_embeddings = {}
    if ulb_embs and len(models) > 0:
        for i in range(N):
            raw_embeddings[ulb_indices[i]] = ulb_embs[0][i].cpu()

    return inconsistencies, preds_per_model, raw_embeddings


def _build_proto_matrix_from_dict(
    prototypes: Dict[int, torch.Tensor],
    embed_dim: int,
) -> torch.Tensor:
    if not prototypes:
        return torch.zeros(1, embed_dim)
    max_cls = max(prototypes.keys()) + 1
    matrix = torch.zeros(max_cls, embed_dim)
    for cls_idx, proto in prototypes.items():
        matrix[cls_idx] = proto
    return matrix
