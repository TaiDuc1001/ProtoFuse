from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def build_dpp_kernel(
    scores: Dict[int, float],
    embeddings: Dict[int, torch.Tensor],
) -> np.ndarray:
    indices = sorted(scores.keys())
    n = len(indices)

    if n == 0:
        return np.zeros((0, 0))

    score_vec = np.array([scores[idx] for idx in indices])

    emb_list = []
    for idx in indices:
        if idx in embeddings:
            e = embeddings[idx]
            if isinstance(e, torch.Tensor):
                e = e.detach().cpu().numpy()
            emb_list.append(e.flatten())
        else:
            emb_list.append(np.zeros(1))

    emb_matrix = np.stack(emb_list)

    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    emb_normed = emb_matrix / norms

    sim_matrix = emb_normed @ emb_normed.T
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    score_outer = np.outer(score_vec, score_vec)
    kernel = score_outer * sim_matrix

    kernel += np.eye(n) * 1e-6

    return kernel


def greedy_dpp_select(
    kernel: np.ndarray,
    budget: int,
    index_mapping: Optional[List[int]] = None,
) -> List[int]:
    n = kernel.shape[0]
    if n == 0:
        return []

    budget = min(budget, n)

    selected = []
    remaining = list(range(n))

    diag = np.diag(kernel).copy()
    first = int(np.argmax(diag))
    selected.append(first)
    remaining.remove(first)

    L = np.zeros((budget, n))
    L[0, :] = kernel[first, :] / np.sqrt(kernel[first, first] + 1e-10)

    for t in range(1, budget):
        if not remaining:
            break

        best_idx = -1
        best_score = -float("inf")

        for idx in remaining:
            c_val = kernel[idx, idx]
            for s in range(t):
                c_val -= L[s, idx] ** 2
            c_val = max(c_val, 1e-10)
            if c_val > best_score:
                best_score = c_val
                best_idx = idx

        if best_idx < 0 or best_score <= 1e-10:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

        sqrt_val = np.sqrt(best_score)
        for idx in remaining:
            dot_sum = 0.0
            for s in range(t):
                dot_sum += L[s, best_idx] * L[s, idx]
            L[t, idx] = (kernel[best_idx, idx] - dot_sum) / sqrt_val

    if index_mapping is not None:
        return [index_mapping[s] for s in selected]
    return selected
