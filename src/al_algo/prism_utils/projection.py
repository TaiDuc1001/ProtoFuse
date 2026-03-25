from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logger


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_projection_heads(
    heads: List[ProjectionHead],
    labeled_embs: List[torch.Tensor],
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> List[ProjectionHead]:
    if len(heads) < 2:
        logger.info("Single model, skipping projection head training")
        return heads

    for h in heads:
        h.to(device)
        h.train()

    all_params = []
    for h in heads:
        all_params.extend(h.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr)
    
    if not labeled_embs or len(labeled_embs[0]) == 0:
        return heads

    dataset = torch.utils.data.TensorDataset(*labeled_embs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        for batch_embs in loader:
            projected = []
            for m_idx, emb in enumerate(batch_embs):
                proj = heads[m_idx](emb.to(device))
                projected.append(proj)

            loss = torch.tensor(0.0, device=device)
            count = 0
            for i in range(len(heads)):
                for j in range(i + 1, len(heads)):
                    diff = projected[i] - projected[j]
                    loss = loss + (diff ** 2).sum(dim=-1).mean()
                    count += 1

            if count > 0:
                loss = loss / count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg = total_loss / max(steps, 1)
            logger.debug(f"Projection head epoch {epoch+1}/{epochs}, align_loss={avg:.6f}")

    for h in heads:
        h.eval()

    return heads


def compute_model_weights(
    models,
    labeled_embs: List[torch.Tensor],
    labeled_labels: List[int],
    prototypes: Dict[int, Dict[int, torch.Tensor]],
    device: torch.device,
    per_class: bool = False,
    num_classes: int = 0,
) -> List[float]:
    per_model_correct = [0] * len(models)
    per_model_total = [0] * len(models)
    per_model_class_correct = None
    per_model_class_total = None

    if per_class and num_classes > 0:
        per_model_class_correct = [[0] * num_classes for _ in range(len(models))]
        per_model_class_total = [[0] * num_classes for _ in range(len(models))]

    if not labeled_embs or len(labeled_embs[0]) == 0:
        return [1.0 / len(models)] * len(models)

    N = labeled_embs[0].size(0)
    batch_size = 512
    labels_tensor = torch.tensor(labeled_labels, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        labels = labels_tensor[start:end]

        for m_idx, model in enumerate(models):
            emb = labeled_embs[m_idx][start:end].to(device)

            model_protos = prototypes.get(m_idx, {})
            if not model_protos:
                continue

            proto_matrix = _build_proto_matrix_from_dict(model_protos, emb.shape[-1]).to(device)
            logits = model.classify_from_embeddings(emb, proto_matrix)
            preds = logits.argmax(dim=-1)

            correct = (preds == labels).sum().item()
            per_model_correct[m_idx] += correct
            per_model_total[m_idx] += labels.size(0)

            if per_class and per_model_class_correct is not None:
                for i in range(labels.size(0)):
                    c = labels[i].item()
                    if c < num_classes:
                        per_model_class_total[m_idx][c] += 1
                        if preds[i].item() == c:
                            per_model_class_correct[m_idx][c] += 1

    accuracies = []
    for m_idx in range(len(models)):
        if per_model_total[m_idx] > 0:
            acc = per_model_correct[m_idx] / per_model_total[m_idx]
        else:
            acc = 1.0 / len(models)
        accuracies.append(acc)

    total_acc = sum(accuracies)
    if total_acc > 0:
        weights = [a / total_acc for a in accuracies]
    else:
        weights = [1.0 / len(models)] * len(models)

    for m_idx, (w, acc) in enumerate(zip(weights, accuracies)):
        logger.info(f"Model {m_idx}: accuracy={acc:.4f}, weight={w:.4f}")

    return weights


def compute_cross_disagreement(
    predictions_per_model: Dict[int, Dict[int, int]],
) -> Dict[int, float]:
    scores = {}
    for idx, model_preds in predictions_per_model.items():
        models_list = sorted(model_preds.keys())
        M = len(models_list)
        if M < 2:
            scores[idx] = 0.0
            continue

        disagreements = 0
        pairs = 0
        for i in range(M):
            for j in range(i + 1, M):
                if model_preds[models_list[i]] != model_preds[models_list[j]]:
                    disagreements += 1
                pairs += 1

        scores[idx] = disagreements / max(pairs, 1)

    return scores


def fuse_embeddings(
    ulb_embs: List[torch.Tensor],
    ulb_indices: List[int],
    weights: List[float],
    heads: List[ProjectionHead],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    fused = {}
    if not ulb_embs or len(ulb_embs[0]) == 0:
        return fused

    N = ulb_embs[0].size(0)
    batch_size = 512
    out_features = heads[0].net[-1].out_features if heads else ulb_embs[0].shape[-1]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_fused = torch.zeros(end - start, out_features, device=device)

        for m_idx in range(len(ulb_embs)):
            emb = ulb_embs[m_idx][start:end].to(device)
            if m_idx < len(heads):
                proj = heads[m_idx](emb)
            else:
                proj = emb
            batch_fused = batch_fused + weights[m_idx] * proj

        batch_fused = batch_fused.cpu()
        for i in range(end - start):
            idx = ulb_indices[start + i]
            fused[idx] = batch_fused[i]

    return fused


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
