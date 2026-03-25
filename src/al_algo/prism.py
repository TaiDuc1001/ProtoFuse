from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset

from .base import BaseStrategy
from .prism_utils.model_zoo import load_models_from_config, ModelWrapper
from .prism_utils.prototypes import generate_pseudo_labels, compute_prototypes
from .prism_utils.alfa_mix import compute_inconsistency_scores
from .prism_utils.projection import (
    ProjectionHead,
    train_projection_heads,
    compute_model_weights,
    compute_cross_disagreement,
    fuse_embeddings,
)
from .prism_utils.dpp import build_dpp_kernel, greedy_dpp_select

from utils import logger, ConfigNode, coerce_to_float, coerce_to_int


class PrismStrategy(BaseStrategy):
    def __init__(self):
        self._models = None
        self._heads = None
        self._round_idx = 0
        self._lambda2_current = None
        self._novel_fraction_current = None

    @property
    def name(self) -> str:
        return "prism"

    def _extract_embeddings(
        self,
        models: List[ModelWrapper],
        dataloader: DataLoader,
        device: torch.device,
        desc: str = "dataset"
    ) -> Tuple[List[torch.Tensor], List[int], List[int]]:
        """
        Extracts embeddings for all models in one pass over the dataloader.
        Returns:
            embs: List[Tensor] where each tensor is shape (N, D) for the m-th model
            labels: List[int] true labels (if available)
            indices: List[int] global dataset indices corresponding to the samples
        """
        if dataloader is None:
            return [], [], []
            
        dataset_len = len(dataloader.dataset)
        logger.info(f"Extracting features for {dataset_len} {desc} samples across {len(models)} models...")
        
        embs = [[] for _ in models]
        all_labels = []
        all_indices = []
        
        sample_idx = 0
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, local_indices = batch[0].to(device), batch[1], batch[2]
            elif len(batch) == 2:
                images, labels = batch[0].to(device), batch[1]
                local_indices = torch.arange(sample_idx, sample_idx + images.size(0))
            else:
                images = batch[0].to(device)
                labels = torch.zeros(images.size(0), dtype=torch.long)
                local_indices = torch.arange(sample_idx, sample_idx + images.size(0))
                
            all_labels.extend(labels.tolist())
            all_indices.extend(local_indices.tolist())
            
            for m_idx, model in enumerate(models):
                with torch.no_grad():
                    embs[m_idx].append(model.embed(images).cpu())
                    
            sample_idx += images.size(0)
            
        embs = [torch.cat(e, dim=0) for e in embs]
        return embs, all_labels, all_indices


    def score(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        **kwargs,
    ) -> Dict[str, Any]:
        config = kwargs.get("config", ConfigNode())
        if not isinstance(config, ConfigNode):
            config = ConfigNode(config)

        prism_cfg = config.get("prism", ConfigNode())
        if not isinstance(prism_cfg, ConfigNode):
            prism_cfg = ConfigNode(prism_cfg)

        labeled_loader = kwargs.get("labeled_loader")
        labeled_indices_kw = kwargs.get("labeled_indices", [])
        dataset = kwargs.get("dataset")
        round_idx = kwargs.get("round_idx", 1)
        self._round_idx = round_idx

        lambda1 = coerce_to_float(prism_cfg.get("lambda1", 0.3), 0.3)
        lambda2 = coerce_to_float(prism_cfg.get("lambda2", 0.5), 0.5)
        lambda2_decay = coerce_to_float(prism_cfg.get("lambda2_decay", 0.8), 0.8)
        lambda2_min = coerce_to_float(prism_cfg.get("lambda2_min", 0.1), 0.1)
        alpha_cap = coerce_to_float(prism_cfg.get("alpha_cap", 0.1), 0.1)
        alpha_growth_step = coerce_to_float(prism_cfg.get("alpha_growth_step", 0.1), 0.1)
        pseudo_threshold = coerce_to_float(prism_cfg.get("pseudo_label_threshold", 0.85), 0.85)
        entropy_threshold = coerce_to_float(prism_cfg.get("agreement_entropy_threshold", 0.5), 0.5)
        pseudo_method = str(prism_cfg.get("pseudo_label_method", "weighted_consensus"))
        adaptive_thr = bool(prism_cfg.get("adaptive_threshold", True))
        adaptive_beta = coerce_to_float(prism_cfg.get("adaptive_threshold_beta", 0.95), 0.95)
        proj_epochs = coerce_to_int(prism_cfg.get("projection_head_epochs", 50), 50)
        proj_lr = coerce_to_float(prism_cfg.get("projection_head_lr", 0.001), 0.001)
        per_class_w = bool(prism_cfg.get("per_class_weights", False))
        novelty_method = str(prism_cfg.get("novelty_method", "mahalanobis"))
        candidate_mult = coerce_to_int(prism_cfg.get("candidate_pool_multiplier", 5), 5)

        if self._lambda2_current is None:
            self._lambda2_current = lambda2
        else:
            self._lambda2_current = max(self._lambda2_current * lambda2_decay, lambda2_min)

        logger.info(f"PRISM round {round_idx}: λ₁={lambda1:.3f}, λ₂={self._lambda2_current:.3f}")

        if self._models is None:
            self._models = load_models_from_config(config, device)
        models = self._models

        num_classes = kwargs.get("num_classes", 0)
        if num_classes == 0 and dataset is not None and hasattr(dataset, "classes"):
            num_classes = len(dataset.classes)

        # -------------------------------------------------------------
        # STEP 1.5: Pre-extract ALL embeddings for blazing fast AL computations
        # -------------------------------------------------------------
        labeled_embs, labeled_labels, out_labeled_indices = self._extract_embeddings(
            models, labeled_loader, device, desc="labeled"
        )
        
        ulb_embs, ulb_labels, out_ulb_indices = self._extract_embeddings(
            models, dataloader, device, desc="unlabeled"
        )
        # -------------------------------------------------------------

        # In case the subset DataLoader wrapper doesn't provide strict dataset global indices:
        if labeled_indices_kw:
            labeled_indices = list(labeled_indices_kw)
        else:
            labeled_indices = out_labeled_indices
            
        # For unlabeled, since dataloader is unmapped, map it sequentially
        unlabeled_indices_kw = kwargs.get("unlabeled_indices", [])
        if unlabeled_indices_kw and len(unlabeled_indices_kw) == len(out_ulb_indices):
            ulb_indices = list(unlabeled_indices_kw)
        else:
            # We assume active_learning.py's Sequential Dataloader passes samples in strict order of unlabeled_indices
            ulb_indices_attr = getattr(getattr(dataloader, 'dataset', None), 'indices', None)
            if ulb_indices_attr is not None:
                ulb_indices = list(ulb_indices_attr)
            else:
                ulb_indices = out_ulb_indices # Fallback local indices

        labeled_labels_dict = dict(zip(labeled_indices, labeled_labels))

        logger.info("Step 2: Computing model weights...")
        dummy_prototypes = {}
        if labeled_embs:
            for m_idx in range(len(models)):
                dummy_prototypes[m_idx] = {}
                cls_embs = defaultdict(list)
                for i, lbl in enumerate(labeled_labels):
                    cls_embs[lbl].append(labeled_embs[m_idx][i])
                for c, embs in cls_embs.items():
                    dummy_prototypes[m_idx][c] = torch.stack(embs).mean(dim=0)

        weights = compute_model_weights(
            models, labeled_embs, labeled_labels, dummy_prototypes, device,
            per_class=per_class_w, num_classes=num_classes,
        ) if labeled_embs else [1.0 / len(models)] * len(models)

        logger.info("Step 3: Generating high-confidence pseudo-labels...")
        pseudo_labels, conflicted_indices, all_predictions = generate_pseudo_labels(
            models, weights, ulb_embs, ulb_indices, device,
            threshold=pseudo_threshold,
            entropy_threshold=entropy_threshold,
            method=pseudo_method,
            adaptive_threshold=adaptive_thr,
            adaptive_beta=adaptive_beta,
            num_classes=num_classes,
        )

        logger.info("Step 3b: Computing class prototypes (labeled + high-conf pseudo)...")
        prototypes, covariances = compute_prototypes(
            models, labeled_indices, labeled_labels_dict, pseudo_labels,
            labeled_embs, ulb_embs, ulb_indices, device,
        ) if dataset is not None else ({}, {})

        if not prototypes:
            prototypes = dummy_prototypes

        logger.info("Step 4: Training projection heads...")
        if self._heads is None:
            common_dim = models[0].embed_dim
            self._heads = [ProjectionHead(m.embed_dim, common_dim) for m in models]

        if labeled_embs and len(models) > 1:
            self._heads = train_projection_heads(
                self._heads, labeled_embs, device,
                epochs=proj_epochs, lr=proj_lr,
            )

        logger.info("Step 6-7: Computing inconsistency scores (ALFA-Mix)...")
        inconsistency, predictions_per_model, raw_embeddings = \
            compute_inconsistency_scores(
                models, weights, prototypes, ulb_embs, ulb_indices, device,
                alpha_cap=alpha_cap,
                alpha_growth_step=alpha_growth_step,
            )

        logger.info("Step 8: Computing cross-model disagreement...")
        cross_dis = compute_cross_disagreement(predictions_per_model)

        logger.info("Step 9: Fusing embeddings + computing novelty...")
        fused = fuse_embeddings(
            ulb_embs, ulb_indices, weights, self._heads, device,
        ) if len(models) > 1 else {idx: emb for idx, emb in raw_embeddings.items()}

        fused_prototypes = {}
        if prototypes and len(models) > 0:
            first_protos = prototypes.get(0, {})
            fused_prototypes = first_protos

        if novelty_method == "mahalanobis":
            novelty = self._compute_novelty(fused, fused_prototypes, covariances)
        else:
            novelty = self._compute_energy_novelty(models, ulb_embs, ulb_indices, device)

        logger.info("Step 11: Computing combined Score = Incons + λ₁·CrossDis + λ₂·Novel...")
        all_indices = sorted(set(inconsistency.keys()) | set(cross_dis.keys()) | set(novelty.keys()))
        scores = {}
        for idx in all_indices:
            inc = inconsistency.get(idx, 0.0)
            cd = cross_dis.get(idx, 0.0)
            nov = novelty.get(idx, 0.0)
            scores[idx] = inc + lambda1 * cd + self._lambda2_current * nov

        proto_predictions = {}
        proto_source = prototypes.get(0, dummy_prototypes.get(0, {}))
        if proto_source:
            proto_classes = sorted(proto_source.keys())
            proto_matrix = torch.stack([proto_source[c] for c in proto_classes])
            proto_matrix = F.normalize(proto_matrix, dim=-1)

            m0_embs = ulb_embs[0]
            batch_size = 512
            for start in range(0, len(ulb_indices), batch_size):
                end = min(start + batch_size, len(ulb_indices))
                batch_embs = F.normalize(m0_embs[start:end], dim=-1)
                sims = batch_embs @ proto_matrix.T
                preds = sims.argmax(dim=-1)
                for i in range(end - start):
                    proto_predictions[ulb_indices[start + i]] = proto_classes[preds[i].item()]

            logger.info(f"Prototype-based predictions: {len(proto_predictions)} samples, "
                        f"{len(set(proto_predictions.values()))} unique classes")
        else:
            proto_predictions = all_predictions

        logger.info(
            f"PRISM scoring complete: {len(scores)} samples, "
            f"Incons range=[{min(inconsistency.values(), default=0):.3f}, {max(inconsistency.values(), default=0):.3f}], "
            f"CrossDis range=[{min(cross_dis.values(), default=0):.3f}, {max(cross_dis.values(), default=0):.3f}], "
            f"Novel range=[{min(novelty.values(), default=0):.3f}, {max(novelty.values(), default=0):.3f}]"
        )

        return {
            "scores": scores,
            "strategy": self.name,
            "embeddings": fused,
            "novelty_scores": novelty,
            "inconsistency_scores": inconsistency,
            "cross_disagreement_scores": cross_dis,
            "conflicted_indices": conflicted_indices,
            "lambda2_current": self._lambda2_current,
            "predicted_labels": proto_predictions,
        }

    def select(
        self,
        scores: Dict[int, float],
        budget: int,
        **kwargs,
    ) -> List[int]:
        embeddings = kwargs.get("embeddings", {})
        novelty_scores = kwargs.get("novelty_scores", {})
        config = kwargs.get("config", ConfigNode())
        conflicted_indices = kwargs.get("conflicted_indices", [])
        predicted_labels = kwargs.get("predicted_labels", {})

        if not isinstance(config, ConfigNode):
            config = ConfigNode(config)

        prism_cfg = config.get("prism", ConfigNode())
        if not isinstance(prism_cfg, ConfigNode):
            prism_cfg = ConfigNode(prism_cfg)

        threshold_pct = coerce_to_float(prism_cfg.get("novelty_threshold_percentile", 75), 75)
        candidate_mult = coerce_to_int(prism_cfg.get("candidate_pool_multiplier", 5), 5)

        novel_fraction = coerce_to_float(prism_cfg.get("novel_budget_fraction", 0.5), 0.5)
        novel_fraction_min = coerce_to_float(prism_cfg.get("novel_budget_fraction_min", 0.2), 0.2)
        novel_fraction_decay = coerce_to_float(prism_cfg.get("novel_budget_fraction_decay", 0.85), 0.85)

        if self._novel_fraction_current is None:
            self._novel_fraction_current = novel_fraction
        else:
            self._novel_fraction_current = max(
                self._novel_fraction_current * novel_fraction_decay,
                novel_fraction_min,
            )

        candidate_k = min(candidate_mult * budget, len(scores))
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        candidate_pool = sorted_indices[:candidate_k]

        for ci in conflicted_indices:
            if ci in scores and ci not in candidate_pool:
                candidate_pool.append(ci)

        if predicted_labels:
            num_classes = len(set(predicted_labels.values()))
        else:
            num_classes = budget // candidate_mult if candidate_mult > 0 else 1
            
        num_classes = max(1, num_classes)
        
        budget_per_class = max(1, budget // num_classes)
        novel_budget_per_class = max(1, int(budget_per_class * self._novel_fraction_current))
        uncertain_budget_per_class = max(0, budget_per_class - novel_budget_per_class)

        class_to_candidates = defaultdict(list)
        for idx in candidate_pool:
            label = predicted_labels.get(idx, 0)
            class_to_candidates[label].append(idx)

        logger.info(
            f"PRISM select (class-balanced): Budget={budget}, Num Classes={num_classes}, "
            f"Per-Class Budget={budget_per_class} (uncertain={uncertain_budget_per_class}, novel={novel_budget_per_class})"
        )

        combined = []
        # Run DPP per class
        for c, c_candidates in class_to_candidates.items():
            if not c_candidates:
                continue

            candidate_novelty = [novelty_scores.get(idx, 0.0) for idx in c_candidates if idx in novelty_scores]
            tau = float(np.percentile(candidate_novelty, threshold_pct)) if candidate_novelty else 0.0

            uncertain_pool = [idx for idx in c_candidates if novelty_scores.get(idx, 0.0) < tau]
            novel_pool = [idx for idx in c_candidates if novelty_scores.get(idx, 0.0) >= tau]

            c_selected_uncertain = self._dpp_select_from_pool(
                uncertain_pool, uncertain_budget_per_class, scores, embeddings,
            )
            c_selected_novel = self._dpp_select_from_pool(
                novel_pool, novel_budget_per_class, scores, embeddings,
            )

            c_combined = list(c_selected_uncertain) + list(c_selected_novel)
            c_combined = list(dict.fromkeys(c_combined))

            # If a class falls short (e.g., very few candidate samples predicted for this class)
            # fill up to budget_per_class using top scoring samples of THIS class.
            if len(c_combined) < budget_per_class:
                remaining = [idx for idx in c_candidates if idx not in set(c_combined)]
                # Sort remaining by score
                remaining = sorted(remaining, key=lambda i: scores[i], reverse=True)
                c_combined.extend(remaining[:budget_per_class - len(c_combined)])

            combined.extend(c_combined[:budget_per_class])

        combined = list(dict.fromkeys(combined))

        if len(combined) < budget:
            remaining = [idx for idx in sorted_indices if idx not in set(combined)]
            combined.extend(remaining[:budget - len(combined)])

        return combined[:budget]

    def update_schedule(self, round_idx: int):
        self._round_idx = round_idx

    def _dpp_select_from_pool(
        self,
        pool: List[int],
        budget: int,
        scores: Dict[int, float],
        embeddings: Dict[int, torch.Tensor],
    ) -> List[int]:
        if not pool or budget <= 0:
            return []

        budget = min(budget, len(pool))

        pool_scores = {idx: scores.get(idx, 0.0) for idx in pool}
        pool_embeddings = {idx: embeddings.get(idx, torch.zeros(1)) for idx in pool}

        kernel = build_dpp_kernel(pool_scores, pool_embeddings)
        index_mapping = sorted(pool_scores.keys())

        selected = greedy_dpp_select(kernel, budget, index_mapping)
        return selected

    def _compute_novelty(
        self,
        fused_embeddings: Dict[int, torch.Tensor],
        prototypes: Dict[int, torch.Tensor],
        covariances: Dict[int, torch.Tensor],
    ) -> Dict[int, float]:
        if not prototypes or not fused_embeddings:
            return {idx: 0.0 for idx in fused_embeddings}

        proto_list = []
        proto_classes = sorted(prototypes.keys())
        for c in proto_classes:
            proto_list.append(prototypes[c])
        proto_matrix = torch.stack(proto_list)

        cov_invs = {}
        for c in proto_classes:
            cov = covariances.get(c)
            if cov is None:
                d = proto_matrix.shape[-1]
                cov_invs[c] = torch.eye(d)
            else:
                try:
                    cov_invs[c] = torch.linalg.inv(cov)
                except Exception:
                    cov_invs[c] = torch.eye(cov.shape[0])

        novelty = {}
        
        # Convert fused_embeddings dict to a single tensor 
        items = list(fused_embeddings.items())
        if not items:
            return novelty
            
        indices = [item[0] for item in items]
        embs = torch.stack([item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1], dtype=torch.float32) for item in items])
        embs = embs.detach().float()
        
        # Pre-allocate squared mahalanobis distances (N, C)
        squared_mahal = torch.zeros((embs.size(0), len(proto_classes)), device=embs.device)
        
        for i, c in enumerate(proto_classes):
            # diff_c shape: (N, D)
            diff_c = embs - proto_matrix[i].unsqueeze(0)
            
            # cov_inv shape: (D, D)
            cov_inv = cov_invs[c].to(embs.device)
            
            # Batched inner product for all N samples:
            # left_term = diff_c @ cov_inv -> shape: (N, D)
            left_term = torch.matmul(diff_c, cov_inv)
            
            # squared distance = sum(left_term * diff_c, dim=-1) -> shape: (N,)
            dist_c = torch.sum(left_term * diff_c, dim=-1)
            
            squared_mahal[:, i] = dist_c
        
        # Clamp to 0 to avoid sqrt(negative) issues from numerical instability
        mahal = torch.sqrt(torch.clamp(squared_mahal, min=0.0))
        
        # Find minimum distance across classes for each sample
        # mahal shape: (N, C) -> min(dim=1) -> (N,)
        min_dists, _ = torch.min(mahal, dim=1)
        
        for i, idx in enumerate(indices):
            novelty[idx] = min_dists[i].item()

        return novelty

    def _compute_energy_novelty(
        self,
        models,
        ulb_embs: List[torch.Tensor],
        ulb_indices: List[int],
        device: torch.device,
    ) -> Dict[int, float]:
        novelty = {}

        if not ulb_embs or len(ulb_embs[0]) == 0:
            return novelty

        N = ulb_embs[0].size(0)
        batch_size = 512
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            
            energies = torch.zeros(end - start, device=device)
            for m_idx, model in enumerate(models):
                emb = ulb_embs[m_idx][start:end].to(device)
                logits = model.get_logits_from_embeddings(emb)
                energy = -torch.logsumexp(logits, dim=-1)
                energies += energy

            energies /= len(models)
            energies = energies.cpu()

            for i in range(end - start):
                idx = ulb_indices[start + i]
                novelty[idx] = energies[i].item()

        return novelty
