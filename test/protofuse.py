import json
import math
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    DEFAULT_ARG_SCHEMA,
    compute_metrics,
    create_argument_parser,
    load_config_file,
    log_experiment_metrics,
    logger,
    merge_configs,
    parse_override_arguments,
    process_parsed_args,
    setup_logging,
)
from src.pipelines.protofuse import ProtoFusePipeline


ARG_SCHEMA = DEFAULT_ARG_SCHEMA
ALPHA_STEP = 0.01


class ProtoFuseDamageForensicsPipeline(ProtoFusePipeline):
    METHOD_NAME = "ProtoFuse Damage Forensics"
    DEFAULT_OUTPUT_DIR = "outputs/protofuse_damage_forensics"

    def _safe_name(self, value):
        return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")

    def _float(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.detach().cpu().item()
        return float(value)

    def _metric_dict(self, labels, preds):
        return {key: float(value) for key, value in compute_metrics(labels, preds).items()}

    def _stats(self, prefix, values):
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float32, device=self.device)
        values = values.detach().float()
        if values.numel() == 0:
            return {
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
                f"{prefix}_min": None,
                f"{prefix}_q25": None,
                f"{prefix}_median": None,
                f"{prefix}_q75": None,
                f"{prefix}_max": None,
            }
        return {
            f"{prefix}_mean": self._float(values.mean()),
            f"{prefix}_std": self._float(values.std(unbiased=False)),
            f"{prefix}_min": self._float(values.min()),
            f"{prefix}_q25": self._float(torch.quantile(values, 0.25)),
            f"{prefix}_median": self._float(torch.quantile(values, 0.50)),
            f"{prefix}_q75": self._float(torch.quantile(values, 0.75)),
            f"{prefix}_max": self._float(values.max()),
        }

    def _corr(self, x, y):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        mask = torch.isfinite(x) & torch.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.numel() < 2:
            return None
        x_std = x.std(unbiased=False)
        y_std = y.std(unbiased=False)
        if self._float(x_std) == 0.0 or self._float(y_std) == 0.0:
            return None
        return self._float(((x - x.mean()) * (y - y.mean())).mean() / (x_std * y_std + 1e-12))

    def _label_remap(self, train_labels, val_labels):
        task_classes = sorted(set(train_labels.tolist()))
        remap = {class_idx: idx for idx, class_idx in enumerate(task_classes)}
        missing = sorted(set(val_labels.tolist()) - set(remap.keys()))
        if missing:
            raise ValueError(f"Validation labels not present in train split: {missing[:10]}")
        remapped_train = torch.tensor([remap[label.item()] for label in train_labels], dtype=torch.long)
        remapped_val = torch.tensor([remap[label.item()] for label in val_labels], dtype=torch.long)
        return remapped_train, remapped_val, len(task_classes)

    def _class_indices(self, labels, num_classes):
        indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels.tolist()):
            indices[label].append(idx)
        return indices

    def _shots_per_class(self, labels, num_classes):
        counts = torch.bincount(labels, minlength=num_classes)
        return int(counts.min().item())

    def _alpha_grid(self):
        return torch.linspace(0.0, 1.0, 101, device=self.device)

    def _masked_margin(self, logits, labels):
        labels = labels.to(logits.device)
        correct = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, labels.view(-1, 1), -float("inf"))
        second = masked.max(dim=1).values
        return correct - second, correct, second

    def _entropy(self, logits, temperature=1.0):
        probs = F.softmax(logits / temperature, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    def _logits_for_alpha(self, alpha, T, V, features, num_classes, one_shot_mode):
        features = F.normalize(features.to(self.device), dim=-1)
        if one_shot_mode:
            logits_text = features @ T.T
            logits_visual = features @ V.T
            probs_visual = F.softmax(logits_visual / 0.05, dim=-1)
            entropy = -(probs_visual * torch.log(probs_visual + 1e-8)).sum(dim=-1)
            confidence = 1.0 - (entropy / math.log(num_classes)).clamp(0.0, 1.0)
            alpha_x = alpha * (0.5 + 0.5 * confidence)
            return (1 - alpha_x).unsqueeze(-1) * logits_text + alpha_x.unsqueeze(-1) * logits_visual

        prototypes = F.normalize((1 - alpha) * T + alpha * V, dim=-1)
        return features @ prototypes.T

    def _pred_margin_matrices(self, T, V, features, labels, num_classes, one_shot_mode):
        labels = labels.to(self.device)
        preds = []
        margins = []
        for alpha in self._alpha_grid():
            logits = self._logits_for_alpha(float(alpha.item()), T, V, features, num_classes, one_shot_mode)
            margin, _, _ = self._masked_margin(logits, labels)
            preds.append(logits.argmax(dim=-1))
            margins.append(margin)
        pred_matrix = torch.stack(preds, dim=0)
        margin_matrix = torch.stack(margins, dim=0)
        correct_matrix = pred_matrix.eq(labels.view(1, -1))
        return pred_matrix, correct_matrix, margin_matrix

    def _pure_endpoint_state(self, T, V, features, labels, num_classes):
        labels = labels.to(self.device)
        features = F.normalize(features.to(self.device), dim=-1)
        logits_text = features @ T.T
        logits_visual = features @ V.T
        text_margin, text_correct_score, text_second_score = self._masked_margin(logits_text, labels)
        visual_margin, visual_correct_score, visual_second_score = self._masked_margin(logits_visual, labels)
        text_pred = logits_text.argmax(dim=-1)
        visual_pred = logits_visual.argmax(dim=-1)
        text_entropy = self._entropy(logits_text) / math.log(num_classes)
        visual_entropy = self._entropy(logits_visual, temperature=0.05) / math.log(num_classes)
        visual_confidence = 1.0 - visual_entropy.clamp(0.0, 1.0)
        return {
            "text_logits": logits_text,
            "visual_logits": logits_visual,
            "text_pred": text_pred,
            "visual_pred": visual_pred,
            "text_correct": text_pred.eq(labels),
            "visual_correct": visual_pred.eq(labels),
            "text_margin": text_margin,
            "visual_margin": visual_margin,
            "text_correct_score": text_correct_score,
            "text_second_score": text_second_score,
            "visual_correct_score": visual_correct_score,
            "visual_second_score": visual_second_score,
            "text_entropy": text_entropy,
            "visual_entropy": visual_entropy,
            "visual_confidence": visual_confidence,
        }

    def _group_types(self, endpoint):
        text_correct = endpoint["text_correct"]
        visual_correct = endpoint["visual_correct"]
        groups = []
        for tc, vc in zip(text_correct.tolist(), visual_correct.tolist()):
            if tc and vc:
                groups.append("both_correct")
            elif (not tc) and (not vc):
                groups.append("both_wrong")
            elif (not tc) and vc:
                groups.append("rescue_candidate")
            else:
                groups.append("damage_candidate")
        return groups

    def _best_from_sweep(self, sweep):
        return max(sweep, key=lambda row: (row.get("accuracy", 0.0), row.get("mca", 0.0)))

    def _sweep_metrics(self, pred_matrix, labels):
        labels_list = labels.tolist()
        sweep = []
        for idx, alpha in enumerate(self._alpha_grid()):
            metrics = self._metric_dict(labels_list, pred_matrix[idx].cpu().tolist())
            metrics["alpha"] = self._float(alpha)
            sweep.append(metrics)
        return sweep

    def _transition_curves(self, correct_matrix, endpoint):
        text_correct = endpoint["text_correct"].to(self.device)
        alphas = self._alpha_grid()
        rescued = ((~text_correct).view(1, -1) & correct_matrix).sum(dim=1).float()
        damaged = (text_correct.view(1, -1) & ~correct_matrix).sum(dim=1).float()
        net = rescued - damaged
        delta_rescue = torch.zeros_like(rescued)
        delta_damage = torch.zeros_like(damaged)
        delta_net = torch.zeros_like(net)
        delta_rescue[1:] = rescued[1:] - rescued[:-1]
        delta_damage[1:] = damaged[1:] - damaged[:-1]
        delta_net[1:] = net[1:] - net[:-1]
        damage_acc = torch.zeros_like(damaged)
        damage_acc[2:] = delta_damage[2:] - delta_damage[1:-1]

        features = {
            "alpha_max_net": self._float(alphas[net.argmax()]),
            "max_net": self._float(net.max()),
            "alpha_damage_exceeds_rescue_first": None,
            "alpha_damage_slope_exceeds_rescue_slope_first": None,
            "alpha_net_starts_decreasing": None,
            "alpha_damage_acceleration_peak": self._float(alphas[damage_acc.argmax()]),
        }

        mask = damaged > rescued
        if mask.any():
            features["alpha_damage_exceeds_rescue_first"] = self._float(alphas[mask.float().argmax()])
        mask = delta_damage > delta_rescue
        if mask[1:].any():
            features["alpha_damage_slope_exceeds_rescue_slope_first"] = self._float(alphas[mask.float().argmax()])
        mask = delta_net < 0
        if mask[1:].any():
            features["alpha_net_starts_decreasing"] = self._float(alphas[mask.float().argmax()])

        for idx in range(len(alphas)):
            features[f"rescue_alpha_{idx:03d}"] = int(rescued[idx].item())
            features[f"damage_alpha_{idx:03d}"] = int(damaged[idx].item())
            features[f"net_alpha_{idx:03d}"] = int(net[idx].item())
            if idx > 0:
                features[f"delta_rescue_alpha_{idx:03d}"] = int(delta_rescue[idx].item())
                features[f"delta_damage_alpha_{idx:03d}"] = int(delta_damage[idx].item())
                features[f"delta_net_alpha_{idx:03d}"] = int(delta_net[idx].item())

        for lam in (1.0, 1.5, 2.0, 3.0, 5.0):
            score = rescued - lam * damaged
            key = self._safe_name(str(lam).replace(".", "_"))
            features[f"best_alpha_net_lam_{key}"] = self._float(alphas[score.argmax()])
            features[f"score_net_lam_{key}_max"] = self._float(score.max())
            if lam in (1.0, 2.0, 3.0):
                for idx in range(len(alphas)):
                    features[f"score_net_lam_{key}_alpha_{idx:03d}"] = self._float(score[idx])
        return features, rescued, damaged, net, delta_rescue, delta_damage

    def _flip_features(self, pred_matrix, correct_matrix, endpoint, oracle_idx, current_idx):
        text_pred = endpoint["text_pred"].to(self.device)
        text_correct = endpoint["text_correct"].to(self.device)
        alphas = self._alpha_grid()
        changed = pred_matrix.ne(text_pred.view(1, -1))
        changed[0] = False
        has_flip = changed.any(dim=0)
        first_flip_idx = changed.float().argmax(dim=0)

        has_correct = correct_matrix.any(dim=0)
        first_correct_idx = correct_matrix.float().argmax(dim=0)
        has_wrong = (~correct_matrix).any(dim=0)
        first_wrong_idx = (~correct_matrix).float().argmax(dim=0)

        rescue = (~text_correct).view(1, -1) & correct_matrix
        rescue[0] = False
        has_rescue = rescue.any(dim=0)
        first_rescue_idx = rescue.float().argmax(dim=0)

        damage = text_correct.view(1, -1) & ~correct_matrix
        damage[0] = False
        has_damage = damage.any(dim=0)
        first_damage_idx = damage.float().argmax(dim=0)

        last_safe_idx = torch.full_like(first_damage_idx, 100)
        last_safe_idx[has_damage] = torch.clamp(first_damage_idx[has_damage] - 1, min=0)

        num_flips = pred_matrix[1:].ne(pred_matrix[:-1]).sum(dim=0)
        features = {}
        features.update(self._stats("first_flip_alpha", alphas[first_flip_idx[has_flip]]))
        features.update(self._stats("first_correct_alpha", alphas[first_correct_idx[has_correct]]))
        features.update(self._stats("first_wrong_alpha", alphas[first_wrong_idx[has_wrong]]))
        features.update(self._stats("rescue_flip_alpha", alphas[first_rescue_idx[has_rescue]]))
        features.update(self._stats("damage_flip_alpha", alphas[first_damage_idx[has_damage]]))
        features.update(self._stats("last_safe_alpha", alphas[last_safe_idx[text_correct]]))
        features.update(self._stats("num_prediction_flips", num_flips.float()))

        text_correct_count = int(text_correct.sum().item())
        text_wrong_count = int((~text_correct).sum().item())
        features["damage_before_oracle_ratio"] = self._ratio((has_damage & (first_damage_idx <= oracle_idx)).sum().item(), text_correct_count)
        features["rescue_before_oracle_ratio"] = self._ratio((has_rescue & (first_rescue_idx <= oracle_idx)).sum().item(), text_wrong_count)
        features["damage_before_current_ratio"] = self._ratio((has_damage & (first_damage_idx <= current_idx)).sum().item(), text_correct_count)
        features["rescue_before_current_ratio"] = self._ratio((has_rescue & (first_rescue_idx <= current_idx)).sum().item(), text_wrong_count)

        per_query = {
            "first_flip_idx": first_flip_idx,
            "has_flip": has_flip,
            "first_correct_idx": first_correct_idx,
            "has_correct": has_correct,
            "first_wrong_idx": first_wrong_idx,
            "has_wrong": has_wrong,
            "first_rescue_idx": first_rescue_idx,
            "has_rescue": has_rescue,
            "first_damage_idx": first_damage_idx,
            "has_damage": has_damage,
            "last_safe_idx": last_safe_idx,
            "num_prediction_flips": num_flips,
        }
        return features, per_query

    def _ratio(self, numerator, denominator):
        denominator = int(denominator)
        if denominator <= 0:
            return None
        return float(numerator) / float(denominator)

    def _margin_cross_features(self, margin_matrix, endpoint, group_types, oracle_idx, current_idx):
        text_margin = endpoint["text_margin"].to(self.device)
        visual_margin = endpoint["visual_margin"].to(self.device)
        text_correct = endpoint["text_correct"].to(self.device)
        visual_correct = endpoint["visual_correct"].to(self.device)
        visual_conf = endpoint["visual_confidence"].to(self.device)
        visual_entropy = endpoint["visual_entropy"].to(self.device)
        text_entropy = endpoint["text_entropy"].to(self.device)
        group_tensor = torch.tensor(
            [0 if g == "both_correct" else 1 if g == "both_wrong" else 2 if g == "rescue_candidate" else 3 for g in group_types],
            device=self.device,
        )
        damage_candidate = group_tensor == 3
        rescue_candidate = group_tensor == 2

        features = {}
        features.update(self._stats("text_margin", text_margin))
        features.update(self._stats("visual_margin", visual_margin))
        features.update(self._stats("text_entropy", text_entropy))
        features.update(self._stats("visual_entropy", visual_entropy))
        features.update(self._stats("visual_confidence", visual_conf))

        group_masks = {
            "damage_candidate": damage_candidate,
            "rescue_candidate": rescue_candidate,
            "actually_damaged_by_current": text_correct & ~margin_matrix[current_idx].gt(0),
            "actually_damaged_by_oracle": text_correct & ~margin_matrix[oracle_idx].gt(0),
            "rescued_by_current": (~text_correct) & margin_matrix[current_idx].gt(0),
            "rescued_by_oracle": (~text_correct) & margin_matrix[oracle_idx].gt(0),
            "visual_correct": visual_correct,
            "visual_wrong": ~visual_correct,
        }
        for name, mask in group_masks.items():
            features.update(self._stats(f"{name}_text_margin", text_margin[mask]))
            features.update(self._stats(f"{name}_visual_margin", visual_margin[mask]))
            features[f"{name}_visual_conf_mean"] = self._float(visual_conf[mask].mean()) if mask.any() else None
            features[f"{name}_visual_entropy_mean"] = self._float(visual_entropy[mask].mean()) if mask.any() else None

        wrong_visual = ~visual_correct
        features["one_shot_visual_conf_wrong_high_ratio_08"] = self._ratio((wrong_visual & (visual_conf >= 0.8)).sum().item(), wrong_visual.sum().item())
        features["one_shot_visual_conf_wrong_high_ratio_09"] = self._ratio((wrong_visual & (visual_conf >= 0.9)).sum().item(), wrong_visual.sum().item())
        features["one_shot_damage_candidate_conf_mean"] = features["damage_candidate_visual_conf_mean"]
        features["one_shot_rescue_candidate_conf_mean"] = features["rescue_candidate_visual_conf_mean"]

        margin_delta = margin_matrix - text_margin.view(1, -1)
        margin_gain = torch.clamp(margin_delta, min=0.0)
        margin_drop = torch.clamp(-margin_delta, min=0.0)
        for idx in range(margin_matrix.size(0)):
            features[f"margin_gain_alpha_{idx:03d}"] = self._float(margin_gain[idx].mean())
            features[f"margin_drop_alpha_{idx:03d}"] = self._float(margin_drop[idx].mean())
            features[f"text_correct_margin_drop_alpha_{idx:03d}"] = self._float(margin_drop[idx, text_correct].mean()) if text_correct.any() else None
            features[f"text_wrong_margin_gain_alpha_{idx:03d}"] = self._float(margin_gain[idx, ~text_correct].mean()) if (~text_correct).any() else None
            features[f"damage_candidate_margin_drop_alpha_{idx:03d}"] = self._float(margin_drop[idx, damage_candidate].mean()) if damage_candidate.any() else None
            features[f"rescue_candidate_margin_gain_alpha_{idx:03d}"] = self._float(margin_gain[idx, rescue_candidate].mean()) if rescue_candidate.any() else None

        for name, idx in (("oracle", oracle_idx), ("current", current_idx)):
            features[f"{name}_text_correct_margin_drop_mean"] = self._float(margin_drop[idx, text_correct].mean()) if text_correct.any() else None
            features[f"{name}_text_wrong_margin_gain_mean"] = self._float(margin_gain[idx, ~text_correct].mean()) if (~text_correct).any() else None
            features[f"{name}_damage_candidate_margin_drop_mean"] = self._float(margin_drop[idx, damage_candidate].mean()) if damage_candidate.any() else None
            features[f"{name}_rescue_candidate_margin_gain_mean"] = self._float(margin_gain[idx, rescue_candidate].mean()) if rescue_candidate.any() else None
        return features

    def _source_features(self, T, V, train_features, train_labels, num_classes):
        train_labels = train_labels.to(self.device)
        train_norm = F.normalize(train_features.to(self.device), dim=-1)
        tv = V @ T.T
        tv_diag = tv.diag()
        tv_offdiag = tv.clone()
        tv_offdiag.fill_diagonal_(-float("inf"))
        tv_offdiag_max = tv_offdiag.max(dim=1).values
        tv_margin = tv_diag - tv_offdiag_max

        text_logits = train_norm @ T.T
        support_text_pred = text_logits.argmax(dim=-1)
        support_text_margin, _, _ = self._masked_margin(text_logits, train_labels)
        visual_sim = V @ V.T
        visual_sim.fill_diagonal_(-float("inf"))
        visual_nearest = visual_sim.max(dim=1).values

        features = {
            "source_tv_margin_mean": self._float(tv_margin.mean()),
            "source_tv_margin_min": self._float(tv_margin.min()),
            "source_tv_margin_neg_ratio": self._float((tv_margin < 0).float().mean()),
            "visual_centroid_text_top1_match_ratio": self._float((tv.argmax(dim=1) == torch.arange(num_classes, device=self.device)).float().mean()),
            "visual_nearest_sim_mean": self._float(visual_nearest.mean()),
            "visual_nearest_sim_max": self._float(visual_nearest.max()),
            "support_text_acc": self._float(support_text_pred.eq(train_labels).float().mean() * 100.0),
            "support_text_margin_mean": self._float(support_text_margin.mean()),
            "support_text_margin_neg_ratio": self._float((support_text_margin < 0).float().mean()),
        }
        return features, {
            "tv_diag": tv_diag,
            "tv_offdiag_max": tv_offdiag_max,
            "tv_margin": tv_margin,
            "support_text_margin": support_text_margin,
            "visual_nearest": visual_nearest,
        }

    def _per_class_damage(self, labels, endpoint, correct_matrix, source_tensors, num_classes, oracle_idx, current_idx):
        labels = labels.to(self.device)
        text_correct = endpoint["text_correct"].to(self.device)
        visual_correct = endpoint["visual_correct"].to(self.device)
        alphas = self._alpha_grid()
        rows = []
        class_alpha_max = []
        damage_ratios = []
        current_negative = 0
        oracle_negative = 0

        support_labels = labels.new_tensor([])
        for class_idx in range(num_classes):
            mask = labels == class_idx
            if not mask.any():
                continue
            rescued = ((~text_correct[mask]).view(1, -1) & correct_matrix[:, mask]).sum(dim=1).float()
            damaged = (text_correct[mask].view(1, -1) & ~correct_matrix[:, mask]).sum(dim=1).float()
            net = rescued - damaged
            candidate_damage = (text_correct[mask] & ~visual_correct[mask]).sum().item()
            candidate_rescue = ((~text_correct[mask]) & visual_correct[mask]).sum().item()
            ratio = self._ratio(candidate_damage, candidate_rescue)
            alpha_max_net = alphas[net.argmax()]
            class_alpha_max.append(alpha_max_net)
            if ratio is not None:
                damage_ratios.append(torch.tensor(ratio, device=self.device))
            if net[current_idx] < 0:
                current_negative += 1
            if net[oracle_idx] < 0:
                oracle_negative += 1
            damage_exceeds = damaged > rescued
            row = {
                "class_id": int(class_idx),
                "class_text_acc": self._float(text_correct[mask].float().mean() * 100.0),
                "class_visual_acc": self._float(visual_correct[mask].float().mean() * 100.0),
                "class_oracle_acc": self._float(correct_matrix[oracle_idx, mask].float().mean() * 100.0),
                "class_current_acc": self._float(correct_matrix[current_idx, mask].float().mean() * 100.0),
                "class_damage_candidate_count": int(candidate_damage),
                "class_rescue_candidate_count": int(candidate_rescue),
                "class_damage_rescue_ratio": ratio,
                "class_oracle_rescued": int(rescued[oracle_idx].item()),
                "class_oracle_damaged": int(damaged[oracle_idx].item()),
                "class_oracle_net": int(net[oracle_idx].item()),
                "class_current_rescued": int(rescued[current_idx].item()),
                "class_current_damaged": int(damaged[current_idx].item()),
                "class_current_net": int(net[current_idx].item()),
                "class_alpha_max_net": self._float(alpha_max_net),
                "class_alpha_damage_exceeds_rescue": self._float(alphas[damage_exceeds.float().argmax()]) if damage_exceeds.any() else None,
                "class_tv_margin": self._float(source_tensors["tv_margin"][class_idx]),
                "class_tv_diag": self._float(source_tensors["tv_diag"][class_idx]),
                "class_tv_offdiag_max": self._float(source_tensors["tv_offdiag_max"][class_idx]),
                "class_visual_sep": self._float(1.0 - source_tensors["visual_nearest"][class_idx]),
            }
            rows.append(row)

        alpha_tensor = torch.stack(class_alpha_max) if class_alpha_max else torch.empty(0, device=self.device)
        ratio_tensor = torch.stack(damage_ratios) if damage_ratios else torch.empty(0, device=self.device)
        features = {
            "class_net_std": self._float(torch.tensor([row["class_current_net"] for row in rows], dtype=torch.float32, device=self.device).std(unbiased=False)) if rows else None,
            "num_classes_current_net_negative": int(current_negative),
            "num_classes_oracle_net_negative": int(oracle_negative),
        }
        features.update(self._stats("class_alpha_max_net", alpha_tensor))
        features.update(self._stats("class_damage_rescue_ratio", ratio_tensor))

        class_tv_margin = torch.tensor([row["class_tv_margin"] for row in rows], dtype=torch.float32, device=self.device)
        class_compact_proxy = class_tv_margin
        class_damage_ratio = torch.tensor(
            [row["class_damage_rescue_ratio"] if row["class_damage_rescue_ratio"] is not None else 0.0 for row in rows],
            dtype=torch.float32,
            device=self.device,
        )
        class_alpha = torch.tensor([row["class_alpha_max_net"] for row in rows], dtype=torch.float32, device=self.device)
        features["corr_class_damage_ratio_with_tv_margin"] = self._corr(class_damage_ratio, class_tv_margin)
        features["corr_class_alpha_max_net_with_tv_margin"] = self._corr(class_alpha, class_tv_margin)
        return features, rows

    def _pair_rows(self, labels, endpoint, pred_matrix, T, V, oracle_idx, current_idx, limit=20):
        labels = labels.to(self.device)
        text_pred = endpoint["text_pred"].to(self.device)
        current_pred = pred_matrix[current_idx]
        oracle_pred = pred_matrix[oracle_idx]
        text_correct = endpoint["text_correct"].to(self.device)
        P_oracle = F.normalize((1 - oracle_idx * ALPHA_STEP) * T + (oracle_idx * ALPHA_STEP) * V, dim=-1)
        P_current = F.normalize((1 - current_idx * ALPHA_STEP) * T + (current_idx * ALPHA_STEP) * V, dim=-1)

        def collect(mask, pred_source):
            counts = {}
            for true_label, pred_label in zip(labels[mask].tolist(), pred_source[mask].tolist()):
                if true_label == pred_label:
                    continue
                pair = (int(true_label), int(pred_label))
                counts[pair] = counts.get(pair, 0) + 1
            rows = []
            for (c, k), count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]:
                rows.append({
                    "true": c,
                    "pred": k,
                    "count": int(count),
                    "T_c_T_k": self._float(T[c] @ T[k]),
                    "V_c_V_k": self._float(V[c] @ V[k]),
                    "P_oracle_c_P_oracle_k": self._float(P_oracle[c] @ P_oracle[k]),
                    "P_current_c_P_current_k": self._float(P_current[c] @ P_current[k]),
                    "V_c_T_c": self._float(V[c] @ T[c]),
                    "V_c_T_k": self._float(V[c] @ T[k]),
                    "T_c_V_k": self._float(T[c] @ V[k]),
                })
            return rows

        return {
            "top_damage_pairs_current": collect(text_correct & current_pred.ne(labels), current_pred),
            "top_damage_pairs_oracle": collect(text_correct & oracle_pred.ne(labels), oracle_pred),
            "top_rescue_pairs_current": collect((~text_correct) & current_pred.eq(labels), text_pred),
            "top_rescue_pairs_oracle": collect((~text_correct) & oracle_pred.eq(labels), text_pred),
        }

    def _query_rows(self, labels, endpoint, pred_matrix, correct_matrix, margin_matrix, group_types, flip_info, oracle_idx, current_idx):
        rows = []
        alphas = self._alpha_grid()
        for qid in range(labels.numel()):
            row = {
                "query_id": int(qid),
                "label": int(labels[qid].item()),
                "text_pred": int(endpoint["text_pred"][qid].item()),
                "visual_pred": int(endpoint["visual_pred"][qid].item()),
                "oracle_pred": int(pred_matrix[oracle_idx, qid].item()),
                "current_pred": int(pred_matrix[current_idx, qid].item()),
                "is_text_correct": bool(endpoint["text_correct"][qid].item()),
                "is_visual_correct": bool(endpoint["visual_correct"][qid].item()),
                "is_oracle_correct": bool(correct_matrix[oracle_idx, qid].item()),
                "is_current_correct": bool(correct_matrix[current_idx, qid].item()),
                "group_type": group_types[qid],
                "text_margin": self._float(endpoint["text_margin"][qid]),
                "visual_margin": self._float(endpoint["visual_margin"][qid]),
                "text_entropy": self._float(endpoint["text_entropy"][qid]),
                "visual_entropy": self._float(endpoint["visual_entropy"][qid]),
                "visual_confidence": self._float(endpoint["visual_confidence"][qid]),
                "first_flip_alpha": self._float(alphas[flip_info["first_flip_idx"][qid]]) if flip_info["has_flip"][qid] else None,
                "first_correct_alpha": self._float(alphas[flip_info["first_correct_idx"][qid]]) if flip_info["has_correct"][qid] else None,
                "first_wrong_alpha": self._float(alphas[flip_info["first_wrong_idx"][qid]]) if flip_info["has_wrong"][qid] else None,
                "first_rescue_alpha": self._float(alphas[flip_info["first_rescue_idx"][qid]]) if flip_info["has_rescue"][qid] else None,
                "first_damage_alpha": self._float(alphas[flip_info["first_damage_idx"][qid]]) if flip_info["has_damage"][qid] else None,
                "last_safe_alpha": self._float(alphas[flip_info["last_safe_idx"][qid]]) if endpoint["text_correct"][qid] else None,
                "num_prediction_flips": int(flip_info["num_prediction_flips"][qid].item()),
            }
            for idx in range(101):
                row[f"pred_alpha_{idx:03d}"] = int(pred_matrix[idx, qid].item())
                row[f"correct_alpha_{idx:03d}"] = bool(correct_matrix[idx, qid].item())
                row[f"margin_alpha_{idx:03d}"] = self._float(margin_matrix[idx, qid])
            rows.append(row)
        return rows

    def _heldout_damage(self, T, train_features, train_labels, num_classes, val_curves, oracle_alpha):
        train_labels = train_labels.to(self.device)
        class_indices = self._class_indices(train_labels.cpu(), num_classes)
        k = min(len(indices) for indices in class_indices)
        if k < 2:
            return {}, []

        alphas = self._alpha_grid()
        class_features = torch.stack([
            train_features[class_indices[class_idx][:k]].to(self.device)
            for class_idx in range(num_classes)
        ])
        target = torch.arange(num_classes, device=self.device)

        all_text_correct = []
        all_visual_correct = []
        all_correct_alpha = []
        all_margin_alpha = []
        rows = []

        for fold_idx in range(k):
            held = F.normalize(class_features[:, fold_idx, :], dim=-1)
            keep = torch.arange(k, device=self.device) != fold_idx
            V_minus = torch.stack([
                self.trainer._weighted_visual_centroid(class_features[class_idx, keep], T[class_idx])
                for class_idx in range(num_classes)
            ])
            text_logits = held @ T.T
            visual_logits = held @ V_minus.T
            text_correct = text_logits.argmax(dim=-1).eq(target)
            visual_correct = visual_logits.argmax(dim=-1).eq(target)
            correct_by_alpha = []
            margin_by_alpha = []
            for alpha in alphas:
                proto = F.normalize((1 - float(alpha.item())) * T + float(alpha.item()) * V_minus, dim=-1)
                logits = held @ proto.T
                margin, _, _ = self._masked_margin(logits, target)
                correct_by_alpha.append(logits.argmax(dim=-1).eq(target))
                margin_by_alpha.append(margin)
            all_text_correct.append(text_correct)
            all_visual_correct.append(visual_correct)
            all_correct_alpha.append(torch.stack(correct_by_alpha, dim=0))
            all_margin_alpha.append(torch.stack(margin_by_alpha, dim=0))

        text_correct = torch.cat(all_text_correct, dim=0)
        visual_correct = torch.cat(all_visual_correct, dim=0)
        correct_matrix = torch.cat(all_correct_alpha, dim=1)
        margin_matrix = torch.cat(all_margin_alpha, dim=1)
        rescued = ((~text_correct).view(1, -1) & correct_matrix).sum(dim=1).float()
        damaged = (text_correct.view(1, -1) & ~correct_matrix).sum(dim=1).float()
        net = rescued - damaged

        features = {
            "heldout_damage_candidate_count": int((text_correct & ~visual_correct).sum().item()),
            "heldout_rescue_candidate_count": int((~text_correct & visual_correct).sum().item()),
            "heldout_damage_rescue_ratio": self._ratio((text_correct & ~visual_correct).sum().item(), (~text_correct & visual_correct).sum().item()),
            "heldout_alpha_max_net": self._float(alphas[net.argmax()]),
            "heldout_abs_alpha_max_net_minus_oracle": abs(self._float(alphas[net.argmax()]) - oracle_alpha),
            "corr_val_net_with_heldout_net": self._corr(val_curves["net"], net),
            "corr_val_damage_with_heldout_damage": self._corr(val_curves["damage"], damaged),
            "corr_val_rescue_with_heldout_rescue": self._corr(val_curves["rescue"], rescued),
            "corr_val_damage_with_heldout_text_margin_drop": self._corr(val_curves["damage"], torch.clamp(-margin_matrix, min=0).mean(dim=1)),
            "corr_val_rescue_with_heldout_text_wrong_gain": self._corr(val_curves["rescue"], torch.clamp(margin_matrix, min=0).mean(dim=1)),
        }
        damage_exceeds = damaged > rescued
        features["heldout_alpha_damage_exceeds_rescue"] = self._float(alphas[damage_exceeds.float().argmax()]) if damage_exceeds.any() else None
        for lam in (1.0, 2.0, 3.0):
            score = rescued - lam * damaged
            key = self._safe_name(str(lam).replace(".", "_"))
            alpha = self._float(alphas[score.argmax()])
            features[f"heldout_best_alpha_net_lam_{key}"] = alpha
            features[f"heldout_abs_best_alpha_net_lam_{key}_minus_oracle"] = abs(alpha - oracle_alpha)
        for idx in range(101):
            features[f"heldout_rescue_alpha_{idx:03d}"] = int(rescued[idx].item())
            features[f"heldout_damage_alpha_{idx:03d}"] = int(damaged[idx].item())
            features[f"heldout_net_alpha_{idx:03d}"] = int(net[idx].item())
        return features, rows

    def _current_protofuse_metrics(self, T, V, train_features, train_labels, eval_features, eval_labels, num_classes, one_shot_mode):
        with torch.no_grad():
            result = self.trainer.hopc_alpha(T, V, train_features, train_labels, num_classes)
            proto, alpha = result[0], result[1]
            if one_shot_mode:
                logits = self._logits_for_alpha(float(alpha), T, V, eval_features, num_classes, True)
                preds = logits.argmax(dim=-1).cpu().tolist()
            else:
                eval_norm = F.normalize(eval_features.to(self.device), dim=-1)
                preds = (eval_norm @ proto.T).argmax(dim=-1).cpu().tolist()
        metrics = self._metric_dict(eval_labels.tolist(), preds)
        metrics["alpha"] = float(alpha)
        return metrics

    def _train_epochs(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before diagnosis.")

        train_features, train_labels = self._cached_train_features()
        val_features, val_labels = self._cached_val_features()
        remapped_train, remapped_val, num_classes = self._label_remap(train_labels, val_labels)
        shots_per_class = self._shots_per_class(remapped_train, num_classes)
        one_shot_mode = shots_per_class < 2

        logger.info(
            f"Running damage forensics: alpha=0.00..1.00, step={ALPHA_STEP:.2f}, "
            f"classes={num_classes}, shots/class={shots_per_class}"
        )

        V = self.trainer.build_visual_centroids(train_features, remapped_train, num_classes)
        T = self.trainer.text_prototypes
        endpoint = self._pure_endpoint_state(T, V, val_features, remapped_val, num_classes)
        pred_matrix, correct_matrix, margin_matrix = self._pred_margin_matrices(
            T, V, val_features, remapped_val, num_classes, one_shot_mode
        )
        sweep = self._sweep_metrics(pred_matrix, remapped_val)
        best = self._best_from_sweep(sweep)
        oracle_alpha = float(best["alpha"])
        oracle_idx = max(0, min(100, int(round(oracle_alpha / ALPHA_STEP))))
        current_metrics = self._current_protofuse_metrics(
            T, V, train_features, remapped_train, val_features, remapped_val, num_classes, one_shot_mode
        )
        current_alpha = float(current_metrics["alpha"])
        current_idx = max(0, min(100, int(round(current_alpha / ALPHA_STEP))))

        self.best_val_acc = float(best["accuracy"])
        self.trainer.best_alpha = oracle_alpha
        self.trainer.fused_prototypes = F.normalize((1 - oracle_alpha) * T + oracle_alpha * V, dim=-1)

        group_types = self._group_types(endpoint)
        source_features, source_tensors = self._source_features(T, V, train_features, remapped_train, num_classes)
        curve_features, rescue_curve, damage_curve, net_curve, _, _ = self._transition_curves(correct_matrix, endpoint)
        flip_features, flip_info = self._flip_features(pred_matrix, correct_matrix, endpoint, oracle_idx, current_idx)
        margin_features = self._margin_cross_features(margin_matrix, endpoint, group_types, oracle_idx, current_idx)
        class_features, class_rows = self._per_class_damage(
            remapped_val, endpoint, correct_matrix, source_tensors, num_classes, oracle_idx, current_idx
        )
        pair_rows = self._pair_rows(remapped_val, endpoint, pred_matrix, T, V, oracle_idx, current_idx)
        heldout_features, heldout_rows = self._heldout_damage(
            T,
            train_features,
            remapped_train,
            num_classes,
            {"rescue": rescue_curve, "damage": damage_curve, "net": net_curve},
            oracle_alpha,
        )
        query_rows = self._query_rows(
            remapped_val.to(self.device), endpoint, pred_matrix, correct_matrix, margin_matrix, group_types, flip_info, oracle_idx, current_idx
        )

        labels_list = remapped_val.tolist()
        text_metrics = self._metric_dict(labels_list, endpoint["text_pred"].cpu().tolist())
        visual_metrics = self._metric_dict(labels_list, endpoint["visual_pred"].cpu().tolist())
        oracle_metrics = self._metric_dict(labels_list, pred_matrix[oracle_idx].cpu().tolist())

        forensics = {
            "dataset": self.config.data.dataset_name,
            "seed": int(self.seed),
            "shot": int(self.kshot),
            "kshot": int(self.kshot),
            "num_classes": int(num_classes),
            "shots_per_class": int(shots_per_class),
            "one_shot_mode": bool(one_shot_mode),
            "alpha_oracle": oracle_alpha,
            "acc_oracle": float(best["accuracy"]),
            "mca_oracle": float(best.get("mca", 0.0)),
            "alpha_current": current_alpha,
            "acc_current": float(current_metrics.get("accuracy", 0.0)),
            "mca_current": float(current_metrics.get("mca", 0.0)),
            "gap": float(best["accuracy"] - current_metrics.get("accuracy", 0.0)),
            "A_T": text_metrics.get("accuracy"),
            "A_V": visual_metrics.get("accuracy"),
            "Delta_VT": visual_metrics.get("accuracy", 0.0) - text_metrics.get("accuracy", 0.0),
            "oracle_rescued": int(rescue_curve[oracle_idx].item()),
            "oracle_damaged": int(damage_curve[oracle_idx].item()),
            "oracle_net": int(net_curve[oracle_idx].item()),
            "current_rescued": int(rescue_curve[current_idx].item()),
            "current_damaged": int(damage_curve[current_idx].item()),
            "current_net": int(net_curve[current_idx].item()),
            "damage_rescue_ratio_current": self._ratio(damage_curve[current_idx].item(), rescue_curve[current_idx].item()),
            "damage_rescue_ratio_oracle": self._ratio(damage_curve[oracle_idx].item(), rescue_curve[oracle_idx].item()),
        }
        forensics.update(source_features)
        forensics.update(curve_features)
        forensics.update(flip_features)
        forensics.update(margin_features)
        forensics.update(class_features)
        forensics.update(heldout_features)
        forensics["corr_alpha_max_net_with_oracle"] = None
        forensics["corr_class_damage_ratio_with_support_text_margin"] = None

        summary = {
            "alpha_step": ALPHA_STEP,
            "best_alpha": oracle_alpha,
            "best_accuracy": float(best["accuracy"]),
            "best_mca": float(best.get("mca", 0.0)),
            "current_alpha": current_alpha,
            "current_accuracy": float(current_metrics.get("accuracy", 0.0)),
            "current_mca": float(current_metrics.get("mca", 0.0)),
            "gap": float(best["accuracy"] - current_metrics.get("accuracy", 0.0)),
            "text_metrics": text_metrics,
            "visual_metrics": visual_metrics,
            "oracle_metrics": oracle_metrics,
            "current_metrics": current_metrics,
            "forensics": forensics,
            "sweep": sweep,
            "query_damage_table": query_rows,
            "class_damage_table": class_rows,
            "pairwise_damage_confusion": pair_rows,
            "heldout_damage_table": heldout_rows,
        }
        self.metrics.append(summary)

        repo_root = Path(__file__).resolve().parent.parent
        dataset_key = self._safe_name(self.config.data.dataset_name)
        sweep_path = repo_root / f"damage_sweep_{dataset_key}_{self.kshot}shot_seed{self.seed}.json"
        forensics_path = repo_root / f"damage_forensics_{self.kshot}shot.jsonl"
        with open(sweep_path, "w") as f:
            json.dump(summary, f, indent=2)
        with open(forensics_path, "a") as f:
            f.write(json.dumps(forensics, sort_keys=True) + "\n")

        logger.info(
            f"Oracle alpha={oracle_alpha:.2f}, rescue={forensics['oracle_rescued']}, "
            f"damage={forensics['oracle_damaged']}, net={forensics['oracle_net']}"
        )
        logger.info(
            f"Current alpha={current_alpha:.2f}, rescue={forensics['current_rescued']}, "
            f"damage={forensics['current_damaged']}, net={forensics['current_net']}"
        )
        logger.info(f"Damage sweep saved to: {sweep_path}")
        logger.info(f"Damage forensics appended to: {forensics_path}")
        log_experiment_metrics(best, title=self._metrics_title())


def parse_args():
    parsed, unknown = create_argument_parser("Run ProtoFuse damage forensics", ARG_SCHEMA).parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, process_parsed_args(parsed, ARG_SCHEMA, overrides)


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, "debug", True), getattr(args, "disable_coloring", True))
    config = merge_configs(load_config_file(args.config), overrides)
    if getattr(args, "output_dir", None) is None:
        config.setdefault("logging", {})
        config["logging"]["output_dir"] = ProtoFuseDamageForensicsPipeline.DEFAULT_OUTPUT_DIR
    pipeline = ProtoFuseDamageForensicsPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
