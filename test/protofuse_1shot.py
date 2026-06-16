import json
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
    run_for_dataset_configs,
    setup_logging,
)
from src.pipelines.protofuse import ProtoFusePipeline


ARG_SCHEMA = DEFAULT_ARG_SCHEMA


class ProtoFuseOneShotCentroidMixCeilingPipeline(ProtoFusePipeline):
    METHOD_NAME = "ProtoFuse 1-Shot Knee Selector Log"
    DEFAULT_OUTPUT_DIR = "outputs/protofuse_1shot"

    def __init__(self, config):
        super().__init__(config)
        output_dir = self.logging_cfg.get("output_dir", self.DEFAULT_OUTPUT_DIR)
        self.run_dir = str(output_dir)
        self.config_path = os.path.join(self.run_dir, "config.json")
        self.metrics_path = os.path.join(
            self.run_dir,
            f"metrics_{self._safe_name(self.config.data.dataset_name)}_{self.kshot}shot_seed{self.seed}.json",
        )
        self.best_model_path = os.path.join(self.run_dir, "best.pt")
        self.last_model_path = os.path.join(self.run_dir, "last.pt")
        self.eda_dir = os.path.join(self.run_dir, "eda")
        logger.info(f"Run directory: {self.run_dir}")

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

    def _tensor_list(self, value):
        return [self._float(v) for v in value.detach().cpu().flatten()]

    def _cfg(self):
        model_cfg = self.config.get("model", {})
        return model_cfg.get("oneshot_ceiling", self.config.get("oneshot_ceiling", {}))

    def _cfg_list(self, key, default):
        value = self._cfg().get(key, default)
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _alpha_grid(self):
        return torch.linspace(0.0, 1.0, 101, device=self.device)

    def _label_remap(self, train_labels, eval_labels):
        task_classes = sorted(set(train_labels.tolist()))
        remap = {class_idx: idx for idx, class_idx in enumerate(task_classes)}
        missing = sorted(set(eval_labels.tolist()) - set(remap.keys()))
        if missing:
            raise ValueError(f"Eval labels not present in train split: {missing[:10]}")
        remapped_train = torch.tensor([remap[label.item()] for label in train_labels], dtype=torch.long)
        remapped_eval = torch.tensor([remap[label.item()] for label in eval_labels], dtype=torch.long)
        return remapped_train, remapped_eval, len(task_classes)

    def _shots_per_class(self, labels, num_classes):
        counts = torch.bincount(labels, minlength=num_classes)
        return int(counts.min().item())

    def _metric_dict(self, labels, preds):
        return {key: float(value) for key, value in compute_metrics(labels, preds).items()}

    def _accuracy(self, preds, labels):
        return 100.0 * preds.eq(labels).float().mean().item()

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

    def _masked_margin(self, logits, labels):
        labels = labels.to(logits.device)
        source_score = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        other_logits = logits.clone()
        other_logits.scatter_(1, labels.view(-1, 1), -float("inf"))
        return source_score - other_logits.max(dim=1).values

    def _neighbor_indices(self, T, V, mode, top_l):
        if mode == "vv":
            score = V @ V.T
        elif mode == "tt":
            score = T @ T.T
        elif mode == "hybrid":
            score = 0.5 * (V @ V.T) + 0.5 * (T @ T.T)
        else:
            raise ValueError(f"Unknown neighbor_score: {mode}")

        score = score.clone()
        score.fill_diagonal_(-float("inf"))
        if top_l == "all":
            k = score.shape[0] - 1
        else:
            k = int(top_l)
        k = min(max(k, 1), score.shape[0] - 1)
        return score.topk(k, dim=1).indices

    def _beta_configs(self, beta_modes, beta_values):
        configs = []
        for mode in beta_modes:
            if mode == "single":
                configs.extend((mode, [beta]) for beta in beta_values)
            elif mode == "cumulative":
                for idx in range(len(beta_values)):
                    configs.append((mode, beta_values[:idx + 1]))
            else:
                raise ValueError(f"Unknown beta_mode: {mode}")
        return configs

    def _weight_for_beta(self, beta, mode):
        if mode == "none":
            return 1.0
        if mode == "linear":
            return 1.0 - 2.0 * beta
        if mode == "quad":
            return (1.0 - 2.0 * beta) ** 2
        raise ValueError(f"Unknown weight_mode: {mode}")

    def _build_pseudo(self, V, neighbors, beta_grid):
        top_l = neighbors.shape[1]
        source_features = V[:, None, :].expand(-1, top_l, -1).reshape(-1, V.shape[-1])
        neighbor_features = V[neighbors.reshape(-1)]
        labels = torch.arange(V.shape[0], device=self.device).repeat_interleave(top_l)

        pseudo_features = []
        pseudo_labels = []
        pseudo_betas = []
        for beta in beta_grid:
            mixed = F.normalize((1.0 - beta) * source_features + beta * neighbor_features, dim=-1)
            pseudo_features.append(mixed)
            pseudo_labels.append(labels)
            pseudo_betas.append(torch.full((mixed.shape[0],), beta, device=self.device))

        return (
            torch.cat(pseudo_features, dim=0),
            torch.cat(pseudo_labels, dim=0),
            torch.cat(pseudo_betas, dim=0),
        )

    def _real_alpha_sweep(self, T, V, eval_features, eval_labels):
        eval_features = F.normalize(eval_features.to(self.device), dim=-1)
        eval_labels = eval_labels.to(self.device)
        alphas = self._alpha_grid()
        preds_by_alpha = []
        acc_by_alpha = []

        for alpha in alphas:
            prototypes = F.normalize((1.0 - alpha) * T + alpha * V, dim=-1)
            preds = (eval_features @ prototypes.T).argmax(dim=-1)
            preds_by_alpha.append(preds)
            acc_by_alpha.append(self._accuracy(preds, eval_labels))

        pred_matrix = torch.stack(preds_by_alpha, dim=0)
        acc_tensor = torch.tensor(acc_by_alpha, dtype=torch.float32, device=self.device)
        correct_matrix = pred_matrix.eq(eval_labels.view(1, -1))
        text_correct = correct_matrix[0]
        rescue = ((~text_correct).view(1, -1) & correct_matrix).sum(dim=1).float()
        damage = (text_correct.view(1, -1) & ~correct_matrix).sum(dim=1).float()
        net = rescue - damage
        transition = self._transition_curves(correct_matrix, text_correct)
        return {
            "alphas": alphas,
            "pred_matrix": pred_matrix,
            "acc": acc_tensor,
            "rescue": rescue,
            "damage": damage,
            "net": net,
            **transition,
        }

    def _pseudo_alpha_sweep_base(self, T, V, pseudo_features, pseudo_labels):
        text_preds = (pseudo_features @ T.T).argmax(dim=-1)
        text_correct = text_preds.eq(pseudo_labels)
        text_acc = self._accuracy(text_preds, pseudo_labels)
        margins = self._masked_margin(pseudo_features @ T.T, pseudo_labels)

        rescue_indicators = []
        damage_indicators = []
        correct_indicators = []
        for alpha in self._alpha_grid():
            prototypes = F.normalize((1.0 - alpha) * T + alpha * V, dim=-1)
            preds = (pseudo_features @ prototypes.T).argmax(dim=-1)
            correct = preds.eq(pseudo_labels)
            correct_indicators.append(correct.float())
            rescue_indicators.append(((~text_correct) & correct).float())
            damage_indicators.append((text_correct & ~correct).float())

        return {
            "pseudo_text_acc": text_acc,
            "text_correct_count": int(text_correct.sum().item()),
            "text_wrong_count": int((~text_correct).sum().item()),
            "pseudo_source_margin_mean": self._float(margins.mean()),
            "pseudo_source_margin_std": self._float(margins.std(unbiased=False)),
            "pseudo_source_margin_neg_ratio": self._float((margins < 0).float().mean()),
            "text_correct": text_correct,
            "correct_indicators": torch.stack(correct_indicators),
            "rescue_indicators": torch.stack(rescue_indicators),
            "damage_indicators": torch.stack(damage_indicators),
        }

    def _apply_pseudo_weights(self, pseudo_base, pseudo_weights):
        rescue = (pseudo_base["rescue_indicators"] * pseudo_weights.view(1, -1)).sum(dim=1)
        damage = (pseudo_base["damage_indicators"] * pseudo_weights.view(1, -1)).sum(dim=1)
        return rescue, damage, rescue - damage

    def _transition_curves(self, correct_matrix, text_correct):
        text_correct = text_correct.bool()
        fused_correct = correct_matrix.bool()
        text_correct_count = int(text_correct.sum().item())
        text_wrong_count = int((~text_correct).sum().item())
        tc_fc = (text_correct.view(1, -1) & fused_correct).sum(dim=1).float()
        tc_fw = (text_correct.view(1, -1) & ~fused_correct).sum(dim=1).float()
        tw_fc = ((~text_correct).view(1, -1) & fused_correct).sum(dim=1).float()
        tw_fw = ((~text_correct).view(1, -1) & ~fused_correct).sum(dim=1).float()
        damage_rate = tc_fw / max(1, text_correct_count)
        rescue_rate = tw_fc / max(1, text_wrong_count)
        return {
            "text_correct_count": text_correct_count,
            "text_wrong_count": text_wrong_count,
            "T_correct_F_correct": tc_fc,
            "T_correct_F_wrong": tc_fw,
            "T_wrong_F_correct": tw_fc,
            "T_wrong_F_wrong": tw_fw,
            "damage_rate": damage_rate,
            "rescue_rate": rescue_rate,
        }

    def _pseudo_transition_curves(self, pseudo_base):
        correct_matrix = pseudo_base["correct_indicators"].bool()
        return self._transition_curves(correct_matrix, pseudo_base["text_correct"])

    def _landmarks(self, alphas, net_curve, acc_curve, damage_curve, rescue_rate_curve):
        max_net = net_curve.max()
        plateau = torch.nonzero(torch.isclose(net_curve, max_net), as_tuple=False).flatten()
        first_damage = torch.nonzero(damage_curve > 0, as_tuple=False).flatten()
        max_rescue_idx = int(rescue_rate_curve.argmax().item())
        return {
            "alpha_max_net": self._float(alphas[int(net_curve.argmax().item())]),
            "alpha_max_acc": self._float(alphas[int(acc_curve.argmax().item())]),
            "alpha_first_damage": None if first_damage.numel() == 0 else self._float(alphas[int(first_damage[0].item())]),
            "alpha_max_rescue_rate": self._float(alphas[max_rescue_idx]),
            "net_max": self._float(max_net),
            "net_plateau_alphas": [self._float(alphas[int(idx.item())]) for idx in plateau],
        }

    def _selected_position_in_plateau(self, selected_alpha, plateau_alphas):
        if not plateau_alphas:
            return None
        for idx, alpha in enumerate(plateau_alphas):
            if abs(float(alpha) - float(selected_alpha)) < 1e-8:
                return idx
        return None

    def _pseudo_validity_by_beta(self, T, pseudo_features, pseudo_labels, pseudo_betas, beta_grid):
        logits = pseudo_features @ T.T
        preds = logits.argmax(dim=-1)
        source_margin = self._masked_margin(logits, pseudo_labels)
        top2 = logits.topk(2, dim=1).values
        top2_margin = top2[:, 0] - top2[:, 1]
        result = {}
        for beta in beta_grid:
            mask = pseudo_betas.eq(beta)
            key = f"{beta:.2f}"
            if not mask.any():
                continue
            result[key] = {
                "pseudo_text_acc": self._accuracy(preds[mask], pseudo_labels[mask]),
                "pseudo_source_margin_mean": self._float(source_margin[mask].mean()),
                "pseudo_source_margin_neg_ratio": self._float((source_margin[mask] < 0).float().mean()),
                "pseudo_nearest_class_is_source_ratio": self._float(preds[mask].eq(pseudo_labels[mask]).float().mean()),
                "pseudo_top2_margin": self._float(top2_margin[mask].mean()),
            }
        return result

    def _neighbor_quality(self, T, V, neighbors, mode, test_text_preds, test_labels):
        if mode == "vv":
            score = V @ V.T
        elif mode == "tt":
            score = T @ T.T
        else:
            score = 0.5 * (V @ V.T) + 0.5 * (T @ T.T)
        selected_scores = score.gather(1, neighbors)

        vv = V @ V.T
        tt = T @ T.T
        vv.fill_diagonal_(-float("inf"))
        tt.fill_diagonal_(-float("inf"))
        vv_top = vv.topk(neighbors.shape[1], dim=1).indices
        tt_top = tt.topk(neighbors.shape[1], dim=1).indices
        agreement = []
        for class_idx in range(neighbors.shape[0]):
            vv_set = set(vv_top[class_idx].tolist())
            tt_set = set(tt_top[class_idx].tolist())
            agreement.append(len(vv_set & tt_set) / max(1, neighbors.shape[1]))

        wrong = test_text_preds.ne(test_labels)
        if wrong.any():
            target_neighbors = neighbors[test_labels[wrong]]
            error_targets = test_text_preds[wrong].view(-1, 1)
            error_target_ratio = target_neighbors.eq(error_targets).any(dim=1).float().mean()
        else:
            error_target_ratio = None

        return {
            "neighbor_similarity_mean": self._float(selected_scores.mean()),
            "neighbor_similarity_std": self._float(selected_scores.std(unbiased=False)),
            "neighbor_rank_agreement_vv_tt": float(sum(agreement) / max(1, len(agreement))),
            "neighbor_is_text_pred_error_target_ratio": self._float(error_target_ratio),
        }

    def _normalize_curve(self, curve):
        curve = curve.float()
        min_value = curve.min()
        max_value = curve.max()
        denom = max_value - min_value
        if self._float(denom) == 0.0:
            return torch.zeros_like(curve)
        return (curve - min_value) / (denom + 1e-12)

    def _knee_index(self, curve):
        y = self._normalize_curve(curve)
        x = torch.linspace(0.0, 1.0, len(y), device=y.device)
        score = y - x
        idx = int(score.argmax().item())
        return idx, self._float(score[idx])

    def _curvature_index(self, curve):
        y = self._normalize_curve(curve)
        if y.numel() < 3:
            return 0
        second = y[2:] - 2.0 * y[1:-1] + y[:-2]
        return int(second.argmax().item()) + 1

    def _curve_shape(self, alphas, curve, selected_idx):
        y = self._normalize_curve(curve)
        max_value = curve.max()
        selected_value = curve[selected_idx]
        plateau = torch.nonzero(torch.isclose(curve, max_value), as_tuple=False).flatten()
        plateau_start = int(plateau[0].item()) if plateau.numel() else int(curve.argmax().item())
        diffs = curve[1:] - curve[:-1]
        return {
            "pseudo_net_at_selected": self._float(selected_value),
            "pseudo_net_max": self._float(max_value),
            "effective_rho": None if self._float(max_value) == 0.0 else self._float(selected_value / (max_value + 1e-12)),
            "curve_auc": self._float(y.mean()),
            "curve_monotonicity": self._float((diffs >= -1e-8).float().mean()) if diffs.numel() else None,
            "plateau_start_alpha": self._float(alphas[plateau_start]),
            "plateau_width": int(plateau.numel()),
        }

    def _support_reliability(self, T, V):
        logits = V @ T.T
        labels = torch.arange(V.shape[0], device=self.device)
        tv_alignment = logits.diag()
        tv_margin = self._masked_margin(logits, labels)
        return {
            "tv_alignment_mean": self._float(tv_alignment.mean()),
            "tv_margin_mean": self._float(tv_margin.mean()),
            "tv_margin_neg_ratio": self._float((tv_margin < 0).float().mean()),
        }

    def _real_at_index(self, real, idx):
        return {
            "real_acc": self._float(real["acc"][idx]),
            "real_rescue": self._float(real["rescue"][idx]),
            "real_damage": self._float(real["damage"][idx]),
            "real_net": self._float(real["net"][idx]),
            "real_damage_rate": self._float(real["damage_rate"][idx]),
            "real_rescue_rate": self._float(real["rescue_rate"][idx]),
        }

    def _nearest_alpha_idx(self, alphas, alpha):
        return int((alphas - float(alpha)).abs().argmin().item())

    def _current_metrics(self, T, V, train_features, train_labels, eval_features, eval_labels, num_classes):
        proto, alpha = self.trainer.hopc_alpha(T, V, train_features, train_labels, num_classes)
        eval_features = F.normalize(eval_features.to(self.device), dim=-1)
        preds = (eval_features @ proto.T).argmax(dim=-1)
        metrics = self._metric_dict(eval_labels.tolist(), preds.cpu().tolist())
        metrics["selected_alpha"] = float(alpha)
        return metrics

    def _aggregate_best(self, per_config, key, value_key=None):
        grouped = {}
        for row in per_config:
            value = row[key] if value_key is None else row[value_key]
            grouped.setdefault(str(value), row["acc"])
            grouped[str(value)] = max(grouped[str(value)], row["acc"])
        best_key, best_acc = max(grouped.items(), key=lambda item: item[1])
        return best_key, best_acc, grouped

    def _train_epochs(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before one-shot knee-selector log.")

        train_features, train_labels = self._cached_train_features()
        test_features, test_labels = self._cached_test_features()
        remapped_train, remapped_test, num_classes = self._label_remap(train_labels, test_labels)
        shots_per_class = self._shots_per_class(remapped_train, num_classes)
        if shots_per_class != 1:
            raise ValueError(f"Expected one-shot data, got shots_per_class={shots_per_class}.")

        T = self.trainer.text_prototypes
        V = self.trainer.build_visual_centroids(train_features, remapped_train, num_classes)
        test_norm = F.normalize(test_features.to(self.device), dim=-1)

        text_preds = (test_norm @ T.T).argmax(dim=-1)
        visual_preds = (test_norm @ V.T).argmax(dim=-1)
        text_metrics = self._metric_dict(remapped_test.tolist(), text_preds.cpu().tolist())
        visual_metrics = self._metric_dict(remapped_test.tolist(), visual_preds.cpu().tolist())
        current_metrics = self._current_metrics(
            T, V, train_features, remapped_train, test_features, remapped_test, num_classes
        )
        real = self._real_alpha_sweep(T, V, test_features, remapped_test)

        beta_values = [float(v) for v in self._cfg_list(
            "beta_values",
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        )]
        beta_values = [b for b in beta_values if 0.0 < b < 0.5]
        neighbor_scores = ["vv", "tt", "hybrid"]
        top_l = 1
        logger.info("Running one-shot parameter-free knee selector log on test")

        dataset_key = self._safe_name(self.config.data.dataset_name)
        self.summary_path = os.path.join(
            self.run_dir,
            f"oneshot_knee_selector_summary_{dataset_key}_seed{self.seed}.json",
        )
        self.curves_path = os.path.join(
            self.run_dir,
            f"oneshot_knee_selector_curves_{dataset_key}_seed{self.seed}.jsonl",
        )

        alphas = real["alphas"]
        alpha_grid = self._tensor_list(alphas)
        oracle_idx = int(real["acc"].argmax().item())
        oracle_alpha = self._float(alphas[oracle_idx])
        oracle_acc = self._float(real["acc"][oracle_idx])
        support_reliability = self._support_reliability(T, V)

        configs = []
        curve_rows = []
        config_id = 0
        for neighbor_score in neighbor_scores:
            neighbors = self._neighbor_indices(T, V, neighbor_score, top_l)
            neighbor_quality = self._neighbor_quality(
                T,
                V,
                neighbors,
                neighbor_score,
                text_preds.to(self.device),
                remapped_test.to(self.device),
            )
            for beta in beta_values:
                pseudo_features, pseudo_labels, pseudo_betas = self._build_pseudo(V, neighbors, [beta])
                pseudo_base = self._pseudo_alpha_sweep_base(T, V, pseudo_features, pseudo_labels)
                pseudo_transition = self._pseudo_transition_curves(pseudo_base)
                pseudo_rescue = pseudo_base["rescue_indicators"].sum(dim=1)
                pseudo_damage = pseudo_base["damage_indicators"].sum(dim=1)
                pseudo_net = pseudo_rescue - pseudo_damage
                pseudo_acc = pseudo_base["correct_indicators"].mean(dim=1) * 100.0
                knee_idx, knee_strength = self._knee_index(pseudo_net)
                maxnet_idx = int(pseudo_net.argmax().item())
                curvature_idx = self._curvature_index(pseudo_net)
                shape = self._curve_shape(alphas, pseudo_net, knee_idx)
                pseudo_validity_by_beta = self._pseudo_validity_by_beta(T, pseudo_features, pseudo_labels, pseudo_betas, [beta])

                config = {
                    "config_id": config_id,
                    "curve_type": "config",
                    "beta": float(beta),
                    "neighbor": neighbor_score,
                    "topL": top_l,
                    "pseudo_rescue": pseudo_rescue,
                    "pseudo_damage": pseudo_damage,
                    "pseudo_net": pseudo_net,
                    "pseudo_acc": pseudo_acc,
                    "pseudo_transition": pseudo_transition,
                    "alpha_maxnet": self._float(alphas[maxnet_idx]),
                    "alpha_knee": self._float(alphas[knee_idx]),
                    "alpha_curvature": self._float(alphas[curvature_idx]),
                    "acc_knee": self._float(real["acc"][knee_idx]),
                    "maxnet_idx": maxnet_idx,
                    "knee_idx": knee_idx,
                    "curvature_idx": curvature_idx,
                    "knee_strength": knee_strength,
                    "pseudo_text_acc": pseudo_base["pseudo_text_acc"],
                    "pseudo_margin_mean": pseudo_base["pseudo_source_margin_mean"],
                    "pseudo_margin_neg_ratio": pseudo_base["pseudo_source_margin_neg_ratio"],
                    "pseudo_top2_margin_mean": list(pseudo_validity_by_beta.values())[0]["pseudo_top2_margin"],
                    **shape,
                    "effective_rho_knee": shape["effective_rho"],
                    **neighbor_quality,
                    **support_reliability,
                    "corr_pseudo_net_real_net": self._corr(pseudo_net, real["net"]),
                    "corr_pseudo_acc_real_acc": self._corr(pseudo_acc, real["acc"]),
                }
                configs.append(config)

                curve_rows.append({
                    **{k: v for k, v in config.items() if not isinstance(v, torch.Tensor) and k != "pseudo_transition"},
                    "alpha_grid": alpha_grid,
                    "pseudo_rescue_curve": self._tensor_list(pseudo_rescue),
                    "pseudo_damage_curve": self._tensor_list(pseudo_damage),
                    "pseudo_net_curve": self._tensor_list(pseudo_net),
                    "pseudo_acc_curve": self._tensor_list(pseudo_acc),
                    "real_rescue_curve": self._tensor_list(real["rescue"]),
                    "real_damage_curve": self._tensor_list(real["damage"]),
                    "real_net_curve": self._tensor_list(real["net"]),
                    "real_acc_curve": self._tensor_list(real["acc"]),
                    "pseudo_T_correct_F_correct": self._tensor_list(pseudo_transition["T_correct_F_correct"]),
                    "pseudo_T_correct_F_wrong": self._tensor_list(pseudo_transition["T_correct_F_wrong"]),
                    "pseudo_T_wrong_F_correct": self._tensor_list(pseudo_transition["T_wrong_F_correct"]),
                    "pseudo_T_wrong_F_wrong": self._tensor_list(pseudo_transition["T_wrong_F_wrong"]),
                    "pseudo_rescue_rate": self._tensor_list(pseudo_transition["rescue_rate"]),
                    "pseudo_damage_rate": self._tensor_list(pseudo_transition["damage_rate"]),
                    "pseudo_text_correct_count": pseudo_transition["text_correct_count"],
                    "pseudo_text_wrong_count": pseudo_transition["text_wrong_count"],
                    "real_T_correct_F_correct": self._tensor_list(real["T_correct_F_correct"]),
                    "real_T_correct_F_wrong": self._tensor_list(real["T_correct_F_wrong"]),
                    "real_T_wrong_F_correct": self._tensor_list(real["T_wrong_F_correct"]),
                    "real_T_wrong_F_wrong": self._tensor_list(real["T_wrong_F_wrong"]),
                    "real_damage_rate_curve": self._tensor_list(real["damage_rate"]),
                    "real_rescue_rate_curve": self._tensor_list(real["rescue_rate"]),
                })
                config_id += 1

        def aggregate_curve(name, members):
            if not members:
                return None
            norm_net = torch.stack([self._normalize_curve(row["pseudo_net"]) for row in members], dim=0)
            net = norm_net.mean(dim=0)
            rescue = torch.stack([self._normalize_curve(row["pseudo_rescue"]) for row in members], dim=0).mean(dim=0)
            damage = torch.stack([self._normalize_curve(row["pseudo_damage"]) for row in members], dim=0).mean(dim=0)
            acc = torch.stack([row["pseudo_acc"] for row in members], dim=0).mean(dim=0)
            knee_idx, knee_strength = self._knee_index(net)
            maxnet_idx = int(net.argmax().item())
            curvature_idx = self._curvature_index(net)
            shape = self._curve_shape(alphas, net, knee_idx)
            record = {
                "config_id": name,
                "curve_type": "aggregate",
                "beta": None,
                "neighbor": None,
                "topL": top_l,
                "member_count": len(members),
                "pseudo_rescue": rescue,
                "pseudo_damage": damage,
                "pseudo_net": net,
                "pseudo_acc": acc,
                "alpha_maxnet": self._float(alphas[maxnet_idx]),
                "alpha_knee": self._float(alphas[knee_idx]),
                "alpha_curvature": self._float(alphas[curvature_idx]),
                "acc_knee": self._float(real["acc"][knee_idx]),
                "maxnet_idx": maxnet_idx,
                "knee_idx": knee_idx,
                "curvature_idx": curvature_idx,
                "knee_strength": knee_strength,
                **shape,
                "effective_rho_knee": shape["effective_rho"],
                "corr_pseudo_net_real_net": self._corr(net, real["net"]),
                "corr_pseudo_acc_real_acc": self._corr(acc, real["acc"]),
            }
            curve_rows.append({
                **{k: v for k, v in record.items() if not isinstance(v, torch.Tensor)},
                "alpha_grid": alpha_grid,
                "pseudo_rescue_curve": self._tensor_list(rescue),
                "pseudo_damage_curve": self._tensor_list(damage),
                "pseudo_net_curve": self._tensor_list(net),
                "pseudo_acc_curve": self._tensor_list(acc),
                "real_rescue_curve": self._tensor_list(real["rescue"]),
                "real_damage_curve": self._tensor_list(real["damage"]),
                "real_net_curve": self._tensor_list(real["net"]),
                "real_acc_curve": self._tensor_list(real["acc"]),
            })
            return record

        beta_ensembles = {
            neighbor: aggregate_curve(
                f"beta_ensemble_{neighbor}",
                [row for row in configs if row["neighbor"] == neighbor],
            )
            for neighbor in neighbor_scores
        }
        full_ensemble = aggregate_curve("full_ensemble", configs)

        config_acc_by_real = []
        for row in configs:
            idx = row["knee_idx"]
            config_acc_by_real.append((row["config_id"], self._float(real["acc"][idx])))
        sorted_config_acc = sorted(config_acc_by_real, key=lambda item: item[1], reverse=True)
        rank_by_config = {config_id: rank + 1 for rank, (config_id, _) in enumerate(sorted_config_acc)}
        best_config_acc = sorted_config_acc[0][1]
        knee_alphas = [row["alpha_knee"] for row in configs]
        median_knee_alpha = float(torch.tensor(knee_alphas).median().item())
        min_knee_alpha = min(knee_alphas)

        def selector_from_idx(name, idx, source=None, selected_beta=None, selected_neighbor=None):
            selected_alpha = self._float(alphas[idx])
            curve = source["pseudo_net"] if source is not None else None
            shape = self._curve_shape(alphas, curve, idx) if curve is not None else {}
            acc = self._float(real["acc"][idx])
            record = {
                "selected_alpha": selected_alpha,
                "selected_beta": selected_beta,
                "selected_neighbor": selected_neighbor,
                "acc": acc,
                "gap_to_current": acc - current_metrics["accuracy"],
                "gap_to_text_only": acc - text_metrics["accuracy"],
                "gap_to_oracle": acc - oracle_acc,
                **self._real_at_index(real, idx),
                "effective_rho": shape.get("effective_rho"),
                "knee_strength": source.get("knee_strength") if source is not None else None,
                "pseudo_net_at_selected": shape.get("pseudo_net_at_selected"),
                "pseudo_net_max": shape.get("pseudo_net_max"),
                "curve_auc": shape.get("curve_auc"),
                "curve_monotonicity": shape.get("curve_monotonicity"),
                "plateau_start_alpha": shape.get("plateau_start_alpha"),
                "plateau_width": shape.get("plateau_width"),
            }
            if source is not None and isinstance(source.get("config_id"), int):
                record["rank_of_selected_config_by_real_acc"] = rank_by_config.get(source["config_id"])
                record["gap_to_best_config"] = acc - best_config_acc
            else:
                record["rank_of_selected_config_by_real_acc"] = None
                record["gap_to_best_config"] = acc - best_config_acc
            return name, record

        selectors = {}
        selectors["current"] = selector_from_idx(
            "current",
            self._nearest_alpha_idx(alphas, current_metrics["selected_alpha"]),
            None,
            None,
            None,
        )[1]
        selectors["text_only"] = selector_from_idx("text_only", 0)[1]
        selectors["visual_only"] = selector_from_idx("visual_only", len(alphas) - 1)[1]
        selectors["oracle_alpha"] = selector_from_idx("oracle_alpha", oracle_idx)[1]

        old_source = max(configs, key=lambda row: self._float(real["acc"][row["maxnet_idx"]]))
        selectors["old_maxnet"] = selector_from_idx(
            "old_maxnet",
            old_source["maxnet_idx"],
            old_source,
            old_source["beta"],
            old_source["neighbor"],
        )[1]

        for beta in (0.25, 0.30):
            for neighbor in ("vv", "tt"):
                source = next((row for row in configs if abs(row["beta"] - beta) < 1e-8 and row["neighbor"] == neighbor), None)
                if source is not None:
                    beta_key = f"{int(round(beta * 100)):03d}"
                    selectors[f"knee_fixed_beta_{beta_key}_{neighbor}"] = selector_from_idx(
                        f"knee_fixed_beta_{beta_key}_{neighbor}",
                        source["knee_idx"],
                        source,
                        beta,
                        neighbor,
                    )[1]

        for neighbor in neighbor_scores:
            neighbor_configs = [row for row in configs if row["neighbor"] == neighbor]
            auto_source = max(neighbor_configs, key=lambda row: row["knee_strength"])
            selectors[f"auto_beta_knee_{neighbor}"] = selector_from_idx(
                f"auto_beta_knee_{neighbor}",
                auto_source["knee_idx"],
                auto_source,
                auto_source["beta"],
                neighbor,
            )[1]
            ensemble = beta_ensembles[neighbor]
            selectors[f"beta_ensemble_knee_{neighbor}"] = selector_from_idx(
                f"beta_ensemble_knee_{neighbor}",
                ensemble["knee_idx"],
                ensemble,
                None,
                neighbor,
            )[1]
            selectors[f"neighbor_{neighbor}_knee"] = dict(selectors[f"beta_ensemble_knee_{neighbor}"])

        selectors["full_ensemble_knee"] = selector_from_idx(
            "full_ensemble_knee",
            full_ensemble["knee_idx"],
            full_ensemble,
            None,
            "vv+tt+hybrid",
        )[1]
        selectors["neighbor_ensemble_knee"] = dict(selectors["full_ensemble_knee"])
        selectors["median_knee_all"] = selector_from_idx(
            "median_knee_all",
            self._nearest_alpha_idx(alphas, median_knee_alpha),
            None,
            None,
            "vv+tt+hybrid",
        )[1]
        selectors["min_knee_all"] = selector_from_idx(
            "min_knee_all",
            self._nearest_alpha_idx(alphas, min_knee_alpha),
            None,
            None,
            "vv+tt+hybrid",
        )[1]

        acc_values = torch.tensor([self._float(real["acc"][row["knee_idx"]]) for row in configs], dtype=torch.float32)
        summary = {
            "dataset": self.config.data.dataset_name,
            "seed": int(self.seed),
            "kshot": int(self.kshot),
            "split": "test",
            "num_classes": int(num_classes),
            "text_only_acc": text_metrics["accuracy"],
            "visual_only_acc": visual_metrics["accuracy"],
            "current_acc": current_metrics["accuracy"],
            "current_alpha": current_metrics["selected_alpha"],
            "oracle_acc": oracle_acc,
            "oracle_alpha": oracle_alpha,
            "selectors": selectors,
            "mean_acc_over_configs": self._float(acc_values.mean()),
            "std_acc_over_configs": self._float(acc_values.std(unbiased=False)),
            "near_best_count_0.5": int((best_config_acc - acc_values <= 0.5).sum().item()),
            "near_best_count_1.0": int((best_config_acc - acc_values <= 1.0).sum().item()),
            "best_config_acc": best_config_acc,
            "best_config_id": sorted_config_acc[0][0],
            "support_reliability": support_reliability,
            "selector_feature_correlations": {
                "selected_alpha_vs_gap_to_oracle": self._corr(
                    torch.tensor([selectors[key]["selected_alpha"] for key in selectors], dtype=torch.float32),
                    torch.tensor([selectors[key]["gap_to_oracle"] for key in selectors], dtype=torch.float32),
                ),
                "selected_alpha_vs_gain_over_current": self._corr(
                    torch.tensor([selectors[key]["selected_alpha"] for key in selectors], dtype=torch.float32),
                    torch.tensor([selectors[key]["gap_to_current"] for key in selectors], dtype=torch.float32),
                ),
            },
            "acc_std_across_configs": self._float(acc_values.std(unbiased=False)),
            "curve_file": self.curves_path,
        }

        with open(self.curves_path, "w") as f:
            for row in curve_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.metrics.append(summary)
        self.best_val_acc = selectors["full_ensemble_knee"]["acc"]
        logger.info(
            f"Full ensemble knee acc={selectors['full_ensemble_knee']['acc']:.2f}% "
            f"at alpha={selectors['full_ensemble_knee']['selected_alpha']:.2f}; "
            f"current={current_metrics['accuracy']:.2f}%, oracle={oracle_acc:.2f}%"
        )
        logger.info(f"One-shot knee selector summary saved to: {self.summary_path}")
        logger.info(f"One-shot knee selector curves saved to: {self.curves_path}")
        log_experiment_metrics(
            {
                "accuracy": selectors["full_ensemble_knee"]["acc"],
                "mca": 0.0,
                "alpha": selectors["full_ensemble_knee"]["selected_alpha"],
            },
            title=self._metrics_title(),
        )

    def _finalize(self):
        return


def parse_args():
    parsed, unknown = create_argument_parser("Run one-shot ProtoFuse centroid-mix deep log", ARG_SCHEMA).parse_known_args()
    overrides = parse_override_arguments(unknown)
    return parsed, process_parsed_args(parsed, ARG_SCHEMA, overrides)


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, "debug", True), getattr(args, "disable_coloring", True))
    config = merge_configs(load_config_file(args.config), overrides)
    if getattr(args, "output_dir", None) is None:
        config.setdefault("logging", {})
        config["logging"]["output_dir"] = ProtoFuseOneShotCentroidMixCeilingPipeline.DEFAULT_OUTPUT_DIR
    run_for_dataset_configs(config, lambda dataset_config, _: ProtoFuseOneShotCentroidMixCeilingPipeline(dataset_config).run())


if __name__ == "__main__":
    main()
