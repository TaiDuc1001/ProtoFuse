import os
import json
from torch.utils.data import DataLoader, Subset

from utils import (
    logger,
    BaseTrainingPipeline,
    log_experiment_metrics,
)

from src.models.protofuse import ProtoFuse
from src.models.tip_adapter import TipAdapter
from src.pipelines.posthoc_protofuse import PosthocProtoFuseMixin


class TipAdapterPipeline(PosthocProtoFuseMixin, BaseTrainingPipeline):
    METHOD_NAME = "TipAdapter"
    DEFAULT_OUTPUT_DIR = "outputs/tip_adapter"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/tip_adapter"
    TRAINER_CLASS = TipAdapter

    def _get_training_epochs(self):
        return 1

    def _make_train_loader(self, shuffle=False):
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before building the Tip-Adapter cache.")
        if not self.train_indices:
            raise RuntimeError("No training samples available for Tip-Adapter cache.")
        train_subset = Subset(self.dataset, list(self.train_indices))
        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def _add_base_novel_metrics(self, metrics):
        if not self.base_novel_enabled:
            return metrics
        preds = metrics.get('predictions', [])
        labels = metrics.get('true_labels', [])
        if not preds or not labels:
            return metrics

        base_set = set(self.base_class_indices)
        novel_set = set(self.novel_class_indices)
        base_total = sum(1 for label in labels if label in base_set)
        novel_total = sum(1 for label in labels if label in novel_set)
        base_correct = sum(1 for pred, label in zip(preds, labels) if label in base_set and pred == label)
        novel_correct = sum(1 for pred, label in zip(preds, labels) if label in novel_set and pred == label)

        base_acc = 100 * base_correct / base_total if base_total > 0 else None
        novel_acc = 100 * novel_correct / novel_total if novel_total > 0 else None
        metrics['base_val_acc'] = base_acc
        metrics['novel_val_acc'] = novel_acc
        if base_acc is not None and novel_acc is not None and (base_acc + novel_acc) > 0:
            metrics['harmonic_mean'] = 2 * base_acc * novel_acc / (base_acc + novel_acc)
        else:
            metrics['harmonic_mean'] = None
        return metrics

    def _train_epochs(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before Tip-Adapter run.")

        # logger.info("Building Tip-Adapter cache from few-shot training split")
        train_features, train_labels = self._cached_train_features()
        self.trainer.build_cache(train_features, train_labels)

        alpha, beta, search_result = self.trainer.tune_alpha_beta(train_features, train_labels)
        finetune_history = []
        if bool(self.model_cfg.get('finetune', {}).get('enabled', False)):
            finetune_history = self.trainer.finetune_adapter_from_features(train_features, train_labels)

        if self.val_loader is not None:
            val_features, val_labels = self._cached_val_features()
            results = self.trainer.evaluate_features(val_features, val_labels)
        else:
            results = self.trainer.evaluate_features(train_features, train_labels, exclude_self=True)

        results['alpha'] = alpha
        results['beta'] = beta
        if search_result is not None:
            results['search'] = search_result
        if finetune_history:
            results['finetune'] = finetune_history
        results = self._add_base_novel_metrics(results)

        self.best_val_acc = results.get('accuracy', 0.0)
        self.metrics.append(results)

        if self._posthoc_protofuse_enabled():
            self._run_posthoc_protofuse(
                train_features,
                train_labels,
                val_features if self.val_loader is not None else train_features,
                val_labels if self.val_loader is not None else train_labels,
                results,
                exclude_self=self.val_loader is None,
            )

        # logger.info(f"Tip-Adapter alpha={alpha:.4f}, beta={beta:.4f}")
        # logger.info(f"Accuracy: {results.get('accuracy', 0.0):.2f}%")
        # logger.info(f"MCA: {results.get('mca', 0.0):.2f}%")

    def _run_posthoc_protofuse(self, train_features, train_labels, eval_features, eval_labels, baseline_metrics, exclude_self=False):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before post-hoc ProtoFuse.")

        cfg = self._posthoc_protofuse_cfg()
        alpha_steps, beta_values, rho = (
            self._posthoc_protofuse_selector_settings()
        )

        logger.info("Applying post-hoc ProtoFuse to Tip-Adapter")
        self.trainer.clear_posthoc_protofuse()
        selection = ProtoFuse.posthoc_fuse(
            self.trainer.get_text_prototypes(),
            train_features,
            train_labels,
            device=self.device,
            alpha_steps=alpha_steps,
            beta_values=beta_values,
            query_features=eval_features,
            rho=rho,
        )
        fused_prototypes = self.trainer.apply_posthoc_protofuse(
            alpha=selection['alpha'],
            fused_prototypes=selection['fused_prototypes'],
            visual_centroids=selection['visual_centroids'],
            centroid_mask=selection['centroid_mask'],
            missing_classes=selection['missing_classes'],
        )
        metrics = self.trainer.evaluate_features(
            eval_features,
            eval_labels,
            exclude_self=exclude_self,
            text_features=fused_prototypes,
        )
        gap_to_tip = metrics.get('accuracy', 0.0) - baseline_metrics.get('accuracy', 0.0)
        logger.info(
            f"Tip-Adapter ProtoFuse alpha={selection['alpha']:.4f} - "
            f"w/o={baseline_metrics.get('accuracy', 0.0):.2f}% - "
            f"w/={metrics.get('accuracy', 0.0):.2f}% - "
            f"gap={gap_to_tip:+.2f}%"
        )

        result = dict(metrics)
        result.update({
            'phase': 'posthoc_protofuse',
            'method': 'TipAdapter+ProtoFuse',
            'alpha': selection['alpha'],
            'protofuse_alpha': selection['alpha'],
            'tip_accuracy': baseline_metrics.get('accuracy'),
            'gap_to_tip': gap_to_tip,
            'missing_centroid_classes': selection['missing_classes'],
        })
        result = self._add_base_novel_metrics(result)
        self.metrics.append(result)
        self.best_val_acc = max(self.best_val_acc, result.get('accuracy', 0.0))
        if not bool(self.logging_cfg.get("summary_only", False)):
            log_experiment_metrics(result, title=self._metrics_title("TipAdapter+ProtoFuse"))

        if bool(cfg.get('save_prototypes', True)):
            proto_path = os.path.join(self.run_dir, 'posthoc_protofuse.pt')
            self.trainer.save_posthoc_protofuse(proto_path)

    def _finalize(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before finalization.")

        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.trainer_cfg.to_dict(), f, indent=4)

        metrics_out = {
            "method": self.METHOD_NAME,
            "dataset": self.config.data.dataset_name,
            "kshot": self.kshot,
            "seed": self.seed,
            "metrics": self.metrics,
        }
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics_out, f, indent=2)
        # logger.info(f"Tip-Adapter complete. Results written to {self.run_dir}")

        final_metrics = self.metrics[-1] if self.metrics else {}
        if not bool(self.logging_cfg.get("summary_only", False)):
            log_experiment_metrics(final_metrics, title=self._metrics_title())


BaseTrainingPipeline.register_extra_pipeline(TipAdapterPipeline)
