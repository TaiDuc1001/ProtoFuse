import os
import json
from torch.utils.data import DataLoader, Subset

from utils import (
    logger,
    BaseTrainingPipeline,
    log_experiment_metrics,
)

from src.models.proto_adapter import ProtoAdapter


class ProtoAdapterPipeline(BaseTrainingPipeline):
    METHOD_NAME = "ProtoAdapter"
    DEFAULT_OUTPUT_DIR = "outputs/proto_adapter"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/proto_adapter"
    TRAINER_CLASS = ProtoAdapter

    def _get_training_epochs(self):
        return 1

    def _make_train_loader(self, shuffle=False):
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before building Proto-Adapter prototypes.")
        if not self.train_indices:
            raise RuntimeError("No training samples available for Proto-Adapter.")
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
            raise RuntimeError("Trainer not initialized before Proto-Adapter run.")

        if os.path.exists(self.best_model_path):
            self.trainer.load_model(self.best_model_path)
            if self.val_loader is not None:
                test_features, test_labels = self._cached_val_features()
            else:
                test_features, test_labels = self._cached_train_features()
            results = self.trainer.evaluate_features(test_features, test_labels)
            results = self._add_base_novel_metrics(results)
            self.best_val_acc = results.get('accuracy', 0.0)
            self.metrics.append(results)
            return

        train_features, train_labels = self._cached_train_features()
        self.trainer.build_prototypes(train_features, train_labels)

        alpha, search_result = self.trainer.tune_alpha(train_features, train_labels)
        finetune_history = []
        if bool(self.model_cfg.get('finetune', {}).get('enabled', False)):
            finetune_history = self.trainer.finetune_adapter_from_features(train_features, train_labels)

        if self.val_loader is not None:
            val_features, val_labels = self._cached_val_features()
            results = self.trainer.evaluate_features(val_features, val_labels)
        else:
            results = self.trainer.evaluate_features(train_features, train_labels)

        results['alpha'] = alpha
        if search_result is not None:
            results['search'] = search_result
        if finetune_history:
            results['finetune'] = finetune_history
        results = self._add_base_novel_metrics(results)

        self.best_val_acc = results.get('accuracy', 0.0)
        self.metrics.append(results)

        # logger.info(f"Proto-Adapter alpha={alpha:.4f}")
        # logger.info(f"Accuracy: {results.get('accuracy', 0.0):.2f}%")
        # logger.info(f"MCA: {results.get('mca', 0.0):.2f}%")

        os.makedirs(self.run_dir, exist_ok=True)
        self.trainer.save_model(self.best_model_path)

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

        self.trainer.save_model(self.last_model_path)
        # logger.info(f"Proto-Adapter complete. Results written to {self.run_dir}")

        final_metrics = self.metrics[-1] if self.metrics else {}
        log_experiment_metrics(final_metrics, title=self._metrics_title())


BaseTrainingPipeline.register_extra_pipeline(ProtoAdapterPipeline)
