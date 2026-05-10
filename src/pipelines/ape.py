import os
import json
import torch
from torch.utils.data import DataLoader, Subset

from utils import (
    logger,
    BaseTrainingPipeline,
    log_experiment_metrics,
)

from src.models.ape import APE


class APEPipeline(BaseTrainingPipeline):
    METHOD_NAME = "APE"
    DEFAULT_OUTPUT_DIR = "outputs/ape"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/ape"
    TRAINER_CLASS = APE

    def _get_training_epochs(self):
        return 1

    def _make_train_loader(self, shuffle=False):
        if not self.train_indices:
            raise RuntimeError("No training samples available for APE cache.")
        train_subset = Subset(self.dataset, list(self.train_indices))
        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def _build_balanced_cache_subset(self, train_features, train_labels):
        num_classes = len(self.classnames)
        counts = torch.bincount(train_labels.long(), minlength=num_classes)
        positive_counts = counts[counts > 0]
        if positive_counts.numel() == 0:
            raise RuntimeError("APE cache cannot be built: no labeled samples found.")

        effective_shots = int(positive_counts.min().item())
        if self.kshot > 0:
            effective_shots = min(effective_shots, self.kshot)
        if effective_shots <= 0:
            raise RuntimeError("APE cache cannot be built: effective shots is zero.")

        selected_indices = []
        for class_idx in range(num_classes):
            class_positions = torch.nonzero(train_labels == class_idx, as_tuple=False).flatten()
            if class_positions.numel() < effective_shots:
                raise RuntimeError(
                    f"APE cache build failed: class {class_idx} has {class_positions.numel()} samples, "
                    f"but {effective_shots} are required."
                )
            selected_indices.append(class_positions[:effective_shots])

        balanced_indices = torch.cat(selected_indices, dim=0)
        return train_features[balanced_indices], train_labels[balanced_indices], effective_shots

    def _train_epochs(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before APE run.")

        # logger.info("Extracting few-shot cache features...")
        train_features, train_labels = self._cached_train_features()
        cache_features, cache_labels, effective_shots = self._build_balanced_cache_subset(train_features, train_labels)
        if effective_shots < self.kshot:
            logger.warning(
                f"APE requested k-shot={self.kshot}, but at least one class has fewer samples. "
                f"Using effective_shots={effective_shots} for cache construction."
            )

        cache_keys = cache_features.t().contiguous()
        cache_values = torch.nn.functional.one_hot(
            cache_labels.long(), num_classes=len(self.classnames)
        ).half().to(self.trainer.device)
        self.trainer.build_cache(cache_keys, cache_values)
        self.trainer.shots = effective_shots

        # logger.info("Using train features for APE hyperparameter search")
        tune_features, tune_labels = train_features, train_labels

        # logger.info("Extracting test features...")
        test_features, test_labels = self._cached_test_features()

        results = self.trainer.evaluate_ape(tune_features, tune_labels, test_features, test_labels)
        self.best_val_acc = results.get('accuracy', 0.0)
        self.metrics.append({'method': 'APE', **results})

        # logger.info(f"APE Accuracy: {results.get('accuracy', 0.0):.2f}%")
        # logger.info(f"APE MCA: {results.get('mca', 0.0):.2f}%")
        log_experiment_metrics(results, title=self._metrics_title("APE"))

        finetune_cfg = self.config.get('model', {}).get('finetune', {})
        if bool(finetune_cfg.get('enabled', False)):
            # logger.info("Running APE-T (trainable variant)...")
            train_loader_aug = self._make_train_loader(shuffle=True)
            ape_t_results = self.trainer.train_ape_t(
                train_loader_aug, tune_features, tune_labels, test_features, test_labels,
                train_features=train_features, train_labels=train_labels
            )
            self.best_val_acc = max(self.best_val_acc, ape_t_results.get('accuracy', 0.0))
            self.metrics.append({'method': 'APE-T', **ape_t_results})
            # logger.info(f"APE-T Accuracy: {ape_t_results.get('accuracy', 0.0):.2f}%")
            log_experiment_metrics(ape_t_results, title=self._metrics_title("APE-T"))

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
        # logger.info(f"APE complete. Results written to {self.run_dir}")


BaseTrainingPipeline.register_extra_pipeline(APEPipeline)
