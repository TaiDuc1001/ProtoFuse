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

    def _train_epochs(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before APE run.")

        # logger.info("Extracting few-shot cache features...")
        train_features, train_labels = self._cached_train_features()
        cache_keys = train_features.t().contiguous()
        cache_values = torch.nn.functional.one_hot(
            train_labels.long(), num_classes=len(self.classnames)
        ).half().to(self.trainer.device)
        self.trainer.build_cache(cache_keys, cache_values)
        self.trainer.shots = self.kshot

        # logger.info("Using train features for APE hyperparameter search")
        tune_features, tune_labels = train_features, train_labels

        # logger.info("Extracting test features...")
        test_features, test_labels = self._cached_test_features()

        results = self.trainer.evaluate_ape(tune_features, tune_labels, test_features, test_labels)
        self.best_val_acc = results.get('accuracy', 0.0)
        self.metrics.append({'method': 'APE', **results})

        # logger.info(f"APE Accuracy: {results.get('accuracy', 0.0):.2f}%")
        # logger.info(f"APE MCA: {results.get('mca', 0.0):.2f}%")
        log_experiment_metrics(results)

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
            log_experiment_metrics(ape_t_results)

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
