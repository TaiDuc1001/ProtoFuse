import os
import json
from torch.utils.data import DataLoader, Subset

from utils import (
    logger,
    BaseTrainingPipeline,
    log_experiment_metrics,
)

from src.models.timo import TIMO


class TIMOPipeline(BaseTrainingPipeline):
    METHOD_NAME = "TIMO"
    DEFAULT_OUTPUT_DIR = "outputs/timo"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/timo"
    TRAINER_CLASS = TIMO
    VARIANT = "TIMO"

    def _get_training_epochs(self):
        return 1

    def _make_train_loader(self, shuffle=False):
        if not self.train_indices:
            raise RuntimeError("No training samples available for TIMO.")
        train_subset = Subset(self.dataset, list(self.train_indices))
        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def _train_epochs(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before TIMO run.")

        self.trainer.shots = self.kshot

        # logger.info(f"Extracting train features for {self.METHOD_NAME}...")
        train_features, train_labels = self._cached_train_features()
        augment_epoch = self.trainer._cfg_int(1, 'model.augment_epoch')
        self.trainer.train_vecs = train_features.repeat((augment_epoch, 1))
        self.trainer.train_labels = train_labels.repeat(augment_epoch)

        # logger.info(f"Using train features for {self.METHOD_NAME} hyperparameter search")
        tune_features, tune_labels = train_features, train_labels

        # logger.info("Extracting test features...")
        test_features, test_labels = self._cached_test_features()

        results = self.trainer.evaluate_timo_variant(
            tune_features,
            tune_labels,
            test_features,
            test_labels,
            variant=self.VARIANT,
        )
        self.metrics.append(results)
        self.best_val_acc = results.get('accuracy', 0.0)

        # logger.info(f"{self.METHOD_NAME} Accuracy: {results.get('accuracy', 0.0):.2f}%")
        # logger.info(f"{self.METHOD_NAME} MCA: {results.get('mca', 0.0):.2f}%")
        log_experiment_metrics(results)

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
        # logger.info(f"{self.METHOD_NAME} complete. Results written to {self.run_dir}")


BaseTrainingPipeline.register_extra_pipeline(TIMOPipeline)


class TIMOSPipeline(TIMOPipeline):
    METHOD_NAME = "TIMOS"
    DEFAULT_OUTPUT_DIR = "outputs/timos"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/timos"
    VARIANT = "TIMOS"


BaseTrainingPipeline.register_extra_pipeline(TIMOSPipeline)
