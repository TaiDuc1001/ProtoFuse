import os
import json
import torch
from utils import (
    logger,
    BaseTrainingPipeline,
    log_experiment_metrics,
)

from src.models.protofuse import ProtoFuse


class ProtoFusePipeline(BaseTrainingPipeline):
    METHOD_NAME = "ProtoFuse"
    DEFAULT_OUTPUT_DIR = "outputs/protofuse"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/protofuse"
    TRAINER_CLASS = ProtoFuse

    def _get_training_epochs(self):
        return 1

    def _remap_labels(self, labels):
        task_classes = sorted(set(labels.tolist()))
        remap = {c: i for i, c in enumerate(task_classes)}
        return torch.tensor([remap[l.item()] for l in labels]), len(task_classes)

    def _train_epochs(self):
        # logger.info("Extracting CLIP features...")
        train_features, train_labels = self._cached_train_features()
        val_features, val_labels = self._cached_val_features()

        remapped_train, num_classes = self._remap_labels(train_labels)
        remapped_val, _ = self._remap_labels(val_labels)

        # logger.info("Running ProtoFuse (LOO-CV α selection)...")
        results = self.trainer.fuse_and_evaluate(
            train_features, remapped_train,
            val_features, remapped_val,
            num_classes,
        )

        self.best_val_acc = results["accuracy"]

        # logger.info(f"Best α: {results['alpha']:.2f}")
        # logger.info(f"Accuracy: {results['accuracy']:.2f}%")
        # logger.info(f"MCA: {results['mca']:.2f}%")

        self.metrics.append(results)
        log_experiment_metrics(results)

    def _finalize(self):
        os.makedirs(self.run_dir, exist_ok=True)
        metrics_out = {
            "method": self.METHOD_NAME,
            "dataset": self.config.data.dataset_name,
            "kshot": self.kshot,
            "seed": self.seed,
            "metrics": self.metrics,
        }
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics_out, f, indent=2)
        # logger.info(f"Metrics saved to {self.metrics_path}")

        proto_path = os.path.join(self.run_dir, 'prototypes.pt')
        self.trainer.save_model(proto_path)

        # logger.info("ProtoFuse complete.")


BaseTrainingPipeline.register_extra_pipeline(ProtoFusePipeline)
