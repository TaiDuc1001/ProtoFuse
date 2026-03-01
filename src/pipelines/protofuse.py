import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    logger,
    ConfigNode,
    BaseTrainingPipeline,
    log_experiment_start,
    log_experiment_metrics,
    compute_metrics,
)

from src.models.protofuse import ProtoFuse


class ProtoFusePipeline(BaseTrainingPipeline):
    METHOD_NAME = "ProtoFuse"
    DEFAULT_OUTPUT_DIR = "outputs/protofuse"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/protofuse"
    TRAINER_CLASS = ProtoFuse

    def _get_training_epochs(self):
        return 1

    def _extract_and_cache_all_features(self):
        cache_cfg = self.config.get('checkpoint', ConfigNode())
        cache_dir = Path(cache_cfg.get('cache_dir', self.DEFAULT_CHECKPOINT_DIR))
        cache_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = Path(self.dataset_root).name
        cache_path = cache_dir / f"features_{dataset_name}.pt"

        if cache_path.exists():
            data = torch.load(cache_path, map_location="cpu", weights_only=True)
            logger.info(f"Loaded cached features from {cache_path}")
            return data["features"], data["labels"]

        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        all_features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="  Extracting features", leave=False):
                images = images.to(self.device)
                features = self.trainer.clip_model.encode_image(images).float()
                all_features.append(features.cpu())
                all_labels.append(labels)

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        torch.save({"features": features, "labels": labels}, cache_path)
        logger.info(f"Cached features to {cache_path}")
        return features, labels

    def _remap_labels(self, labels):
        task_classes = sorted(set(labels.tolist()))
        remap = {c: i for i, c in enumerate(task_classes)}
        return torch.tensor([remap[l.item()] for l in labels]), len(task_classes)

    def _train_epochs(self):
        logger.info("Extracting CLIP features...")
        all_features, all_labels = self._extract_and_cache_all_features()

        train_features = all_features[self.train_indices]
        train_labels = all_labels[self.train_indices]
        val_features = all_features[self.val_indices]
        val_labels = all_labels[self.val_indices]

        remapped_train, num_classes = self._remap_labels(train_labels)
        remapped_val, _ = self._remap_labels(val_labels)

        logger.info("Running ProtoFuse (LOO-CV α selection)...")
        results = self.trainer.fuse_and_evaluate(
            train_features, remapped_train,
            val_features, remapped_val,
            num_classes,
        )

        self.best_val_acc = results["accuracy"]

        logger.info(f"Best α: {results['alpha']:.2f}")
        logger.info(f"Accuracy: {results['accuracy']:.2f}%")
        logger.info(f"MCA: {results['mca']:.2f}%")

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
        logger.info(f"Metrics saved to {self.metrics_path}")

        proto_path = os.path.join(self.run_dir, 'prototypes.pt')
        self.trainer.save_model(proto_path)

        logger.info("ProtoFuse complete.")


BaseTrainingPipeline.register_extra_pipeline(ProtoFusePipeline)
