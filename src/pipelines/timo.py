import os
import json
from torch.utils.data import DataLoader, Subset

from utils import (
    logger,
    BaseTrainingPipeline,
    log_experiment_metrics,
)

from src.models.protofuse import ProtoFuse
from src.models.timo import TIMO
from src.pipelines.posthoc_protofuse import PosthocProtoFuseMixin


class TIMOPipeline(PosthocProtoFuseMixin, BaseTrainingPipeline):
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
        log_experiment_metrics(results, title=self._metrics_title(self.METHOD_NAME))

        if self._posthoc_protofuse_enabled():
            self._run_posthoc_protofuse(train_features, train_labels, tune_features, tune_labels, test_features, test_labels, results)

    def _run_posthoc_protofuse(self, train_features, train_labels, tune_features, tune_labels, test_features, test_labels, baseline_metrics):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before post-hoc ProtoFuse.")

        cfg = self._posthoc_protofuse_cfg()
        alpha_steps, beta_values, force_loo_accuracy = self._posthoc_protofuse_selector_settings()

        logger.info(f"Applying post-hoc ProtoFuse to {self.METHOD_NAME}")
        self.trainer.clear_posthoc_protofuse()
        selection = ProtoFuse.posthoc_fuse(
            self.trainer.get_text_prototypes(),
            train_features,
            train_labels,
            device=self.device,
            alpha_steps=alpha_steps,
            beta_values=beta_values,
            force_loo_accuracy=force_loo_accuracy,
        )
        fused_clip_weights, fused_text_features_all = self.trainer.apply_posthoc_protofuse(
            alpha=selection['alpha'],
            fused_prototypes=selection['fused_prototypes'],
            visual_centroids=selection['visual_centroids'],
            centroid_mask=selection['centroid_mask'],
            missing_classes=selection['missing_classes'],
        )
        metrics = self.trainer.evaluate_timo_variant(
            tune_features,
            tune_labels,
            test_features,
            test_labels,
            variant=self.VARIANT,
            clip_weights=fused_clip_weights,
            text_features_all=fused_text_features_all,
        )
        gap_to_timo = metrics.get('accuracy', 0.0) - baseline_metrics.get('accuracy', 0.0)
        logger.info(
            f"{self.METHOD_NAME} ProtoFuse alpha={selection['alpha']:.4f} - "
            f"w/o={baseline_metrics.get('accuracy', 0.0):.2f}% - "
            f"w/={metrics.get('accuracy', 0.0):.2f}% - "
            f"gap={gap_to_timo:+.2f}%"
        )

        result = dict(metrics)
        result.update({
            'phase': 'posthoc_protofuse',
            'method': f'{self.METHOD_NAME}+ProtoFuse',
            'protofuse_alpha': selection['alpha'],
            'timo_accuracy': baseline_metrics.get('accuracy'),
            'gap_to_timo': gap_to_timo,
            'missing_centroid_classes': selection['missing_classes'],
        })
        self.metrics.append(result)
        self.best_val_acc = max(self.best_val_acc, result.get('accuracy', 0.0))
        log_experiment_metrics(result, title=self._metrics_title(f"{self.METHOD_NAME}+ProtoFuse"))

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
        # logger.info(f"{self.METHOD_NAME} complete. Results written to {self.run_dir}")


BaseTrainingPipeline.register_extra_pipeline(TIMOPipeline)


class TIMOSPipeline(TIMOPipeline):
    METHOD_NAME = "TIMOS"
    DEFAULT_OUTPUT_DIR = "outputs/timos"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/timos"
    VARIANT = "TIMOS"


BaseTrainingPipeline.register_extra_pipeline(TIMOSPipeline)
