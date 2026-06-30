import os

from torch.utils.data import DataLoader, Subset

from utils import (
    BaseTrainingPipeline,
    ConfigNode,
    coerce_to_float,
    coerce_to_int,
    get_config_value,
    load_config_file,
    log_experiment_metrics,
    logger,
)

from src.models.coop import CoOP
from src.models.protofuse import ProtoFuse


class CoOPTrainingPipeline(BaseTrainingPipeline):
    METHOD_NAME = "CoOP"
    SAVE_BEST_LAST = False
    DEFAULT_OUTPUT_DIR = "outputs/coop"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/coop"
    TRAINER_CLASS = CoOP

    def _posthoc_protofuse_cfg(self):
        return self.config.get('posthoc_protofuse', ConfigNode())

    def _posthoc_protofuse_enabled(self):
        return bool(self._posthoc_protofuse_cfg().get('enabled', False))

    def _posthoc_protofuse_selector_settings(self):
        cfg = self._posthoc_protofuse_cfg()
        proto_cfg = {}
        config_path = cfg.get('config_path', 'configs/protofuse.yaml')
        if config_path:
            try:
                proto_cfg = load_config_file(config_path)
            except FileNotFoundError:
                logger.warning(f"ProtoFuse config not found at {config_path}; using post-hoc defaults.")

        alpha_steps = coerce_to_int(
            cfg.get('alpha_steps', get_config_value(proto_cfg, 'model.alpha_steps', 101)),
            101,
            key='posthoc_protofuse.alpha_steps',
        )
        proto_beta_values = get_config_value(proto_cfg, 'model.centroid_mix.beta_values', None)
        centroid_mix_cfg = cfg.get('centroid_mix', ConfigNode())
        beta_values = centroid_mix_cfg.get('beta_values', proto_beta_values)
        rho = None
        return alpha_steps, beta_values, rho

    def _train_epochs(self):
        super()._train_epochs()
        if self._posthoc_protofuse_enabled():
            self._run_posthoc_protofuse()

    def _run_posthoc_protofuse(self):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before post-hoc ProtoFuse.")
        if self.val_loader is None:
            logger.warning("Skipping post-hoc ProtoFuse because no validation/test loader is available.")
            return

        train_subset = Subset(self.dataset, list(self.train_indices))
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        cfg = self._posthoc_protofuse_cfg()
        alpha_steps, beta_values, rho = (
            self._posthoc_protofuse_selector_settings()
        )

        logger.info("Applying post-hoc ProtoFuse to frozen CoOp")
        self.trainer.freeze()

        train_features, train_labels = self.trainer.extract_features(train_loader)
        eval_features, eval_labels = self.trainer.extract_features(self.val_loader)

        text_prototypes = self.trainer.get_text_prototypes()
        selection = ProtoFuse.posthoc_fuse(
            text_prototypes,
            train_features,
            train_labels,
            device=self.device,
            alpha_steps=alpha_steps,
            beta_values=beta_values,
            rho=rho,
        )
        alpha = selection['alpha']

        coop_metrics = self.trainer.evaluate_features(
            eval_features,
            eval_labels,
            prototypes=text_prototypes,
            alpha=0.0,
        )
        log_experiment_metrics(coop_metrics, title=self._metrics_title("CoOP w/o ProtoFuse"))

        fused_prototypes = self.trainer.apply_posthoc_protofuse(
            alpha=alpha,
            fused_prototypes=selection['fused_prototypes'],
            visual_centroids=selection['visual_centroids'],
            missing_classes=selection['missing_classes'],
        )
        metrics = self.trainer.evaluate_features(
            eval_features,
            eval_labels,
            prototypes=fused_prototypes,
            alpha=alpha,
        )
        gap_to_coop = metrics.get('accuracy', 0.0) - coop_metrics.get('accuracy', 0.0)
        logger.info(
            f"CoOP ProtoFuse alpha={alpha:.4f} - "
            f"w/o={coop_metrics.get('accuracy', 0.0):.2f}% - "
            f"w/={metrics.get('accuracy', 0.0):.2f}% - "
            f"gap={gap_to_coop:+.2f}%"
        )

        result = dict(metrics)
        result.update({
            'epoch': self.global_epoch,
            'phase': 'posthoc_protofuse',
            'method': 'CoOP+ProtoFuse',
            'alpha': alpha,
            'train_loss': None,
            'train_acc': None,
            'val_loss': metrics.get('loss'),
            'val_acc': metrics.get('accuracy'),
            'coop_text_accuracy': coop_metrics.get('accuracy'),
            'gap_to_coop_text': gap_to_coop,
            'missing_centroid_classes': selection['missing_classes'],
        })

        self.metrics.append(result)
        self.best_val_acc = max(self.best_val_acc, result.get('accuracy', 0.0))

        if bool(cfg.get('save_prototypes', True)):
            proto_path = os.path.join(self.run_dir, 'posthoc_protofuse.pt')
            self.trainer.save_posthoc_protofuse(proto_path)
