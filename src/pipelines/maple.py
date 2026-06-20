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

from src.models.maple import MaPLe
from src.models.protofuse import ProtoFuse


class MaPLeTrainingPipeline(BaseTrainingPipeline):
    METHOD_NAME = "MaPLe"
    SAVE_BEST_LAST = False
    DEFAULT_OUTPUT_DIR = "outputs/maple"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/maple"
    TRAINER_CLASS = MaPLe

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
        rho = coerce_to_float(
            cfg.get('rho', get_config_value(proto_cfg, 'model.rho', 0.5)),
            0.5,
            key='posthoc_protofuse.rho',
        )
        return alpha_steps, beta_values, rho

    def _try_load_checkpoint(self) -> bool:
        if self.checkpoint_cache is None or self.checkpoint_id is None:
            return False
        if not self.checkpoint_cache.exists(self.checkpoint_id):
            return False
        ckpt = self.checkpoint_cache.load(self.checkpoint_id)
        if ckpt is None:
            return False
        if self.trainer is None:
            return False

        model_state = ckpt['model_state_dict']
        prompt_state = model_state.get('prompt_learner_state_dict')
        if prompt_state is not None:
            prompt_state = dict(prompt_state)
            if "token_prefix" in prompt_state:
                del prompt_state["token_prefix"]
            if "token_suffix" in prompt_state:
                del prompt_state["token_suffix"]
            self.trainer.model.prompt_learner.load_state_dict(prompt_state, strict=False)
        elif 'ctx' in model_state:
            logger.warning("Loaded legacy MaPLe checkpoint with ctx only; compound prompts were not cached.")
            self.trainer.model.prompt_learner.ctx.data = model_state['ctx'].to(self.device)

        self.trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.labeled_indices = ckpt['labeled_indices']
        self.unlabeled_indices = ckpt['unlabeled_indices']
        self.metrics = ckpt['metrics']
        self.global_epoch = len(self.metrics)
        logger.info(f"Loaded checkpoint: {self.checkpoint_id} (epoch {self.global_epoch})")
        return True

    def _save_checkpoint(self):
        if self.checkpoint_cache is None or self.checkpoint_id is None:
            return
        if self.trainer is None:
            return
        model_state = {
            'prompt_learner_state_dict': self.trainer.model.prompt_learner.state_dict(),
        }
        path = self.checkpoint_cache.save(
            self.checkpoint_id,
            model_state,
            self.trainer.optimizer.state_dict(),
            self.trainer.scheduler.state_dict(),
            self.labeled_indices,
            self.unlabeled_indices,
            self.metrics,
            self.config
        )
        logger.debug(f"Saved checkpoint to: {path}")

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

        logger.info("Applying post-hoc ProtoFuse to frozen MaPLe")
        self.trainer.clear_posthoc_protofuse()
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
            query_features=eval_features,
            rho=rho,
        )
        alpha = selection['alpha']

        maple_metrics = self.trainer.evaluate_features(
            eval_features,
            eval_labels,
            prototypes=text_prototypes,
            alpha=0.0,
        )
        log_experiment_metrics(maple_metrics, title=self._metrics_title("MaPLe w/o ProtoFuse"))

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
        gap_to_maple = metrics.get('accuracy', 0.0) - maple_metrics.get('accuracy', 0.0)
        logger.info(
            f"MaPLe ProtoFuse alpha={alpha:.4f} - "
            f"w/o={maple_metrics.get('accuracy', 0.0):.2f}% - "
            f"w/={metrics.get('accuracy', 0.0):.2f}% - "
            f"gap={gap_to_maple:+.2f}%"
        )

        result = dict(metrics)
        result.update({
            'epoch': self.global_epoch,
            'phase': 'posthoc_protofuse',
            'method': 'MaPLe+ProtoFuse',
            'alpha': alpha,
            'train_loss': None,
            'train_acc': None,
            'val_loss': metrics.get('loss'),
            'val_acc': metrics.get('accuracy'),
            'maple_text_accuracy': maple_metrics.get('accuracy'),
            'gap_to_maple_text': gap_to_maple,
            'missing_centroid_classes': selection['missing_classes'],
        })

        self.metrics.append(result)
        self.best_val_acc = max(self.best_val_acc, result.get('accuracy', 0.0))

        if bool(cfg.get('save_prototypes', True)):
            proto_path = os.path.join(self.run_dir, 'posthoc_protofuse.pt')
            self.trainer.save_posthoc_protofuse(proto_path)
