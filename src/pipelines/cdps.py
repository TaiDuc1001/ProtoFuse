import os
import json
import torch
from torch.utils.data import DataLoader, Subset

from utils import (
    logger,
    BaseTrainingPipeline,
    log_experiment_metrics,
)

from src.models.protofuse import ProtoFuse
from src.models.cdps import CDPS, extract_vit_features_and_masks
from src.pipelines.posthoc_protofuse import PosthocProtoFuseMixin


class CDPSPipeline(PosthocProtoFuseMixin, BaseTrainingPipeline):
    METHOD_NAME = "CDPS"
    DEFAULT_OUTPUT_DIR = "outputs/cdps"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/cdps"
    TRAINER_CLASS = CDPS

    def _get_training_epochs(self):
        return 1

    def _make_train_loader(self, shuffle=False):
        if not self.train_indices:
            raise RuntimeError("No training samples available for CDPS.")
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

    def _cached_patch_payload(self, dataset, split_name):
        dataset_name = self.config.data.dataset_name
        checkpoint_cfg = self.config.get("checkpoint", {})
        base_cache_dir = checkpoint_cfg.get("cache_dir", self.DEFAULT_CHECKPOINT_DIR)
        cache_dir = os.path.join(base_cache_dir, "patch_cache", str(dataset_name))
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{split_name}_patches.pt")

        if os.path.exists(cache_file):
            return torch.load(cache_file, map_location="cpu", weights_only=False)

        logger.info(f"Extracting ViT patch features and attention masks for {dataset_name} [{split_name}]...")
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        masked_feats, global_feats, labels = extract_vit_features_and_masks(
            self.trainer.clip_model,
            loader,
            self.device,
            tgt_area=self.trainer.tgt_area,
            top_percent=self.trainer.top_percent,
            topk=self.trainer.topk,
        )
        payload = {
            "masked_feats": masked_feats,
            "global_feats": global_feats,
            "labels": labels,
        }
        torch.save(payload, cache_file)
        return payload

    def _train_epochs(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before CDPS run.")

        # 1. Load full dataset patch cache & slice training indices
        full_payload = self._cached_patch_payload(self.dataset, "train_or_full")
        train_indices = list(self.train_indices)
        train_masked_feats = [full_payload["masked_feats"][i] for i in train_indices]
        train_global_feats = full_payload["global_feats"][train_indices]
        train_labels = full_payload["labels"][train_indices]

        # 2. Load test split patch cache
        if getattr(self, "_val_dataset", None) is not None:
            test_payload = self._cached_patch_payload(self._val_dataset, "val_or_test")
            test_masked_feats = test_payload["masked_feats"]
            test_global_feats = test_payload["global_feats"]
            test_labels = test_payload["labels"]
        else:
            val_indices = list(self.val_indices)
            test_masked_feats = [full_payload["masked_feats"][i] for i in val_indices]
            test_global_feats = full_payload["global_feats"][val_indices]
            test_labels = full_payload["labels"][val_indices]

        # 3. Execute CDPS evaluation
        results = self.trainer.evaluate_cdps(
            train_masked_feats,
            train_global_feats,
            train_labels,
            test_masked_feats,
            test_global_feats,
            test_labels,
        )
        results = self._add_base_novel_metrics(results)
        self.metrics.append(results)
        self.best_val_acc = results.get('accuracy', 0.0)

        if not bool(self.logging_cfg.get("summary_only", False)):
            log_experiment_metrics(results, title=self._metrics_title(self.METHOD_NAME))

        # 4. Post-hoc ProtoFuse extension
        if self._posthoc_protofuse_enabled():
            self._run_posthoc_protofuse(
                train_masked_feats,
                train_global_feats,
                train_labels,
                test_masked_feats,
                test_global_feats,
                test_labels,
                results,
            )

    def _run_posthoc_protofuse(
        self,
        train_masked_feats,
        train_global_feats,
        train_labels,
        test_masked_feats,
        test_global_feats,
        test_labels,
        baseline_metrics,
    ):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized before post-hoc ProtoFuse.")

        cfg = self._posthoc_protofuse_cfg()
        alpha_steps, beta_values, rho = self._posthoc_protofuse_selector_settings()

        logger.info(f"Applying post-hoc ProtoFuse to {self.METHOD_NAME}")
        self.trainer.clear_posthoc_protofuse()
        selection = ProtoFuse.posthoc_fuse(
            self.trainer.get_text_prototypes(),
            train_global_feats,
            train_labels,
            device=self.device,
            alpha_steps=alpha_steps,
            beta_values=beta_values,
            rho=rho,
        )
        fused_clip_weights, fused_prototypes = self.trainer.apply_posthoc_protofuse(
            alpha=selection['alpha'],
            fused_prototypes=selection['fused_prototypes'],
            visual_centroids=selection['visual_centroids'],
            centroid_mask=selection['centroid_mask'],
            missing_classes=selection['missing_classes'],
        )
        metrics = self.trainer.evaluate_cdps(
            train_masked_feats,
            train_global_feats,
            train_labels,
            test_masked_feats,
            test_global_feats,
            test_labels,
            clip_weights=fused_clip_weights,
        )
        gap_to_cdps = metrics.get('accuracy', 0.0) - baseline_metrics.get('accuracy', 0.0)
        logger.info(
            f"{self.METHOD_NAME} ProtoFuse alpha={selection['alpha']:.4f} - "
            f"w/o={baseline_metrics.get('accuracy', 0.0):.2f}% - "
            f"w/={metrics.get('accuracy', 0.0):.2f}% - "
            f"gap={gap_to_cdps:+.2f}%"
        )

        result = dict(metrics)
        result.update({
            'phase': 'posthoc_protofuse',
            'method': f'{self.METHOD_NAME}+ProtoFuse',
            'alpha': selection['alpha'],
            'protofuse_alpha': selection['alpha'],
            'before_protofuse_accuracy': baseline_metrics.get('accuracy'),
            'protofuse_gain': gap_to_cdps,
            'cdps_accuracy': baseline_metrics.get('accuracy'),
            'gap_to_cdps': gap_to_cdps,
            'missing_centroid_classes': selection['missing_classes'],
        })
        result = self._add_base_novel_metrics(result)
        self.metrics.append(result)
        self.best_val_acc = max(self.best_val_acc, result.get('accuracy', 0.0))
        if not bool(self.logging_cfg.get("summary_only", False)):
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

        final_metrics = self.metrics[-1] if self.metrics else {}
        if not bool(self.logging_cfg.get("summary_only", False)):
            log_experiment_metrics(final_metrics, title=self._metrics_title())


BaseTrainingPipeline.register_extra_pipeline(CDPSPipeline)

