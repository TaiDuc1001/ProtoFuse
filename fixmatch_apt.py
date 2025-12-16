import os
import time
import json
import math
import random
import torch
import datetime
import torch.nn.functional as F
from torchvision import transforms
from collections import defaultdict
from torchvision.datasets import ImageFolder
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader, Subset

from utils import (
    logger,
    setup_logging,
    ConfigNode,
    set_global_seed,
    build_config_namespace,
    create_argument_parser,
    process_parsed_args,
    parse_override_arguments,
    merge_configs,
    load_config_file,
    get_config_value,
    coerce_to_str,
    coerce_to_int,
    coerce_to_float,
    log_experiment_start,
    log_experiment_accuracy,
)

from apt_original import APT, APTTrainingPipeline
from fixmatch_utils import FixMatchMixin

ARG_SCHEMA = {
    'config': {'type': str, 'required': True, 'help': 'Path to YAML configuration file'},
    'output_dir': {'type': str, 'help': 'Override logging.output_dir from config', 'config_path': 'logging.output_dir'},
    'device': {'type': str, 'help': 'Override training.device from config', 'config_path': 'training.device'},
    'debug': {'type': bool, 'help': 'Enable debug output', 'default': False},
    'disable_coloring': {'type': bool, 'help': 'Disable colored output for log files', 'default': False},
}


class FixMatchAPTTrainingPipeline(APTTrainingPipeline, FixMatchMixin):
    def __init__(self, config):
        super().__init__(config)
        self._init_fixmatch_config()
        
        base_output_value = self.logging_cfg.get("output_dir", "outputs/fixmatch_apt")
        base_output = coerce_to_str(base_output_value, "outputs/fixmatch_apt", key="logging.output_dir")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(base_output, timestamp)
        self.config_path = os.path.join(self.run_dir, 'config.json')
        self.metrics_path = os.path.join(self.run_dir, 'metrics.json')
        self.best_model_path = os.path.join(self.run_dir, 'best.pt')
        self.last_model_path = os.path.join(self.run_dir, 'last.pt')
        self.eda_dir = os.path.join(self.run_dir, 'eda')
    
    def run(self):
        set_global_seed(self.seed)
        
        logger.section("Initialization", "config")
        self._prepare_directories()
        self._load_dataset()
        self._split_dataset()
        self._initialize_trainer()

        dataset_name = self.config.model.dataset_name
        log_experiment_start("APT+FixMatch", dataset_name, self.kshot, self.seed)
        
        if len(self.unlabeled_indices) == 0:
            logger.warning("No unlabeled samples available, falling back to standard APT training")
            logger.section("APT Training (No Unlabeled Data)", "train")
            self._train_epochs()
        else:
            logger.section("FixMatch Training", "train")
            logger.info(f"FixMatch config: confidence={self.confidence}, wu={self.wu}, unlabeled_batch={self.unlabeled_batch_size}")
            self._train_fixmatch_epochs()
        
        logger.section("Finalization", "save")
        self._finalize()
    
    def _train_fixmatch_epochs(self):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before training.")
        if not self.labeled_indices:
            raise RuntimeError("No labeled samples available.")
        if not self.unlabeled_indices:
            raise RuntimeError("No unlabeled samples available for FixMatch.")
        
        labeled_loader, unlabeled_loader = self._create_fixmatch_loaders()
        epochs_total = self._get_training_epochs()
        
        for epoch_idx in range(1, epochs_total + 1):
            self._run_fixmatch_epoch(epoch_idx, epochs_total, labeled_loader, unlabeled_loader)
    
    def _run_fixmatch_epoch(self, epoch_idx, epochs_total, labeled_loader, unlabeled_loader):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized.")
        
        self.global_epoch += 1
        start_time = time.time()
        self.trainer.model.train()
        
        running_loss = 0.0
        running_loss_xe = 0.0
        running_loss_u = 0.0
        running_accuracy = 0.0
        running_mask_ratio = 0.0
        steps = 0
        
        unlabeled_iter = iter(unlabeled_loader)
        
        for labeled_batch in labeled_loader:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)
            
            metrics = self._fixmatch_train_step(labeled_batch, unlabeled_batch)
            
            running_loss += metrics['loss']
            running_loss_xe += metrics['loss_xe']
            running_loss_u += metrics['loss_u']
            running_accuracy += metrics['accuracy']
            running_mask_ratio += metrics['mask_ratio']
            steps += 1
        
        avg_loss = running_loss / max(1, steps)
        avg_loss_xe = running_loss_xe / max(1, steps)
        avg_loss_u = running_loss_u / max(1, steps)
        avg_acc = running_accuracy / max(1, steps)
        avg_mask = running_mask_ratio / max(1, steps)
        
        if self.val_loader is not None:
            results = self.trainer.evaluate(self.val_loader)
            val_acc = results['accuracy']
            val_loss = results['loss']
        else:
            val_acc = 0.0
            val_loss = 0.0
        
        epoch_dir = os.path.join(self.run_dir, f'epoch_{epoch_idx:03d}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        epoch_time = time.time() - start_time
        epoch_result = {
            'epoch': epoch_idx,
            'train_loss': avg_loss,
            'train_loss_xe': avg_loss_xe,
            'train_loss_u': avg_loss_u,
            'train_acc': avg_acc,
            'mask_ratio': avg_mask,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': epoch_time
        }
        with open(os.path.join(epoch_dir, 'result.json'), 'w') as f:
            json.dump(epoch_result, f, indent=2)
        
        self.metrics.append(epoch_result)
        
        if self.val_loader is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.trainer.save_model(self.best_model_path)
        
        val_acc_display = f"{val_acc:.2f}%" if self.val_loader is not None else "N/A"
        logger.info(f"Epoch {epoch_idx} - loss={avg_loss:.4f} (xe={avg_loss_xe:.4f}, u={avg_loss_u:.4f}) - acc={avg_acc:.2f}% - mask={avg_mask:.2%} - val_acc={val_acc_display} - {epoch_time:.2f}s")
        
        if self.trainer.scheduler is not None:
            self.trainer.scheduler.step()


def parse_args():
    parser = create_argument_parser("Train APT+FixMatch model", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True), getattr(args, 'disable_coloring', False))
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = FixMatchAPTTrainingPipeline(merged)
    pipeline.run()


if __name__ == "__main__":
    main()
