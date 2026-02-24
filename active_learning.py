import os
import json
import time
import torch
import datetime
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

import apt
import coop
import cocoop
import maple
import vife

from utils import (
    logger,
    setup_logging,
    ConfigNode,
    BaseTrainingPipeline,
    set_global_seed,
    compute_metrics,
    log_experiment_start,
    log_experiment_metrics,
    create_argument_parser,
    process_parsed_args,
    parse_override_arguments,
    merge_configs,
    load_config_file,
    coerce_to_str,
    coerce_to_int,
    get_config_value,
)
from active_learning_algorithms import get_strategy


ARG_SCHEMA = {
    'config': {'type': str, 'required': True, 'help': 'Path to YAML configuration file'},
    'output_dir': {'type': str, 'help': 'Override logging.output_dir from config', 'config_path': 'logging.output_dir'},
    'device': {'type': str, 'help': 'Override training.device from config', 'config_path': 'training.device'},
    'debug': {'type': bool, 'help': 'Enable debug output', 'default': True},
    'disable_coloring': {'type': bool, 'help': 'Disable colored output for log files', 'default': False},
}


class IndexTrackingDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.base_dataset[real_idx]
        return image, label, idx


class ActiveLearningPipeline:
    def __init__(self, config):
        if not isinstance(config, ConfigNode):
            config = ConfigNode(config)
        self.config = config

        al_cfg = self.config.get('active_learning', ConfigNode())
        if not isinstance(al_cfg, ConfigNode):
            al_cfg = ConfigNode(al_cfg)
        self.al_cfg = al_cfg

        model_type = coerce_to_str(al_cfg.get('model_type', 'APT'), 'APT')
        pipeline_cls = BaseTrainingPipeline.get_pipeline_by_name(model_type)
        self.inner = pipeline_cls(config)

        strategy_name = coerce_to_str(al_cfg.get('strategy', 'entropy'), 'entropy')
        self.strategy = get_strategy(strategy_name)

        self.kshot_budget = coerce_to_int(al_cfg.get('kshot_budget', self.inner.kshot), self.inner.kshot)
        self.num_rounds = coerce_to_int(al_cfg.get('num_rounds', 5), 5)

        base_output_value = self.inner.logging_cfg.get("output_dir", "outputs/active_learning")
        base_output = coerce_to_str(base_output_value, "outputs/active_learning", key="logging.output_dir")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(base_output, timestamp)
        self.config_path = os.path.join(self.run_dir, 'config.json')
        self.metrics_path = os.path.join(self.run_dir, 'metrics.json')
        self.best_model_path = os.path.join(self.run_dir, 'best.pt')
        self.last_model_path = os.path.join(self.run_dir, 'last.pt')
        self.selection_log_path = os.path.join(self.run_dir, 'al_selections.json')
        self.eda_dir = os.path.join(self.run_dir, 'eda')

        self.inner.run_dir = self.run_dir
        self.inner.config_path = self.config_path
        self.inner.metrics_path = self.metrics_path
        self.inner.best_model_path = self.best_model_path
        self.inner.last_model_path = self.last_model_path
        self.inner.eda_dir = self.eda_dir

        self.round_metrics = []
        self.all_selections = []

    def run(self):
        set_global_seed(self.inner.seed)

        logger.section("Initialization", "config")
        self.inner._prepare_directories()
        self.inner._load_dataset()
        self.inner._split_dataset()
        self.inner._initialize_trainer()

        dataset_name = self.config.data.dataset_name
        method_name = f"ActiveLearning+{self.inner.METHOD_NAME}"
        log_experiment_start(method_name, dataset_name, self.inner.kshot, self.inner.seed)

        num_classes = len(self.inner.classnames)
        total_budget_per_round = self.kshot_budget * num_classes
        unlabeled_count = len(self.inner.unlabeled_indices)

        max_possible_rounds = unlabeled_count // total_budget_per_round if total_budget_per_round > 0 else 0
        effective_rounds = min(self.num_rounds, max_possible_rounds + 1)
        remaining_after = unlabeled_count - (min(effective_rounds - 1, max_possible_rounds) * total_budget_per_round)

        logger.info(f"Active Learning config: strategy={self.strategy.name}, "
                     f"model={self.inner.METHOD_NAME}, "
                     f"kshot_budget={self.kshot_budget}, num_rounds={self.num_rounds}, "
                     f"total_budget_per_round={total_budget_per_round}")
        logger.info(f"Initial labeled: {len(self.inner.labeled_indices)}, "
                     f"unlabeled: {unlabeled_count}")
        if effective_rounds < self.num_rounds:
            logger.warning(f"Requested {self.num_rounds} rounds but only {effective_rounds} "
                           f"possible with {unlabeled_count} unlabeled samples "
                           f"and budget {total_budget_per_round}/round")
        logger.info(f"Effective rounds: {effective_rounds}, "
                     f"remaining unlabeled after AL: {remaining_after}")

        for round_idx in range(1, effective_rounds + 1):
            logger.section(f"Active Learning Round {round_idx}/{effective_rounds}", "train")
            self._run_al_round(round_idx, total_budget_per_round)

            if len(self.inner.unlabeled_indices) == 0:
                logger.info("No more unlabeled samples. Stopping AL loop.")
                break

        logger.section("Finalization", "save")
        self._finalize_al()

    def _run_al_round(self, round_idx, budget):
        round_dir = os.path.join(self.run_dir, f'round_{round_idx:02d}')
        os.makedirs(round_dir, exist_ok=True)

        logger.info(f"Round {round_idx}: labeled={len(self.inner.labeled_indices)}, "
                     f"unlabeled={len(self.inner.unlabeled_indices)}")

        if round_idx > 1:
            self.inner.trainer.reset_optimizer_scheduler()

        train_subset = Subset(self.inner.dataset, list(self.inner.labeled_indices))
        train_loader = DataLoader(
            train_subset, batch_size=self.inner.batch_size,
            shuffle=True, num_workers=self.inner.num_workers,
        )

        epochs_total = self.inner._get_training_epochs()
        self.inner.best_val_acc = -float('inf')

        round_start_time = time.time()
        for epoch_idx in range(1, epochs_total + 1):
            self.inner._run_epoch(epoch_idx, epochs_total, train_loader, round_dir)
        round_train_time = time.time() - round_start_time

        val_metrics = {}
        if self.inner.val_loader is not None:
            val_metrics = self.inner.trainer.evaluate(self.inner.val_loader)
            logger.info(f"Round {round_idx} val accuracy: {val_metrics.get('accuracy', 0.0):.2f}%")

        selected_indices = []
        if len(self.inner.unlabeled_indices) > 0 and round_idx < self.num_rounds:
            selected_indices = self._select_samples(budget)
            self._move_to_labeled(selected_indices)
            logger.info(f"Selected {len(selected_indices)} samples. "
                         f"New labeled={len(self.inner.labeled_indices)}, "
                         f"unlabeled={len(self.inner.unlabeled_indices)}")

        round_result = {
            'round': round_idx,
            'labeled_count': len(self.inner.labeled_indices),
            'unlabeled_count': len(self.inner.unlabeled_indices),
            'selected_count': len(selected_indices),
            'val_accuracy': val_metrics.get('accuracy', 0.0),
            'val_mca': val_metrics.get('mca', 0.0),
            'val_f1_macro': val_metrics.get('f1_macro', 0.0),
            'train_time': round_train_time,
        }
        self.round_metrics.append(round_result)

        selection_record = {
            'round': round_idx,
            'strategy': self.strategy.name,
            'budget': budget,
            'selected_indices': selected_indices,
            'selected_paths': [
                self.inner.dataset.samples[i][0] for i in selected_indices
            ] if selected_indices else [],
        }
        self.all_selections.append(selection_record)

        with open(os.path.join(round_dir, 'round_result.json'), 'w') as f:
            serializable = {k: v for k, v in round_result.items()}
            json.dump(serializable, f, indent=2)

        with open(os.path.join(round_dir, 'selections.json'), 'w') as f:
            json.dump(selection_record, f, indent=2)

    def _select_samples(self, budget):
        actual_budget = min(budget, len(self.inner.unlabeled_indices))

        unlabeled_ds = IndexTrackingDataset(self.inner.dataset, self.inner.unlabeled_indices)
        unlabeled_loader = DataLoader(
            unlabeled_ds, batch_size=self.inner.batch_size * 4,
            shuffle=False, num_workers=self.inner.num_workers,
        )

        logger.info(f"Scoring {len(self.inner.unlabeled_indices)} unlabeled samples "
                     f"with {self.strategy.name} strategy...")

        score_result = self.strategy.score(
            self.inner.trainer.model, unlabeled_loader, self.inner.device,
        )
        local_scores = score_result['scores']

        global_scores = {
            self.inner.unlabeled_indices[local_idx]: score
            for local_idx, score in local_scores.items()
        }

        selected_global = self.strategy.select(global_scores, actual_budget)

        return selected_global

    def _move_to_labeled(self, selected_indices):
        selected_set = set(selected_indices)
        self.inner.labeled_indices.extend(selected_indices)
        self.inner.unlabeled_indices = [
            i for i in self.inner.unlabeled_indices if i not in selected_set
        ]

    def _finalize_al(self):
        if self.inner.trainer is None:
            raise RuntimeError("Trainer not initialized before finalization.")

        with open(self.config_path, 'w') as f:
            json.dump(self.inner.trainer_cfg.to_dict(), f, indent=4)

        with open(self.metrics_path, 'w') as f:
            json.dump({
                'round_metrics': self.round_metrics,
                'epoch_metrics': self.inner.metrics,
            }, f, indent=4)

        with open(self.selection_log_path, 'w') as f:
            json.dump(self.all_selections, f, indent=2)

        self.inner.trainer.save_model(self.last_model_path)

        logger.info(f"Active Learning completed. Results written to {self.run_dir}")
        logger.info(f"Rounds: {len(self.round_metrics)}, "
                     f"Final labeled: {len(self.inner.labeled_indices)}")

        if self.round_metrics:
            final = self.round_metrics[-1]
            logger.info(f"Final round val accuracy: {final.get('val_accuracy', 0.0):.2f}%")

        for i, rm in enumerate(self.round_metrics):
            logger.info(f"  Round {rm['round']}: "
                         f"labeled={rm['labeled_count']}, "
                         f"selected={rm['selected_count']}, "
                         f"val_acc={rm['val_accuracy']:.2f}%")


def parse_args():
    parser = create_argument_parser("Train with Active Learning", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True), getattr(args, 'disable_coloring', False))
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = ActiveLearningPipeline(merged)
    pipeline.run()


if __name__ == "__main__":
    main()
