import argparse
import json
import math
import os
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from apt import (
    ConfigNode,
    APTTrainingPipeline,
    ARG_SCHEMA,
    coerce_to_int,
    create_argument_parser,
    load_config_file,
    merge_configs,
    parse_override_arguments,
    process_parsed_args,
    set_nested_value,
)

AL_ARG_SCHEMA = {
    **ARG_SCHEMA,
}


def compute_conflict_scores_cache(trainer, dataset, indices, batch_size, num_workers):
    if not indices:
        return defaultdict(list)

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    conflict_per_class = defaultdict(list)
    position = 0
    trainer.model.eval()

    with torch.no_grad():
        for images, labels in loader:
            batch_size_local = images.size(0)
            batch_indices = indices[position:position + batch_size_local]
            position += batch_size_local
            images = images.to(trainer.device)

            apt_logits = trainer.model(images)
            if isinstance(apt_logits, (list, tuple)):
                apt_logits = apt_logits[0]
            prob_apt = torch.softmax(apt_logits, dim=1)

            visual_out = trainer.model.vis_encoder(images)
            img_feats = visual_out[1]
            img_feats = F.normalize(img_feats, dim=-1)

            cache_logits_only = trainer.cache_adapter.get_cache_logits(img_feats)
            prob_cache = torch.softmax(cache_logits_only, dim=1)

            kl_div = F.kl_div(prob_apt.log(), prob_cache, reduction='none', log_target=False).sum(dim=1)

            for score, lbl, idx in zip(kl_div.cpu().tolist(), labels.cpu().tolist(), batch_indices):
                conflict_per_class[int(lbl)].append((float(score), int(idx)))

    return conflict_per_class


def select_global_topk_indices(conflict_per_class, nshot):
    if nshot <= 0:
        return []

    all_candidates = []
    for scores in conflict_per_class.values():
        all_candidates.extend(scores)

    if not all_candidates:
        return []

    k = nshot * max(1, len(conflict_per_class))
    sorted_candidates = sorted(all_candidates, key=lambda item: item[0], reverse=True)
    return [idx for _, idx in sorted_candidates[:k]]


def compute_entropy_scores(trainer, dataset, indices, batch_size, num_workers):
    if not indices:
        return defaultdict(list)

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    entropy_per_class = defaultdict(list)
    position = 0
    eps = 1e-12

    trainer.model.eval()

    with torch.no_grad():
        for images, labels in loader:
            batch_size_local = images.size(0)
            batch_indices = indices[position:position + batch_size_local]
            position += batch_size_local

            images = images.to(trainer.device)
            logits = trainer.model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + eps)).sum(dim=1)

            for ent, lbl, idx in zip(entropy.cpu().tolist(), labels.cpu().tolist(), batch_indices):
                entropy_per_class[int(lbl)].append((float(ent), int(idx)))

    return entropy_per_class


def select_high_entropy_indices(entropy_per_class, nshot):
    if nshot <= 0:
        return []

    selected = []
    for class_id in sorted(entropy_per_class.keys()):
        scores = entropy_per_class[class_id]
        if not scores:
            continue
        sorted_scores = sorted(scores, key=lambda item: item[0], reverse=True)
        selected.extend(idx for _, idx in sorted_scores[:nshot])
    return selected


def _group_indices_by_class(dataset, indices):
    grouped = defaultdict(list)
    for idx in indices:
        _, class_idx = dataset.samples[idx]
        grouped[int(class_idx)].append(int(idx))
    return grouped


def select_random_indices(dataset, indices, nshot, seed=None):
    if nshot <= 0 or not indices:
        return []

    rng = random.Random(seed)
    grouped = _group_indices_by_class(dataset, indices)
    selected = []

    for class_id in sorted(grouped.keys()):
        candidates = grouped[class_id]
        if not candidates:
            continue
        k = min(nshot, len(candidates))
        if k == len(candidates):
            chosen = list(candidates)
        else:
            chosen = rng.sample(candidates, k)
        selected.extend(chosen)

    return selected


def compute_coreset_embeddings(trainer, dataset, indices, batch_size, num_workers):
    if not indices:
        return {}

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    embeddings = {}
    position = 0

    original_training_state = trainer.model.training
    trainer.model.eval()

    old_mode = trainer.model.cfg.get('mode', None) if isinstance(trainer.model.cfg, dict) else None
    if isinstance(trainer.model.cfg, dict):
        trainer.model.cfg['mode'] = 'features'

    with torch.no_grad():
        for images, _ in loader:
            batch_size_local = images.size(0)
            batch_indices = indices[position:position + batch_size_local]
            position += batch_size_local

            images = images.to(trainer.device)

            outputs = trainer.model(images)
            if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                logits, text_features = outputs
            else:
                raise RuntimeError("Model in 'features' mode is expected to return (logits, text_features).")

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            predicted = torch.argmax(logits, dim=1)

            visual_output = trainer.model.vis_encoder(images)
            _, image_features = visual_output

            image_features = image_features.to(torch.float32)
            tuned_features = text_features[torch.arange(text_features.size(0)), predicted].to(torch.float32)

            image_features = F.normalize(image_features, dim=-1)
            tuned_features = F.normalize(tuned_features, dim=-1)

            combined = torch.cat([image_features, tuned_features], dim=-1)
            combined_cpu = combined.detach().cpu().to(torch.float32)

            for idx, vec in zip(batch_indices, combined_cpu):
                embeddings[int(idx)] = vec

    if isinstance(trainer.model.cfg, dict):
        if old_mode is None:
            trainer.model.cfg.pop('mode', None)
        else:
            trainer.model.cfg['mode'] = old_mode

    trainer.model.train(original_training_state)
    return embeddings


def _coreset_greedy_selection(candidates, centers, embeddings, k):
    if k <= 0 or not candidates:
        return []

    candidate_pool = list(candidates)
    selected = []

    center_vectors = [embeddings[idx] for idx in centers if idx in embeddings]

    if not center_vectors:
        candidate_matrix = torch.stack([embeddings[idx] for idx in candidate_pool]).to(torch.float32)
        norms = torch.norm(candidate_matrix, dim=1)
        first_choice = int(torch.argmax(norms).item())
        first_idx = candidate_pool.pop(first_choice)
        selected.append(first_idx)
        center_vectors = [embeddings[first_idx]]

    center_matrix = torch.stack(center_vectors).to(torch.float32)

    while candidate_pool and len(selected) < k:
        candidate_matrix = torch.stack([embeddings[idx] for idx in candidate_pool]).to(torch.float32)
        distances = torch.cdist(candidate_matrix, center_matrix)
        min_distances, _ = torch.min(distances, dim=1)
        next_choice = int(torch.argmax(min_distances).item())
        chosen_idx = candidate_pool.pop(next_choice)
        selected.append(chosen_idx)
        center_matrix = torch.cat([center_matrix, embeddings[chosen_idx].unsqueeze(0).to(torch.float32)], dim=0)

    return selected


def select_coreset_indices(trainer, dataset, labeled_indices, unlabeled_indices, nshot, batch_size, num_workers):
    if nshot <= 0 or not unlabeled_indices:
        return []

    grouped_unlabeled = _group_indices_by_class(dataset, unlabeled_indices)
    grouped_labeled = _group_indices_by_class(dataset, labeled_indices)

    all_needed_indices = set(unlabeled_indices) | set(labeled_indices)
    embeddings = compute_coreset_embeddings(trainer, dataset, list(all_needed_indices), batch_size, num_workers)

    selected = []
    for class_id in sorted(grouped_unlabeled.keys()):
        candidates = grouped_unlabeled[class_id]
        if not candidates:
            continue
        k = min(nshot, len(candidates))
        centers = grouped_labeled.get(class_id, [])
        chosen = _coreset_greedy_selection(candidates, centers, embeddings, k)
        selected.extend(chosen)

    return selected


class ActiveLearningPipeline(APTTrainingPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.active_cfg = self.config.get('active_learning', ConfigNode())

        rounds_value = self.active_cfg.get("rounds", None)
        self.rounds = max(1, coerce_to_int(rounds_value, 1, key="active_learning.rounds"))

        incr_value = self.training_cfg.get("increment_epochs", None)
        self.incr_epochs = coerce_to_int(incr_value, 0, key="training.increment_epochs")

        kshot_value = self.data_cfg.get("kshot", None)
        self.initial_kshot = coerce_to_int(kshot_value, 16, key="data.kshot")

        strategy_value = self.active_cfg.get("strategy", None)
        if strategy_value is not None and not isinstance(strategy_value, str):
            raise ValueError("active_learning.strategy must be a string or null.")
        self.strategy = strategy_value
        if self.strategy not in (None, "entropy", "random", "coreset", "conflict"):
            raise ValueError("active_learning.strategy must be one of null, 'entropy', 'random', 'coreset', 'conflict'.")

        nshot_value = self.active_cfg.get("nshot", None)
        self.nshot = coerce_to_int(nshot_value, 0, key="active_learning.nshot")

    def run(self):
        self._prepare_directories()
        self._load_dataset()
        self._split_dataset()
        self._initialize_trainer()
        self._active_learning_loop()
        self._finalize()

    def _prepare_directories(self):
        super()._prepare_directories()
        with open(self.selection_log_path, 'w') as f:
            f.write('')

    def _split_dataset(self):
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before splitting.")
        samples_by_class_idx = defaultdict(list)
        for idx, (_, class_idx) in enumerate(self.dataset.samples):
            samples_by_class_idx[class_idx].append(idx)

        rng = random.Random(self.seed)
        val_indices = []
        labeled_indices = []
        unlabeled_indices = []

        for class_idx in sorted(samples_by_class_idx.keys()):
            class_samples = list(samples_by_class_idx[class_idx])
            class_samples.sort()
            rng.shuffle(class_samples)

            val_count = int(math.floor(len(class_samples) * self.val_fraction))
            if self.val_fraction > 0 and val_count == 0 and len(class_samples) > 0:
                val_count = 1

            val_part = class_samples[:val_count]
            remaining = class_samples[val_count:]

            labeled_count = min(len(remaining), self.initial_kshot)
            labeled_part = remaining[:labeled_count]
            unlabeled_part = remaining[labeled_count:]

            val_indices.extend(val_part)
            labeled_indices.extend(labeled_part)
            unlabeled_indices.extend(unlabeled_part)

        self.val_indices = val_indices
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices

        if len(self.val_indices) > 0:
            val_ds = Subset(self.dataset, self.val_indices)
            self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        else:
            print("Warning: validation split is empty; skipping validation metrics.")

        self.classnames = list(self.dataset.classes)

        stats = {
            'total_images': len(self.dataset),
            'val_count': len(self.val_indices),
            'labeled_count': len(self.labeled_indices),
            'unlabeled_count': len(self.unlabeled_indices),
            'train_count': len(self.labeled_indices) + len(self.unlabeled_indices),
        }
        print(f"Dataset loaded: {stats['total_images']} total images.")
        val_percentage = (stats['val_count'] / stats['total_images'] * 100.0) if stats['total_images'] > 0 else 0.0
        print(f"Validation split: {stats['val_count']} images ({val_percentage:.2f}%).")
        print(
            f"Train pool size: {stats['labeled_count'] + stats['unlabeled_count']} images "
            f"({stats['labeled_count']} labeled, {stats['unlabeled_count']} unlabeled)."
        )

        trainer_cfg = self._build_trainer_config(stats, val_percentage)
        with open(self.config_path, 'w') as f:
            json.dump(trainer_cfg.to_dict(), f, indent=4)

        with open(self.log_file, 'w') as f:
            f.write(f"Config: {json.dumps(trainer_cfg.to_dict(), indent=2)}\n\n")
            f.write('=' * 50 + '\n')

    def _build_trainer_config(self, stats, val_percentage):
        trainer_cfg = super()._build_trainer_config(stats, val_percentage)
        meta = trainer_cfg.meta
        meta.active_learning = self.strategy
        meta.nshot = self.nshot
        meta.initial_kshot = self.initial_kshot
        meta.initial_labeled_size = stats.get('labeled_count', 0)
        meta.initial_unlabeled_size = stats.get('unlabeled_count', 0)
        meta.train_pool_size = stats.get('train_count', meta.get('train_pool_size', 0))
        meta.al_selection_log = self.selection_log_path
        meta.reset_optimizer_per_round = True
        self.trainer_cfg = trainer_cfg
        return trainer_cfg

    def _active_learning_loop(self):
        print('\n')
        print('=' * 50)

        self.trainer.update_cache_memory(self.dataset, self.labeled_indices)  # type: ignore

        for round_idx in range(1, self.rounds + 1):
            self._run_round(round_idx)

    def _run_round(self, round_idx):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before running rounds.")
        round_dir = os.path.join(self.run_dir, f'round_{round_idx:02d}')
        os.makedirs(round_dir, exist_ok=True)

        if len(self.labeled_indices) == 0:
            msg = f"Round {round_idx}: no labeled samples available; stopping training."
            print(msg)
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')
                self.trainer_cfg.meta.completed_rounds = round_idx - 1
            return

        train_subset = Subset(self.dataset, list(self.labeled_indices))
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        msg = (
            f"Starting round {round_idx}/{self.rounds}: {len(self.labeled_indices)} labeled | "
            f"{len(self.unlabeled_indices)} unlabeled"
        )
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')

        base_epochs_value = self.training_cfg.get('epochs', None)
        base_epochs = coerce_to_int(base_epochs_value, 150, key='training.epochs')
        epochs_this_round = base_epochs + (round_idx - 1) * self.incr_epochs

        with open(self.log_file, 'a') as f:
            f.write(f"  Epochs this round: {epochs_this_round} (base: {base_epochs})\n")

        for epoch_in_round in range(1, epochs_this_round + 1):
            self._run_epoch(round_idx, epoch_in_round, epochs_this_round, train_loader, round_dir)

        if self.strategy in ('entropy', 'random', 'coreset', 'conflict') and round_idx < self.rounds:
            self._perform_active_selection(round_idx)
            self.trainer.reset_optimizer_scheduler()
            self.trainer.update_cache_memory(self.dataset, self.labeled_indices)

        self.trainer_cfg.meta.completed_rounds = round_idx

    def _perform_active_selection(self, round_idx):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before active selection.")

        strategy = self.strategy
        if strategy not in ('entropy', 'random', 'coreset', 'conflict'):
            return

        if not self.unlabeled_indices:
            skip_msg = (
                f"Active learning selection ({strategy}) (round {round_idx} -> {round_idx + 1}) skipped (no unlabeled samples)."
            )
            print(skip_msg)
            with open(self.log_file, 'a') as f:
                f.write(skip_msg + '\n')
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none" + '\n')
            return

        if self.nshot <= 0:
            no_shot_msg = (
                f"Active learning selection ({strategy}) (round {round_idx} -> {round_idx + 1}) skipped (nshot={self.nshot})."
            )
            print(no_shot_msg)
            with open(self.log_file, 'a') as f:
                f.write(no_shot_msg + '\n')
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none" + '\n')
            return

        raw_selected = []

        if strategy == 'entropy':
            entropy_scores = compute_entropy_scores(
                self.trainer, self.dataset, self.unlabeled_indices, self.batch_size, self.num_workers
            )
            raw_selected = select_high_entropy_indices(entropy_scores, self.nshot)
        elif strategy == 'random':
            seed = self.seed + round_idx
            raw_selected = select_random_indices(
                self.dataset, self.unlabeled_indices, self.nshot, seed=seed
            )
        elif strategy == 'coreset':
            raw_selected = select_coreset_indices(
                self.trainer,
                self.dataset,
                self.labeled_indices,
                self.unlabeled_indices,
                self.nshot,
                self.batch_size,
                self.num_workers
            )
        elif strategy == 'conflict':
            conflict_scores = compute_conflict_scores_cache(
                self.trainer, self.dataset, self.unlabeled_indices, self.batch_size, self.num_workers
            )
            raw_selected = select_global_topk_indices(conflict_scores, self.nshot)

        if not raw_selected:
            empty_msg = (
                f"Active learning selection ({strategy}) (round {round_idx} -> {round_idx + 1}) selected no samples."
            )
            print(empty_msg)
            with open(self.log_file, 'a') as f:
                f.write(empty_msg + '\n')
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none" + '\n')
            return

        unlabeled_set = set(self.unlabeled_indices)
        seen = set()
        selected_indices = []
        for idx in raw_selected:
            if idx in unlabeled_set and idx not in seen:
                seen.add(idx)
                selected_indices.append(idx)

        if not selected_indices:
            duplicate_msg = (
                f"Active learning selection ({strategy}) (round {round_idx} -> {round_idx + 1}): "
                "suggested samples were already labeled."
            )
            print(duplicate_msg)
            with open(self.log_file, 'a') as f:
                f.write(duplicate_msg + '\n')
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none" + '\n')
            return

        existing_labeled = set(self.labeled_indices)
        new_indices = [idx for idx in selected_indices if idx not in existing_labeled]

        if not new_indices:
            no_new_msg = (
                f"Active learning selection ({strategy}) (round {round_idx} -> {round_idx + 1}): "
                "all suggested samples were already labeled."
            )
            print(no_new_msg)
            with open(self.log_file, 'a') as f:
                f.write(no_new_msg + '\n')
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none" + '\n')
            return

        prev_labeled = len(self.labeled_indices)
        new_set = set(new_indices)
        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = [idx for idx in self.unlabeled_indices if idx not in new_set]
        after_labeled = len(self.labeled_indices)

        summary = f"Selected {len(new_indices)} new samples. Labeled: {after_labeled} (was {prev_labeled})."
        print(summary)
        with open(self.log_file, 'a') as f:
            f.write(summary + '\n')

        round_selected_paths = [os.path.abspath(self.dataset.samples[idx][0]) for idx in new_indices]
        with open(self.selection_log_path, 'a') as f:
            line = ';'.join(round_selected_paths) if round_selected_paths else f"round {round_idx}: none"
            f.write(line + '\n')


def parse_args():
    parser = create_argument_parser("Train APT model with active learning", AL_ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, AL_ARG_SCHEMA, overrides)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = ActiveLearningPipeline(merged)
    pipeline.run()


if __name__ == "__main__":
    main()