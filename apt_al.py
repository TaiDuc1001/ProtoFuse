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
    coerce_to_float,
    create_argument_parser,
    load_config_file,
    merge_configs,
    parse_override_arguments,
    process_parsed_args,
    set_global_seed,
)
from utils import (
    logger,
    setup_logging,
    plot_bald_distribution,
    plot_entropy_distribution,
)

AL_ARG_SCHEMA = {
    **ARG_SCHEMA,
}

def compute_entropy_scores(trainer, dataset, indices, batch_size, num_workers):
    if not indices:
        return defaultdict(list)

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_probs = []
    all_labels = []
    all_indices = []
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
            all_probs.append(probs.cpu())
            all_labels.extend(labels.cpu().tolist())
            all_indices.extend(batch_indices)

    all_probs = torch.cat(all_probs, dim=0)
    num_classes = all_probs.shape[1]

    class_probs = defaultdict(list)
    for i, lbl in enumerate(all_labels):
        class_probs[int(lbl)].append(all_probs[i])

    contextual_prior = torch.zeros(num_classes)
    top_k_fraction = 0.1

    for c in range(num_classes):
        if c in class_probs and len(class_probs[c]) > 0:
            class_prob_tensor = torch.stack(class_probs[c])
            class_scores = class_prob_tensor[:, c]
            k = max(1, int(len(class_scores) * top_k_fraction))
            top_k_scores, _ = torch.topk(class_scores, k)
            contextual_prior[c] = top_k_scores.mean()
        else:
            contextual_prior[c] = 1.0

    contextual_prior = torch.clamp(contextual_prior, min=eps)

    calibrated_probs = all_probs / contextual_prior.unsqueeze(0)
    calibrated_probs = calibrated_probs / calibrated_probs.sum(dim=1, keepdim=True)

    calibrated_entropy = -(calibrated_probs * torch.log(calibrated_probs + eps)).sum(dim=1)

    entropy_per_class = defaultdict(list)
    for i, (ent, lbl, idx) in enumerate(zip(calibrated_entropy.tolist(), all_labels, all_indices)):
        entropy_per_class[int(lbl)].append((float(ent), int(idx)))

    return entropy_per_class


def compute_bald_scores(trainer, dataset, indices, batch_size, num_workers, mc_samples=10):
    if not indices:
        return defaultdict(list)

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    eps = 1e-12
    all_mc_probs = []
    all_labels = []
    all_indices = []

    original_training_state = trainer.model.training
    trainer.model.train()

    for m in range(mc_samples):
        mc_probs = []
        position = 0

        with torch.no_grad():
            for images, labels in loader:
                batch_size_local = images.size(0)
                if m == 0:
                    batch_indices = indices[position:position + batch_size_local]
                    all_labels.extend(labels.cpu().tolist())
                    all_indices.extend(batch_indices)
                position += batch_size_local

                images = images.to(trainer.device)
                logits = trainer.model(images)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]

                probs = torch.softmax(logits, dim=1)
                mc_probs.append(probs.cpu())

        all_mc_probs.append(torch.cat(mc_probs, dim=0))

    trainer.model.train(original_training_state)

    stacked_probs = torch.stack(all_mc_probs, dim=0)
    mean_probs = stacked_probs.mean(dim=0)

    mean_entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=1)

    individual_entropies = -(stacked_probs * torch.log(stacked_probs + eps)).sum(dim=2)
    avg_entropy = individual_entropies.mean(dim=0)

    bald_scores = mean_entropy - avg_entropy

    bald_per_class = defaultdict(list)
    for i, (score, lbl, idx) in enumerate(zip(bald_scores.tolist(), all_labels, all_indices)):
        bald_per_class[int(lbl)].append((float(score), int(idx)))

    return bald_per_class


def select_high_bald_indices(bald_per_class, nshot):
    if nshot <= 0:
        return []

    selected = []
    for class_id in sorted(bald_per_class.keys()):
        scores = bald_per_class[class_id]
        if not scores:
            continue
        sorted_scores = sorted(scores, key=lambda item: item[0], reverse=True)
        selected.extend(idx for _, idx in sorted_scores[:nshot])
    return selected


def select_high_entropy_indices(entropy_per_class, nshot, fraction=0.5):
    if nshot <= 0:
        return []

    selected = []
    for class_id in sorted(entropy_per_class.keys()):
        scores = entropy_per_class[class_id]
        if not scores:
            continue
        sorted_scores = sorted(scores, key=lambda item: item[0], reverse=False)
        start_idx = int(len(sorted_scores) * fraction)
        chosen = sorted_scores[start_idx:start_idx + nshot]
        selected.extend(idx for _, idx in chosen)
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

    with torch.no_grad():
        for images, _ in loader:
            batch_size_local = images.size(0)
            batch_indices = indices[position:position + batch_size_local]
            position += batch_size_local

            images = images.to(trainer.device)

            visual_out = trainer.model.vis_encoder(images)
            if isinstance(visual_out, (list, tuple)) and len(visual_out) == 2:
                unpooled_levels, image_features = visual_out
            else:
                unpooled_levels = visual_out
                image_features = None

            if isinstance(unpooled_levels, list):
                unpooled = unpooled_levels[0]
            else:
                unpooled = unpooled_levels

            if isinstance(unpooled, (list, tuple)):
                unpooled = unpooled[0]

            unpooled_images = unpooled.permute(1, 0, 2)

            base_text_features = trainer.model._prepare_text_features().to(images.device).to(unpooled_images.dtype)
            text_features = base_text_features.unsqueeze(1).expand(-1, unpooled_images.shape[1], -1)

            for layer in trainer.model._prompt_layers_iter():
                text_features, _ = layer(unpooled_images, text_features)

            text_features = text_features.permute(1, 0, 2)
            text_features = F.normalize(text_features, dim=-1).to(torch.float32)

            if image_features is not None:
                image_features = F.normalize(image_features.to(torch.float32), dim=-1)
                logit_scale = trainer.model.logit_scale.exp()
                logits = logit_scale * F.cosine_similarity(image_features.unsqueeze(1), text_features, dim=-1)
                predicted = torch.argmax(logits, dim=1)
            else:
                norms = torch.norm(text_features, dim=-1)
                predicted = torch.argmax(norms, dim=1)

            tuned_features = text_features[torch.arange(text_features.size(0)), predicted].to(torch.float32)
            tuned_features_cpu = tuned_features.detach().cpu()

            for idx, vec in zip(batch_indices, tuned_features_cpu):
                embeddings[int(idx)] = vec

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
        logger.debug(f"ActiveLearningPipeline: rounds={self.rounds}")

        incr_value = self.training_cfg.get("increment_epochs", None)
        self.incr_epochs = coerce_to_int(incr_value, 0, key="training.increment_epochs")

        kshot_value = self.data_cfg.get("kshot", None)
        self.initial_kshot = coerce_to_int(kshot_value, 16, key="data.kshot")

        strategy_value = self.active_cfg.get("strategy", None)
        if strategy_value is not None and not isinstance(strategy_value, str):
            raise ValueError("active_learning.strategy must be a string or null.")
        self.strategy = strategy_value
        if self.strategy not in (None, "entropy", "random", "coreset", "bald"):
            raise ValueError("active_learning.strategy must be one of null, 'entropy', 'random', 'coreset', 'bald'.")

        alpha_cap_value = self.active_cfg.get("alpha_cap", None)
        self.alpha_cap = coerce_to_float(alpha_cap_value, 0.5, key="active_learning.alpha_cap")

        mc_samples_value = self.active_cfg.get("mc_samples", None)
        self.mc_samples = coerce_to_int(mc_samples_value, 10, key="active_learning.mc_samples")

        nshot_value = self.active_cfg.get("nshot", None)
        self.nshot = coerce_to_int(nshot_value, 0, key="active_learning.nshot")

        self.reset_model_per_round = bool(self.active_cfg.get("reset_model_per_round", False))
        self.plot_entropy_distribution = bool(self.config.get("plot_entropy_distribution", self.active_cfg.get("plot_entropy_distribution", False)))
        self.plot_bald_distribution = bool(self.config.get("plot_bald_distribution", self.active_cfg.get("plot_bald_distribution", False)))
        self.plot_coreset_embedding_umap = bool(self.config.get("plot_coreset_embedding_umap", self.active_cfg.get("plot_coreset_embedding_umap", False)))
        self.plot_coreset_embedding_tsne = bool(self.config.get("plot_coreset_embedding_tsne", self.active_cfg.get("plot_coreset_embedding_tsne", False)))
        logger.debug(f"ActiveLearningPipeline: strategy={self.strategy}, nshot={self.nshot}")

    def run(self):
        set_global_seed(self.seed)
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
            logger.warning("Validation split is empty; skipping validation metrics")

        self.classnames = list(self.dataset.classes)

        stats = {
            'total_images': len(self.dataset),
            'val_count': len(self.val_indices),
            'labeled_count': len(self.labeled_indices),
            'unlabeled_count': len(self.unlabeled_indices),
            'train_count': len(self.labeled_indices),
            'train_pool_size': len(self.labeled_indices) + len(self.unlabeled_indices),
        }
        logger.info(f"Dataset: {stats['total_images']} total, {stats['val_count']} val, {stats['labeled_count']} labeled, {stats['unlabeled_count']} unlabeled")

        val_percentage = (stats['val_count'] / stats['total_images'] * 100.0) if stats['total_images'] > 0 else 0.0
        trainer_cfg = self._build_trainer_config(stats, val_percentage)
        with open(self.config_path, 'w') as f:
            json.dump(trainer_cfg.to_dict(), f, indent=4)

    def _build_trainer_config(self, stats, val_percentage):
        trainer_cfg = super()._build_trainer_config(stats, val_percentage)
        meta = trainer_cfg.meta
        meta.active_learning = self.strategy
        meta.nshot = self.nshot
        meta.initial_kshot = self.initial_kshot
        meta.initial_labeled_size = stats.get('labeled_count', 0)
        meta.initial_unlabeled_size = stats.get('unlabeled_count', 0)
        meta.train_pool_size = stats.get('train_pool_size', stats.get('train_count', meta.get('train_pool_size', 0)))
        meta.al_selection_log = self.selection_log_path
        meta.reset_optimizer_per_round = self.active_cfg.get('reset_optimizer_per_round', True)
        meta.reset_model_per_round = self.reset_model_per_round

        self.trainer_cfg = trainer_cfg
        return trainer_cfg

    def _active_learning_loop(self):
        logger.info("Starting Active Learning")

        base_epochs = self._get_training_epochs()
        incr_epochs = self.incr_epochs
        self.total_epochs = sum(base_epochs + (r - 1) * incr_epochs for r in range(1, self.rounds + 1))

        for round_idx in range(1, self.rounds + 1):
            self._run_round(round_idx)

    def _run_round(self, round_idx):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before running rounds.")
        round_dir = os.path.join(self.run_dir, f'round_{round_idx:02d}')
        os.makedirs(round_dir, exist_ok=True)

        if len(self.labeled_indices) == 0:
            logger.warning(f"Round {round_idx}: no labeled samples; stopping")
            self.trainer_cfg.meta.completed_rounds = round_idx - 1
            return

        if round_idx == 1 and self._try_load_checkpoint():
            logger.info("Skipping round 1 training (loaded from checkpoint)")
        else:
            train_subset = Subset(self.dataset, list(self.labeled_indices))
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

            logger.info(f"Round {round_idx}/{self.rounds}: {len(self.labeled_indices)} labeled | {len(self.unlabeled_indices)} unlabeled")

            base_epochs = self._get_training_epochs()
            epochs_this_round = base_epochs + (round_idx - 1) * self.incr_epochs
            logger.debug(f"Epochs this round: {epochs_this_round}")

            for epoch_in_round in range(1, epochs_this_round + 1):
                self._run_epoch(epoch_in_round, epochs_this_round, train_loader, round_dir)

            if round_idx == 1:
                self._save_checkpoint()

        if self.strategy in ('entropy', 'random', 'coreset', 'bald') and round_idx < self.rounds:
            self._perform_active_selection(round_idx)
            self.trainer.reset_optimizer_scheduler()
            if self.reset_model_per_round:
                self.trainer.reset_model()

        self.trainer_cfg.meta.completed_rounds = round_idx

    def _perform_active_selection(self, round_idx):
        if self.dataset is None or self.trainer is None:
            raise RuntimeError("Pipeline not initialized before active selection.")

        strategy = self.strategy
        if strategy not in ('entropy', 'random', 'coreset', 'bald'):
            return

        if not self.unlabeled_indices:
            logger.warning(f"AL selection ({strategy}) skipped: no unlabeled samples")
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none\n")
            return

        if self.nshot <= 0:
            logger.warning(f"AL selection ({strategy}) skipped: nshot={self.nshot}")
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none\n")
            return

        need_entropy_plot = self.plot_entropy_distribution and strategy == 'entropy'
        need_bald_plot = self.plot_bald_distribution and strategy == 'bald'
        need_coreset_plot = (
            (self.plot_coreset_embedding_umap or self.plot_coreset_embedding_tsne) and strategy == 'coreset'
        )

        selection_plot_dir = None
        if need_entropy_plot or need_bald_plot or need_coreset_plot:
            selection_plot_dir = os.path.join(self.run_dir, f'round_{round_idx:02d}', 'selection_plots')
            os.makedirs(selection_plot_dir, exist_ok=True)

        raw_selected = []
        coreset_embeddings = None

        if strategy == 'entropy':
            entropy_scores = compute_entropy_scores(
                self.trainer, self.dataset, self.unlabeled_indices, self.batch_size, self.num_workers
            )
            if need_entropy_plot and selection_plot_dir is not None:
                entropy_plot_path = os.path.join(
                    selection_plot_dir, f'entropy_distribution_round_{round_idx:02d}.pdf'
                )
                plot_entropy_distribution(entropy_scores, round_idx, entropy_plot_path)
            raw_selected = select_high_entropy_indices(entropy_scores, self.nshot)
        elif strategy == 'bald':
            bald_scores = compute_bald_scores(
                self.trainer, self.dataset, self.unlabeled_indices, self.batch_size, self.num_workers, self.mc_samples
            )
            if need_bald_plot and selection_plot_dir is not None:
                bald_plot_path = os.path.join(
                    selection_plot_dir, f'bald_distribution_round_{round_idx:02d}.pdf'
                )
                plot_bald_distribution(bald_scores, round_idx, bald_plot_path)
            raw_selected = select_high_bald_indices(bald_scores, self.nshot)
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
        
        if not raw_selected:
            logger.warning(f"AL selection ({strategy}) selected no samples")
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none\n")
            return

        unlabeled_set = set(self.unlabeled_indices)
        seen = set()
        selected_indices = []
        for idx in raw_selected:
            if idx in unlabeled_set and idx not in seen:
                seen.add(idx)
                selected_indices.append(idx)

        if not selected_indices:
            logger.warning(f"AL selection ({strategy}): samples already labeled")
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none\n")
            return

        existing_labeled = set(self.labeled_indices)
        new_indices = [idx for idx in selected_indices if idx not in existing_labeled]

        if not new_indices:
            logger.warning(f"AL selection ({strategy}): all already labeled")
            with open(self.selection_log_path, 'a') as f:
                f.write(f"round {round_idx}: none\n")
            return

        prev_labeled = len(self.labeled_indices)
        new_set = set(new_indices)
        self.labeled_indices.extend(new_indices)
        self.unlabeled_indices = [idx for idx in self.unlabeled_indices if idx not in new_set]
        after_labeled = len(self.labeled_indices)

        logger.info(f"Selected {len(new_indices)} new samples. Labeled: {after_labeled} (was {prev_labeled})")

        if new_indices:
            new_subset = Subset(self.dataset, new_indices)
            new_loader = DataLoader(new_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            correct = 0
            total = 0
            self.trainer.model.eval()
            with torch.no_grad():
                for images, labels in new_loader:
                    images = images.to(self.trainer.device)
                    labels = labels.to(self.trainer.device)
                    logits = self.trainer.model(images)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            accuracy = correct / total * 100 if total > 0 else 0
            logger.debug(f"Newly selected: {correct}/{total} correct ({accuracy:.2f}%)")

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
    setup_logging(getattr(args, 'debug', False))
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = ActiveLearningPipeline(merged)
    pipeline.run()


if __name__ == "__main__":
    main()