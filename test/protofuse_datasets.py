import csv
import json
import math
import os
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clip import clip
from src.models.apt import CUSTOM_TEMPLATES
from utils import (
    CLIPFeatureCache,
    DEFAULT_ARG_SCHEMA,
    create_argument_parser,
    get_config_value,
    iter_dataset_configs,
    load_config_file,
    log_experiment_start,
    logger,
    merge_configs,
    parse_override_arguments,
    process_parsed_args,
    setup_logging,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CLIP_MODEL_PATH = REPO_ROOT / "models" / "ViT-B-16.pt"

DEFAULT_DEVICE = "cuda:0"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "protofuse_datasets"
DEFAULT_CACHE_DIR = CLIPFeatureCache.DEFAULT_CACHE_DIR

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

DEFAULT_ALPHA_STEPS = 101
DEFAULT_BETA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


DEVICE = DEFAULT_DEVICE
BATCH_SIZE = 128
NUM_WORKERS = 4
CACHE_DIR = DEFAULT_CACHE_DIR


def _float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().cpu().item()
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _safe_name(value):
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _stats(prefix, values):
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, dtype=torch.float32, device=DEVICE)
    values = values.detach().float()
    if values.numel() == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_q25": None,
            f"{prefix}_median": None,
            f"{prefix}_q75": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_mean": _float(values.mean()),
        f"{prefix}_std": _float(values.std(unbiased=False)),
        f"{prefix}_min": _float(values.min()),
        f"{prefix}_q25": _float(torch.quantile(values, 0.25)),
        f"{prefix}_median": _float(torch.quantile(values, 0.50)),
        f"{prefix}_q75": _float(torch.quantile(values, 0.75)),
        f"{prefix}_max": _float(values.max()),
    }


def _ratio(numerator, denominator):
    denominator = int(denominator)
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def normalize_dataset_root(root):
    path = Path(root).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def dataset_spec_from_config(config):
    dataset_root = get_config_value(config, "data.root")
    if not dataset_root:
        raise ValueError("Config must set data.root for this single-dataset diagnostics script.")
    dataset_name = get_config_value(config, "data.dataset_name") or infer_dataset_name(dataset_root)
    return {
        "key": _safe_name(dataset_name).lower(),
        "display_name": dataset_name,
        "template_name": dataset_name,
        "path": normalize_dataset_root(dataset_root),
    }


def load_clip():
    model = torch.jit.load(str(CLIP_MODEL_PATH), map_location="cpu").eval()
    state_dict = model.state_dict()
    model = clip.build_model(state_dict)
    model = model.to(DEVICE).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def clip_transform_spec():
    return {
        "resize": 256,
        "center_crop": 224,
        "normalize_mean": list(CLIP_MEAN),
        "normalize_std": list(CLIP_STD),
    }


def load_clip_features(clip_model, dataset, dataset_name, classnames, args, dataset_id):
    template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")
    cache = CLIPFeatureCache(str(CACHE_DIR), enabled=args.feature_cache_enabled)
    transform_spec = clip_transform_spec()

    if args.force_cache and args.feature_cache_enabled:
        cache_key, _ = cache.compute_cache_key(
            dataset,
            dataset_id,
            classnames,
            template,
            args.backbone,
            args.precision,
            CLIP_MEAN,
            CLIP_STD,
            transform_spec,
        )
        cache_path = Path(cache._cache_path(cache_key))
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Removed CLIP feature cache ({dataset_id}) at {cache_path}")

    payload = cache.load_or_compute(
        dataset=dataset,
        dataset_id=dataset_id,
        clip_model=clip_model,
        classnames=classnames,
        template=template,
        backbone=args.backbone,
        precision=args.precision,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        clip_mean=CLIP_MEAN,
        clip_std=CLIP_STD,
        transform_spec=transform_spec,
    )
    return payload["image_features"], payload["labels"]


def split_by_class(dataset, val_size, kshot, seed):
    samples_by_class = defaultdict(list)
    for idx, (_, class_idx) in enumerate(dataset.samples):
        samples_by_class[class_idx].append(idx)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for class_idx in sorted(samples_by_class.keys()):
        class_samples = list(samples_by_class[class_idx])
        class_samples.sort()
        rng.shuffle(class_samples)

        val_count = int(math.floor(len(class_samples) * val_size))
        if val_size > 0 and val_count == 0 and len(class_samples) > 0:
            val_count = 1

        val_part = class_samples[:val_count]
        train_candidates = class_samples[val_count:]

        val_indices.extend(val_part)
        train_indices.extend(train_candidates[:kshot])

    return train_indices, val_indices


def sample_kshot_by_class(dataset, kshot, seed):
    samples_by_class = defaultdict(list)
    for idx, (_, class_idx) in enumerate(dataset.samples):
        samples_by_class[class_idx].append(idx)

    rng = random.Random(seed)
    train_indices = []
    for class_idx in sorted(samples_by_class.keys()):
        class_samples = list(samples_by_class[class_idx])
        class_samples.sort()
        rng.shuffle(class_samples)
        if kshot > 0:
            train_indices.extend(class_samples[:kshot])
        else:
            train_indices.extend(class_samples)
    return train_indices


def infer_dataset_name(dataset_path):
    path_name = Path(dataset_path).name.lower()
    if "cub" in path_name:
        return "CUB-200-2011"
    if "flower" in path_name:
        return "Flowers102"
    if "aircraft" in path_name:
        return "FGVCAircraft"
    if "car" in path_name:
        return "StanfordCars"
    if "dog" in path_name or "pet" in path_name:
        return "OxfordPets"
    if "food" in path_name:
        return "Food-101"
    if "euro" in path_name:
        return "EuroSAT"
    if "ucf" in path_name:
        return "UCF101"
    if "dtd" in path_name or "texture" in path_name:
        return "DTD"
    return Path(dataset_path).name


def get_task_text_features(clip_model, classnames, task_classes, dataset_name):
    sorted_classes = sorted(task_classes)
    template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")
    prompts = [template.format(classnames[c].replace("_", " ")) for c in sorted_classes]

    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(DEVICE)
        text_features = clip_model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)

    class_remap = {c: i for i, c in enumerate(sorted_classes)}
    return text_features, class_remap, prompts


def remap_labels(labels, class_remap, device=None):
    remapped = torch.tensor([class_remap[label.item()] for label in labels], dtype=torch.long)
    if device is not None:
        remapped = remapped.to(device, non_blocking=True)
    return remapped


def prepare_split_tensors(all_features, all_labels, train_indices, val_indices, class_remap):
    train_features = all_features[train_indices].to(DEVICE, non_blocking=True)
    val_features = all_features[val_indices].to(DEVICE, non_blocking=True)
    train_labels_raw = all_labels[train_indices]
    val_labels_raw = all_labels[val_indices]
    remapped_train = remap_labels(train_labels_raw, class_remap, device=DEVICE)
    remapped_val = remap_labels(val_labels_raw, class_remap, device=DEVICE)
    return train_features, val_features, remapped_train, remapped_val


def prepare_train_eval_tensors(train_all_features, train_all_labels, train_indices, eval_features, eval_labels, class_remap):
    train_features = train_all_features[train_indices].to(DEVICE, non_blocking=True)
    val_features = eval_features.to(DEVICE, non_blocking=True)
    train_labels_raw = train_all_labels[train_indices]
    remapped_train = remap_labels(train_labels_raw, class_remap, device=DEVICE)
    remapped_val = remap_labels(eval_labels, class_remap, device=DEVICE)
    return train_features, val_features, remapped_train, remapped_val


def weighted_visual_centroid(class_features, text_prototype):
    class_features = class_features.to(DEVICE)
    text_prototype = text_prototype.to(DEVICE)
    similarities = F.cosine_similarity(
        F.normalize(class_features, dim=-1),
        F.normalize(text_prototype, dim=-1).unsqueeze(0),
        dim=-1,
    ).clamp_min(0.0)
    sim_sum = similarities.sum()
    if sim_sum <= 1e-12:
        weights = torch.full_like(similarities, 1.0 / max(1, similarities.numel()))
    else:
        weights = similarities / sim_sum
    return F.normalize((weights.unsqueeze(-1) * class_features).sum(dim=0), dim=-1)


def build_visual_centroids(train_features, remapped_train_labels, text_features, num_classes):
    embed_dim = text_features.shape[-1]
    centroids = torch.zeros(num_classes, embed_dim, device=DEVICE, dtype=train_features.dtype)
    for class_idx in range(num_classes):
        mask = remapped_train_labels == class_idx
        if mask.any():
            centroids[class_idx] = weighted_visual_centroid(train_features[mask], text_features[class_idx])
    return centroids


def shots_per_class(labels, num_classes):
    counts = torch.bincount(labels.to("cpu"), minlength=num_classes)
    return int(counts.min().item())


def class_indices(labels, num_classes):
    indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(labels.detach().cpu().tolist()):
        indices[label].append(idx)
    return indices


def masked_margin(logits, labels):
    labels = labels.to(logits.device)
    source_score = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    other_logits = logits.clone()
    other_logits.scatter_(1, labels.view(-1, 1), -float("inf"))
    nearest_wrong = other_logits.max(dim=1).values
    return source_score - nearest_wrong, source_score, nearest_wrong


def normalized_entropy(logits, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    if logits.shape[-1] <= 1:
        return torch.zeros_like(entropy)
    return entropy / math.log(logits.shape[-1])


def accuracy_from_preds(preds, labels):
    return _float(preds.eq(labels).float().mean() * 100.0)


def evaluate_alpha(T, V, alpha, eval_features, eval_labels):
    eval_norm = F.normalize(eval_features, dim=-1)
    prototypes = F.normalize((1.0 - alpha) * T + alpha * V, dim=-1)
    logits = eval_norm @ prototypes.T
    preds = logits.argmax(dim=-1)
    return preds, accuracy_from_preds(preds, eval_labels)


def curve_knee(values):
    values = values.float()
    span = values.max() - values.min()
    if span <= 1e-12:
        return None, 0.0, 0.0
    y = (values - values.min()) / (span + 1e-12)
    x = torch.linspace(0.0, 1.0, len(values), device=values.device)
    knee_scores = y - x
    knee_idx = int(knee_scores.argmax().item())
    return knee_idx, _float(knee_scores[knee_idx]), _float(span)


def centroid_mix_neighbors(T, V, mode):
    if mode == "vv":
        similarity = V @ V.T
    elif mode == "tt":
        similarity = T @ T.T
    elif mode == "hybrid":
        similarity = 0.5 * (V @ V.T) + 0.5 * (T @ T.T)
    else:
        raise ValueError(f"Unknown neighbor mode: {mode}")
    similarity = similarity.clone()
    similarity.fill_diagonal_(-float("inf"))
    return similarity.argmax(dim=1)


def centroid_mix_net_curve(T, V, neighbors, beta, alphas, num_classes):
    labels = torch.arange(num_classes, device=DEVICE)
    pseudo_features = F.normalize((1.0 - beta) * V + beta * V[neighbors], dim=-1)
    text_preds = (pseudo_features @ T.T).argmax(dim=-1)
    text_correct = text_preds.eq(labels)

    net_scores = []
    for alpha in alphas:
        prototypes = F.normalize((1.0 - alpha) * T + alpha * V, dim=-1)
        fused_preds = (pseudo_features @ prototypes.T).argmax(dim=-1)
        fused_correct = fused_preds.eq(labels)
        rescue = (~text_correct) & fused_correct
        damage = text_correct & ~fused_correct
        net_scores.append(rescue.sum().float() - damage.sum().float())
    return torch.stack(net_scores)


def centroid_mix_alpha(T, V, num_classes, alphas, beta_values):
    beta_values = sorted({round(float(beta), 6) for beta in beta_values if 0.0 < float(beta) < 0.5})
    if 0.45 not in beta_values:
        beta_values.append(0.45)
        beta_values.sort()
    if num_classes < 2 or not beta_values:
        return 0.0

    best = {"score": -float("inf"), "alpha": 0.0}
    for mode in ("vv", "tt", "hybrid"):
        neighbors = centroid_mix_neighbors(T, V, mode)
        for beta in beta_values:
            net_curve = centroid_mix_net_curve(T, V, neighbors, beta, alphas, num_classes)
            knee_idx, knee_strength, signal_span = curve_knee(net_curve)
            if knee_idx is None:
                continue
            amplitude = signal_span / max(1, num_classes)
            quality = knee_strength * amplitude
            alpha = _float(alphas[knee_idx])
            if quality > best["score"] or (quality == best["score"] and alpha < best["alpha"]):
                best = {"score": quality, "alpha": alpha}

    return best["alpha"] if best["score"] > 0.0 else 0.0


def select_protofuse_alpha(T, V, train_features, train_labels, num_classes, alphas, beta_values):
    indices = class_indices(train_labels, num_classes)
    min_shots = min(len(idxs) for idxs in indices)

    if min_shots < 2:
        return centroid_mix_alpha(T, V, num_classes, alphas, beta_values)

    class_feat = torch.stack([train_features[indices[c][:min_shots]].to(DEVICE) for c in range(num_classes)])
    net_scores = torch.zeros(len(alphas), device=DEVICE)
    targets = torch.arange(num_classes, device=DEVICE)

    for hold_idx in range(min_shots):
        held = F.normalize(class_feat[:, hold_idx, :], dim=-1)
        keep = torch.arange(min_shots, device=DEVICE) != hold_idx
        v_minus = torch.stack(
            [weighted_visual_centroid(class_feat[c, keep], T[c]) for c in range(num_classes)]
        )

        text_preds = (held @ T.T).argmax(dim=-1)
        text_correct = text_preds.eq(targets)

        refined = F.normalize(
            (1.0 - alphas).view(-1, 1, 1) * T + alphas.view(-1, 1, 1) * v_minus,
            dim=-1,
        )
        fused_preds = torch.einsum("cd,akd->ack", held, refined).argmax(dim=-1)
        fused_correct = fused_preds.eq(targets.view(1, -1))

        rescue = (~text_correct).view(1, -1) & fused_correct
        damage = text_correct.view(1, -1) & ~fused_correct
        net_scores += rescue.sum(dim=1).float() - damage.sum(dim=1).float()

    return _float(alphas[net_scores.argmax()])


def endpoint_state(T, V, eval_features, eval_labels, margin_low_threshold):
    eval_norm = F.normalize(eval_features, dim=-1)
    text_logits = eval_norm @ T.T
    visual_logits = eval_norm @ V.T

    text_pred = text_logits.argmax(dim=-1)
    visual_pred = visual_logits.argmax(dim=-1)

    text_margin, text_same, text_nearest_wrong = masked_margin(text_logits, eval_labels)
    visual_margin, visual_same, visual_nearest_wrong = masked_margin(visual_logits, eval_labels)

    text_entropy = normalized_entropy(text_logits)
    visual_entropy = normalized_entropy(visual_logits)

    result = {
        "text_logits": text_logits,
        "visual_logits": visual_logits,
        "text_pred": text_pred,
        "visual_pred": visual_pred,
        "text_correct": text_pred.eq(eval_labels),
        "visual_correct": visual_pred.eq(eval_labels),
        "text_margin": text_margin,
        "visual_margin": visual_margin,
        "text_same_class_sim": text_same,
        "text_nearest_wrong_sim": text_nearest_wrong,
        "visual_same_class_sim": visual_same,
        "visual_nearest_wrong_sim": visual_nearest_wrong,
        "text_entropy": text_entropy,
        "visual_entropy": visual_entropy,
    }
    result["summary"] = {
        "text_only_acc": accuracy_from_preds(text_pred, eval_labels),
        "text_margin_mean": _float(text_margin.mean()),
        "text_margin_std": _float(text_margin.std(unbiased=False)),
        "text_margin_low_ratio": _float((text_margin <= margin_low_threshold).float().mean()),
        "text_entropy_mean": _float(text_entropy.mean()),
        "visual_only_acc": accuracy_from_preds(visual_pred, eval_labels),
        "visual_margin_mean": _float(visual_margin.mean()),
        "visual_margin_std": _float(visual_margin.std(unbiased=False)),
        "visual_margin_neg_ratio": _float((visual_margin < 0).float().mean()),
        "visual_entropy_mean": _float(visual_entropy.mean()),
        "same_class_query_to_centroid_sim": _float(visual_same.mean()),
        "nearest_wrong_centroid_sim": _float(visual_nearest_wrong.mean()),
    }
    return result


def text_visual_conflict_rows(T, V, classnames, base_row):
    logits = V @ T.T
    labels = torch.arange(V.shape[0], device=DEVICE)
    tv_margin, tv_alignment, tv_nearest_wrong = masked_margin(logits, labels)
    nearest_text = logits.clone()
    nearest_text.fill_diagonal_(-float("inf"))
    nearest_text_idx = nearest_text.argmax(dim=1)

    rows = []
    for class_idx in range(V.shape[0]):
        nearest_idx = int(nearest_text_idx[class_idx].item())
        rows.append(
            {
                **base_row,
                "class_id": int(class_idx),
                "class_name": classnames[class_idx],
                "tv_alignment": _float(tv_alignment[class_idx]),
                "tv_margin": _float(tv_margin[class_idx]),
                "tv_nearest_wrong_text_sim": _float(tv_nearest_wrong[class_idx]),
                "tv_nearest_wrong_text_class_id": nearest_idx,
                "tv_nearest_wrong_text_class_name": classnames[nearest_idx],
                "tv_is_negative": bool(tv_margin[class_idx].item() < 0),
            }
        )

    summary = {
        "tv_alignment_mean": _float(tv_alignment.mean()),
        "tv_alignment_std": _float(tv_alignment.std(unbiased=False)),
        "tv_margin_mean": _float(tv_margin.mean()),
        "tv_margin_std": _float(tv_margin.std(unbiased=False)),
        "tv_margin_min": _float(tv_margin.min()),
        "tv_neg_ratio": _float((tv_margin < 0).float().mean()),
    }
    return rows, summary


def alpha_sweep(T, V, eval_features, eval_labels, endpoint, alphas, plateau_eps):
    eval_norm = F.normalize(eval_features, dim=-1)
    text_correct = endpoint["text_correct"].to(DEVICE)
    text_correct_count = int(text_correct.sum().item())
    text_wrong_count = int((~text_correct).sum().item())

    acc_curve = []
    rescue_curve = []
    damage_curve = []
    net_curve = []
    damage_rate_curve = []
    rescue_rate_curve = []

    for alpha in alphas:
        prototypes = F.normalize((1.0 - alpha) * T + alpha * V, dim=-1)
        preds = (eval_norm @ prototypes.T).argmax(dim=-1)
        correct = preds.eq(eval_labels)
        rescue = ((~text_correct) & correct).sum().item()
        damage = (text_correct & ~correct).sum().item()
        acc_curve.append(_float(correct.float().mean() * 100.0))
        rescue_curve.append(int(rescue))
        damage_curve.append(int(damage))
        net_curve.append(int(rescue - damage))
        damage_rate_curve.append(_ratio(damage, text_correct_count))
        rescue_rate_curve.append(_ratio(rescue, text_wrong_count))

    acc_tensor = torch.tensor(acc_curve, dtype=torch.float32, device=DEVICE)
    oracle_idx = int(acc_tensor.argmax().item())
    max_acc = float(acc_curve[oracle_idx])
    plateau_indices = [idx for idx, acc in enumerate(acc_curve) if acc >= max_acc - plateau_eps]
    if plateau_indices:
        plateau_width = float(alphas[plateau_indices[-1]].item() - alphas[plateau_indices[0]].item())
    else:
        plateau_width = 0.0

    return {
        "alpha_grid": [_float(alpha) for alpha in alphas],
        "acc": acc_curve,
        "rescue": rescue_curve,
        "damage": damage_curve,
        "net": net_curve,
        "damage_rate": damage_rate_curve,
        "rescue_rate": rescue_rate_curve,
        "oracle_idx": oracle_idx,
        "oracle_alpha": _float(alphas[oracle_idx]),
        "oracle_acc": max_acc,
        "plateau_width": plateau_width,
        "plateau_grid_count": len(plateau_indices),
        "text_correct_count": text_correct_count,
        "text_wrong_count": text_wrong_count,
    }


def curve_rows(base_row, sweep, selected_alpha):
    rows = []
    selected_idx = nearest_alpha_idx(sweep["alpha_grid"], selected_alpha)
    oracle_idx = sweep["oracle_idx"]
    for idx, alpha in enumerate(sweep["alpha_grid"]):
        rows.append(
            {
                **base_row,
                "alpha": alpha,
                "alpha_idx": idx,
                "acc": sweep["acc"][idx],
                "rescue": sweep["rescue"][idx],
                "damage": sweep["damage"][idx],
                "net": sweep["net"][idx],
                "damage_rate": sweep["damage_rate"][idx],
                "rescue_rate": sweep["rescue_rate"][idx],
                "is_selected_alpha": idx == selected_idx,
                "is_oracle_alpha": idx == oracle_idx,
                "selected_alpha": selected_alpha,
                "oracle_alpha": sweep["oracle_alpha"],
                "plateau_width": sweep["plateau_width"],
                "plateau_grid_count": sweep["plateau_grid_count"],
            }
        )
    return rows


def nearest_alpha_idx(alpha_grid, alpha):
    alpha = float(alpha)
    return min(range(len(alpha_grid)), key=lambda idx: abs(float(alpha_grid[idx]) - alpha))


def rescue_damage_at(sweep, idx, prefix):
    return {
        f"{prefix}_rescue": int(sweep["rescue"][idx]),
        f"{prefix}_damage": int(sweep["damage"][idx]),
        f"{prefix}_net": int(sweep["net"][idx]),
        f"{prefix}_damage_rate": sweep["damage_rate"][idx],
        f"{prefix}_rescue_rate": sweep["rescue_rate"][idx],
    }


def support_representativeness_rows(
    dataset,
    train_indices,
    train_features,
    train_labels,
    val_features,
    val_labels,
    V,
    classnames,
    base_row,
):
    train_norm = F.normalize(train_features, dim=-1)
    val_norm = F.normalize(val_features, dim=-1)
    rows = []
    support_outlier_scores = []
    support_means = []

    for local_idx, global_idx in enumerate(train_indices):
        class_idx = int(train_labels[local_idx].item())
        query_mask = val_labels == class_idx
        query_features = val_norm[query_mask]
        if query_features.numel() == 0:
            same_mean = None
            same_std = None
            centroid_mean = None
            relative_outlier = None
            outlier_score = None
        else:
            sims = query_features @ train_norm[local_idx]
            centroid_sims = query_features @ V[class_idx]
            same_mean = _float(sims.mean())
            same_std = _float(sims.std(unbiased=False))
            centroid_mean = _float(centroid_sims.mean())
            outlier_score = None if same_mean is None else 1.0 - same_mean
            relative_outlier = None if centroid_mean is None or same_mean is None else centroid_mean - same_mean

        if outlier_score is not None:
            support_outlier_scores.append(outlier_score)
        if same_mean is not None:
            support_means.append(same_mean)

        rows.append(
            {
                **base_row,
                "support_local_idx": int(local_idx),
                "support_global_idx": int(global_idx),
                "support_path": dataset.samples[global_idx][0],
                "class_id": class_idx,
                "class_name": classnames[class_idx],
                "support_to_query_same_class_mean": same_mean,
                "support_to_query_same_class_std": same_std,
                "support_to_centroid_sim": _float(train_norm[local_idx] @ V[class_idx]),
                "query_to_centroid_same_class_mean": centroid_mean,
                "support_outlier_score": outlier_score,
                "support_relative_outlier_score": relative_outlier,
            }
        )

    summary = {}
    if support_outlier_scores:
        summary.update(_stats("support_outlier_score", support_outlier_scores))
    else:
        summary.update(_stats("support_outlier_score", torch.empty(0, device=DEVICE)))
    if support_means:
        summary.update(_stats("support_to_query_same_class", support_means))
    else:
        summary.update(_stats("support_to_query_same_class", torch.empty(0, device=DEVICE)))
    return rows, summary


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset_key"], row["kshot"])].append(row)

    aggregate_rows = []
    for (dataset_key, kshot), members in sorted(groups.items()):
        out = {
            "dataset_key": dataset_key,
            "dataset": members[0]["dataset"],
            "kshot": kshot,
            "seeds": [int(row["seed"]) for row in members],
        }
        metric_keys = sorted(
            key
            for key in members[0].keys()
            if isinstance(members[0].get(key), (int, float)) and key not in {"seed", "kshot", "num_classes"}
        )
        for key in metric_keys:
            values = [row.get(key) for row in members if isinstance(row.get(key), (int, float))]
            if not values:
                continue
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_std"] = float(np.std(values))
        aggregate_rows.append(out)
    return aggregate_rows


def evaluate_split(
    spec,
    dataset,
    classnames,
    all_features,
    all_labels,
    class_remap,
    text_features,
    kshot,
    seed,
    val_size,
    alphas,
    beta_values,
    margin_low_threshold,
    plateau_eps,
    eval_features=None,
    eval_labels=None,
):
    if val_size is None:
        if eval_features is None or eval_labels is None:
            raise ValueError("eval_features/eval_labels are required when data.val_size is not set.")
        train_indices = sample_kshot_by_class(dataset, kshot, seed)
        val_indices = list(range(len(eval_labels)))
        train_features, val_features, remapped_train, remapped_val = prepare_train_eval_tensors(
            all_features,
            all_labels,
            train_indices,
            eval_features,
            eval_labels,
            class_remap,
        )
    else:
        train_indices, val_indices = split_by_class(dataset, val_size, kshot, seed)
        train_features, val_features, remapped_train, remapped_val = prepare_split_tensors(
            all_features,
            all_labels,
            train_indices,
            val_indices,
            class_remap,
        )

    num_classes = text_features.shape[0]
    V = build_visual_centroids(train_features, remapped_train, text_features, num_classes)
    endpoint = endpoint_state(text_features, V, val_features, remapped_val, margin_low_threshold)
    selected_alpha = select_protofuse_alpha(
        text_features,
        V,
        train_features,
        remapped_train,
        num_classes,
        alphas,
        beta_values,
    )
    selected_preds, selected_acc = evaluate_alpha(text_features, V, selected_alpha, val_features, remapped_val)
    sweep = alpha_sweep(text_features, V, val_features, remapped_val, endpoint, alphas, plateau_eps)
    selected_idx = nearest_alpha_idx(sweep["alpha_grid"], selected_alpha)
    oracle_idx = sweep["oracle_idx"]

    base_row = {
        "dataset_key": spec["key"],
        "dataset": spec["display_name"],
        "template_name": spec["template_name"],
        "path": str(spec["path"]),
        "seed": int(seed),
        "kshot": int(kshot),
        "num_classes": int(num_classes),
        "train_size": int(len(train_indices)),
        "eval_size": int(len(val_indices)),
        "shots_per_class": shots_per_class(remapped_train, num_classes),
    }

    tv_rows, tv_summary = text_visual_conflict_rows(text_features, V, classnames, base_row)
    support_rows, support_summary = support_representativeness_rows(
        dataset,
        train_indices,
        train_features,
        remapped_train,
        val_features,
        remapped_val,
        V,
        classnames,
        base_row,
    )

    summary = {
        **base_row,
        **endpoint["summary"],
        **tv_summary,
        **support_summary,
        "selected_alpha": float(selected_alpha),
        "selected_acc": float(selected_acc),
        "selected_correct_count": int(selected_preds.eq(remapped_val).sum().item()),
        "oracle_alpha": sweep["oracle_alpha"],
        "oracle_acc": sweep["oracle_acc"],
        "plateau_width": sweep["plateau_width"],
        "plateau_grid_count": sweep["plateau_grid_count"],
        "text_correct_count": sweep["text_correct_count"],
        "text_wrong_count": sweep["text_wrong_count"],
        "gap_selected_to_text_only": float(selected_acc - endpoint["summary"]["text_only_acc"]),
        "gap_oracle_to_text_only": float(sweep["oracle_acc"] - endpoint["summary"]["text_only_acc"]),
        "gap_selected_to_oracle": float(selected_acc - sweep["oracle_acc"]),
    }
    summary.update(rescue_damage_at(sweep, selected_idx, "selected"))
    summary.update(rescue_damage_at(sweep, oracle_idx, "oracle"))

    alpha_rows = curve_rows(base_row, sweep, selected_alpha)

    return {
        "summary": summary,
        "alpha_rows": alpha_rows,
        "class_tv_rows": tv_rows,
        "support_rows": support_rows,
        "V": V,
    }


def centroid_stability_for_k(
    dataset,
    all_features,
    all_labels,
    class_remap,
    text_features,
    kshot,
    seed,
    val_size,
):
    if kshot <= 1:
        return None

    if val_size is None:
        train_indices_k = sample_kshot_by_class(dataset, kshot, seed)
        train_indices_prev = sample_kshot_by_class(dataset, kshot - 1, seed)
    else:
        train_indices_k, _ = split_by_class(dataset, val_size, kshot, seed)
        train_indices_prev, _ = split_by_class(dataset, val_size, kshot - 1, seed)
    train_features_k = all_features[train_indices_k].to(DEVICE, non_blocking=True)
    train_features_prev = all_features[train_indices_prev].to(DEVICE, non_blocking=True)
    labels_k = remap_labels(all_labels[train_indices_k], class_remap, device=DEVICE)
    labels_prev = remap_labels(all_labels[train_indices_prev], class_remap, device=DEVICE)
    V_k = build_visual_centroids(train_features_k, labels_k, text_features, text_features.shape[0])
    V_prev = build_visual_centroids(train_features_prev, labels_prev, text_features, text_features.shape[0])
    return _float((V_k * V_prev).sum(dim=-1).mean())


def evaluate_dataset(
    spec,
    clip_model,
    args,
    all_summary_rows,
    all_alpha_rows,
    all_class_tv_rows,
    all_support_rows,
    all_shot_scaling_rows,
):
    dataset_path = spec.get("path")
    if dataset_path is None or not Path(dataset_path).exists():
        raise FileNotFoundError(f"Missing dataset path for {spec['display_name']}: {dataset_path}")

    spec["path"] = Path(dataset_path)
    dataset_name = spec.get("template_name") or infer_dataset_name(dataset_path)
    spec["template_name"] = dataset_name

    logger.info(f"Dataset:  {spec['display_name']} ({dataset_name})")
    logger.info(f"Path:     {spec['path']}")

    if args.val_size is None:
        train_path = spec["path"] / "train"
        eval_path = spec["path"] / "test"
        if not train_path.exists() or not eval_path.exists():
            raise FileNotFoundError(
                f"data.val_size is not set, so expected train/test folders under {spec['path']}"
            )
        dataset = ImageFolder(str(train_path), transform=get_transform())
        eval_dataset = ImageFolder(str(eval_path), transform=get_transform())
        if dataset.classes != eval_dataset.classes:
            raise ValueError("Train/test class folders do not match.")
        classnames = [name.replace("_", " ") for name in dataset.classes]
        task_classes = sorted(
            set(label for _, label in dataset.samples) | set(label for _, label in eval_dataset.samples)
        )
        logger.info(
            f"Dataset loaded: train={len(dataset.samples)} images, test={len(eval_dataset.samples)} images, "
            f"{len(task_classes)} classes"
        )
    else:
        dataset = ImageFolder(str(spec["path"]), transform=get_transform())
        eval_dataset = None
        classnames = [name.replace("_", " ") for name in dataset.classes]
        task_classes = sorted(set(label for _, label in dataset.samples))
        logger.info(f"Dataset loaded: {len(dataset.samples)} images, {len(task_classes)} classes")

    all_features, all_labels = load_clip_features(
        clip_model,
        dataset,
        dataset_name,
        classnames,
        args,
        "train_or_full",
    )
    if eval_dataset is None:
        eval_features = None
        eval_labels = None
    else:
        eval_features, eval_labels = load_clip_features(
            clip_model,
            eval_dataset,
            dataset_name,
            classnames,
            args,
            "val_or_test",
        )

    text_features, class_remap, prompts = get_task_text_features(
        clip_model,
        classnames,
        task_classes,
        dataset_name,
    )
    alphas = torch.linspace(0.0, 1.0, args.alpha_steps, device=DEVICE)

    dataset_summary_rows = []
    for seed in args.seeds:
        previous_results_by_k = {}
        for kshot in args.kshots:
            with torch.inference_mode():
                result = evaluate_split(
                    spec,
                    dataset,
                    classnames,
                    all_features,
                    all_labels,
                    class_remap,
                    text_features,
                    kshot,
                    seed,
                    args.val_size,
                    alphas,
                    args.beta_values,
                    args.margin_low_threshold,
                    args.plateau_eps,
                    eval_features=eval_features,
                    eval_labels=eval_labels,
                )

            summary = result["summary"]
            stability = centroid_stability_for_k(
                dataset,
                all_features,
                all_labels,
                class_remap,
                text_features,
                kshot,
                seed,
                args.val_size,
            )
            summary["centroid_stability_K"] = stability
            summary["prompt_example"] = prompts[0] if prompts else None

            shot_row = {
                "dataset_key": spec["key"],
                "dataset": spec["display_name"],
                "seed": int(seed),
                "kshot": int(kshot),
                "centroid_stability_K": stability,
                "visual_only_acc_K": summary["visual_only_acc"],
                "oracle_alpha_K": summary["oracle_alpha"],
                "selected_alpha_K": summary["selected_alpha"],
                "selected_acc_K": summary["selected_acc"],
                "text_only_acc_K": summary["text_only_acc"],
                "gap_to_text_only_K": summary["gap_selected_to_text_only"],
                "oracle_gap_to_text_only_K": summary["gap_oracle_to_text_only"],
            }
            if previous_results_by_k:
                previous_k = max(previous_results_by_k)
                shot_row["previous_logged_kshot"] = int(previous_k)
                shot_row["selected_alpha_delta_from_previous_logged_K"] = (
                    summary["selected_alpha"] - previous_results_by_k[previous_k]["selected_alpha"]
                )
                shot_row["oracle_alpha_delta_from_previous_logged_K"] = (
                    summary["oracle_alpha"] - previous_results_by_k[previous_k]["oracle_alpha"]
                )
            previous_results_by_k[kshot] = summary

            dataset_summary_rows.append(summary)
            all_summary_rows.append(summary)
            all_alpha_rows.extend(result["alpha_rows"])
            all_class_tv_rows.extend(result["class_tv_rows"])
            all_support_rows.extend(result["support_rows"])
            all_shot_scaling_rows.append(shot_row)

            logger.info(
                f"ProtoFuse Diagnostics - seed={seed} - kshot={kshot} - "
                f"text={summary['text_only_acc']:.2f}% - visual={summary['visual_only_acc']:.2f}% - "
                f"selected={summary['selected_acc']:.2f}%@{summary['selected_alpha']:.2f} - "
                f"oracle={summary['oracle_acc']:.2f}%@{summary['oracle_alpha']:.2f} - "
                f"damage={summary['selected_damage']} - rescue={summary['selected_rescue']}"
            )

    summary_table_rows = []
    for kshot in args.kshots:
        members = [row for row in dataset_summary_rows if row["kshot"] == kshot]
        if not members:
            continue
        summary_table_rows.append(
            {
                "K": int(kshot),
                "Text": f"{np.mean([row['text_only_acc'] for row in members]):.2f}%",
                "Visual": f"{np.mean([row['visual_only_acc'] for row in members]):.2f}%",
                "Selected": f"{np.mean([row['selected_acc'] for row in members]):.2f}%",
                "Oracle": f"{np.mean([row['oracle_acc'] for row in members]):.2f}%",
                "sel a": f"{np.mean([row['selected_alpha'] for row in members]):.2f}",
                "oracle a": f"{np.mean([row['oracle_alpha'] for row in members]):.2f}",
                "damage": f"{np.mean([row['selected_damage'] for row in members]):.1f}",
                "rescue": f"{np.mean([row['selected_rescue'] for row in members]):.1f}",
            }
        )
    if summary_table_rows:
        logger.comparison_table(
            rows=summary_table_rows,
            columns=["K", "Text", "Visual", "Selected", "Oracle", "sel a", "oracle a", "damage", "rescue"],
            title=f"{spec['display_name']} Mean Summary",
        )


def save_outputs(args, summary_rows, alpha_rows, class_tv_rows, support_rows, shot_scaling_rows):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows = aggregate_summary(summary_rows)
    files = {
        "summary_jsonl": output_dir / "summary.jsonl",
        "alpha_curves_jsonl": output_dir / "alpha_curves.jsonl",
        "class_tv_conflict_jsonl": output_dir / "class_tv_conflict.jsonl",
        "support_jsonl": output_dir / "support_representativeness.jsonl",
        "shot_scaling_jsonl": output_dir / "shot_scaling.jsonl",
        "aggregate_json": output_dir / "aggregate_summary.json",
        "summary_csv": output_dir / "summary.csv",
        "shot_scaling_csv": output_dir / "shot_scaling.csv",
    }

    write_jsonl(files["summary_jsonl"], summary_rows)
    write_jsonl(files["alpha_curves_jsonl"], alpha_rows)
    write_jsonl(files["class_tv_conflict_jsonl"], class_tv_rows)
    write_jsonl(files["support_jsonl"], support_rows)
    write_jsonl(files["shot_scaling_jsonl"], shot_scaling_rows)
    write_csv(files["summary_csv"], summary_rows)
    write_csv(files["shot_scaling_csv"], shot_scaling_rows)

    with open(files["aggregate_json"], "w") as f:
        json.dump(aggregate_rows, f, indent=2, sort_keys=True)

    logger.info(f"Results written to {output_dir}")
    for path in files.values():
        logger.info(f"Saved: {path}")


ARG_SCHEMA = DEFAULT_ARG_SCHEMA


def parse_args():
    parser = create_argument_parser("Run ProtoFuse dataset diagnostics", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, merge_configs(load_config_file(parsed.config), overrides)


def runtime_args_from_config(config, spec):
    val_size_value = get_config_value(config, "data.val_size", None)
    val_size = None
    if val_size_value is not None:
        val_size = float(val_size_value)
        if val_size > 1.0:
            val_size = val_size / 100.0
        if val_size < 0 or val_size >= 1.0:
            raise ValueError("data.val_size must be in [0, 1) or 0-100 range when expressed as a percentage.")

    output_base = Path(get_config_value(config, "logging.output_dir", DEFAULT_OUTPUT_DIR))
    output_dir = output_base / spec["key"] / f"k{int(get_config_value(config, 'data.kshot', 16))}" / (
        f"seed{int(get_config_value(config, 'data.seed', 1))}"
    )

    return SimpleNamespace(
        device=str(get_config_value(config, "training.device", DEFAULT_DEVICE)),
        batch_size=int(get_config_value(config, "training.batch_size", BATCH_SIZE)),
        num_workers=int(get_config_value(config, "data.num_workers", NUM_WORKERS)),
        val_size=val_size,
        seed=int(get_config_value(config, "data.seed", 1)),
        kshot=int(get_config_value(config, "data.kshot", 16)),
        seeds=[int(get_config_value(config, "data.seed", 1))],
        kshots=[int(get_config_value(config, "data.kshot", 16))],
        alpha_steps=int(get_config_value(config, "model.alpha_steps", DEFAULT_ALPHA_STEPS)),
        beta_values=parse_float_list(
            get_config_value(config, "model.centroid_mix.beta_values", DEFAULT_BETA_VALUES)
        ),
        backbone=str(get_config_value(config, "model.backbone", "ViT-B/16")),
        precision=str(get_config_value(config, "training.precision", "fp32")),
        clip_mean=list(get_config_value(config, "data.clip_mean", CLIP_MEAN)),
        clip_std=list(get_config_value(config, "data.clip_std", CLIP_STD)),
        cache_dir=get_config_value(config, "feature_cache.cache_dir", DEFAULT_CACHE_DIR),
        feature_cache_enabled=bool(get_config_value(config, "feature_cache.enabled", True)),
        force_cache=bool(get_config_value(config, "diagnostics.force_cache", False)),
        margin_low_threshold=float(get_config_value(config, "diagnostics.margin_low_threshold", 0.0)),
        plateau_eps=float(get_config_value(config, "diagnostics.plateau_eps", 0.0)),
        output_dir=str(get_config_value(config, "diagnostics.output_dir", output_dir)),
    )


def run_one(parsed, config):
    global DEVICE, BATCH_SIZE, NUM_WORKERS, CACHE_DIR, CLIP_MEAN, CLIP_STD

    spec = dataset_spec_from_config(config)
    args = runtime_args_from_config(config, spec)

    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    CACHE_DIR = Path(args.cache_dir)
    CLIP_MEAN = args.clip_mean
    CLIP_STD = args.clip_std

    logger.section("Initialization", "config")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Run directory: {args.output_dir}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Alpha steps: {args.alpha_steps}")
    logger.info(f"CLIP feature cache: {CACHE_DIR}")
    log_experiment_start("ProtoFuse Dataset Diagnostics", spec["display_name"], args.kshot, args.seed)

    logger.section("ProtoFuse Dataset Diagnostics", "eval")

    clip_model = load_clip()

    summary_rows = []
    alpha_rows = []
    class_tv_rows = []
    support_rows = []
    shot_scaling_rows = []

    evaluate_dataset(
        spec,
        clip_model,
        args,
        summary_rows,
        alpha_rows,
        class_tv_rows,
        support_rows,
        shot_scaling_rows,
    )

    logger.section("Finalization", "save")
    save_outputs(args, summary_rows, alpha_rows, class_tv_rows, support_rows, shot_scaling_rows)


def run():
    parsed, config = parse_args()
    setup_logging(getattr(parsed, "debug", True), getattr(parsed, "disable_coloring", True))
    for dataset_config, _ in iter_dataset_configs(config):
        run_one(parsed, dataset_config)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run()
