import argparse
import gc
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.ape import APE, _search_ape_hyperparameters
from src.models.apt import CUSTOM_TEMPLATES
from src.models.proto_adapter import ProtoAdapter
from src.models.protofuse import ProtoFuse
from src.models.timo import TIMO, _gda, _image_guide_text, _vec_sort
from src.models.tip_adapter import TipAdapter
from utils import (
    ConfigNode,
    deep_merge_dicts,
    discover_dataset_envs,
    load_clip_to_cpu,
    load_config_file,
    set_global_seed,
    fast_image_folder,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
KSHOTS = [1, 2, 4, 8, 16]
SEEDS = [1, 10, 100, 1000, 10000]
METHODS = ["zeroshot", "tip", "protoadapter", "ape", "timo", "protofuse"]
MB = 1024.0 * 1024.0
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DEFAULT_INFER_FEATURES = 10_000
DEFAULT_INFER_WARMUP = 5
DEFAULT_INFER_ITERS = 30

console = Console()


class DummyClip(nn.Module):
    def __init__(self, logit_scale):
        super().__init__()
        value = torch.as_tensor(logit_scale).detach().float().log()
        self.logit_scale = nn.Parameter(value.clone(), requires_grad=False)


def parse_int_list(raw):
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def parse_method_list(raw):
    methods = [part.strip().lower() for part in str(raw).split(",") if part.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid methods: {METHODS}")
    return methods


def load_method_config(method, args, kshot, seed, dataset_root, dataset_name):
    config_name = {
        "zeroshot": "protofuse",
        "tip": "tip_adapter",
        "protoadapter": "proto_adapter",
        "ape": "ape",
        "timo": "timo",
        "protofuse": "protofuse",
    }[method]
    config = load_config_file(CONFIG_DIR / f"{config_name}.yaml")

    overrides = {
        "model": {"backbone": args.backbone},
        "training": {
            "device": args.device,
            "precision": args.precision,
            "batch_size": args.batch_size,
        },
        "data": {
            "dataset_name": dataset_name,
            "root": str(dataset_root),
            "kshot": kshot,
            "seed": seed,
            "num_workers": args.num_workers,
            "run_eda": False,
        },
        "checkpoint": {"enabled": False},
        "feature_cache": {"enabled": False},
        "logging": {"output_dir": str(REPO_ROOT / "outputs" / "computational_cost")},
    }
    deep_merge_dicts(config, overrides)
    if args.no_finetune and method in {"tip", "protoadapter"}:
        config.setdefault("model", {}).setdefault("finetune", {})["enabled"] = False
    return ConfigNode(config)


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def load_train_dataset(root):
    root = Path(root)
    train_root = root / "train"
    dataset_root = train_root if train_root.exists() else root
    return fast_image_folder(str(dataset_root), transform=get_transform())


def load_test_dataset(root):
    root = Path(root)
    test_root = root / "test"
    dataset_root = test_root if test_root.exists() else root
    return fast_image_folder(str(dataset_root), transform=get_transform())


def extract_image_features(clip_model, dataset, device, batch_size, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    features = []
    labels = []
    clip_model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            batch_features = clip_model.encode_image(images).float()
            batch_features = F.normalize(batch_features, dim=-1)
            features.append(batch_features.cpu())
            labels.append(batch_labels.cpu().long())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def build_text_features(clip_model, classnames, dataset_name, device):
    template = CUSTOM_TEMPLATES.get(dataset_name, "a photo of a {}.")
    prompts = [template.format(name.replace("_", " ")) for name in classnames]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens).float()
    return F.normalize(text_features, dim=-1).detach()


def support_positions(labels, kshot, seed):
    by_class = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        by_class[int(label)].append(idx)

    rng = random.Random(seed)
    selected = []
    for class_idx in sorted(by_class):
        positions = list(by_class[class_idx])
        positions.sort()
        rng.shuffle(positions)
        if len(positions) < kshot:
            raise RuntimeError(f"Class {class_idx} has {len(positions)} samples, cannot sample {kshot}-shot.")
        selected.extend(positions[:kshot])
    return torch.tensor(selected, dtype=torch.long)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def reset_peak(device):
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def memory_allocated(device):
    if str(device).startswith("cuda"):
        return torch.cuda.memory_allocated(device)
    return 0


def max_memory_allocated(device):
    if str(device).startswith("cuda"):
        return torch.cuda.max_memory_allocated(device)
    return 0


def tensor_bytes(tensor):
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def module_parameter_bytes(module):
    if module is None:
        return 0
    return sum(tensor_bytes(param) for param in module.parameters())


def benchmark_setup(device, fn):
    sync(device)
    reset_peak(device)
    start_alloc = memory_allocated(device)
    start_time = time.perf_counter()
    state_bytes = fn()
    sync(device)
    setup_time = time.perf_counter() - start_time
    peak_delta = max(0, max_memory_allocated(device) - start_alloc)
    return setup_time, peak_delta / MB, state_bytes / MB


def prepare_infer_features(features, count, device):
    if features.shape[0] <= 0:
        raise RuntimeError("Cannot benchmark inference with an empty feature tensor.")
    repeats = math.ceil(count / features.shape[0])
    return features.repeat((repeats, 1))[:count].contiguous().to(device)


def benchmark_infer_logits(device, logits_fn, features, warmup, iters):
    sync(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = logits_fn(features)
        sync(device)

        if str(device).startswith("cuda"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _ = logits_fn(features)
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / max(1, iters)

        start_time = time.perf_counter()
        for _ in range(iters):
            _ = logits_fn(features)
        return 1000.0 * (time.perf_counter() - start_time) / max(1, iters)


def make_tip_trainer(cfg, classnames, text_features, logit_scale, device):
    trainer = TipAdapter.__new__(TipAdapter)
    trainer.cfg = cfg
    trainer.training_cfg = cfg.get("training", ConfigNode())
    trainer.model_cfg = cfg.get("model", ConfigNode())
    trainer.data_cfg = cfg.get("data", ConfigNode())
    trainer.classnames = classnames
    trainer.device = device
    trainer.alpha = trainer._cfg_float(1.0, "model.alpha")
    trainer.beta = trainer._cfg_float(5.5, "model.beta")
    trainer.text_features = text_features.to(device)
    trainer.clip_model = DummyClip(logit_scale).to(device).eval()
    trainer.embed_dim = text_features.shape[-1]
    trainer.num_classes = len(classnames)
    trainer.cache_keys = None
    trainer.cache_values = None
    trainer.cache_labels = None
    trainer.adapter = None
    trainer.model = nn.Module()
    return trainer


def make_proto_adapter_trainer(cfg, classnames, text_features, logit_scale, device):
    trainer = ProtoAdapter.__new__(ProtoAdapter)
    trainer.cfg = cfg
    trainer.training_cfg = cfg.get("training", ConfigNode())
    trainer.model_cfg = cfg.get("model", ConfigNode())
    trainer.data_cfg = cfg.get("data", ConfigNode())
    trainer.classnames = classnames
    trainer.device = device
    trainer.alpha = trainer._cfg_float(1.0, "model.alpha")
    trainer.text_features = text_features.to(device)
    trainer.clip_model = DummyClip(logit_scale).to(device).eval()
    trainer.embed_dim = text_features.shape[-1]
    trainer.num_classes = len(classnames)
    trainer.proto_weights = None
    trainer.adapter = None
    trainer.model = nn.Module()
    return trainer


def classifier_state_bytes(text_features):
    return tensor_bytes(text_features)


def compact_tip_state_bytes(trainer):
    base = classifier_state_bytes(trainer.text_features)
    if trainer.adapter is not None:
        return base + module_parameter_bytes(trainer.adapter) + tensor_bytes(trainer.cache_values)
    return base + tensor_bytes(trainer.cache_keys) + tensor_bytes(trainer.cache_values)


def compact_proto_adapter_state_bytes(trainer):
    base = classifier_state_bytes(trainer.text_features)
    if trainer.adapter is not None:
        return base + module_parameter_bytes(trainer.adapter)
    return base + tensor_bytes(trainer.proto_weights)


def run_zeroshot(text_features, logit_scale, device):
    text_features = text_features.to(device)

    def logits_fn(features):
        return features @ text_features.t()

    return (*benchmark_setup(device, lambda: classifier_state_bytes(text_features)), logits_fn)


def run_tip(cfg, classnames, text_features, logit_scale, train_features, train_labels, device):
    trainer = make_tip_trainer(cfg, classnames, text_features, logit_scale, device)

    def setup():
        trainer.build_cache(train_features, train_labels)
        trainer.tune_alpha_beta(train_features, train_labels)
        if bool(trainer.model_cfg.get("finetune", ConfigNode()).get("enabled", False)):
            trainer.finetune_adapter_from_features(train_features, train_labels)
        return compact_tip_state_bytes(trainer)

    return (*benchmark_setup(device, setup), trainer.logits_from_features)


def run_proto_adapter(cfg, classnames, text_features, logit_scale, train_features, train_labels, device):
    trainer = make_proto_adapter_trainer(cfg, classnames, text_features, logit_scale, device)

    def setup():
        trainer.build_prototypes(train_features, train_labels)
        trainer.tune_alpha(train_features, train_labels)
        if bool(trainer.model_cfg.get("finetune", ConfigNode()).get("enabled", False)):
            trainer.finetune_adapter_from_features(train_features, train_labels)
        return compact_proto_adapter_state_bytes(trainer)

    return (*benchmark_setup(device, setup), trainer.logits_from_features)


def balanced_cache_subset(train_features, train_labels, num_classes, kshot):
    counts = torch.bincount(train_labels.long(), minlength=num_classes)
    positive = counts[counts > 0]
    if positive.numel() == 0:
        raise RuntimeError("APE cache cannot be built without labeled samples.")
    effective_shots = min(int(positive.min().item()), int(kshot))
    selected = []
    for class_idx in range(num_classes):
        positions = torch.nonzero(train_labels == class_idx, as_tuple=False).flatten()
        if positions.numel() < effective_shots:
            raise RuntimeError(f"Class {class_idx} has too few samples for APE cache.")
        selected.append(positions[:effective_shots])
    selected = torch.cat(selected, dim=0)
    return train_features[selected], train_labels[selected], effective_shots


def ape_criterion_no_cache(cfg, clip_weights, cache_keys, only_use_txt, training_free):
    feat_dim, cate_num = clip_weights.shape
    text_feat = clip_weights.t().unsqueeze(1)
    if only_use_txt:
        feats = text_feat.squeeze(1)
        total = feats.sum(dim=0)
        sim = total * total - (feats * feats).sum(dim=0)
        sim = sim / (cate_num * (cate_num - 1))
    else:
        shots = cfg.get("shots", 1)
        cache_feat = cache_keys.reshape(cate_num, shots, feat_dim)
        feats = torch.cat([text_feat, cache_feat], dim=1)
        sample_num = feats.shape[1]
        total = feats.reshape(-1, feat_dim).sum(dim=0)
        class_total = feats.reshape(cate_num, sample_num, feat_dim).sum(dim=1)
        sim = total * total - (class_total * class_total).sum(dim=0)
        sim = sim / (cate_num * (cate_num - 1) * sample_num * sample_num)

    w = cfg.get("w", cfg.get("w_training_free", [0.5, 0.5]))
    criterion = (-1) * w[0] * sim + w[1] * torch.var(clip_weights, dim=1)
    feat_num_key = "training_free_feat_num" if training_free else "training_feat_num"
    k = cfg.get(feat_num_key, 800)
    ratio = 1024 / clip_weights.shape[0]
    k = max(1, min(int(k // ratio), clip_weights.shape[0]))
    return torch.topk(criterion, k=k).indices


def make_ape_trainer(cfg, classnames, text_features, device):
    trainer = APE.__new__(APE)
    trainer.cfg = cfg
    trainer.training_cfg = cfg.get("training", ConfigNode())
    trainer.model_cfg = cfg.get("model", ConfigNode())
    trainer.data_cfg = cfg.get("data", ConfigNode())
    trainer.classnames = classnames
    trainer.device = device
    trainer.shots = trainer._cfg_int(1, "data.kshot")
    trainer.init_alpha = trainer._cfg_float(1.0, "model.init_alpha")
    trainer.init_beta = trainer._cfg_float(1.0, "model.init_beta")
    trainer.init_gamma = trainer._cfg_float(0.1, "model.init_gamma")
    search_cfg = cfg.get("model", ConfigNode()).get("search", ConfigNode())
    trainer.search_scale = list(search_cfg.get("scale", [7, 7, 1]))
    trainer.search_step = list(search_cfg.get("step", [200, 20, 20]))
    trainer.w_training_free = list(cfg.get("model", ConfigNode()).get("w_training_free", [0.5, 0.5]))
    trainer.w_training = list(cfg.get("model", ConfigNode()).get("w_training", [0.2, 0.8]))
    trainer.training_free_feat_num = trainer._cfg_int(800, "model.training_free_feat_num")
    trainer.training_feat_num = trainer._cfg_int(900, "model.training_feat_num")
    trainer.clip_weights = text_features.to(device).t().contiguous()
    trainer.cache_keys = None
    trainer.cache_values = None
    trainer.num_classes = len(classnames)
    return trainer


def run_ape(cfg, classnames, text_features, train_features, train_labels, device, kshot):
    trainer = make_ape_trainer(cfg, classnames, text_features, device)

    def setup():
        cache_features, cache_labels, effective_shots = balanced_cache_subset(
            train_features, train_labels, len(classnames), kshot
        )
        trainer.shots = effective_shots
        cache_keys = cache_features.t().contiguous().to(device)
        cache_values = F.one_hot(cache_labels.long(), num_classes=len(classnames)).half().to(device)
        trainer.build_cache(cache_keys, cache_values)

        clip_weights = trainer.clip_weights
        feat_dim, cate_num = clip_weights.shape
        reshaped_values = trainer.cache_values.reshape(cate_num, -1, cate_num).to(device)
        reshaped_keys = trainer.cache_keys.t().reshape(cate_num, trainer.shots, feat_dim).reshape(cate_num, -1, feat_dim).to(device)
        flat_keys = reshaped_keys.reshape(-1, feat_dim)
        flat_values = reshaped_values.reshape(-1, cate_num)

        cfg_dict = trainer._build_cfg_dict("training_free")
        indices = ape_criterion_no_cache(cfg_dict, clip_weights, flat_keys, only_use_txt=False, training_free=True)

        new_clip_weights = F.normalize(clip_weights[indices, :], dim=0)
        new_cache_keys = F.normalize(flat_keys[:, indices], dim=-1)
        new_val_features = F.normalize(train_features.to(device)[:, indices], dim=-1)

        key_logits = (new_cache_keys @ new_clip_weights).softmax(dim=1)
        cache_div = torch.sum(
            flat_values * torch.log2((flat_values + 1e-6) / (key_logits + 1e-6)),
            dim=1,
        )[:, None]

        val_labels_dev = train_labels.to(device)
        raw_val_features = train_features.to(device)
        r_ff_val = new_val_features @ new_cache_keys.t()
        r_fw_val = 100.0 * raw_val_features @ clip_weights

        best_acc, best_alpha, best_beta, best_gamma = (
            _search_ape_hyperparameters(
                r_ff_val,
                r_fw_val,
                flat_values,
                cache_div,
                val_labels_dev,
                trainer.search_scale,
                trainer.search_step,
                trainer.init_alpha,
                trainer.init_beta,
                trainer.init_gamma,
            )
        )

        inference_soft_values = flat_values * (cache_div * best_gamma).exp()
        trainer.benchmark_state = SimpleNamespace(
            indices=indices,
            cache_keys=new_cache_keys,
            soft_values=inference_soft_values,
            alpha=best_alpha,
            beta=best_beta,
            gamma=best_gamma,
        )
        return (
            classifier_state_bytes(trainer.clip_weights)
            + tensor_bytes(indices)
            + tensor_bytes(new_cache_keys)
            + tensor_bytes(inference_soft_values)
        )

    setup_metrics = benchmark_setup(device, setup)

    def logits_fn(features):
        state = trainer.benchmark_state
        image_features = features.to(device).float()
        selected_features = F.normalize(image_features[:, state.indices], dim=-1)
        cache_logits = ((-1) * (state.beta - state.beta * (selected_features @ state.cache_keys.t()))).exp() @ state.soft_values
        clip_logits = 100.0 * image_features @ trainer.clip_weights
        return clip_logits + cache_logits * state.alpha

    return (*setup_metrics, logits_fn)


def make_timo_trainer(cfg, classnames, text_features, device):
    trainer = TIMO.__new__(TIMO)
    trainer.cfg = cfg
    trainer.training_cfg = cfg.get("training", ConfigNode())
    trainer.model_cfg = cfg.get("model", ConfigNode())
    trainer.data_cfg = cfg.get("data", ConfigNode())
    trainer.classnames = classnames
    trainer.device = device
    trainer.shots = trainer._cfg_int(1, "data.kshot")
    trainer.augment_epoch = trainer._cfg_int(1, "model.augment_epoch")
    trainer.clip_weights = text_features.to(device).t().contiguous()
    trainer.text_features_all = text_features.to(device).unsqueeze(1)
    trainer.num_classes = len(classnames)
    trainer.train_vecs = None
    trainer.train_labels = None
    return trainer


def run_timo(cfg, classnames, text_features, train_features, train_labels, device):
    trainer = make_timo_trainer(cfg, classnames, text_features, device)

    def setup():
        augment_epoch = trainer._cfg_int(1, "model.augment_epoch")
        trainer.train_vecs = train_features.repeat((augment_epoch, 1))
        trainer.train_labels = train_labels.repeat(augment_epoch)

        train_vecs = trainer.train_vecs.float().to(device)
        train_labels_dev = trainer.train_labels.long().to(device)
        val_features = train_features.float().to(device)
        val_labels = train_labels.long().to(device)

        image_weights = torch.stack(
            [train_vecs[train_labels_dev == class_idx].mean(dim=0) for class_idx in range(trainer.num_classes)]
        )
        image_weights = F.normalize(image_weights, dim=-1)

        dataset_name = cfg.get("data", ConfigNode()).get("dataset_name", "ImageNet")
        clip_weights_igt, matching_score = _image_guide_text(
            dataset_name, trainer.text_features_all.float().to(device), image_weights
        )
        clip_weights_igt = clip_weights_igt.t().contiguous()
        sorted_text, sorted_weights = _vec_sort(trainer.text_features_all.float().to(device), matching_score)
        prompt_num = sorted_text.shape[1]

        sliced_text = sorted_text.repeat(1, 2, 1)[:, :prompt_num, :]
        sliced_weights = sorted_weights.repeat(1, 2)[:, :prompt_num]
        sliced_text = sliced_text * sliced_weights.unsqueeze(-1)
        text_vecs = sliced_text.reshape(trainer.num_classes * prompt_num, -1)
        text_labels = torch.arange(trainer.num_classes, device=device).unsqueeze(1).repeat(1, prompt_num).flatten()

        combined_vecs = torch.cat([text_vecs, train_vecs], dim=0)
        combined_labels = torch.cat([text_labels, train_labels_dev], dim=0)
        alpha, weights, bias, val_acc = _gda(
            combined_vecs,
            combined_labels,
            clip_weights_igt,
            val_features,
            val_labels,
            alpha_shift=True,
        )
        trainer.benchmark_state = SimpleNamespace(
            alpha=alpha,
            beta=prompt_num,
            clip_weights_igt=clip_weights_igt,
            weights=weights,
            bias=bias,
            val_acc=val_acc,
        )
        return tensor_bytes(clip_weights_igt) + tensor_bytes(weights) + tensor_bytes(bias)

    setup_metrics = benchmark_setup(device, setup)

    def logits_fn(features):
        state = trainer.benchmark_state
        image_features = features.to(device).float()
        return state.alpha * image_features @ state.clip_weights_igt.float() + (image_features @ state.weights + state.bias)

    return (*setup_metrics, logits_fn)


def run_protofuse(
    cfg,
    classnames,
    text_features,
    train_features,
    train_labels,
    query_features,
    query_labels,
    device,
):
    alpha_steps = int(cfg.get("model", ConfigNode()).get("alpha_steps", 101))
    rho = float(cfg.get("model", ConfigNode()).get("rho", 0.5))
    beta_values = cfg.get("model", ConfigNode()).get("centroid_mix", ConfigNode()).get(
        "beta_values", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    )

    trainer = ProtoFuse.from_precomputed(
        text_features,
        device,
        alpha_steps=alpha_steps,
        beta_values=beta_values,
        rho=rho,
        classnames=classnames,
    )

    def setup():
        trainer.fuse_and_evaluate(
            train_features,
            train_labels,
            query_features,
            query_labels,
            len(classnames),
        )
        return tensor_bytes(trainer.fused_prototypes)

    setup_metrics = benchmark_setup(device, setup)

    def logits_fn(features):
        return features.to(device).float() @ trainer.fused_prototypes.T

    return (*setup_metrics, logits_fn)


RUNNERS = {
    "zeroshot": run_zeroshot,
    "tip": run_tip,
    "protoadapter": run_proto_adapter,
    "ape": run_ape,
    "timo": run_timo,
    "protofuse": run_protofuse,
}


def mean(values):
    return sum(values) / max(1, len(values))


def std(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def format_mean_std(values, decimals=4):
    return f"{mean(values):.{decimals}f} +/- {std(values):.{decimals}f}"


def method_label(method):
    return {
        "zeroshot": "Zero-shot CLIP",
        "tip": "TIP",
        "protoadapter": "ProtoAdapter",
        "ape": "APE",
        "timo": "TIMO",
        "protofuse": "ProtoFuse",
    }[method]


def build_summary_table(results, device, dataset_count=1):
    table = Table(
        title=(
            f"Average Computational Cost "
            f"({dataset_count} datasets x {len(KSHOTS)} k-shot x {len(SEEDS)} seeds)"
        )
    )
    table.add_column("method", style="bold")
    table.add_column("Setup (s)", justify="right")
    table.add_column("Peak Mem (MB)", justify="right")
    table.add_column("State Mem (MB)", justify="right")
    table.add_column("Infer Time (us/10k)", justify="right")

    for method in METHODS:
        rows = [row for row in results if row["method"] == method]
        if not rows:
            continue
        table.add_row(
            method_label(method),
            format_mean_std([row["setup_time_s"] for row in rows], decimals=4),
            format_mean_std([row["peak_delta_mb"] for row in rows], decimals=2) if str(device).startswith("cuda") else "N/A",
            format_mean_std([row["state_delta_mb"] for row in rows], decimals=2) if str(device).startswith("cuda") else "N/A",
            format_mean_std([row["infer_time_us_10k"] for row in rows], decimals=2),
        )
    return table


def build_kshot_table(results, device):
    table = Table(title="Per-k-shot Averages")
    table.add_column("method", style="bold")
    table.add_column("kshot", justify="right")
    table.add_column("Setup (s)", justify="right")
    table.add_column("Peak Mem (MB)", justify="right")
    table.add_column("State Mem (MB)", justify="right")
    table.add_column("Infer Time (us/10k)", justify="right")
    for method in METHODS:
        for kshot in KSHOTS:
            rows = [row for row in results if row["method"] == method and row["kshot"] == kshot]
            if not rows:
                continue
            table.add_row(
                method_label(method),
                str(kshot),
                format_mean_std([row["setup_time_s"] for row in rows], decimals=4),
                format_mean_std([row["peak_delta_mb"] for row in rows], decimals=2) if str(device).startswith("cuda") else "N/A",
                format_mean_std([row["state_delta_mb"] for row in rows], decimals=2) if str(device).startswith("cuda") else "N/A",
                format_mean_std([row["infer_time_us_10k"] for row in rows], decimals=2),
            )
    return table


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark few-shot setup computational cost.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasets" / "DTD")
    parser.add_argument("--dataset-name", default="DTD")
    parser.add_argument(
        "--data.all",
        dest="all_datasets",
        nargs="?",
        const="true",
        default="false",
        help="Run every dataset discovered from *_DATA_ROOT environment variables.",
    )
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--kshots", default=",".join(str(v) for v in KSHOTS))
    parser.add_argument("--seeds", default=",".join(str(v) for v in SEEDS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--show-kshot-table", action="store_true")
    parser.add_argument("--no-finetune", action="store_true", help="Disable TIP/ProtoAdapter finetune blocks.")
    parser.add_argument("--infer-features", type=int, default=DEFAULT_INFER_FEATURES)
    parser.add_argument("--infer-warmup", type=int, default=DEFAULT_INFER_WARMUP)
    parser.add_argument("--infer-iters", type=int, default=DEFAULT_INFER_ITERS)
    return parser.parse_args()


def run_dataset(args, dataset_root, dataset_name, requested_device, clip_model):
    console.print(
        f"[bold]Running {dataset_name} from {dataset_root}[/bold]"
    )
    dataset = load_train_dataset(dataset_root)
    test_dataset = load_test_dataset(dataset_root)
    classnames = list(dataset.classes)
    set_global_seed(1)

    image_features, labels = extract_image_features(
        clip_model,
        dataset,
        requested_device,
        args.batch_size,
        args.num_workers,
    )
    test_features, test_labels = extract_image_features(
        clip_model,
        test_dataset,
        requested_device,
        args.batch_size,
        args.num_workers,
    )
    text_features = build_text_features(
        clip_model,
        classnames,
        dataset_name,
        requested_device,
    )
    logit_scale = clip_model.logit_scale.detach().to(requested_device).exp()
    infer_features = prepare_infer_features(test_features, args.infer_features, requested_device)

    console.print(
        f"Dataset={dataset_name}, classes={len(classnames)}, support pool={len(dataset)}, "
        f"test pool={len(test_dataset)}, backbone={args.backbone}, device={requested_device}"
    )

    results = []
    total = len(METHODS) * len(KSHOTS) * len(SEEDS)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("benchmarking", total=total)
        for kshot in KSHOTS:
            for seed in SEEDS:
                positions = support_positions(labels, kshot, seed)
                support_features = image_features[positions].contiguous()
                support_labels = labels[positions].contiguous()
                for method in METHODS:
                    progress.update(task, description=f"{method_label(method)} | {kshot}-shot | seed {seed}")
                    set_global_seed(seed)
                    cfg = load_method_config(
                        method,
                        args,
                        kshot,
                        seed,
                        dataset_root,
                        dataset_name,
                    )
                    runner = RUNNERS[method]
                    if method == "zeroshot":
                        metrics = runner(text_features, logit_scale, requested_device)
                    elif method == "ape":
                        metrics = runner(
                            cfg,
                            classnames,
                            text_features,
                            support_features,
                            support_labels,
                            requested_device,
                            kshot,
                        )
                    elif method in {"tip", "protoadapter"}:
                        metrics = runner(
                            cfg,
                            classnames,
                            text_features,
                            logit_scale,
                            support_features,
                            support_labels,
                            requested_device,
                        )
                    elif method == "protofuse":
                        metrics = runner(
                            cfg,
                            classnames,
                            text_features,
                            support_features,
                            support_labels,
                            test_features,
                            test_labels,
                            requested_device,
                        )
                    else:
                        metrics = runner(
                            cfg,
                            classnames,
                            text_features,
                            support_features,
                            support_labels,
                            requested_device,
                        )
                    setup_time_s, peak_delta_mb, state_delta_mb, logits_fn = metrics
                    infer_time_us = 1000.0 * benchmark_infer_logits(
                        requested_device,
                        logits_fn,
                        infer_features,
                        args.infer_warmup,
                        args.infer_iters,
                    )
                    results.append(
                        {
                            "dataset": dataset_name,
                            "method": method,
                            "kshot": kshot,
                            "seed": seed,
                            "setup_time_s": setup_time_s,
                            "peak_delta_mb": peak_delta_mb,
                            "state_delta_mb": state_delta_mb,
                            "infer_time_us_10k": infer_time_us,
                        }
                    )
                    gc.collect()
                    if str(requested_device).startswith("cuda"):
                        torch.cuda.empty_cache()
                    progress.advance(task)
    return results


def main():
    global KSHOTS, SEEDS, METHODS
    args = parse_args()
    KSHOTS = parse_int_list(args.kshots)
    SEEDS = parse_int_list(args.seeds)
    METHODS = parse_method_list(args.methods)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not str(requested_device).startswith("cuda"):
        console.print("[yellow]CUDA is not available; GPU memory columns will be N/A.[/yellow]")

    all_datasets = str(args.all_datasets).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if all_datasets:
        discovered = discover_dataset_envs()
        if not discovered:
            raise RuntimeError(
                "No environment variables ending with _DATA_ROOT were found."
            )
        dataset_runs = [
            (Path(item["root"]), item["dataset_name"])
            for item in discovered
        ]
    else:
        dataset_runs = [(args.dataset_root, args.dataset_name)]

    console.print("[bold]Loading CLIP once; dataset features are extracted per dataset.[/bold]")
    clip_model = load_clip_to_cpu(args.backbone)
    if args.precision in {"fp32", "amp"}:
        clip_model.float()
    clip_model = clip_model.to(requested_device).eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    console.print("Disk feature/checkpoint caches are disabled; APE criterion cache is bypassed in-memory.")
    console.print(
        "Infer Time is measured after adaptation on cached test image features and reports the time to "
        f"compute logits for {args.infer_features:,} features. It excludes CLIP image encoding, data loading, "
        "and setup/adaptation time."
    )

    results = []
    completed_datasets = []
    for dataset_root, dataset_name in dataset_runs:
        results.extend(
            run_dataset(
                args,
                dataset_root,
                dataset_name,
                requested_device,
                clip_model,
            )
        )
        completed_datasets.append(dataset_name)
        gc.collect()
        if str(requested_device).startswith("cuda"):
            torch.cuda.empty_cache()

    console.print()
    console.print(
        "[bold]Datasets run:[/bold] "
        + ", ".join(completed_datasets)
    )
    console.print(
        build_summary_table(
            results,
            requested_device,
            dataset_count=len(completed_datasets),
        )
    )
    if args.show_kshot_table and not all_datasets:
        console.print()
        console.print(build_kshot_table(results, requested_device))


if __name__ == "__main__":
    main()
