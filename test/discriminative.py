import sys
import pandas as pd
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
import random
import warnings
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from clip import clip
from torchvision.datasets import ImageFolder

CLIP_MODEL_PATH = Path(__file__).parent.parent / "models" / "ViT-B-16.pt"
DEFAULT_DATASET = Path(__file__).parent.parent / "datasets" / "cub-200-2011-renamed"
DEFAULT_DEVICE = "cuda:0"

DEVICE = DEFAULT_DEVICE
BATCH_SIZE = 128
VAL_SIZE = 0.7
KSHOT = 4
SEED = 1

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CACHE_DIR = Path(__file__).parent.parent / "checkpoints" / "disc_cache"


def load_clip():
    model = torch.jit.load(str(CLIP_MODEL_PATH), map_location="cpu").eval()
    state_dict = model.state_dict()
    model = clip.build_model(state_dict)
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def extract_and_cache_features(clip_model, dataset, cache_name):
    cache_path = CACHE_DIR / f"{cache_name}.pt"
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  Loaded cached features from {cache_path}")
        return data["features"], data["labels"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  Extracting {cache_name}", leave=False):
            images = images.to(DEVICE)
            features = clip_model.encode_image(images).float()
            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({"features": features, "labels": labels}, cache_path)
    print(f"  Cached features to {cache_path}")
    return features, labels


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
        if kshot > 0:
            train_indices.extend(train_candidates[:kshot])
        else:
            train_indices.extend(train_candidates)

    return train_indices, val_indices


def compute_acc(labels_np, preds_np):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return {
            "acc": accuracy_score(labels_np, preds_np) * 100,
            "mca": balanced_accuracy_score(labels_np, preds_np) * 100,
        }


def zeroshot_eval(features, labels, clip_model, classnames, template="a photo of a {}, a type of bird."):
    features = F.normalize(features.to(DEVICE), dim=-1)

    with torch.no_grad():
        prompts = [template.format(name.replace("_", " ")) for name in classnames]
        tokens = clip.tokenize(prompts).to(DEVICE)
        text_features = clip_model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)

    logits = features @ text_features.T
    preds = logits.argmax(dim=-1).cpu().numpy()

    label_set = sorted(set(labels.tolist()))
    label_remap = {old: new for new, old in enumerate(label_set)}
    remapped_labels = np.array([label_remap[l.item()] for l in labels])

    return compute_acc(remapped_labels, preds)


def get_task_text_features(clip_model, classnames, task_classes, template="a photo of a {}, a type of bird."):
    sorted_classes = sorted(task_classes)
    prompts = [template.format(classnames[c].replace("_", " ")) for c in sorted_classes]
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(DEVICE)
        text_features = clip_model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)
    class_remap = {c: i for i, c in enumerate(sorted_classes)}
    return text_features, class_remap


def eval_prototypes(prototypes, features, labels, class_remap):
    features = F.normalize(features.to(DEVICE), dim=-1)
    remapped_labels = np.array([class_remap[l.item()] for l in labels])
    logits = features @ prototypes.T
    preds = logits.argmax(dim=-1).cpu().numpy()
    return compute_acc(remapped_labels, preds)



def _build_visual_centroids(train_features, remapped_train_labels, num_classes, embed_dim):
    visual_centroids = torch.zeros(num_classes, embed_dim, device=DEVICE)
    for i in range(num_classes):
        mask = (remapped_train_labels == i)
        if mask.any():
            visual_centroids[i] = F.normalize(train_features[mask].to(DEVICE).mean(0), dim=-1)
    return visual_centroids


def oracle_alpha(T, V, val_features, val_labels_remapped, alphas):
    val_norm = F.normalize(val_features.to(DEVICE), dim=-1)
    refined = F.normalize(
        (1 - alphas).view(-1, 1, 1) * T + alphas.view(-1, 1, 1) * V,
        dim=-1
    )
    logits = torch.einsum("qd,apd->aqp", val_norm, refined)
    preds = logits.argmax(dim=-1)
    scores = (preds == val_labels_remapped).float().mean(dim=-1)
    best_idx = scores.argmax()
    best_alpha = alphas[best_idx].item()
    best_proto = refined[best_idx]
    return best_proto, best_alpha


def loo_cv_alpha(T, V_all, train_features, remapped_train_labels, num_classes, alphas):
    class_indices = [[] for _ in range(num_classes)]
    for idx, lbl in enumerate(remapped_train_labels.tolist()):
        class_indices[lbl].append(idx)

    shots_per_class = min(len(idxs) for idxs in class_indices)

    if shots_per_class < 2:
        train_norm = F.normalize(train_features.to(DEVICE), dim=-1)
        refined = F.normalize(
            (1 - alphas).view(-1, 1, 1) * T + alphas.view(-1, 1, 1) * V_all,
            dim=-1
        )
        logits = torch.einsum("qd,apd->aqp", train_norm, refined)
        preds = logits.argmax(dim=-1)
        scores = (preds == remapped_train_labels.to(DEVICE)).float().mean(dim=-1)
        best_alpha = alphas[scores.argmax()].item()
        best_proto = F.normalize((1 - best_alpha) * T + best_alpha * V_all, dim=-1)
    else:
        k = shots_per_class
        class_feat = torch.stack([
            train_features[class_indices[c][:k]].to(DEVICE) for c in range(num_classes)
        ])
        class_sums = class_feat.sum(dim=1)
        loo_scores = torch.zeros(len(alphas), device=DEVICE)

        for fold in range(k):
            held = F.normalize(class_feat[:, fold, :], dim=-1)
            V_loo = F.normalize((class_sums - class_feat[:, fold, :]) / (k - 1), dim=-1)
            refined = F.normalize(
                (1 - alphas).view(-1, 1, 1) * T + alphas.view(-1, 1, 1) * V_loo,
                dim=-1
            )
            logits = torch.einsum("qd,apd->aqp", held, refined)
            preds = logits.argmax(dim=-1)
            loo_scores += (preds == torch.arange(num_classes, device=DEVICE)).float().mean(dim=-1)

        best_alpha = alphas[loo_scores.argmax()].item()
        best_proto = F.normalize((1 - best_alpha) * T + best_alpha * V_all, dim=-1)

    return best_proto, best_alpha


def run():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()
    seed = args.seed
    DEVICE = args.device
    kshots = [1, 2, 4, 8, 16]

    print("DISCRIMINATIVE CLIP — CLOSED-FORM PROTOTYPE")
    print(f"Device: {DEVICE} | val_size={VAL_SIZE} | seed={seed}")
    print(f"kshots: {kshots}")
    print()

    clip_model = load_clip()
    transform = get_transform()
    embed_dim = 512

    dataset = ImageFolder(args.dataset, transform=transform)
    num_classes = len(dataset.classes)
    classnames = list(dataset.classes)
    print(f"Dataset: {Path(args.dataset).name}")
    print(f"  Total: {len(dataset)} samples, {num_classes} classes")

    print("\nExtracting CLIP features...")
    cache_name = f"disc_{Path(args.dataset).name}"
    all_features, all_labels = extract_and_cache_features(clip_model, dataset, cache_name)

    task_classes = sorted(set(c for _, c in dataset.samples))
    text_features, class_remap = get_task_text_features(clip_model, classnames, task_classes)
    T = F.normalize(text_features, dim=-1)
    alphas = torch.linspace(0, 1, 101, device=DEVICE)

    results = []

    for kshot in kshots:
        print(f"  Running kshot={kshot}...", end=" ", flush=True)
        train_indices, val_indices = split_by_class(dataset, VAL_SIZE, kshot, seed)
        train_features = all_features[train_indices]
        train_labels = all_labels[train_indices]
        val_features = all_features[val_indices]
        val_labels = all_labels[val_indices]

        remapped_train = torch.tensor([class_remap[l.item()] for l in train_labels])
        remapped_val = torch.tensor([class_remap[l.item()] for l in val_labels], device=DEVICE)
        V = _build_visual_centroids(train_features, remapped_train, num_classes, embed_dim)

        vanilla = zeroshot_eval(val_features, val_labels, clip_model, classnames)

        proto_oracle, oracle_a = oracle_alpha(T, V, val_features, remapped_val, alphas)
        oracle_val = eval_prototypes(proto_oracle, val_features, val_labels, class_remap)

        proto_loo, loo_a = loo_cv_alpha(T, V, train_features, remapped_train, num_classes, alphas)
        loo_val = eval_prototypes(proto_loo, val_features, val_labels, class_remap)
        print("done")

        results.append({
            "kshot": kshot,
            "zs_acc": vanilla["acc"],
            "zs_mca": vanilla["mca"],
            "oracle_alpha": oracle_a,
            "oracle_acc": oracle_val["acc"],
            "oracle_mca": oracle_val["mca"],
            "loo_alpha": loo_a,
            "loo_acc": loo_val["acc"],
            "loo_mca": loo_val["mca"],
        })

    df = pd.DataFrame(results)
    df["gap"] = df["oracle_acc"] - df["loo_acc"]
    df = df.rename(columns={
        "kshot": "K", "zs_acc": "ZS Acc", "zs_mca": "ZS MCA",
        "oracle_alpha": "Oracle alpha", "oracle_acc": "Oracle Acc",
        "oracle_mca": "Oracle MCA", "loo_alpha": "LOO alpha",
        "loo_acc": "LOO Acc", "loo_mca": "LOO MCA", "gap": "Gap",
    })
    print(f"\n{Path(args.dataset).name}  |  seed={seed}")
    print(df.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    run()


