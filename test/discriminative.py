import sys
import argparse
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
from rich.console import Console
from rich.table import Table
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))
from clip import clip
from torchvision.datasets import ImageFolder
from src.models.apt import CUSTOM_TEMPLATES
from utils import compute_metrics

CLIP_MODEL_PATH = Path(__file__).parent.parent / "models" / "ViT-B-16.pt"
DEFAULT_DATASET = Path(__file__).parent.parent / "datasets" / "cub-200-2011-renamed"
DEFAULT_DEVICE = "cuda:0"

DEVICE = DEFAULT_DEVICE
BATCH_SIZE = 128
VAL_SIZE = 0.7
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
        return data["features"], data["labels"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            features = clip_model.encode_image(images).float()
            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({"features": features, "labels": labels}, cache_path)
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
    if "dog" in path_name:
        return "OxfordPets"
    if "food" in path_name:
        return "Food101"
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
    return text_features, class_remap


def build_visual_centroids(train_features, remapped_train_labels, num_classes, embed_dim):
    visual_centroids = torch.zeros(num_classes, embed_dim, device=DEVICE)
    for i in range(num_classes):
        mask = (remapped_train_labels == i)
        if mask.any():
            visual_centroids[i] = F.normalize(train_features[mask].to(DEVICE).mean(0), dim=-1)
    return visual_centroids


def opt_base_eval(T, V_all, alpha, val_features, val_labels, class_remap):
    val_norm = F.normalize(val_features.to(DEVICE), dim=-1)
    refined = F.normalize((1 - alpha) * T + alpha * V_all, dim=-1)
    logits = val_norm @ refined.T
    preds = logits.argmax(dim=-1).cpu().numpy()
    remapped_labels = np.array([class_remap[l.item()] for l in val_labels])
    return compute_metrics(remapped_labels.tolist(), preds.tolist())


def opt_base_loo_cv_alpha(T, V_all, train_features, remapped_train_labels, num_classes, alphas):
    class_indices = [[] for _ in range(num_classes)]
    for idx, lbl in enumerate(remapped_train_labels.tolist()):
        class_indices[lbl].append(idx)
    shots_per_class = min(len(idxs) for idxs in class_indices)
    
    if shots_per_class < 2:
        train_norm = F.normalize(train_features.to(DEVICE), dim=-1)
        refined = F.normalize((1 - alphas).view(-1, 1, 1) * T + alphas.view(-1, 1, 1) * V_all, dim=-1)
        logits = torch.einsum("qd,apd->aqp", train_norm, refined)
        preds = logits.argmax(dim=-1)
        scores = (preds == remapped_train_labels.to(DEVICE)).float().mean(dim=-1)
        return alphas[scores.argmax()].item()
        
    k = shots_per_class
    class_feat = torch.stack([train_features[class_indices[c][:k]].to(DEVICE) for c in range(num_classes)])
    class_sums = class_feat.sum(dim=1)
    loo_scores = torch.zeros(len(alphas), device=DEVICE)
    
    for fold in range(k):
        held = F.normalize(class_feat[:, fold, :], dim=-1)
        V_loo = F.normalize((class_sums - class_feat[:, fold, :]) / (k - 1), dim=-1)
        refined = F.normalize((1 - alphas).view(-1, 1, 1) * T + alphas.view(-1, 1, 1) * V_loo, dim=-1)
        logits = torch.einsum("qd,apd->aqp", held, refined)
        preds = logits.argmax(dim=-1)
        loo_scores += (preds == torch.arange(num_classes, device=DEVICE)).float().mean(dim=-1)
        
    return alphas[loo_scores.argmax()].item()


def opt_entropy_eval(T, V_all, alpha_base, val_features, val_labels, class_remap, num_classes, tau=0.05):
    val_norm = F.normalize(val_features.to(DEVICE), dim=-1)
    L_T = val_norm @ T.T
    L_V = val_norm @ V_all.T
    
    p = F.softmax(L_V / tau, dim=-1)
    H_x = -torch.sum(p * torch.log(p + 1e-8), dim=-1)
    w_x = 1.0 - torch.clamp(H_x / math.log(num_classes), 0.0, 1.0)
    alpha_x = alpha_base * (0.5 + 0.5 * w_x)
    
    L_final = (1 - alpha_x).unsqueeze(-1) * L_T + alpha_x.unsqueeze(-1) * L_V
    preds = L_final.argmax(dim=-1).cpu().numpy()
    remapped_labels = np.array([class_remap[l.item()] for l in val_labels])
    return compute_metrics(remapped_labels.tolist(), preds.tolist())


def opt_variance_eval(T, V_all, var_global, alpha, val_features, val_labels, class_remap):
    W = 1.0 / (var_global + 1e-5)
    W = W / W.mean()
    T_w = F.normalize(T * W, dim=-1)
    V_w = F.normalize(V_all * W, dim=-1)
    val_norm = F.normalize(val_features.to(DEVICE) * W, dim=-1)
    
    refined = F.normalize((1 - alpha) * T_w + alpha * V_w, dim=-1)
    logits = val_norm @ refined.T
    preds = logits.argmax(dim=-1).cpu().numpy()
    
    remapped_labels = np.array([class_remap[l.item()] for l in val_labels])
    return compute_metrics(remapped_labels.tolist(), preds.tolist())


def opt_variance_loo_cv_alpha(T, V_all, var_global, train_features, remapped_train_labels, num_classes, alphas):
    class_indices = [[] for _ in range(num_classes)]
    for idx, lbl in enumerate(remapped_train_labels.tolist()):
        class_indices[lbl].append(idx)
    shots_per_class = min(len(idxs) for idxs in class_indices)
    
    W = 1.0 / (var_global + 1e-5)
    W = W / W.mean()
    T_w = F.normalize(T * W, dim=-1)
    
    if shots_per_class < 2:
        train_norm = F.normalize(train_features.to(DEVICE) * W, dim=-1)
        V_w = F.normalize(V_all * W, dim=-1)
        refined = F.normalize((1 - alphas).view(-1, 1, 1) * T_w + alphas.view(-1, 1, 1) * V_w, dim=-1)
        logits = torch.einsum("qd,apd->aqp", train_norm, refined)
        preds = logits.argmax(dim=-1)
        scores = (preds == remapped_train_labels.to(DEVICE)).float().mean(dim=-1)
        return alphas[scores.argmax()].item()
        
    k = shots_per_class
    train_feat_w = train_features.to(DEVICE) * W
    class_feat = torch.stack([train_feat_w[class_indices[c][:k]] for c in range(num_classes)])
    class_sums = class_feat.sum(dim=1)
    loo_scores = torch.zeros(len(alphas), device=DEVICE)
    
    for fold in range(k):
        held = F.normalize(class_feat[:, fold, :], dim=-1)
        V_loo_w = F.normalize((class_sums - class_feat[:, fold, :]) / (k - 1), dim=-1)
        refined = F.normalize((1 - alphas).view(-1, 1, 1) * T_w + alphas.view(-1, 1, 1) * V_loo_w, dim=-1)
        logits = torch.einsum("qd,apd->aqp", held, refined)
        preds = logits.argmax(dim=-1)
        loo_scores += (preds == torch.arange(num_classes, device=DEVICE)).float().mean(dim=-1)
        
    return alphas[loo_scores.argmax()].item()


def format_color(val):
    if val > 0:
        return f"[green]+{val:.2f}[/green]"
    elif val < 0:
        return f"[red]{val:.2f}[/red]"
    return f"{val:.2f}"


def run():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()
    DEVICE = args.device
    dataset_name = infer_dataset_name(args.dataset)
    kshots = [1, 2, 4, 8, 16]
    seeds = [1, 10, 100]

    clip_model = load_clip()
    transform = get_transform()
    embed_dim = 512

    dataset = ImageFolder(args.dataset, transform=transform)
    num_classes = len(dataset.classes)
    classnames = list(dataset.classes)

    cache_name = f"disc_{Path(args.dataset).name}"
    all_features, all_labels = extract_and_cache_features(clip_model, dataset, cache_name)

    task_classes = sorted(set(c for _, c in dataset.samples))
    text_features, class_remap = get_task_text_features(clip_model, classnames, task_classes, dataset_name)
    T = F.normalize(text_features, dim=-1)
    alphas = torch.linspace(0, 1, 11, device=DEVICE)

    data_b = []
    data_e = []
    data_v = []
    delta_e = []
    delta_v = []

    console = Console()
    console.print(f"[bold blue]DATASET:[/bold blue] {Path(args.dataset).name}")
    console.print(f"[bold blue]SEEDS:[/bold blue] {seeds}")
    console.print(f"[bold blue]DEVICE:[/bold blue] {DEVICE}")
    console.print()

    with console.status("[bold green]Evaluating new Uncertainty Options..."):
        for kshot in kshots:
            m_b = defaultdict(list)
            m_e = defaultdict(list)
            m_v = defaultdict(list)
            
            for seed in seeds:
                train_indices, val_indices = split_by_class(dataset, VAL_SIZE, kshot, seed)
                train_features = all_features[train_indices]
                train_labels = all_labels[train_indices]
                val_features = all_features[val_indices]
                val_labels = all_labels[val_indices]

                remapped_train = torch.tensor([class_remap[l.item()] for l in train_labels])
                V = build_visual_centroids(train_features, remapped_train, num_classes, embed_dim)

                alpha_b = opt_base_loo_cv_alpha(T, V, train_features, remapped_train, num_classes, alphas)
                if kshot < 2:
                    base_metrics = opt_entropy_eval(T, V, alpha_b, val_features, val_labels, class_remap, num_classes)
                else:
                    base_metrics = opt_base_eval(T, V, alpha_b, val_features, val_labels, class_remap)
                for k_met, v_met in base_metrics.items():
                    m_b[k_met].append(v_met)
                    
                for k_met, v_met in opt_entropy_eval(T, V, alpha_b, val_features, val_labels, class_remap, num_classes).items():
                    m_e[k_met].append(v_met)
                    
                var_global = train_features.to(DEVICE).var(dim=0, unbiased=False)
                alpha_v = opt_variance_loo_cv_alpha(T, V, var_global, train_features, remapped_train, num_classes, alphas)
                for k_met, v_met in opt_variance_eval(T, V, var_global, alpha_v, val_features, val_labels, class_remap).items():
                    m_v[k_met].append(v_met)

            means_b = {}
            means_e = {}
            means_v = {}
            
            for met_name in ["accuracy", "mca", "f1_macro", "precision_macro", "recall_macro"]:
                if met_name not in ["accuracy", "mca"]:
                    m_b[met_name] = [x * 100.0 for x in m_b[met_name]]
                    m_e[met_name] = [x * 100.0 for x in m_e[met_name]]
                    m_v[met_name] = [x * 100.0 for x in m_v[met_name]]
                    
                means_b[met_name] = (np.mean(m_b[met_name]), np.mean(m_b[met_name]) if len(m_b[met_name]) == 1 else np.std(m_b[met_name]))
                means_e[met_name] = (np.mean(m_e[met_name]), np.mean(m_e[met_name]) if len(m_e[met_name]) == 1 else np.std(m_e[met_name]))
                means_v[met_name] = (np.mean(m_v[met_name]), np.mean(m_v[met_name]) if len(m_v[met_name]) == 1 else np.std(m_v[met_name]))
                
            data_b.append((kshot, means_b))
            data_e.append((kshot, means_e))
            data_v.append((kshot, means_v))
            
            diff_e = {k: means_e[k][0] - means_b[k][0] for k in means_b}
            diff_v = {k: means_v[k][0] - means_b[k][0] for k in means_b}
            delta_e.append((kshot, diff_e))
            delta_v.append((kshot, diff_v))

    def create_table(title, items):
        table = Table(title=title)
        table.add_column("K", justify="right", style="cyan", no_wrap=True)
        table.add_column("Acc", justify="right")
        table.add_column("MCA", justify="right")
        table.add_column("F1-Ma", justify="right")
        table.add_column("P-Ma", justify="right")
        table.add_column("R-Ma", justify="right")
        
        for k_val, means in items:
            table.add_row(
                str(k_val),
                f"{means['accuracy'][0]:.2f} ± {means['accuracy'][1]:.2f}",
                f"{means['mca'][0]:.2f} ± {means['mca'][1]:.2f}",
                f"{means['f1_macro'][0]:.2f} ± {means['f1_macro'][1]:.2f}",
                f"{means['precision_macro'][0]:.2f} ± {means['precision_macro'][1]:.2f}",
                f"{means['recall_macro'][0]:.2f} ± {means['recall_macro'][1]:.2f}"
            )
        return table

    def create_delta_table(title, diff_e_list, diff_v_list):
        table = Table(title=title)
        table.add_column("K", justify="right", style="cyan", no_wrap=True)
        table.add_column("Entropy Acc", justify="right")
        table.add_column("Entropy MCA", justify="right")
        table.add_column("Variance Acc", justify="right")
        table.add_column("Variance MCA", justify="right")
        
        for (k_val, d_e), (_, d_v) in zip(diff_e_list, diff_v_list):
            table.add_row(
                str(k_val),
                format_color(d_e['accuracy']),
                format_color(d_e['mca']),
                format_color(d_v['accuracy']),
                format_color(d_v['mca'])
            )
        return table

    t_b = create_table("Base (LOO-CV)", data_b)
    console.print(t_b)
    console.print()

    table_opts = Table.grid(padding=(0, 4))
    t_e = create_table("Instance Entropy Anchor", data_e)
    t_v = create_table("Global Variance Re-weighting", data_v)
    table_opts.add_row(t_e, t_v)

    console.print(table_opts)
    console.print()
    console.print(create_delta_table("Improvement over Base (-/+) (%)", delta_e, delta_v))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run()
