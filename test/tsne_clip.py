import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))
from clip import clip
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

CLIP_MODEL_PATH = Path(__file__).parent.parent / "models" / "ViT-B-16.pt"
CUB_DATASET_PATH = Path(__file__).parent.parent / "datasets" / "CUB_200_2011"
OUTPUT_DIR = Path(__file__).parent / "tsne_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

NUM_CLASSES = 200
BATCH_SIZE = 64
PERPLEXITY = 50
N_ITER = 2000
RANDOM_STATE = 42


def load_clip_model():
    model = torch.jit.load(str(CLIP_MODEL_PATH), map_location="cpu").eval()
    state_dict = model.state_dict()
    model = clip.build_model(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


def get_preprocessing():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def load_class_info():
    classes_file = CUB_DATASET_PATH / "classes.txt"
    classes = {}
    with open(classes_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                class_id = int(parts[0])
                class_name = parts[1]
                classes[class_id] = class_name
    return classes


def load_image_class_mapping():
    images_file = CUB_DATASET_PATH / "images.txt"
    labels_file = CUB_DATASET_PATH / "image_class_labels.txt"
    
    images = {}
    with open(images_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_id = int(parts[0])
                img_path = parts[1]
                images[img_id] = img_path
    
    labels = {}
    with open(labels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_id = int(parts[0])
                class_id = int(parts[1])
                labels[img_id] = class_id
    
    class_to_images = {}
    for img_id, class_id in labels.items():
        if class_id not in class_to_images:
            class_to_images[class_id] = []
        class_to_images[class_id].append(images[img_id])
    
    return class_to_images


def select_random_classes(classes, n_classes, seed=42):
    np.random.seed(seed)
    all_class_ids = list(classes.keys())
    selected_ids = np.random.choice(all_class_ids, n_classes, replace=False)
    return {cid: classes[cid] for cid in selected_ids}


class ImageDataset(Dataset):
    def __init__(self, paths, class_ids, transform):
        self.paths = paths
        self.class_ids = class_ids
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.paths[idx]).convert("RGB")
            image_tensor = self.transform(image)
            return image_tensor, self.class_ids[idx], True
        except:
            return torch.zeros(3, 224, 224), self.class_ids[idx], False


def extract_features(model, preprocess, class_to_images, selected_classes):
    images_dir = CUB_DATASET_PATH / "images"
    
    all_image_paths = []
    all_class_ids = []
    
    for class_id in selected_classes.keys():
        if class_id not in class_to_images:
            continue
        for img_rel_path in class_to_images[class_id]:
            img_path = images_dir / img_rel_path
            if img_path.exists():
                all_image_paths.append(img_path)
                all_class_ids.append(class_id)
    
    dataset = ImageDataset(all_image_paths, all_class_ids, preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Extracting features for {len(selected_classes)} classes ({len(dataset)} images)...")
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch_images, batch_class_ids, batch_valid in tqdm(dataloader, desc="Extracting features"):
            batch_images = batch_images.to(DEVICE)
            features = model.encode_image(batch_images)
            features = features / features.norm(dim=-1, keepdim=True)
            
            for feat, class_id, valid in zip(features, batch_class_ids, batch_valid):
                if valid:
                    all_features.append(feat.cpu().numpy())
                    all_labels.append(class_id.item())
    
    return np.array(all_features), np.array(all_labels)


def run_tsne(features, perplexity=PERPLEXITY, max_iter=N_ITER, random_state=RANDOM_STATE):
    print(f"Running t-SNE (perplexity={perplexity}, max_iter={max_iter})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto',
    )
    embeddings = tsne.fit_transform(features)
    return embeddings


def plot_tsne(embeddings, labels, selected_classes, output_path):
    plt.figure(figsize=(12, 10))
    
    n_classes = len(selected_classes)
    colors = plt.cm.gist_ncar(np.linspace(0, 0.9, n_classes))
    class_id_to_idx = {cid: i for i, cid in enumerate(sorted(selected_classes.keys()))}
    
    for class_id in selected_classes.keys():
        mask = labels == class_id
        idx = class_id_to_idx[class_id]
        plt.scatter(
            embeddings[mask, 0], 
            embeddings[mask, 1],
            c=[colors[idx]],
            alpha=0.6,
            s=25,
            edgecolors='white',
            linewidths=0.2,
        )
    
    plt.title(f't-SNE Visualization ({len(selected_classes)} Classes)\nCLIP ViT-B/16 | perplexity={PERPLEXITY}', fontsize=12)
    plt.xlabel('t-SNE Dimension 1', fontsize=10)
    plt.ylabel('t-SNE Dimension 2', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Plot saved to: {output_path}")
    plt.close()


def compute_cluster_metrics(embeddings, labels):
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    silhouette = silhouette_score(embeddings, labels)
    calinski = calinski_harabasz_score(embeddings, labels)
    davies = davies_bouldin_score(embeddings, labels)
    
    print("\n" + "="*50)
    print("CLUSTER QUALITY METRICS")
    print("="*50)
    print(f"  Silhouette Score:       {silhouette:.4f}")
    print(f"  Calinski-Harabasz:      {calinski:.2f}")
    print(f"  Davies-Bouldin Index:   {davies:.4f}")
    
    return silhouette, calinski, davies


def main():
    print("="*50)
    print("t-SNE: RANDOM CLASS SELECTION")
    print("="*50)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"\nDevice: {DEVICE}")
    print(f"Classes: {NUM_CLASSES} (random)")
    print(f"Perplexity: {PERPLEXITY}")
    
    print("\n[1/5] Loading CLIP model...")
    model = load_clip_model()
    preprocess = get_preprocessing()
    
    print("[2/5] Selecting random classes...")
    all_classes = load_class_info()
    selected_classes = select_random_classes(all_classes, NUM_CLASSES)
    print(f"       Selected {len(selected_classes)} random classes:")
    for cid, name in selected_classes.items():
        print(f"         - {cid}: {name}")
    
    print("\n[3/5] Loading image mappings...")
    class_to_images = load_image_class_mapping()
    
    print("[4/5] Extracting CLIP features...")
    features, labels = extract_features(model, preprocess, class_to_images, selected_classes)
    print(f"       Extracted {len(features)} feature vectors")
    
    print("[5/5] Running t-SNE...")
    embeddings = run_tsne(features)
    
    output_path = OUTPUT_DIR / "tsne_random_classes.png"
    plot_tsne(embeddings, labels, selected_classes, output_path)
    
    compute_cluster_metrics(embeddings, labels)
    
    with open(OUTPUT_DIR / "random_classes.txt", 'w') as f:
        f.write(f"# Random selection of {NUM_CLASSES} classes\n")
        f.write(f"# Seed: {RANDOM_STATE}\n\n")
        for cid, name in selected_classes.items():
            f.write(f"{cid} {name}\n")
    
    return embeddings, labels, selected_classes


if __name__ == "__main__":
    embeddings, labels, selected = main()
