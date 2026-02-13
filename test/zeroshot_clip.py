import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from clip import clip

CLIP_MODEL_PATH = Path(__file__).parent.parent / "models" / "ViT-B-16.pt"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"

CUB_PATH = Path("/state/partition1/tri.pm/APT/datasets/cub-200-2011-renamed")
FLOWER_PATH = Path("/state/partition1/tri.pm/APT/datasets/flowers102")
AIRCRAFT_PATH = Path("/state/partition1/tri.pm/APT/datasets/fgvc_aircraft")
CARS_PATH = Path("datasets/stanford_cars")

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

DATASET_CONFIGS = {
    # "cub": {
    #     "path": CUB_PATH,
    #     "template": "a photo of a {}, a type of bird.",
    #     "name": "CUB-200-2011",
    #     "type": "cub",
    # },
    # "flower": {
    #     "path": FLOWER_PATH,
    #     "template": "a photo of a {}, a type of flower.",
    #     "name": "Flowers102",
    #     "type": "imagefolder",
    # },
    "aircraft": {
        "path": AIRCRAFT_PATH,
        "template": "a photo of a {}, a type of aircraft.",
        "name": "FGVCAircraft",
    },
    "cars": {
        "path": CARS_PATH,
        "template": "a photo of a {}.",
        "name": "StanfordCars",
    },
}


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


class ImageFolderWithClassnames:
    def __init__(self, root, transform):
        self.dataset = ImageFolder(str(root), transform=transform)
        self.classnames = [name.replace("_", " ") for name in self.dataset.classes]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def zeroshot_classifier(model, classnames, template):
    with torch.no_grad():
        prompts = [template.format(name) for name in classnames]
        texts = clip.tokenize(prompts).to(DEVICE)
        text_features = model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def evaluate_zeroshot(model, dataset, template):
    dataloader = DataLoader(
        dataset.dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    text_features = zeroshot_classifier(model, dataset.classnames, template)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Evaluating", leave=False):
            images = images.to(DEVICE)
            
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = 100.0 * image_features @ text_features.T
            predictions = logits.argmax(dim=-1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    metrics = {
        "acc": accuracy_score(all_labels, all_preds) * 100,
        "mca": balanced_accuracy_score(all_labels, all_preds) * 100,
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted") * 100,
        "f1_macro": f1_score(all_labels, all_preds, average="macro") * 100,
        "f1_micro": f1_score(all_labels, all_preds, average="micro") * 100,
        "precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100,
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100,
        "precision_micro": precision_score(all_labels, all_preds, average="micro", zero_division=0) * 100,
        "recall_weighted": recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100,
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100,
        "recall_micro": recall_score(all_labels, all_preds, average="micro", zero_division=0) * 100,
    }
    return metrics


def run_experiment():
    print("=" * 60)
    print("ZERO-SHOT CLIP EVALUATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Datasets: {list(DATASET_CONFIGS.keys())}")
    print()
    
    print("Loading CLIP model...")
    model = load_clip_model()
    transform = get_preprocessing()
    
    results = {}
    
    for dataset_name in DATASET_CONFIGS.keys():
        config = DATASET_CONFIGS[dataset_name]
        print(f"\n[{config['name']}]")
        
        if not config["path"].exists():
            print(f"  ⚠ Not found at {config['path']}")
            continue
        
        try:
            dataset = ImageFolderWithClassnames(config["path"], transform)
            print(f"  {len(dataset)} samples, {len(dataset.classnames)} classes")
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
            continue
        
        metrics = evaluate_zeroshot(model, dataset, config["template"])
        results[dataset_name] = metrics
        
        print(f"  → Accuracy: {metrics['acc']:.2f}%")
        print(f"  → MCA: {metrics['mca']:.2f}%")
        print("  → F1:")
        print(f"      Weighted: {metrics['f1_weighted']:.2f}%")
        print(f"      Macro:    {metrics['f1_macro']:.2f}%")
        print(f"      Micro:    {metrics['f1_micro']:.2f}%")
        print("  → Precision:")
        print(f"      Weighted: {metrics['precision_weighted']:.2f}%")
        print(f"      Macro:    {metrics['precision_macro']:.2f}%")
        print(f"      Micro:    {metrics['precision_micro']:.2f}%")
        print("  → Recall:")
        print(f"      Weighted: {metrics['recall_weighted']:.2f}%")
        print(f"      Macro:    {metrics['recall_macro']:.2f}%")
        print(f"      Micro:    {metrics['recall_micro']:.2f}%")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<15} {'Acc':>8} {'MCA':>8} {'F1-Ma':>8} {'P-Ma':>8} {'R-Ma':>8}")
    print("-" * 60)
    
    for dataset_name, m in results.items():
        print(f"{dataset_name:<15} {m['acc']:>8.2f} {m['mca']:>8.2f} {m['f1_macro']:>8.2f} {m['precision_macro']:>8.2f} {m['recall_macro']:>8.2f}")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
