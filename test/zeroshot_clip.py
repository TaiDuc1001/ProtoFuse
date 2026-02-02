import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Evaluating", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = 100.0 * image_features @ text_features.T
            predictions = logits.argmax(dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy


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
        
        acc = evaluate_zeroshot(model, dataset, config["template"])
        results[dataset_name] = acc
        print(f"  → Accuracy: {acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<15} {'Accuracy':>10}")
    print("-" * 30)
    
    for dataset_name, acc in results.items():
        print(f"{dataset_name:<15} {acc:>9.2f}%")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_experiment()
