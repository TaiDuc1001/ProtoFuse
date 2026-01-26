import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from clip import clip

CLIP_MODEL_PATH = Path(__file__).parent.parent / "models" / "ViT-B-16.pt"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"

CUB_PATH = DATASETS_DIR / "CUB_200_2011"
FLOWER_PATH = DATASETS_DIR / "flowers102"
AIRCRAFT_PATH = DATASETS_DIR / "fgvc-aircraft-2013b"
CARS_PATH = DATASETS_DIR / "stanford_cars"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

DATASET_CONFIGS = {
    "cub": {
        "path": CUB_PATH,
        "template": "a photo of a {}, a type of bird.",
        "name": "CUB-200-2011",
        "type": "cub",
    },
    "flower": {
        "path": FLOWER_PATH,
        "template": "a photo of a {}, a type of flower.",
        "name": "Flowers102",
        "type": "imagefolder",
    },
    "aircraft": {
        "path": AIRCRAFT_PATH,
        "template": "a photo of a {}, a type of aircraft.",
        "name": "FGVCAircraft",
        "type": "aircraft",
    },
    "cars": {
        "path": CARS_PATH,
        "template": "a photo of a {}.",
        "name": "StanfordCars",
        "type": "cars",
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


class CUBDataset(Dataset):
    def __init__(self, root, transform, split="test"):
        self.root = Path(root)
        self.transform = transform
        self.images_dir = self.root / "images"
        
        classes = {}
        with open(self.root / "classes.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    classes[int(parts[0])] = parts[1].split(".")[-1].replace("_", " ")
        self.classnames = [classes[i] for i in sorted(classes.keys())]
        
        images = {}
        with open(self.root / "images.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    images[int(parts[0])] = parts[1]
        
        labels = {}
        with open(self.root / "image_class_labels.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[int(parts[0])] = int(parts[1]) - 1
        
        train_test = {}
        with open(self.root / "train_test_split.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    train_test[int(parts[0])] = int(parts[1])
        
        is_test = 0 if split == "test" else 1
        self.samples = []
        for img_id, img_path in images.items():
            if train_test.get(img_id, 0) == is_test:
                self.samples.append((self.images_dir / img_path, labels[img_id]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            return image, label
        except:
            return torch.zeros(3, 224, 224), label


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.dataset = ImageFolder(str(root), transform=transform)
        folder_names = sorted(os.listdir(root))
        self.classnames = [name.replace("_", " ") for name in folder_names if (Path(root) / name).is_dir()]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class FGVCAircraftDataset(Dataset):
    def __init__(self, root, transform, split="test"):
        self.root = Path(root) / "data"
        self.transform = transform
        self.images_dir = self.root / "images"
        
        variants_path = self.root / "variants.txt"
        with open(variants_path) as f:
            self.classnames = [line.strip() for line in f]
        self.class_to_idx = {name: i for i, name in enumerate(self.classnames)}
        
        split_file = self.root / f"images_variant_{split}.txt"
        self.samples = []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) >= 2:
                    img_name, variant = parts
                    img_path = self.images_dir / f"{img_name}.jpg"
                    if img_path.exists() and variant in self.class_to_idx:
                        self.samples.append((img_path, self.class_to_idx[variant]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            return image, label
        except:
            return torch.zeros(3, 224, 224), label


class StanfordCarsDataset(Dataset):
    def __init__(self, root, transform, split="test"):
        self.root = Path(root)
        self.transform = transform
        
        import scipy.io
        
        if split == "test":
            anno_path = self.root / "cars_test_annos_withlabels.mat"
            self.images_dir = self.root / "cars_test"
        else:
            anno_path = self.root / "devkit" / "cars_train_annos.mat"
            self.images_dir = self.root / "cars_train"
        
        meta_path = self.root / "devkit" / "cars_meta.mat"
        meta = scipy.io.loadmat(str(meta_path))
        self.classnames = [name[0] for name in meta["class_names"][0]]
        
        annos = scipy.io.loadmat(str(anno_path))
        annotations = annos["annotations"][0]
        
        self.samples = []
        for anno in annotations:
            img_name = anno[5][0] if len(anno) > 5 else anno[-1][0]
            label = anno[4][0][0] - 1
            img_path = self.images_dir / img_name
            if img_path.exists():
                self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            return image, label
        except:
            return torch.zeros(3, 224, 224), label


def get_dataset(dataset_name, transform):
    config = DATASET_CONFIGS[dataset_name]
    path = config["path"]
    ds_type = config["type"]
    
    if ds_type == "cub":
        return CUBDataset(path, transform, split="test")
    elif ds_type == "imagefolder":
        return ImageFolderDataset(path, transform)
    elif ds_type == "aircraft":
        return FGVCAircraftDataset(path, transform, split="test")
    elif ds_type == "cars":
        return StanfordCarsDataset(path, transform, split="test")
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")


def zeroshot_classifier(model, classnames, template):
    with torch.no_grad():
        prompts = [template.format(name) for name in classnames]
        texts = clip.tokenize(prompts).to(DEVICE)
        text_features = model.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def evaluate_zeroshot(model, dataset, template):
    dataloader = DataLoader(
        dataset, 
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
            dataset = get_dataset(dataset_name, transform)
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
