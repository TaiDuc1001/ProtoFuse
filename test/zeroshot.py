import argparse
import contextlib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).parent.parent))

from clip import clip
from src.models.apt import CUSTOM_TEMPLATES
from utils import load_clip_to_cpu


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def parse_args():
    parser = argparse.ArgumentParser(description="Print one zero-shot CLIP accuracy number.")
    parser.add_argument("--data.root", "--root", dest="root", required=True, help="Dataset root.")
    parser.add_argument("--data.dataset_name", "--name", dest="dataset_name", required=True, help="Dataset name for prompt template.")
    parser.add_argument("--model.backbone", "--model", dest="backbone", required=True, help="CLIP backbone, e.g. ViT-B/16.")
    parser.add_argument("--device", "--training.device", dest="device", default=None, help="Device override.")
    parser.add_argument("--batch-size", "--training.batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--num-workers", "--data.num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--precision", "--training.precision", dest="precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--digits", type=int, default=2, help="Number of decimals to print.")
    parser.add_argument("--quiet", action="store_true", help="Suppress stderr run metadata.")
    return parser.parse_args()


def resolve_path(raw):
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def load_datasets(root, transform):
    train_root = root / "train"
    test_root = root / "test"
    if train_root.is_dir() and test_root.is_dir():
        train_dataset = ImageFolder(str(train_root), transform=transform)
        eval_dataset = ImageFolder(str(test_root), transform=transform)
        if train_dataset.classes != eval_dataset.classes:
            raise ValueError("Train/test class folders do not match.")
        classnames = [name.replace("_", " ") for name in train_dataset.classes]
        return classnames, eval_dataset

    eval_dataset = ImageFolder(str(root), transform=transform)
    classnames = [name.replace("_", " ") for name in eval_dataset.classes]
    return classnames, eval_dataset


def load_model(backbone, device, precision):
    with contextlib.redirect_stdout(sys.stderr):
        model = load_clip_to_cpu(backbone)
    model = model.to(device).eval()
    if precision == "fp32":
        model.float()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def get_template(dataset_name):
    aliases = {
        "FGVC-Aircraft": "FGVCAircraft",
        "FGVC Aircraft": "FGVCAircraft",
        "Food-101": "Food101",
    }
    return CUSTOM_TEMPLATES.get(
        dataset_name,
        CUSTOM_TEMPLATES.get(aliases.get(dataset_name, ""), "a photo of a {}."),
    )


def build_text_features(model, classnames, dataset_name, device):
    template = get_template(dataset_name)
    prompts = [template.format(name) for name in classnames]
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)
    return text_features


def zeroshot_accuracy(model, dataset, text_features, device, batch_size, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            image_features = model.encode_image(images).float()
            image_features = F.normalize(image_features, dim=-1)
            preds = (image_features @ text_features.T).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    if total == 0:
        raise ValueError("Evaluation dataset is empty.")
    return 100.0 * correct / total


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    transform = get_transform()
    classnames, dataset = load_datasets(root, transform)
    if not args.quiet:
        print(
            f"zeroshot dataset={args.dataset_name} backbone={args.backbone} "
            f"device={device} classes={len(classnames)} images={len(dataset)}",
            file=sys.stderr,
        )
    model = load_model(args.backbone, device, args.precision)
    text_features = build_text_features(model, classnames, args.dataset_name, device)
    accuracy = zeroshot_accuracy(model, dataset, text_features, device, args.batch_size, args.num_workers)
    print(f"{accuracy:.{args.digits}f}")


if __name__ == "__main__":
    main()
