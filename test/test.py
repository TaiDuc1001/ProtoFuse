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
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).parent.parent))

from clip import clip
from src.models.protofuse import ProtoFuse
from src.models.apt import CUSTOM_TEMPLATES
from utils import load_clip_to_cpu


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print cosine similarity between the first image and its true class text feature."
    )
    parser.add_argument("--config", default="configs/protofuse.yaml")
    parser.add_argument("--data.root", "--root", dest="root", default=None)
    parser.add_argument("--data.dataset_name", "--name", dest="dataset_name", default=None)
    parser.add_argument("--model.backbone", "--model", dest="backbone", default=None)
    parser.add_argument("--split", default="train", help="Dataset split to read first, if it exists.")
    parser.add_argument("--support-split", default="train", help="Dataset split used to build ProtoFuse prototypes.")
    parser.add_argument("--device", "--training.device", dest="device", default=None)
    parser.add_argument("--precision", "--training.precision", dest="precision", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--batch-size", "--training.batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--num-workers", "--data.num_workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--kshot", "--data.kshot", dest="kshot", type=int, default=None)
    parser.add_argument("--seed", "--data.seed", dest="seed", type=int, default=None)
    parser.add_argument("--digits", type=int, default=6)
    return parser.parse_args()


def resolve_path(raw):
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def load_config(path):
    config_path = resolve_path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}


def config_get(config, section, key, default=None):
    value = config.get(section, {})
    if not isinstance(value, dict):
        return default
    return value.get(key, default)


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def load_dataset(root, split, transform):
    split_root = root / split
    dataset_root = split_root if split_root.is_dir() else root
    return ImageFolder(str(dataset_root), transform=transform)


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


def load_model(backbone, device, precision):
    with contextlib.redirect_stdout(sys.stderr):
        model = load_clip_to_cpu(backbone)
    model = model.to(device).eval()
    if precision == "fp32":
        model.float()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def kshot_indices(dataset, kshot, seed):
    if kshot is None or kshot <= 0:
        return list(range(len(dataset)))

    generator = torch.Generator().manual_seed(seed)
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)

    selected = []
    for label in sorted(by_class):
        indices = torch.tensor(sorted(by_class[label]), dtype=torch.long)
        perm = torch.randperm(len(indices), generator=generator)
        selected.extend(indices[perm[:kshot]].tolist())
    return sorted(selected)


def encode_images(model, dataset, device, batch_size, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )

    features = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            batch_features = model.encode_image(images).float()
            features.append(F.normalize(batch_features, dim=-1).cpu())
            labels.append(batch_labels.long().cpu())

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def main():
    args = parse_args()
    config = load_config(args.config)

    root = args.root or config_get(config, "data", "root")
    dataset_name = args.dataset_name or config_get(config, "data", "dataset_name", "ImageNet")
    backbone = args.backbone or config_get(config, "model", "backbone", "ViT-B/16")
    precision = args.precision or config_get(config, "training", "precision", "fp32")
    device = args.device or config_get(config, "training", "device") or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or config_get(config, "training", "batch_size", 128)
    num_workers = args.num_workers if args.num_workers is not None else config_get(config, "data", "num_workers", 4)
    kshot = args.kshot if args.kshot is not None else config_get(config, "data", "kshot", -1)
    seed = args.seed if args.seed is not None else config_get(config, "data", "seed", 42)
    alpha_steps = config_get(config, "model", "alpha_steps", 101)
    centroid_mix = config_get(config, "model", "centroid_mix", {})
    beta_values = centroid_mix.get("beta_values") if isinstance(centroid_mix, dict) else None

    if root is None:
        raise ValueError("Missing dataset root. Pass --root or set data.root in the config.")

    root = resolve_path(root)
    transform = get_transform()
    dataset = load_dataset(root, args.split, transform)
    support_dataset = load_dataset(root, args.support_split, transform)
    if dataset.classes != support_dataset.classes:
        raise ValueError("Image split and support split class folders do not match.")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    image, label = dataset[0]
    classnames = [class_name.replace("_", " ") for class_name in dataset.classes]
    prompts = [get_template(dataset_name).format(class_name) for class_name in classnames]

    model = load_model(backbone, device, precision)

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        tokens = clip.tokenize(prompts).to(device)

        image_feature = F.normalize(model.encode_image(image).float(), dim=-1)
        text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
        zeroshot_cosine = (image_feature @ text_features[label].unsqueeze(0).T).item()

    support_indices = kshot_indices(support_dataset, kshot, seed)
    support_features, support_labels = encode_images(
        model,
        Subset(support_dataset, support_indices),
        device,
        batch_size,
        num_workers,
    )
    fused = ProtoFuse.posthoc_fuse(
        text_features.detach().cpu(),
        support_features,
        support_labels,
        device,
        alpha_steps=alpha_steps,
        beta_values=beta_values,
    )
    fused_prototypes = fused["fused_prototypes"].to(device)
    protofuse_cosine = (image_feature @ fused_prototypes[label].unsqueeze(0).T).item()

    print(f"zeroshot_cosine: {zeroshot_cosine:.{args.digits}f}")
    print(f"protofuse_cosine: {protofuse_cosine:.{args.digits}f}")
    print(f"protofuse_alpha: {float(fused['alpha']):.{args.digits}f}")


if __name__ == "__main__":
    main()
