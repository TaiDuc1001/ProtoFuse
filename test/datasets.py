import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}


def resolve_path(raw):
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def class_dirs(split_root):
    if not split_root.exists():
        return {}
    return {
        item.name: item
        for item in sorted(split_root.iterdir())
        if item.is_dir()
    }


def count_images(root):
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_counts(split_root):
    dirs = class_dirs(split_root)
    return {class_name: count_images(path) for class_name, path in dirs.items()}


def build_table(dataset_name, dataset_root, train_counts, test_counts):
    classes = sorted(set(train_counts) | set(test_counts))
    num_classes = len(classes)
    train_images = sum(train_counts.values())
    test_images = sum(test_counts.values())
    total_images = train_images + test_images
    avg_per_class = total_images / num_classes if num_classes else 0.0
    per_class_totals = [train_counts.get(class_name, 0) + test_counts.get(class_name, 0) for class_name in classes]
    min_per_class = min(per_class_totals) if per_class_totals else 0
    max_per_class = max(per_class_totals) if per_class_totals else 0

    table = Table(title=dataset_name)
    table.add_column("root")
    table.add_column("# classes", justify="right")
    table.add_column("# train images", justify="right")
    table.add_column("# test images", justify="right")
    table.add_column("avg. # image / class", justify="right")
    table.add_column("min / max", justify="right")
    table.add_row(
        str(dataset_root),
        str(num_classes),
        str(train_images),
        str(test_images),
        f"{avg_per_class:.2f}",
        f"{min_per_class} / {max_per_class}",
    )
    return table


def parse_args():
    parser = argparse.ArgumentParser(description="Print train/test dataset image counts.")
    parser.add_argument("--root", "--data.root", dest="root", required=True, help="Dataset root containing train/test folders.")
    parser.add_argument("--name", "--data.dataset_name", dest="name", required=True, help="Dataset display name.")
    parser.add_argument("--disable-coloring", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    console = Console(no_color=args.disable_coloring)
    dataset_root = resolve_path(args.root)
    train_counts = split_counts(dataset_root / "train")
    test_counts = split_counts(dataset_root / "test")
    console.print(build_table(args.name, dataset_root, train_counts, test_counts))


if __name__ == "__main__":
    main()
