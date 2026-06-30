import copy
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.protofuse import ProtoFuse
from utils import get_config_value, iter_dataset_configs


TABLE_DATASETS = (
    "CUB-200-2011",
    "FGVC-Aircraft",
    "Stanford Cars",
    "Flowers102",
    "Food-101",
    "OxfordPets",
)

_DATASET_ALIASES = {
    "cub": "CUB-200-2011",
    "cub200": "CUB-200-2011",
    "cub2002011": "CUB-200-2011",
    "aircraft": "FGVC-Aircraft",
    "fgvcaircraft": "FGVC-Aircraft",
    "cars": "Stanford Cars",
    "stanfordcars": "Stanford Cars",
    "flowers": "Flowers102",
    "flowers102": "Flowers102",
    "food": "Food-101",
    "food101": "Food-101",
    "pet": "OxfordPets",
    "pets": "OxfordPets",
    "oxfordpet": "OxfordPets",
    "oxfordpets": "OxfordPets",
}


def _dataset_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _canonical_dataset_name(dataset_config, dataset_metadata):
    candidates = [get_config_value(dataset_config, "data.dataset_name", "")]
    if dataset_metadata is not None:
        candidates.extend(
            [
                dataset_metadata.get("dataset_name", ""),
                str(dataset_metadata.get("env_key", "")).removesuffix("_DATA_ROOT"),
            ]
        )
    for candidate in candidates:
        canonical = _DATASET_ALIASES.get(_dataset_key(candidate))
        if canonical is not None:
            return canonical
    return None


def table_dataset_configs(config):
    all_config = copy.deepcopy(config)
    all_config.setdefault("data", {})["all"] = True
    discovered = list(iter_dataset_configs(all_config))

    selected = []
    seen = set()
    for dataset_config, metadata in discovered:
        canonical = _canonical_dataset_name(dataset_config, metadata)
        table_name = canonical or str(get_config_value(dataset_config, "data.dataset_name", "Dataset"))
        if table_name in seen:
            continue
        seen.add(table_name)
        if canonical is not None:
            dataset_config["data"]["dataset_name"] = {
                "FGVC-Aircraft": "FGVCAircraft",
                "Stanford Cars": "StanfordCars",
            }.get(canonical, canonical)
        selected.append((table_name, dataset_config, metadata))
    return selected


def build_selector(
    text_features,
    support_features,
    support_labels,
    device,
    alpha_steps,
    beta_values,
    rho,
):
    selector = ProtoFuse.from_precomputed(
        text_features,
        device,
        alpha_steps=alpha_steps,
        beta_values=beta_values,
        rho=rho,
    )
    support_features = F.normalize(support_features.to(device).float(), dim=-1)
    support_labels = support_labels.to(device).long()
    visual = selector.build_visual_centroids(
        support_features,
        support_labels,
        selector.text_prototypes.shape[0],
    )
    return selector, selector.text_prototypes, visual, support_features, support_labels


def fused_prototypes(text, visual, alpha):
    return F.normalize((1.0 - float(alpha)) * text + float(alpha) * visual, dim=-1)


def batched_variant_accuracies(
    variants,
    eval_features,
    eval_labels,
    device,
    batch_size,
):
    names = [name for name, _ in variants]
    prototypes = torch.stack([prototype for _, prototype in variants], dim=0)
    correct = torch.zeros(len(variants), device=device, dtype=torch.long)
    total = int(eval_labels.numel())
    batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, total, batch_size):
            features = F.normalize(
                eval_features[start:start + batch_size].to(device).float(),
                dim=-1,
            )
            labels = eval_labels[start:start + batch_size].to(device).long()
            logits = torch.einsum("nd,vcd->vnc", features, prototypes)
            predictions = logits.argmax(dim=-1)
            correct += predictions.eq(labels.unsqueeze(0)).sum(dim=1)

    values = correct.float().mul_(100.0 / max(1, total)).cpu().tolist()
    return dict(zip(names, values))


def summarize_table(raw, variants, kshots, datasets=None):
    datasets = list(datasets or dict.fromkeys(row["dataset"] for row in raw))
    if not datasets:
        raise RuntimeError("Cannot summarize an ablation table without dataset results.")
    kshot_set = {int(kshot) for kshot in kshots}
    rows = []
    for variant in variants:
        by_dataset = {}
        variant_rows = [row for row in raw if row["variant"] == variant]
        seeds = sorted({int(row["seed"]) for row in variant_rows})
        for dataset in datasets:
            seed_macros = []
            for seed in seeds:
                values = [
                    row["accuracy"]
                    for row in variant_rows
                    if row["dataset"] == dataset
                    and row["kshot"] in kshot_set
                    and int(row["seed"]) == seed
                ]
                if len(values) != len(kshots):
                    raise RuntimeError(
                        f"Expected {len(kshots)} settings for variant={variant!r}, "
                        f"dataset={dataset!r}, seed={seed}; found {len(values)}."
                    )
                seed_macros.append(float(np.mean(values)))
            by_dataset[dataset] = {
                "mean": float(np.mean(seed_macros)),
                "std": float(np.std(seed_macros)),
                "runs": len(seed_macros),
                "settings_per_run": len(kshots),
            }

        overall_seed_macros = []
        for seed in seeds:
            values = [
                row["accuracy"]
                for row in variant_rows
                if row["dataset"] in datasets
                and row["kshot"] in kshot_set
                and int(row["seed"]) == seed
            ]
            expected = len(datasets) * len(kshots)
            if len(values) != expected:
                raise RuntimeError(
                    f"Expected {expected} macro-average settings for variant={variant!r}, "
                    f"seed={seed}; found {len(values)}."
                )
            overall_seed_macros.append(float(np.mean(values)))
        rows.append(
            {
                "variant": variant,
                "by_dataset": by_dataset,
                "average": {
                    "mean": float(np.mean(overall_seed_macros)),
                    "std": float(np.std(overall_seed_macros)),
                    "runs": len(overall_seed_macros),
                    "settings_per_run": len(datasets) * len(kshots),
                },
            }
        )
    return rows


def format_stat(stat):
    return f"{stat['mean']:.2f} ± {stat['std']:.2f}"


def build_table(title, summary, datasets=None):
    datasets = list(datasets or summary[0]["by_dataset"])
    rows = []
    for row in summary:
        rows.append(
            {
                "Variant": row["variant"],
                **{
                    dataset: format_stat(row["by_dataset"][dataset])
                    for dataset in datasets
                },
                "Avg.": format_stat(row["average"]),
            }
        )
    frame = pd.DataFrame(rows, columns=["Variant", *datasets, "Avg."])
    return f"{title}\n{frame.to_string(index=False)}"


def latex_rows(summary, datasets=None):
    datasets = list(datasets or summary[0]["by_dataset"])
    return "\n".join(
        "{} & {} & {:.2f} $\\pm$ {:.2f} \\\\".format(
            row["variant"],
            " & ".join(
                (
                    f"{row['by_dataset'][dataset]['mean']:.2f} "
                    f"$\\pm$ {row['by_dataset'][dataset]['std']:.2f}"
                )
                for dataset in datasets
            ),
            row["average"]["mean"],
            row["average"]["std"],
        )
        for row in summary
    )
