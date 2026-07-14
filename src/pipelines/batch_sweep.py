import copy
import logging

import torch
import torch.nn as nn

from utils import (
    get_config_value,
    iter_dataset_configs,
    logger,
    set_global_seed,
)


DEFAULT_KSHOTS = [1, 2, 4, 8, 16]
DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]


def _parse_int_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _has_config_path(config, path):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def sweep_values(config, overrides):
    kshots = _parse_int_list(
        get_config_value(config, "data.kshots", None),
        [get_config_value(config, "data.kshot")]
        if _has_config_path(overrides, "data.kshot")
        else DEFAULT_KSHOTS,
    )
    seeds = _parse_int_list(
        get_config_value(config, "data.seeds", None),
        [get_config_value(config, "data.seed")]
        if _has_config_path(overrides, "data.seed")
        else DEFAULT_SEEDS,
    )
    return kshots, seeds


def _snapshot_trainer_defaults(trainer):
    names = (
        "alpha",
        "beta",
        "init_alpha",
        "init_beta",
        "init_gamma",
    )
    return {
        name: getattr(trainer, name)
        for name in names
        if hasattr(trainer, name)
    }


def _reset_trainer_for_run(pipeline, defaults, kshot, seed):
    trainer = pipeline.trainer
    if trainer is None:
        raise RuntimeError("Trainer must be initialized before a batch run.")

    for name, value in defaults.items():
        setattr(trainer, name, value)

    for name in (
        "cache_keys",
        "cache_values",
        "cache_labels",
        "proto_weights",
        "adapter",
        "ape_adapter",
        "train_vecs",
        "train_labels",
        "last_ape_params",
        "last_timo_params",
        "benchmark_state",
    ):
        if hasattr(trainer, name):
            setattr(trainer, name, None)

    if hasattr(trainer, "clear_posthoc_protofuse"):
        trainer.clear_posthoc_protofuse()
    if hasattr(trainer, "model"):
        trainer.model = nn.Module()
    if hasattr(trainer, "shots"):
        trainer.shots = int(kshot)

    trainer.cfg.setdefault("data", {})["kshot"] = int(kshot)
    trainer.cfg.setdefault("data", {})["seed"] = int(seed)
    trainer.data_cfg = trainer.cfg.get("data", trainer.data_cfg)


def _prepare_shared_pipeline(pipeline_cls, config, kshots, seeds):
    run_config = copy.deepcopy(config)
    data_cfg = run_config.setdefault("data", {})
    data_cfg["kshot"] = int(max(kshots))
    data_cfg["seed"] = int(seeds[0])
    data_cfg["run_eda"] = False
    run_config.setdefault("checkpoint", {})["enabled"] = False
    run_config.setdefault("logging", {})["summary_only"] = True
    run_config.setdefault("posthoc_protofuse", {})["save_prototypes"] = False

    set_global_seed(int(seeds[0]))
    pipeline = pipeline_cls(run_config)
    pipeline._prepare_directories()
    pipeline._load_dataset()
    pipeline._split_dataset()
    pipeline._initialize_trainer()

    pipeline._full_dataset_clip_features()
    pipeline._cached_test_features()
    return pipeline, _snapshot_trainer_defaults(pipeline.trainer)


def _run_once(pipeline, defaults, kshot, seed):
    set_global_seed(int(seed))
    pipeline.kshot = int(kshot)
    pipeline.seed = int(seed)
    pipeline.config.data.kshot = int(kshot)
    pipeline.config.data.seed = int(seed)
    pipeline.data_cfg["kshot"] = int(kshot)
    pipeline.data_cfg["seed"] = int(seed)
    pipeline._split_dataset()
    _reset_trainer_for_run(pipeline, defaults, kshot, seed)

    pipeline.metrics = []
    pipeline.best_val_acc = -float("inf")
    pipeline.global_epoch = 0
    pipeline._train_epochs()

    if not pipeline.metrics:
        raise RuntimeError(
            f"{pipeline.METHOD_NAME} produced no metrics for "
            f"{kshot}-shot seed {seed}."
        )
    return dict(pipeline.metrics[-1])


def run_dataset_sweep(pipeline_cls, config, kshots, seeds):
    pipeline, defaults = _prepare_shared_pipeline(
        pipeline_cls,
        config,
        kshots,
        seeds,
    )
    results = {}
    for kshot in kshots:
        for seed in seeds:
            results[(int(kshot), int(seed))] = _run_once(
                pipeline,
                defaults,
                int(kshot),
                int(seed),
            )
    return results


def _mean_std(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())


def _format_table(rows, columns):
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    lines = [
        "  ".join(column.ljust(widths[column]) for column in columns),
        "  ".join("-" * widths[column] for column in columns),
    ]
    for row in rows:
        lines.append(
            "  ".join(str(row[column]).ljust(widths[column]) for column in columns)
        )
    return "\n".join(lines)


def print_summary(dataset_name, method_name, results, kshots, seeds):
    rows = []
    include_protofuse_columns = True
    for kshot in kshots:
        members = [results[(int(kshot), int(seed))] for seed in seeds]
        accuracies = [float(member.get("accuracy", 0.0)) for member in members]
        mean, std = _mean_std(accuracies)
        row = {
            "kshot": f"{int(kshot)}-shot",
            "runs": str(len(members)),
            "acc": f"{mean:.2f}% +/- {std:.2f}%",
        }
        has_protofuse_metrics = all(
            member.get("before_protofuse_accuracy") is not None
            and member.get("protofuse_gain") is not None
            for member in members
        )
        include_protofuse_columns = (
            include_protofuse_columns and has_protofuse_metrics
        )
        if has_protofuse_metrics:
            before_mean, before_std = _mean_std(
                [
                    float(member["before_protofuse_accuracy"])
                    for member in members
                ]
            )
            gain_mean, gain_std = _mean_std(
                [float(member["protofuse_gain"]) for member in members]
            )
            row["before ProtoFuse"] = (
                f"{before_mean:.2f}% +/- {before_std:.2f}%"
            )
            row["gain"] = f"{gain_mean:+.2f}% +/- {gain_std:.2f}%"
        rows.append(row)

    print(f"\n{dataset_name} x {method_name} x seed mean +/- std")
    columns = ["kshot", "runs", "acc"]
    # if include_protofuse_columns:
    #     columns.extend(["before ProtoFuse", "gain"])
    print(_format_table(rows, columns), flush=True)


def run_batch_sweep(config, overrides, pipeline_cls):
    previous_level = logger._logger.level
    logger._logger.setLevel(logging.WARNING)
    try:
        dataset_configs = list(iter_dataset_configs(config))
        outputs = []
        for dataset_config, _ in dataset_configs:
            kshots, seeds = sweep_values(dataset_config, overrides)
            results = run_dataset_sweep(
                pipeline_cls,
                dataset_config,
                kshots,
                seeds,
            )
            dataset_name = str(dataset_config["data"]["dataset_name"])
            first_result = next(iter(results.values()), {})
            method_name = str(first_result.get("method", pipeline_cls.METHOD_NAME))
            print_summary(
                dataset_name,
                method_name,
                results,
                kshots,
                seeds,
            )
            outputs.append(results)
        return outputs
    finally:
        logger._logger.setLevel(previous_level)
