from utils import (
    setup_logging,
    DEFAULT_ARG_SCHEMA,
    create_argument_parser,
    process_parsed_args,
    parse_override_arguments,
    merge_configs,
    load_config_file,
    run_for_dataset_configs,
)

from src.pipelines.maple import MaPLeTrainingPipeline

ARG_SCHEMA = DEFAULT_ARG_SCHEMA


def parse_args():
    parser = create_argument_parser("Train MaPLe model", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True), getattr(args, 'disable_coloring', True))
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    run_for_dataset_configs(merged, lambda dataset_config, _: MaPLeTrainingPipeline(dataset_config).run())


if __name__ == "__main__":
    main()
