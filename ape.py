from utils import (
    setup_logging,
    DEFAULT_ARG_SCHEMA,
    create_argument_parser,
    process_parsed_args,
    parse_override_arguments,
    merge_configs,
    load_config_file,
)

from src.pipelines.ape import APEPipeline
from src.pipelines.batch_sweep import run_batch_sweep

ARG_SCHEMA = DEFAULT_ARG_SCHEMA


def parse_args():
    parser = create_argument_parser("Run APE batch sweep", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True), getattr(args, 'disable_coloring', True))
    config = merge_configs(load_config_file(args.config), overrides)
    run_batch_sweep(config, overrides, APEPipeline)


if __name__ == "__main__":
    main()
