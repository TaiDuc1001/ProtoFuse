import os
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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

from src.pipelines.protofuse import ProtoFusePipeline

ARG_SCHEMA = DEFAULT_ARG_SCHEMA

def parse_args():
    p, u = create_argument_parser("Run", ARG_SCHEMA).parse_known_args()
    return p, process_parsed_args(p, ARG_SCHEMA, parse_override_arguments(u))

def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True), getattr(args, 'disable_coloring', True))
    config = merge_configs(load_config_file(args.config), overrides)
    run_for_dataset_configs(config, lambda dataset_config, _: ProtoFusePipeline(dataset_config).run())

if __name__ == "__main__": main()
