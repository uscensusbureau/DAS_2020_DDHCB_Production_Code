#!/usr/bin/env python3
"""Command line interface for SafeTab-H."""

# Copyright 2024 Tumult Labs
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import argparse
import json
import os
import sys
import tempfile

from smart_open import open  # pylint: disable=redefined-builtin

from tmlt.common.io_helpers import get_logger_stream, write_log_file
from tmlt.safetab_h.paths import ALT_INPUT_CONFIG_DIR_SAFETAB_H, setup_input_config_dir
from tmlt.safetab_h.safetab_h_analytics import run_plan_h_analytics
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.regions import validate_state_filter_us
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    safetab_input_reader,
)


def main():
    """Parse command line arguments and run SafeTab-H."""
    parser = argparse.ArgumentParser(prog="safetab-h")
    subparsers = parser.add_subparsers(help="safetab-h sub-commands", dest="mode")

    def add_parameters_path(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "parameters_path",
            help="name of directory that contains iteration csv files and config.json",
            type=str,
        )

    def add_data_path(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "data_path",
            help=(
                "string used by the reader. The string is interpreted as an "
                "input csv files directory path "
                "for a csv reader or as a reader config file path for a cef reader."
            ),
            type=str,
        )

    def add_output(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            dest="output_directory",
            help="name of directory that contains all output files",
            type=str,
        )

    def add_log(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "-l",
            "--log",
            dest="log_filename",
            help="name of log file",
            type=str,
            default="safetab_h.log",
        )

    def add_validate_private_output(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "-vo",
            "--validate-private-output",
            dest="output_validation_flag",
            help="validate private outputs after running safetab-h private algorithm",
            action="store_true",
            default=False,
        )

    add_log(parser)

    parser_validate = subparsers.add_parser("validate", help="validate input files")
    for add_arg_func in [add_parameters_path, add_data_path, add_log]:
        add_arg_func(parser_validate)

    parser_execute = subparsers.add_parser("execute", help="execute mechanism")
    for add_arg_func in [
        add_parameters_path,
        add_data_path,
        add_output,
        add_log,
        add_validate_private_output,
    ]:
        add_arg_func(parser_execute)

    args = parser.parse_args()

    # Set up logging.
    logger, io_stream = get_logger_stream()

    if not args.mode:
        logger.error("No command was provided. Exiting...")
        sys.exit(1)

    if args.mode == "validate":
        logger.info("Validating SafeTab-H inputs and config...")
        setup_input_config_dir()
        with tempfile.TemporaryDirectory() as updated_config_dir:
            with open(os.path.join(args.parameters_path, "config.json"), "r") as f:
                config_json = json.load(f)
                reader = config_json[READER_FLAG]
                state_filter = []
                if config_json["run_us"] and validate_state_filter_us(
                    config_json[STATE_FILTER_FLAG]
                ):
                    state_filter += config_json[STATE_FILTER_FLAG]
                if config_json["run_pr"]:
                    state_filter += ["72"]

            okay = validate_input(
                parameters_path=args.parameters_path,
                input_data_configs_path=ALT_INPUT_CONFIG_DIR_SAFETAB_H,
                output_path=updated_config_dir,
                program="safetab-h",
                input_reader=safetab_input_reader(
                    reader=reader,
                    data_path=args.data_path,
                    state_filter=state_filter,
                    program="safetab-h",
                ),
                state_filter=state_filter,
            )
        if not okay:
            logger.error("SafeTab-H input validation failed. Exiting...")
            sys.exit(1)

    if args.mode == "execute":
        logger.info("Running SafeTab-H in 'execute' mode...")
        run_plan_h_analytics(
            parameters_path=args.parameters_path,
            data_path=args.data_path,
            output_path=args.output_directory,
            should_validate_private_output=args.output_validation_flag,
        )

    if args.log_filename:
        log_content = io_stream.getvalue()
        io_stream.close()
        write_log_file(args.log_filename, log_content)


if __name__ == "__main__":
    main()
