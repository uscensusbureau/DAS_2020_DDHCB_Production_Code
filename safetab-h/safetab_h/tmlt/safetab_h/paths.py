"""Paths used by SafeTab-H."""

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

import os
import pkgutil
from pathlib import Path
from typing import Dict

from tmlt.common.configuration import Config
from tmlt.safetab_utils.paths import INPUT_FILES_SAFETAB_H

RESOURCES_PACKAGE_NAME = "resources"
"""The name of the directory containing resources.

This is used by pkgutil.
"""

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), RESOURCES_PACKAGE_NAME)
)
"""Path to directory containing resources for SafeTab-H."""

INPUT_CONFIG_DIR = os.path.join(RESOURCES_DIR, "config/input")
"""Directory containing initial configs for input files."""

ALT_INPUT_CONFIG_DIR_SAFETAB_H = "/tmp/safetab_h_input_configs"
ALT_OUTPUT_CONFIG_DIR_SAFETAB_H = "/tmp/safetab_h_output_configs"
"""The config directories to use for spark-compatible version of SafeTab-H.

Config files are copied to this directory from safetab_h resources. They cannot be used
directly because SafeTab-H resources may be zipped.
"""

OUTPUT_CONFIG_FILES = ["t3", "t4"]


def setup_input_config_dir():
    """Copy INPUT_CONFIG_DIR contents to temp directory ALT_INPUT_CONFIG_DIR_SAFETAB_H.

    NOTE: This setup is required to ensure zip-compatibility of Safetab.
    In particular, configs in resources directory of Safetab can not read
    when Safetab is distributed (and invoked) as zip archive (See Issue #331)
    """
    os.makedirs(ALT_INPUT_CONFIG_DIR_SAFETAB_H, exist_ok=True)
    for cfg_file in set(INPUT_FILES_SAFETAB_H):
        json_filename = os.path.splitext(cfg_file)[0] + ".json"
        json_file = Path(os.path.join(ALT_INPUT_CONFIG_DIR_SAFETAB_H, json_filename))
        json_file.touch(exist_ok=True)
        json_file.write_bytes(
            pkgutil.get_data(
                "tmlt.safetab_h", os.path.join("resources/config/input", json_filename)
            )
        )


def setup_safetab_h_output_config_dir():
    """Copy resources/config/output contents to temp ALT_OUTPUT_CONFIG_DIR_SAFETAB_H.

    NOTE: This setup is required to ensure zip-compatibility of Safetab.
    In particular, configs in resources directory of Safetab can not read
    when Safetab is distributed (and invoked) as zip archive (See Issue #331)
    """
    os.makedirs(ALT_OUTPUT_CONFIG_DIR_SAFETAB_H, exist_ok=True)
    for cfg_file in OUTPUT_CONFIG_FILES:
        json_filename = cfg_file + ".json"
        json_file = Path(os.path.join(ALT_OUTPUT_CONFIG_DIR_SAFETAB_H, json_filename))
        json_file.touch(exist_ok=True)
        json_file.write_bytes(
            pkgutil.get_data(
                "tmlt.safetab_h", os.path.join("resources/config/output", json_filename)
            )
        )


def get_safetab_h_output_configs() -> Dict[str, Config]:
    """Returns a map from output subdirs to configs for the files in those dirs."""
    configs = {}
    for cfg_file in OUTPUT_CONFIG_FILES:
        json_filename = f"{cfg_file}.json"
        configs[cfg_file] = Config.load_json(
            os.path.join(ALT_OUTPUT_CONFIG_DIR_SAFETAB_H, json_filename)
        )
    return configs
