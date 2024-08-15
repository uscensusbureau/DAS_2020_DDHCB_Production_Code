#!/usr/bin/env python
"""Run the single run error report on SafeTab-H."""

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

import json
import os
import tempfile

from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_h.accuracy_report import create_error_report_h
from tmlt.safetab_h.paths import INPUT_CONFIG_DIR, RESOURCES_DIR
from tmlt.safetab_h.safetab_h_analytics import run_plan_h_analytics
from tmlt.safetab_h.target_counts_h import create_ground_truth_h
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.regions import validate_state_filter_us
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    safetab_input_reader,
)

PySparkTest.setUpClass()
data_path = os.path.join(RESOURCES_DIR, "toy_dataset")
parameters_path = os.path.join(data_path, "input_dir_puredp")
noisy_dir = "example_output/single_run_error_report_h/actual"
target_dir = "example_output/single_run_error_report_h/expected"
output_dir = "example_output/single_run_error_report_h/output"
with open(os.path.join(parameters_path, "config.json"), "r") as f:
    config_json = json.load(f)

run_plan_h_analytics(parameters_path, data_path, noisy_dir)
with tempfile.TemporaryDirectory() as updated_config_dir:
    with open(os.path.join(parameters_path, "config.json"), "r") as f:
        config_json = json.load(f)
        reader = config_json[READER_FLAG]
        state_filter = []
        if config_json["run_us"] and validate_state_filter_us(
            config_json[STATE_FILTER_FLAG]
        ):
            state_filter += config_json[STATE_FILTER_FLAG]
        if config_json["run_pr"]:
            state_filter += ["72"]

    if validate_input(
        parameters_path=parameters_path,
        input_data_configs_path=INPUT_CONFIG_DIR,
        output_path=updated_config_dir,
        program="safetab-h",
        input_reader=safetab_input_reader(
            reader=reader,
            data_path=data_path,
            state_filter=state_filter,
            program="safetab-h",
        ),
        state_filter=state_filter,
    ):
        create_ground_truth_h(
            parameters_path=parameters_path,
            data_path=data_path,
            output_path=target_dir,
            config_path=updated_config_dir,
        )
create_error_report_h(
    noisy_path=noisy_dir,
    ground_truth_path=target_dir,
    parameters_path=parameters_path,
    output_path=output_dir,
)
PySparkTest.tearDownClass()
