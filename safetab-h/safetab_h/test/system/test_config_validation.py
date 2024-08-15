"""Tests SafeTab-H config validation."""

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

import copy
import json
import os
import re
import unittest
from typing import Dict, List

import pytest
from parameterized import parameterized

from tmlt.safetab_h.paths import RESOURCES_DIR
from tmlt.safetab_utils.config_validation import validate_config_values


@pytest.mark.usefixtures("spark")
class TestConfigValidation(unittest.TestCase):
    """Tests invalid SafeTab-H config fails config validation."""

    def setUp(self):
        """Set up test."""
        self.input_path = os.path.join(RESOURCES_DIR, "toy_dataset/input_dir_zcdp")
        with open(os.path.join(self.input_path, "config.json"), "r") as f:
            self.temp_config_json = json.load(f)

    @parameterized.expand(
        [
            (
                {"privacy_defn": "pure"},
                r"Invalid config: Supported privacy definitions are Rho zCDP \(zcdp\)"
                r" and Pure DP \(puredp\).",
            ),
            (
                {"max_race_codes": "random"},
                "Invalid config: expected 'max_race_codes' to have int value between 1"
                " and 8.",
            ),
            (
                {"max_race_codes": 9},
                "expected 'max_race_codes' to have int value between 1 and 8.",
            ),
            (
                {"allow_negative_counts": "xyz"},
                "Supported value for 'allow_negative_counts' are true and false.",
            ),
            ({"run_us": "Yes"}, "Supported value for 'run_us' are true and false."),
            ({"run_pr": 10}, "Supported value for 'run_pr' are true and false."),
        ]
    )
    def test_invalid_config(self, overwrite_config: Dict, error_msg_regex: str):
        """validate_config_values errors for invalid common config keys.

        Args:
            overwrite_config: JSON data to be validated, as a Python dict
            error_msg_regex: A regular expression to check the error message against
        """
        invalid_config = copy.deepcopy(self.temp_config_json)
        for key, value in overwrite_config.items():
            invalid_config[key] = value

        with pytest.raises(ValueError, match=error_msg_regex):
            validate_config_values(invalid_config, "safetab-h", ["US", "PR"])

    @parameterized.expand(
        [
            (
                {},
                ["US", "PR"],
                "Invalid config: missing required keys {allow_negative_counts,"
                " max_race_codes, privacy_budget_h_t3_level_1_aiannh,"
                " privacy_budget_h_t3_level_1_county,"
                " privacy_budget_h_t3_level_1_place,"
                " privacy_budget_h_t3_level_1_pr_county,"
                " privacy_budget_h_t3_level_1_pr_place,"
                " privacy_budget_h_t3_level_1_pr_state,"
                " privacy_budget_h_t3_level_1_pr_tract,"
                " privacy_budget_h_t3_level_1_state, privacy_budget_h_t3_level_1_tract,"
                " privacy_budget_h_t3_level_1_usa, privacy_budget_h_t3_level_2_aiannh,"
                " privacy_budget_h_t3_level_2_county,"
                " privacy_budget_h_t3_level_2_place,"
                " privacy_budget_h_t3_level_2_pr_county,"
                " privacy_budget_h_t3_level_2_pr_place,"
                " privacy_budget_h_t3_level_2_pr_state,"
                " privacy_budget_h_t3_level_2_pr_tract,"
                " privacy_budget_h_t3_level_2_state, privacy_budget_h_t3_level_2_tract,"
                " privacy_budget_h_t3_level_2_usa, privacy_budget_h_t4_level_1_aiannh,"
                " privacy_budget_h_t4_level_1_county,"
                " privacy_budget_h_t4_level_1_place,"
                " privacy_budget_h_t4_level_1_pr_county,"
                " privacy_budget_h_t4_level_1_pr_place,"
                " privacy_budget_h_t4_level_1_pr_state,"
                " privacy_budget_h_t4_level_1_pr_tract,"
                " privacy_budget_h_t4_level_1_state, privacy_budget_h_t4_level_1_tract,"
                " privacy_budget_h_t4_level_1_usa, privacy_budget_h_t4_level_2_aiannh,"
                " privacy_budget_h_t4_level_2_county,"
                " privacy_budget_h_t4_level_2_place,"
                " privacy_budget_h_t4_level_2_pr_county,"
                " privacy_budget_h_t4_level_2_pr_place,"
                " privacy_budget_h_t4_level_2_pr_state,"
                " privacy_budget_h_t4_level_2_pr_tract,"
                " privacy_budget_h_t4_level_2_state, privacy_budget_h_t4_level_2_tract,"
                " privacy_budget_h_t4_level_2_usa, privacy_defn, reader, run_pr,"
                " run_us, state_filter_us, thresholds_h_t3, thresholds_h_t4}",
            ),
            (
                {"privacy_budget_h_t3_level_2_pr_place": "1"},
                ["US", "PR"],
                "expected 'privacy_budget_h_t3_level_2_pr_place' key to have float"
                " value, not str.",
            ),
            (
                {"privacy_budget_h_t4_level_2_county": -1},
                ["US", "PR"],
                "'privacy_budget_h_t4_level_2_county' must be non-negative",
            ),
        ]
    )
    def test_invalid_program_specific_config(
        self,
        overwrite_config: Dict,
        us_or_puerto_rico_values: List[str],
        error_msg_regex: str,
    ):
        """validate_config_values errors for invalid specific safetab-h config keys.

        Args:
            overwrite_config: JSON data to be validated, as a Python dict
            us_or_puerto_rico_values: Speficies run - US or PR or both
            error_msg_regex: A regular expression to check the error message against
        """
        error_msg_regex = re.escape(error_msg_regex)
        if overwrite_config:
            invalid_config = copy.deepcopy(self.temp_config_json)
            for key, value in overwrite_config.items():
                invalid_config[key] = value
        else:
            invalid_config = {}

        with pytest.raises(ValueError, match=error_msg_regex):
            validate_config_values(
                invalid_config, "safetab-h", us_or_puerto_rico_values
            )
