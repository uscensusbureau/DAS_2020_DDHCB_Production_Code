"""System tests for SafeTab-H, making sure that the algorithm has the correct
output for all adaptive levels."""

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
import shutil
import tempfile
import unittest
from itertools import product
from test.system.constants import SAFETAB_H_OUTPUT_FILES
from typing import Any, Dict

import pytest
from parameterized import parameterized, parameterized_class

from tmlt.common.io_helpers import multi_read_csv
from tmlt.safetab_h.paths import INPUT_CONFIG_DIR, RESOURCES_DIR
from tmlt.safetab_h.preprocessing import (
    T3_FAMILY_HOUSEHOLDS,
    T3_FEMALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
    T3_HOUSEHOLDER_LIVING_ALONE,
    T3_HOUSEHOLDER_NOT_LIVING_ALONE,
    T3_MALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
    T3_MARRIED_COUPLE_FAMILY,
    T3_NONFAMILY_HOUSEHOLDS,
    T3_OTHER_FAMILY,
    T3_TOTAL,
    T4_OWNED_FREE_CLEAR,
    T4_OWNED_MORTGAGE_LOAN,
    T4_RENTER_OCCUPIED,
    T4_TOTAL,
)
from tmlt.safetab_h.safetab_h_analytics import execute_plan_h_analytics
from tmlt.safetab_utils.config_validation import CONFIG_PARAMS_H
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.regions import REGION_TYPES, validate_state_filter_us
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    safetab_input_reader,
)


@pytest.mark.usefixtures("spark")
# pylint: disable=no-member
@parameterized_class(
    [
        {
            "name": "SafeTab-H US Analytics Pure DP",
            "us_or_puerto_rico": "US",
            "config_input_dir": "input_dir_puredp",
        },
        {
            "name": "SafeTab-H PR Analytics Pure DP",
            "us_or_puerto_rico": "PR",
            "config_input_dir": "input_dir_puredp",
        },
        {
            "name": "SafeTab-H US Analytics zCDP",
            "us_or_puerto_rico": "US",
            "config_input_dir": "input_dir_zcdp",
        },
        {
            "name": "SafeTab-H PR Analytics zCDP",
            "us_or_puerto_rico": "PR",
            "config_input_dir": "input_dir_zcdp",
        },
    ]
)
class TestAlgorithmWithAdaptiveOutput(unittest.TestCase):
    """Test algorithm compared to target output."""

    name: str
    us_or_puerto_rico: str
    config_input_dir: str

    def setUp(self):
        """Create temporary directories."""
        print(self.name)  # Hint to determine what algorithm is being tested.
        # pylint: disable=consider-using-with
        self.input_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.input_dir.name, "dataset")
        self.actual_dir = tempfile.TemporaryDirectory()
        self.expected_dir = tempfile.TemporaryDirectory()
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)
        self.config_dir = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with
        with open(
            os.path.join(
                os.path.join(self.data_path, self.config_input_dir), "config.json"
            ),
            "r",
        ) as f:
            config_json = json.load(f)
            reader = config_json[READER_FLAG]
            # This test runs either US or PR, never both
            if self.us_or_puerto_rico == "US" and validate_state_filter_us(
                config_json[STATE_FILTER_FLAG]
            ):
                state_filter = config_json[STATE_FILTER_FLAG]
            else:
                state_filter = ["72"]
            validate_input(  # Create configs used to run SafeTab.
                parameters_path=os.path.join(self.data_path, self.config_input_dir),
                input_data_configs_path=INPUT_CONFIG_DIR,
                output_path=self.config_dir.name,
                program="safetab-h",
                input_reader=safetab_input_reader(
                    reader, self.data_path, state_filter, "safetab-h"
                ),
                state_filter=state_filter,
            )

    # 999999 and -999999 are chosen because they will trigger the levels of adaptivity
    # because all counts will be between these values.
    @parameterized.expand(
        [
            [
                {
                    "t3_thresholds": [999999, 999999, 999999],
                    "t4_thresholds": [999999],
                    "t3_domain": [str(T3_TOTAL)],
                    "t4_domain": [str(T4_TOTAL)],
                    "output_files": [
                        os.path.join("t3", "T03001"),
                        os.path.join("t4", "T04001"),
                    ],
                }
            ],
            [
                {
                    "t3_thresholds": [-999999, 999999, 999999],
                    "t4_thresholds": [-999999],
                    "t3_domain": [
                        str(i)
                        for i in [
                            T3_TOTAL,
                            T3_FAMILY_HOUSEHOLDS,
                            T3_NONFAMILY_HOUSEHOLDS,
                        ]
                    ],
                    "t4_domain": [
                        str(i)
                        for i in [
                            T4_TOTAL,
                            T4_OWNED_MORTGAGE_LOAN,
                            T4_OWNED_FREE_CLEAR,
                            T4_RENTER_OCCUPIED,
                        ]
                    ],
                    "output_files": [
                        os.path.join("t3", "T03002"),
                        os.path.join("t4", "T04002"),
                    ],
                }
            ],
            [
                {
                    "t3_thresholds": [-999999, -999999, 999999],
                    "t4_thresholds": [-999999],
                    "t3_domain": [
                        str(i)
                        for i in [
                            T3_TOTAL,
                            T3_FAMILY_HOUSEHOLDS,
                            T3_NONFAMILY_HOUSEHOLDS,
                            T3_OTHER_FAMILY,
                            T3_MARRIED_COUPLE_FAMILY,
                            T3_HOUSEHOLDER_LIVING_ALONE,
                            T3_HOUSEHOLDER_NOT_LIVING_ALONE,
                        ]
                    ],
                    "t4_domain": [
                        str(i)
                        for i in [
                            T4_TOTAL,
                            T4_OWNED_MORTGAGE_LOAN,
                            T4_OWNED_FREE_CLEAR,
                            T4_RENTER_OCCUPIED,
                        ]
                    ],
                    "output_files": [
                        os.path.join("t3", "T03003"),
                        os.path.join("t4", "T04002"),
                    ],
                }
            ],
            [
                {
                    "t3_thresholds": [-999999, -999999, -999999],
                    "t4_thresholds": [-999999],
                    "t3_domain": [
                        str(i)
                        for i in [
                            T3_TOTAL,
                            T3_FAMILY_HOUSEHOLDS,
                            T3_NONFAMILY_HOUSEHOLDS,
                            T3_OTHER_FAMILY,
                            T3_MARRIED_COUPLE_FAMILY,
                            T3_MALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
                            T3_FEMALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
                            T3_HOUSEHOLDER_LIVING_ALONE,
                            T3_HOUSEHOLDER_NOT_LIVING_ALONE,
                        ]
                    ],
                    "t4_domain": [
                        str(i)
                        for i in [
                            T4_TOTAL,
                            T4_OWNED_MORTGAGE_LOAN,
                            T4_OWNED_FREE_CLEAR,
                            T4_RENTER_OCCUPIED,
                        ]
                    ],
                    "output_files": [
                        os.path.join("t3", "T03004"),
                        os.path.join("t4", "T04002"),
                    ],
                }
            ],
        ]
    )
    @pytest.mark.slow
    # This test is not run frequently based on the criticality of the test and runtime
    def test_thresholds(self, params: Dict[str, Any]) -> None:
        """SafeTab-H correctly sums the correct data cells for the corresponding
        threshold."""

        # t3_thresholds: list,
        # t4_thresholds: list,
        # t3_domain: str,
        # t4_domain: str,
        # output_files: list,

        t3_thresholds = params["t3_thresholds"]
        t4_thresholds = params["t4_thresholds"]
        t3_domain = params["t3_domain"]
        t4_domain = params["t4_domain"]
        output_files = params["output_files"]

        inf_budget: Dict[str, Any] = {
            key: float("inf")
            for key in CONFIG_PARAMS_H
            if key.startswith("privacy_budget_h_t")
        }
        thresholds_t3 = {
            "thresholds_h_t3": {
                f"({region_type}, {iteration_level})": t3_thresholds
                for region_type, iteration_level in product(
                    REGION_TYPES["US"] + REGION_TYPES["PR"], ["1", "2"]
                )
            }
        }
        thresholds_t4 = {
            "thresholds_h_t4": {
                f"({region_type}, {iteration_level})": t4_thresholds
                for region_type, iteration_level in product(
                    REGION_TYPES["US"] + REGION_TYPES["PR"], ["1", "2"]
                )
            }
        }
        overwrite_config = {**inf_budget, **thresholds_t3, **thresholds_t4}
        execute_plan_h_analytics(
            os.path.join(self.data_path, self.config_input_dir),
            self.data_path,
            self.actual_dir.name,
            config_path=self.config_dir.name,
            overwrite_config=overwrite_config,
            us_or_puerto_rico=self.us_or_puerto_rico,
        )

        for output_file in SAFETAB_H_OUTPUT_FILES:
            print(f"Checking {output_file}")
            data = multi_read_csv(
                os.path.join(self.actual_dir.name, output_file), sep="|", dtype=str
            )
            if output_file in output_files:
                if output_file[0:2] == "t3":
                    assert set(data["T3_DATA_CELL"].unique()) == set(t3_domain)
                else:
                    assert set(data["T4_DATA_CELL"].unique()) == set(t4_domain)
            else:
                assert len(data) == 0  # The df should be empty.
