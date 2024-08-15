"""System tests for SafeTab-H, making sure that the algorithm has the correct output."""

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

# pylint: disable=protected-access, no-member
import json
import os
import shutil
import tempfile
import unittest
from io import StringIO
from test.conftest import dict_parametrize
from test.system.constants import SAFETAB_H_OUTPUT_FILES
from textwrap import dedent
from typing import Callable, Dict, List

import pandas as pd
import pytest
from parameterized import parameterized, parameterized_class
from pyspark.sql import functions as sf

from tmlt.common.io_helpers import multi_read_csv
from tmlt.common.pyspark_test_tools import assert_frame_equal_with_sort
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
from tmlt.safetab_h.safetab_h_analytics import (
    execute_plan_h_analytics,
    run_plan_h_analytics,
)
from tmlt.safetab_h.target_counts_h import create_ground_truth_h
from tmlt.safetab_utils.config_validation import CONFIG_PARAMS_H
from tmlt.safetab_utils.csv_reader import CSVHReader
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.regions import validate_state_filter_us
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    get_augmented_df,
    safetab_input_reader,
    validate_directory_single_config,
)


@pytest.mark.usefixtures("spark")
class AlgorithmWithTargetOutput(unittest.TestCase):
    """Test algorithm compared to target output."""

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
                    reader=reader,
                    data_path=self.data_path,
                    state_filter=state_filter,
                    program="safetab-h",
                ),
                state_filter=state_filter,
            )

    def infinite_eps_calc(self):
        """SafeTab-H has zero error with infinite budget."""

        overwrite_config = {
            key: float("inf")
            for key in CONFIG_PARAMS_H
            if key.startswith("privacy_budget_h_t")
        }

        overwrite_config["thresholds_h_t3"] = {
            "(USA, 1)": [0, 1, 2],
            "(USA, 2)": [0, 1, 2],
            "(STATE, 1)": [0, 1, 2],
            "(STATE, 2)": [0, 1, 2],
            "(COUNTY, 1)": [0, 1, 2],
            "(COUNTY, 2)": [0, 1, 2],
            "(TRACT, 1)": [0, 1, 2],
            "(TRACT, 2)": [0, 1, 2],
            "(PLACE, 1)": [0, 1, 2],
            "(PLACE, 2)": [0, 1, 2],
            "(AIANNH, 1)": [0, 1, 2],
            "(AIANNH, 2)": [0, 1, 2],
            "(PR-STATE, 1)": [0, 1, 2],
            "(PR-STATE, 2)": [0, 1, 2],
            "(PR-COUNTY, 1)": [0, 1, 2],
            "(PR-COUNTY, 2)": [0, 1, 2],
            "(PR-TRACT, 1)": [0, 1, 2],
            "(PR-TRACT, 2)": [0, 1, 2],
            "(PR-PLACE, 1)": [0, 1, 2],
            "(PR-PLACE, 2)": [0, 1, 2],
        }

        overwrite_config["thresholds_h_t4"] = {
            "(USA, 1)": [0],
            "(USA, 2)": [0],
            "(STATE, 1)": [0],
            "(STATE, 2)": [0],
            "(COUNTY, 1)": [0],
            "(COUNTY, 2)": [0],
            "(TRACT, 1)": [0],
            "(TRACT, 2)": [0],
            "(PLACE, 1)": [0],
            "(PLACE, 2)": [0],
            "(AIANNH, 1)": [0],
            "(AIANNH, 2)": [0],
            "(PR-STATE, 1)": [0],
            "(PR-STATE, 2)": [0],
            "(PR-COUNTY, 1)": [0],
            "(PR-COUNTY, 2)": [0],
            "(PR-TRACT, 1)": [0],
            "(PR-TRACT, 2)": [0],
            "(PR-PLACE, 1)": [0],
            "(PR-PLACE, 2)": [0],
        }

        self.algorithms["execute"](
            parameters_path=os.path.join(self.data_path, self.config_input_dir),
            data_path=self.data_path,
            output_path=self.actual_dir.name,
            config_path=self.config_dir.name,
            overwrite_config=overwrite_config,
            us_or_puerto_rico=self.us_or_puerto_rico,
        )
        self.algorithms["ground_truth"](
            parameters_path=os.path.join(self.data_path, self.config_input_dir),
            data_path=self.data_path,
            output_path=self.expected_dir.name,
            config_path=self.config_dir.name,
            # Overwriting the privacy budget is necessary to tabulate all geo/iteration
            # levels, including those with budget set to 0 in the default config.json.
            overwrite_config=overwrite_config,
            us_or_puerto_rico=self.us_or_puerto_rico,
        )

        for output_file in SAFETAB_H_OUTPUT_FILES:
            print(f"Checking {output_file}")
            augmented_df = get_augmented_df(
                output_file, self.actual_dir.name, self.expected_dir.name
            )
            assert_frame_equal_with_sort(
                augmented_df.rename(columns={"NOISY": "COUNT"}).drop(
                    columns="GROUND_TRUTH"
                ),
                augmented_df.rename(columns={"GROUND_TRUTH": "COUNT"}).drop(
                    columns="NOISY"
                ),
                list(set(augmented_df.columns) - {"NOISY", "GROUND_TRUTH"}),
            )


@parameterized_class(
    [
        {
            "name": "SafeTab-H US Analytics Pure DP",
            "us_or_puerto_rico": "US",
            "algorithms": {
                "execute": execute_plan_h_analytics,
                "ground_truth": create_ground_truth_h,
            },
            "config_input_dir": "input_dir_puredp",
        }
    ]
)
@pytest.mark.usefixtures("spark")
class TestAlgorithmByTargetOutputFast(AlgorithmWithTargetOutput):
    """Child class that is made to run AlgorithmWithTargetOutput in the fast tests."""

    def test_infinite_eps_calc(self):
        """Runs the infinite_eps_calc test every time tests are run."""
        return self.infinite_eps_calc()


@parameterized_class(
    [
        {
            "name": "SafeTab-H PR Analytics Pure DP",
            "us_or_puerto_rico": "PR",
            "algorithms": {
                "execute": execute_plan_h_analytics,
                "ground_truth": create_ground_truth_h,
            },
            "config_input_dir": "input_dir_puredp",
        },
        {
            "name": "SafeTab-H US Analytics zCDP",
            "us_or_puerto_rico": "US",
            "algorithms": {
                "execute": execute_plan_h_analytics,
                "ground_truth": create_ground_truth_h,
            },
            "config_input_dir": "input_dir_zcdp",
        },
        {
            "name": "SafeTab-H PR Analytics zCDP",
            "us_or_puerto_rico": "PR",
            "algorithms": {
                "execute": execute_plan_h_analytics,
                "ground_truth": create_ground_truth_h,
            },
            "config_input_dir": "input_dir_zcdp",
        },
    ]
)
@pytest.mark.usefixtures("spark")
class TestAlgorithmByTargetOutputSlow(AlgorithmWithTargetOutput):
    """Child class that is made to run AlgorithmWithTargetOutput in the slow tests."""

    @pytest.mark.slow
    def test_infinite_eps_calc(self):
        """Runs the infinite_eps_calc test when the program is running slow tests."""
        return self.infinite_eps_calc()


# pylint: disable=line-too-long
GENERAL_ADAPTIVITY_HOUSEHOLDS = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|HHRACE|QRACE1|QRACE2|QRACE3|QRACE4|QRACE5|QRACE6|QRACE7|QRACE8|QSPAN|HOUSEHOLD_TYPE|TEN
01|001|000001|0001|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|1|1
01|001|000001|0001|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|2|2
01|001|000001|0001|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|3|3
01|001|000001|0001|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|4|4
01|001|000001|0001|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|5|4
02|001|000001|0001|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|5|4
"""

GENERAL_ADAPTIVITY_POP_COUNT = """REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
01|STATE|0002|3
01|STATE|0113|3
01001|COUNTY|0002|2
01001|COUNTY|0113|2
01001000001|TRACT|0002|1
01001000001|TRACT|0113|1
02|STATE|0002|0
02|STATE|0113|0
"""
# There are less counts than households, but this input comes from SafeTab-P,
# which is noisy, so this isn't actually unrealistic.


AIANNH_ADAPTIVITY_HOUSEHOLDS = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|HHRACE|QRACE1|QRACE2|QRACE3|QRACE4|QRACE5|QRACE6|QRACE7|QRACE8|QSPAN|HOUSEHOLD_TYPE|TEN
72|031|050902|2041|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|5|4
01|001|000001|0001|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|1|1
01|001|000001|0001|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|2|2
01|001|000001|0001|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|3|3
01|001|000001|0001|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|4|4
01|001|000001|0001|01|1001|Null|Null|Null|Null|Null|Null|Null|1000|5|4
"""
# - The bottom 5 check that adaptivity is working with AIANNH as the only row dif is
# household type and the STATE, COUNTY, and TRACT thresholds will trigger each level.
# - The top row checks that the State filter is removing AIANNH values that aren't in
# the state filter.


AIANNH_ADAPTIVITY_POP_COUNT = """REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
0001|AIANNH|0002|4
0001|AIANNH|0113|4
9999|AIANNH|0002|4
9999|AIANNH|0113|4
"""

AIANNH_GRFC_MULTISTATE = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|PLACEFP|AIANNHCE
01|001|000001|0001|22800|0001
01|001|000001|0002|22800|0001
01|001|000002|0001|22800|0001
01|001|000002|0002|22800|0001
02|001|000002|0001|22801|0001
"""

T3_COLUMNS = ["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "T3_DATA_CELL", "COUNT"]

T4_COLUMNS = ["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "T4_DATA_CELL", "COUNT"]


@parameterized_class(
    [
        {
            "name": "TestTargetCountAIANNHInputSimple",
            "overwrite_config": {
                "state_filter_us": ["01"],
                "thresholds_h_t3": {
                    "(USA, 1)": [1, 2, 3],
                    "(USA, 2)": [1, 2, 3],
                    "(STATE, 1)": [1, 2, 3],
                    "(STATE, 2)": [1, 2, 3],
                    "(COUNTY, 1)": [1, 2, 3],
                    "(COUNTY, 2)": [1, 2, 3],
                    "(TRACT, 1)": [1, 2, 3],
                    "(TRACT, 2)": [1, 2, 3],
                    "(PLACE, 1)": [1, 2, 3],
                    "(PLACE, 2)": [1, 2, 3],
                    "(AIANNH, 1)": [1, 2, 3],
                    "(AIANNH, 2)": [1, 2, 3],
                    "(PR-STATE, 1)": [1, 2, 3],
                    "(PR-STATE, 2)": [1, 2, 3],
                    "(PR-COUNTY, 1)": [1, 2, 3],
                    "(PR-COUNTY, 2)": [1, 2, 3],
                    "(PR-TRACT, 1)": [1, 2, 3],
                    "(PR-TRACT, 2)": [1, 2, 3],
                    "(PR-PLACE, 1)": [1, 2, 3],
                    "(PR-PLACE, 2)": [1, 2, 3],
                },
                "thresholds_h_t4": {
                    "(USA, 1)": [1],
                    "(USA, 2)": [1],
                    "(STATE, 1)": [1],
                    "(STATE, 2)": [1],
                    "(COUNTY, 1)": [1],
                    "(COUNTY, 2)": [1],
                    "(TRACT, 1)": [1],
                    "(TRACT, 2)": [1],
                    "(PLACE, 1)": [1],
                    "(PLACE, 2)": [1],
                    "(AIANNH, 1)": [1],
                    "(AIANNH, 2)": [1],
                    "(PR-STATE, 1)": [1],
                    "(PR-STATE, 2)": [1],
                    "(PR-COUNTY, 1)": [1],
                    "(PR-COUNTY, 2)": [1],
                    "(PR-TRACT, 1)": [1],
                    "(PR-TRACT, 2)": [1],
                    "(PR-PLACE, 1)": [1],
                    "(PR-PLACE, 2)": [1],
                },
            },
            "output_files": {
                os.path.join("t3", "T03004"): pd.DataFrame(
                    [
                        # High Detail
                        ["0001", "AIANNH", "0002", "2", "1"],
                        ["0001", "AIANNH", "0002", "4", "1"],
                        ["0001", "AIANNH", "0002", "5", "1"],
                        ["0001", "AIANNH", "0002", "7", "1"],
                        ["0001", "AIANNH", "0002", "8", "1"],
                        ["0001", "AIANNH", "0113", "2", "1"],
                        ["0001", "AIANNH", "0113", "4", "1"],
                        ["0001", "AIANNH", "0113", "5", "1"],
                        ["0001", "AIANNH", "0113", "7", "1"],
                        ["0001", "AIANNH", "0113", "8", "1"],
                        # Medium Detial
                        ["0001", "AIANNH", "0002", "3", "2"],
                        ["0001", "AIANNH", "0113", "3", "2"],
                        # Low Detail
                        ["0001", "AIANNH", "0002", "1", "3"],
                        ["0001", "AIANNH", "0113", "1", "3"],
                        ["0001", "AIANNH", "0002", "6", "2"],
                        ["0001", "AIANNH", "0113", "6", "2"],
                        # Totals
                        ["0001", "AIANNH", "0002", "0", "5"],
                        ["0001", "AIANNH", "0113", "0", "5"],
                    ],
                    columns=(T3_COLUMNS),
                ),
                os.path.join("t4", "T04002"): pd.DataFrame(
                    [
                        # High Detail
                        ["0001", "AIANNH", "0002", "1", "1"],
                        ["0001", "AIANNH", "0002", "2", "1"],
                        ["0001", "AIANNH", "0002", "3", "3"],
                        ["0001", "AIANNH", "0113", "1", "1"],
                        ["0001", "AIANNH", "0113", "2", "1"],
                        ["0001", "AIANNH", "0113", "3", "3"],
                        # Totals
                        ["0001", "AIANNH", "0002", "0", "5"],
                        ["0001", "AIANNH", "0113", "0", "5"],
                    ],
                    columns=(T4_COLUMNS),
                ),
            },
            "file_overwrite": {
                "household-records.txt": AIANNH_ADAPTIVITY_HOUSEHOLDS,
                "pop-group-totals.txt": AIANNH_ADAPTIVITY_POP_COUNT,
            },
        },
        {
            "name": "TestTargetCountAIANNHInputMultiState",
            "overwrite_config": {
                "state_filter_us": ["01"],
                "thresholds_h_t3": {
                    "(USA, 1)": [1, 2, 3],
                    "(USA, 2)": [1, 2, 3],
                    "(STATE, 1)": [1, 2, 3],
                    "(STATE, 2)": [1, 2, 3],
                    "(COUNTY, 1)": [1, 2, 3],
                    "(COUNTY, 2)": [1, 2, 3],
                    "(TRACT, 1)": [1, 2, 3],
                    "(TRACT, 2)": [1, 2, 3],
                    "(PLACE, 1)": [1, 2, 3],
                    "(PLACE, 2)": [1, 2, 3],
                    "(AIANNH, 1)": [1, 2, 3],
                    "(AIANNH, 2)": [1, 2, 3],
                    "(PR-STATE, 1)": [1, 2, 3],
                    "(PR-STATE, 2)": [1, 2, 3],
                    "(PR-COUNTY, 1)": [1, 2, 3],
                    "(PR-COUNTY, 2)": [1, 2, 3],
                    "(PR-TRACT, 1)": [1, 2, 3],
                    "(PR-TRACT, 2)": [1, 2, 3],
                    "(PR-PLACE, 1)": [1, 2, 3],
                    "(PR-PLACE, 2)": [1, 2, 3],
                },
                "thresholds_h_t4": {
                    "(USA, 1)": [1],
                    "(USA, 2)": [1],
                    "(STATE, 1)": [1],
                    "(STATE, 2)": [1],
                    "(COUNTY, 1)": [1],
                    "(COUNTY, 2)": [1],
                    "(TRACT, 1)": [1],
                    "(TRACT, 2)": [1],
                    "(PLACE, 1)": [1],
                    "(PLACE, 2)": [1],
                    "(AIANNH, 1)": [1],
                    "(AIANNH, 2)": [1],
                    "(PR-STATE, 1)": [1],
                    "(PR-STATE, 2)": [1],
                    "(PR-COUNTY, 1)": [1],
                    "(PR-COUNTY, 2)": [1],
                    "(PR-TRACT, 1)": [1],
                    "(PR-TRACT, 2)": [1],
                    "(PR-PLACE, 1)": [1],
                    "(PR-PLACE, 2)": [1],
                },
            },
            "output_files": {
                os.path.join("t3", "T03004"): pd.DataFrame(
                    [
                        # High Detail
                        ["0001", "AIANNH", "0002", "2", "1"],
                        ["0001", "AIANNH", "0002", "4", "1"],
                        ["0001", "AIANNH", "0002", "5", "1"],
                        ["0001", "AIANNH", "0002", "7", "1"],
                        ["0001", "AIANNH", "0002", "8", "1"],
                        ["0001", "AIANNH", "0113", "2", "1"],
                        ["0001", "AIANNH", "0113", "4", "1"],
                        ["0001", "AIANNH", "0113", "5", "1"],
                        ["0001", "AIANNH", "0113", "7", "1"],
                        ["0001", "AIANNH", "0113", "8", "1"],
                        # Medium Detial
                        ["0001", "AIANNH", "0002", "3", "2"],
                        ["0001", "AIANNH", "0113", "3", "2"],
                        # Low Detail
                        ["0001", "AIANNH", "0002", "1", "3"],
                        ["0001", "AIANNH", "0113", "1", "3"],
                        ["0001", "AIANNH", "0002", "6", "2"],
                        ["0001", "AIANNH", "0113", "6", "2"],
                        # Totals
                        ["0001", "AIANNH", "0002", "0", "5"],
                        ["0001", "AIANNH", "0113", "0", "5"],
                    ],
                    columns=(T3_COLUMNS),
                ),
                os.path.join("t4", "T04002"): pd.DataFrame(
                    [
                        # High Detail
                        ["0001", "AIANNH", "0002", "1", "1"],
                        ["0001", "AIANNH", "0002", "2", "1"],
                        ["0001", "AIANNH", "0002", "3", "3"],
                        ["0001", "AIANNH", "0113", "1", "1"],
                        ["0001", "AIANNH", "0113", "2", "1"],
                        ["0001", "AIANNH", "0113", "3", "3"],
                        # Totals
                        ["0001", "AIANNH", "0002", "0", "5"],
                        ["0001", "AIANNH", "0113", "0", "5"],
                    ],
                    columns=(T4_COLUMNS),
                ),
            },
            "file_overwrite": {
                "household-records.txt": AIANNH_ADAPTIVITY_HOUSEHOLDS,
                "pop-group-totals.txt": AIANNH_ADAPTIVITY_POP_COUNT,
                "GRF-C.txt": AIANNH_GRFC_MULTISTATE,
            },
        },
        {
            "name": "TestTargetCountSmallInput",
            "overwrite_config": {
                "state_filter_us": ["01", "02"],
                "thresholds_h_t3": {
                    "(USA, 1)": [1, 2, 3],
                    "(USA, 2)": [1, 2, 3],
                    "(STATE, 1)": [1, 2, 3],
                    "(STATE, 2)": [1, 2, 3],
                    "(COUNTY, 1)": [1, 2, 3],
                    "(COUNTY, 2)": [1, 2, 3],
                    "(TRACT, 1)": [1, 2, 3],
                    "(TRACT, 2)": [1, 2, 3],
                    "(PLACE, 1)": [1, 2, 3],
                    "(PLACE, 2)": [1, 2, 3],
                    "(AIANNH, 1)": [1, 2, 3],
                    "(AIANNH, 2)": [1, 2, 3],
                    "(PR-STATE, 1)": [1, 2, 3],
                    "(PR-STATE, 2)": [1, 2, 3],
                    "(PR-COUNTY, 1)": [1, 2, 3],
                    "(PR-COUNTY, 2)": [1, 2, 3],
                    "(PR-TRACT, 1)": [1, 2, 3],
                    "(PR-TRACT, 2)": [1, 2, 3],
                    "(PR-PLACE, 1)": [1, 2, 3],
                    "(PR-PLACE, 2)": [1, 2, 3],
                },
                "thresholds_h_t4": {
                    "(USA, 1)": [1],
                    "(USA, 2)": [1],
                    "(STATE, 1)": [1],
                    "(STATE, 2)": [1],
                    "(COUNTY, 1)": [1],
                    "(COUNTY, 2)": [1],
                    "(TRACT, 1)": [1],
                    "(TRACT, 2)": [1],
                    "(PLACE, 1)": [1],
                    "(PLACE, 2)": [1],
                    "(AIANNH, 1)": [1],
                    "(AIANNH, 2)": [1],
                    "(PR-STATE, 1)": [1],
                    "(PR-STATE, 2)": [1],
                    "(PR-COUNTY, 1)": [1],
                    "(PR-COUNTY, 2)": [1],
                    "(PR-TRACT, 1)": [1],
                    "(PR-TRACT, 2)": [1],
                    "(PR-PLACE, 1)": [1],
                    "(PR-PLACE, 2)": [1],
                },
            },
            "output_files": {
                os.path.join("t3", "T03004"): pd.DataFrame(
                    [
                        # High Detail
                        ["01", "STATE", "0002", "2", "1"],
                        ["01", "STATE", "0002", "4", "1"],
                        ["01", "STATE", "0002", "5", "1"],
                        ["01", "STATE", "0002", "7", "1"],
                        ["01", "STATE", "0002", "8", "1"],
                        ["01", "STATE", "0113", "2", "1"],
                        ["01", "STATE", "0113", "4", "1"],
                        ["01", "STATE", "0113", "5", "1"],
                        ["01", "STATE", "0113", "7", "1"],
                        ["01", "STATE", "0113", "8", "1"],
                        # Medium Detail
                        ["01", "STATE", "0002", "3", "2"],
                        ["01", "STATE", "0113", "3", "2"],
                        # Low Detail
                        ["01", "STATE", "0002", "1", "3"],
                        ["01", "STATE", "0002", "6", "2"],
                        ["01", "STATE", "0113", "1", "3"],
                        ["01", "STATE", "0113", "6", "2"],
                        # Totals
                        ["01", "STATE", "0002", "0", "5"],
                        ["01", "STATE", "0113", "0", "5"],
                    ],
                    columns=(T3_COLUMNS),
                ),
                os.path.join("t3", "T03003"): pd.DataFrame(
                    [
                        # Medium Detail
                        ["01001", "COUNTY", "0002", "2", "1"],
                        ["01001", "COUNTY", "0002", "3", "2"],
                        ["01001", "COUNTY", "0002", "7", "1"],
                        ["01001", "COUNTY", "0002", "8", "1"],
                        ["01001", "COUNTY", "0113", "2", "1"],
                        ["01001", "COUNTY", "0113", "3", "2"],
                        ["01001", "COUNTY", "0113", "7", "1"],
                        ["01001", "COUNTY", "0113", "8", "1"],
                        # Low Detail
                        ["01001", "COUNTY", "0002", "1", "3"],
                        ["01001", "COUNTY", "0113", "1", "3"],
                        ["01001", "COUNTY", "0002", "6", "2"],
                        ["01001", "COUNTY", "0113", "6", "2"],
                        # Totals
                        ["01001", "COUNTY", "0002", "0", "5"],
                        ["01001", "COUNTY", "0113", "0", "5"],
                    ],
                    columns=(T3_COLUMNS),
                ),
                os.path.join("t3", "T03002"): pd.DataFrame(
                    [
                        # Low Detail
                        ["01001000001", "TRACT", "0002", "1", "3"],
                        ["01001000001", "TRACT", "0002", "6", "2"],
                        ["01001000001", "TRACT", "0113", "1", "3"],
                        ["01001000001", "TRACT", "0113", "6", "2"],
                        # Totals
                        ["01001000001", "TRACT", "0002", "0", "5"],
                        ["01001000001", "TRACT", "0113", "0", "5"],
                    ],
                    columns=(T3_COLUMNS),
                ),
                os.path.join("t3", "T03001"): pd.DataFrame(
                    [
                        # Total Detail
                        ["02", "STATE", "0002", "0", "1"],
                        ["02", "STATE", "0113", "0", "1"],
                    ],
                    columns=(T3_COLUMNS),
                ),
                os.path.join("t4", "T04002"): pd.DataFrame(
                    [
                        # High Detail
                        ["01", "STATE", "0002", "1", "1"],
                        ["01", "STATE", "0002", "2", "1"],
                        ["01", "STATE", "0002", "3", "3"],
                        ["01", "STATE", "0113", "1", "1"],
                        ["01", "STATE", "0113", "2", "1"],
                        ["01", "STATE", "0113", "3", "3"],
                        ["01001", "COUNTY", "0002", "1", "1"],
                        ["01001", "COUNTY", "0002", "2", "1"],
                        ["01001", "COUNTY", "0002", "3", "3"],
                        ["01001", "COUNTY", "0113", "1", "1"],
                        ["01001", "COUNTY", "0113", "2", "1"],
                        ["01001", "COUNTY", "0113", "3", "3"],
                        ["01001000001", "TRACT", "0002", "1", "1"],
                        ["01001000001", "TRACT", "0002", "2", "1"],
                        ["01001000001", "TRACT", "0002", "3", "3"],
                        ["01001000001", "TRACT", "0113", "1", "1"],
                        ["01001000001", "TRACT", "0113", "2", "1"],
                        ["01001000001", "TRACT", "0113", "3", "3"],
                        # Low Detail
                        ["01", "STATE", "0002", "0", "5"],
                        ["01", "STATE", "0113", "0", "5"],
                        ["01001", "COUNTY", "0002", "0", "5"],
                        ["01001", "COUNTY", "0113", "0", "5"],
                        ["01001000001", "TRACT", "0002", "0", "5"],
                        ["01001000001", "TRACT", "0113", "0", "5"],
                    ],
                    columns=(T4_COLUMNS),
                ),
                os.path.join("t4", "T04001"): pd.DataFrame(
                    [
                        # Low Detail
                        ["02", "STATE", "0002", "0", "1"],
                        ["02", "STATE", "0113", "0", "1"],
                    ],
                    columns=(T4_COLUMNS),
                ),
            },
            "file_overwrite": {
                "household-records.txt": GENERAL_ADAPTIVITY_HOUSEHOLDS,
                "pop-group-totals.txt": GENERAL_ADAPTIVITY_POP_COUNT,
            },
        },
    ]
)
# pylint: enable=line-too-long
@pytest.mark.usefixtures("spark")
class TestTargetCount(unittest.TestCase):
    """Tests that the target_counts_h.py algorithm is correctly calculating the counts
    completely, including adaptivity. The test before this one checks that
    safetab_h_analytics.py creates the same output as target_counts_h.py when the
    epsilon is set to infinity.
    """

    name: str
    overwrite_config: dict
    t3_output: pd.DataFrame
    t4_output: pd.DataFrame
    file_overwrite: dict

    def setUp(self):
        """Create temporary directories."""
        # pylint: disable=consider-using-with
        self.input_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.input_dir.name, "dataset")
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)
        self.output_dir = tempfile.TemporaryDirectory()
        self.config_dir = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with

        for file_path in self.file_overwrite.keys():
            with open(os.path.join(self.data_path, file_path), "w") as datafile:
                datafile.write(self.file_overwrite[file_path])

        validate_input(  # Create configs used to run SafeTab.
            parameters_path=os.path.join(self.data_path, "input_dir_puredp"),
            input_data_configs_path=INPUT_CONFIG_DIR,
            output_path=self.config_dir.name,
            program="safetab-h",
            input_reader=safetab_input_reader(
                reader="csv",
                data_path=self.data_path,
                state_filter=self.overwrite_config["state_filter_us"],
                program="safetab-h",
            ),
            state_filter=self.overwrite_config["state_filter_us"],
        )

    def test_check_accuracy(self):
        """Checks that target_counts_h.py correctly evaluates the counts."""
        create_ground_truth_h(
            parameters_path=os.path.join(self.data_path, "input_dir_puredp"),
            data_path=self.data_path,
            output_path=os.path.join(self.output_dir.name, "ground_truth"),
            config_path=self.config_dir.name,
            # Overwriting the privacy budget is necessary to tabulate all geo/iteration
            # levels, including those with budget set to 0 in the default config.json.
            overwrite_config=self.overwrite_config,
            us_or_puerto_rico="US",
        )

        self.overwrite_config.update(
            {
                key: float("inf")
                for key in CONFIG_PARAMS_H
                if key.startswith("privacy_budget_h_t")
            }
        )

        execute_plan_h_analytics(
            parameters_path=os.path.join(self.data_path, "input_dir_puredp"),
            data_path=self.data_path,
            output_path=os.path.join(self.output_dir.name, "analytics"),
            config_path=self.config_dir.name,
            # Overwriting the privacy budget is necessary to tabulate all geo/iteration
            # levels, including those with budget set to 0 in the default config.json.
            overwrite_config=self.overwrite_config,
            us_or_puerto_rico="US",
        )

        for output_file in self.output_files.keys():
            print(f"Checking {output_file}")
            print("Checking Ground Truth Files")
            ground_truth_df = multi_read_csv(
                os.path.join(self.output_dir.name, "ground_truth", output_file),
                sep="|",
                dtype=str,
            )
            print("Checking Analytics Files")
            analytics_df = multi_read_csv(
                os.path.join(self.output_dir.name, "analytics", output_file),
                sep="|",
                dtype=str,
            )
            expected_df = self.output_files[output_file]

            assert_frame_equal_with_sort(ground_truth_df, expected_df)
            assert_frame_equal_with_sort(analytics_df, expected_df)


@parameterized_class(
    [
        {
            "algorithms": {"run": run_plan_h_analytics},
            "config_input_dir": "input_dir_puredp",
        },
        {
            "algorithms": {"run": run_plan_h_analytics},
            "config_input_dir": "input_dir_zcdp",
        },
    ]
)
@pytest.mark.usefixtures("spark")
class TestAlgorithmAlone(unittest.TestCase):
    """Test algorithm output by itself."""

    algorithms: Dict[str, Callable]
    config_input_dir: str

    def setUp(self):
        """Create temporary directories."""
        # pylint: disable=consider-using-with
        self.input_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.input_dir.name, "dataset")
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)
        self.output_dir = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with

    @parameterized.expand([(False,), (True,)])
    @pytest.mark.slow
    # This test is not run frequently based on the criticality of the test and runtime
    def test_non_negativity_post_processing(self, allow_negative_counts_flag: bool):
        """SafeTab-H eliminates negative values in output based on flag.

        Args:
            allow_negative_counts_flag: Whether or not negative counts are allowed.
        """
        self.algorithms["run"](
            os.path.join(self.data_path, self.config_input_dir),
            self.data_path,
            self.output_dir.name,
            overwrite_config={"allow_negative_counts": allow_negative_counts_flag},
        )
        output_dfs = []
        for output_file in SAFETAB_H_OUTPUT_FILES:
            output_dfs.append(
                multi_read_csv(
                    os.path.join(self.output_dir.name, output_file),
                    dtype=int,
                    sep="|",
                    usecols=["COUNT"],
                )
            )
        df = pd.concat(output_dfs)
        contains_negatives = any(df["COUNT"] < 0)
        assert contains_negatives == allow_negative_counts_flag

    @pytest.mark.slow  # This test is not run frequently as it takes longer than 10 minutes
    def test_output_format(self):
        """SafeTab-H output files pass validation.

        See resources/config/output, and `Appendix A` for details about the expected
        output formats.
        """
        self.algorithms["run"](
            os.path.join(self.data_path, self.config_input_dir),
            self.data_path,
            self.output_dir.name,
            overwrite_config={
                "run_us": True,
                "run_pr": True,
                "allow_negative_counts": True,
            },
            should_validate_private_output=True,
        )
        for output_file in SAFETAB_H_OUTPUT_FILES:
            output_type = "t3" if output_file.startswith("t3") else "t4"
            assert validate_directory_single_config(
                os.path.join(self.output_dir.name, output_file),
                os.path.join(RESOURCES_DIR, f"config/output/{output_type}.json"),
                delimiter="|",
            )

    @pytest.mark.slow
    # This test is not run frequently based on the criticality of the test and runtime
    def test_exclude_states(self):
        """SafeTab-H can exclude specific states from tabulation."""
        include_states = ["02"]

        # Setting the privacy budget for aiannh to 0 because these can cross state lines.
        overwrite_config = {"state_filter_us": include_states}

        self.algorithms["run"](
            os.path.join(self.data_path, self.config_input_dir),
            self.data_path,
            self.output_dir.name,
            overwrite_config=overwrite_config,
        )

        output_dfs = []
        for output_file in SAFETAB_H_OUTPUT_FILES:
            output_dfs.append(
                multi_read_csv(
                    os.path.join(self.output_dir.name, output_file),
                    dtype=str,
                    sep="|",
                    usecols=["REGION_TYPE", "REGION_ID"],
                )
            )
        df = pd.concat(output_dfs)
        df = df[df["REGION_TYPE"] == "STATE"]
        actual = df["REGION_ID"].unique()
        assert actual == include_states


@parameterized_class(
    [
        {
            "algorithms": {"run": run_plan_h_analytics},
            "config_input_dir": "input_dir_puredp",
        },
        {
            "algorithms": {"run": run_plan_h_analytics},
            "config_input_dir": "input_dir_zcdp",
        },
    ]
)
@pytest.mark.usefixtures("spark")
class TestAlgorithmsUSandPR(unittest.TestCase):
    """Test algorithm runs on US and/or Puerto Rico."""

    algorithms: Dict[str, Callable]
    output_files: List[str]
    config_input_dir: str

    def setUp(self):
        """Create temporary directories."""
        # pylint: disable=consider-using-with
        self.input_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.input_dir.name, "dataset")
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)
        self.output_dir = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with

    @parameterized.expand([(False, True), (True, False), (True, True)])
    @pytest.mark.slow
    # This test is not run frequently based on the criticality of the test and runtime
    def test_run_us_run_pr(self, run_us: bool, run_pr: bool):
        """SafeTab-H can run for US, PR or both.

        Args:
            run_us: Whether to run SafeTab-H on the US geographies.
            run_pr: Whether to run SafeTab-H on Puerto Rico.
        """

        overwrite_config = {
            "thresholds_h_t3": {
                "(USA, 1)": [5, 10, 15],
                "(USA, 2)": [5, 10, 15],
                "(STATE, 1)": [5, 10, 15],
                "(STATE, 2)": [5, 10, 15],
                "(COUNTY, 1)": [5, 10, 15],
                "(COUNTY, 2)": [5, 10, 15],
                "(TRACT, 1)": [5, 10, 15],
                "(TRACT, 2)": [5, 10, 15],
                "(PLACE, 1)": [5, 10, 15],
                "(PLACE, 2)": [5, 10, 15],
                "(AIANNH, 1)": [5, 10, 15],
                "(AIANNH, 2)": [5, 10, 15],
                "(PR-STATE, 1)": [5, 10, 15],
                "(PR-STATE, 2)": [5, 10, 15],
                "(PR-COUNTY, 1)": [5, 10, 15],
                "(PR-COUNTY, 2)": [5, 10, 15],
                "(PR-TRACT, 1)": [5, 10, 15],
                "(PR-TRACT, 2)": [5, 10, 15],
                "(PR-PLACE, 1)": [5, 10, 15],
                "(PR-PLACE, 2)": [5, 10, 15],
            },
            "thresholds_h_t4": {
                "(USA, 1)": [10],
                "(USA, 2)": [10],
                "(STATE, 1)": [10],
                "(STATE, 2)": [10],
                "(COUNTY, 1)": [10],
                "(COUNTY, 2)": [10],
                "(TRACT, 1)": [10],
                "(TRACT, 2)": [10],
                "(PLACE, 1)": [10],
                "(PLACE, 2)": [10],
                "(AIANNH, 1)": [10],
                "(AIANNH, 2)": [10],
                "(PR-STATE, 1)": [10],
                "(PR-STATE, 2)": [10],
                "(PR-COUNTY, 1)": [10],
                "(PR-COUNTY, 2)": [10],
                "(PR-TRACT, 1)": [10],
                "(PR-TRACT, 2)": [10],
                "(PR-PLACE, 1)": [10],
                "(PR-PLACE, 2)": [10],
            },
            "run_us": run_us,
            "run_pr": run_pr,
        }

        self.algorithms["run"](
            os.path.join(self.data_path, self.config_input_dir),
            self.data_path,
            self.output_dir.name,
            overwrite_config=overwrite_config,
        )
        output_dfs = []
        for output_file in SAFETAB_H_OUTPUT_FILES:
            output_dfs.append(
                multi_read_csv(
                    os.path.join(self.output_dir.name, output_file),
                    dtype=str,
                    sep="|",
                    usecols=["REGION_TYPE", "REGION_ID"],
                )
            )
        df = pd.concat(output_dfs)
        assert ("STATE" in df["REGION_TYPE"].values) == run_us
        assert ("PR-STATE" in df["REGION_TYPE"].values) == run_pr

    @pytest.mark.slow
    # This test is not run frequently based on the criticality of the test and runtime
    def test_run_us_run_pr_both_false(self):
        """SafeTab-H fails if run_us and run_pr are False."""
        with pytest.raises(
            ValueError,
            match="Invalid config: At least one of 'run_us', 'run_pr' must be True.",
        ):
            self.algorithms["run"](
                os.path.join(self.data_path, self.config_input_dir),
                self.data_path,
                self.output_dir.name,
                overwrite_config={"run_us": False, "run_pr": False},
            )


@parameterized_class(
    [
        {
            "name": "safetab_h_puredp",
            "algorithm": {"run": run_plan_h_analytics},
            "input_dir_name": "input_dir_puredp",
        },
        {
            "name": "safetab_h_zcdp",
            "algorithm": {"run": run_plan_h_analytics},
            "input_dir_name": "input_dir_zcdp",
        },
    ]
)
@pytest.mark.usefixtures("spark")
class TestPopulationGroups(unittest.TestCase):
    """Test that SafeTab-H tabulates the correct population groups."""

    algorithm: Dict[str, Callable]
    output_files: List[str]
    input_dir_name: str

    def setUp(self):
        # pylint: disable=consider-using-with
        self.reader_config_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.reader_config_dir.name, "dataset")
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)
        self.parameters_path = os.path.join(self.data_path, self.input_dir_name)
        self.output_dir = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with
        self.output_path = self.output_dir.name

    @parameterized.expand(
        [
            (True, False, ["01", "02", "11"]),
            (False, True, ["01", "02", "11"]),
            (True, True, ["01", "02", "11"]),
            (True, True, ["01"]),
        ]
    )
    @pytest.mark.slow
    def test_pop_groups(self, run_us: bool, run_pr: bool, include_states: List[str]):
        """Test that SafeTab-H tabulates the correct population groups."""
        overwrite_config: Dict = {
            query: float("inf")
            for query in CONFIG_PARAMS_H
            if query.startswith("privacy_budget_h_t")
        }
        overwrite_config.update(
            {"run_us": run_us, "run_pr": run_pr, "state_filter_us": include_states}
        )

        run_plan_h_analytics(
            parameters_path=self.parameters_path,
            data_path=self.data_path,
            output_path=self.output_path,
            overwrite_config=overwrite_config,
        )

        actual_dfs = []
        for output_file in SAFETAB_H_OUTPUT_FILES:
            actual_dfs.append(
                multi_read_csv(
                    os.path.join(self.output_path, output_file),
                    dtype=str,
                    sep="|",
                    usecols=["REGION_ID", "REGION_TYPE", "ITERATION_CODE"],
                )
            )
        actual_df = pd.concat(actual_dfs).drop_duplicates()

        # Reading in the pop_groups_df with CSVHReader drops population groups with
        # RegionIDs that do not fall in a state included in the state filter.
        if run_pr and run_us:
            reader_state_filter = include_states + ["72"]
        elif run_pr and not run_us:
            reader_state_filter = ["72"]
        else:
            assert run_us and not run_pr
            reader_state_filter = include_states.copy()

        csv_reader = CSVHReader(
            config_path=self.data_path, state_filter=reader_state_filter
        )
        pop_groups_df = csv_reader.get_pop_group_details_df().select(
            "REGION_ID", "REGION_TYPE", "ITERATION_CODE"
        )
        if not run_us:
            pop_groups_df = pop_groups_df.filter(sf.col("REGION_TYPE") != "USA")

        assert_frame_equal_with_sort(actual_df, pop_groups_df.toPandas())


@parameterized_class(
    [
        {
            "algorithms": {"run": execute_plan_h_analytics},
            "config_input_dir": "input_dir_puredp",
        }
    ]
)
@pytest.mark.usefixtures("spark")
class TestInvalidInput(unittest.TestCase):
    """Verify that SafeTab-H raises exceptions on invalid input."""

    algorithms: Dict[str, Callable]
    config_input_dir: str

    def setUp(self):
        """Create temporary directories."""
        self.data_path = str(os.path.join(RESOURCES_DIR, "toy_dataset"))
        # pylint: disable=consider-using-with
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
            # This test just runs US because execute_plan_h_analytics defaults to US
            _ = validate_state_filter_us(config_json[STATE_FILTER_FLAG])
            state_filter = config_json[STATE_FILTER_FLAG]
            validate_input(  # Create configs used to run SafeTab.
                parameters_path=os.path.join(self.data_path, self.config_input_dir),
                input_data_configs_path=INPUT_CONFIG_DIR,
                output_path=self.config_dir.name,
                program="safetab-h",
                input_reader=safetab_input_reader(
                    reader=reader,
                    data_path=self.data_path,
                    state_filter=state_filter,
                    program="safetab-h",
                ),
                state_filter=state_filter,
            )

    @parameterized.expand(
        [
            ("s3://dummy", "dummy/"),
            ("dummy/", "s3://dummy"),
            ("s3://dummy", "s3://dummy"),
            ("s3a://dummy", "dummy/"),
            ("dummy/", "s3a://dummy"),
            ("s3a://dummy", "s3a://dummy"),
        ]
    )
    @pytest.mark.slow
    # This test is not run frequently based on the criticality of the test and runtime
    def test_read_from_s3_spark_local(self, input_path: str, output_path: str):
        """SafeTab-H fails if input directory is on s3 and in Spark local mode.

        Args:
            input_path: The directory that contains the input files.
            output_path: The path where the output should be saved.
        """
        with pytest.raises(
            RuntimeError,
            match=(
                "Reading and writing to and from s3"
                " is not supported when running Spark in local mode."
            ),
        ):
            self.algorithms["run"](
                parameters_path=input_path,
                data_path=input_path,
                output_path=output_path,
                config_path="dummy_config_dir",
            )


SIMPLE_HOUSEHOLDS_FILE = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|HHRACE|QRACE1|QRACE2|QRACE3|QRACE4|QRACE5|QRACE6|QRACE7|QRACE8|QSPAN|HOUSEHOLD_TYPE|TEN
01|001|000001|0001|01|1000|Null|Null|Null|Null|Null|Null|Null|1000|1|4
"""

SIMPLE_PR_HOUSEHOLDS_FILE = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|HHRACE|QRACE1|QRACE2|QRACE3|QRACE4|QRACE5|QRACE6|QRACE7|QRACE8|QSPAN|HOUSEHOLD_TYPE|TEN
72|001|000001|0001|01|1000|Null|Null|Null|Null|Null|Null|Null|1000|1|4
"""

SIMPLE_GRFC = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|PLACEFP|AIANNHCE
01|001|000001|0001|22800|0001
"""


SIMPLE_ETHNICITY_ITERATIONS = """ITERATION_CODE|ITERATION_NAME|LEVEL|DETAILED_ONLY|COARSE_ONLY
3010|Central American|1|False|False
3011|Costa Rican|2|False|False"""

SIMPLE_RACE_ITERATIONS = """ITERATION_CODE|ITERATION_NAME|LEVEL|ALONE|DETAILED_ONLY|COARSE_ONLY
0002|European alone|1|True|False|False
0003|Albanian alone|2|True|False|False"""

SIMPLE_RACE_CODES = """RACE_ETH_CODE|RACE_ETH_NAME
1000|White
1340|Slovak"""

SIMPLE_CODE_ITERATION_MAP = """ITERATION_CODE|RACE_ETH_CODE
0002|1000
0003|1000
3010|1340
3011|1340"""

SIMPLE_US_GRFC = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|PLACEFP|AIANNHCE
01|001|000001|0001|22800|0001
"""

SIMPLE_PR_GRFC = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|PLACEFP|AIANNHCE
72|001|000001|0001|22800|0001
"""

FOUR_POPGROUP_T1_OUTPUT = """REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
1|USA|0002|5
1|USA|0003|15
1|USA|3010|25
1|USA|3011|35"""

SIMPLE_CONFIG = """{
  "max_race_codes": 8,
  "privacy_budget_h_t3_level_1_usa": 1,
  "privacy_budget_h_t3_level_2_usa": 1,
  "privacy_budget_h_t3_level_1_state": 1,
  "privacy_budget_h_t3_level_2_state": 1,
  "privacy_budget_h_t3_level_1_county": 1,
  "privacy_budget_h_t3_level_2_county": 1,
  "privacy_budget_h_t3_level_1_tract": 1,
  "privacy_budget_h_t3_level_2_tract": 1,
  "privacy_budget_h_t3_level_1_place": 1,
  "privacy_budget_h_t3_level_2_place": 1,
  "privacy_budget_h_t3_level_1_aiannh": 1,
  "privacy_budget_h_t3_level_2_aiannh": 1,
  "privacy_budget_h_t3_level_1_pr_state": 1,
  "privacy_budget_h_t3_level_2_pr_state": 1,
  "privacy_budget_h_t3_level_1_pr_county": 1,
  "privacy_budget_h_t3_level_2_pr_county": 1,
  "privacy_budget_h_t3_level_1_pr_tract": 1,
  "privacy_budget_h_t3_level_2_pr_tract": 1,
  "privacy_budget_h_t3_level_1_pr_place": 1,
  "privacy_budget_h_t3_level_2_pr_place": 1,
  "privacy_budget_h_t4_level_1_usa": 1,
  "privacy_budget_h_t4_level_2_usa": 1,
  "privacy_budget_h_t4_level_1_state": 1,
  "privacy_budget_h_t4_level_2_state": 1,
  "privacy_budget_h_t4_level_1_county": 1,
  "privacy_budget_h_t4_level_2_county": 1,
  "privacy_budget_h_t4_level_1_tract": 1,
  "privacy_budget_h_t4_level_2_tract": 1,
  "privacy_budget_h_t4_level_1_place": 1,
  "privacy_budget_h_t4_level_2_place": 1,
  "privacy_budget_h_t4_level_1_aiannh": 1,
  "privacy_budget_h_t4_level_2_aiannh": 1,
  "privacy_budget_h_t4_level_1_pr_state": 1,
  "privacy_budget_h_t4_level_2_pr_state": 1,
  "privacy_budget_h_t4_level_1_pr_county": 1,
  "privacy_budget_h_t4_level_2_pr_county": 1,
  "privacy_budget_h_t4_level_1_pr_tract": 1,
  "privacy_budget_h_t4_level_2_pr_tract": 1,
  "privacy_budget_h_t4_level_1_pr_place": 1,
  "privacy_budget_h_t4_level_2_pr_place": 1,
  "thresholds_h_t3": {
    "(USA, 1)": [5000, 20000, 150000],
    "(USA, 2)": [500, 1000, 7000],
    "(STATE, 1)": [5000, 20000, 150000],
    "(STATE, 2)": [500, 1000, 7000],
    "(COUNTY, 1)": [5000, 20000, 150000],
    "(COUNTY, 2)": [1000, 5000, 20000],
    "(TRACT, 1)": [5000, 20000, 150000],
    "(TRACT, 2)": [1000, 5000, 20000],
    "(PLACE, 1)": [5000, 20000, 150000],
    "(PLACE, 2)": [1000, 5000, 20000],
    "(AIANNH, 1)": [5000, 20000, 150000],
    "(AIANNH, 2)": [1000, 5000, 20000],
    "(PR-STATE, 1)": [5000, 20000, 150000],
    "(PR-STATE, 2)": [500, 1000, 7000],
    "(PR-COUNTY, 1)": [5000, 20000, 150000],
    "(PR-COUNTY, 2)": [1000, 5000, 20000],
    "(PR-TRACT, 1)": [5000, 20000, 150000],
    "(PR-TRACT, 2)": [1000, 5000, 20000],
    "(PR-PLACE, 1)": [5000, 20000, 150000],
    "(PR-PLACE, 2)": [1000, 5000, 20000]
  },
  "thresholds_h_t4": {
    "(USA, 1)": [5000],
    "(USA, 2)": [500],
    "(STATE, 1)": [5000],
    "(STATE, 2)": [500],
    "(COUNTY, 1)": [5000],
    "(COUNTY, 2)": [1000],
    "(TRACT, 1)": [5000],
    "(TRACT, 2)": [1000],
    "(PLACE, 1)": [5000],
    "(PLACE, 2)": [1000],
    "(AIANNH, 1)": [5000],
    "(AIANNH, 2)": [1000],
    "(PR-STATE, 1)": [5000],
    "(PR-STATE, 2)": [500],
    "(PR-COUNTY, 1)": [5000],
    "(PR-COUNTY, 2)": [1000],
    "(PR-TRACT, 1)": [5000],
    "(PR-TRACT, 2)": [1000],
    "(PR-PLACE, 1)": [5000],
    "(PR-PLACE, 2)": [1000]
  },
  "allow_negative_counts": true,
  "run_us": true,
  "run_pr": false,
  "reader": "csv",
  "state_filter_us": ["01"],
  "privacy_defn": "puredp"
}
"""

THRESHOLDS_T3_10_20_30 = {
    "(USA, 1)": [10, 20, 30],
    "(USA, 2)": [10, 20, 30],
    "(STATE, 1)": [10, 20, 30],
    "(STATE, 2)": [10, 20, 30],
    "(COUNTY, 1)": [10, 20, 30],
    "(COUNTY, 2)": [10, 20, 30],
    "(TRACT, 1)": [10, 20, 30],
    "(TRACT, 2)": [10, 20, 30],
    "(PLACE, 1)": [10, 20, 30],
    "(PLACE, 2)": [10, 20, 30],
    "(AIANNH, 1)": [10, 20, 30],
    "(AIANNH, 2)": [10, 20, 30],
    "(PR-STATE, 1)": [10, 20, 30],
    "(PR-STATE, 2)": [10, 20, 30],
    "(PR-COUNTY, 1)": [10, 20, 30],
    "(PR-COUNTY, 2)": [10, 20, 30],
    "(PR-TRACT, 1)": [10, 20, 30],
    "(PR-TRACT, 2)": [10, 20, 30],
    "(PR-PLACE, 1)": [10, 20, 30],
    "(PR-PLACE, 2)": [10, 20, 30],
}

THRESHOLDS_T4_20 = {
    "(USA, 1)": [20],
    "(USA, 2)": [20],
    "(STATE, 1)": [20],
    "(STATE, 2)": [20],
    "(COUNTY, 1)": [20],
    "(COUNTY, 2)": [20],
    "(TRACT, 1)": [20],
    "(TRACT, 2)": [20],
    "(PLACE, 1)": [20],
    "(PLACE, 2)": [20],
    "(AIANNH, 1)": [20],
    "(AIANNH, 2)": [20],
    "(PR-STATE, 1)": [20],
    "(PR-STATE, 2)": [20],
    "(PR-COUNTY, 1)": [20],
    "(PR-COUNTY, 2)": [20],
    "(PR-TRACT, 1)": [20],
    "(PR-TRACT, 2)": [20],
    "(PR-PLACE, 1)": [20],
    "(PR-PLACE, 2)": [20],
}
# pylint: enable=line-too-long


def write_input_files(
    input_dir: str,
    households: str,
    t1_output: str,
    ethnicity_iterations: str,
    race_iterations: str,
    race_eth_codes: str,
    iteration_map: str,
    grfc: str,
    config: str,
):
    """Write a complete set of input files to a directory."""
    with open(os.path.join(input_dir, "GRF-C.txt"), "w") as grfc_file:
        grfc_file.write(grfc)

    with open(os.path.join(input_dir, "household-records.txt"), "w") as households_file:
        households_file.write(households)

    with open(os.path.join(input_dir, "pop-group-totals.txt"), "w") as t1_file:
        t1_file.write(t1_output)

    config_dir = os.path.join(input_dir, "config")
    os.mkdir(config_dir)

    with open(os.path.join(config_dir, "config.json"), "w") as config_file:
        config_file.write(config)

    with open(
        os.path.join(config_dir, "ethnicity-characteristic-iterations.txt"), "w"
    ) as ethnicity_file:
        ethnicity_file.write(ethnicity_iterations)

    with open(
        os.path.join(config_dir, "race-characteristic-iterations.txt"), "w"
    ) as race_file:
        race_file.write(race_iterations)

    with open(
        os.path.join(config_dir, "race-and-ethnicity-codes.txt"), "w"
    ) as race_eth_codes_file:
        race_eth_codes_file.write(race_eth_codes)

    with open(
        os.path.join(config_dir, "race-and-ethnicity-code-to-iteration.txt"), "w"
    ) as iteration_map_file:
        iteration_map_file.write(iteration_map)


@pytest.mark.usefixtures("spark")
class TestOutputPostprocessing(unittest.TestCase):
    """Verify that safetab-h output is correctly postprocessed."""

    def test_split_full_tables(self):
        """Check that output is divided based on stat level, and that we produce full
        output tables (with marginals and totals).

        The input is arranged so that one popgroup will meet each stat level."""

        input_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        write_input_files(
            input_dir=input_dir.name,
            households=SIMPLE_HOUSEHOLDS_FILE,
            t1_output=FOUR_POPGROUP_T1_OUTPUT,
            ethnicity_iterations=SIMPLE_ETHNICITY_ITERATIONS,
            race_iterations=SIMPLE_RACE_ITERATIONS,
            race_eth_codes=SIMPLE_RACE_CODES,
            iteration_map=SIMPLE_CODE_ITERATION_MAP,
            grfc=SIMPLE_GRFC,
            config=SIMPLE_CONFIG,
        )
        # pylint: disable=consider-using-with
        output_dir = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with

        run_plan_h_analytics(
            parameters_path=os.path.join(input_dir.name, "config"),
            data_path=input_dir.name,
            output_path=output_dir.name,
            overwrite_config={
                "thresholds_h_t3": THRESHOLDS_T3_10_20_30,
                "thresholds_h_t4": THRESHOLDS_T4_20,
            },
        )

        t03001 = multi_read_csv(
            os.path.join(output_dir.name, "t3", "T03001"), dtype=str, sep="|"
        )
        t03002 = multi_read_csv(
            os.path.join(output_dir.name, "t3", "T03002"), dtype=str, sep="|"
        )
        t03003 = multi_read_csv(
            os.path.join(output_dir.name, "t3", "T03003"), dtype=str, sep="|"
        )
        t03004 = multi_read_csv(
            os.path.join(output_dir.name, "t3", "T03004"), dtype=str, sep="|"
        )
        t04001 = multi_read_csv(
            os.path.join(output_dir.name, "t4", "T04001"), dtype=str, sep="|"
        )
        t04002 = multi_read_csv(
            os.path.join(output_dir.name, "t4", "T04002"), dtype=str, sep="|"
        )

        assert len(t03001.index) == 1
        assert set(t03001["T3_DATA_CELL"].astype(int).unique()) == {T3_TOTAL}

        assert len(t03002.index) == 3
        assert set(t03002["T3_DATA_CELL"].astype(int).unique()) == {
            T3_TOTAL,
            T3_FAMILY_HOUSEHOLDS,
            T3_NONFAMILY_HOUSEHOLDS,
        }

        assert len(t03003.index) == 7
        assert set(t03003["T3_DATA_CELL"].astype(int).unique()) == {
            T3_TOTAL,
            T3_FAMILY_HOUSEHOLDS,
            T3_MARRIED_COUPLE_FAMILY,
            T3_OTHER_FAMILY,
            T3_NONFAMILY_HOUSEHOLDS,
            T3_HOUSEHOLDER_LIVING_ALONE,
            T3_HOUSEHOLDER_NOT_LIVING_ALONE,
        }

        assert len(t03004.index) == 9
        assert set(t03004["T3_DATA_CELL"].astype(int).unique()) == {
            T3_TOTAL,
            T3_FAMILY_HOUSEHOLDS,
            T3_MARRIED_COUPLE_FAMILY,
            T3_MALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
            T3_FEMALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
            T3_OTHER_FAMILY,
            T3_NONFAMILY_HOUSEHOLDS,
            T3_HOUSEHOLDER_LIVING_ALONE,
            T3_HOUSEHOLDER_NOT_LIVING_ALONE,
        }

        # Each of the t4 dataframes will have two popgroups in it.
        assert len(t04001.index) == 2
        assert set(t04001["T4_DATA_CELL"].astype(int).unique()) == {T4_TOTAL}

        assert len(t04002.index) == 8
        assert set(t04002["T4_DATA_CELL"].astype(int).unique()) == {
            T4_TOTAL,
            T4_OWNED_MORTGAGE_LOAN,
            T4_OWNED_FREE_CLEAR,
            T4_RENTER_OCCUPIED,
        }


INFINITE_BUDGETS = {
    key: float("inf") for key in CONFIG_PARAMS_H if key.startswith("privacy_budget_h_t")
}


RACE_ITERATIONS_WITH_DETAILED_ONLY = dedent(
    """    ITERATION_CODE|ITERATION_NAME|LEVEL|ALONE|DETAILED_ONLY|COARSE_ONLY
    0002|European alone|1|True|False|False
    0003|Albanian alone|2|True|True|False"""
)


@pytest.fixture(name="input_dir")
def fixture_input_dir():
    """Returns a temporary directory to use for input files."""
    with tempfile.TemporaryDirectory() as to_return:
        yield to_return


@pytest.fixture(name="output_dir")
def fixture_output_dir():
    """Returns a temporary directory to use for input files."""
    with tempfile.TemporaryDirectory() as to_return:
        yield to_return


@dict_parametrize(
    {
        "detailed_only_USA": {
            "detailed_only": True,
            "coarse_only": False,
            "households": SIMPLE_HOUSEHOLDS_FILE,
            "region_type_to_test": "USA",
            "region_id_to_test": "1",
            "grfc": SIMPLE_GRFC,
            "run_pr": False,
        },
        "detailed_only_PR-STATE": {
            "detailed_only": True,
            "coarse_only": False,
            "households": SIMPLE_PR_HOUSEHOLDS_FILE,
            "region_type_to_test": "PR-STATE",
            "region_id_to_test": "72",
            "grfc": SIMPLE_PR_GRFC,
            "run_pr": True,
        },
        "coarse_only_COUNTY": {
            "detailed_only": False,
            "coarse_only": True,
            "households": SIMPLE_HOUSEHOLDS_FILE,
            "region_type_to_test": "COUNTY",
            "region_id_to_test": "01001",
            "grfc": SIMPLE_GRFC,
            "run_pr": False,
        },
        "coarse_only_PR-COUNTY": {
            "detailed_only": False,
            "coarse_only": True,
            "households": SIMPLE_PR_HOUSEHOLDS_FILE,
            "region_type_to_test": "PR-COUNTY",
            "region_id_to_test": "72001",
            "grfc": SIMPLE_PR_GRFC,
            "run_pr": True,
        },
        "common_STATE": {
            "detailed_only": False,
            "coarse_only": False,
            "households": SIMPLE_HOUSEHOLDS_FILE,
            "region_type_to_test": "STATE",
            "region_id_to_test": "01",
            "grfc": SIMPLE_GRFC,
            "run_pr": False,
        },
        "common_PR-STATE": {
            "detailed_only": False,
            "coarse_only": False,
            "households": SIMPLE_PR_HOUSEHOLDS_FILE,
            "region_type_to_test": "PR-STATE",
            "region_id_to_test": "72",
            "grfc": SIMPLE_PR_GRFC,
            "run_pr": True,
        },
        "common_COUNTY": {
            "detailed_only": False,
            "coarse_only": False,
            "households": SIMPLE_HOUSEHOLDS_FILE,
            "region_type_to_test": "COUNTY",
            "region_id_to_test": "01001",
            "grfc": SIMPLE_GRFC,
            "run_pr": False,
        },
        "common_PR-COUNTY": {
            "detailed_only": False,
            "coarse_only": False,
            "households": SIMPLE_PR_HOUSEHOLDS_FILE,
            "region_type_to_test": "PR-COUNTY",
            "region_id_to_test": "72001",
            "grfc": SIMPLE_PR_GRFC,
            "run_pr": True,
        },
    }
)
@dict_parametrize(
    {
        "T03001+T04001": {
            "t1_count": "5",
            "t3_output_file": "T03001",
            "t4_output_file": "T04001",
        },
        "T03002+T04001": {
            "t1_count": "15",
            "t3_output_file": "T03002",
            "t4_output_file": "T04001",
        },
        "T03003+T04002": {
            "t1_count": "25",
            "t3_output_file": "T03003",
            "t4_output_file": "T04002",
        },
        "T03004+T04002": {
            "t1_count": "35",
            "t3_output_file": "T03004",
            "t4_output_file": "T04002",
        },
    }
)
@dict_parametrize(
    {
        "Race_Alone": {"iteration_type": "RACE", "alone": True},
        "Race_Not_Alone": {"iteration_type": "RACE", "alone": False},
        "Ethnicity": {"iteration_type": "ETHNICITY", "alone": True},
    }
)
@pytest.mark.usefixtures("spark")
@pytest.mark.slow
def test_iteration_has_data(
    input_dir,
    output_dir,
    households,
    iteration_type,
    coarse_only,
    detailed_only,
    alone,
    grfc,
    region_id_to_test,
    region_type_to_test,
    t1_count,
    t3_output_file,
    t4_output_file,
    run_pr,
):
    """Test that we correctly count detail-only iterations."""
    if iteration_type == "RACE":
        race_iterations = (
            "ITERATION_CODE|ITERATION_NAME|LEVEL|ALONE|DETAILED_ONLY|COARSE_ONLY\n"
            f"0002|European alone|1|{alone}|{detailed_only}|{coarse_only}\n"
            "0003|Albanian alone|2|True|False|False"
        )
        ethnicity_iterations = SIMPLE_ETHNICITY_ITERATIONS
        iteration_to_test = "0002"
    else:
        assert iteration_type == "ETHNICITY"
        race_iterations = SIMPLE_RACE_ITERATIONS
        ethnicity_iterations = (
            "ITERATION_CODE|ITERATION_NAME|LEVEL|DETAILED_ONLY|COARSE_ONLY\n"
            f"3010|Central American|1|{detailed_only}|{coarse_only}\n"
            "3011|Costa Rican|2|False|False"
        )
        iteration_to_test = "3010"

    write_input_files(
        input_dir=input_dir,
        households=households,
        t1_output=(
            "REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT\n"
            f"{region_id_to_test}|{region_type_to_test}|{iteration_to_test}|{t1_count}"
        ),
        ethnicity_iterations=ethnicity_iterations,
        race_iterations=race_iterations,
        race_eth_codes=SIMPLE_RACE_CODES,
        iteration_map=f"ITERATION_CODE|RACE_ETH_CODE\n{iteration_to_test}|1000",
        grfc=grfc,
        config=SIMPLE_CONFIG,
    )

    run_plan_h_analytics(
        parameters_path=os.path.join(input_dir, "config"),
        data_path=input_dir,
        output_path=output_dir,
        overwrite_config={
            "thresholds_h_t3": THRESHOLDS_T3_10_20_30,
            "thresholds_h_t4": THRESHOLDS_T4_20,
            "run_us": not run_pr,
            "run_pr": run_pr,
            **INFINITE_BUDGETS,
        },
    )

    for subdir, output_file in [
        ("t3", "T03001"),
        ("t3", "T03002"),
        ("t3", "T03003"),
        ("t3", "T03004"),
        ("t4", "T04001"),
        ("t4", "T04002"),
    ]:
        output_df = multi_read_csv(
            os.path.join(output_dir, subdir, output_file), dtype=str, sep="|"
        )

        if output_file not in [t3_output_file, t4_output_file]:
            assert len(output_df) == 0
            continue

        output_for_target_iteration = output_df[
            output_df["ITERATION_CODE"] == iteration_to_test
        ]
        assert (output_for_target_iteration["COUNT"].astype(int) == 1).any(), (
            f"All counts for iteration {iteration_to_test} in "
            f"{output_file} should be 1, but got: \n{output_df}"
        )


# pylint: disable=line-too-long
# Below are inputs to a test with different household counts by iteration code for detailed only and coarse only.
# There are different counts by state to trigger the adaptivity logic.

COMPLEX_HOUSEHOLDS_FILE = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|HHRACE|QRACE1|QRACE2|QRACE3|QRACE4|QRACE5|QRACE6|QRACE7|QRACE8|QSPAN|HOUSEHOLD_TYPE|TEN
01|001|000001|0001|01|1000|Null|Null|Null|Null|Null|Null|Null|1000|1|1
01|001|000001|0001|01|1340|Null|Null|Null|Null|Null|Null|Null|2002|1|1
01|001|000001|0001|45|1035|Null|Null|Null|Null|Null|Null|Null|2100|1|1
01|001|000001|0001|01|5011|Null|Null|Null|Null|Null|Null|Null|2000|1|1
02|105|000200|1000|01|1000|Null|Null|Null|Null|Null|Null|Null|1000|1|1
02|105|000200|1000|01|1000|Null|Null|Null|Null|Null|Null|Null|1000|1|1
02|105|000200|1000|01|1340|Null|Null|Null|Null|Null|Null|Null|2002|1|1
02|105|000200|1000|01|1340|Null|Null|Null|Null|Null|Null|Null|2002|1|1
02|105|000200|1000|45|1035|Null|Null|Null|Null|Null|Null|Null|2100|1|1
02|105|000200|1000|45|1035|Null|Null|Null|Null|Null|Null|Null|2100|1|1
02|105|000200|1000|01|5011|Null|Null|Null|Null|Null|Null|Null|2000|1|1
02|105|000200|1000|01|5011|Null|Null|Null|Null|Null|Null|Null|2000|1|1
04|007|940200|1002|01|1000|Null|Null|Null|Null|Null|Null|Null|1000|1|1
04|007|940200|1002|01|1000|Null|Null|Null|Null|Null|Null|Null|1000|1|1
04|007|940200|1002|01|1000|Null|Null|Null|Null|Null|Null|Null|1000|1|1
04|007|940200|1002|01|1340|Null|Null|Null|Null|Null|Null|Null|2002|1|1
04|007|940200|1002|01|1340|Null|Null|Null|Null|Null|Null|Null|2002|1|1
04|007|940200|1002|01|1340|Null|Null|Null|Null|Null|Null|Null|2002|1|1
04|007|940200|1002|45|1035|Null|Null|Null|Null|Null|Null|Null|2100|1|1
04|007|940200|1002|45|1035|Null|Null|Null|Null|Null|Null|Null|2100|1|1
04|007|940200|1002|45|1035|Null|Null|Null|Null|Null|Null|Null|2100|1|1
04|007|940200|1002|01|5011|Null|Null|Null|Null|Null|Null|Null|2000|1|1
04|007|940200|1002|01|5011|Null|Null|Null|Null|Null|Null|Null|2000|1|1
04|007|940200|1002|01|5011|Null|Null|Null|Null|Null|Null|Null|2000|1|1
06|027|000500|2088|01|1000|Null|Null|Null|Null|Null|Null|Null|1000|1|1
06|027|000500|2088|01|1340|Null|Null|Null|Null|Null|Null|Null|2002|1|1
06|027|000500|2088|45|1035|Null|Null|Null|Null|Null|Null|Null|2100|1|1
06|027|000500|2088|01|5011|Null|Null|Null|Null|Null|Null|Null|2000|1|1
"""

COMPLEX_GRFC = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|PLACEFP|AIANNHCE
01|001|000001|0001|22800|0001
02|105|000200|1000|40510|6310
04|007|940200|1002|11370|1140
06|027|000500|2088|06616|0250
"""

COMPLEX_ETHNICITY_ITERATIONS = """ITERATION_CODE|ITERATION_NAME|LEVEL|DETAILED_ONLY|COARSE_ONLY
4017|Costa Rican|2|False|False
4046|Other Hispanic or Latino, not specified (All geos)|2|False|True
4047|Hispanic|2|True|False
"""

# White Alone won't have any tabulations because the level is 0.
COMPLEX_RACE_ITERATIONS = """ITERATION_CODE|ITERATION_NAME|LEVEL|ALONE|DETAILED_ONLY|COARSE_ONLY
1001|White alone|0|True|Null|Null
1002|European alone|1|True|False|False
1008|Azerbaijani alone|2|True|True|False
2832|Other Alaska Native alone or in any combinatio...|2|False|False|True
"""

COMPLEX_RACE_CODES = """RACE_ETH_CODE|RACE_ETH_NAME
1000|White alone
1340|Slovak
1035|Azerbaijani
5011|Alaska Native
2100|Costa Rican
2000|Other Hispanic or Latino, not specified (All geos)
2002|Hispanic
"""

COMPLEX_CODE_ITERATION_MAP = """ITERATION_CODE|RACE_ETH_CODE
1001|1000
1002|1340
1008|1035
2832|5011
4017|2100
4046|2000
4047|2002
"""

COMPLEX_T1_INPUT = """REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
1|USA|1002|61
1|USA|1008|61
06|STATE|1002|1
06|STATE|1008|1
01|STATE|1002|10
01|STATE|1008|10
02|STATE|1002|20
02|STATE|1008|20
04|STATE|1002|30
04|STATE|1008|30
06027|COUNTY|1002|1
06027|COUNTY|2832|1
01001|COUNTY|1002|10
01001|COUNTY|2832|10
02105|COUNTY|2832|20
02105|COUNTY|1002|20
04007|COUNTY|2832|30
04007|COUNTY|1002|30
06027000500|TRACT|2832|1
06027000500|TRACT|1002|1
01001000001|TRACT|2832|10
01001000001|TRACT|1002|10
02105000200|TRACT|1002|20
02105000200|TRACT|2832|20
04007940200|TRACT|2832|30
04007940200|TRACT|1002|30
0606616|PLACE|1002|1
0606616|PLACE|2832|1
0122800|PLACE|1002|10
0122800|PLACE|2832|10
0240510|PLACE|1002|20
0240510|PLACE|2832|20
0411370|PLACE|1002|30
0411370|PLACE|2832|30
0250|AIANNH|1002|1
0250|AIANNH|2832|1
0001|AIANNH|1002|10
0001|AIANNH|2832|10
6310|AIANNH|1002|20
6310|AIANNH|2832|20
1140|AIANNH|1002|30
1140|AIANNH|2832|30
1|USA|4017|61
1|USA|4047|61
06|STATE|4017|1
06|STATE|4047|1
01|STATE|4017|10
01|STATE|4047|10
02|STATE|4017|20
02|STATE|4047|20
04|STATE|4017|30
04|STATE|4047|30
06027|COUNTY|4017|1
06027|COUNTY|4046|1
01001|COUNTY|4017|10
01001|COUNTY|4046|10
02105|COUNTY|4017|20
02105|COUNTY|4046|20
04007|COUNTY|4017|30
04007|COUNTY|4046|30
06027000500|TRACT|4017|1
06027000500|TRACT|4046|1
01001000001|TRACT|4017|10
01001000001|TRACT|4046|10
02105000200|TRACT|4017|20
02105000200|TRACT|4046|20
04007940200|TRACT|4017|30
04007940200|TRACT|4046|30
0606616|PLACE|4017|1
0606616|PLACE|4046|1
0122800|PLACE|4017|10
0122800|PLACE|4046|10
0240510|PLACE|4017|20
0240510|PLACE|4046|20
0411370|PLACE|4017|30
0411370|PLACE|4046|30
0250|AIANNH|4017|1
0250|AIANNH|4046|1
0001|AIANNH|4017|10
0001|AIANNH|4046|10
6310|AIANNH|4017|20
6310|AIANNH|4046|20
1140|AIANNH|4017|30
1140|AIANNH|4046|30
"""

OUTPUT_3001 = """REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
06|STATE|1002|0|1
06|STATE|1008|0|1
06|STATE|4017|0|1
06|STATE|4047|0|1
06027|COUNTY|2832|0|1
06027|COUNTY|1002|0|1
06027|COUNTY|4046|0|1
06027|COUNTY|4017|0|1
06027000500|TRACT|2832|0|1
06027000500|TRACT|1002|0|1
06027000500|TRACT|4017|0|1
06027000500|TRACT|4046|0|1
0606616|PLACE|1002|0|1
0606616|PLACE|2832|0|1
0606616|PLACE|4017|0|1
0606616|PLACE|4046|0|1
0250|AIANNH|1002|0|1
0250|AIANNH|2832|0|1
0250|AIANNH|4017|0|1
0250|AIANNH|4046|0|1
"""

OUTPUT_3002 = """REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
01|STATE|1002|0|1
01|STATE|1008|0|1
01|STATE|1002|1|1
01|STATE|1008|1|1
01001|COUNTY|2832|0|1
01001|COUNTY|1002|0|1
01001|COUNTY|2832|1|1
01001|COUNTY|1002|1|1
01001000001|TRACT|2832|0|1
01001000001|TRACT|1002|0|1
01001000001|TRACT|2832|1|1
01001000001|TRACT|1002|1|1
0122800|PLACE|2832|0|1
0122800|PLACE|1002|0|1
0122800|PLACE|2832|1|1
0122800|PLACE|1002|1|1
0001|AIANNH|2832|0|1
0001|AIANNH|1002|0|1
0001|AIANNH|2832|1|1
0001|AIANNH|1002|1|1
01|STATE|4017|0|1
01|STATE|4047|0|1
01|STATE|4017|1|1
01|STATE|4047|1|1
01001|COUNTY|4046|0|1
01001|COUNTY|4017|0|1
01001|COUNTY|4046|1|1
01001|COUNTY|4017|1|1
01001000001|TRACT|4046|0|1
01001000001|TRACT|4017|0|1
01001000001|TRACT|4046|1|1
01001000001|TRACT|4017|1|1
0122800|PLACE|4046|0|1
0122800|PLACE|4017|0|1
0122800|PLACE|4046|1|1
0122800|PLACE|4017|1|1
0001|AIANNH|4046|0|1
0001|AIANNH|4017|0|1
0001|AIANNH|4046|1|1
0001|AIANNH|4017|1|1
01|STATE|1002|6|0
01|STATE|1008|6|0
01001|COUNTY|2832|6|0
01001|COUNTY|1002|6|0
01001000001|TRACT|2832|6|0
01001000001|TRACT|1002|6|0
0122800|PLACE|2832|6|0
0122800|PLACE|1002|6|0
0001|AIANNH|2832|6|0
0001|AIANNH|1002|6|0
01|STATE|4017|6|0
01|STATE|4047|6|0
01001|COUNTY|4046|6|0
01001|COUNTY|4017|6|0
01001000001|TRACT|4046|6|0
01001000001|TRACT|4017|6|0
0122800|PLACE|4046|6|0
0122800|PLACE|4017|6|0
0001|AIANNH|4046|6|0
0001|AIANNH|4017|6|0
"""

OUTPUT_3003 = """REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
02|STATE|1002|0|2
02|STATE|1008|0|2
02|STATE|1002|1|2
02|STATE|1008|1|2
02|STATE|1002|2|2
02|STATE|1008|2|2
02|STATE|1002|6|0
02|STATE|1008|6|0
02105|COUNTY|2832|0|2
02105|COUNTY|1002|0|2
02105|COUNTY|2832|1|2
02105|COUNTY|1002|1|2
02105|COUNTY|2832|2|2
02105|COUNTY|1002|2|2
02105000200|TRACT|2832|0|2
02105000200|TRACT|1002|0|2
02105000200|TRACT|2832|1|2
02105000200|TRACT|1002|1|2
02105000200|TRACT|2832|2|2
02105000200|TRACT|1002|2|2
0240510|PLACE|2832|0|2
0240510|PLACE|1002|0|2
0240510|PLACE|2832|1|2
0240510|PLACE|1002|1|2
0240510|PLACE|2832|2|2
0240510|PLACE|1002|2|2
6310|AIANNH|2832|0|2
6310|AIANNH|1002|0|2
6310|AIANNH|2832|1|2
6310|AIANNH|1002|1|2
6310|AIANNH|2832|2|2
6310|AIANNH|1002|2|2
02|STATE|4017|0|2
02|STATE|4047|0|2
02|STATE|4017|1|2
02|STATE|4047|1|2
02|STATE|4017|2|2
02|STATE|4047|2|2
02105|COUNTY|4046|0|2
02105|COUNTY|4017|0|2
02105|COUNTY|4046|1|2
02105|COUNTY|4017|1|2
02105|COUNTY|4046|2|2
02105|COUNTY|4017|2|2
02105000200|TRACT|4046|0|2
02105000200|TRACT|4017|0|2
02105000200|TRACT|4046|1|2
02105000200|TRACT|4017|1|2
02105000200|TRACT|4046|2|2
02105000200|TRACT|4017|2|2
0240510|PLACE|4046|0|2
0240510|PLACE|4017|0|2
0240510|PLACE|4046|1|2
0240510|PLACE|4017|1|2
0240510|PLACE|4046|2|2
0240510|PLACE|4017|2|2
6310|AIANNH|4046|0|2
6310|AIANNH|4017|0|2
6310|AIANNH|4046|1|2
6310|AIANNH|4017|1|2
6310|AIANNH|4046|2|2
6310|AIANNH|4017|2|2
02|STATE|1002|3|0
02|STATE|1008|3|0
02105|COUNTY|2832|3|0
02105|COUNTY|1002|3|0
02105000200|TRACT|2832|3|0
02105000200|TRACT|1002|3|0
0240510|PLACE|2832|3|0
0240510|PLACE|1002|3|0
6310|AIANNH|2832|3|0
6310|AIANNH|1002|3|0
02|STATE|4017|3|0
02|STATE|4047|3|0
02105|COUNTY|4046|3|0
02105|COUNTY|4017|3|0
02105000200|TRACT|4046|3|0
02105000200|TRACT|4017|3|0
0240510|PLACE|4046|3|0
0240510|PLACE|4017|3|0
6310|AIANNH|4046|3|0
6310|AIANNH|4017|3|0
02|STATE|1002|7|0
02|STATE|1008|7|0
02105|COUNTY|2832|6|0
02105|COUNTY|1002|6|0
02105000200|TRACT|2832|6|0
02105000200|TRACT|1002|6|0
0240510|PLACE|2832|6|0
0240510|PLACE|1002|6|0
6310|AIANNH|2832|6|0
6310|AIANNH|1002|6|0
02|STATE|4017|6|0
02|STATE|4047|6|0
02105|COUNTY|4046|6|0
02105|COUNTY|4017|6|0
02105000200|TRACT|4046|6|0
02105000200|TRACT|4017|6|0
0240510|PLACE|4046|6|0
0240510|PLACE|4017|6|0
6310|AIANNH|4046|6|0
6310|AIANNH|4017|6|0
02105|COUNTY|2832|7|0
02105|COUNTY|1002|7|0
02105000200|TRACT|2832|7|0
02105000200|TRACT|1002|7|0
0240510|PLACE|2832|7|0
0240510|PLACE|1002|7|0
6310|AIANNH|2832|7|0
6310|AIANNH|1002|7|0
02|STATE|4017|7|0
02|STATE|4047|7|0
02105|COUNTY|4046|7|0
02105|COUNTY|4017|7|0
02105000200|TRACT|4046|7|0
02105000200|TRACT|4017|7|0
0240510|PLACE|4046|7|0
0240510|PLACE|4017|7|0
6310|AIANNH|4046|7|0
6310|AIANNH|4017|7|0
02|STATE|1002|8|0
02|STATE|1008|8|0
02105|COUNTY|2832|8|0
02105|COUNTY|1002|8|0
02105000200|TRACT|2832|8|0
02105000200|TRACT|1002|8|0
0240510|PLACE|2832|8|0
0240510|PLACE|1002|8|0
6310|AIANNH|2832|8|0
6310|AIANNH|1002|8|0
02|STATE|4017|8|0
02|STATE|4047|8|0
02105|COUNTY|4046|8|0
02105|COUNTY|4017|8|0
02105000200|TRACT|4046|8|0
02105000200|TRACT|4017|8|0
0240510|PLACE|4046|8|0
0240510|PLACE|4017|8|0
6310|AIANNH|4046|8|0
6310|AIANNH|4017|8|0
"""

OUTPUT_3004 = """REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
1|USA|1002|0|7
1|USA|1008|0|7
1|USA|1002|1|7
1|USA|1008|1|7
1|USA|1002|2|7
1|USA|1008|2|7
04|STATE|1002|0|3
04|STATE|1008|0|3
04|STATE|1002|1|3
04|STATE|1008|1|3
04|STATE|1002|2|3
04|STATE|1008|2|3
04007|COUNTY|2832|0|3
04007|COUNTY|1002|0|3
04007|COUNTY|2832|1|3
04007|COUNTY|1002|1|3
04007|COUNTY|2832|2|3
04007|COUNTY|1002|2|3
04007940200|TRACT|2832|0|3
04007940200|TRACT|1002|0|3
04007940200|TRACT|2832|1|3
04007940200|TRACT|1002|1|3
04007940200|TRACT|2832|2|3
04007940200|TRACT|1002|2|3
0411370|PLACE|2832|0|3
0411370|PLACE|1002|0|3
0411370|PLACE|2832|1|3
0411370|PLACE|1002|1|3
0411370|PLACE|2832|2|3
0411370|PLACE|1002|2|3
1140|AIANNH|2832|0|3
1140|AIANNH|1002|0|3
1140|AIANNH|2832|1|3
1140|AIANNH|1002|1|3
1140|AIANNH|2832|2|3
1140|AIANNH|1002|2|3
1|USA|4017|0|7
1|USA|4047|0|7
1|USA|4017|1|7
1|USA|4047|1|7
1|USA|4017|2|7
1|USA|4047|2|7
04|STATE|4017|0|3
04|STATE|4047|0|3
04|STATE|4017|1|3
04|STATE|4047|1|3
04|STATE|4017|2|3
04|STATE|4047|2|3
04007|COUNTY|4046|0|3
04007|COUNTY|4017|0|3
04007|COUNTY|4046|1|3
04007|COUNTY|4017|1|3
04007|COUNTY|4046|2|3
04007|COUNTY|4017|2|3
04007940200|TRACT|4046|0|3
04007940200|TRACT|4017|0|3
04007940200|TRACT|4046|1|3
04007940200|TRACT|4017|1|3
04007940200|TRACT|4046|2|3
04007940200|TRACT|4017|2|3
0411370|PLACE|4046|0|3
0411370|PLACE|4017|0|3
0411370|PLACE|4046|1|3
0411370|PLACE|4017|1|3
0411370|PLACE|4046|2|3
0411370|PLACE|4017|2|3
1140|AIANNH|4046|0|3
1140|AIANNH|4017|0|3
1140|AIANNH|4046|1|3
1140|AIANNH|4017|1|3
1140|AIANNH|4046|2|3
1140|AIANNH|4017|2|3
1|USA|1002|3|0
1|USA|1008|3|0
04|STATE|1002|3|0
04|STATE|1008|3|0
04007|COUNTY|2832|3|0
04007|COUNTY|1002|3|0
04007940200|TRACT|2832|3|0
04007940200|TRACT|1002|3|0
0411370|PLACE|2832|3|0
0411370|PLACE|1002|3|0
1140|AIANNH|2832|3|0
1140|AIANNH|1002|3|0
1|USA|4017|3|0
1|USA|4047|3|0
04|STATE|4017|3|0
04|STATE|4047|3|0
04007|COUNTY|4046|3|0
04007|COUNTY|4017|3|0
04007940200|TRACT|4046|3|0
04007940200|TRACT|4017|3|0
0411370|PLACE|4046|3|0
0411370|PLACE|4017|3|0
1140|AIANNH|4046|3|0
1140|AIANNH|4017|3|0
1|USA|1002|4|0
1|USA|1008|4|0
04|STATE|1002|4|0
04|STATE|1008|4|0
04007|COUNTY|2832|4|0
04007|COUNTY|1002|4|0
04007940200|TRACT|2832|4|0
04007940200|TRACT|1002|4|0
0411370|PLACE|2832|4|0
0411370|PLACE|1002|4|0
1140|AIANNH|2832|4|0
1140|AIANNH|1002|4|0
1|USA|4017|4|0
1|USA|4047|4|0
04|STATE|4017|4|0
04|STATE|4047|4|0
04007|COUNTY|4046|4|0
04007|COUNTY|4017|4|0
04007940200|TRACT|4046|4|0
04007940200|TRACT|4017|4|0
0411370|PLACE|4046|4|0
0411370|PLACE|4017|4|0
1140|AIANNH|4046|4|0
1140|AIANNH|4017|4|0
1|USA|1002|5|0
1|USA|1008|5|0
04|STATE|1002|5|0
04|STATE|1008|5|0
04007|COUNTY|2832|5|0
04007|COUNTY|1002|5|0
04007940200|TRACT|2832|5|0
04007940200|TRACT|1002|5|0
0411370|PLACE|2832|5|0
0411370|PLACE|1002|5|0
1140|AIANNH|2832|5|0
1140|AIANNH|1002|5|0
1|USA|4017|5|0
1|USA|4047|5|0
04|STATE|4017|5|0
04|STATE|4047|5|0
04007|COUNTY|4046|5|0
04007|COUNTY|4017|5|0
04007940200|TRACT|4046|5|0
04007940200|TRACT|4017|5|0
0411370|PLACE|4046|5|0
0411370|PLACE|4017|5|0
1140|AIANNH|4046|5|0
1140|AIANNH|4017|5|0
1|USA|1002|6|0
1|USA|1008|6|0
04|STATE|1002|6|0
04|STATE|1008|6|0
04007|COUNTY|2832|6|0
04007|COUNTY|1002|6|0
04007940200|TRACT|2832|6|0
04007940200|TRACT|1002|6|0
0411370|PLACE|2832|6|0
0411370|PLACE|1002|6|0
1140|AIANNH|2832|6|0
1140|AIANNH|1002|6|0
1|USA|4017|6|0
1|USA|4047|6|0
04|STATE|4017|6|0
04|STATE|4047|6|0
04007|COUNTY|4046|6|0
04007|COUNTY|4017|6|0
04007940200|TRACT|4046|6|0
04007940200|TRACT|4017|6|0
0411370|PLACE|4046|6|0
0411370|PLACE|4017|6|0
1140|AIANNH|4046|6|0
1140|AIANNH|4017|6|0
1|USA|1002|7|0
1|USA|1008|7|0
04|STATE|1002|7|0
04|STATE|1008|7|0
04007|COUNTY|2832|7|0
04007|COUNTY|1002|7|0
04007940200|TRACT|2832|7|0
04007940200|TRACT|1002|7|0
0411370|PLACE|2832|7|0
0411370|PLACE|1002|7|0
1140|AIANNH|2832|7|0
1140|AIANNH|1002|7|0
1|USA|4017|7|0
1|USA|4047|7|0
04|STATE|4017|7|0
04|STATE|4047|7|0
04007|COUNTY|4046|7|0
04007|COUNTY|4017|7|0
04007940200|TRACT|4046|7|0
04007940200|TRACT|4017|7|0
0411370|PLACE|4046|7|0
0411370|PLACE|4017|7|0
1140|AIANNH|4046|7|0
1140|AIANNH|4017|7|0
1|USA|1002|8|0
1|USA|1008|8|0
04|STATE|1002|8|0
04|STATE|1008|8|0
04007|COUNTY|2832|8|0
04007|COUNTY|1002|8|0
04007940200|TRACT|2832|8|0
04007940200|TRACT|1002|8|0
0411370|PLACE|2832|8|0
0411370|PLACE|1002|8|0
1140|AIANNH|2832|8|0
1140|AIANNH|1002|8|0
1|USA|4017|8|0
1|USA|4047|8|0
04|STATE|4017|8|0
04|STATE|4047|8|0
04007|COUNTY|4046|8|0
04007|COUNTY|4017|8|0
04007940200|TRACT|4046|8|0
04007940200|TRACT|4017|8|0
0411370|PLACE|4046|8|0
0411370|PLACE|4017|8|0
1140|AIANNH|4046|8|0
1140|AIANNH|4017|8|0
"""

OUTPUT_4001 = """REGION_ID|REGION_TYPE|ITERATION_CODE|T4_DATA_CELL|COUNT
06|STATE|1002|0|1
06|STATE|1008|0|1
06027|COUNTY|2832|0|1
06027|COUNTY|1002|0|1
06027000500|TRACT|2832|0|1
06027000500|TRACT|1002|0|1
0606616|PLACE|2832|0|1
0606616|PLACE|1002|0|1
0250|AIANNH|2832|0|1
0250|AIANNH|1002|0|1
01|STATE|1002|0|1
01|STATE|1008|0|1
01001|COUNTY|2832|0|1
01001|COUNTY|1002|0|1
01001000001|TRACT|2832|0|1
01001000001|TRACT|1002|0|1
0122800|PLACE|2832|0|1
0122800|PLACE|1002|0|1
0001|AIANNH|2832|0|1
0001|AIANNH|1002|0|1
06|STATE|4017|0|1
06|STATE|4047|0|1
06027|COUNTY|4046|0|1
06027|COUNTY|4017|0|1
06027000500|TRACT|4046|0|1
06027000500|TRACT|4017|0|1
0606616|PLACE|4046|0|1
0606616|PLACE|4017|0|1
0250|AIANNH|4046|0|1
0250|AIANNH|4017|0|1
01|STATE|4017|0|1
01|STATE|4047|0|1
01001|COUNTY|4046|0|1
01001|COUNTY|4017|0|1
01001000001|TRACT|4046|0|1
01001000001|TRACT|4017|0|1
0122800|PLACE|4046|0|1
0122800|PLACE|4017|0|1
0001|AIANNH|4046|0|1
0001|AIANNH|4017|0|1
"""

OUTPUT_4002 = """REGION_ID|REGION_TYPE|ITERATION_CODE|T4_DATA_CELL|COUNT
1|USA|1002|0|7
1|USA|1008|0|7
1|USA|1002|1|7
1|USA|1008|1|7
1|USA|4017|0|7
1|USA|4047|0|7
1|USA|4017|1|7
1|USA|4047|1|7
1|USA|1002|2|0
1|USA|1008|2|0
1|USA|4017|2|0
1|USA|4047|2|0
1|USA|1002|3|0
1|USA|1008|3|0
1|USA|4017|3|0
1|USA|4047|3|0
02|STATE|1002|0|2
02|STATE|1008|0|2
02|STATE|1002|1|2
02|STATE|1008|1|2
04|STATE|1002|0|3
04|STATE|1008|0|3
04|STATE|1002|1|3
04|STATE|1008|1|3
02105|COUNTY|2832|0|2
02105|COUNTY|1002|0|2
02105|COUNTY|2832|1|2
02105|COUNTY|1002|1|2
04007|COUNTY|2832|0|3
04007|COUNTY|1002|0|3
04007|COUNTY|2832|1|3
04007|COUNTY|1002|1|3
02105000200|TRACT|2832|0|2
02105000200|TRACT|1002|0|2
02105000200|TRACT|2832|1|2
02105000200|TRACT|1002|1|2
04007940200|TRACT|2832|0|3
04007940200|TRACT|1002|0|3
04007940200|TRACT|2832|1|3
04007940200|TRACT|1002|1|3
0240510|PLACE|2832|0|2
0240510|PLACE|1002|0|2
0240510|PLACE|2832|1|2
0240510|PLACE|1002|1|2
0411370|PLACE|2832|0|3
0411370|PLACE|1002|0|3
0411370|PLACE|2832|1|3
0411370|PLACE|1002|1|3
6310|AIANNH|2832|0|2
6310|AIANNH|1002|0|2
6310|AIANNH|2832|1|2
6310|AIANNH|1002|1|2
1140|AIANNH|2832|0|3
1140|AIANNH|1002|0|3
1140|AIANNH|2832|1|3
1140|AIANNH|1002|1|3
02|STATE|4017|0|2
02|STATE|4047|0|2
02|STATE|4017|1|2
02|STATE|4047|1|2
04|STATE|4017|0|3
04|STATE|4047|0|3
04|STATE|4017|1|3
04|STATE|4047|1|3
02105|COUNTY|4046|0|2
02105|COUNTY|4017|0|2
02105|COUNTY|4046|1|2
02105|COUNTY|4017|1|2
04007|COUNTY|4046|0|3
04007|COUNTY|4017|0|3
04007|COUNTY|4046|1|3
04007|COUNTY|4017|1|3
02105000200|TRACT|4046|0|2
02105000200|TRACT|4017|0|2
02105000200|TRACT|4046|1|2
02105000200|TRACT|4017|1|2
04007940200|TRACT|4046|0|3
04007940200|TRACT|4017|0|3
04007940200|TRACT|4046|1|3
04007940200|TRACT|4017|1|3
0240510|PLACE|4046|0|2
0240510|PLACE|4017|0|2
0240510|PLACE|4046|1|2
0240510|PLACE|4017|1|2
0411370|PLACE|4046|0|3
0411370|PLACE|4017|0|3
0411370|PLACE|4046|1|3
0411370|PLACE|4017|1|3
6310|AIANNH|4046|0|2
6310|AIANNH|4017|0|2
6310|AIANNH|4046|1|2
6310|AIANNH|4017|1|2
1140|AIANNH|4046|0|3
1140|AIANNH|4017|0|3
1140|AIANNH|4046|1|3
1140|AIANNH|4017|1|3
02|STATE|1002|2|0
02|STATE|1008|2|0
04|STATE|1002|2|0
04|STATE|1008|2|0
02105|COUNTY|2832|2|0
02105|COUNTY|1002|2|0
04007|COUNTY|2832|2|0
04007|COUNTY|1002|2|0
02105000200|TRACT|2832|2|0
02105000200|TRACT|1002|2|0
04007940200|TRACT|2832|2|0
04007940200|TRACT|1002|2|0
0240510|PLACE|2832|2|0
0240510|PLACE|1002|2|0
0411370|PLACE|2832|2|0
0411370|PLACE|1002|2|0
6310|AIANNH|2832|2|0
6310|AIANNH|1002|2|0
1140|AIANNH|2832|2|0
1140|AIANNH|1002|2|0
02|STATE|4017|2|0
02|STATE|4047|2|0
04|STATE|4017|2|0
04|STATE|4047|2|0
02105|COUNTY|4046|2|0
02105|COUNTY|4017|2|0
04007|COUNTY|4046|2|0
04007|COUNTY|4017|2|0
02105000200|TRACT|4046|2|0
02105000200|TRACT|4017|2|0
04007940200|TRACT|4046|2|0
04007940200|TRACT|4017|2|0
0240510|PLACE|4046|2|0
0240510|PLACE|4017|2|0
0411370|PLACE|4046|2|0
0411370|PLACE|4017|2|0
6310|AIANNH|4046|2|0
6310|AIANNH|4017|2|0
1140|AIANNH|4046|2|0
1140|AIANNH|4017|2|0
02|STATE|1002|3|0
02|STATE|1008|3|0
04|STATE|1002|3|0
04|STATE|1008|3|0
02105|COUNTY|2832|3|0
02105|COUNTY|1002|3|0
04007|COUNTY|2832|3|0
04007|COUNTY|1002|3|0
02105000200|TRACT|2832|3|0
02105000200|TRACT|1002|3|0
04007940200|TRACT|2832|3|0
04007940200|TRACT|1002|3|0
0240510|PLACE|2832|3|0
0240510|PLACE|1002|3|0
0411370|PLACE|2832|3|0
0411370|PLACE|1002|3|0
6310|AIANNH|2832|3|0
6310|AIANNH|1002|3|0
1140|AIANNH|2832|3|0
1140|AIANNH|1002|3|0
02|STATE|4017|3|0
02|STATE|4047|3|0
04|STATE|4017|3|0
04|STATE|4047|3|0
02105|COUNTY|4046|3|0
02105|COUNTY|4017|3|0
04007|COUNTY|4046|3|0
04007|COUNTY|4017|3|0
02105000200|TRACT|4046|3|0
02105000200|TRACT|4017|3|0
04007940200|TRACT|4046|3|0
04007940200|TRACT|4017|3|0
0240510|PLACE|4046|3|0
0240510|PLACE|4017|3|0
0411370|PLACE|4046|3|0
0411370|PLACE|4017|3|0
6310|AIANNH|4046|3|0
6310|AIANNH|4017|3|0
1140|AIANNH|4046|3|0
1140|AIANNH|4017|3|0
"""

# pylint: enable=line-too-long


@pytest.mark.usefixtures("spark")
def test_detailed_coarse_iterations_are_correct(input_dir, output_dir):
    """Test that we correctly count detail-only and coarse only iterations."""
    write_input_files(
        input_dir=input_dir,
        households=COMPLEX_HOUSEHOLDS_FILE,
        t1_output=COMPLEX_T1_INPUT,
        ethnicity_iterations=COMPLEX_ETHNICITY_ITERATIONS,
        race_iterations=COMPLEX_RACE_ITERATIONS,
        race_eth_codes=COMPLEX_RACE_CODES,
        iteration_map=COMPLEX_CODE_ITERATION_MAP,
        grfc=COMPLEX_GRFC,
        config=SIMPLE_CONFIG,
    )

    overwrite_config = {
        "thresholds_h_t3": THRESHOLDS_T3_10_20_30,
        "thresholds_h_t4": THRESHOLDS_T4_20,
        "state_filter_us": ["01", "02", "04", "06"],
        **INFINITE_BUDGETS,
    }

    run_plan_h_analytics(
        parameters_path=os.path.join(input_dir, "config"),
        data_path=input_dir,
        output_path=output_dir,
        overwrite_config=overwrite_config,
    )

    t03001 = multi_read_csv(
        os.path.join(output_dir, "t3", "T03001"), dtype=str, sep="|"
    )
    t03002 = multi_read_csv(
        os.path.join(output_dir, "t3", "T03002"), dtype=str, sep="|"
    )
    t03003 = multi_read_csv(
        os.path.join(output_dir, "t3", "T03003"), dtype=str, sep="|"
    )
    t03004 = multi_read_csv(
        os.path.join(output_dir, "t3", "T03004"), dtype=str, sep="|"
    )
    t04001 = multi_read_csv(
        os.path.join(output_dir, "t4", "T04001"), dtype=str, sep="|"
    )
    t04002 = multi_read_csv(
        os.path.join(output_dir, "t4", "T04002"), dtype=str, sep="|"
    )

    assert_frame_equal_with_sort(
        t03001, pd.read_csv(StringIO(OUTPUT_3001), sep="|", dtype=str)
    )

    assert_frame_equal_with_sort(
        t03002, pd.read_csv(StringIO(OUTPUT_3002), sep="|", dtype=str)
    )

    assert_frame_equal_with_sort(
        t03003, pd.read_csv(StringIO(OUTPUT_3003), sep="|", dtype=str)
    )

    assert_frame_equal_with_sort(
        t03004, pd.read_csv(StringIO(OUTPUT_3004), sep="|", dtype=str)
    )

    assert_frame_equal_with_sort(
        t04001, pd.read_csv(StringIO(OUTPUT_4001), sep="|", dtype=str)
    )

    assert_frame_equal_with_sort(
        t04002, pd.read_csv(StringIO(OUTPUT_4002), sep="|", dtype=str)
    )


@dict_parametrize(
    {
        "t1_with_coarse_at_USA_level": {
            "households": SIMPLE_HOUSEHOLDS_FILE,
            "t1_output": dedent(
                """
                REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
                1|USA|0002|5"""
            ),
            "race_iterations": dedent(
                """
                ITERATION_CODE|ITERATION_NAME|LEVEL|ALONE|DETAILED_ONLY|COARSE_ONLY
                0002|European alone|1|True|False|True
                0003|Albanian alone|2|True|False|False"""
            ),
            "iterations_map": dedent(
                """
                ITERATION_CODE|RACE_ETH_CODE
                0002|1000"""
            ),
            "grfc": SIMPLE_GRFC,
            "run_pr": False,
        },
        "t1_with_coarse_at_PR-STATE_level": {
            "households": SIMPLE_PR_HOUSEHOLDS_FILE,
            "t1_output": dedent(
                """
                REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
                72|PR-STATE|0002|5"""
            ),
            "race_iterations": dedent(
                """
                ITERATION_CODE|ITERATION_NAME|LEVEL|ALONE|DETAILED_ONLY|COARSE_ONLY
                0002|European alone|1|True|False|True
                0003|Albanian alone|2|True|False|False"""
            ),
            "iterations_map": dedent(
                """
                ITERATION_CODE|RACE_ETH_CODE
                0002|1000"""
            ),
            "grfc": SIMPLE_PR_GRFC,
            "run_pr": True,
        },
        "t1_with_detailed_at_COUNTY_level": {
            "households": SIMPLE_HOUSEHOLDS_FILE,
            "t1_output": dedent(
                """
                REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
                01001|COUNTY|0002|5"""
            ),
            "race_iterations": dedent(
                """
                ITERATION_CODE|ITERATION_NAME|LEVEL|ALONE|DETAILED_ONLY|COARSE_ONLY
                0002|European alone|1|True|True|False
                0003|Albanian alone|2|True|False|False"""
            ),
            "iterations_map": dedent(
                """
                ITERATION_CODE|RACE_ETH_CODE
                0002|1000"""
            ),
            "grfc": SIMPLE_GRFC,
            "run_pr": False,
        },
        "t1_with_detailed_at_PR-COUNTY_level": {
            "households": SIMPLE_PR_HOUSEHOLDS_FILE,
            "t1_output": dedent(
                """
                REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
                72001|PR-COUNTY|0002|5"""
            ),
            "race_iterations": dedent(
                """
                ITERATION_CODE|ITERATION_NAME|LEVEL|ALONE|DETAILED_ONLY|COARSE_ONLY
                0002|European alone|1|True|True|False
                0003|Albanian alone|2|True|False|False"""
            ),
            "iterations_map": dedent(
                """
                ITERATION_CODE|RACE_ETH_CODE
                0002|1000"""
            ),
            "grfc": SIMPLE_PR_GRFC,
            "run_pr": True,
        },
    }
)
@pytest.mark.usefixtures("spark")
def test_invalid_popgroups_pruned(
    input_dir,
    output_dir,
    households,
    t1_output,
    race_iterations,
    iterations_map,
    grfc,
    run_pr,
):
    """Tests that invalid pop groups in the T1 file are not tabulated."""
    write_input_files(
        input_dir=input_dir,
        households=households,
        t1_output=t1_output,
        ethnicity_iterations=SIMPLE_ETHNICITY_ITERATIONS,
        race_iterations=race_iterations,
        race_eth_codes=SIMPLE_RACE_CODES,
        iteration_map=iterations_map,
        grfc=grfc,
        config=SIMPLE_CONFIG,
    )

    run_plan_h_analytics(
        parameters_path=os.path.join(input_dir, "config"),
        data_path=input_dir,
        output_path=output_dir,
        overwrite_config={"run_us": not run_pr, "run_pr": run_pr, **INFINITE_BUDGETS},
    )

    for subdir, output in [
        ("t3", "T03001"),
        ("t3", "T03002"),
        ("t3", "T03003"),
        ("t3", "T03004"),
        ("t4", "T04001"),
        ("t4", "T04002"),
    ]:
        output_df = multi_read_csv(
            os.path.join(output_dir, subdir, output), dtype=str, sep="|"
        )
        assert (
            len(output_df) == 0
        ), f"Expected {output} to be empty, but got:\n {output_df}"


# A GRFC with US and PR values.
GRFC_WITH_INTERSECT_REGIONS = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|PLACEFP|AIANNHCE
01|001|000001|0001|22800|0001
01|002|000001|0001|22800|0001
02|001|000001|0001|22800|0001
72|001|000001|0001|22800|0001
"""

# pylint:disable=line-too-long
SIMPLE_US_AND_PR_HOUSEHOLDS_FILE = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|HHRACE|QRACE1|QRACE2|QRACE3|QRACE4|QRACE5|QRACE6|QRACE7|QRACE8|QSPAN|HOUSEHOLD_TYPE|TEN
01|001|000001|0001|01|1000|Null|Null|Null|Null|Null|Null|Null|1000|1|4
72|001|000001|0001|01|1000|Null|Null|Null|Null|Null|Null|Null|1000|1|4
"""
# pylint:enable=line-too-long


@dict_parametrize(
    {
        "US_t1_all_households": {
            "households": SIMPLE_US_AND_PR_HOUSEHOLDS_FILE,
            "pop_group_totals": dedent(
                """
                REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
                1|USA|0002|5"""
            ),
            "run_us": True,
            "run_pr": True,
            "error_message": (
                "No households in the selected population groups. Check to make sure"
                " you're using the correct pop_group_totals file for your US/PR"
                " settings."
            ),
        },
        "PR_t1_all_households": {
            "households": SIMPLE_US_AND_PR_HOUSEHOLDS_FILE,
            "pop_group_totals": dedent(
                """
                REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
                72|PR-STATE|0002|5"""
            ),
            "run_us": True,
            "run_pr": True,
            "error_message": (
                "No T1 pop groups data for the US run. Check to make sure you're using"
                " the correct pop_group_totals file for your US/PR settings."
            ),
        },
        "different_states": {
            "households": SIMPLE_HOUSEHOLDS_FILE,
            "pop_group_totals": dedent(
                """
                REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
                02|STATE|0002|5"""
            ),
            "run_us": True,
            "run_pr": False,
            "error_message": (
                "No households in the selected population groups. Check to make sure"
                " you're using the correct pop_group_totals file for your US/PR"
                " settings."
            ),
        },
        "different_counties": {
            "households": SIMPLE_HOUSEHOLDS_FILE,
            "pop_group_totals": dedent(
                """
                REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
                01002|COUNTY|0002|5"""
            ),
            "run_us": True,
            "run_pr": False,
            "error_message": (
                "No households in the selected population groups. Check to make sure"
                " you're using the correct pop_group_totals file for your US/PR"
                " settings."
            ),
        },
        "different_iterations": {
            "households": SIMPLE_HOUSEHOLDS_FILE,
            "pop_group_totals": dedent(
                """
                REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT
                01|STATE|3010|5"""
            ),
            "run_us": True,
            "run_pr": False,
            "error_message": (
                "No households in the selected population groups. Check to make sure"
                " you're using the correct pop_group_totals file for your US/PR"
                " settings."
            ),
        },
    }
)
@pytest.mark.usefixtures("spark")
def test_validates_t1_units_overlap(
    input_dir,
    output_dir,
    households,
    pop_group_totals,
    run_us,
    run_pr,
    error_message,
    caplog,
):
    """Checks for an error if there are no t1 pop groups in the households data."""
    write_input_files(
        input_dir=input_dir,
        households=households,
        t1_output=pop_group_totals,
        ethnicity_iterations=SIMPLE_ETHNICITY_ITERATIONS,
        race_iterations=SIMPLE_RACE_ITERATIONS,
        race_eth_codes=SIMPLE_RACE_CODES,
        iteration_map=SIMPLE_CODE_ITERATION_MAP,
        grfc=GRFC_WITH_INTERSECT_REGIONS,
        config=SIMPLE_CONFIG,
    )

    with pytest.raises(RuntimeError) as excinfo:
        run_plan_h_analytics(
            parameters_path=os.path.join(input_dir, "config"),
            data_path=input_dir,
            output_path=output_dir,
            overwrite_config={
                "run_us": run_us,
                "run_pr": run_pr,
                "state_filter_us": ["01", "02"],
            },
        )

    assert "Input validation failed." in str(excinfo.value)

    assert error_message in caplog.text
