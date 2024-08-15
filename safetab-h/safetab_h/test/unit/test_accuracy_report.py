"""Unit tests for :mod:`tmlt.safetab_h.accuracy_report`."""

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
import tempfile
import unittest
from textwrap import dedent

import pandas as pd
import pytest

from tmlt.common.io_helpers import to_csv_with_create_dir
from tmlt.common.pyspark_test_tools import assert_frame_equal_with_sort
from tmlt.safetab_h.accuracy_report import (
    create_aggregated_error_report_h,
    create_error_report_h,
)


@pytest.mark.usefixtures("spark")
class TestSingleRunAccuracyReports(unittest.TestCase):
    """TestCase for single run accuracy reports for SafeTab-H."""

    def setUp(self):
        """Create shared input files."""
        # pylint:disable=consider-using-with
        self.parameters_dir = tempfile.TemporaryDirectory()
        self.noisy_dir = tempfile.TemporaryDirectory()
        self.target_dir = tempfile.TemporaryDirectory()
        self.output_dir = tempfile.TemporaryDirectory()
        # pylint:enable=consider-using-with

        pd.DataFrame(
            [["1", "False", "False"]],
            columns=["ITERATION_CODE", "DETAILED_ONLY", "COARSE_ONLY"],
        ).to_csv(
            os.path.join(
                self.parameters_dir.name, "race-characteristic-iterations.txt"
            ),
            sep="|",
            index=False,
        )

        pd.DataFrame(
            [["2", "True", "False"]],
            columns=["ITERATION_CODE", "DETAILED_ONLY", "COARSE_ONLY"],
        ).to_csv(
            os.path.join(
                self.parameters_dir.name, "ethnicity-characteristic-iterations.txt"
            ),
            sep="|",
            index=False,
        )

    def test_create_error_report_h(self):
        """create_error_report_h creates the expected output."""

        # REGION_ID, REGION_TYPE, ITERATION_CODE
        population_groups = [["01", "A", "1"], ["01", "A", "2"], ["02", "B", "1"]]

        household_type_values = ["1", "2", "3", "4", "5"]
        household_tenure_values = ["0", "1", "2", "3", "4"]

        t3_df = pd.DataFrame(
            [
                population_group + [household_type]
                for population_group in population_groups
                for household_type in household_type_values
            ],
            columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "T3_DATA_CELL"],
        )
        # Save noisy counts
        t3_df["COUNT"] = [-1, 3, -2, 1, 3, 3, 1, -2, 0, 3, 3, 1, 3, 3, 3]

        # The create_error_report function takes in the T03001, T03002, ect. dirs.
        # This function needs to write to these locations.
        to_csv_with_create_dir(
            t3_df,
            os.path.join(self.noisy_dir.name, "t3/T03001", "t3.csv"),
            sep="|",
            index=False,
        )
        # Save target counts
        t3_df["COUNT"] = [0, 2, 0, 2, 2, 1, 2, 0, 1, 0, 1, 2, 2, 0, 1]
        to_csv_with_create_dir(
            t3_df,
            os.path.join(self.target_dir.name, "t3/T03001", "t3.csv"),
            sep="|",
            index=False,
        )

        t4_df = pd.DataFrame(
            [
                population_group + [household_tenure]
                for population_group in population_groups
                for household_tenure in household_tenure_values
            ],
            columns=["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "T4_DATA_CELL"],
        )
        # Save noisy counts
        t4_df["COUNT"] = [-1, -4, -4, -3, 0, -4, -1, 2, 1, 3, 3, -2, -2, 3, 0]
        to_csv_with_create_dir(
            t4_df,
            os.path.join(self.noisy_dir.name, "t4/T04001", "t4.csv"),
            sep="|",
            index=False,
        )
        # Save target counts
        t4_df["COUNT"] = [0, 0, 0, 3, 0, 2, 2, 2, 3, 0, 2, 3, 1, 2, 2]
        to_csv_with_create_dir(
            t4_df,
            os.path.join(self.target_dir.name, "t4/T04001", "t4.csv"),
            sep="|",
            index=False,
        )

        expected_df = pd.DataFrame(
            [
                ["01", "A", "1", "False", "False", "6", "6", "8", "15", "69"],
                ["01", "A", "2", "True", "False", "4", "9", "19", "14", "58"],
                ["02", "B", "1", "False", "False", "6", "9", "19", "12", "40"],
            ],
            columns=[
                "REGION_ID",
                "REGION_TYPE",
                "ITERATION_CODE",
                "DETAILED_ONLY",
                "COARSE_ONLY",
                "Population group size",  # Uses total count from T3
                "Total absolute error in T3",
                "Total squared error in T3",
                "Total absolute error in T4",
                "Total squared error in T4",
            ],
        )

        create_error_report_h(
            noisy_path=self.noisy_dir.name,
            ground_truth_path=self.target_dir.name,
            parameters_path=self.parameters_dir.name,
            output_path=self.output_dir.name,
            outputs="1",
        )

        actual_df = pd.read_csv(
            os.path.join(self.output_dir.name, "error_report.csv"), dtype=str
        )

        pd.testing.assert_frame_equal(
            expected_df.set_index(["REGION_ID", "REGION_TYPE", "ITERATION_CODE"]),
            actual_df.set_index(["REGION_ID", "REGION_TYPE", "ITERATION_CODE"]),
        )


@pytest.mark.usefixtures("spark")
class TestMultiRunSafeTabHAccuracyReport(unittest.TestCase):
    """TestCase for SafeTab-H multi-run error report."""

    def setUp(self):
        """Create temporary directories and fill with inputs for subsequent tests."""
        # pylint: disable=consider-using-with
        self.parameters_dir = tempfile.TemporaryDirectory()
        self.noisy_run_dir_1 = tempfile.TemporaryDirectory()
        self.noisy_run_dir_2 = tempfile.TemporaryDirectory()
        self.ground_truth_dir = tempfile.TemporaryDirectory()
        self.output_dir = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with

        # Set up iterations files.
        with open(
            os.path.join(
                self.parameters_dir.name, "race-characteristic-iterations.txt"
            ),
            "w",
        ) as f:
            f.write(
                dedent(
                    """
                    ITERATION_CODE|DETAILED_ONLY|COARSE_ONLY|LEVEL
                    1|False|False|1
                    2|False|False|1
                    """
                )
            )
        with open(
            os.path.join(
                self.parameters_dir.name, "ethnicity-characteristic-iterations.txt"
            ),
            "w",
        ) as f:
            f.write(
                dedent(
                    """
                    ITERATION_CODE|DETAILED_ONLY|COARSE_ONLY|LEVEL
                    3|True|False|2
                    """
                )
            )

        # Set up config.
        with open(os.path.join(self.parameters_dir.name, "config.json"), "w") as f:
            f.write(
                dedent(
                    """{
                    "privacy_budget_h_t3_level_1_a": 0.008,
                    "privacy_budget_h_t3_level_2_a": 0.534,
                    "privacy_budget_h_t3_level_1_b": 0.534,
                    "privacy_budget_h_t3_level_2_b": 2.43,
                    "privacy_budget_h_t3_level_1_c": 0.534,
                    "privacy_budget_h_t3_level_2_c": 0.534,
                    "privacy_budget_h_t4_level_1_a": 0.008,
                    "privacy_budget_h_t4_level_2_a": 0.534,
                    "privacy_budget_h_t4_level_1_b": 0.534,
                    "privacy_budget_h_t4_level_2_b": 2.43,
                    "privacy_budget_h_t4_level_1_c": 0.534,
                    "privacy_budget_h_t4_level_2_c": 0.534,
                    "privacy_defn": "zcdp"
                    }
                    """
                )
            )

        # Set up ground truth.
        os.makedirs(os.path.join(self.ground_truth_dir.name, "t3", "T03001"))
        with open(
            os.path.join(self.ground_truth_dir.name, "t3", "T03001", "t31.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|4
                    02|A|2|0|1
                    03|B|3|0|0
                    """
                )
            )
        os.makedirs(os.path.join(self.ground_truth_dir.name, "t3", "T03002"))
        with open(
            os.path.join(self.ground_truth_dir.name, "t3", "T03002", "t3.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|8
                    01|A|1|1|2
                    01|A|1|6|6
                    """
                )
            )
        os.makedirs(os.path.join(self.ground_truth_dir.name, "t3", "T03003"))
        with open(
            os.path.join(self.ground_truth_dir.name, "t3", "T03003", "t3.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|9
                    01|A|1|1|3
                    01|A|1|2|3
                    01|A|1|3|0
                    01|A|1|6|6
                    01|A|1|7|5
                    01|A|1|8|1
                    """
                )
            )
        os.makedirs(os.path.join(self.ground_truth_dir.name, "t3", "T03004"))
        with open(
            os.path.join(self.ground_truth_dir.name, "t3", "T03004", "t3.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|20
                    01|A|1|1|12
                    01|A|1|2|2
                    01|A|1|3|10
                    01|A|1|4|9
                    01|A|1|5|1
                    01|A|1|6|8
                    01|A|1|7|6
                    01|A|1|8|2
                    """
                )
            )
        os.makedirs(os.path.join(self.ground_truth_dir.name, "t4", "T04001"))
        with open(
            os.path.join(self.ground_truth_dir.name, "t4", "T04001", "t4.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T4_DATA_CELL|COUNT
                    01|A|1|0|5
                    """
                )
            )
        os.makedirs(os.path.join(self.ground_truth_dir.name, "t4", "T04002"))
        with open(
            os.path.join(self.ground_truth_dir.name, "t4", "T04002", "t4.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T4_DATA_CELL|COUNT
                    01|A|1|0|2
                    01|A|1|1|2
                    01|A|1|2|0
                    01|A|1|3|0
                    """
                )
            )

        # Set up noisy counts.
        os.makedirs(os.path.join(self.noisy_run_dir_1.name, "t3", "T03001"))
        with open(
            os.path.join(self.noisy_run_dir_1.name, "t3", "T03001", "t31.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|2
                    02|A|2|0|0
                    03|B|3|0|1
                    """
                )
            )
        os.makedirs(os.path.join(self.noisy_run_dir_1.name, "t3", "T03002"))
        with open(
            os.path.join(self.noisy_run_dir_1.name, "t3", "T03002", "t3.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|5
                    01|A|1|1|2
                    01|A|1|6|3
                    """
                )
            )
        os.makedirs(os.path.join(self.noisy_run_dir_1.name, "t3", "T03003"))
        with open(
            os.path.join(self.noisy_run_dir_1.name, "t3", "T03003", "t3.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|9
                    01|A|1|1|3
                    01|A|1|2|2
                    01|A|1|3|1
                    01|A|1|6|6
                    01|A|1|7|3
                    01|A|1|8|3
                    """
                )
            )
        os.makedirs(os.path.join(self.noisy_run_dir_1.name, "t3", "T03004"))
        with open(
            os.path.join(self.noisy_run_dir_1.name, "t3", "T03004", "t3.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|20
                    01|A|1|1|12
                    01|A|1|2|2
                    01|A|1|3|10
                    01|A|1|4|5
                    01|A|1|5|5
                    01|A|1|6|8
                    01|A|1|7|5
                    01|A|1|8|3
                    """
                )
            )
        os.makedirs(os.path.join(self.noisy_run_dir_1.name, "t4", "T04001"))
        with open(
            os.path.join(self.noisy_run_dir_1.name, "t4", "T04001", "t4.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T4_DATA_CELL|COUNT
                    01|A|1|0|3
                    """
                )
            )
        os.makedirs(os.path.join(self.noisy_run_dir_1.name, "t4", "T04002"))
        with open(
            os.path.join(self.noisy_run_dir_1.name, "t4", "T04002", "t4.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T4_DATA_CELL|COUNT
                    01|A|1|0|3
                    01|A|1|1|1
                    01|A|1|2|1
                    01|A|1|3|1
                    """
                )
            )

        # Set up noisy counts 2.
        os.makedirs(os.path.join(self.noisy_run_dir_2.name, "t3", "T03001"))
        with open(
            os.path.join(self.noisy_run_dir_2.name, "t3", "T03001", "t31.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|1
                    02|A|2|0|8
                    03|B|3|0|5
                    """
                )
            )
        os.makedirs(os.path.join(self.noisy_run_dir_2.name, "t3", "T03002"))
        with open(
            os.path.join(self.noisy_run_dir_2.name, "t3", "T03002", "t3.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|2
                    01|A|1|1|1
                    01|A|1|6|1
                    """
                )
            )
        os.makedirs(os.path.join(self.noisy_run_dir_2.name, "t3", "T03003"))
        with open(
            os.path.join(self.noisy_run_dir_2.name, "t3", "T03003", "t3.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|7
                    01|A|1|1|2
                    01|A|1|2|1
                    01|A|1|3|1
                    01|A|1|6|5
                    01|A|1|7|3
                    01|A|1|8|2
                    """
                )
            )
        os.makedirs(os.path.join(self.noisy_run_dir_2.name, "t3", "T03004"))
        with open(
            os.path.join(self.noisy_run_dir_2.name, "t3", "T03004", "t3.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T3_DATA_CELL|COUNT
                    01|A|1|0|16
                    01|A|1|1|6
                    01|A|1|2|1
                    01|A|1|3|5
                    01|A|1|4|3
                    01|A|1|5|2
                    01|A|1|6|10
                    01|A|1|7|8
                    01|A|1|8|2
                    """
                )
            )
        os.makedirs(os.path.join(self.noisy_run_dir_2.name, "t4", "T04001"))
        with open(
            os.path.join(self.noisy_run_dir_2.name, "t4", "T04001", "t4.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T4_DATA_CELL|COUNT
                    01|A|1|0|9
                    """
                )
            )
        os.makedirs(os.path.join(self.noisy_run_dir_2.name, "t4", "T04002"))
        with open(
            os.path.join(self.noisy_run_dir_2.name, "t4", "T04002", "t4.csv"), "w"
        ) as f:
            f.write(
                dedent(
                    """
                    REGION_ID|REGION_TYPE|ITERATION_CODE|T4_DATA_CELL|COUNT
                    01|A|1|0|7
                    01|A|1|1|3
                    01|A|1|2|4
                    01|A|1|3|0
                    """
                )
            )

    def test_aggregate_error_report(self):
        """create_aggregated_error_report_h creates the expected output."""

        expected_df = pd.DataFrame(
            [
                ["t3", "A", "1", "[0.0, 1.0]", "0", "1", "46.485481604475176", "6.70"],
                [
                    "t3",
                    "A",
                    "1",
                    "(10.0, 100.0]",
                    "0",
                    "1",
                    "46.485481604475176",
                    "2.95",
                ],
                [
                    "t3",
                    "A",
                    "1",
                    "(10.0, 100.0]",
                    "1",
                    "2",
                    "46.485481604475176",
                    "4.70",
                ],
                [
                    "t3",
                    "A",
                    "1",
                    "(10.0, 100.0]",
                    "2",
                    "4",
                    "46.485481604475176",
                    "2.00",
                ],
                [
                    "t3",
                    "A",
                    "1",
                    "(10.0, 100.0]",
                    "3",
                    "5",
                    "46.485481604475176",
                    "5.10",
                ],
                ["t3", "B", "2", "[0.0, 1.0]", "0", "1", "2.667222164363905", "4.80"],
                ["t4", "A", "1", "(1.0, 10.0]", "0", "1", "46.485481604475176", "3.90"],
                ["t4", "A", "1", "(1.0, 10.0]", "1", "3", "46.485481604475176", "3.25"],
            ],
            columns=[
                "Table",
                "REGION_TYPE",
                "ITERATION_LEVEL",
                "Population group size",
                "STAT_LEVEL",
                "Number of pop groups",
                "Expected MOE",
                "MOE",
            ],
        )

        create_aggregated_error_report_h(
            single_run_paths=[self.noisy_run_dir_1.name, self.noisy_run_dir_2.name],
            parameters_path=self.parameters_dir.name,
            ground_truth_path=self.ground_truth_dir.name,
            output_path=self.output_dir.name,
        )
        actual_df = pd.read_csv(
            os.path.join(self.output_dir.name, "multi_run_error_report.csv"), dtype=str
        )
        assert_frame_equal_with_sort(actual_df, expected_df)
