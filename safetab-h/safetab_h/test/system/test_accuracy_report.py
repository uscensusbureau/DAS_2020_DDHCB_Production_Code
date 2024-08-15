"""System tests for :mod:`tmlt.safetab_h.accuracy_report`."""

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
import shutil
import tempfile
import unittest

import pytest
from parameterized import parameterized

from tmlt.safetab_h.accuracy_report import run_full_error_report_h
from tmlt.safetab_h.paths import RESOURCES_DIR


@pytest.mark.usefixtures("spark")
class TestRunFullErrorReport(unittest.TestCase):
    """Run the full error report."""

    def setUp(self):
        """Create temporary directories."""
        # pylint: disable=consider-using-with
        self.input_dir = tempfile.TemporaryDirectory()
        self.data_path = os.path.join(self.input_dir.name, "dataset")
        self.parameters_path = os.path.join(self.data_path, "input_dir_puredp")
        self.config_dir = tempfile.TemporaryDirectory()
        self.config_path = self.config_dir.name
        self.output_dir = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with
        shutil.copytree(os.path.join(RESOURCES_DIR, "toy_dataset"), self.data_path)

    @parameterized.expand(["US", "PR"])
    @pytest.mark.slow
    def test_safetab_h_multi_run_error_report(self, us_or_puerto_rico: str):
        """Run full error report on SafeTab-H."""
        temp_output_dir = os.path.join(self.output_dir.name, "safetab-h")
        run_full_error_report_h(
            parameters_path=self.parameters_path,
            data_path=self.data_path,
            output_path=temp_output_dir,
            config_path=self.config_path,
            trials=5,
            us_or_puerto_rico=us_or_puerto_rico,
        )
        expected_subdir = ["full_error_report", "ground_truth", "single_runs"]
        actual_subdir = next(os.walk(temp_output_dir))[1]
        assert expected_subdir == sorted(actual_subdir)
