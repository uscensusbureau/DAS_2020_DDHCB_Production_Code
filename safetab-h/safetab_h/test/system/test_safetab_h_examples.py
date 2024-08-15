"""Tests for :mod:`examples`."""

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

import logging
import os
import unittest
from typing import Callable

import pytest
from parameterized import parameterized

from tmlt.safetab_h.paths import RESOURCES_DIR
from tmlt.safetab_h.target_counts_h import main as target_counts_h_main


@pytest.mark.usefixtures("spark")
class TestExampleScripts(unittest.TestCase):
    """Smoke tests for the target counts command line interface."""

    @parameterized.expand(
        [
            (
                target_counts_h_main,
                [
                    "--input",
                    os.path.join(RESOURCES_DIR, "toy_dataset", "input_dir_puredp"),
                    "--reader",
                    os.path.join(RESOURCES_DIR, "toy_dataset"),
                    "--output",
                    "example_output/target_counts_h",
                ],
            )
        ]
    )
    @pytest.mark.slow
    def test_examples(self, main_func: Callable[[list], None], args: list):
        """Tests the examples by calling main function.

        Args:
            main_func: The main function for the module that is run.
            args: List of arguments passed to ArgumentParser.
        """
        logging.getLogger().handlers = []
        main_func(args)
