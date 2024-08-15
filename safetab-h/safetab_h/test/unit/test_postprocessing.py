"""Unit tests for :mod:`tmlt.safetab_h.postprocessing`."""

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

import unittest
from io import StringIO
from textwrap import dedent
from typing import List

import pandas as pd
import pytest
from parameterized import parameterized
from pyspark.sql import SparkSession

from tmlt.common.pyspark_test_tools import assert_frame_equal_with_sort
from tmlt.safetab_h.postprocessing import (
    add_marginals,
    t3_postprocessing,
    t4_postprocessing,
)


def text_csv_to_pandas(text_csv: str):
    """Helper function to convert a text csv file to a pandas dataframe.

    Converts the "COUNT" column to integer type.
    """
    to_return = pd.read_csv(StringIO(text_csv))
    return to_return


@pytest.mark.usefixtures("spark")
class TestPostprocessing(unittest.TestCase):
    """TestCase for postprocessing functions."""

    @parameterized.expand(
        [
            (
                "t3 level 0",
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                        01,       STATE,      0002,          0,            9
                        """
                    )
                ),
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                        01,       STATE,      0002,          0,           9
                        """
                    )
                ),
            ),
            (
                "t3 level 1",
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                        01,       STATE,      0002,          1,           5
                        01,       STATE,      0002,          6,           4
                        """
                    )
                ),
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                        01,       STATE,      0002,          0,           9
                        01,       STATE,      0002,          1,           5
                        01,       STATE,      0002,          6,           4
                        """
                    )
                ),
            ),
            (
                "t3 level 2",
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                        01,       STATE,      0002,          2,           3
                        01,       STATE,      0002,          3,           2
                        01,       STATE,      0002,          7,           3
                        01,       STATE,      0002,          8,           1
                        """
                    )
                ),
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                        01,       STATE,      0002,          0,           9
                        01,       STATE,      0002,          1,           5
                        01,       STATE,      0002,          6,           4
                        01,       STATE,      0002,          2,           3
                        01,       STATE,      0002,          3,           2
                        01,       STATE,      0002,          7,           3
                        01,       STATE,      0002,          8,           1
                        """
                    )
                ),
            ),
            (
                "t3 level 3",
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                        01,       STATE,      0002,          2,           3
                        01,       STATE,      0002,          4,           1
                        01,       STATE,      0002,          5,           1
                        01,       STATE,      0002,          7,           3
                        01,       STATE,      0002,          8,           1
                        """
                    )
                ),
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                        01,       STATE,      0002,          0,           9
                        01,       STATE,      0002,          1,           5
                        01,       STATE,      0002,          6,           4
                        01,       STATE,      0002,          3,           2
                        01,       STATE,      0002,          2,           3
                        01,       STATE,      0002,          4,           1
                        01,       STATE,      0002,          5,           1
                        01,       STATE,      0002,          7,           3
                        01,       STATE,      0002,          8,           1
                        """
                    )
                ),
            ),
            (
                "t4 level 0",
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T4_DATA_CELL,COUNT
                        01,       STATE,      0002,          0,           9
                        """
                    )
                ),
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T4_DATA_CELL,COUNT
                        01,       STATE,      0002,          0,           9
                        """
                    )
                ),
            ),
            (
                "t4 level 1",
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T4_DATA_CELL,COUNT
                        01,       STATE,      0002,          1,           2
                        01,       STATE,      0002,          2,           3
                        01,       STATE,      0002,          3,           4
                        """
                    )
                ),
                text_csv_to_pandas(
                    dedent(
                        """
                        REGION_ID,REGION_TYPE,ITERATION_CODE,T4_DATA_CELL,COUNT
                        01,       STATE,      0002,          0,           9
                        01,       STATE,      0002,          1,           2
                        01,       STATE,      0002,          2,           3
                        01,       STATE,      0002,          3,           4
                        """
                    )
                ),
            ),
        ]
    )
    def test_add_marginals(self, _, input_data: pd.DataFrame, expected: pd.DataFrame):
        """add_marginals adds the correct rows to the output."""
        spark = SparkSession.builder.getOrCreate()
        input_sdf = spark.createDataFrame(input_data)
        output_sdf = add_marginals(input_sdf)
        assert_frame_equal_with_sort(output_sdf.toPandas(), expected)

    # pylint: disable=line-too-long
    @parameterized.expand(
        [
            (
                "all levels represented",
                [
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,STAT_LEVEL,T3_DATA_CELL,COUNT
                            01,       STATE,      0002,          0,         0,           1
                            01,       STATE,      0002,          1,         1,           2
                            """
                        )
                    ),
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,STAT_LEVEL,T3_DATA_CELL,COUNT
                            01,       STATE,      0002,          2,         2,           3
                            01,       STATE,      0002,          3,         3,           4
                            """
                        )
                    ),
                ],
                [
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                            01,       STATE,      0002,          0,           1
                            """
                        )
                    ),
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                            01,       STATE,      0002,          1,           2
                            """
                        )
                    ),
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                            01,       STATE,      0002,          2,           3
                            """
                        )
                    ),
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                            01,       STATE,      0002,          3,           4
                            """
                        )
                    ),
                ],
            ),
            (
                "level 2 missing",
                [
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,STAT_LEVEL,T3_DATA_CELL,COUNT
                            01,       STATE,      0002,          0,         0,           1
                            01,       STATE,      0002,          1,         1,           2
                            """
                        )
                    ),
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,STAT_LEVEL,T3_DATA_CELL,COUNT
                            01,       STATE,      0002,          3,         3,           4
                            """
                        )
                    ),
                ],
                [
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                            01,       STATE,      0002,          0,           1
                            """
                        )
                    ),
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                            01,       STATE,      0002,          1,           2
                            """
                        )
                    ),
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                            """
                        )
                    ),
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T3_DATA_CELL,COUNT
                            01,       STATE,      0002,          3,           4
                            """
                        )
                    ),
                ],
            ),
        ]
    )
    # pylint: enable=line-too-long
    def test_t3_postprocessing(
        self, _, input_data: List[pd.DataFrame], expected: List[pd.DataFrame]
    ):
        """Tests that the T3 output is correctly divided into 4 files per adaptivity
        level."""
        spark = SparkSession.builder.getOrCreate()
        input_sdfs = [spark.createDataFrame(pdf) for pdf in input_data]
        output_sdfs = t3_postprocessing(input_sdfs)

        assert len(output_sdfs) == len(expected)

        for output_sdf, expected_pdf in zip(output_sdfs, expected):
            assert_frame_equal_with_sort(output_sdf.toPandas(), expected_pdf)

    @parameterized.expand(
        [
            (
                "all levels represented",
                [
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,STAT_LEVEL,T4_DATA_CELL,COUNT
                            01,       STATE,      0002,          0,         0,           1
                            """
                        )
                    ),
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,STAT_LEVEL,T4_DATA_CELL,COUNT
                            01,       STATE,      0002,          1,         1,           2
                            """
                        )
                    ),
                ],
                [
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T4_DATA_CELL,COUNT
                            01,       STATE,      0002,          0,           1
                            """
                        )
                    ),
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T4_DATA_CELL,COUNT
                            01,       STATE,      0002,          1,           2
                            """
                        )
                    ),
                ],
            ),
            (
                "level 2 missing",
                [
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,STAT_LEVEL,T4_DATA_CELL,COUNT
                            01,       STATE,      0002,          1,         1,           2
                            """
                        )
                    )
                ],
                [
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T4_DATA_CELL,COUNT
                            """
                        )
                    ),
                    text_csv_to_pandas(
                        dedent(
                            """
                            REGION_ID,REGION_TYPE,ITERATION_CODE,T4_DATA_CELL,COUNT
                            01,       STATE,      0002,          1,           2
                            """
                        )
                    ),
                ],
            ),
        ]
    )
    def test_t4_postprocessing(
        self, _, input_data: List[pd.DataFrame], expected: List[pd.DataFrame]
    ):
        """Tests that the T4 output is correctly divided into 2 files per adaptivity
        level."""
        spark = SparkSession.builder.getOrCreate()
        input_sdfs = [spark.createDataFrame(pdf) for pdf in input_data]
        output_sdfs = t4_postprocessing(input_sdfs)

        assert len(output_sdfs) == len(expected)

        for output_sdf, expected_pdf in zip(output_sdfs, expected):
            assert_frame_equal_with_sort(output_sdf.toPandas(), expected_pdf)
