"""Fixtures to be shared across all safetab-h tests."""

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
import shutil
from typing import Any, Dict

import pytest
from pyspark.sql import SparkSession


def quiet_py4j():
    """Remove noise in the logs irrelevant to testing."""
    print("Calling PySparkTest:suppress_py4j_logging")
    logger = logging.getLogger("py4j")
    # This is to silence py4j.java_gateway: DEBUG logs.
    logger.setLevel(logging.ERROR)


# this initializes one shared spark session for the duration of the test session.
# another option may be to set the scope to "module", which changes the duration to
# one session per module
@pytest.fixture(scope="session", name="spark")
def pyspark():
    """Setup a context to execute pyspark tests."""
    quiet_py4j()
    print("Setting up spark session.")
    # pylint: disable=no-member
    spark = (
        SparkSession.builder.appName(__name__)
        .master("local[4]")
        .config("spark.sql.warehouse.dir", "/tmp/hive_tables")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.eventLog.enabled", "false")
        .config("spark.driver.allowMultipleContexts", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.default.parallelism", "5")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "2g")
        .getOrCreate()
    )
    # pylint: enable=no-member

    # This is to silence pyspark logs.
    spark.sparkContext.setLogLevel("OFF")
    yield spark
    shutil.rmtree("/tmp/hive_tables", ignore_errors=True)
    spark.stop()


def dict_parametrize(data: Dict[str, Dict[str, Any]], **kwargs):
    """Parameterize a test function based on a dictionary.

    E.g.:
    @dict_parametrize(
        {
            "these keys become the ids that identify each run": {
                "x": 1,
                "y": 2,
                "expected_value": 3,
            },
            "some_edge_case": {
                "x": 0,
                "y": 0,
                "expected_value": 0,
            },
        }
    )
    def test_func(x, y, expected_value):
    """
    first_parameter_set = list(data.values())[0]
    parameters = list(first_parameter_set.keys())
    formatted_data = [
        [parameter_set[parameter] for parameter in parameters]
        for parameter_set in data.values()
    ]
    ids = list(data.keys())
    return pytest.mark.parametrize(parameters, formatted_data, ids=ids, **kwargs)
