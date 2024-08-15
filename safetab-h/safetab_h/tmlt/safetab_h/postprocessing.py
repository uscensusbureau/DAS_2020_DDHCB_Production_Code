"""Functions for postprocessing output files."""

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

import functools
from typing import List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (  # pylint: disable=no-name-in-module
    array,
    col,
    explode,
    lit,
    when,
)
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

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

T3_COLUMNS = {
    "REGION_ID": StringType(),
    "REGION_TYPE": StringType(),
    "ITERATION_CODE": StringType(),
    "T3_DATA_CELL": StringType(),
    "COUNT": IntegerType(),
}
T3_SCHEMA = StructType(
    [StructField(column, dtype) for column, dtype in T3_COLUMNS.items()]
)
"""Expected columns with their types, in order, for t3."""

T4_COLUMNS = {
    "REGION_ID": StringType(),
    "REGION_TYPE": StringType(),
    "ITERATION_CODE": StringType(),
    "T4_DATA_CELL": StringType(),
    "COUNT": IntegerType(),
}
T4_SCHEMA = StructType(
    [StructField(column, dtype) for column, dtype in T4_COLUMNS.items()]
)
"""Expected columns, in order, for t4."""


def t3_postprocessing(t3_results: List[DataFrame]) -> List[DataFrame]:
    """Returns four postprocessed t3 dfs matching the format of t3.

    Each dataframe contains all the rows calculated at a given stat level, and drops
    the stat level.

    Expected input format is a list of data frames with the following columns

    * REGION_ID: In the correct format
    * REGION_TYPE: In the correct format
    * ITERATION_CODE: In the correct format
    * STAT_LEVEL: An integer in [0, 1, 2, 3].
    * T3_DATA_CELL: In the correct format
    * COUNT: In the correct format

    Args:
        t3_results: List of data frames in expected input format.
    """
    if len(t3_results) == 0:
        spark = SparkSession.builder.getOrCreate()
        return [spark.createDataFrame([], schema=T3_SCHEMA)] * 4
    t3_sdf = functools.reduce(DataFrame.union, t3_results)
    t3_sdf = t3_sdf.withColumn("COUNT", col("COUNT").cast(IntegerType()))
    return [
        t3_sdf.filter(col("STAT_LEVEL") == stat_level).select(list(T3_COLUMNS.keys()))
        for stat_level in [0, 1, 2, 3]
    ]


def t4_postprocessing(t4_results: List[DataFrame]) -> List[DataFrame]:
    """Returns two postprocessed t4 dfs matching the format of t4.

    Each dataframe contains all the rows calculated at a given stat level.

    Expected input format is a list of data frames with the following columns

    * REGION_ID: In the correct format
    * REGION_TYPE: In the correct format
    * ITERATION_CODE: In the correct format
    * STAT_LEVEL: An integer in [0, 1].
    * T4_DATA_CELL: In the correct format
    * COUNT: In the correct format

    Args:
        t4_results: List of data frames in expected input format.
    """
    if len(t4_results) == 0:
        spark = SparkSession.builder.getOrCreate()
        return [spark.createDataFrame([], schema=T4_SCHEMA)] * 2
    t4_sdf = functools.reduce(DataFrame.union, t4_results)
    t4_sdf = t4_sdf.withColumn("COUNT", col("COUNT").cast(IntegerType()))
    return [
        t4_sdf.filter(col("STAT_LEVEL") == stat_level).select(list(T4_COLUMNS.keys()))
        for stat_level in [0, 1]
    ]


def _add_t3_marginals_column(results_sdf: DataFrame):
    """Adds a MARGINALS column to the dataframe.

    The MARGINALS columns will contain a list of all marginal rows (including the
    row itself) that should be calculated from the row in question.
    """
    return results_sdf.withColumn(
        "MARGINALS",
        when(col("T3_DATA_CELL") == T3_TOTAL, array(lit(T3_TOTAL)))
        .when(
            col("T3_DATA_CELL") == T3_FAMILY_HOUSEHOLDS,
            array([lit(i) for i in [T3_TOTAL, T3_FAMILY_HOUSEHOLDS]]),
        )
        .when(
            col("T3_DATA_CELL") == T3_MARRIED_COUPLE_FAMILY,
            array(
                [
                    lit(i)
                    for i in [T3_TOTAL, T3_FAMILY_HOUSEHOLDS, T3_MARRIED_COUPLE_FAMILY]
                ]
            ),
        )
        .when(
            col("T3_DATA_CELL") == T3_OTHER_FAMILY,
            array([lit(i) for i in [T3_TOTAL, T3_FAMILY_HOUSEHOLDS, T3_OTHER_FAMILY]]),
        )
        .when(
            col("T3_DATA_CELL") == T3_MALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
            array(
                [
                    lit(i)
                    for i in [
                        T3_TOTAL,
                        T3_FAMILY_HOUSEHOLDS,
                        T3_OTHER_FAMILY,
                        T3_MALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
                    ]
                ]
            ),
        )
        .when(
            col("T3_DATA_CELL") == T3_FEMALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
            array(
                [
                    lit(i)
                    for i in [
                        T3_TOTAL,
                        T3_FAMILY_HOUSEHOLDS,
                        T3_OTHER_FAMILY,
                        T3_FEMALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
                    ]
                ]
            ),
        )
        .when(
            col("T3_DATA_CELL") == T3_NONFAMILY_HOUSEHOLDS,
            array([lit(i) for i in [T3_TOTAL, T3_NONFAMILY_HOUSEHOLDS]]),
        )
        .when(
            col("T3_DATA_CELL") == T3_HOUSEHOLDER_LIVING_ALONE,
            array(
                [
                    lit(i)
                    for i in [
                        T3_TOTAL,
                        T3_NONFAMILY_HOUSEHOLDS,
                        T3_HOUSEHOLDER_LIVING_ALONE,
                    ]
                ]
            ),
        )
        .when(
            col("T3_DATA_CELL") == T3_HOUSEHOLDER_NOT_LIVING_ALONE,
            array(
                [
                    lit(i)
                    for i in [
                        T3_TOTAL,
                        T3_NONFAMILY_HOUSEHOLDS,
                        T3_HOUSEHOLDER_NOT_LIVING_ALONE,
                    ]
                ]
            ),
        ),
    )


def _add_t4_marginals_column(results_sdf: DataFrame):
    """Adds a MARGINALS column to the dataframe.

    The MARGINALS column will contain a list of all marginal rows (including the
    row itself) that should be calculated from the row in question.
    """
    return results_sdf.withColumn(
        "MARGINALS",
        when(col("T4_DATA_CELL") == T4_TOTAL, array(lit(T4_TOTAL)))
        .when(
            col("T4_DATA_CELL") == T4_OWNED_MORTGAGE_LOAN,
            array(lit(T4_TOTAL), lit(T4_OWNED_MORTGAGE_LOAN)),
        )
        .when(
            col("T4_DATA_CELL") == T4_OWNED_FREE_CLEAR,
            array(lit(T4_TOTAL), lit(T4_OWNED_FREE_CLEAR)),
        )
        .when(
            col("T4_DATA_CELL") == T4_RENTER_OCCUPIED,
            array(lit(T4_TOTAL), lit(T4_RENTER_OCCUPIED)),
        ),
    )


def add_marginals(results_sdf: DataFrame) -> DataFrame:
    """Returns a results dataframe with the marginals added. Works for t3 or t4.

    Expected input format is a dataframe with the following columns:
    * REGION_ID
    * REGION_TYPE
    * ITERATION CODE
    * T3_DATA_CELL or T4_DATA_CELL
    * COUNT
    """
    data_cell_column = (
        "T3_DATA_CELL" if "T3_DATA_CELL" in results_sdf.columns else "T4_DATA_CELL"
    )
    # T3 or T4
    output_type = data_cell_column[:2]
    if output_type == "T3":
        results_sdf = _add_t3_marginals_column(results_sdf)
    else:
        results_sdf = _add_t4_marginals_column(results_sdf)

    # Replace the data cell column with the expanded marginals.
    results_sdf = results_sdf.drop(data_cell_column).withColumnRenamed(
        "MARGINALS", data_cell_column
    )

    return (
        results_sdf.select(
            col("REGION_ID"),
            col("REGION_TYPE"),
            col("ITERATION_CODE"),
            explode(col(data_cell_column)).alias(data_cell_column),
            col("COUNT"),
        )
        .groupBy("REGION_ID", "REGION_TYPE", "ITERATION_CODE", data_cell_column)
        .sum("COUNT")
        .withColumnRenamed("sum(COUNT)", "COUNT")
    )
