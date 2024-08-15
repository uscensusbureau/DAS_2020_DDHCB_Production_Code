"""Functions for preprocessing input files."""

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

import itertools
from typing import Dict, Tuple

from pyspark.sql import DataFrame
from pyspark.sql.functions import (  # pylint: disable=no-name-in-module
    array,
    col,
    explode,
    lit,
    when,
)
from pyspark.sql.types import StringType

from tmlt.safetab_utils.regions import REGION_TYPES

T3_TOTAL = 0
T3_FAMILY_HOUSEHOLDS = 1
T3_MARRIED_COUPLE_FAMILY = 2
T3_OTHER_FAMILY = 3
T3_MALE_HOUSEHOLDER_NO_SPOUSE_PARTNER = 4
T3_FEMALE_HOUSEHOLDER_NO_SPOUSE_PARTNER = 5
T3_NONFAMILY_HOUSEHOLDS = 6
T3_HOUSEHOLDER_LIVING_ALONE = 7
T3_HOUSEHOLDER_NOT_LIVING_ALONE = 8

T4_TOTAL = 0
T4_OWNED_MORTGAGE_LOAN = 1
T4_OWNED_FREE_CLEAR = 2
T4_RENTER_OCCUPIED = 3


MARRIED_COUPLE_FAMILY = 1
MALE_FAMILY_HOUSEHOLD = 2
FEMALE_FAMILY_HOUSEHOLD = 3
HOUSEHOLDER_LIVING_ALONE = 4
HOUSEHOLDER_NOT_LIVING_ALONE = 5

OWNED_MORTAGE_LOAN = 1
OWNED_FREE_CLEAR = 2
RENTER_OCCUPIED = 3
OCCUPIED_NON_OWNER_NO_RENT = 4


def preprocess_pop_group(
    pop_group_df: DataFrame, us_or_puerto_rico: str, iterations_df: DataFrame
) -> Dict[Tuple[str, str], DataFrame]:
    """Return mapping of region_type x iteration level to respective filtered dataframe.

    This function partitions the population group dataframe by region type and iteration
    level.

    Expected input format is a dataframe with the following columns

    * REGION_ID
    * REGION_TYPE
    * ITERATION_CODE
    * COUNT

    The format of each column's data is defined in
    tmlt/safetab_h/resources/config/input/pop-group-totals.json

    The dataframes in the output have the following columns:
    * <REGION_TYPE>: region_id -- The REGION_TYPE ("STATE", "COUNTY", ect.) is turned
        into a column with the region_id. This matches the unit DF input.
    * REGION_TYPE: Unaltered.
    * ITERATION_CODE: Unaltered
    * T1_COUNT: int matching the prior COUNT variable.

    Args:
        pop_group_df: The data frame in expected input format.
        us_or_puerto_rico: A string indicating whether the data is for the US or Puerto
        Rico.
        iterations_df: A Spark Dataframe from the iteration manager.
    """
    regions = set(REGION_TYPES[us_or_puerto_rico])

    # ensures that the ITERATION_CODE in pop_groups_df is a str.
    if pop_group_df.schema["ITERATION_CODE"].dataType != StringType():
        pop_group_df = pop_group_df.withColumn(
            "ITERATION_CODE", col("ITERATION_CODE").cast(StringType())
        )

    # Move / Rename Columns in respect to REGION_TYPE

    region_dfs: Dict[Tuple[str, str], DataFrame] = {}
    for region, iteration_level in itertools.product(regions, ["1", "2"]):
        region_dfs[(region, iteration_level)] = (
            pop_group_df.filter(col("REGION_TYPE") == region)
            .join(
                iterations_df.filter(col("LEVEL") == iteration_level).select(
                    "ITERATION_CODE"
                ),
                on=["ITERATION_CODE"],
            )
            .withColumnRenamed(
                "REGION_ID", region
            )  # This enables a join on region_id as region_id is renamed to REGION_TYPE
            .withColumnRenamed("COUNT", "T1_COUNT")
            .cache()
        )

    return region_dfs


def pop_groups_t3(sdf: DataFrame, thresholds: list) -> DataFrame:
    """Adds columns for household_type and T3_DATA_CELL to the pop_groups df.

    T3_DATA_CELL is based on the thresholds. The output will include all valid
    combinations of T3_DATA_CELL and HOUSEHOLD_TYPE. HOUSEHOLD_TYPE rows that do not
    match the data will need to be removed by the joining to the units data.

    Args:
        sdf: A Spark Dataframe with T1_COUNT and HOUSEHOLD_TYPE columns added.
        thresholds: A list of the tier 3 threshold values to determine datacell values.
    """
    # Three threshold levels for T3
    high_threshold = thresholds[2]
    medium_threshold = thresholds[1]
    low_threshold = thresholds[0]

    sdf_processed = sdf.withColumn(
        "HOUSEHOLD_TYPE",
        explode(
            array(
                [
                    lit(MARRIED_COUPLE_FAMILY),
                    lit(MALE_FAMILY_HOUSEHOLD),
                    lit(FEMALE_FAMILY_HOUSEHOLD),
                    lit(HOUSEHOLDER_LIVING_ALONE),
                    lit(HOUSEHOLDER_NOT_LIVING_ALONE),
                ]
            )
        ),
    )

    sdf_with_stat_levels = sdf_processed.withColumn(
        "STAT_LEVEL",
        when((col("T1_COUNT") >= high_threshold), 3)
        .when((col("T1_COUNT") >= medium_threshold), 2)
        .when((col("T1_COUNT") >= low_threshold), 1)
        .otherwise(0),
    )

    # Adaptively calculates the T3_DATA_CELL values based on public safetab-p info.
    return sdf_with_stat_levels.withColumn(
        "T3_DATA_CELL",
        # High Detail Level
        when(
            (col("STAT_LEVEL") == 3) & (col("HOUSEHOLD_TYPE") == MALE_FAMILY_HOUSEHOLD),
            T3_MALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
        )
        .when(
            (col("STAT_LEVEL") == 3)
            & (col("HOUSEHOLD_TYPE") == FEMALE_FAMILY_HOUSEHOLD),
            T3_FEMALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
        )
        .when(
            (col("STAT_LEVEL") == 3) & (col("HOUSEHOLD_TYPE") == MARRIED_COUPLE_FAMILY),
            T3_MARRIED_COUPLE_FAMILY,
        )
        .when(
            (col("STAT_LEVEL") == 3)
            & (col("HOUSEHOLD_TYPE") == HOUSEHOLDER_LIVING_ALONE),
            T3_HOUSEHOLDER_LIVING_ALONE,
        )
        .when(
            (col("STAT_LEVEL") == 3)
            & (col("HOUSEHOLD_TYPE") == HOUSEHOLDER_NOT_LIVING_ALONE),
            T3_HOUSEHOLDER_NOT_LIVING_ALONE,
        )
        # Medium Detail Level
        .when(
            (col("STAT_LEVEL") == 2) & (col("HOUSEHOLD_TYPE") == MARRIED_COUPLE_FAMILY),
            T3_MARRIED_COUPLE_FAMILY,
        )
        .when(
            (col("STAT_LEVEL") == 2)
            & col("HOUSEHOLD_TYPE").isin(
                [MALE_FAMILY_HOUSEHOLD, FEMALE_FAMILY_HOUSEHOLD]
            ),
            T3_OTHER_FAMILY,
        )
        .when(
            (col("STAT_LEVEL") == 2)
            & (col("HOUSEHOLD_TYPE") == HOUSEHOLDER_LIVING_ALONE),
            T3_HOUSEHOLDER_LIVING_ALONE,
        )
        .when(
            (col("STAT_LEVEL") == 2)
            & (col("HOUSEHOLD_TYPE") == HOUSEHOLDER_NOT_LIVING_ALONE),
            T3_HOUSEHOLDER_NOT_LIVING_ALONE,
        )
        # Low Detail Level
        .when(
            (col("STAT_LEVEL") == 1)
            & (
                col("HOUSEHOLD_TYPE").isin(
                    [
                        MARRIED_COUPLE_FAMILY,
                        MALE_FAMILY_HOUSEHOLD,
                        FEMALE_FAMILY_HOUSEHOLD,
                    ]
                )
            ),
            T3_FAMILY_HOUSEHOLDS,
        ).when(
            (col("STAT_LEVEL") == 1)
            & (
                col("HOUSEHOLD_TYPE").isin(
                    [HOUSEHOLDER_LIVING_ALONE, HOUSEHOLDER_NOT_LIVING_ALONE]
                )
            ),
            T3_NONFAMILY_HOUSEHOLDS,
        )
        # Overall Totals
        .otherwise(T3_TOTAL),
    )


def pop_groups_t4(sdf: DataFrame, thresholds: list) -> DataFrame:
    """Adds columns for TEN and T4_DATA_CELL to the pop_groups df.

    T4_DATA_CELL is based on the threshold level. The output will include all valid
    combinations of TEN and T4_DATA_CELL. TEN rows that do not match the data will
    need to be removed by joining to the units data.

    Args:
        sdf: A Spark Dataframe with T1_COUNT and TEN columns added.
        thresholds: A list with the tier 3 threshold value
            to determine the datacell values.
    """
    threshold = thresholds[0]

    sdf_processed = sdf.withColumn(
        "TEN",
        explode(
            array(
                [
                    lit(OWNED_MORTAGE_LOAN),
                    lit(OWNED_FREE_CLEAR),
                    lit(RENTER_OCCUPIED),
                    lit(OCCUPIED_NON_OWNER_NO_RENT),
                ]
            )
        ),
    )

    sdf_with_stat_levels = sdf_processed.withColumn(
        "STAT_LEVEL", when((col("T1_COUNT") >= threshold), 1).otherwise(0)
    )

    return sdf_with_stat_levels.withColumn(
        "T4_DATA_CELL",
        when(
            (col("STAT_LEVEL") == 1) & (col("TEN") == OCCUPIED_NON_OWNER_NO_RENT),
            T4_RENTER_OCCUPIED,
        )  # relist tenure to 3 per datacell requirements.
        .when(
            (col("STAT_LEVEL") == 1) & (col("TEN") == RENTER_OCCUPIED),
            T4_RENTER_OCCUPIED,
        )
        .when(
            (col("STAT_LEVEL") == 1) & (col("TEN") == OWNED_FREE_CLEAR),
            T4_OWNED_FREE_CLEAR,
        )
        .when(
            (col("STAT_LEVEL") == 1) & (col("TEN") == OWNED_MORTAGE_LOAN),
            T4_OWNED_MORTGAGE_LOAN,
        )
        .otherwise(T4_TOTAL),
    )
