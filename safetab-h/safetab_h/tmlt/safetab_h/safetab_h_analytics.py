"""SafeTab-H Algorithm on Analytics.

Runs a differentially private mechanism to create t3 and t4.
"""

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
import json
import logging
import os
import shutil
import tempfile
from enum import Enum
from typing import Any, Dict, List, Optional, cast

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, when  # pylint: disable=no-name-in-module
from pyspark.sql.types import StringType, StructField, StructType
from smart_open import open  # pylint: disable=redefined-builtin

from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PureDPBudget, RhoZCDPBudget
from tmlt.analytics.protected_change import AddOneRow
from tmlt.analytics.query_builder import ColumnType, QueryBuilder
from tmlt.analytics.query_expr import CountMechanism
from tmlt.analytics.session import Session
from tmlt.common.configuration import CategoricalStr
from tmlt.common.io_helpers import is_s3_path
from tmlt.safetab_h.paths import (
    ALT_INPUT_CONFIG_DIR_SAFETAB_H,
    get_safetab_h_output_configs,
    setup_input_config_dir,
    setup_safetab_h_output_config_dir,
)
from tmlt.safetab_h.postprocessing import (
    add_marginals,
    t3_postprocessing,
    t4_postprocessing,
)
from tmlt.safetab_h.preprocessing import (
    pop_groups_t3,
    pop_groups_t4,
    preprocess_pop_group,
)
from tmlt.safetab_utils.characteristic_iterations import (
    AttributeFilter,
    IterationManager,
    LevelFilter,
)
from tmlt.safetab_utils.config_validation import CONFIG_PARAMS_H, validate_config_values
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.output_validation import validate_output
from tmlt.safetab_utils.regions import (
    BLOCK_COLUMNS,
    REGION_TYPES,
    preprocess_geography_df,
    validate_state_filter_us,
)
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    create_augmenting_map,
    safetab_input_reader,
)

T3_STAT_LEVELS = [0, 1, 2, 3]

T4_STAT_LEVELS = [0, 1]


class RegionGranularity(Enum):
    """The type of iterations to tabulate for a region type."""

    COARSE = 1
    DETAILED = 2


REGION_TYPE_TO_GRANULARITY = {
    "USA": RegionGranularity.DETAILED,
    "STATE": RegionGranularity.DETAILED,
    "PR-STATE": RegionGranularity.DETAILED,
    "COUNTY": RegionGranularity.COARSE,
    "PR-COUNTY": RegionGranularity.COARSE,
    "TRACT": RegionGranularity.COARSE,
    "PR-TRACT": RegionGranularity.COARSE,
    "PLACE": RegionGranularity.COARSE,
    "PR-PLACE": RegionGranularity.COARSE,
    "AIANNH": RegionGranularity.COARSE,
}


def _check_popgroups_units_intersect(
    config_json: Any,
    parameters_path: str,
    data_path: str,
    state_filter: List[str],
    updated_config_dir: str,
    us_or_puerto_rico: str,
) -> bool:
    """Check if the pop groups from the t1 file have units in them.

    If there are no units in the population groups from the t1 file, this is a
    validation failure. The user likely used a puerto rico T1 with a US units file,
    or vice versa.
    """
    logger = logging.getLogger(__name__)

    reader = safetab_input_reader(
        reader=config_json[READER_FLAG],
        data_path=data_path,
        state_filter=state_filter,
        program="safetab-h",
    )
    spark = SparkSession.builder.getOrCreate()

    units = reader.get_unit_df()
    if units.count() == 0:
        logger.error(
            f"No units data for the {us_or_puerto_rico} run. Check to make sure you're"
            " using the correct units file for your US/PR settings."
        )
        return False
    iteration_manager = IterationManager(parameters_path, config_json["max_race_codes"])
    _, _, flatmap = iteration_manager.create_add_iterations_flat_map(
        detailed_only=AttributeFilter.BOTH, coarse_only=AttributeFilter.BOTH
    )
    units = units.rdd.flatMap(create_augmenting_map(flatmap)).toDF()

    geos = preprocess_geography_df(reader, us_or_puerto_rico, updated_config_dir)

    units_complete = units.join(geos, on=BLOCK_COLUMNS)

    t1_pop_groups = reader.get_pop_group_details_df()
    if t1_pop_groups.count() == 0:
        logger.error(
            f"No T1 pop groups data for the {us_or_puerto_rico} run. Check to make sure"
            " you're using the correct pop_group_totals file for your US/PR settings."
        )
        return False

    t1_pop_groups_by_level = preprocess_pop_group(
        t1_pop_groups,
        us_or_puerto_rico,
        spark.createDataFrame(iteration_manager.get_iteration_df()),
    )

    for region_type, iteration_level in itertools.product(
        REGION_TYPES[us_or_puerto_rico], ["1", "2"]
    ):

        t1_pop_groups_subset = t1_pop_groups_by_level[(region_type, iteration_level)]

        intersect_groups = t1_pop_groups_subset.join(
            units_complete,
            on=[region_type, "ITERATION_CODE"],
            how="inner",
        )

        if len(intersect_groups.head(1)) > 0:
            return True

    logger.error(
        "No households in the selected population groups. Check to make sure you're"
        " using the correct pop_group_totals file for your US/PR settings."
    )
    return False


def execute_plan_h_analytics(
    parameters_path: str,
    data_path: str,
    output_path: str,
    config_path: str,
    overwrite_config: Optional[Dict[str, Any]] = None,
    us_or_puerto_rico: str = "US",
    append: bool = False,
    should_validate_private_output: bool = False,
):
    """Save a differentially private tabulation of the given relation.

    Args:
        parameters_path: The location of the parameters files.
        data_path: If csv reader, the location of data files.
            If cef reader, the file path to the reader config.
        output_path: The location to save t3 and t4. Appends to existing
            files if they exist, and the append flag is set. Notice that
            :func:`run_safetab_h` overwrites existing files if they exist.
        config_path: The location of the directory containing the schema files.
        overwrite_config: Optional partial config that will overwrite any values for
            matching keys in the config that is read from config.json.
        us_or_puerto_rico: Whether to tabulate for the 50 states + DC ("US") or
            Puerto Rico ("PR").
        append: Whether to append to existing files, or overwrite.
        should_validate_private_output: If True, validate
        private output after tabulations.
    """
    spark_local_mode = (
        SparkSession.builder.getOrCreate().conf.get("spark.master").startswith("local")
    )
    if (is_s3_path(parameters_path) or is_s3_path(output_path)) and spark_local_mode:
        raise RuntimeError(
            "Reading and writing to and from s3"
            " is not supported when running Spark in local mode."
        )

    logger = logging.getLogger(__name__)
    logger.info("Starting SafeTab-H execution...")
    logger.info("with the following parameters:.")
    with open(os.path.join(parameters_path, "config.json"), "r") as f:
        config_json: Dict[str, Any] = json.load(f)
    if overwrite_config is not None:
        for key, value in overwrite_config.items():
            if key in config_json:
                config_json[key] = value
            else:
                raise KeyError(key)
    for key in CONFIG_PARAMS_H:
        if key not in ["run_us", "run_pr"]:  # us_or_puerto_rico is logged instead.
            logger.info("\t%s: %s", key, config_json[key])
    logger.info("\tus_or_puerto_rico: %s", us_or_puerto_rico)

    validate_config_values(config_json, "safetab-h", [us_or_puerto_rico])

    budget_sum: float = 0
    for budget_key in config_json.keys():
        if not budget_key.startswith("privacy_budget_h_t"):
            continue
        budget = config_json[budget_key]
        if isinstance(budget, int):
            config_json[budget_key] = float(budget)
            budget = float(budget)
        else:
            assert isinstance(budget, float)
        budget_sum += budget
    (total_budget, budget_type, noise_mechanism) = (
        (PureDPBudget(budget_sum), PureDPBudget, CountMechanism.LAPLACE)
        if config_json["privacy_defn"] == "puredp"
        else (RhoZCDPBudget(budget_sum), RhoZCDPBudget, CountMechanism.GAUSSIAN)
    )
    logger.info("Total budget: %s", str(total_budget))

    logger.info("Getting data...")
    # Validate state filtering
    if us_or_puerto_rico == "US" and validate_state_filter_us(
        config_json[STATE_FILTER_FLAG]
    ):
        state_filter = config_json[STATE_FILTER_FLAG]
    else:
        state_filter = ["72"]

    # Get input reader
    input_reader = safetab_input_reader(
        reader=config_json[READER_FLAG],
        data_path=data_path,
        state_filter=state_filter,
        program="safetab-h",
    )

    # Get household data
    unit_df = input_reader.get_unit_df()

    # Preprocess geography_df.
    # Contains one column for each column in REGION_TYPES[us_or_puerto_rico] and shares
    # BLOCK_COLUMNS with person-records.txt.
    geography_df = preprocess_geography_df(input_reader, us_or_puerto_rico, config_path)

    logger.info("Creating session...")
    # Create Session
    session = Session.from_dataframe(
        privacy_budget=total_budget,
        source_id="unit_source",
        dataframe=unit_df,
        protected_change=AddOneRow(),
    )
    session.add_public_dataframe("geo_source", geography_df)

    logger.info("Building root query...")
    # CREATE THE ROOT QUERY BUILDER.
    # Do a public join with the geo dataframe.
    root_builder = QueryBuilder(source_id="unit_source").join_public(
        public_table="geo_source"
    )
    session.create_view(root_builder, "root", cache=True)

    spark = SparkSession.builder.getOrCreate()
    # Initialize domains that will be used in the main computation loop
    iteration_manager = IterationManager(parameters_path, config_json["max_race_codes"])

    # Get SafeTab-P output
    pop_group_df = input_reader.get_pop_group_details_df()

    # This partitions the dataframe by region_type and iteration code.
    pop_group_dfs = preprocess_pop_group(
        pop_group_df,
        us_or_puerto_rico,
        spark.createDataFrame(iteration_manager.get_iteration_df()),
    )

    enforce_nonnegativity = not config_json["allow_negative_counts"]

    t3_results: List[DataFrame] = []
    t4_results: List[DataFrame] = []

    # Create a flat map.
    iterations_domains = {}
    for iteration_level in ["1", "2"]:
        for granularity in [RegionGranularity.COARSE, RegionGranularity.DETAILED]:
            detailed_only = (
                AttributeFilter.BOTH
                if granularity == RegionGranularity.DETAILED
                else AttributeFilter.EXCLUDE
            )
            coarse_only = (
                AttributeFilter.BOTH
                if granularity == RegionGranularity.COARSE
                else AttributeFilter.EXCLUDE
            )
            flat_map_container = iteration_manager.create_add_iterations_flat_map(
                detailed_only=detailed_only,
                coarse_only=coarse_only,
                level=LevelFilter.ONE if iteration_level == "1" else LevelFilter.TWO,
            )

            session.create_view(
                QueryBuilder(source_id="root").flat_map(
                    flat_map_container.flat_map,
                    max_num_rows=flat_map_container.sensitivity,
                    new_column_types={"ITERATION_CODE": ColumnType.VARCHAR},
                    augment=True,
                    grouping=True,
                ),
                f"root_{iteration_level}_{granularity.name}",
                cache=True,
            )
            iterations_domains[(iteration_level, granularity)] = cast(
                CategoricalStr, flat_map_container.output_domain["ITERATION_CODE"]
            ).values

    session.delete_view("root")

    for region_type, iteration_level in itertools.product(
        REGION_TYPES[us_or_puerto_rico], ["1", "2"]
    ):
        granularity = REGION_TYPE_TO_GRANULARITY[region_type]

        # Intersect the possible iterations domain of the flatmap with the iterations
        # from the pop groups from SafeTab-P to drop "disallowed" combinations created
        # via SafeTab-P coterminus geo postprocessing.
        iteration_domain_df = spark.createDataFrame(
            [
                (iteration,)
                for iteration in iterations_domains[(iteration_level, granularity)]
            ],
            schema=StructType([StructField("ITERATION_CODE", StringType())]),
        )
        pop_group_df = pop_group_dfs[(region_type, iteration_level)].join(
            iteration_domain_df, on=["ITERATION_CODE"], how="inner"
        )

        # If there are no groups for this region type x iteration level, continue.
        if pop_group_df.count() == 0:
            continue

        t3_thresholds = config_json["thresholds_h_t3"][
            f"({region_type}, {iteration_level})"
        ]
        t4_thresholds = config_json["thresholds_h_t4"][
            f"({region_type}, {iteration_level})"
        ]

        pop_counts_t3 = pop_groups_t3(pop_group_df, t3_thresholds).cache()
        pop_counts_t4 = pop_groups_t4(pop_group_df, t4_thresholds).cache()

        builder = (
            QueryBuilder(source_id=f"root_{iteration_level}_{granularity.name}")
            .join_public(
                pop_counts_t3,
                join_columns=[region_type, "ITERATION_CODE", "HOUSEHOLD_TYPE"],
            )
            .rename({"STAT_LEVEL": "STAT_LEVEL_T3"})
            .join_public(
                pop_counts_t4, join_columns=[region_type, "ITERATION_CODE", "TEN"]
            )
            .rename({"STAT_LEVEL": "STAT_LEVEL_T4"})
            .select(
                columns=[
                    "ITERATION_CODE",
                    "T3_DATA_CELL",
                    "T4_DATA_CELL",
                    "STAT_LEVEL_T3",
                    "STAT_LEVEL_T4",
                ]
                + REGION_TYPES[us_or_puerto_rico]
            )
        )

        # The public join above is a defacto inner join on region_id because the
        # region_id has been switched to region_type.

        sanitized_region_type = region_type.replace("-", "_").lower()
        local_view = f"safetab_{sanitized_region_type}_{iteration_level}"
        session.create_view(builder, local_view, cache=True)

        logger.info(
            f"Creating and evaluating queries for {region_type}, iteration level"
            f" {iteration_level} T3..."
        )
        t3_raw_budget = config_json[
            f"privacy_budget_h_t3_level_{iteration_level}_{sanitized_region_type}"
        ]
        if t3_raw_budget > 0:
            t3_keyset = pop_counts_t3.select(
                [region_type, "ITERATION_CODE", "T3_DATA_CELL", "STAT_LEVEL"]
            )

            t3_budget = budget_type(t3_raw_budget)
            t3_query = (
                QueryBuilder(local_view)
                .rename({"STAT_LEVEL_T3": "STAT_LEVEL"})
                .groupby(KeySet.from_dataframe(t3_keyset))
                .count(name="COUNT", mechanism=noise_mechanism)
            )

            # The input of column named "STATE" gets renamed to "REGION_ID"
            # because the column STATE has an ID that is the region id.
            t3_result = (
                session.evaluate(t3_query, privacy_budget=t3_budget)
                .withColumnRenamed(region_type, "REGION_ID")
                .withColumn("REGION_TYPE", lit(region_type))
                # adding the region_type column based on the current region,
                # in this case STATE.
            )
            if enforce_nonnegativity:
                t3_result = t3_result.withColumn(
                    "COUNT", when(col("COUNT") < 0, 0).otherwise(col("COUNT"))
                )
            t3_results.append(t3_result)

        logger.info(
            f"Creating and evaluating queries for {region_type}, iteration level"
            f" {iteration_level} T4..."
        )
        t4_raw_budget = config_json[
            f"privacy_budget_h_t4_level_{iteration_level}_{sanitized_region_type}"
        ]
        if t4_raw_budget > 0:
            t4_keyset = pop_counts_t4.select(
                [region_type, "ITERATION_CODE", "T4_DATA_CELL", "STAT_LEVEL"]
            )

            t4_budget = budget_type(t4_raw_budget)
            t4_query = (
                QueryBuilder(local_view)
                .rename({"STAT_LEVEL_T4": "STAT_LEVEL"})
                .groupby(KeySet.from_dataframe(t4_keyset))
                .count(name="COUNT", mechanism=noise_mechanism)
            )
            t4_result = (
                session.evaluate(t4_query, privacy_budget=t4_budget)
                .withColumnRenamed(region_type, "REGION_ID")
                .withColumn("REGION_TYPE", lit(region_type))
            )
            if enforce_nonnegativity:
                t4_result = t4_result.withColumn(
                    "COUNT", when(col("COUNT") < 0, 0).otherwise(col("COUNT"))
                )
            t4_results.append(t4_result)

        # Unpersist pop_counts DataFrames.
        pop_counts_t3.unpersist()
        pop_counts_t4.unpersist()
        session.delete_view(local_view)

        # Unpersist pop_group_dfs DataFrames.
        pop_group_dfs[(region_type, iteration_level)].unpersist()

    t3_sdfs = t3_postprocessing(t3_results)
    for stat_level in T3_STAT_LEVELS:
        t3_sdf = add_marginals(t3_sdfs[stat_level])
        t3_sdf.repartition(1).write.csv(
            os.path.join(output_path, "t3", f"T0300{stat_level+1}"),
            sep="|",
            mode="append" if append else "overwrite",
            header=True,
        )

    t4_sdfs = t4_postprocessing(t4_results)
    for stat_level in T4_STAT_LEVELS:
        t4_sdf = add_marginals(t4_sdfs[stat_level])
        t4_sdf.repartition(1).write.csv(
            os.path.join(output_path, "t4", f"T0400{stat_level+1}"),
            sep="|",
            mode="append" if append else "overwrite",
            header=True,
        )

    for iteration_level in ["1", "2"]:
        for granularity in [RegionGranularity.COARSE, RegionGranularity.DETAILED]:
            session.delete_view(f"root_{iteration_level}_{granularity.name}")

    if should_validate_private_output:
        if not validate_output(
            output_sdfs={"t3": t3_sdf, "t4": t4_sdf},
            expected_output_configs=get_safetab_h_output_configs(),
            state_filter=state_filter,
            allow_negative_counts_flag=config_json["allow_negative_counts"],
        ):
            logger.error("SafeTab-H output validation failed. Exiting...")
            raise RuntimeError("Output validation Failed.")

    logger.info("SafeTab-H completed successfully.")


def run_plan_h_analytics(
    parameters_path: str,
    data_path: str,
    output_path: str,
    overwrite_config: Optional[Dict[str, Any]] = None,
    should_validate_private_output: bool = False,
):
    """Entry point for SafeTab-H algorithm.

    First validates input files, and builds the expected domain of
    `household-records.txt` from files such as `GRF-C.txt`. See
    :mod:`.input_validation` for more details.

    .. warning::
        During validation, `household-records.txt` is checked against the expected
        domain, to make sure that the input files are consistent.

    Args:
        parameters_path: The location of the parameters files.
        data_path: If csv reader, the location of input files.
            If cef reader, the file path to the reader config.
        output_path: The location to save t3 and t4. Overwrites existing
            files if they exist. Notice that :func:`execute_plan_h_analytics` appends
            to existing files if they exist, and the append flag is set.
        overwrite_config: Optional partial config that will overwrite any values for
            matching keys in the config that is read from config.json.
        should_validate_private_output: If True, validate private output after
            tabulations.
    """
    setup_input_config_dir()
    setup_safetab_h_output_config_dir()

    us_or_puerto_rico_values = []
    if overwrite_config is None:
        overwrite_config = {}
    with open(os.path.join(parameters_path, "config.json"), "r") as f:
        config_json = json.load(f)
        config_json.update(overwrite_config)
    if config_json["run_us"]:
        us_or_puerto_rico_values.append("US")
    if config_json["run_pr"]:
        us_or_puerto_rico_values.append("PR")
    if not us_or_puerto_rico_values:
        raise ValueError(
            "Invalid config: At least one of 'run_us', 'run_pr' must be True."
        )

    validate_config_values(config_json, "safetab-h", us_or_puerto_rico_values)

    with tempfile.TemporaryDirectory() as updated_config_dir:
        # Find states used in this execution to validate input.
        state_filter = []
        if "US" in us_or_puerto_rico_values and validate_state_filter_us(
            config_json[STATE_FILTER_FLAG]
        ):
            state_filter += config_json[STATE_FILTER_FLAG]
        if "PR" in us_or_puerto_rico_values:
            state_filter += ["72"]

        if not validate_input(
            parameters_path=parameters_path,
            input_data_configs_path=ALT_INPUT_CONFIG_DIR_SAFETAB_H,
            output_path=updated_config_dir,
            program="safetab-h",
            input_reader=safetab_input_reader(
                reader=config_json[READER_FLAG],
                data_path=data_path,
                state_filter=state_filter,
                program="safetab-h",
            ),
            state_filter=state_filter,
        ):
            raise RuntimeError("Input validation failed.")
        for us_or_puerto_rico in us_or_puerto_rico_values:
            state_filter = (
                config_json[STATE_FILTER_FLAG] if us_or_puerto_rico == "US" else ["72"]
            )

            if not _check_popgroups_units_intersect(
                config_json=config_json,
                parameters_path=parameters_path,
                data_path=data_path,
                state_filter=state_filter,
                updated_config_dir=updated_config_dir,
                us_or_puerto_rico=us_or_puerto_rico,
            ):
                raise RuntimeError("Input validation failed.")

        for tx in ["t3", "t4"]:
            filepath = os.path.join(output_path, tx)
            if os.path.exists(filepath):
                shutil.rmtree(filepath)
        for us_or_puerto_rico in us_or_puerto_rico_values:
            execute_plan_h_analytics(
                parameters_path=parameters_path,
                data_path=data_path,
                output_path=output_path,
                config_path=updated_config_dir,
                overwrite_config=overwrite_config,
                us_or_puerto_rico=us_or_puerto_rico,
                # If US and PR are run together, append PR results to existing US
                # results.
                append=(
                    (us_or_puerto_rico == "PR") and ("US" in us_or_puerto_rico_values)
                ),
                should_validate_private_output=should_validate_private_output,
            )
