"""Create ground truth t3 and t4 counts for SafeTab-H using pure spark computation."""

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
import argparse
import itertools
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit  # pylint: disable=no-name-in-module
from smart_open import open  # pylint: disable=redefined-builtin

from tmlt.common.io_helpers import is_s3_path
from tmlt.safetab_h.paths import ALT_INPUT_CONFIG_DIR_SAFETAB_H, setup_input_config_dir
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
from tmlt.safetab_h.safetab_h_analytics import (
    CONFIG_PARAMS_H,
    T3_STAT_LEVELS,
    T4_STAT_LEVELS,
)
from tmlt.safetab_utils.characteristic_iterations import (
    AttributeFilter,
    IterationManager,
)
from tmlt.safetab_utils.input_validation import validate_input
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


def create_ground_truth_h(
    parameters_path: str,
    data_path: str,
    output_path: str,
    config_path: str,
    overwrite_config: Optional[Dict[str, Any]] = None,
    us_or_puerto_rico: str = "US",
    append: bool = False,
):
    """Save ground truth counts using native spark groupby calculations.

    Args:
        parameters_path: The location of the iterations csv files and config.json.
        data_path: If csv reader, the location of input files.
            If cef reader, the file path to the reader config.
        output_path: The location to save t3 and t4. Appends to existing
            files if they exist.
        config_path: The location of the directory containing the schema files.
        overwrite_config: Optional partial config that will overwrite any values for
            matching keys in the config that is read from config.json.
        us_or_puerto_rico: Whether to tabulate for the 50 states + DC ("US") or
            Puerto Rico ("PR").
        append: Whether to append to existing files, or overwrite.
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
        config_json = json.load(f)
    if overwrite_config is not None:
        for key, value in overwrite_config.items():
            if key in config_json:
                config_json[key] = value
            else:
                raise KeyError(key)
    for key in CONFIG_PARAMS_H:
        if key not in [
            # us_or_puerto_rico is logged instead.
            "run_us",
            "run_pr",
            # These aren't used for the nonprivate algorithm.
            "privacy_defn",
            "privacy_budget_h",
        ]:
            logger.info("\t%s: %s", key, config_json[key])
    logger.info("\tus_or_puerto_rico: %s", us_or_puerto_rico)

    # Validate state filtering
    if us_or_puerto_rico == "US" and validate_state_filter_us(
        config_json[STATE_FILTER_FLAG]
    ):
        state_filter = config_json[STATE_FILTER_FLAG]
    else:
        state_filter = ["72"]

    spark = SparkSession.builder.getOrCreate()
    # Get input reader
    input_reader = safetab_input_reader(
        reader=config_json[READER_FLAG],
        data_path=data_path,
        state_filter=state_filter,
        program="safetab-h",
    )

    # Get household data
    unit_sdf = input_reader.get_unit_df()

    # Create flat map function and apply it.
    # Unlike the private version, we don't worry about separating out coarse-only and
    # detailed-only iterations, because invalid combinations of iteration and region
    # type (e.g. coarse_only at state-level) will be eliminated when
    # we join against the pop_group_dfs, and we're not tracking sensitivity.
    iteration_manager = IterationManager(parameters_path, config_json["max_race_codes"])
    _, _, flatmap = iteration_manager.create_add_iterations_flat_map(
        detailed_only=AttributeFilter.BOTH, coarse_only=AttributeFilter.BOTH
    )
    unit_sdf = unit_sdf.rdd.flatMap(create_augmenting_map(flatmap)).toDF()

    geography_df = preprocess_geography_df(input_reader, us_or_puerto_rico, config_path)

    units_complete = unit_sdf.join(geography_df, on=BLOCK_COLUMNS)

    pop_group_df = input_reader.get_pop_group_details_df()

    # This partitions the dataframe by region_type and iteration code.
    pop_group_dfs = preprocess_pop_group(
        pop_group_df,
        us_or_puerto_rico,
        spark.createDataFrame(iteration_manager.get_iteration_df()),
    )

    t3_dfs = []
    t4_dfs = []

    for region_type, iteration_level in itertools.product(
        REGION_TYPES[us_or_puerto_rico], ["1", "2"]
    ):
        # This adds the t3 and t4 datacell columns to dfs which are filtered on region.
        # These dataframes are then added to dfs to collapse
        # which can create the end DF to use.

        sdf_filtered = pop_group_dfs[(region_type, iteration_level)]

        t3_thresholds = config_json["thresholds_h_t3"][
            f"({region_type}, {iteration_level})"
        ]
        t4_thresholds = config_json["thresholds_h_t4"][
            f"({region_type}, {iteration_level})"
        ]

        pop_counts_t3 = pop_groups_t3(sdf_filtered, t3_thresholds)
        pop_counts_t4 = pop_groups_t4(sdf_filtered, t4_thresholds)

        pop_counts_t3 = pop_counts_t3.join(
            units_complete.withColumn("COUNT", lit(1)),
            on=[region_type, "ITERATION_CODE", "HOUSEHOLD_TYPE"],
            how="left",
        ).withColumnRenamed(region_type, "REGION_ID")

        pop_counts_t4 = pop_counts_t4.join(
            units_complete.withColumn("COUNT", lit(1)),
            on=[region_type, "ITERATION_CODE", "TEN"],
            how="left",
        ).withColumnRenamed(region_type, "REGION_ID")

        t3_dfs.append(pop_counts_t3)
        t4_dfs.append(pop_counts_t4)

    final_pop_t3 = t3_dfs[0]
    for sdf in t3_dfs[1:]:
        final_pop_t3 = final_pop_t3.union(sdf)

    t3 = (
        final_pop_t3.fillna({"COUNT": 0})
        .groupBy(
            "REGION_ID", "REGION_TYPE", "ITERATION_CODE", "T3_DATA_CELL", "STAT_LEVEL"
        )
        .agg({"COUNT": "sum"})
        .select(
            col("REGION_ID"),
            col("REGION_TYPE"),
            col("ITERATION_CODE"),
            col("T3_DATA_CELL"),
            col("STAT_LEVEL"),
            col("sum(COUNT)").alias("COUNT"),
        )
    )

    t3_split = t3_postprocessing([t3])

    print("t3_df")

    final_pop_t4 = t4_dfs[0]
    for sdf in t4_dfs[1:]:
        final_pop_t4 = final_pop_t4.union(sdf)

    t4 = (
        final_pop_t4.fillna({"COUNT": 0})
        .groupBy(
            "REGION_ID", "REGION_TYPE", "ITERATION_CODE", "T4_DATA_CELL", "STAT_LEVEL"
        )
        .agg({"COUNT": "sum"})
        .select(
            col("REGION_ID"),
            col("REGION_TYPE"),
            col("ITERATION_CODE"),
            col("T4_DATA_CELL"),
            col("STAT_LEVEL"),
            col("sum(COUNT)").alias("COUNT"),
        )
    )

    t4_split = t4_postprocessing([t4])

    for stat_level in T3_STAT_LEVELS:
        full_t3_sdf = add_marginals(t3_split[stat_level])
        full_t3_sdf.repartition(1).write.csv(
            os.path.join(output_path, "t3", f"T0300{stat_level+1}"),
            sep="|",
            mode="append" if append else "overwrite",
            header=True,
        )

    for stat_level in T4_STAT_LEVELS:
        full_t4_sdf = add_marginals(t4_split[stat_level])
        full_t4_sdf.repartition(1).write.csv(
            os.path.join(output_path, "t4", f"T0400{stat_level+1}"),
            sep="|",
            mode="append" if append else "overwrite",
            header=True,
        )

    # Unpersist PreProcessing dataframes.
    for value in pop_group_dfs.values():
        value.unpersist()


def main(arglst: List[str] = None) -> None:
    """Main function.

    Args:
        arglst: Optional. List of args usually passed to commandline.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(
        description="Create target counts for SafeTab-H."
    )  # pylint: disable-msg=C0103

    # Standard args
    parser.add_argument(
        "-i",
        "--input",
        dest="parameters_path",
        help="The directory the input files are stored.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-r",
        "--reader",
        dest="data_path",
        help=(
            "string used by the reader. The string is interpreted as an "
            "input csv files directory path "
            "for a csv reader or as a reader config file path for a cef reader."
        ),
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="The directory to store the output in.",
        required=True,
        type=str,  
    )

    args = parser.parse_args(arglst)

    with tempfile.TemporaryDirectory() as updated_config_dir:
        with open(os.path.join(args.parameters_path, "config.json"), "r") as f:
            config_json = json.load(f)
            reader = config_json[READER_FLAG]
            state_filter = []
            if config_json["run_us"] and validate_state_filter_us(
                config_json[STATE_FILTER_FLAG]
            ):
                state_filter += config_json[STATE_FILTER_FLAG]
            if config_json["run_pr"]:
                state_filter += ["72"]

        setup_input_config_dir()

        if validate_input(
            parameters_path=args.parameters_path,
            input_data_configs_path=ALT_INPUT_CONFIG_DIR_SAFETAB_H,
            output_path=updated_config_dir,
            program="safetab-h",
            input_reader=safetab_input_reader(
                reader=reader,
                data_path=args.data_path,
                state_filter=state_filter,
                program="safetab-h",
            ),
            state_filter=state_filter,
        ):
            create_ground_truth_h(
                parameters_path=args.parameters_path,
                data_path=args.data_path,
                output_path=args.output_path,
                config_path=updated_config_dir,
            )


if __name__ == "__main__":
    main()
