"""Utilities for comparing the results from SafeTab-H against the ground truth.

As this report uses the ground truth counts, it violates differential privacy,
and should not be created using sensitive data. Rather its purpose is to test
SafeTab-H on non-sensitive or synthetic datasets to help tune the algorithms and
to predict the performance on the private data.
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

import argparse
import functools
import itertools
import json
import math
import os
import tempfile
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as PySparkDataFrame
from pyspark.sql import SparkSession
from smart_open import open  # pylint: disable=redefined-builtin

from tmlt.common.io_helpers import is_s3_path, read_csv, to_csv_with_create_dir
from tmlt.safetab_h.paths import ALT_INPUT_CONFIG_DIR_SAFETAB_H, setup_input_config_dir
from tmlt.safetab_h.postprocessing import T3_SCHEMA, T4_SCHEMA
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
from tmlt.safetab_h.safetab_h_analytics import execute_plan_h_analytics
from tmlt.safetab_h.target_counts_h import create_ground_truth_h
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.regions import validate_state_filter_us
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    get_augmented_df,
    safetab_input_reader,
)


class SafeTabOutputDir(NamedTuple):
    """The output directory and associated information."""

    relative_path: str
    stat_level: str
    directly_computed_cell_codes: List[str]
    data_cell_column: str


TABLE_TO_DIRS = {
    "t3": [
        SafeTabOutputDir(
            relative_path=os.path.join("t3", "T03001"),
            stat_level="0",
            directly_computed_cell_codes=[str(T3_TOTAL)],
            data_cell_column="T3_DATA_CELL",
        ),
        SafeTabOutputDir(
            relative_path=os.path.join("t3", "T03002"),
            stat_level="1",
            directly_computed_cell_codes=[
                str(i) for i in [T3_FAMILY_HOUSEHOLDS, T3_NONFAMILY_HOUSEHOLDS]
            ],
            data_cell_column="T3_DATA_CELL",
        ),
        SafeTabOutputDir(
            relative_path=os.path.join("t3", "T03003"),
            stat_level="2",
            directly_computed_cell_codes=[
                str(i)
                for i in [
                    T3_MARRIED_COUPLE_FAMILY,
                    T3_OTHER_FAMILY,
                    T3_HOUSEHOLDER_LIVING_ALONE,
                    T3_HOUSEHOLDER_NOT_LIVING_ALONE,
                ]
            ],
            data_cell_column="T3_DATA_CELL",
        ),
        SafeTabOutputDir(
            relative_path=os.path.join("t3", "T03004"),
            stat_level="3",
            directly_computed_cell_codes=[
                str(i)
                for i in [
                    T3_MARRIED_COUPLE_FAMILY,
                    T3_MALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
                    T3_FEMALE_HOUSEHOLDER_NO_SPOUSE_PARTNER,
                    T3_HOUSEHOLDER_LIVING_ALONE,
                    T3_HOUSEHOLDER_NOT_LIVING_ALONE,
                ]
            ],
            data_cell_column="T3_DATA_CELL",
        ),
    ],
    "t4": [
        SafeTabOutputDir(
            relative_path=os.path.join("t4", "T04001"),
            stat_level="0",
            directly_computed_cell_codes=[str(T4_TOTAL)],
            data_cell_column="T4_DATA_CELL",
        ),
        SafeTabOutputDir(
            relative_path=os.path.join("t4", "T04002"),
            stat_level="1",
            directly_computed_cell_codes=[
                str(i)
                for i in [
                    T4_OWNED_MORTGAGE_LOAN,
                    T4_OWNED_FREE_CLEAR,
                    T4_RENTER_OCCUPIED,
                ]
            ],
            data_cell_column="T4_DATA_CELL",
        ),
    ],
}


def create_error_report_h(
    noisy_path: str,
    ground_truth_path: str,
    parameters_path: str,
    output_path: str,
    outputs: str = "All",
):
    """Create an error report from a single run of SafeTab-H.

    The error report is saved to {output_path}/error_report.csv

    Args:
        noisy_path: the path containing the noisy t3 and t4.
        ground_truth_path: the path containing the ground_truth t3 and t4.
        parameters_path: the path containing the iterations files (used to gather
            additional information about iteration codes).
        output_path: the path where the output should be saved.
        outputs: the output file to evaluate with the error report
    """
    population_group_columns = ["REGION_ID", "REGION_TYPE", "ITERATION_CODE"]

    t3_dirs_used = (
        [dir.relative_path for dir in TABLE_TO_DIRS["t3"]]
        if outputs == "All"
        else ["t3/T0300" + outputs]
    )
    t4_dirs_used = (
        [dir.relative_path for dir in TABLE_TO_DIRS["t4"]]
        if outputs == "All"
        else ["t4/T0400" + outputs]
    )

    t3_dfs = []
    for output_dir in t3_dirs_used:
        dir_args = output_dir.split("/")

        t3_dfs.append(
            get_augmented_df(
                dir_args[1],
                os.path.join(noisy_path, dir_args[0]),
                os.path.join(ground_truth_path, dir_args[0]),
            )
        )
    t3_df = pd.concat(t3_dfs)
    t3_df["Population group size"] = t3_df["GROUND_TRUTH"]
    t3_df["Total absolute error in T3"] = np.abs(t3_df["NOISY"] - t3_df["GROUND_TRUTH"])
    t3_df["Total squared error in T3"] = np.square(t3_df["Total absolute error in T3"])
    t3_df = (
        t3_df[
            population_group_columns
            + [
                "Population group size",
                "Total absolute error in T3",
                "Total squared error in T3",
            ]
        ]
        .groupby(population_group_columns)
        .agg("sum")
        .reset_index()
    )

    t4_dfs = []
    for output_dir in t4_dirs_used:
        t4_dfs.append(get_augmented_df(output_dir, noisy_path, ground_truth_path))
    t4_df = pd.concat(t4_dfs)
    t4_df["Total absolute error in T4"] = np.abs(t4_df["NOISY"] - t4_df["GROUND_TRUTH"])
    t4_df["Total squared error in T4"] = np.square(t4_df["Total absolute error in T4"])
    t4_df = (
        t4_df[
            population_group_columns
            + ["Total absolute error in T4", "Total squared error in T4"]
        ]
        .groupby(population_group_columns)
        .agg("sum")
        .reset_index()
    )
    results = pd.merge(t3_df, t4_df, on=population_group_columns)

    # Load DETAILED_ONLY and COARSE_ONLY info, and join with results
    ethnicity_iterations_df = read_csv(
        os.path.join(parameters_path, "ethnicity-characteristic-iterations.txt"),
        delimiter="|",
        dtype=str,
        usecols=["ITERATION_CODE", "DETAILED_ONLY", "COARSE_ONLY"],
    )
    race_iterations_df = read_csv(
        os.path.join(parameters_path, "race-characteristic-iterations.txt"),
        delimiter="|",
        dtype=str,
        usecols=["ITERATION_CODE", "DETAILED_ONLY", "COARSE_ONLY"],
    )
    iterations_df = pd.concat([ethnicity_iterations_df, race_iterations_df])
    results = results.merge(iterations_df, how="left")

    # Select and order output columns
    results = results.loc[
        :,
        [
            "REGION_ID",
            "REGION_TYPE",
            "ITERATION_CODE",
            "DETAILED_ONLY",
            "COARSE_ONLY",
            "Population group size",
            "Total absolute error in T3",
            "Total squared error in T3",
            "Total absolute error in T4",
            "Total squared error in T4",
        ],
    ]

    to_csv_with_create_dir(
        results, os.path.join(output_path, "error_report.csv"), index=False
    )


def run_full_error_report_h(
    parameters_path: str,
    data_path: str,
    output_path: str,
    config_path: str,
    trials: int,
    overwrite_config: Optional[Dict[str, Any]] = None,
    us_or_puerto_rico: str = "US",
):
    """Run SafeTab-H for multiple trials and create an error report.

    Run SafeTab-H for <trials> trials . Create a ground truth for these runs and
    noisy answers for each run. Finally, create an
    aggregated error report from the single-run
    reports. This aggregated error report is saved in
    "<output_path>/multi_run_error_report.csv".

    Args:
        parameters_path: The path containing the input files for SafeTab-H.
        data_path: If csv reader, the location of input files.
            If cef reader, the file path to the reader config.
        output_path: The path where the output file "multi_run_error_report.csv" is
            saved.
        config_path: The location of the directory containing the schema files.
        trials: The number of trials to run for each privacy parameter.
        overwrite_config: Optional partial config that will overwrite any values for
            matching keys in the config that is read from config.json.
        us_or_puerto_rico: Whether to tabulate for the 50 states + DC ("US") or
            Puerto Rico ("PR").
    """
    setup_input_config_dir()

    with open(os.path.join(parameters_path, "config.json"), "r") as f:
        config_json = json.load(f)
        reader = config_json[READER_FLAG]
        if us_or_puerto_rico == "US" and validate_state_filter_us(
            config_json[STATE_FILTER_FLAG]
        ):
            state_filter = config_json[STATE_FILTER_FLAG]
        else:
            state_filter = ["72"]
    input_reader = safetab_input_reader(
        reader=reader,
        data_path=data_path,
        state_filter=state_filter,
        program="safetab-h",
    )
    validate_input(
        parameters_path=parameters_path,
        input_data_configs_path=ALT_INPUT_CONFIG_DIR_SAFETAB_H,
        output_path=config_path,
        program="safetab-h",
        input_reader=input_reader,
        state_filter=state_filter,
    )

    # All paths corresponding to safetab runs.
    single_run_paths: List[str] = []

    # Run safetab for <trials> trials and save the results.
    for trial in range(trials):
        dir_name = f"trial_{trial}"
        single_run_path = os.path.join(output_path, "single_runs", dir_name)
        single_run_paths.append(single_run_path)
        execute_plan_h_analytics(
            parameters_path=parameters_path,
            data_path=data_path,
            output_path=single_run_path,
            config_path=config_path,
            overwrite_config=overwrite_config,
            us_or_puerto_rico=us_or_puerto_rico,
        )

    # Create ground truth using all safetab runs for all privacy budget values.
    ground_truth_path = os.path.join(output_path, "ground_truth")
    create_ground_truth_h(
        parameters_path=parameters_path,
        data_path=data_path,
        output_path=ground_truth_path,
        config_path=config_path,
        overwrite_config=overwrite_config,
        us_or_puerto_rico=us_or_puerto_rico,
    )

    # Create the aggregated error report.
    multi_run_path = os.path.join(output_path, "full_error_report")
    create_aggregated_error_report_h(
        single_run_paths=single_run_paths,
        parameters_path=parameters_path,
        ground_truth_path=ground_truth_path,
        output_path=multi_run_path,
    )


def get_augmented_df_from_spark(
    noisy_df: PySparkDataFrame, ground_truth_df: PySparkDataFrame
) -> PySparkDataFrame:
    """Joins the noisy and ground truth results into one dataframe in spark.

    Args:
        noisy_df: The noisy differentially private count or sum.
        ground_truth_df: The ground truth count or sum.

    Returns:
        A dataframe containing all of the expected columns from `Appendix A`,
        minus "COUNT"/"SUM". The dataframe is augmented with the following
        columns:

        * NOISY: The counts from the mechanism, as integers.
        * GROUND_TRUTH: The ground truth counts, as integers.
    """
    if "SUM" in noisy_df.columns:
        col = "SUM"
    else:
        if "COUNT" not in noisy_df.columns:
            raise ValueError("Noisy data must contain either a SUM or COUNT column.")
        col = "COUNT"
    noisy_df = noisy_df.withColumnRenamed(col, "NOISY")
    ground_truth_df = ground_truth_df.withColumnRenamed(col, "GROUND_TRUTH")
    join_columns = list(set(noisy_df.columns).intersection(ground_truth_df.columns))
    # The noisy data will have the full keyset, while the ground truth may not.
    # Thus we left join to ensure the full keyset exists in the accuracy report.
    return noisy_df.join(ground_truth_df, on=join_columns, how="left").fillna(0)


def create_aggregated_error_report_h(
    single_run_paths: List[str],
    parameters_path: str,
    ground_truth_path: str,
    output_path: str,
):
    """Create an error report with population groups aggregated by size.

    Create an error report with population groups aggregated by size and grouped by
    iteration level, and geography level.

    Args:
        single_run_paths: A list of paths to directories containing single runs of
            SafeTab-p.
        parameters_path: The path containing the input files.
        ground_truth_path: The path containing the ground truth t3 and t4 counts.
        output_path: The path where the output file "multi_run_error_report.csv" is
            saved.
    """
    spark = SparkSession.builder.getOrCreate()
    dfs = []
    for table, schema in [("t3", T3_SCHEMA), ("t4", T4_SCHEMA)]:
        for output_dir in TABLE_TO_DIRS[table]:
            ground_truth = spark.read.csv(
                os.path.join(ground_truth_path, output_dir.relative_path),
                schema=schema,
                header=True,
                sep="|",
            )
            for trial, single_run_path in enumerate(single_run_paths):
                noisy = spark.read.csv(
                    os.path.join(single_run_path, output_dir.relative_path),
                    schema=schema,
                    header=True,
                    sep="|",
                )
                df = get_augmented_df_from_spark(noisy, ground_truth).toPandas()

                # Drop rows that are not directly computed.
                df = df.loc[
                    df[output_dir.data_cell_column].isin(
                        output_dir.directly_computed_cell_codes
                    )
                ]

                df["Table"] = table
                df["STAT_LEVEL"] = output_dir.stat_level
                df["Trial"] = trial
                dfs.append(df)

    error_report = pd.concat(dfs)

    # Compute the size of each population group
    error_report = (
        error_report.groupby(
            ["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "Trial", "Table"]
        )
        .agg({"GROUND_TRUTH": sum})
        .rename(columns={"GROUND_TRUTH": "Population group size"})
        .reset_index()
        .merge(error_report)
    )

    # Compute total absolute error in each workload by aggregating over each workload
    error_report["Error"] = abs(error_report["NOISY"] - error_report["GROUND_TRUTH"])

    # Load LEVEL info, and join with aggregated_error_report.
    ethnicity_iterations_df = read_csv(
        os.path.join(parameters_path, "ethnicity-characteristic-iterations.txt"),
        delimiter="|",
        dtype=str,
        usecols=["ITERATION_CODE", "LEVEL"],
    )
    race_iterations_df = read_csv(
        os.path.join(parameters_path, "race-characteristic-iterations.txt"),
        delimiter="|",
        dtype=str,
        usecols=["ITERATION_CODE", "LEVEL"],
    )
    iterations_df = pd.concat([ethnicity_iterations_df, race_iterations_df])
    error_report = error_report.merge(iterations_df, how="left")

    error_report = error_report.rename(columns={"LEVEL": "ITERATION_LEVEL"})

    # Bin population group sizes by powers of 10.
    max_pop_group_size = error_report["Population group size"].max()
    bins = [0] + [
        10 ** i for i in range(0, math.ceil(np.log10(max_pop_group_size)) + 1)
    ]
    error_report["Population group size"] = pd.cut(
        error_report["Population group size"], bins, include_lowest=True
    )

    # Group and calculate the margin of error (95%).
    compute_95_moe = functools.partial(np.quantile, q=0.95, interpolation="linear")

    # Group and calculate the margin of error (95%).
    grouped_error_report = error_report.groupby(
        [
            "REGION_TYPE",
            "ITERATION_LEVEL",
            "Table",
            "STAT_LEVEL",
            "Population group size",
        ]
    )
    aggregated_error_report = (
        grouped_error_report.agg({"Error": compute_95_moe})
        .rename(columns={"Error": "MOE"})
        .reset_index()
    )

    # Compute the number of population groups. Account for multiple trials, and sanity
    # check that we get the same stat level on each trial.
    aggregated_error_report = (
        grouped_error_report.size()
        .reset_index(name="Number of pop groups")
        .merge(aggregated_error_report)
    )
    num_trials = len(single_run_paths)
    aggregated_error_report["Number of pop groups"] /= num_trials
    num_pop_groups_as_int = aggregated_error_report["Number of pop groups"].astype(int)
    assert (
        aggregated_error_report["Number of pop groups"] == num_pop_groups_as_int
    ).all()
    aggregated_error_report["Number of pop groups"] = num_pop_groups_as_int

    # Remove rows without population groups.
    aggregated_error_report = aggregated_error_report.loc[
        aggregated_error_report["Number of pop groups"] != 0
    ]

    # Add expected MOEs for zcdp runs. Expected MOEs are not supported for puredp.
    with open(os.path.join(parameters_path, "config.json"), "r") as f:
        config_json: Dict[str, Any] = json.load(f)
    if config_json["privacy_defn"] == "zcdp":
        expected_moes = get_expected_moes(
            config_json, aggregated_error_report["REGION_TYPE"].unique()
        )
        aggregated_error_report = aggregated_error_report.merge(expected_moes)

    # Sort columns
    output_columns = (
        [
            "Table",
            "REGION_TYPE",
            "ITERATION_LEVEL",
            "Population group size",
            "STAT_LEVEL",
            "Number of pop groups",
            "Expected MOE",
            "MOE",
        ]
        if "Expected MOE" in aggregated_error_report.columns
        else [
            "Table",
            "REGION_TYPE",
            "ITERATION_LEVEL",
            "Population group size",
            "STAT_LEVEL",
            "Number of pop groups",
            "MOE",
        ]
    )

    aggregated_error_report = aggregated_error_report.loc[:, output_columns]

    # Convert imprecise intervals to precise intervals as strings.
    aggregated_error_report["Population group size"] = aggregated_error_report[
        "Population group size"
    ].astype(str)
    aggregated_error_report.loc[
        aggregated_error_report["Population group size"] == "(-0.001, 1.0]",
        "Population group size",
    ] = "[0.0, 1.0]"

    # Format floats.
    aggregated_error_report["MOE"] = aggregated_error_report["MOE"].map(
        lambda x: f"{x:.2E}" if 0 < x < 0.01 else f"{x:.2f}"
    )

    error_report_write_path = os.path.join(output_path, "multi_run_error_report.csv")
    if is_s3_path(error_report_write_path):
        with open(error_report_write_path, mode="w") as f:
            aggregated_error_report.to_csv(f, index=False)
    else:
        to_csv_with_create_dir(
            aggregated_error_report,
            os.path.join(output_path, "multi_run_error_report.csv"),
            index=False,
        )


def get_expected_moes(config: Dict[str, Any], region_types: List[str]) -> pd.DataFrame:
    """Get expect MOEs for each table and pop group level.

    Args:
        config: The SafeTab H config.json file containing the privacy budgets.
        region_types: The region types for get expected MOEs for.
    """
    # We don't use pure DP anymore. We can update this function with the formula for
    # pure dp if it is ever needed.
    if config["privacy_defn"] != "zcdp":
        raise NotImplementedError("Error report only implemented for zCDP.")

    rows = []
    for region_type, level, table in itertools.product(
        region_types, ["1", "2"], ["t3", "t4"]
    ):
        budget = config[
            f"privacy_budget_h_{table}_level_"
            f"{level}_{region_type.replace('-', '_').lower()}"
        ]
        if budget != 0:
            # Formula for expected MOE is 1.96 * sqrt(sensitivity / (2 * budget)).
            # Sensitivity for us is 9, since the iteration flatmap sensitivity is 9.
            expected_moe = 1.96 * np.sqrt(9 / (2 * budget))
            rows.append([region_type, level, table, expected_moe])

    return pd.DataFrame(
        rows, columns=["REGION_TYPE", "ITERATION_LEVEL", "Table", "Expected MOE"]
    )


def main():
    """Parse arguments and run the report."""
    parser = argparse.ArgumentParser(prog="SafeTab-H error report")
    parser.add_argument(
        dest="parameters_path",
        help="path to SafeTab-H config and iterations files",
        type=str,
    )
    parser.add_argument(dest="data_path", help="path to input CSV files", type=str)
    parser.add_argument(
        dest="output_path", help="path to write the output files to", type=str
    )
    parser.add_argument(
        "--trials", help="The number of noisy trials to run.", type=int, default=1
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as config_path:
        run_full_error_report_h(
            args.parameters_path,
            args.data_path,
            args.output_path,
            config_path,
            args.trials,
        )


if __name__ == "__main__":
    main()
