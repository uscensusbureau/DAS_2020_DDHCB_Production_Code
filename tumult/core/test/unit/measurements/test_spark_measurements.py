"""Unit tests for :mod:`~tmlt.core.measurements.spark_measurements`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

# pylint: disable=no-self-use
from fractions import Fraction
from typing import Dict, List

import numpy as np
import pandas as pd
import sympy as sp
from parameterized import parameterized
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from tmlt.core.domains.numpy_domains import NumpyIntegerDomain
from tmlt.core.domains.pandas_domains import PandasDataFrameDomain, PandasSeriesDomain
from tmlt.core.domains.spark_domains import (
    SparkColumnDescriptor,
    SparkDataFrameDomain,
    SparkFloatColumnDescriptor,
    SparkGroupedDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.noise_mechanisms import AddGeometricNoise, AddLaplaceNoise
from tmlt.core.measurements.pandas_measurements.dataframe import AggregateByColumn
from tmlt.core.measurements.pandas_measurements.series import (
    AddNoiseToSeries,
    NoisyQuantile,
)
from tmlt.core.measurements.spark_measurements import (
    AddNoiseToColumn,
    ApplyInPandas,
    BoundSelection,
    GeometricPartitionSelection,
    _get_materialized_df,
)
from tmlt.core.measures import ApproxDP, PureDP
from tmlt.core.metrics import SumOf, SymmetricDifference
from tmlt.core.utils.distributions import double_sided_geometric_cmf_exact
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.grouped_dataframe import GroupedDataFrame
from tmlt.core.utils.testing import (
    FakeAggregate,
    PySparkTest,
    assert_property_immutability,
    get_all_props,
)


class TestApplyInPandas(PySparkTest):
    """Tests for ApplyInPandas."""

    def setUp(self):
        """Setup."""
        self.aggregation_function = AggregateByColumn(
            input_domain=PandasDataFrameDomain(
                {"B": PandasSeriesDomain(NumpyIntegerDomain())}
            ),
            column_to_aggregation={
                "B": NoisyQuantile(
                    PandasSeriesDomain(NumpyIntegerDomain()),
                    output_measure=PureDP(),
                    quantile=0.5,
                    lower=22,
                    upper=29,
                    epsilon=sp.Integer(1),
                )
            },
        )
        self.domain = SparkGroupedDataFrameDomain(
            schema={
                "A": SparkStringColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
            },
            groupby_columns=["A"],
        )
        self.measurement = ApplyInPandas(
            input_domain=self.domain,
            input_metric=SumOf(SymmetricDifference()),
            aggregation_function=self.aggregation_function,
        )

    @parameterized.expand(get_all_props(ApplyInPandas))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.measurement, prop_name)

    def test_properties(self):
        """ApplyInPandas's properties have the expected values."""
        aggregation_function = FakeAggregate()
        input_domain = SparkGroupedDataFrameDomain(
            schema={
                "A": SparkStringColumnDescriptor(),
                "B": SparkFloatColumnDescriptor(allow_nan=True),
            },
            groupby_columns=["A"],
        )
        measurement = ApplyInPandas(
            input_domain=input_domain,
            input_metric=SumOf(SymmetricDifference()),
            aggregation_function=aggregation_function,
        )
        self.assertEqual(measurement.input_domain, input_domain)
        self.assertEqual(measurement.input_metric, SumOf(SymmetricDifference()))
        self.assertEqual(measurement.output_measure, PureDP())
        self.assertEqual(measurement.is_interactive, False)
        self.assertEqual(measurement.aggregation_function, aggregation_function)

    @parameterized.expand(
        [
            # test with one groupby column
            (
                {
                    "A": ["1", "2", "2", "3"],
                    "B": [1.0, 2.0, 1.0, np.nan],
                    "C": [np.nan] * 4,
                    "D": [np.nan] * 4,
                },
                {
                    "A": SparkStringColumnDescriptor(),
                    "B": SparkFloatColumnDescriptor(allow_nan=True),
                    "C": SparkFloatColumnDescriptor(allow_nan=True),
                    "D": SparkFloatColumnDescriptor(allow_nan=True),
                },
                {"A": ["1", "2", "3", "4"]},
                {
                    "A": ["1", "2", "3", "4"],
                    "C": [1.0, 3.0, None, -1.0],
                    "C_str": ["1.0", "3.0", "nan", "-1.0"],
                },
            ),
            # test with two groupby columns
            (
                {
                    "A_1": ["1", "2", "2", "3"],
                    "A_2": ["1", "2", "2", "1"],
                    "B": [1.0, 2.0, 1.0, np.nan],
                    "C": [np.nan] * 4,
                    "D": [np.nan] * 4,
                },
                {
                    "A_1": SparkStringColumnDescriptor(),
                    "A_2": SparkStringColumnDescriptor(),
                    "B": SparkFloatColumnDescriptor(allow_nan=True),
                    "C": SparkFloatColumnDescriptor(allow_nan=True),
                    "D": SparkFloatColumnDescriptor(allow_nan=True),
                },
                {"A_1": ["1", "1", "2", "2"], "A_2": ["1", "2", "1", "2"]},
                {
                    "A_1": ["1", "1", "2", "2"],
                    "A_2": ["1", "2", "1", "2"],
                    "C": [1.0, -1.0, -1.0, 3.0],
                    "C_str": ["1.0", "-1.0", "-1.0", "3.0"],
                },
            ),
        ]
    )
    def test_correctness_test_measure(
        self,
        df_dict: Dict[str, List],
        schema: Dict[str, SparkColumnDescriptor],
        groupby_domains: Dict[str, List],
        expected_dict: Dict[str, List],
    ):
        """Test correctness for a GroupByApplyInPandas aggregation."""
        group_keys = self.spark.createDataFrame(pd.DataFrame(groupby_domains))
        input_domain = SparkGroupedDataFrameDomain(
            schema=schema, groupby_columns=list(groupby_domains)
        )
        grouped_dataframe = GroupedDataFrame(
            dataframe=self.spark.createDataFrame(pd.DataFrame(df_dict)),
            group_keys=group_keys,
        )
        actual = ApplyInPandas(
            input_domain=input_domain,
            input_metric=SumOf(SymmetricDifference()),
            aggregation_function=FakeAggregate(),
        )(grouped_dataframe).toPandas()
        expected = pd.DataFrame(expected_dict)
        # It looks like python nans get converted to nulls when the return value
        # from a python udf gets converted back to spark land.
        self.assert_frame_equal_with_sort(actual, expected)

    def test_privacy_function_and_relation(self):
        """Test that the privacy function and relation are computed correctly."""

        quantile_measurement = NoisyQuantile(
            PandasSeriesDomain(NumpyIntegerDomain()),
            output_measure=PureDP(),
            quantile=0.5,
            lower=22,
            upper=29,
            epsilon=sp.Integer(2),
        )

        df_aggregation_function = AggregateByColumn(
            input_domain=PandasDataFrameDomain(
                {"Age": PandasSeriesDomain(NumpyIntegerDomain())}
            ),
            column_to_aggregation={"Age": quantile_measurement},
        )
        measurement = ApplyInPandas(
            input_domain=SparkGroupedDataFrameDomain(
                schema={
                    "Gender": SparkStringColumnDescriptor(),
                    "Age": SparkIntegerColumnDescriptor(),
                },
                groupby_columns=["Gender"],
            ),
            input_metric=SumOf(SymmetricDifference()),
            aggregation_function=df_aggregation_function,
        )

        self.assertTrue(measurement.privacy_function(sp.Integer(1)), sp.Integer(2))
        self.assertTrue(measurement.privacy_relation(sp.Integer(1), sp.Integer(2)))
        self.assertFalse(
            measurement.privacy_relation(sp.Integer(1), sp.Rational("1.99999"))
        )


class TestAddNoiseToColumn(PySparkTest):
    """Tests for AddNoiseToColumn.

    Tests :class:`~tmlt.core.measurements.spark_measurements.AddNoiseToColumn`.
    """

    def setUp(self):
        """Test Setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(),
                "count": SparkIntegerColumnDescriptor(),
            }
        )

    @parameterized.expand(get_all_props(AddNoiseToColumn))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        measurement = AddNoiseToColumn(
            input_domain=self.input_domain,
            measurement=AddNoiseToSeries(
                AddLaplaceNoise(input_domain=NumpyIntegerDomain(), scale=sp.Integer(1))
            ),
            measure_column="count",
        )
        assert_property_immutability(measurement, prop_name)

    def test_correctness(self):
        """Tests that AddNoiseToColumn works correctly."""
        expected = pd.DataFrame({"A": [0, 1, 2, 3], "count": [0, 1, 2, 3]})
        sdf = self.spark.createDataFrame(expected)
        measurement = AddNoiseToColumn(
            input_domain=self.input_domain,
            measurement=AddNoiseToSeries(AddGeometricNoise(alpha=0)),
            measure_column="count",
        )
        actual = measurement(sdf).toPandas()
        self.assert_frame_equal_with_sort(actual, expected)


class TestGeometricPartitionSelection(PySparkTest):
    """Tests for GeometricPartitionSelection.

    Tests
    :class:`~tmlt.core.measurements.spark_measurements.GeometricPartitionSelection`.
    """

    def setUp(self):
        """Test Setup."""
        self.input_domain = SparkDataFrameDomain(
            {"A": SparkStringColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
        )
        self.threshold = 5
        self.alpha = ExactNumber(3)
        self.count_column = "noisy counts"
        self.measurement = GeometricPartitionSelection(
            input_domain=self.input_domain,
            alpha=self.alpha,
            threshold=self.threshold,
            count_column=self.count_column,
        )

    @parameterized.expand(get_all_props(GeometricPartitionSelection))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.measurement, prop_name)

    def test_properties(self):
        """GeometricPartitionSelection has the expected properties."""
        self.assertEqual(self.measurement.input_domain, self.input_domain)
        self.assertEqual(self.measurement.input_metric, SymmetricDifference())
        self.assertEqual(self.measurement.output_measure, ApproxDP())
        self.assertEqual(self.measurement.alpha, self.alpha)
        self.assertEqual(self.measurement.threshold, self.threshold)
        self.assertEqual(self.measurement.count_column, self.count_column)

    def test_empty(self):
        """Tests that empty inputs/outputs don't cause any issues."""
        sdf = self.spark.createDataFrame(
            [],
            schema=StructType(
                [StructField("A", StringType()), StructField("B", IntegerType())]
            ),
        )
        expected = pd.DataFrame(
            {
                "A": pd.Series(dtype=str),
                "B": pd.Series(dtype=int),
                self.count_column: pd.Series(dtype=int),
            }
        )
        actual = self.measurement(sdf).toPandas()
        self.assert_frame_equal_with_sort(actual, expected)

    def test_negative_threshold(self):
        """Tests that negative thresholds don't cause any issues."""
        sdf = self.spark.createDataFrame(
            pd.DataFrame({"A": ["a1"] * 100, "B": [1] * 100})
        )
        measurement = GeometricPartitionSelection(
            input_domain=self.input_domain,
            alpha=1,
            threshold=-1,
            count_column=self.count_column,
        )
        actual = measurement(sdf).toPandas()
        expected_without_count = pd.DataFrame({"A": ["a1"], "B": [1]})
        self.assertIsInstance(actual, pd.DataFrame)
        assert isinstance(actual, pd.DataFrame)
        self.assert_frame_equal_with_sort(actual[["A", "B"]], expected_without_count)
        # Threshold -1 should give worse guarantee than for threshold of 0 or 1
        measurement_threshold_0 = GeometricPartitionSelection(
            input_domain=self.input_domain,
            alpha=1,
            threshold=0,
            count_column=self.count_column,
        )
        measurement_threshold_1 = GeometricPartitionSelection(
            input_domain=self.input_domain,
            alpha=1,
            threshold=1,
            count_column=self.count_column,
        )
        # Guarantee isn't infinitely bad
        self.assertFalse(
            ApproxDP().compare((sp.oo, 1), measurement.privacy_function(1))
        )
        # But is worse than for 0, which is worse than for 1
        self.assertFalse(
            ApproxDP().compare(
                measurement.privacy_function(1),
                measurement_threshold_0.privacy_function(1),
            )
        )
        self.assertFalse(
            ApproxDP().compare(
                measurement.privacy_function(1),
                measurement_threshold_1.privacy_function(1),
            )
        )
        self.assertFalse(
            ApproxDP().compare(
                measurement_threshold_0.privacy_function(1),
                measurement_threshold_1.privacy_function(1),
            )
        )

    def test_no_noise(self):
        """Tests that the no noise works correctly."""
        sdf = self.spark.createDataFrame(
            pd.DataFrame(
                {"A": ["a1", "a2", "a2", "a3", "a3", "a3"], "B": [1, 2, 2, 3, 3, 3]}
            )
        )
        expected = pd.DataFrame({"A": ["a2", "a3"], "B": [2, 3], "count": [2, 3]})
        measurement = GeometricPartitionSelection(
            input_domain=self.input_domain, alpha=0, threshold=2
        )
        actual = measurement(sdf).toPandas()
        self.assert_frame_equal_with_sort(actual, expected)

    def test_privacy_function(self):
        """GeometricPartitionSelection's privacy function is correct."""
        alpha = ExactNumber(3)
        threshold = 100
        measurement = GeometricPartitionSelection(
            input_domain=self.input_domain, alpha=alpha, threshold=threshold
        )
        self.assertEqual(measurement.privacy_function(0), (0, 0))
        base_epsilon = 1 / alpha
        base_delta = 1 - double_sided_geometric_cmf_exact(threshold - 2, alpha)
        self.assertEqual(measurement.privacy_function(1), (base_epsilon, base_delta))

        self.assertEqual(
            measurement.privacy_function(3),
            (
                3 * base_epsilon,
                3 * ExactNumber(sp.E) ** (3 * base_epsilon) * base_delta,
            ),
        )


class TestSanitization(PySparkTest):
    """Output DataFrames from Spark measurements are correctly sanitized."""

    @parameterized.expand(
        [
            (
                pd.DataFrame({"col1": [1, 2, 3], "col2": ["abc", "def", "ghi"]}),
                "simple_table",
            ),
            (
                pd.DataFrame(
                    {
                        "bad;column;name": ["a", "b", "c"],
                        "big_numbers": [
                            100000000000000,
                            100000000000000000,
                            99999999999999999,
                        ],
                    }
                ),
                "table_123456",
            ),
        ]
    )
    def test_get_materialized_df(self, df, table_name):
        """Tests that _get_materialized_df works correctly."""
        current_db = self.spark.catalog.currentDatabase()
        sdf = self.spark.createDataFrame(df)
        materialized_df = _get_materialized_df(sdf, table_name)
        self.assertEqual(current_db, self.spark.catalog.currentDatabase())
        self.assert_frame_equal_with_sort(materialized_df.toPandas(), df)

    def test_repartition_works_as_expected(self):
        """Tests that repartitioning randomly works as expected.

        Note: This is a sanity test that checks repartition by a random
        column works as expected regardless of the internal representation of
        the DataFrame being repartitioned. This does not test any unit
        in :mod:`~tmlt.core.measurements.spark_measurements`.
        """
        df = self.spark.createDataFrame(
            [(i, f"{j}") for i in range(10) for j in range(20)]
        )
        df = df.withColumn("partitioningColumn", sf.round(sf.rand() * 1000))
        # Random partitioning column
        partitions1 = df.repartition("partitioningColumn").rdd.glom().collect()
        df_shuffled = df.repartition(1000)
        partitions2 = df_shuffled.repartition("partitioningColumn").rdd.glom().collect()
        self.assertListEqual(partitions1, partitions2)


class TestBoundSelection(PySparkTest):
    """Tests for BoundSelection.

    Tests
    :class:`~tmlt.core.measurements.spark_measurements.BoundSelection`.
    """

    def setUp(self):
        """Test Setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(),
                "B": SparkIntegerColumnDescriptor(),
                "C": SparkFloatColumnDescriptor(),
            }
        )
        self.spark_schema = StructType(
            [
                StructField("A", StringType(), True),
                StructField("B", IntegerType(), True),
                StructField("C", DoubleType(), True),
            ]
        )
        self.output_measure = PureDP()
        self.bound_column = "B"
        self.alpha = ExactNumber(0)
        self.threshold = 0.8
        self.sdf = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "A": ["a1", "a2", "a2", "a3", "a3", "a3"],
                    "B": [1, 2, 2, 3, 3, 8],
                    "C": [1.0, 2.0, 2.0, 3.0, 3.0, 8.0],
                }
            )
        )
        self.expected_min, self.expected_max = -4.0, 4.0
        self.measurement = BoundSelection(
            input_domain=self.input_domain,
            bound_column=self.bound_column,
            alpha=self.alpha,
            threshold=self.threshold,
        )

    @parameterized.expand(get_all_props(BoundSelection))
    def test_property_immutability(self, prop_name: str):
        """Tests that given property is immutable."""
        assert_property_immutability(self.measurement, prop_name)

    def test_properties(self):
        """BoundSelection has the expected properties."""
        self.assertEqual(self.measurement.input_domain, self.input_domain)
        self.assertEqual(self.measurement.input_metric, SymmetricDifference())
        self.assertIsInstance(self.measurement.output_measure, PureDP)
        self.assertEqual(self.measurement.bound_column, self.bound_column)
        self.assertEqual(self.measurement.alpha, self.alpha)
        self.assertEqual(self.measurement.threshold, self.threshold)

    @parameterized.expand(
        [("B", SparkIntegerColumnDescriptor()), ("C", SparkFloatColumnDescriptor())]
    )
    def test_splits(self, bound_column: str, column_type: SparkColumnDescriptor):
        """BoundSelection creates the correct splits."""
        measurement = BoundSelection(
            input_domain=self.input_domain,
            bound_column=bound_column,
            alpha=self.alpha,
            threshold=self.threshold,
        )
        if isinstance(column_type, SparkFloatColumnDescriptor):
            expected_splits = (
                [float("-inf")]
                + [-(2**i) * 2**-100 for i in range(200, -1, -1)]
                + [0]
                + [2**i * 2**-100 for i in range(201)]
                + [float("inf")]
            )
        elif isinstance(column_type, SparkIntegerColumnDescriptor):
            expected_splits = (
                [-(2 ** (column_type.size - 1)) + 1]
                + [-(2**i) for i in range(column_type.size - 2, -1, -1)]
                + [0]
                + [2**i for i in range(column_type.size)]
            )
        self.assertEqual(measurement.splits, expected_splits)

    def test_empty(self):
        """Tests that empty inputs don't cause any issues."""
        sdf = self.spark.createDataFrame(
            pd.DataFrame({"A": [], "B": [], "C": []}), schema=self.spark_schema
        )
        expected_min, expected_max = -1.0, 1.0
        actual_min, actual_max = self.measurement(sdf)
        self.assertEqual(actual_min, expected_min)
        self.assertEqual(actual_max, expected_max)

    def test_negative_threshold(self):
        """Tests that negative thresholds aren't allowed."""
        with self.assertRaises(ValueError):
            BoundSelection(
                input_domain=self.input_domain,
                bound_column=self.bound_column,
                alpha=self.alpha,
                threshold=-0.95,
            )

    def test_no_noise(self):
        """Tests that the no noise works correctly."""
        actual_min, actual_max = self.measurement(self.sdf)
        self.assertEqual(actual_min, self.expected_min)
        self.assertEqual(actual_max, self.expected_max)

    @parameterized.expand(
        [
            # Tightly clustered
            # Binning the bounds work on a [x, y) interval,
            # hence, why the bounds are at 32,
            # as 16 would be found in the [16, 32) bound
            ([16] * 10, 0.95, -32, 32),
            # Loosely clustered
            ([-50, -30, 0, 10, 30, 30, 30, 50, 60, 70], 0.9, -64, 64),
            # Tightly clustered, but with a few outliers
            ([16] * 15 + [500] * 5, 0.95, -512, 512),
            # Tightly clustered, with little to no outliers
            ([-777] * 10, 0.95, -1024, 1024),
            # First bin
            ([-1] * 2 + [0] * 16 + [1] * 2, 0.8, -1, 1),
            # Second bin
            ([-1] * 3 + [0] * 4 + [1] * 3, 0.95, -2, 2),
            # Empty DataFrame
            ([], 0.95, -1.0, 1.0),
        ]
    )
    def test_different_clusterings(
        self,
        column_values: List[int],
        threshold: float,
        expected_min: int,
        expected_max: int,
    ):
        """Test different clustering cases for integer columned BoundSelection."""
        df = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "A": ["a" for _ in column_values],
                    "B": list(column_values),
                    "C": [1.0 for _ in column_values],
                }
            ),
            schema=self.spark_schema,
        )
        measurement = BoundSelection(
            input_domain=self.input_domain,
            bound_column="B",
            alpha=0,
            threshold=threshold,
        )
        actual_min, actual_max = measurement(df)
        self.assertEqual(actual_min, expected_min)
        self.assertEqual(actual_max, expected_max)

    @parameterized.expand(
        [
            # Tightly clustered
            # Binning the bounds work on a [x, y) interval,
            # hence, why the bounds are at 32,
            # as 16 would be found in the [16, 32) bound
            ([16.0] * 10, 0.95, -32.0, 32.0),
            # Loosely clustered
            (
                [-50.0, -30.0, 0.0, 10.0, 30.0, 30.0, 30.0, 50.0, 60.0, 70.0],
                0.9,
                -64.0,
                64.0,
            ),
            # Tightly clustered, but with a few outliers
            ([16.0] * 15 + [500.0] * 5, 0.95, -512.0, 512.0),
            # Tightly clustered, with little to no outliers
            ([-777.0] * 10, 0.95, -1024.0, 1024.0),
            # First bin
            (
                [-(2**-150)] * 2 + [0.0] * 16 + [2**-150] * 2,
                0.8,
                -(2**-100),
                2**-100,
            ),
            # Second bin
            (
                [-(2**-99.5)] * 8 + [0.0] * 10 + [2**-99.5] * 8,
                0.95,
                -(2**-99),
                2**-99,
            ),
            # Empty DataFrame
            ([], 0.95, -(2**-100), 2**-100),
            # Beyond largest split
            ([2**101] * 10, 0.95, -(2**-100), 2**-100),
        ]
    )
    def test_floats(
        self,
        column_values: List[float],
        threshold: float,
        expected_min: float,
        expected_max: float,
    ):
        """Test different clustering cases for float columned BoundSelection."""
        df = self.spark.createDataFrame(
            pd.DataFrame(
                {
                    "A": ["a" for _ in column_values],
                    "B": [1 for _ in column_values],
                    "C": [float(v) for v in column_values],
                }
            ),
            schema=self.spark_schema,
        )
        measurement = BoundSelection(
            input_domain=self.input_domain,
            bound_column="C",
            alpha=0,
            threshold=threshold,
        )
        actual_min, actual_max = measurement(df)
        self.assertEqual(actual_min, expected_min)
        self.assertEqual(actual_max, expected_max)

    def test_privacy_function(self):
        """Test that BoundSelection's privacy function is correct."""
        alpha = ExactNumber(3)
        threshold = 1
        measurement = BoundSelection(
            input_domain=self.input_domain,
            bound_column="B",
            alpha=alpha,
            threshold=threshold,
        )
        self.assertEqual(measurement.privacy_function(0), 0)
        self.assertEqual(measurement.privacy_function(1), ExactNumber(Fraction(4, 3)))
        self.assertEqual(measurement.privacy_function(3), 4)
