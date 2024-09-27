"""Unit tests for :mod:`~tmlt.core.measurements.aggregations`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023
import functools
import unittest
from typing import Callable, List, Optional, Tuple, Union, cast

import sympy as sp
from parameterized import parameterized, parameterized_class
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, StructField, StructType

from tmlt.core.domains.spark_domains import (
    SparkDataFrameDomain,
    SparkIntegerColumnDescriptor,
    SparkStringColumnDescriptor,
)
from tmlt.core.measurements.aggregations import (
    NoiseMechanism,
    create_average_measurement,
    create_bound_selection_measurement,
    create_count_distinct_measurement,
    create_count_measurement,
    create_partition_selection_measurement,
    create_quantile_measurement,
    create_standard_deviation_measurement,
    create_sum_measurement,
    create_variance_measurement,
)
from tmlt.core.measurements.converters import PureDPToApproxDP, PureDPToRhoZCDP
from tmlt.core.measurements.spark_measurements import BoundSelection
from tmlt.core.measures import (
    ApproxDP,
    ApproxDPBudget,
    PrivacyBudget,
    PrivacyBudgetInput,
    PureDP,
    RhoZCDP,
)
from tmlt.core.metrics import (
    HammingDistance,
    IfGroupedBy,
    RootSumOfSquared,
    SumOf,
    SymmetricDifference,
)
from tmlt.core.transformations.spark_transformations.groupby import GroupBy
from tmlt.core.utils.distributions import double_sided_geometric_cmf_exact
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.utils.testing import PySparkTest


@parameterized_class(
    [
        {"group_keys_list": [], "struct_fields": [], "groupby_columns": []},
        {
            "group_keys_list": [("x1",), ("x2",), ("x3",), (None,)],
            "struct_fields": [StructField("A", StringType())],
            "groupby_columns": ["A"],
        },
    ]
)
class TestGroupByAggregationMeasurements(PySparkTest):
    """Tests for :mod:`tmlt.core.measurements.aggregations`."""

    group_keys_list: List[Tuple[str, ...]]
    struct_fields: List[StructField]
    groupby_columns: List[str]

    def setUp(self):
        """Test setup."""
        self.input_domain = SparkDataFrameDomain(
            {
                "A": SparkStringColumnDescriptor(allow_null=True),
                "B": SparkIntegerColumnDescriptor(),
            }
        )
        self.group_keys = self.spark.createDataFrame(
            self.group_keys_list, schema=StructType(self.struct_fields.copy())
        )
        self.sdf = self.spark.createDataFrame(
            [("x1", 2), ("x1", 2), ("x2", 4)], schema=["A", "B"]
        )

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                d_out,
                noise_mechanism,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    "A", cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(2)),
                (RhoZCDP(), sp.Integer(2)),
                (ApproxDP(), (sp.Integer(2), sp.Integer(0))),
            ]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_count_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP, ApproxDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_count_measurement works correctly with groupby."""
        if self.groupby_columns == [] and isinstance(input_metric, IfGroupedBy):
            return
        count_measurement = create_count_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            count_column="test_count",
        )
        self.assertEqual(count_measurement.input_domain, self.input_domain)
        self.assertEqual(count_measurement.output_measure, output_measure)
        self.assertEqual(count_measurement.privacy_function(sp.Integer(1)), d_out)
        answer = count_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        self.assertEqual(answer.columns, self.groupby_columns + ["test_count"])

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                d_out,
                noise_mechanism,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    "A", cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(2)),
                (RhoZCDP(), sp.Integer(2)),
                (ApproxDP(), (sp.Integer(2), sp.Integer(0))),
            ]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_count_distinct_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP, ApproxDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_count_distinct_measurement works correctly with groupby."""
        if self.groupby_columns == [] and isinstance(input_metric, IfGroupedBy):
            return
        count_distinct_measurement = create_count_distinct_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            count_column="test_count",
        )
        self.assertEqual(count_distinct_measurement.input_domain, self.input_domain)
        self.assertEqual(count_distinct_measurement.output_measure, output_measure)
        self.assertEqual(
            count_distinct_measurement.privacy_function(sp.Integer(1)), d_out
        )
        answer = count_distinct_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        self.assertEqual(answer.columns, self.groupby_columns + ["test_count"])

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                d_out,
                noise_mechanism,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    "A", cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_sum_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_sum_measurement works correctly with groupby."""
        if self.groupby_columns == [] and isinstance(input_metric, IfGroupedBy):
            return
        sum_measurement = create_sum_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            sum_column="sumB",
        )
        self.assertEqual(sum_measurement.input_domain, self.input_domain)
        self.assertEqual(sum_measurement.output_measure, output_measure)
        self.assertEqual(sum_measurement.privacy_function(sp.Integer(1)), d_out)
        answer = sum_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        self.assertEqual(answer.columns, self.groupby_columns + ["sumB"])

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                d_out,
                noise_mechanism,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    "A", cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_average_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP, ApproxDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_average_measurement works correctly with groupby."""
        if self.groupby_columns == [] and isinstance(input_metric, IfGroupedBy):
            return
        average_measurement = create_average_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            average_column="AVG(B)",
        )
        self.assertEqual(average_measurement.input_domain, self.input_domain)
        self.assertEqual(average_measurement.output_measure, output_measure)
        self.assertEqual(average_measurement.privacy_function(sp.Integer(1)), d_out)
        answer = average_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        self.assertEqual(answer.columns, self.groupby_columns + ["AVG(B)"])

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                noise_mechanism,
                d_out,
                output_column,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    "A", cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            for output_column in ["XYZ", None]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_standard_deviation_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP],
        noise_mechanism: NoiseMechanism,
        d_out: PrivacyBudgetInput,
        output_column: Optional[str] = None,
    ):
        """Tests that create_standard_deviation_measurement works correctly."""
        if self.groupby_columns == [] and isinstance(input_metric, IfGroupedBy):
            return
        standard_deviation_measurement = create_standard_deviation_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            keep_intermediates=False,
            standard_deviation_column=output_column,
        )
        self.assertEqual(standard_deviation_measurement.input_domain, self.input_domain)
        self.assertEqual(standard_deviation_measurement.output_measure, output_measure)
        self.assertEqual(
            standard_deviation_measurement.privacy_function(sp.Integer(1)), d_out
        )
        answer = standard_deviation_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        if not output_column:
            output_column = "stddev(B)"
        self.assertEqual(answer.columns, self.groupby_columns + [output_column])
        answer.first()

    @parameterized.expand(
        [
            (
                input_metric,
                groupby_output_metric,
                output_measure,
                noise_mechanism,
                d_out,
                output_column,
            )
            for noise_mechanism, groupby_output_metric in [
                (NoiseMechanism.LAPLACE, SumOf(SymmetricDifference())),
                (NoiseMechanism.GEOMETRIC, SumOf(SymmetricDifference())),
                (
                    NoiseMechanism.DISCRETE_GAUSSIAN,
                    RootSumOfSquared(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    "A", cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            for output_column in ["XYZ", None]
            if not (
                noise_mechanism == NoiseMechanism.DISCRETE_GAUSSIAN
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_variance_measurement_with_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance, IfGroupedBy],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        output_measure: Union[PureDP, RhoZCDP],
        noise_mechanism: NoiseMechanism,
        d_out: PrivacyBudgetInput,
        output_column: Optional[str] = None,
    ):
        """Tests that create_variance_measurement works correctly with groupby."""
        if self.groupby_columns == [] and isinstance(input_metric, IfGroupedBy):
            return
        variance_measurement = create_variance_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            keep_intermediates=False,
            variance_column=output_column,
        )
        self.assertEqual(variance_measurement.input_domain, self.input_domain)
        self.assertEqual(variance_measurement.output_measure, output_measure)
        self.assertEqual(variance_measurement.privacy_function(sp.Integer(1)), d_out)
        answer = variance_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        if not output_column:
            output_column = "var(B)"
        self.assertEqual(answer.columns, self.groupby_columns + [output_column])
        answer.first()

    @parameterized.expand(
        [
            (input_metric, groupby_output_metric, d_out, output_measure)
            for output_measure, d_out, groupby_output_metric in [
                (PureDP(), sp.Integer(4), SumOf(SymmetricDifference())),
                (RhoZCDP(), sp.Integer(4), RootSumOfSquared(SymmetricDifference())),
                (
                    ApproxDP(),
                    (sp.Integer(4), sp.Integer(0)),
                    SumOf(SymmetricDifference()),
                ),
            ]
            for input_metric in [
                SymmetricDifference(),
                HammingDistance(),
                IfGroupedBy(
                    "A", cast(Union[SumOf, RootSumOfSquared], groupby_output_metric)
                ),
            ]
        ]
    )
    def test_create_quantile_measurement_with_groupby(
        self,
        input_metric: Union[HammingDistance, SymmetricDifference],
        groupby_output_metric: Union[SumOf, RootSumOfSquared],
        d_out: PrivacyBudgetInput,
        output_measure: Union[PureDP, RhoZCDP],
    ):
        """Tests that create_quantile_measurement works correctly with groupby."""
        if self.groupby_columns == [] and isinstance(input_metric, IfGroupedBy):
            return
        quantile_measurement = create_quantile_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="B",
            quantile=0.5,
            upper=10,
            lower=0,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=GroupBy(
                input_domain=self.input_domain,
                input_metric=input_metric,
                use_l2=isinstance(groupby_output_metric, RootSumOfSquared),
                group_keys=self.group_keys,
            ),
            quantile_column="MEDIAN(B)",
        )
        self.assertEqual(quantile_measurement.input_domain, self.input_domain)
        self.assertEqual(quantile_measurement.input_metric, input_metric)
        self.assertEqual(quantile_measurement.output_measure, output_measure)
        self.assertEqual(quantile_measurement.privacy_function(sp.Integer(1)), d_out)
        answer = quantile_measurement(self.sdf)
        self.assertIsInstance(answer, DataFrame)
        self.assertEqual(answer.columns, self.groupby_columns + ["MEDIAN(B)"])
        df = answer.toPandas()
        self.assertTrue(((df["MEDIAN(B)"] <= 10) & (df["MEDIAN(B)"] >= 0)).all())


class TestAggregationMeasurement(PySparkTest):
    """Tests for :mod:`tmlt.core.measurements.aggregations`."""

    def setUp(self):
        """Test setup."""
        self.input_domain = SparkDataFrameDomain(
            {"A": SparkStringColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
        )
        self.sdf = self.spark.createDataFrame([("x1", 2), ("x2", 4)], schema=["A", "B"])

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(2)),
                (RhoZCDP(), sp.Integer(2)),
                (ApproxDP(), (sp.Integer(2), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_count_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_count_measurement works correctly without groupby."""
        count_measurement = create_count_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            output_measure=output_measure,
        )
        self.assertEqual(count_measurement.input_domain, self.input_domain)
        self.assertEqual(count_measurement.input_metric, input_metric)
        self.assertEqual(count_measurement.output_measure, output_measure)
        self.assertEqual(count_measurement.privacy_function(1), d_out)
        answer = count_measurement(self.sdf)
        self.assertIsInstance(answer, (float, int))

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(2)),
                (RhoZCDP(), sp.Integer(2)),
                (ApproxDP(), (sp.Integer(2), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_count_distinct_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests create_count_distinct_measurement without groupby."""
        count_distinct_measurement = create_count_distinct_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            output_measure=output_measure,
        )

        self.assertEqual(count_distinct_measurement.input_domain, self.input_domain)
        self.assertEqual(count_distinct_measurement.input_metric, input_metric)
        self.assertEqual(count_distinct_measurement.output_measure, output_measure)
        self.assertEqual(count_distinct_measurement.privacy_function(1), d_out)
        answer = count_distinct_measurement(self.sdf)
        self.assertIsInstance(answer, (int, float))

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_sum_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_sum_measurement works correctly without groupby."""
        sum_measurement = create_sum_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            output_measure=output_measure,
        )

        self.assertEqual(sum_measurement.input_domain, self.input_domain)
        self.assertEqual(sum_measurement.input_metric, input_metric)
        self.assertEqual(sum_measurement.output_measure, output_measure)
        self.assertEqual(sum_measurement.privacy_function(1), d_out)
        answer = sum_measurement(self.sdf)
        self.assertIsInstance(answer, (float, int))

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_average_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_average_measurement works correctly without groupby."""
        average_measurement = create_average_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            keep_intermediates=False,
            output_measure=output_measure,
        )

        self.assertEqual(average_measurement.input_domain, self.input_domain)
        self.assertEqual(average_measurement.input_metric, input_metric)
        self.assertEqual(average_measurement.output_measure, output_measure)
        self.assertEqual(average_measurement.privacy_function(1), d_out)
        answer = average_measurement(self.sdf)
        self.assertIsInstance(answer, (float, int))

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_standard_deviation_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_standard_deviation_measurement works correctly."""
        standard_deviation_measurement = create_standard_deviation_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            keep_intermediates=False,
            output_measure=output_measure,
        )

        self.assertEqual(standard_deviation_measurement.input_domain, self.input_domain)
        self.assertEqual(standard_deviation_measurement.input_metric, input_metric)
        self.assertEqual(standard_deviation_measurement.output_measure, output_measure)
        self.assertEqual(standard_deviation_measurement.privacy_function(1), d_out)
        answer = standard_deviation_measurement(self.sdf)
        self.assertIsInstance(answer, float)

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out, noise_mechanism)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for noise_mechanism in [
                NoiseMechanism.LAPLACE,
                NoiseMechanism.GEOMETRIC,
                NoiseMechanism.DISCRETE_GAUSSIAN,
                NoiseMechanism.GAUSSIAN,
            ]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
            if not (
                noise_mechanism
                in [NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN]
                and output_measure != RhoZCDP()
            )
        ]
    )
    def test_create_variance_measurement_without_groupby(
        self,
        input_metric: Union[SymmetricDifference, HammingDistance],
        output_measure: Union[PureDP, RhoZCDP],
        d_out: PrivacyBudgetInput,
        noise_mechanism: NoiseMechanism,
    ):
        """Tests that create_variance_measurement works correctly without groupby."""
        variance_measurement = create_variance_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            measure_column="B",
            upper=sp.Integer(10),
            lower=sp.Integer(0),
            noise_mechanism=noise_mechanism,
            d_in=sp.Integer(1),
            d_out=d_out,
            keep_intermediates=False,
            output_measure=output_measure,
        )

        self.assertEqual(variance_measurement.input_domain, self.input_domain)
        self.assertEqual(variance_measurement.input_metric, input_metric)
        self.assertEqual(variance_measurement.output_measure, output_measure)
        self.assertEqual(variance_measurement.privacy_function(1), d_out)
        answer = variance_measurement(self.sdf)
        self.assertIsInstance(answer, (int, float))

    @parameterized.expand(
        [
            (input_metric, output_measure, d_out)
            for input_metric in [SymmetricDifference(), HammingDistance()]
            for output_measure, d_out in [
                (PureDP(), sp.Integer(4)),
                (RhoZCDP(), sp.Integer(4)),
                (ApproxDP(), (sp.Integer(4), sp.Integer(0))),
            ]
        ]
    )
    def test_create_quantile_measurement_without_groupby(
        self,
        input_metric: Union[HammingDistance, SymmetricDifference],
        output_measure: Union[PureDP, RhoZCDP, ApproxDP],
        d_out: PrivacyBudgetInput,
    ):
        """Tests that create_quantile_measurement works correctly without groupby."""
        quantile_measurement = create_quantile_measurement(
            input_domain=self.input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            measure_column="B",
            quantile=0.5,
            upper=10,
            lower=0,
            d_in=sp.Integer(1),
            d_out=d_out,
            groupby_transformation=None,
            quantile_column="MEDIAN(B)",
        )
        self.assertEqual(quantile_measurement.input_domain, self.input_domain)
        self.assertEqual(quantile_measurement.input_metric, input_metric)
        self.assertEqual(quantile_measurement.output_measure, output_measure)
        self.assertEqual(quantile_measurement.privacy_function(1), d_out)
        answer = quantile_measurement(self.sdf)
        self.assertIsInstance(answer, float)
        self.assertLessEqual(answer, 10)
        self.assertGreaterEqual(answer, 0)

    @parameterized.expand(
        [
            (float("inf"), 0, 1, 0, 0, None),
            # Test with alternate definition of infinite budget.
            (1, 1, 1, 0, 0, None),
            # Test large value of epsilon succeeds without error.
            (
                10000,
                1
                - double_sided_geometric_cmf_exact(
                    5 - 2, ExactNumber(1) / ExactNumber(10000)
                ),
                1,
                ExactNumber(1) / ExactNumber(10000),
                5,
                None,
            ),
            (
                ExactNumber(1) / ExactNumber(3),
                1 - double_sided_geometric_cmf_exact(7 - 2, 3),
                1,
                3,
                7,
                None,
            ),
            (
                ExactNumber(1) / ExactNumber(17),
                1 - double_sided_geometric_cmf_exact(10 - 2, 17),
                1,
                17,
                10,
                None,
            ),
            (
                ExactNumber(2) / ExactNumber(13),
                2
                * ExactNumber(sp.E) ** (ExactNumber(2) / ExactNumber(13))
                * (1 - double_sided_geometric_cmf_exact(50 - 2, 13)),
                2,
                13,
                50,
                "my_count_column",
            ),
        ]
    )
    def test_create_partition_selection_measurement(
        self,
        epsilon: ExactNumberInput,
        delta: ExactNumberInput,
        d_in: ExactNumberInput,
        expected_alpha: ExactNumberInput,
        expected_threshold: ExactNumberInput,
        count_column: Optional[str] = None,
    ) -> None:
        """Test create_partition_selection_measurement works correctly."""
        measurement = create_partition_selection_measurement(
            input_domain=self.input_domain,
            epsilon=epsilon,
            delta=delta,
            d_in=d_in,
            count_column=count_column,
        )
        self.assertEqual(measurement.alpha, expected_alpha)
        self.assertEqual(measurement.threshold, expected_threshold)
        self.assertEqual(measurement.input_domain, self.input_domain)
        if count_column is not None:
            self.assertEqual(measurement.count_column, count_column)
        # Check that measurement.privacy_function(d_in) = (epsilon, delta)
        measurement_epsilon, measurement_delta = measurement.privacy_function(d_in)
        if ApproxDPBudget((epsilon, delta)).is_finite():
            self.assertEqual(measurement_epsilon, epsilon)
            self.assertEqual(measurement_delta, delta)
        else:
            self.assertFalse(
                ApproxDPBudget((measurement_epsilon, measurement_delta)).is_finite()
            )

    @parameterized.expand(
        [
            (PureDP(), 1, "B", 0.7, 1),
            (RhoZCDP(), 1, "B", 0.7, 1),
            (ApproxDP(), (1, 0), "B", 0.7, 1),
        ]
    )
    def test_create_bound_selection_measurement(
        self,
        output_measure: Union[PureDP, RhoZCDP, ApproxDP],
        d_out: PrivacyBudgetInput,
        bound_column: str,
        threshold: float,
        d_in: ExactNumberInput = 1,
    ):
        """Test create_bound_selection_measurement works correctly."""
        measurement = create_bound_selection_measurement(
            input_domain=self.input_domain,
            output_measure=output_measure,
            d_out=d_out,
            bound_column=bound_column,
            threshold=threshold,
            d_in=d_in,
        )
        d_out = PrivacyBudget.cast(output_measure, d_out).value
        if isinstance(measurement, PureDPToRhoZCDP):
            measurement = measurement.pure_dp_measurement
            epsilon = sp.sqrt(ExactNumber(sp.Integer(2) * d_out).expr)
        elif isinstance(measurement, PureDPToApproxDP):
            measurement = measurement.pure_dp_measurement
            assert isinstance(d_out, tuple)
            epsilon = d_out[0]
        else:
            epsilon = d_out
        # Appease mypy
        if not isinstance(measurement, BoundSelection):
            raise TypeError(
                f"Expected measurement to be a BoundSelection, got {measurement}"
            )
        d_in = ExactNumber(d_in)
        self.assertEqual(measurement.input_domain, self.input_domain)
        self.assertEqual(measurement.output_measure, PureDP())
        self.assertEqual(measurement.privacy_function(d_in), epsilon)
        if d_out == float("inf"):
            expected_alpha = ExactNumber(0)
        else:
            expected_alpha = (4 / epsilon) * d_in
        self.assertEqual(measurement.alpha, expected_alpha)
        self.assertEqual(measurement.bound_column, bound_column)
        self.assertEqual(measurement.threshold, threshold)


INPUT_DOMAIN = SparkDataFrameDomain(
    {"A": SparkStringColumnDescriptor(), "B": SparkIntegerColumnDescriptor()}
)


class TestBadDelta(unittest.TestCase):
    """Tests for :mod:`tmlt.core.measurements.aggregations`."""

    @parameterized.expand(
        [
            (noise_mechanism, d_out, f)
            for noise_mechanism, d_out in [
                (NoiseMechanism.LAPLACE, (sp.Integer(1), sp.Rational(1, 2))),
                (NoiseMechanism.GEOMETRIC, (sp.Integer(1), sp.Rational(1, 2))),
                (NoiseMechanism.GAUSSIAN, (sp.Integer(1), sp.Rational(1, 2))),
                (NoiseMechanism.DISCRETE_GAUSSIAN, (sp.Integer(1), sp.Rational(1, 2))),
                (NoiseMechanism.GAUSSIAN, (sp.Integer(1), sp.Integer(0))),
                (NoiseMechanism.DISCRETE_GAUSSIAN, (sp.Integer(1), sp.Integer(0))),
            ]
            for f in [
                functools.partial(
                    create_count_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_count_distinct_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_sum_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    measure_column="B",
                    upper=sp.Integer(10),
                    lower=sp.Integer(0),
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_average_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    measure_column="B",
                    upper=sp.Integer(10),
                    lower=sp.Integer(0),
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_standard_deviation_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    measure_column="B",
                    upper=sp.Integer(10),
                    lower=sp.Integer(0),
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_variance_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    measure_column="B",
                    upper=sp.Integer(10),
                    lower=sp.Integer(0),
                    output_measure=ApproxDP(),
                ),
            ]
        ]
    )
    def test_functions_with_noise_mechanism(
        self, noise_mechanism: NoiseMechanism, d_out: PrivacyBudgetInput, f: Callable
    ) -> None:
        """Test error is raised for invalid delta/noise mechanism combination."""
        with self.assertRaises(ValueError):
            f(noise_mechanism=noise_mechanism, d_out=d_out)

    @parameterized.expand(
        [
            (d_out, f)
            for d_out in [
                (sp.Integer(1), sp.Rational(1, 2)),
                (sp.Integer(1), sp.Rational(1, 3)),
            ]
            for f in [
                functools.partial(
                    create_bound_selection_measurement,
                    input_domain=INPUT_DOMAIN,
                    bound_column="B",
                    threshold=0.5,
                    output_measure=ApproxDP(),
                ),
                functools.partial(
                    create_quantile_measurement,
                    input_domain=INPUT_DOMAIN,
                    input_metric=SymmetricDifference(),
                    measure_column="B",
                    quantile=0.5,
                    upper=10,
                    lower=0,
                    output_measure=ApproxDP(),
                ),
            ]
        ]
    )
    def test_functions_without_noise_mechanism(
        self, d_out: PrivacyBudgetInput, f: Callable
    ) -> None:
        """Test error is raised for invalid deltas."""
        with self.assertRaises(ValueError):
            f(d_out=d_out)
