"""Tests that Sum measurement adds noise sampled from the correct distributions."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

# pylint: disable=no-member, no-self-use

from typing import Dict, Union

import pytest

from tmlt.core.measurements.aggregations import NoiseMechanism, create_sum_measurement
from tmlt.core.measures import PureDP, RhoZCDP
from tmlt.core.metrics import SymmetricDifference
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.testing import (
    ChiSquaredTestCase,
    FixedGroupDataSet,
    KSTestCase,
    PySparkTest,
    get_noise_scales,
    get_prob_functions,
    get_sampler,
    get_values_summing_to_loc,
    run_test_using_chi_squared_test,
    run_test_using_ks_test,
)

from . import NOISE_SCALE_FUDGE_FACTOR, P_THRESHOLD, SAMPLE_SIZE


def _get_sum_test_cases(noise_mechanism: NoiseMechanism):
    """Returns sum test cases.

    This returns a list of 4 test cases specifying the sampler (that produces
    a sum sample), expected sum location, expected noise scale and corresponding
    cdf (if noise mechanism is Laplace) or cmf and pmf (if noise mechanism is not
    Laplace).

    Each of the 4 samplers produces a sample of size SAMPLE_SIZE.
      * 2 samplers that compute noisy groupby-sum once on a DataFrame with
         # groups = SAMPLE_SIZE. These two samplers have different true sums
         and different noise scales.

      * 2 samplers that compute noisy groupby-sum 200 times on a DataFrame with
         # groups = SAMPLE_SIZE/200. These two samplers have different true sums
         and different noise scales.

    """
    test_cases = []
    supports_continuous = noise_mechanism in (
        NoiseMechanism.LAPLACE,
        NoiseMechanism.GAUSSIAN,
    )
    sum_locations = [3.5, 111.3] if supports_continuous else [3, 111]
    privacy_budgets = ["3.3", "0.11"]
    for sum_loc, budget in zip(sum_locations, privacy_budgets):
        group_values = get_values_summing_to_loc(sum_loc, n=3)  # Fixed group size of 3
        dataset = FixedGroupDataSet(
            group_vals=group_values,
            num_groups=SAMPLE_SIZE,
            float_measure_column=supports_continuous,
        )

        true_answers: Dict[str, Union[float, int]] = {"sum": sum(dataset.group_vals)}
        measurement = create_sum_measurement(
            input_domain=dataset.domain,
            input_metric=SymmetricDifference(),
            measure_column="B",
            output_measure=PureDP()
            if noise_mechanism
            not in (NoiseMechanism.DISCRETE_GAUSSIAN, NoiseMechanism.GAUSSIAN)
            else RhoZCDP(),
            lower=ExactNumber.from_float(min(group_values), round_up=False),
            upper=ExactNumber.from_float(max(group_values), round_up=True),
            noise_mechanism=noise_mechanism,
            d_out=budget,
            groupby_transformation=dataset.groupby(noise_mechanism),
            sum_column="sum",
        )
        print(measurement)
        sampler = get_sampler(measurement, dataset, lambda df: df.select("sum"))
        noise_scales = get_noise_scales(
            agg="sum", budget=budget, dataset=dataset, noise_mechanism=noise_mechanism
        )
        prob_functions = get_prob_functions(noise_mechanism, true_answers)
        test_cases.append(
            {
                "sampler": sampler,
                "locations": true_answers,
                "scales": noise_scales,
                **prob_functions,
            }
        )

    return test_cases


class TestUsingKSTest(PySparkTest):
    """Distribution tests for create_sum_measurement."""

    @pytest.mark.slow
    def test_sum_with_laplace_noise(self):
        """`create_sum_measurement` has expected geometric distribution."""
        cases = [
            KSTestCase.from_dict(e) for e in _get_sum_test_cases(NoiseMechanism.LAPLACE)
        ]
        for case in cases:
            run_test_using_ks_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @pytest.mark.slow
    def test_sum_with_geometric_noise(self):
        """`create_sum_measurement` has expected discrete Gaussian distribution."""
        cases = [
            ChiSquaredTestCase.from_dict(e)
            for e in _get_sum_test_cases(NoiseMechanism.GEOMETRIC)
        ]
        for case in cases:
            run_test_using_chi_squared_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @pytest.mark.slow
    def test_sum_with_discrete_gaussian_noise(self):
        """`create_sum_measurement` has expected Laplace distribution."""
        cases = [
            ChiSquaredTestCase.from_dict(e)
            for e in _get_sum_test_cases(NoiseMechanism.DISCRETE_GAUSSIAN)
        ]
        for case in cases:
            run_test_using_chi_squared_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)

    @pytest.mark.slow
    def test_sum_with_gaussian_noise(self):
        """`create_sum_measurement` has expected Gaussian distribution."""
        cases = [
            KSTestCase.from_dict(e)
            for e in _get_sum_test_cases(NoiseMechanism.GAUSSIAN)
        ]
        for case in cases:
            run_test_using_ks_test(case, P_THRESHOLD, NOISE_SCALE_FUDGE_FACTOR)
