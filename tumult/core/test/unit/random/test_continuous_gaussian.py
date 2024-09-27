"""Tests for :mod:`~tmlt.core.random.continuous_gaussian`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

import math
from unittest import TestCase

from parameterized import parameterized
from scipy.stats import norm

from tmlt.core.random.continuous_gaussian import gaussian_inverse_cdf
from tmlt.core.utils.arb import Arb


class TestContinuousGaussianInverseCDF(TestCase):
    """Tests for :func:`~.continuous_gaussian_inverse_cdf`."""

    @parameterized.expand(
        [
            (1, 1, 2.0, r"`p` should be in \(0,1\)"),
            (float("inf"), 1, 0.4, "Location `u` should be finite and non-nan"),
            (float("nan"), 1, 0.4, "Location `u` should be finite and non-nan"),
            (-float("inf"), 1, 0.4, "Location `u` should be finite and non-nan"),
            (1, float("inf"), 0.5, "Scale should be finite, non-nan and non-negative"),
            (1, float("nan"), 0.5, "Scale should be finite, non-nan and non-negative"),
            (1, -1, 0.5, "Scale should be finite, non-nan and non-negative"),
        ]
    )
    def test_bad_arguments(self, u: float, b: float, p: float, error_msg: str):
        """`gaussian_inverse_cdf` raises error when called with bad arguments."""
        with self.assertRaisesRegex(ValueError, error_msg):
            gaussian_inverse_cdf(u, b, Arb.from_float(p), 63)

    @parameterized.expand(
        [
            (0, 1, 0.5),
            (10, 100, 0.1),
            (10, 100, 0.5),
            (10, 100, 0.9),
            (0, 1, 0.9),
            (0, 5, 0.1),
            (2000, 5, 0.1),
            (-10, 0.5, 0.5),
            (-10, 0.5, 0.09),
        ]
    )
    def test_correctness(self, u: float, sigma_squared: float, p: float):
        """Sanity tests for :func:`gaussian_inverse_cdf`."""
        self.assertAlmostEqual(
            float(gaussian_inverse_cdf(u, sigma_squared, Arb.from_float(p), 63)),
            norm.ppf(p, loc=u, scale=math.sqrt(sigma_squared)),
        )
