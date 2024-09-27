.. _core-changelog:

Changelog
=========

0.12.0 - 2024-02-26
-------------------

Added
~~~~~
- Added a non-truncating truncation strategy with infinite stability.
- Added functions implementing various mechanisms to support slow scaling PRDP.

Changed
~~~~~~~
- Changed :func:`~.truncate_large_groups` and :func:`~.limit_keys_per_group` to use
  SHA-2 (256 bits) instead of Spark's default hash (Murmur3). This results in a minor
  performance hit, but these functions should be less likely to have collisions which
  could impact utility. **Note that this may change the output of transformations which
  use these functions.** In particular, :class:`~.PrivateJoin`,
  :class:`~.LimitRowsPerGroup`, :class:`~.LimitKeysPerGroup`, and
  :class:`~.LimitRowsPerKeyPerGroup`.
- Expanded the explanation of `GroupingFlatMap`'s stability.
- Support all metrics for the flat map transformation.

Fixed
~~~~~
- Fixed missing minus sign in the documentation of the discrete Gaussian pmf.
- Fixed :func:`~.create_partition_selection_measurement` behavior when called
  with infinite budgets.
- Fixed :func:`~.create_partition_selection_measurement` crashing when called
  with very large budgets.


0.11.6 - 2024-02-21
-------------------

0.11.6 was yanked. Those changes will be released in 0.12.0.


0.11.5 - 2023-11-29
-------------------

Fixed
~~~~~
-  Addressed a serious security vulnerability in PyArrow: `CVE-2023-47248 <https://nvd.nist.gov/vuln/detail/CVE-2023-47248>`__.

   -  Python 3.8+ now requires PyArrow 14.0.1 or higher, which is the recommended fix and addresses the vulnerability.
   -  Python 3.7 uses the hotfix, as PyArrow 14.0.1 is not compatible with Python 3.7. Note that if you are using 3.7 the hotfix must be imported before your Spark code. Core imports the hotfix, so importing Core before Spark will also work.
   -  **It is strongly recommended to upgrade if you are using an older version of Core.**
   -  Also see the `GitHub Advisory entry <https://github.com/advisories/GHSA-5wvp-7f3h-6wmm>`__ for more information.

- Fixed a reference to an uninitialized variable that could cause :func:`~.arb_union` to crash the Python interpreter.

0.11.4 - 2023-11-01
-------------------

Fixed a typo that prevented PyArrow from being installed on Python 3.8.

0.11.3 - 2023-10-31
-------------------

Fixed a typo that prevented PySpark from being installed on Python 3.8.

0.11.2 - 2023-10-27
-------------------

Added
~~~~~
- Added support for Python 3.11.

0.11.1 - 2023-09-25
-------------------

Added
~~~~~
- Added documentation for known vulnerabilities related to Parallel Composition and the use of SymPy.

0.11.0 - 2023-08-15
-------------------

Changed
~~~~~~~
- Replaced the `group_keys` for constructing :class:`~.SparkGroupedDataFrameDomain`\ s with `groupby_columns`.
- Modified :class:`~.SymmetricDifference` to define the distance
  between two elements of :class:`~.SparkGroupedDataFrameDomain`\ s to be infinite when the two elements have different `group_keys`.
- Updated maximum version for `pyspark` from 3.3.1 to 3.3.2.

0.10.2 - 2023-07-18
-------------------

Changed
~~~~~~~
- Build wheels for macOS 11 instead of macOS 13.
- Updated dependency version for `typing_extenstions` to 4.1.0

0.10.1 - 2023-06-08
-------------------

Added
~~~~~
- Added support for Python 3.10.
- Added the :func:`~.arb_exp`, :func:`~.arb_const_pi`, :func:`~.arb_neg`, :func:`~.arb_product`, :func:`~.arb_sum`, :func:`~.arb_union`, :func:`~.arb_erf`, and :func:`~.arb_erfc` functions.
- Added a new error, :class:`~.DomainMismatchError`, which is raised when two or more domains should match but do not.
- Added a new error, :class:`~.UnsupportedMetricError`, which is raised when an unsupported metric is used.
- Added a new error, :class:`~.MetricMismatchError`, which is raised when two or more metrics should match but do not.
- Added a new error, :class:`~.UnsupportedMeasureError`, which is raised when an unsupported measure is used.
- Added a new error, :class:`~.MeasureMismatchError`, which is raised when two or more measures should match but do not.
- Added a new error, :class:`~.UnsupportedCombinationError`, which is raised when some combination of domain, metric, and measure is not supported (but each one is individually valid).
- Added a new error, :class:`~.UnsupportedNoiseMechanismError`, which is raised when a user tries to create a measurement with a noise mechanism that is not supported.
- Added a new error, :class:`~.UnsupportedSympyExprError`, which is raised when a user tries to create an :class:`~.ExactNumber` with an invalid SymPy expression.

Changed
~~~~~~~
- Restructured the repository to keep code under the `src` directory.

0.10.0 - 2023-05-17
-------------------

Added
~~~~~
- Added the :class:`~.BoundSelection` spark measurement.

Changed
~~~~~~~
- Replaced many existing exceptions in Core with new classes that contain metadata about the inputs causing the exception.

Fixed
~~~~~
- Fixed bug in :func:`~.limit_keys_per_group`.
- Fixed bug in :func:`~.gaussian`.
- :func:`~tmlt.core.utils.cleanup.cleanup` now emits a warning rather than an exception if it fails to get a Spark session.
  This should prevent unexpected exceptions in the ``atexit`` cleanup handler.

0.9.2 - 2023-05-16
------------------

0.9.2 was yanked, as it contained breaking changes. Those changes will be released in 0.10.0.

0.9.1 - 2023-04-20
------------------

Added
~~~~~
- Subclasses of :class:`~.Measure` now have equations defining the distance they represent.

0.9.0 - 2023-04-14
------------------

Added
~~~~~

- :mod:`~.utils.join`, which contains utilities for validating join parameters, propogating domains through joins, and joining dataframes.

Changed
~~~~~~~

- :func:`~.truncate_large_groups` does not clump identical records together in hash-based ordering.
- :class:`~.TransformValue` no longer fails when renaming the id column using :class:`~.RenameValue`.

Fixed
~~~~~

- groupby no longer outputs nan values when both tables are views on the same original table
- private join no longer drops Nulls on non-join columns when join_on_nulls=False
- groupby average and variance no longer drops groups containing null values

0.8.3 - 2023-03-08
------------------

Changed
~~~~~~~

- Functions in :mod:`~.aggregations` now support :class:`~.ApproxDP`.

0.8.2 - 2023-03-02
------------------

Added
~~~~~
- Added `LimitKeysPerGroupValue` transformation

Changed
~~~~~~~
- Updated `LimitKeysPerGroup` to require an output metric, and to support the
  `IfGroupedBy(grouping_column, SymmetricDifference())` output metric. Dropped the 'use_l2' parameter.

0.8.1 - 2023-02-24
------------------

Added
~~~~~

- Added `LimitRowsPerKeyPerGroup` and `LimitRowsPerKeyPerGroupValue` transformations

Changed
~~~~~~~

- Faster implementation of discrete_gaussian_inverse_cmf.

0.8.0 - 2023-02-14
------------------

Added
~~~~~

- Added `LimitRowsPerGroupValue` transformation

Changed
~~~~~~~

- Updated `LimitRowsPerGroup` to require an output metric, and to support the
  `IfGroupedBy(column, SymmetricDifference())` output metric.
- Added a check so that `TransformValue` can no longer be instantiated without
  subclassing.


0.7.0 - 2023-02-02
------------------

Added
~~~~~

- Added measurement for adding Gaussian noise.

0.6.3 - 2022-12-20
------------------

Changed
~~~~~~~

- On Linux, Core previously used `MPIR <https://en.wikipedia.org/wiki/MPIR_(mathematics_software)>`__ as a multi-precision arithmetic library to support `FLINT <https://flintlib.org/>`__ and `Arb <https://arblib.org/>`__.
  MPIR is no longer maintained, so Core now uses `GMP <https://gmplib.org/>`__ instead.
  This change does not affect macOS builds, which have always used GMP, and does not change Core's Python API.

Fixed
~~~~~

- Fixed a bug where PrivateJoin's privacy relation would only accept string keys in the d_in. It now accepts any type of key.


0.6.2 - 2022-12-07
------------------

This is a maintenance release which introduces a number of documentation improvements, but has no publicly-visible API changes.

Fixed
~~~~~

- :func:`~tmlt.core.utils.configuration.check_java11()` now has the correct behavior when Java is not installed.

0.6.1 - 2022-12-05
------------------

Added
~~~~~

-  Added approximate DP support to interactive mechanisms.
-  Added support for Spark 3.1 through 3.3, in addition to existing support for Spark 3.0.

Fixed
~~~~~

-  Validation for ``SparkedGroupDataFrameDomain``\ s used to fail with a Spark ``AnalysisException`` in some environments.
   That should no longer happen.

0.6.0 - 2022-11-14
------------------

Added
~~~~~

-  Added new ``PrivateJoinOnKey`` transformation that works with ``AddRemoveKeys``.
-  Added inverse CDF methods to noise mechanisms.

0.5.1 - 2022-11-03
------------------

Fixed
~~~~~

-  Domains and metrics make copies of mutable constructor arguments and return copies of mutable properties.

0.5.0 - 2022-10-14
------------------

Changed
~~~~~~~

-  Core no longer depends on the ``python-flint`` package, and instead packages libflint and libarb itself.
   Binary wheels are available, and the source distribution includes scripting to build these dependencies from source.

Fixed
~~~~~

-  Equality checks on ``SparkGroupedDataFrameDomain``\ s used to occasionally fail with a Spark ``AnalysisException`` in some environments.
   That should no longer happen.
-  ``AddRemoveKeys`` now allows different names for the key column in each dataframe.

0.4.3 - 2022-09-01
------------------

-  Core now checks to see if the user is running Java 11 or higher. If they are, Core either sets the appropriate Spark options (if Spark is not yet running) or raises an informative exception (if Spark is running and configured incorrectly).

0.4.2 - 2022-08-24
------------------

Changed
~~~~~~~

-  Replaced uses of PySpark DataFrame’s ``intersect`` with inner joins. See https://issues.apache.org/jira/browse/SPARK-40181 for background.

0.4.1 - 2022-07-25
------------------

Added
~~~~~

-  Added an alternate prng for non-intel architectures that don’t support RDRAND.
-  Add new metric ``AddRemoveKeys`` for multiple tables using ``IfGroupedBy(X, SymmetricDifference())``.
-  Add new ``TransformValue`` base class for wrapping transformations to support ``AddRemoveKeys``.
-  Add many new transformations using ``TransformValue``: ``FilterValue``, ``PublicJoinValue``, ``FlatMapValue``, ``MapValue``, ``DropInfsValue``, ``DropNaNsValue``, ``DropNullsValue``, ``ReplaceInfsValue``, ``ReplaceNaNsValue``, ``ReplaceNullsValue``, ``PersistValue``, ``UnpersistValue``, ``SparkActionValue``, ``RenameValue``, ``SelectValue``.

Changed
~~~~~~~

-  Fixed bug in ``ReplaceNulls`` to not allow replacing values for grouping column in ``IfGroupedBy``.
-  Changed ``ReplaceNulls``, ``ReplaceNaNs``, and ``ReplaceInfs`` to only support specific ``IfGroupedBy`` metrics.

0.3.2 - 2022-06-23
------------------

Changed
~~~~~~~

-  Moved ``IMMUTABLE_TYPES`` from ``utils/testing.py`` to ``utils/type_utils.py`` to avoid importing nose when accessing ``IMMUTABLE_TYPES``.

0.3.1 - 2022-06-23
------------------

Changed
~~~~~~~

-  Fixed ``copy_if_mutable`` so that it works with containers that can’t be deep-copied.
-  Reverted change from 0.3.0 “Add checks in ``ParallelComposition`` constructor to only permit L1/L2 over SymmetricDifference or AbsoluteDifference.”
-  Temporarily disabled flaky statistical tests.

0.3.0 - 2022-06-22
------------------

Added
~~~~~

-  Added new transformations ``DropInfs`` and ``ReplaceInfs`` for handling infinities in data.
-  Added ``IfGroupedBy(X, SymmetricDifference())`` input metric.

   -  Added support for this metric to ``Filter``, ``Map``, ``FlatMap``, ``PublicJoin``, ``Select``, ``Rename``, ``DropNaNs``, ``DropNulls``, ``DropInfs``, ``ReplaceNulls``, ``ReplaceNaNs``, and ``ReplaceInfs``.

-  Added new truncation transformations for ``IfGroupedBy(X, SymmetricDifference())``: ``LimitRowsPerGroup``, ``LimitKeysPerGroup``
-  Added ``AddUniqueColumn`` for switching from ``SymmetricDifference`` to ``IfGroupedBy(X, SymmetricDifference())``.
-  Added a topic guide around NaNs, nulls and infinities.

Changed
~~~~~~~

-  Moved truncation transformations used by ``PrivateJoin`` to be functions (now in ``utils/truncation.py``).
-  Change ``GroupBy`` and ``PartitionByKeys`` to have an ``use_l2`` argument instead of ``output_metric``.
-  Fixed bug in ``AddUniqueColumn``.
-  Operations that group on null values are now supported.
-  Modify ``CountDistinctGrouped`` and ``CountDistinct`` so they work as expected with null values.
-  Changed ``ReplaceNulls``, ``ReplaceNaNs``, and ``ReplaceInfs`` to only support specific ``IfGroupedBy`` metrics.
-  Fixed bug in ``ReplaceNulls`` to not allow replacing values for grouping column in ``IfGroupedBy``.
-  ``PrivateJoin`` has a new parameter for ``__init__``: ``join_on_nulls``.
   When ``join_on_nulls`` is ``True``, the ``PrivateJoin`` can join null values between both dataframes.
-  Changed transformations and measurements to make a copy of mutable constructor arguments.
-  Add checks in ``ParallelComposition`` constructor to only permit L1/L2 over SymmetricDifference or AbsoluteDifference.

Removed
~~~~~~~

-  Removed old examples from ``examples/``.
   Future examples will be added directly to the documentation.

0.2.0 - 2022-04-12 (internal release)
-------------------------------------

Added
~~~~~

-  Added ``SparkDateColumnDescriptor`` and ``SparkTimestampColumnDescriptor``, enabling support for Spark dates and timestamps.
-  Added two exception types, ``InsufficientBudgetError`` and ``InactiveAccountantError``, to PrivacyAccountants.
-  Future documentation will include any exceptions defined in this library.
-  Added ``cleanup.remove_all_temp_tables()`` function, which will remove all temporary tables created by Core.
-  Added new components ``DropNaNs``, ``DropNulls``, ``ReplaceNulls``, and ``ReplaceNaNs``.

0.1.1 - 2022-02-24 (internal release)
-------------------------------------

Added
~~~~~

-  Added new implementations for SequentialComposition and ParallelComposition.
-  Added new spark transformations: Persist, Unpersist and SparkAction.
-  Added PrivacyAccountant.
-  Installation on Python 3.7.1 through 3.7.3 is now allowed.
-  Added ``DecorateQueryable``, ``DecoratedQueryable`` and ``create_adaptive_composition`` components.

Changed
~~~~~~~

-  Fixed a bug where ``create_quantile_measurement`` would always be created with PureDP as the output measure.
-  ``PySparkTest`` now runs ``tmlt.core.utils.cleanup.cleanup()`` during ``tearDownClass``.
-  Refactored noise distribution tests.
-  Remove sorting from ``GroupedDataFrame.apply_in_pandas`` and ``GroupedDataFrame.agg``.
-  Repartition DataFrames output by ``SparkMeasurement`` to prevent privacy violation.
-  Updated repartitioning in ``SparkMeasurement`` to use a random column.
-  Changed quantile implementation to use arblib.
-  Changed Laplace implementation to use arblib.

Removed
~~~~~~~

-  Removed ``ExponentialMechanism`` and ``PermuteAndFlip`` components.
-  Removed ``AddNoise``, ``AddLaplaceNoise``, ``AddGeometricNoise``, and ``AddDiscreteGaussianNoise`` from ``tmlt.core.measurements.pandas.series``.
-  Removed ``SequentialComposition``, ``ParallelComposition`` and corresponding Queryables from ``tmlt.core.measurements.composition``.
-  Removed ``tmlt.core.transformations.cache``.

0.1.0 - 2022-02-14 (internal release)
-------------------------------------

Added
~~~~~

-  Initial release.
