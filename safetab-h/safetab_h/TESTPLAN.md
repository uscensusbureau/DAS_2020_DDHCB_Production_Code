# SafeTab-H Tests

SafeTab-H and supporting libraries provide a range of tests to ensure that the software is working correctly.

Our tests are divided into unit and system tests. Unit tests verify that the implementation of a class, and its associated methods, match the behavior specified in its documentation. System tests are designed to determine whether the assembled system meets its specifications.

We also divide tests into fast and slow tests. Fast tests complete relatively quickly, and can be run often, while slow tests are longer-running and less frequently exercised. While unit tests tend to be fast and system tests tend to be slow, there are some slow unit tests and fast system tests.

All tests are run using pytest. Core is provided as a binary wheel, and thus does not have runnable tests in this release.

All tests are run on a single machine. Runtimes mentioned in this document were measured on an `r4.16xlarge` machine.

All tests commands in this file should be run from the repository root, unless otherwise noted.

## Running all tests

Execute the following to run the tests:

*Fast Tests:*

```bash
python3.11 -m pytest common -m 'not slow'
python3.11 -m pytest analytics -m 'not slow'
python3.11 -m pytest safetab_h -m 'not slow'
python3.11 -m pytest safetab_utils -m 'not slow'
# (Total runtime estimate: 50 minutes)
```

*Slow Tests:*

```bash
python3.11 -m pytest common -m 'slow' --suppress-no-test-exit-code
python3.11 -m pytest analytics -m 'slow'
python3.11 -m pytest safetab_h -m 'slow'
python3.11 -m pytest safetab_utils -m 'slow' --suppress-no-test-exit-code
# (Total runtime estimate: 3 hours)
```

Note: Common and SafeTab-Utils have no slow tests, and pytest treats finding no tests as an error. The `suppress-no-test-exit-code` flag will prevent the error, but requires the `pytest-custom-exit-code` package be installed. If you do not have it, you can remove the flag and ignore the error.


## SafeTab-H's tests

### Unit Tests:

SafeTab-H unit tests test the individual components of the algorithm like characteristic iteration flat maps, region preprocessing, and accuracy report components.

```bash
python3.11 -m pytest safetab_h/test/unit
# (Runtime estimate: 20 seconds)
```

### System Tests:

System tests are designed to determine whether the assembled system meets its specifications.

##### **Input Validation**:

   * Tests that the input to SafeTab-H matches input specification.
   * Tests that SafeTab-H raises appropriate exceptions when invalid arguments are supplied and that SafeTab-H runs the full algorithm for US and/or Puerto Rico depending on input supplied.

```bash
python3.11 -m pytest safetab_h/test/system/test_input_validation.py
# (Runtime estimate: 3.5 minutes)
```

 The below spark-submit commands demonstrates running SafeTab-H command line program in input `validate` mode on toy dataset and a csv reader. To validate with the CEF reader, see instructions in the [README](./README.md).

```bash
pushd safetab_h/tmlt/safetab_h
spark-submit \
        --conf spark.pyspark.python=/usr/local/bin/python3.11 \
        --properties-file resources/spark_configs/spark_local_properties.conf \
        ./safetab-h.py validate resources/toy_dataset/input_dir_zcdp \
         resources/toy_dataset
popd
# (Runtime estimate: 40 seconds)
```


##### **Output Validation**:

   * Tests that the output to SafeTab-H conforms to the output specification and varies appropriately with changes in the input. Also tests that flags for non-negativity and excluding states work as expected.

The below spark-submit commands demonstrates running SafeTab-H command line program to produce private tabulations followed by output validation on toy dataset and a csv reader.

```bash
pushd safetab_h/tmlt/safetab_h
spark-submit \
        --conf spark.pyspark.python=/usr/local/bin/python3.11 \
        --properties-file resources/spark_configs/spark_local_properties.conf \
        ./safetab-h.py execute resources/toy_dataset/input_dir_zcdp  \
        resources/toy_dataset  \
        example_output/safetab_h --validate-private-output
popd
# (Runtime estimate: 4 minutes)
```

##### **Correctness Tests (also test of consistency)**:

   * A test checks that when all values of privacy budget goes to infinity, the output converges to the correct (noise-free/ground truth) answer.

   * A test ensures that the SafeTab-H algorithm outputs the correct population groups (characteristic iteration and geographic entity) with the correct aggregate counts when non-zero privacy budget is allocated to the iteration/geo level.

   * A test ensures that the SafeTab-H algorithm determines computes the correct aggregate counts when the noise addition is turned off.

   * A test that ensures that SafeTab-H produces correctly-formatted output.

   * A test to ensure that adaptivity produces the correct detail levels.

```bash
python3.11 -m pytest safetab_h/test/system/test_correctness.py
# (Runtime estimate: 2 hours 20 minutes)
```

##### **Adaptivity Test**:

   * Tests that SafeTab-H produces the breakdowns with the correct level of detail for a given T1 output.

```bash
python3.11 -m pytest safetab_h/test/system/test_adaptivity.py
# (Runtime estimate: 35 minutes)
```

##### **Accuracy Test**:

   * Run `examples/multi_run_error_report_h.py` for comparing the results from SafeTab-H algorithm execution against the ground truth (non-private) answers across multiple trials and privacy budgets. This example script runs the SafeTab-H program on non-sensitive data present in input path (`tmlt/safetab/resources/toy_dataset`) for 5 trials and epsilons 1.0 and 30.0. It produces ground truth tabulations `t3/*/*.csv` and `t4/*/*.csv` in directory (`example_output/multi_run_error_report_h/ground_truth`). Error report `error_report.csv` and private tabulations `t3/*/*.csv` and `t4/*/*.csv` for each run can be found in the directory (`example_output/multi_run_error_report_h/single_runs`). The aggregated error report `multi_run_error_report.csv` is saved to output directory (`example_output/multi_run_error_report_h/`).

```bash
bash safetab_h/examples/multi_run_error_report.sh
# (Runtime estimate: 20 minutes)
```

Note: Multi-run error report uses the ground truth counts. It violates differential privacy, and should not be created using sensitive data. Its purpose is to test SafeTab-H on non-sensitive or synthetic datasets to help tune the algorithms and to predict the performance on the private data.

   * Tests that running the full accuracy report creates the appropriate output directories for SafeTab-H algorithm.

```bash
python3.11 -m pytest safetab_h/test/system/test_accuracy_report.py
# (Runtime estimate: 20 minutes)
```
