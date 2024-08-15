# SafeTab-H for Detailed Race and Ethnicity



SPDX-License-Identifier: Apache-2.0
Copyright 2024 Tumult Labs

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

## Overview

SafeTab-H produces differentially private tables of statistics (counts) of demographic and housing characteristics crossed with detailed races and tribes at varying levels of geography (national, state, county, tract, place and AIANNH).

The data product derived from the output of SafeTab-H is known as Detailed DHC-B.

More information about the SafeTab algorithm can be found in the [SafeTab-H specifications document](SafeTab_H_Documentation.pdf), which describes the problem, general approach, and (in Appendix A), the input and output file formats.

See [CHANGELOG](CHANGELOG.md) for version number information and changes from past versions.

## System Requirements

SafeTab-H is designed to run on a Linux machine with Python 3.11 and PySpark 3. It can be run either locally on a single machine, or on a Spark cluster. We have developed and tested with Amazon EMR 6.12 in mind. When running on a nation-scale dataset, we recommend using an EMR cluster having at least 1 master and 2 core nodes of instance type: r4.16xlarge or higher.

## Installing and Running SafeTab-H Locally

These instructions assume that the default python3 version is Python 3.11. Once Python 3.11 is installed, you can add the following to your `.bashrc` to make it the default version:

```bash
alias python3=python3.11
```

### 1. Installing Dependencies
First, make sure you have Java 8 or later with `JAVA_HOME` properly set. If Java is not yet installed on your system, you can [install OpenJDK 8](https://openjdk.org/install/) (installation will vary based on the system type).

All python dependencies, as specified in [requirements.txt](requirements.txt) must be installed and available on the PYTHONPATH.

You can do this by running:

```bash
sudo python3 -m pip install -r safetab_h/requirements.txt
```

### 2. Installing Tumult Core

SafeTab-H also requires the Tumult Core library to be installed. Tumult Core can either be installed from the wheel file provided with this repository, or from PyPI (like external dependencies in the previous step). Users like the Census who prefer to avoid installing packages from PyPI will likely prefer installing from a wheel. Users who do not have such concerns will likely find it easier to install from PyPI.

#### Wheel installation (Linux only)

Tumult Core can be installed by calling:

```bash
sudo python3 -m pip install --no-deps core/tmlt_core-0.12.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

This will only work on a Linux machine, as the provided wheel is Linux-only. All other operating systems should install from PyPI.

#### Pip installation

Tumult Core can also be installed from PyPI:

```sudo python3 -m pip install tmlt.core==0.12.0```

### 3. Setting up your environment

`PYTHONPATH` needs to updated to include the Tumult source directories. This assumes that required Python dependencies have been installed.

```bash
DIR=<absolute path of cloned tumult repository>
export PYTHONPATH=$PYTHONPATH:$DIR/safetab_h
export PYTHONPATH=$PYTHONPATH:$DIR/safetab_utils
export PYTHONPATH=$PYTHONPATH:$DIR/common
export PYTHONPATH=$PYTHONPATH:$DIR/analytics
```

`PYTHONPATH` also needs to be updated to include a Census Edited File (CEF) reader module. If you are reading CSV files, you can use the built-in mock CEF reader:

```bash
export PYTHONPATH=$DIR/mock_cef_reader:$PYTHONPATH
```

If you are reading CEF files, you will need to use MITRE’s CEF reader (developed separately):

```bash
export PYTHONPATH=<absolute path to directory containing safetab_cef_reader.py>:$PYTHONPATH
```

If using MITRE's CEF reader, you will also need to include its dependencies. Consult the CEF reader README for more details.

### 4. Running PHSafe

#### Input Directory Structure
There are two path inputs to SafeTab-H, a `parameters path` and a `data path`. This setup is replicated in the  `safetab_h/tmlt/safetab_h/resources/toy_dataset` directory. 

The parameters path should point to a directory containing:
  - config.json
  - ethnicity-characteristic-iterations.txt
  - race-and-ethnicity-code-to-iteration.txt
  - race-and-ethnicity-codes.txt
  - race-characteristic-iterations.txt
  
The `data path` has different requirements depending on the type of input reader being used. If a CEF reader is specified, the data path should point to the CEF reader's config file. If a CSV reader is being used, the data path should point to a directory containing:
  - household-records.txt
  - pop-group-totals.txt
  - GRF-C.txt

Note:
- The parameters directory contains non-CEF input files. `config.json` specifies the privacy parameters. [`safetab_h/tmlt/safetab_h/resources/toy_dataset/input_dir_puredp/config.json`](tmlt/safetab_h/resources/toy_dataset/input_dir_puredp/config.json) contains PureDP privacy parameters and sets the algorithm to run using PureDP, while [`safetab_h/tmlt/safetab_h/resources/toy_dataset/input_dir_zcdp/config.json`](tmlt/safetab_h/resources/toy_dataset/input_dir_zcdp/config.json) contains Rho zCDP privacy parameters and sets the algorithm to run using Rho zCDP.

#### Command line tool
The primary command line interface is driven by [`safetab_h/tmlt/safetab_h/safetab-h.py`](tmlt/safetab_h/safetab-h.py). We recommend running SafeTab-H from `safetab_h/tmlt/safetab_h` because the provided Spark properties files need to assume a directory from which the program is running.

The SafeTab-H command line program expects one of the following subcommands: `validate` or `execute`. These modes are explained below.

To view the list of available subcommands on console, enter:

```bash
safetab-h.py -h
```

To view the arguments for running in a given mode on console, enter:

```bash
safetab-h.py <subcommand> -h
```

The following subcommands are supported:

##### Validate
`validate` mode validates the input data files against the input specification and reports any discrepancies.  Validation errors are written to the user-specified log file.

An example command to validate in local mode:

```bash
spark-submit \
   --properties-file resources/spark_configs/spark_local_properties.conf \
   safetab-h.py validate \
   <path to parameters folder>  \
   <path to reader configuration> \
   --log <log_file_name>
```

Note: If using csv readers replace `<path to reader configuration>` with `<path to data folder>`.

##### Execute
The `execute` subcommand first validates the input files and then executes SafeTab-H. Both input validation and execution of the private algorithm use spark.

The SafeTab-H algorithm (executed with `safetab-h.py`) produces t3 and t4 tabulations. The output files `t3/T0300*/*.csv` and `t4/T0400*/*.csv` are generated and saved to the output folder specified on the command-line. The optional `--validate-private-output` flag can be passed to validate the generated output files.

Input and output directories must correspond to locations on the local machine (and not S3 or HDFS).

An example command to execute in local mode:

```bash
spark-submit \
      --properties-file resources/spark_configs/spark_local_properties.conf \
      safetab-h.py execute \
      <path to parameters folder>/  \
      <path to reader configuration> \
      <path to output folder> \
      --log <log_file_name> \
      --validate-private-output
```

Notes:

If using csv readers replace `<path to reader configuration>` with `<path to data folder>`.

If, by default, the machine does not have enough disk space for the SafeTab output,
a larger directory can be mounted to a drive on the local machine prior to
execution with commands similar to the following:

```
> lsblk

NAME    MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
xvda    202:0    0   10G  0 disk
└─xvda1 202:1    0   10G  0 part /
xvdb    202:16   0  128G  0 disk
├─xvdb1 202:17   0    5G  0 part /emr
└─xvdb2 202:18   0  123G  0 part /mnt
xvdc    202:32   0  128G  0 disk /mnt1
xvdd    202:48   0  128G  0 disk /mnt2
xvde    202:64   0  128G  0 disk /mnt3

> mkdir <path to output folder>
> sudo mount /dev/xvdc <path to output folder>
```

A sample Spark custom properties file for local mode execution is located in [`safetab_h/tmlt/safetab_h/resources/spark_configs/spark_local_properties.conf`](tmlt/safetab_h/resources/spark_configs/spark_local_properties.conf).
While Spark properties are often specific to the environment (number of cores,
memory allocation, etc.), we strongly recommend that the `spark.sql.execution.arrow.enabled`
be set to `true` as is done in the example config file for local mode.
When pyarrow is enabled, the data exchange between Python and Java is much faster and results
in orders-of-magnitude differences in runtime performance. 

#### Example scripts

The shell script [`safetab_h/examples/validate_input_safetab_h.sh`](examples/validate_input_safetab_h.sh) demonstrates running the SafeTab-H command line program in validate mode on the toy dataset using the csv reader.  An excerpt is shown here with comments:

```bash
safetab-h.py validate \  # validate the SafeTab-H inputs
resources/toy_dataset/input_dir_puredp \   # the parameters directory (see note below)
resources/toy_dataset \   # the data_path (see note below)
-l example_output_p/safetab_toydataset.log  # desired log location
```

The shell script [`safetab_h/examples/run_safetab_h.sh`](examples/run_safetab_h.sh) demonstrates running the SafeTab-H command line program in execute mode using the csv reader with input and output validation.  An excerpt is shown here with comments:

```bash
safetab-h.py execute \  # execute the SafeTab-H algorithm
resources/toy_dataset/input_dir_puredp \   # the parameters directory (see note below)
resources/toy_dataset \   # the data_path (see note below)
example_output_p \        # desired output location
-l example_output_p/safetab_toydataset.log \  # desired log location
--validate-private-output      # validate output after executing algorithm
```

See `safetab_h/examples` for examples of other features of the SafeTab-H command line program.

When running these examples, the SafeTab-H config file containing the input privacy-parameters is defined in [`safetab_h/tmlt/safetab_h/resources/toy_dataset/input_dir_puredp/config.json`](tmlt/safetab_h/resources/toy_dataset/input_dir_puredp/config.json). The default privacy definition is PureDP. If zCDP is desired, switch the parameters folder argument to `resources/toy_dataset/input_dir_zcdp`, which will utilize [`safetab_h/tmlt/safetab_h/resources/toy_dataset/input_dir_zcdp/config.json`](tmlt/safetab_h/resources/toy_dataset/input_dir_zcdp/config.json) as the SafeTab-H config.

Note that the toy dataset (located at `safetab_h/tmlt/safetab_h/resources/toy_dataset`) used in these examples is small and not representative of a realistic input. The output from SafeTab-H when run on this dataset will not be comparable to a run on a realistic input. We include the toy dataset to provide an example of input formats, an example of output formats, and a way to quickly experiment with running SafeTab-H, but not as a way to generate representative outputs.

## Installing and running SafeTab-H on an EMR cluster

### 1. Installing dependencies and Tumult Core
Before you can run SafeTab-H, you must install all python dependencies specified in [requirements.txt](requirements.txt), plus Tumult Core.

We have designed with EMR 6.12 in mind, which does not come with Python 3.11. Therefore, you will need to install Python 3.11, and reinstall PySpark. This can be done with a [bootstrap action](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-plan-bootstrap.html). We include sample bootstrap scripts that install Python3.11, the public dependencies, and (optionally), Tumult Core. Bootstrap actions must be configured when starting a cluster, so start a new cluster with the bootstrap actions specified in this section (we recomend at least 1 master and 2 core nodes of instance type: r4.16xlarge or higher).

Tumult Core can either be installed from the wheel file provided with this repository, or from PyPI (like the other dependencies). Users like the Census who prefer to avoid installing packages from PyPI will likely prefer installing from a wheel. Users who do not have such concerns will likely find it easier to install from PyPI.

If you would like to install Tumult Core from PyPI, use [bootstrap_with_core.sh](safetab_h/tmlt/safetab_h/resources/installation/bootstrap_with_core.sh). Upload the script to an S3 bucket, create a new bootstrap action, and provide the s3 location of the script as its script location.

If you would like to install Tumult Core from the provided wheel, use [bootstrap_without_core.sh](safetab_h/tmlt/safetab_h/resources/installation/bootstrap_without_core.sh). Upload that script, [`core/bootstrap_script.sh`](../core/bootstrap_script.sh), and [`core/tmlt_core-0.12.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`](../core/tmlt_core-0.12.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl) to an S3 bucket. Then, add the following bootstrap steps (in this order):
1. Set the script location to the S3 path where you uploaded `bootstrap_without_core.sh`.
1. Set the script location to the S3 path where you uploaded `core/bootstrap_script.sh`. Set the Optional Argument to the S3 path where you uploaded the wheel file.

### 2. Running SafeTab-H

Once your cluster has started, you can use the AWS Management Console to configure a step that will invoke the `spark-submit` command. 

#### Uploading PHSafe
There are three important preconditions:

1. All of the inputs files must be uploaded to S3.
1. A zip file containing the repository source code must be created and placed in S3.
1. The main driver python program, [`safetab_h/tmlt/safetab_h/safetab-h.py`](tmlt/safetab_h/safetab-h.py), must be placed in S3.

The zip file can be created using the following command, which creates a packaged repository `repo.zip` that contains Tumult’s products and MITRE’s CEF reader (it does not contain any other dependencies).

```bash
bash <path to cloned tumult repo>/safetab_h/examples/repo_zip.sh \
-t <path to cloned tumult repo> \
-r <path to CEF reader>
```

If you're using the built-in CSV reader, the `-r` option can point to the included `mock_cef_reader` package. If you have a CEF reader installed into your environment, you can skip the `-r` flag entirely.

Note: The `repo_zip.sh` script has a dependency on associative arrays and works with bash version 4.0 or newer.

#### <a id="steps"></a>Steps:

Once you've uploaded the zip file, the driver program, and the input files, you can [add a step](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-work-with-steps.html) to your cluster:

1. In the [Amazon EMR console](https://console.aws.amazon.com/elasticmapreduce), on the Cluster List page, select the link for your cluster.
1. On the `Cluster Details` page, choose the `Steps` tab.
1. On the `Steps` tab, choose `Add step`.
1. Type appropriate values in the fields in the `Add Step` dialog, and then choose `Add`. Here are the sample values:

                Step type: Custom JAR

                Name: <any name that you want>

                JAR location: command-runner.jar

                Arguments:
                        spark-submit
                        --deploy-mode client --master yarn
                        --conf spark.pyspark.python=/usr/local/bin/python3.11
                        --conf spark.driver.maxResultSize=20g
                        --conf spark.sql.execution.arrow.enabled=true
                        --py-files s3://<s3 repo.zip path>
                        s3://<s3 safetab-h main file path>/safetab-h.py
                        execute
                        s3://<s3 sample parameters files path>
                        s3://<s3 data_path>
                        s3://<s3 output directory>
                        --log s3://<s3 output directory>/<log file path>
                        --validate-private-output

                Action on Failure: Cancel and wait

Notes: 
- Since Python 3.11 is not the default python version on an EMR cluster, you must manually install it and point PySpark to it (using the `spark.pyspark.python` configuration option) for your step. This example step assumes that Python 3.11 has been installed at `/usr/local/bin/python3.11`, which is where our sample bootstrap script installs it.
- Output locations must be S3 paths. 
- The `--log` and `--validate-private-output` arguments are optional. 
- You can run SafeTab-H this way on either the sample input files or the toy dataset. You can also run the program in `validate` mode by replacing `execute` with `validate`, removing the output directory argument and `--validate-private-output` flag.

#### Spark properties

While Spark properties are often specific to the environment (number of cores, memory allocation, etc.), we strongly recommend that the `spark.sql.execution.arrow.enabled` be set to `true`. An example custom Spark properties config file for cluster mode is located in [`safetab_h/tmlt/safetab_h/resources/spark_configs/spark_cluster_properties.conf`](tmlt/safetab_h/resources/spark_configs/spark_cluster_properties.conf).

A properties file must be located on the local machine (we recommend using a bootstrap action to accomplish this), and can be specified by adding the `--properties-file` option to `spark-submit` in the step specification.

## Testing

*See [TESTPLAN](TESTPLAN.md)*

## Warnings and errors

### List of error codes

SafeTab-H is distributed as Python source code plus associated scripts and resource files. The errors that occur during the SafeTab-H command line tool usage are human readable Python exceptions.

### Known Warnings

These warnings can be safely ignored:

1. PyTest warning:

```
PytestUnknownMarkWarning: Unknown pytest.mark.slow - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.slow
```


2. In order to prevent the following warning:

```
WARN NativeCodeLoader: Unable to load native-hadoop library for your platform
```

`LD_LIBRARY_PATH` must be set correctly. Use the following:

```bash
export LD_LIBRARY_PATH=/usr/lib/hadoop/lib/native/
```

If `HADOOP_HOME` is set correctly (usually `/usr/lib/hadoop`), this may be replaced with

```bash
export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native/
```

3. Other known warnings:

```
FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise
comparison
```
```
UserWarning: In Python 3.6+ and Spark 3.0+, it is preferred to specify type hints for pandas UDF instead of specifying
pandas UDF type which will be deprecated in the future releases. See SPARK-28264 for more details.
```
```
UserWarning: It is preferred to use 'applyInPandas' over this API. This API will be deprecated in the future releases.
See SPARK-28264 for more details.
```

### Known Errors

SafeTab-H occasionally produces a `ValueError: I/O operation on closed file` about attempting to write to the logs file after all logs are written. This error can be safely ignored, as all log records should still be written.
