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

## SafeTab-H

This repository contains SafeTab-H and its supporting Tumult-developed libraries. For instructions on running SafeTab-H, see its [README](safetab_h/README.md).

### Contents

In the repository there are six folders, each of which contains a component of the release:

- **Core 0.12.0**: A Python library for performing differentially private computations. The design of Tumult Core is based on the design proposed in the [OpenDP White Paper](https://projects.iq.harvard.edu/files/opendp/files/opendp_programming_framework_11may2020_1_01.pdf), and can automatically verify the privacy properties of algorithms constructed from Tumult Core components. Tumult Core is scalable, includes a wide variety of components to handle various query types, and supports multiple privacy definitions. This library is available as an independent open-source release. For more, see its software documentation at https://docs.tmlt.dev/core/v0.12/.
- **Analytics 0.8.3**: A Python library for privately answering statistical queries on tabular data, implemented using Tumult Core. It is built on PySpark, allowing it to scale to large datasets. Its privacy guarantees are based on differential privacy, a technique that perturbs statistics to provably protect the data of individuals in the original dataset. This library is available as an independent open-source release. For more, see its software documentation at https://docs.tmlt.dev/analytics/v0.8/.
- **Common 0.8.7**: A Python library with utilities for reading and validating data. Code in Common is designed not to be specific to Census applications.
- **Mock CEF Reader 0.4.1**: A Python library that substitutes for the required CEF reader when the CSV reader is being used.
- **SafeTab-Utils 0.7.11**: A Python library that implements pieces of Census business logic that are shared between multiple Census products. Includes code for processing characteristic iterations and geographies, performing additional, product-specific validation, and reading in Census data.
- **SafeTab-H 3.0.0**: The main program of this release. It produces the Detailed DHC-B.

SafeTab-H also requires a CEF reader module for reading data from Census' file formats. The CEF reader is implemented separately, and is therefore not included in this release. We do include a built-in alternative reader that gets data from CSV files. When using the CSV reader, the CEF reader can be replaced by the included [mock CEF reader](mock_cef_reader).

For details, consult each library's `README` within its respective subfolder. To see which new features have been added since the previous versions, consult their respective `CHANGELOG`s.

<<<<<<< HEAD
### Synthetic Data

This release also comes with a set of synthetic data files that can be used to test SafeTab-H. The ZIP file containing the sample files is hosted on Amazon Simple Storage Service (Amazon S3). Please note that the download link will be valid until 2024-04-09 at 12:00 pm Eastern.

**To download the ZIP file from the browser:**

Please click this [Amazon S3 URL](https://s3.us-east-1.amazonaws.com/tumult.data.census/safetab_h-3.0.0/safetab-h-full-size-synthetic-data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA25LEV2NNTS4WZ777%2F20240402%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240402T160243Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c505584c3891bc57a8f6c237473c0c21a694b9d0df117f140176ba74ae1be0c8) to download the ZIP file.

**To download the ZIP file from the command line:**

- Execute the following on the command line:

```bash
curl "https://s3.us-east-1.amazonaws.com/tumult.data.census/safetab_h-3.0.0/safetab-h-full-size-synthetic-data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA25LEV2NNTS4WZ777%2F20240402%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240402T160243Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c505584c3891bc57a8f6c237473c0c21a694b9d0df117f140176ba74ae1be0c8" -L -o safetab-h-full-size-synthetic-data.zip
```

- To unzip the zip file:

```bash
unzip safetab-h-full-size-synthetic-data.zip -d <path to directory>/safetab-h-full-size-input
```

The download file is `safetab-h-full-size-synthetic-data.zip` will contain the following input files that work with the CSV reader:

- `input_dir`: Directory containing iteration code files and a config containing input parameters for running SafeTab-H.
  - `ethnicity-characteristic-iterations.txt`
  - `race-and-ethnicity-code-to-iteration.txt`
  - `race-and-ethnicity-codes.txt`
  - `race-characteristic-iterations.txt`
  - `config.json`: SafeTab-H config file containing input parameters for running SafeTab-H (e.g. `privacy_budgets`, `state_filter_us`, `reader`, `privacy_defn`).
    This is the file that the SafeTab-H algorithm uses to get input parameters. The default parameters in this file are Rho zCDP parameters (same as in config_zcdp.json). The `state_filter_us` is set to all 50 states and DC.
  - `config_puredp.json`: SafeTab-H config file containing input parameters for running SafeTab under PureDP. Note: This file is not used by the SafeTab algorithm. In order to use these parameters in the SafeTab-H algorithm copy them to `config.json`.
  - `config_zcdp.json`: SafeTab-H config file containing input parameters for running SafeTab under Rho zCDP. Note: This file is not used by the SafeTab algorithm. In order to use these parameters in the SafeTab-H algorithm copy them to `config.json`.
- `GRF-C.txt`: A representation of the GRFC that is input to DAS.
- `household-records.txt`: A representation of the custom household records derived from the CEF Person file that is input to DAS.
- `pop-group-totals.txt`: The T1 output file from a SafeTab-P run on a 300 million record synthetic dataset.

See [SafeTab-H Spec Doc](safetab_h/SafeTab_H_Documentation.pdf) for a description of each file. See the [SafeTab-H Library `README`](safetab_h/README.md) for more input directory setup notes.
=======
>>>>>>> main
