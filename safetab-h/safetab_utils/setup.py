# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['tmlt', 'tmlt.safetab_utils']

package_data = \
{'': ['*'], 'tmlt.safetab_utils': ['resources/test/*']}

install_requires = \
['parameterized>=0.7.4,<0.8.0',
 'pyarrow>=14.0.1,<15.0.0',
 'pyspark[sql]>=3.0.0,<4.0.0',
 'pytest>=7.1.2,<8.0.0',
 'tmlt.analytics>=0.7.0,<0.9.0',
 'tmlt.common>=0.8.0,<0.9.0',
 'tmlt.core>=0.10.0,<0.13.0']

extras_require = \
{':python_version < "3.8"': ['numpy>=1.21.0,<1.22.0', 'pandas>=1.2.0,<1.4.0'],
 ':python_version >= "3.10" and python_version < "3.11"': ['pandas>=1.4.0,<2.0.0'],
 ':python_version >= "3.11"': ['pandas>=1.5.0,<2.0.0'],
 ':python_version >= "3.8" and python_version < "3.10"': ['pandas>=1.2.0,<2.0.0'],
 ':python_version >= "3.8" and python_version < "3.9"': ['numpy>=1.22.0,<1.25.0'],
 ':python_version >= "3.9"': ['numpy>=1.23.2,<=1.26.1']}

setup_kwargs = {
    'name': 'tmlt-safetab-utils',
    'version': '0.7.11',
    'description': 'SafeTab Utilities for Detailed Race/AIANNH',
    'long_description': "# SafeTab-Utils\n\nThis module primarily contains common utility functions used by different SafeTab products.\n\nThe SafeTab product produces differentially private tables of statistics (counts) of demographic and housing characteristics crossed with detailed races and ethnicities at varying levels of geography (national, state, county, tract, place and AIANNH).\n\n<placeholder: add notice if required>\n\nSee [CHANGELOG](CHANGELOG.md) for version number information and changes from past versions.\n\n## Testing\n\n*Fast Tests:*\n\n```\npytest test/unit test/system -m 'not slow'\n```\n\n*Slow Tests:*\n\nSlow tests are tests that we run less frequently because they take a long time to run, or the functionality has been tested by other fast tests.\n\n```\npytest test/unit test/system -m 'slow'\n```\n\n*Detailed test plan is included in [TESTPLAN](TESTPLAN.md)*\n\n\n### Warnings\n\n1. Pytest warning:\n\n```\nDeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.\n```\n\n\n2. In order to prevent the following warning:\n\n```\nWARN NativeCodeLoader: Unable to load native-hadoop library for your platform\n```\n\n`LD_LIBRARY_PATH` must be set correctly. Use the following:\n\n```bash\nexport LD_LIBRARY_PATH=/usr/lib/hadoop/lib/native/\n```\n\nIf `HADOOP_HOME` is set correctly (usually `/usr/lib/hadoop`), this may be replaced with\n\n```bash\nexport LD_LIBRARY_PATH=$HADOOP_HOME/lib/native/\n```\n\n2. Other known warnings:\n\n```\nFutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise\ncomparison\n```\n```\nUserWarning: In Python 3.6+ and Spark 3.0+, it is preferred to specify type hints for pandas UDF instead of specifying\npandas UDF type which will be deprecated in the future releases. See SPARK-28264 for more details.\n```\n```\nUserWarning: It is preferred to use 'applyInPandas' over this API. This API will be deprecated in the future releases.\nSee SPARK-28264 for more details.\n```\n\n## Additional resources\n\nWe also provide a standalone script for converting short form iteration mapping to long form.\n\nThe script is located in `tmlt/safetab_utils`.\n\nTo view the arguments for running,\n\n```bash\npython convert_short_form.py -h\n```\n\nExample cmd using all arguments,\n\n```bash\npython convert_short_form.py \\\n--short-form [INPUT_FILE] \\      # Name of short-form race file.\n--race-codes [RACE_CODES_FILE] \\ # Name of race codes file.\n--output [OUTPUT_FILE]           # Name of output file for long-form races.\n```\n",
    'author': 'None',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'python_requires': '>=3.9.0,<3.12',
}


setup(**setup_kwargs)
