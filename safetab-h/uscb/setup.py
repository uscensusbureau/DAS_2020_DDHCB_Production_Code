# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['tmlt', 'tmlt.safetab_h']

package_data = \
{'': ['*'], 'tmlt.safetab_h': ['resources/config/input/*',
                               'resources/config/output/*']}

install_requires = \
['tmlt.common>=0.8.0,<0.9.0',
 'tmlt.safetab_utils==0.7.11',
 'tmlt.analytics==0.8.3',
 'pyspark[sql]>=3.0.0,<4.0.0',
 'smart-open==5.2.1',

]

extras_require = \
{':python_version < "3.8"': ['pandas>=1.2.0,<1.4.0',
                             'pyspark[sql]>=3.0.0,<3.4.0'],
 ':python_version >= "3.10" and python_version < "3.11"': ['pandas>=1.4.0,<2.0.0'],
 ':python_version >= "3.11"': ['pandas>=1.5.0,<2.0.0',
                               'pyspark[sql]>=3.4.0,<3.6.0'],
 ':python_version >= "3.8" and python_version < "3.10"': ['pandas>=1.2.0,<2.0.0'],
 ':python_version >= "3.8" and python_version < "3.11"': ['pyspark[sql]>=3.0.0,<3.6.0']}


setup_kwargs = {
    'name': 'tmlt-safetab-h',
    'version': '3.0.0',
    'description': 'SafeTab H for Detailed Race/AIANNH',
    'long_description': "# SafeTab-H",
    'author': 'None',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7.4,<3.12',
}


setup(**setup_kwargs)
