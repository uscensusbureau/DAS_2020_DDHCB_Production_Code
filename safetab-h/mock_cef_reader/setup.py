# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['phsafe_safetab_reader']

package_data = \
{'': ['*']}

setup_kwargs = {
    'name': 'tmlt-mock-cef-reader',
    'version': '0.4.1',
    'description': 'Mock CEF Reader',
    'long_description': '# Mock CEF Reader\nTumult software for the Decennial Census supports two different reader modules: one for reading CSV files, and one for reading CEF files. The CEF reader was developed separately. This mock CEF reader allows software to be tested without the real CEF reader. Rather than reading CEF files, it wraps the CSV reader in the same package as the CEF reader uses.',
    'author': 'None',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'python_requires': '>=3.7.4,<4.0.0',
}


setup(**setup_kwargs)
