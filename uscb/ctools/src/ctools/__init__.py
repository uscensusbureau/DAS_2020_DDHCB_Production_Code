import os

MAJOR = "3"
MINOR = "2"
PATCH = "3"
BUILD = "2"

__version__ = f"{MAJOR}.{MINOR}.{PATCH}.{BUILD}"

# Check for the AWS default region env var and set it if it is not already set
AWS_DEFAULT_REGION = 'AWS_DEFAULT_REGION'
DEFAULT_REGION = 'us-gov-west-1'

if AWS_DEFAULT_REGION not in os.environ:
    os.environ[AWS_DEFAULT_REGION] = DEFAULT_REGION
