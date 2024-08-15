#!/usr/bin/env bash
# Assumes PySpark will use Python 3.11.
# If it does not by default, you can set:
# --conf spark.pyspark.python=/usr/local/bin/python3.11
# or similar.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR/../tmlt/safetab_h/

spark-submit \
        --properties-file resources/spark_configs/spark_local_properties.conf \
        safetab-h.py validate resources/toy_dataset/input_dir_puredp resources/toy_dataset
