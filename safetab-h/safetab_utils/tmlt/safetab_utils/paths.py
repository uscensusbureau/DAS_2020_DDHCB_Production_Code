"""Paths and file names used by SafeTab."""

# Copyright 2024 Tumult Labs
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
ETHNICITY_ITERATIONS_FILENAME = "ethnicity-characteristic-iterations.txt"
RACE_ITERATIONS_FILENAME = "race-characteristic-iterations.txt"

INPUT_FILES_SAFETAB_H = [
    ETHNICITY_ITERATIONS_FILENAME,
    "GRF-C.txt",
    "household-records.txt",
    "pop-group-totals.txt",
    "race-and-ethnicity-code-to-iteration.txt",
    "race-and-ethnicity-codes.txt",
    RACE_ITERATIONS_FILENAME,
]
"""List of all expected SafeTab-H input files that have a csv format."""

INPUT_FILES_SAFETAB_P = [
    ETHNICITY_ITERATIONS_FILENAME,
    "GRF-C.txt",
    "person-records.txt",
    "race-and-ethnicity-code-to-iteration.txt",
    "race-and-ethnicity-codes.txt",
    RACE_ITERATIONS_FILENAME,
]
"""List of all expected SafeTab-P input files that have a csv format."""
