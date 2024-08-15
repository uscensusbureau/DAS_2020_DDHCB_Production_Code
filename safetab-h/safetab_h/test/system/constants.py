"""Constants that are shared across tests."""

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

import os

# A list of all output directories in which safetab-h will write a csv file.
# All paths are relative to the overall output directory passed to the program.
SAFETAB_H_OUTPUT_FILES = [
    os.path.join("t3", "T03001"),
    os.path.join("t3", "T03002"),
    os.path.join("t3", "T03003"),
    os.path.join("t3", "T03004"),
    os.path.join("t4", "T04001"),
    os.path.join("t4", "T04002"),
]
