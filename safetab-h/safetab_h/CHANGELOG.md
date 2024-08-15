# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 3.0.0 - 2024-04-03
### Added
- Input validation now detects if you use the wrong pop-group-totals file.
- Assorted typo fixes, extra comments, and minor refactors for clarity.

### Changed
- Re-organized instructions to make them easier to follow.
- SafeTab-H now exits with a non-zero return code for all validation failures.
- Updated to tmlt.analytics 0.8.3 and tmlt.core 0.12.0 which incorporate an improved truncation hash function and Core documentation.

## 2.0.4 - 2023-12-08
### Changed
- Now requires PyArrow 14.0.1, to address CVE-2023-47248.

## 2.0.3 - 2023-11-08
### Changed
- Modified the bootstrap_script.sh script in the core directory to read the whl file from the provided arguement.
- Supports Python 3.11 and PySpark 3.4.0.

## 2.0.2 - 2023-08-18
### Fixed
- `DETAILED_ONLY = True` iterations no longer have all units filtered out.
- Otherwise-invalid pop groups created by coterminous geos are no longer tabulated at all (previously tabulated as 0+noise).
- Use new, faster, versions of Core and Analytics.

## 2.0.1 - 2023-08-01
### Changed
- Updated to the newest version of SafeTab-Utils, which fixes import errors Census had working with the CEF reader.

## 2.0.0 - 2023-07-27
### Changed
- Made readability changes like updating docstring formats and using string constants.
- Stop filtering total only (`DETAILED_ONLY = True`) iterations from SafeTab-H output.
- Add public installation instructions and reformat documentation to match SafeTab-P.
- Updated tmlt.common to 0.8.2 at a minimum.
- Updated tmlt.safetab_utils to 0.6.0 at a minimum.
- Switched from nose to pytest for running tests.
- Updated to support Python 3.9.
- Added new CEF reader import path, and updated zip script so `mock_cef_reader` is not included by default.
- Update to Core 0.10.1 and Analytics 0.7.3.

## 1.0.1 - 2023-02-17
### Changed
- Removed the suggestion to use extraLibraryPath options on EMR.
- Updated to tmlt.analytics 0.6.0 from 0.5.1

### Fixed
- Accuracy report (and related tests) now complete under puredp as well as zcdp.

## 1.0.0 - 2023-01-27
### Added
- Implement adaptivity - for each population group, we determine how granular the T3 or T4 breakdown should be based on the noisy T1 count from SafeTab-P.
- Output "full tables" - split the output into separate files based on level of detail, and postprocess to add less detailed data cells.

### Changed
- Added adaptivity based on SafeTab-P output.
- Fixed test cases in regards to tabulating distinct population groups.
- Changed reader interface to accept `safetab-h` as an additional required input parameter.
- Removed code "99999" from the output for PLACE.
- Update tmlt.core and switched to a wheel install process. 
- Updated tmlt.analytics.
- Change base stability of unit dataframe from 2 to 1.
- Update accuracy report to use privacy budgets given in the config instead of dividing the budget equally, and report the MOE instead of the absolute error.
- Minor style changes to address Galois feedback.
- When a race code is mapped to multiple iteration codes in the same level, update error message to be more descriptive and include a list of offending codes.

## 0.1.1 - 2022-09-09
### Fixed
- Updated tmlt.core and tmlt.analytics to fix a bug where queries failed to evaluate. 

## 0.1.0 - 2022-07-27
### Added
- New system test for race and ethnicity disjointness input validation.
- New project to house SafeTab-H product - SafeTab-H command-line, validation, DP and non-DP algorithms.
- Added Tract and Place support to PR runs of safetab-h.
- Add `--validate-private-output` flag to private `execute` mode in CLI.

### Fixed
- Fixed output validation config for HOUSEHOLD_TYPE and HOUSEHOLD_TENURE columns.
