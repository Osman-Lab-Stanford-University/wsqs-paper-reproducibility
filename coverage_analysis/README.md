# WSQS supplementary coverage analysis (N = 6,039)

This module documents the descriptive sample-coverage calculations accompanying the Water Service Quality Score study. It retains the original coverage measures and table categories, and adds state-level PIN coverage. It is separate from model development and does not fit or modify forecasting models.

## Cohort and definitions

- Source survey: 6,145 original households, with 106 excluded by the authoritative private ResponseId manifest to yield 6,039 analytical households.
- Household-size comparison: 6,013 nonmissing household-size responses, grouped into 1, 2, 3, 4, 5, 6 and 7+.
- Housing tenure: Owned outright, Owned with mortgage/loan, Rented, Other; original displayed panel combines the two ownership classes.
- Car ownership: binary presentation using the original coverage-analysis convention, where English-form blank entries contribute to the No car category. The questionnaire response option is "None". The available source comparison does not independently establish whether every blank reflects a literal "None" import, so the code and documentation retain this explicit operational convention.
- Primary domestic provision: six original Q1.1 categories without conflating water source and provider.
- Geographic coverage: normalized six-digit PIN entries; 2024 India-wide postal directory for PIN-to-state classification; original Tamil Nadu district/geographic lookup for mapping. Prefix-based state-only sensitivity is separate from full-PIN matches.
- Settlement: Global Human Settlement Layer GHS-SMOD 2025 (R2023A) matched to postal reference coordinates and population density from the 3x3 neighbourhood of GHS-POP. Postal reference points are not household locations.

## Inputs required locally (NOT included)

The raw survey workbook, the 106-ResponseId exclusion manifest, geographic and benchmark source workbooks, the 2024 India-wide postal CSV and PIN-level intermediate files are stored in the authorized study workspace and are not included in this repository.

Suggested source layout on the analyst's authorized Mac (the individual source scripts preserve the local original path configuration):

- `Chapter 3 MANUAL RERUN/merged_cleaned_survey_data.xlsx`
- `Chapter 3 MANUAL RERUN/WSQS_MODEL_RERUN_EXCLUSION_IDS/WSQS_EXCLUDE_106_PRIMARY_N6039.csv`
- `Chapter 3 MANUAL RERUN/Demographid analysis/` (Census, NSS, NFHS, TN postal and GHS spatial references)
- `~/WSQS_PRIVATE_COVERAGE_N6039_NOT_FOR_PUBLIC/all_india_pincode_mirror_2024.csv`

## Analysis files

1. `source_code/01_household_composition_N6039.py`: cohort validation and household category harmonization (including tenure and car-response audit). Requires openpyxl.
2. `source_code/02_PIN_to_state_N6039.py`: six-digit PIN normalization and 2024 postal-directory state classification. Writes respondent-level intermediate records to a private local folder only.
3. `source_code/03_state_prefix_sensitivity_N6039.py`: supplemental state-only inferences from state-exclusive three-digit prefixes. Never overwrites a six-digit PIN or a state-directory match.
4. `source_code/04_geographic_aggregates_N6039.py`: district, settlement and density aggregates from the already validated private 629-PIN geographic join.
5. `source_code/05_GHS_SMOD_Table_S31_N6039.py`: verifies S31 against the original district roster, exact-cell GHS-SMOD codes and 3x3 neighbourhood modes. Computes the 452/628 comparison, with a separate 629-PIN aggregated settlement denominator.

6. `source_code/06_Census_household_size_overlap_N6039.py`: independently calculates the Census household-size overlap and total variation distance.

These scripts are preserved as source-stage workflows and use validated intermediate files. The module requires access to the analyst's authorized input workbooks for full reruns. Do not commit private inputs or intermediate record-level outputs.

## Public repository contents

Include: code, methodology/reference documentation, aggregate tables S30, S31 and state table S32, Census/demographic aggregate distributions, validation summary and source/checksum manifest.

Exclude: all raw respondent records, ResponseIds, private PIN-level geocodes, respondent-to-PIN crosswalks, original or regenerated figures, Figure 1 and all plotting code, raster inputs, historical/experimental scripts and forecasting model files.

## Important numerical distinctions

- Full analytical N = 6,039
- Household-size-complete N = 6,013
- Six-digit formatted PIN records = 5,991 (817 distinct values)
- Tamil Nadu mapped N = 5,778, associated with 629 postal reference PIN locations in 37 of 38 districts (Chengalpattu not represented)
- Fine-grained GHS-SMOD sensitivity = 452 of 628 complete PIN locations = 71.97%; grouped settlement and density use all 629 locations
- 2024 directory matches = 5,869 households; state-only prefix sensitivity adds 88 (total 5,957). Those two assignment methods remain separate.

The new coverage analysis is descriptive and does not change the model sample, scores or predictive validation.
