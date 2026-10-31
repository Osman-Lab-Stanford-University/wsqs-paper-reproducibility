# Analysis coverage and repository contents

This repository provides the computational sources, verified aggregate results, manuscript tables and figure numerical sources for the N = 6,039 WSQS study analysis.

## Measurement and predictive modeling

- CORE-8 measurement, seven-domain applicability handling, priority-weighting comparisons and the 33-variable predictor specification
- Seven nonempty S/C/P predictor-block models with a frozen held-out set of 1,208 households
- Ten matched-split learner benchmarks and domain-specific cross-domain prediction models
- Three-block R² Shapley allocation and native CatBoost SHAP attribution with 1,000 held-out bootstrap resamples
- Priority-null analysis with 1,000 fitted repetitions
- Four service-response profiles with 200 development-refit intervals
- Internal measurement reliability, factor analysis and provider-applicability diagnostics

## Study coverage and publication materials

- S4 geographic and demographic analyses and 13 aggregate outputs in [`../coverage_analysis/`](../coverage_analysis/README.md)
- 55 N6039 modeling, inference, measurement and figure-input aggregate files in [`../analysis/results/`](../analysis/results/)
- Main manuscript Tables 1–8 in [`../publication/tables/`](../publication/tables/)
- Figure-source mapping and exhibit dependencies in [`../publication/figures/`](../publication/figures/) and `analysis/audits/`

## Reproducing and verifying

Use [`REPRODUCE.md`](REPRODUCE.md), [`../analysis/README.md`](../analysis/README.md) and [`../coverage_analysis/README.md`](../coverage_analysis/README.md) to identify input contracts and execution procedures. Run `python tools/verify_n6039_package.py` from the repository root for aggregate-source and package checks.

The `docs/FROZEN_ORIGINAL_SHA256SUMS.txt` historical checksum ledger, `analysis/audits/ORIGINAL_V01_FILE_SHA256.json` and `results/aggregate/` preserve the earlier N = 6,145 computational reference state. They are not the checksums or results to use for the final N = 6,039 study. For the current repository, use `SHA256SUMS.txt`, `tools/verify_n6039_package.py` and `analysis/results/`.
