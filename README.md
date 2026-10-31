# Water Service Quality Score (WSQS)

**Measuring and forecasting household-experienced water service quality**

Research code, verified numerical results, manuscript tables and figure-source data accompanying the study by **Aadhityaa Mohanavelu and Khalid K. Osman** at Stanford University.

Water-service assessments frequently describe the infrastructure available to a household or its principal water source. These measures do not fully capture the quality of service that people experience. The Water Service Quality Score (WSQS) brings together eight dimensions of reported service experience and examines which observed service conditions are most informative for predicting variation in that score.

This repository documents the **6,039-household study analysis**, including score construction, predictor definitions, model evaluation, uncertainty analyses, measurement diagnostics and study-coverage calculations. It also provides executable synthetic examples and integrity checks for the archived scientific results.

## Study at a glance

| Component | Study specification |
| --- | --- |
| Setting | Household water-service survey in India, primarily Tamil Nadu |
| Analytical sample | 6,039 households |
| Primary outcome | Equal-weight CORE-8 WSQS, scaled 0–100 |
| Service dimensions | Affordability, access, reliability, efficiency, acceptability, perceived quality, quantity and customer service |
| Forecasting predictors | 22 service-support variables, 3 household-context variables and 8 stated priorities |
| Model evaluation | Frozen held-out sample of 1,208 households and 10 matched-split comparisons |
| Methods | CatBoost, learner benchmarks, block Shapley R², SHAP, priority-null refits, service-response analysis and measurement diagnostics |

### Measuring service quality

CORE-8 uses one questionnaire assessment from each of the eight service dimensions. Each response is coded on a **1–5 severity scale**, where 1 is most favorable. Each applicable domain is transformed to a 0–100 score using `25 × (5 − severity)`, and the primary household WSQS is the equal-weight mean across valid, applicable domains. A higher WSQS indicates better reported service conditions. The analytical cohort contains **266 households contributing seven valid domains**; unavailable or inapplicable responses are not scored as adverse experiences.

The analysis also examines population-priority and household-priority weighting as alternative outcome definitions, while keeping the equal-weight CORE-8 score as the primary outcome. Survey-item definitions are in [`questionbook/`](questionbook/) and scoring code is in [`src/wsqs/`](src/wsqs/).

### Selected empirical results

The following figures are drawn from the verified N = 6,039 analysis files, not the synthetic demonstration.

| Result | Verified estimate |
| --- | ---: |
| Mean equal-weight WSQS | **71.74 / 100** |
| Held-out CatBoost full-model R² | **0.54895** |
| Held-out full-model RMSE | **10.657 WSQS points** |
| Mean full-model R² over 10 matched splits | **0.55901** |
| Service-support share of total *absolute* SHAP attribution | **94.33%** |
| Households reporting different priorities across domains | **4,920 / 6,039 (81.47%)** |

Service-support responses contain most of the predictive information among the variables evaluated, with prominent model contributions from value for money, complaint resolution, appearance acceptability, water clarity, maintenance, pressure and shortage experience. SHAP values describe predictive attribution, not causal effects. Full numerical results, uncertainty intervals and alternative model specifications are included in the repository.

## Repository guide

| Location | What you will find |
| --- | --- |
| [`src/wsqs/`](src/wsqs/) | CORE-8 scoring, feature construction, response coding and model utilities |
| [`questionbook/`](questionbook/) and [`config/`](config/) | Survey-item mappings, predictor inventory and model specifications |
| [`analysis/source_code/`](analysis/source_code/) | Executed N = 6,039 analysis and verification scripts |
| [`analysis/results/`](analysis/results/) | **55** verified aggregate result and figure-data files across modeling, measurement and inference |
| [`analysis/audits/`](analysis/audits/) | Source provenance, numerical-audit records and the 56-exhibit dependency register |
| [`coverage_analysis/`](coverage_analysis/) | Demographic and geographic coverage methods, six scripts and **13** aggregate outputs |
| [`publication/tables/`](publication/tables/) | Main manuscript **Tables 1–8**, in Word and Markdown |
| [`publication/figures/`](publication/figures/) | Mapping from manuscript Figures 1–8 to their underlying numerical sources |
| [`example/`](example/) and [`tests/`](tests/) | Fully synthetic 25-record example and automated scientific unit tests |
| [`results/aggregate/`](results/aggregate/) | Separately labeled historical aggregate reference results |
| [`tools/verify_n6039_package.py`](tools/verify_n6039_package.py) | Offline source-integrity and aggregate-validation checks |

For the full computational workflow, start with [`analysis/README.md`](analysis/README.md) and [`docs/REPRODUCE.md`](docs/REPRODUCE.md). Supplementary coverage methods are described in [`coverage_analysis/README.md`](coverage_analysis/README.md). The [figure-source map](publication/figures/FIGURE_SOURCE_MAP.md) identifies the files underlying the manuscript panels; finalized graphic artwork is maintained separately with the manuscript.

## Quick start

Use **Python 3.11** and run the following from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-demo.txt

PYTHONPATH=src python -m unittest discover -s tests -v
python run.py --mode demo
python run_extended.py --mode demo
python tools/verify_n6039_package.py
```

The first three execution commands test the scoring, feature construction and modeling interfaces using the included **25 synthetic households**. The final command verifies the archived repository files, N = 6,039 result aggregates, coverage-analysis outputs and main manuscript tables against their checksums and source manifests.

### Reproducing the study analyses

The empirical workflows are organized by analytical stage under [`analysis/source_code/`](analysis/source_code/). They cover seven predictor-block specifications, eight cross-domain comparisons, outcome and feature sensitivities, 10-split model benchmarking, 1,000 priority-null fits, native SHAP attribution with 1,000 held-out bootstrap resamples, 200 service-response refits, factor and reliability analysis and numerical source reconciliation.

The household-level study inputs and fitted checkpoints remain in the authorized research environment. Researchers with access to those inputs can use the documented stage scripts and configurations to rerun the analyses in a separate secure workspace. The repository itself makes the **source code, synthetic demonstration, aggregate outputs, tables and source-verification checks** available. See [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) and [`docs/INPUT_SCHEMA.md`](docs/INPUT_SCHEMA.md).

## Manuscript and companion toolkit

**Associated research**  
Mohanavelu, A., and Osman, K. K. (2026). *Operational Service Conditions are the primary predictors of Household Water Service Quality Across Heterogeneous Provision Systems*. Research Square. [https://doi.org/10.21203/rs.3.rs-10888037/v1](https://doi.org/10.21203/rs.3.rs-10888037/v1)

The linked Research Square preprint presents an earlier manuscript cohort. The verified numerical files in this repository document the **N = 6,039** analysis. Please cite the study and identify this repository when using its computational materials. GitHub citation metadata is provided in [`CITATION.cff`](CITATION.cff).

The separate [international WSQS toolkit](https://github.com/Osman-Lab-Stanford-University/wsqs-toolkit) provides a general-purpose implementation for questionnaire and WSQS use beyond this study.

## Authorship, licensing and data

**Software and repository implementation:** Aadhityaa Mohanavelu  
**Study authors:** Aadhityaa Mohanavelu and Khalid K. Osman  
**Research affiliation:** Stanford University

Original software is licensed under [MIT](LICENSE). Eligible original documentation and aggregate research content are licensed under [CC BY 4.0](LICENSE-CONTENT.md). See [`LICENSING.md`](LICENSING.md) and [`RIGHTS_AND_ATTRIBUTION.md`](RIGHTS_AND_ATTRIBUTION.md) for the allocation of rights and scholarly attribution.

No observed household records, respondent identifiers or household-level prediction files are published here. See [`DATA_PRIVACY.md`](DATA_PRIVACY.md) for the repository's data-handling provisions.
