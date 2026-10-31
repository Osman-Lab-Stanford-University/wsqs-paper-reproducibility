# N = 6,039 WSQS computational analysis

This directory contains the executed analysis scripts, result sources, integrity records and exhibit dependencies used for the N6039 WSQS study.

## Analytical specifications

- Analytical sample: **6,039 households**, after 106 exclusions from the original 6,145.
- CORE-8 outcome: eight service domains on a 0–100 scale; 266 records contribute seven valid domains.
- Features: 22 service-support (S), three household-context (C), eight priorities (P).
- Frozen test N = **1,208**, development N = **4,831**; primary full-model CatBoost uses **500 trees**.
- Original sample SHA-256: `c72b8480d3cff223d1d4cf0ebcc29a3dd80204d563cd63a4295f6739f0ed77a1`.

## Analysis modules

| Stage | Analysis |
| --- | --- |
| 04 | Seven predictor-block models and eight cross-domain full-versus-CP comparisons |
| 05 | CORE-7 and priority-weighting outcome sensitivity, feature sensitivities and Q6.1 profiles |
| 06 | Ten matched validation splits, benchmarking and paired performance |
| 07 | Priority-null inference with B = 1,000 model refits |
| 08 | CatBoost SHAP attribution and B = 1,000 held-out bootstrap resamples |
| 09 | Four service-response profiles and B = 200 development bootstrap refits |
| 10 | Reliability, 30-item factor structure, weighting and provider-applicability analysis |
| 11–14 | Numerical-source validation, source integration and figure data |
| 16 | Reconciliation of Table 8 held-out observed/predicted means |

Execution code is in [`source_code/`](source_code/). Numerical outputs are under [`results/`](results/), and checksum records are in [`audits/`](audits/). The master figure and table inventory is [`audits/02_exhibit_56_dependency_register.csv`](audits/02_exhibit_56_dependency_register.csv).

To rerun the original empirical analyses, place the authorized private input data and model checkpoints into the analysis workspace required by the stage scripts. Python 3.11 and the documented package environment are used for the scientific runs. For general synthetic execution, use the repository root `run.py` and `run_extended.py`.

The source scripts, verified results and accompanying audits in this directory document the N = 6,039 study analysis. Historical reference aggregates are retained separately at the repository root for computational provenance.
