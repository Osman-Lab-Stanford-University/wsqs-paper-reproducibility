# Water Service Quality Score (WSQS) | Study reproducibility

Computational methods, source code, analysis outputs and manuscript table/figure data accompanying the water-service-quality research by **Aadhityaa Mohanavelu and Khalid K. Osman**.

**Software implementation:** Aadhityaa Mohanavelu  
**Study authors:** Aadhityaa Mohanavelu and Khalid K. Osman  
**Study analysis:** N = 6,039 households

## Repository contents

| Location | Contents |
| --- | --- |
| [`src/wsqs/`](src/wsqs/) | CORE-8 scoring, feature encodings and prediction methods |
| [`analysis/source_code/`](analysis/source_code/) | Executed N6039 analysis and verification scripts |
| [`analysis/results/`](analysis/results/) | 55 checksum-verified modeling, inference, measurement and figure-source aggregate files |
| [`coverage_analysis/`](coverage_analysis/) | S4 coverage code and 13 aggregate demographic, geographic and reference-comparison results |
| [`publication/tables/`](publication/tables/) | N6039 main manuscript Tables 1–8 in Word and Markdown |
| [`publication/figures/`](publication/figures/) | Manuscript figure-to-source map and supporting numerical inputs |
| [`questionbook/`](questionbook/) and [`config/`](config/) | Survey-variable specification, coding and modeling configurations |
| [`example/`](example/), [`tests/`](tests/) | Synthetic verification dataset and unit tests |
| [`tools/verify_n6039_package.py`](tools/verify_n6039_package.py) | Offline integrity verification |

## Run the reproducibility checks

Use Python 3.11. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-demo.txt
PYTHONPATH=src python -m unittest discover -s tests -v
python run.py --mode demo
python run_extended.py --mode demo
python tools/verify_n6039_package.py
```

These demonstration commands use the included 25-record synthetic dataset. The N6039 computational procedures, execution sources, validation settings and aggregate outputs are organized in [`analysis/`](analysis/README.md). To execute the empirical analyses, use the authorized study data in a secure workspace, following the stage scripts and the [`N6039 computational analysis guide`](analysis/README.md).

## Study analysis

The primary study analyzes **6,039 households** and uses the eight-domain WSQS on a 0–100 scale. The forecasting inventory contains **33 predictors**: 22 service-support variables, three household-context variables and eight stated-priority ratings.

The repository includes predictor-block modeling, ten-split performance comparisons, cross-domain prediction, outcome-weighting sensitivities, the 1,000-refit priority-null analysis, SHAP attribution and bootstrap intervals, service-response analysis with 200 development refits, factor and reliability analysis, provider-applicability diagnostics and S4 study-coverage analyses.

The manuscript's main tables and verified figure-source aggregates are available under [`publication/`](publication/). The verified numerical results supporting the N = 6,039 manuscript are under [`analysis/results/`](analysis/results/).

## Data access

Study participant-level data and model checkpoints are maintained in the authorized research environment. This repository provides the software, aggregate analysis outputs and a synthetic example. See [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) for the data-access description.

## Citation

Mohanavelu, A., and Osman, K. K. (2026). *Operational Service Conditions are the primary predictors of Household Water Service Quality Across Heterogeneous Provision Systems*. Research Square. https://doi.org/10.21203/rs.3.rs-10888037/v1

Cite the associated research article and link to this repository when using its computational materials. Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## Licensing

The repository uses the [MIT License](LICENSE) for original software and [CC BY 4.0](LICENSE-CONTENT.md) for eligible original documentation and aggregate content. Details appear in [`LICENSING.md`](LICENSING.md) and [`RIGHTS_AND_ATTRIBUTION.md`](RIGHTS_AND_ATTRIBUTION.md).
