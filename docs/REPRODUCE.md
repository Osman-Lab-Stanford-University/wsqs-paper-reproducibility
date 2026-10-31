# Computational reproduction guide

The study computational analyses use **6,039 household observations** and the 0–100 eight-domain WSQS outcome. Analysis scripts, input specifications, aggregate outputs and integrity records are documented in [`../analysis/`](../analysis/README.md). The geographical and demographic coverage methods are under [`../coverage_analysis/`](../coverage_analysis/README.md).

## Verification using included files

Run these commands from the repository root using Python 3.11:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-demo.txt
PYTHONPATH=src python -m unittest discover -s tests -v
python run.py --mode demo
python run_extended.py --mode demo
python tools/verify_n6039_package.py
```

The example dataset is synthetic and contains 25 records. The offline package check validates the supplied scientific results, file integrity and manuscript tables.

## Empirical study analysis

The frozen analytical cohort contains 6,039 households. The development partition contains 4,831 households, including 3,864 fitting and 967 validation observations, and the held-out test partition contains 1,208 households. The forecasting inventory comprises 22 service-support responses, three household-context predictors and eight stated priorities.

For empirical reproduction with authorized inputs, follow the stage-specific source scripts in [`../analysis/source_code/`](../analysis/source_code/) and the input contracts in [`../analysis/README.md`](../analysis/README.md). Keep confidential analytical records, respondent identifiers and saved model checkpoints in the authorized research workspace. The distributed `analysis/results/` and `coverage_analysis/results/` directories contain aggregate results.
