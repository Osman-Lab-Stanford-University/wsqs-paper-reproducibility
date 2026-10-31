# Repository verification checks

Run these commands from the repository root after installing the dependencies described in [`../README.md`](../README.md).

1. `python tools/verify_n6039_package.py` verifies repository source checksums, the N6039 aggregate data and the S4 coverage records.
2. `PYTHONPATH=src python -m unittest discover -s tests -v` runs the original unit tests.
3. `python run.py --mode demo` and `python run_extended.py --mode demo` run the synthetic examples.
4. Inspect [`../publication/tables/`](../publication/tables/) and [`../publication/figures/FIGURE_SOURCE_MAP.md`](../publication/figures/FIGURE_SOURCE_MAP.md) with the manuscript.
5. Confirm that the repository contains only approved public files before creating the initial GitHub commit.

This repository is intended as a single study reproducibility resource. No GitHub Release or version tag is required.
