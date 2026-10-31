# Additional scientific analyses

The package implements the seven nonempty predictor-block
coalitions for service-support (S), household context (C)
and stated priorities (P).

Three-block Shapley R² allocation averages each block's
incremental predictive contribution over all possible
coalition entry orders. It is not causal attribution.

Additional outcome specifications include CORE-7A
(excluding Acceptability), CORE-7B (excluding Customer
Service), population-priority weighting and household-
specific priority weighting. Population weights are
calculated on development observations only.

For the final N = 6,039 study analysis, **CORE-7A, CORE-7B and both priority-weighted WSQS outcomes use the same 25 service-support and context (SC) predictors**. Their held-out results and selected iterations are recorded in `analysis/results/04_numerical_results/modeling/STAGE05_SENSITIVITY_SATISFACTION_N6039/01_original_eight_sensitivity_model_metrics.csv`. The corresponding execution script is `analysis/source_code/WSQS_STAGE05_SENSITIVITY_SATISFACTION_N6039.py`.

The preserved general-purpose implementation in `src/wsqs/extended.py` includes the original 33-predictor SCP CORE-7B specification. That software configuration and its historical results are separate from the N = 6,039 Stage 05 outcome comparison. For the final study's CORE-7B result, use the Stage 05 script and aggregate above.

Feature sensitivity specifications comprise reported
income as a nominal predictor, alternative pressure
coding, respondent age and respondent gender.

Cross-domain models exclude service-support predictors
from the target domain. They retain other service-support
predictors, context and priorities. Each model uses its
observed domain outcomes and the common partition.

For a no-data demonstration:

    python run_extended.py --mode demo

For authorized empirical reproduction:

    python run_extended.py --mode reproduce --data /secure/analytical_master.csv --output /secure/new_results_folder

The command creates aggregate evaluation tables only.
It does not write household predictions, respondent
identifiers or survey microdata.

The demonstration and the original general-purpose routine are distinct from the N = 6,039 empirical Stage 05 execution. For the study's analysis, consult the stage scripts and verified aggregate results in `analysis/`.
