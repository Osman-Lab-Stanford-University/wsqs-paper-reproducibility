# Main manuscript Tables 1–8 | N = 6,039

All values are source-derived from the Stage 14 verified aggregate handoff, except the six Table 8 means verified by the Stage 16 user-run private-checkpoint audit.

## Table 1. CORE-8 domain and aggregate WSQS scores.

| Measure | Mean | SD |
| --- | --- | --- |
| Affordability | 68.83 | 27.10 |
| Access | 75.26 | 28.00 |
| Reliability | 76.03 | 25.52 |
| Efficiency | 71.22 | 23.26 |
| Acceptability | 64.95 | 30.62 |
| Perceived quality | 72.94 | 22.68 |
| Quantity | 75.53 | 24.41 |
| Customer service | 68.85 | 24.51 |
| Equal-weight WSQS | 71.74 | 15.82 |

Note. N = 6,039. Perceived quality uses 6,035 valid ratings; Customer service uses 5,777. The aggregate retains all 6,039 households. Sources: Stage 10 domain-level summary and CORE-8 measurement JSON.

## Table 2. Priority-weight summary by WSQS domain.

| Domain | Mean priority | SD | Median | P25 | P75 | Population weight |
| --- | --- | --- | --- | --- | --- | --- |
| Affordability | 6.693 | 2.258 | 7 | 5 | 8 | 0.120 |
| Access | 6.942 | 2.136 | 7 | 5 | 8 | 0.124 |
| Reliability | 6.901 | 2.140 | 7 | 5 | 8 | 0.124 |
| Efficiency | 6.915 | 2.155 | 7 | 5 | 9 | 0.124 |
| Acceptability | 7.061 | 2.126 | 7 | 6 | 9 | 0.126 |
| Perceived quality | 7.070 | 2.115 | 7 | 6 | 9 | 0.127 |
| Quantity | 7.122 | 2.084 | 7 | 6 | 9 | 0.128 |
| Customer service | 7.114 | 2.130 | 7 | 6 | 9 | 0.127 |

Note. Priorities are 0–10 ratings (N = 6,039). Percentile values recomputed from the archived 0–10 response frequencies. Population weights are estimated from the full analytical sample. Stage 13: 4,920 households (81.47%) assigned different ratings to at least two domains.

## Table 3. Robustness across WSQS weighting schemes.

| Metric | Equal weight | Population priority | Individual priority |
| --- | --- | --- | --- |
| N | 6039 | 6039 | 6039 |
| Mean WSQS | 71.742 | 71.745 | 71.770 |
| SD | 15.820 | 15.837 | 15.951 |
| Pearson correlation with equal weight | 1.0000 | 0.9999 | 0.9908 |
| Spearman correlation with equal weight | 1.0000 | 0.9985 | 0.9916 |
| Mean signed shift | 0.000 | 0.003 | 0.028 |
| Mean absolute shift | 0.000 | 0.133 | 0.948 |
| 95th-percentile absolute shift | 0.000 | 0.338 | 3.694 |

Note. N = 6,039 for all three weighting schemes. Population-priority weights are calculated from the sample-average ratings and renormalized across applicable domains. Score shifts are relative to equal-weight CORE-8 (WSQS points). Descriptive full-sample weights are distinguished from development-only weights used for prediction.

## Table 4. Held-out model performance across nested predictor sets.

| Model | Predictor blocks | Predictors | Test RMSE | Test MAE | Test R² | Spearman ρ | Repeated-split RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | Mean-only baseline | 0 | 15.8692 | 12.9861 | -0.000140 | — | 15.846 ± 0.202 |
| M1 | Service-support | 22 | 10.6924 | 8.1706 | 0.545951 | 0.726623 | 10.559 ± 0.224 |
| M2 | Service-support + context | 25 | 10.7309 | 8.1905 | 0.542675 | 0.724357 | 10.560 ± 0.213 |
| M3 | Service-support + context + priorities | 33 | 10.6570 | 8.1663 | 0.548950 | 0.729408 | 10.516 ± 0.211 |

Note. Held-out n = 1,208. Repeated-split quantities summarize 10 matched partitions. R² values retained at six decimals to distinguish closely related models. Sources: Stage 04 primary metrics and Stage 06 repeated-split summary.

## Table 5. Grouped SHAP and order-independent block Shapley contributions.

| Predictor block | Features | Gross abs. SHAP | Gross share (%) | Net abs. SHAP | Allocated R² | Allocated share (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Service-support | 22 | 20.527141 | 94.330 | 9.276436 | 0.509049 | 92.708 |
| Household context | 3 | 0.271185 | 1.246 | 0.187219 | 0.004183 | 0.762 |
| Stated priorities | 8 | 0.962764 | 4.424 | 0.652397 | 0.035859 | 6.531 |
| Total | 33 | 21.761089 | 100.000 | — | 0.549091 | 100.000 |

Note. SHAP computed on the unchanged 1,208-household test sample; 1,000 held-out resamples provide conditional intervals elsewhere. Block Shapley distributes full-model R² improvement over the mean baseline. Sources: Stages 04, 08 and 13.

## Table 6. Top predictors by mean absolute SHAP in the final M3 model.

| Rank | Predictor | Survey item | Block | Mean abs. SHAP | Share (%) |
| --- | --- | --- | --- | --- | --- |
| 1 | Value for money | Q2.5 | Service-support | 2.592 | 11.9 |
| 2 | Complaint resolution | Q9.2 | Service-support | 2.497 | 11.5 |
| 3 | Appearance acceptability | Q6.2 | Service-support | 2.007 | 9.2 |
| 4 | Water clarity | Q7.3 | Service-support | 1.617 | 7.4 |
| 5 | Infrastructure maintenance | Q5.3 | Service-support | 1.507 | 6.9 |
| 6 | Water pressure | Q4.2 | Service-support | 1.253 | 5.8 |
| 7 | Shortage frequency | Q8.3 | Service-support | 1.247 | 5.7 |
| 8 | Change in water quality | Q7.4 | Service-support | 0.892 | 4.1 |
| 9 | Water odor | Q6.4 | Service-support | 0.831 | 3.8 |
| 10 | Backup during failure | Q4.4 | Service-support | 0.774 | 3.6 |
| 11 | Water taste | Q6.3 | Service-support | 0.743 | 3.4 |
| 12 | Advance disruption notification | Q4.3 | Service-support | 0.668 | 3.1 |
| 13 | Daily water-access time | Q3.3 | Service-support | 0.594 | 2.7 |
| 14 | Weekly municipal-water volume | Q8.2 | Service-support | 0.572 | 2.6 |
| 15 | Municipal-water supply frequency | Q3.6 | Service-support | 0.571 | 2.6 |

Note. Predictor labels follow original survey-item definitions. Importance describes model attribution and is not causal. Stage 08 held-out n = 1,208. Top 15 account for 84.397% of total mean absolute SHAP.

## Table 7. Priority-conditioned service-response contrasts on the 0–100 WSQS scale.

| Service condition | Evaluation N | Severity 5 − 1 (95% CI) | Severity 3 − 1 (95% CI) | Largest absolute adjacent change (95% CI) | Priority modification at 5 − 1 (95% CI) |
| --- | --- | --- | --- | --- | --- |
| Complaint resolution | 1208 | -7.6724 [-9.0402, -6.2707] | -6.2284 [-7.4868, -5.2998] | 3.7216 [3.0610, 4.5480] | -0.0496 [-0.2920, 0.0710] |
| Value for money | 1208 | -7.7289 [-9.0068, -6.1368] | -6.5306 [-7.2074, -5.2786] | 3.7403 [2.9183, 4.2104] | -0.1035 [-0.4000, 0.1680] |
| Water pressure | 943 | -5.5799 [-7.1999, -3.7859] | -3.7685 [-4.6906, -2.2651] | 2.2169 [1.1227, 2.9359] | 0.0184 [-0.0797, 0.4377] |
| Shortage frequency | 1208 | -4.3127 [-5.9113, -2.6565] | -2.9733 [-4.1053, -2.0031] | 1.5353 [0.6237, 2.4029] | 0.1037 [-0.1020, 0.6502] |

Note. Model-derived changes in WSQS points. The largest absolute adjacent changes occur at severity 2–3 for complaint resolution and value for money, and 1–2 for water pressure and shortage frequency. Intervals are 95% percentiles from 200 development-sample model refits with 500 fixed trees. Water-pressure eligibility N = 943. Priority modification is the high-priority minus low-priority fitted 5-minus-1 response contrast. All four priority-modification intervals include zero.

## Table 8. Robustness across alternative WSQS outcome definitions.

| WSQS outcome | Test N | RMSE | MAE | R² | Spearman ρ | Observed mean | Predicted mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Equal-weight CORE-8 | 1208 | 10.7309 | 8.1905 | 0.542675 | 0.724357 | 71.8920 | 71.4832 |
| Population-priority CORE-8 | 1208 | 10.6796 | 8.1563 | 0.548139 | 0.729961 | 71.8876 | 71.6139 |
| Individual-priority CORE-8 | 1208 | 10.8149 | 8.2331 | 0.535322 | 0.720038 | 71.8935 | 71.6053 |

Note. Same frozen 1,208-household test set for all outcome definitions. Each model uses 22 service-support and 3 household-context predictors; stated priorities are excluded as predictors and used only to construct the weighted outcomes. Predicted and observed means verified from the saved Stage 04/05 checkpoint files in the completed Stage 16 audit; zero model refitting. All scores on a 0–100 WSQS scale.
