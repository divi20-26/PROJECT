# Macro-augmented pilot comparison

This is an isolated, exploratory model branch; frozen baseline artifacts are not overwritten.
Baseline and augmented XGBoost models use the same complete-eligible label strategy,
training/validation/test vintages, hyperparameters, seed, and macro-complete cohort.
The 2017–2019 test vintages were previously examined, so these comparisons are not confirmatory.

The exact origination date is unavailable in Release 47. The experiment uses first-payment
month minus two calendar months as an approximate decision-date proxy. Macro release dates
are conservatively modeled with fixed lags (UNRATE 7 days after month end, real GDP 45 days
after quarter end, and FHFA HPI 90 days after quarter end). FRED current-vintage values can
include later revisions; this is not a real-time-vintage backtest. All measures are national.

| Horizon | Metric | Baseline | Macro augmented | Difference (paired 95% interval) |
|---:|---|---:|---:|---:|
| 12 | roc_auc | 0.662324 | 0.656901 | -0.005423 [-0.011977, 0.000368] |
| 12 | pr_auc | 0.018480 | 0.019013 | 0.000533 [-0.000890, 0.002141] |
| 12 | brier_score | 0.009787 | 0.010715 | 0.000928 [0.000861, 0.001002] |
| 24 | roc_auc | 0.731818 | 0.716307 | -0.015511 [-0.018198, -0.013003] |
| 24 | pr_auc | 0.072610 | 0.063547 | -0.009064 [-0.012113, -0.006681] |
| 24 | brier_score | 0.025902 | 0.026635 | 0.000733 [0.000590, 0.000876] |
| 36 | roc_auc | 0.722746 | 0.720719 | -0.002027 [-0.003585, -0.000598] |
| 36 | pr_auc | 0.105385 | 0.099594 | -0.005791 [-0.008265, -0.003572] |
| 36 | brier_score | 0.041140 | 0.041074 | -0.000066 [-0.000253, 0.000113] |

Differences are empirical and should be judged on practical size, paired interval,
calibration, and the documented timing limitations. SHAP values are model associations,
not causal effects. See `macro_augmented_shap_importance.csv` for the macro indicator
contributions and `matched_cohort_report.csv` for sample exclusions.
