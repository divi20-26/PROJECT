# Exploratory Later-Vintage Evaluation

**Exploratory only; not a confirmatory holdout.** Outcome coverage, maturity, and default-rate summaries for these vintages were previously generated in `tables/mature_outcome_coverage_asof_20260331.csv`.

- Vintages: 2020, 2021, 2022
- Outcome snapshot: 2026-03-31
- Models: frozen pilot model artifacts trained on 2005-2014; no refitting or model selection on these vintages.
- Thresholds: previously selected using 2015-2016 validation data.
- Cohort: common complete-horizon test population for each horizon; payoff is a non-default competing outcome, and early other termination is excluded.
- Cost values are hypothetical normalized scenarios, not observed dollar losses.
- Active pilot settings now match the exact validation thresholds and 1,000-replicate bootstrap outputs; pre-audit metadata is retained in the settings JSON.

## Common-cohort metrics

| Horizon | Model | Strategy | N | Defaults | ROC-AUC | PR-AUC | Brier | ECE |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 12 | cause_specific_cox | delayed_entry_competing_risk | 148,299 | 1,147 | 0.7017 | 0.0176 | 0.007672 | 0.003794 |
| 12 | logistic_regression | complete_eligible | 148,299 | 1,147 | 0.6759 | 0.0141 | 0.007779 | 0.005509 |
| 12 | logistic_regression | matched_naive | 148,299 | 1,147 | 0.6767 | 0.0142 | 0.007797 | 0.005380 |
| 12 | logistic_regression | naive_full | 148,299 | 1,147 | 0.6773 | 0.0141 | 0.007793 | 0.005578 |
| 12 | xgboost | complete_eligible | 148,299 | 1,147 | 0.6934 | 0.0157 | 0.007762 | 0.005543 |
| 12 | xgboost | matched_naive | 148,299 | 1,147 | 0.6919 | 0.0156 | 0.007761 | 0.005549 |
| 12 | xgboost | naive_full | 148,299 | 1,147 | 0.6893 | 0.0150 | 0.007768 | 0.005831 |
| 24 | cause_specific_cox | delayed_entry_competing_risk | 148,245 | 2,156 | 0.7400 | 0.0397 | 0.014206 | 0.000570 |
| 24 | logistic_regression | complete_eligible | 148,245 | 2,156 | 0.7166 | 0.0338 | 0.014527 | 0.009135 |
| 24 | logistic_regression | matched_naive | 148,245 | 2,156 | 0.7124 | 0.0331 | 0.014576 | 0.009526 |
| 24 | logistic_regression | naive_full | 148,245 | 2,156 | 0.7115 | 0.0324 | 0.014644 | 0.010360 |
| 24 | xgboost | complete_eligible | 148,245 | 2,156 | 0.7479 | 0.0398 | 0.014353 | 0.006569 |
| 24 | xgboost | matched_naive | 148,245 | 2,156 | 0.7394 | 0.0375 | 0.014376 | 0.007602 |
| 24 | xgboost | naive_full | 148,245 | 2,156 | 0.7271 | 0.0352 | 0.014452 | 0.009340 |
| 36 | cause_specific_cox | delayed_entry_competing_risk | 148,221 | 3,053 | 0.7538 | 0.0605 | 0.019876 | 0.005853 |
| 36 | logistic_regression | complete_eligible | 148,221 | 3,053 | 0.7356 | 0.0553 | 0.020248 | 0.011231 |
| 36 | logistic_regression | matched_naive | 148,221 | 3,053 | 0.7281 | 0.0524 | 0.020460 | 0.013211 |
| 36 | logistic_regression | naive_full | 148,221 | 3,053 | 0.7267 | 0.0514 | 0.020653 | 0.014765 |
| 36 | xgboost | complete_eligible | 148,221 | 3,053 | 0.7665 | 0.0633 | 0.019880 | 0.004350 |
| 36 | xgboost | matched_naive | 148,221 | 3,053 | 0.7523 | 0.0572 | 0.020079 | 0.009451 |
| 36 | xgboost | naive_full | 148,221 | 3,053 | 0.7382 | 0.0527 | 0.020280 | 0.012500 |

## Lowest descriptive test cost

| Horizon | Scenario | Mean cost | Model | Strategy |
|---:|---|---:|---|---|
| 12 | review_to_missed_cost_0.001 | 0.002723 | logistic_regression | matched_naive |
| 12 | review_to_missed_cost_0.005 | 0.006078 | cause_specific_cox | delayed_entry_competing_risk |
| 12 | review_to_missed_cost_0.01 | 0.007217 | xgboost | matched_naive |
| 12 | review_to_missed_cost_0.025 | 0.007675 | cause_specific_cox | delayed_entry_competing_risk |
| 12 | review_to_missed_cost_0.05 | 0.007734 | cause_specific_cox | delayed_entry_competing_risk |
| 12 | review_to_missed_cost_0.1 | 0.007734 | cause_specific_cox | delayed_entry_competing_risk |
| 12 | review_to_missed_cost_0.25 | 0.007734 | cause_specific_cox | delayed_entry_competing_risk |
| 12 | review_to_missed_cost_0.5 | 0.007734 | cause_specific_cox | delayed_entry_competing_risk |
| 12 | review_to_missed_cost_1 | 0.007734 | cause_specific_cox | delayed_entry_competing_risk |
| 12 | review_to_missed_cost_10 | 0.007734 | cause_specific_cox | delayed_entry_competing_risk |
| 12 | review_to_missed_cost_100 | 0.007734 | cause_specific_cox | delayed_entry_competing_risk |
| 12 | review_to_missed_cost_2 | 0.007734 | cause_specific_cox | delayed_entry_competing_risk |
| 12 | review_to_missed_cost_25 | 0.007734 | cause_specific_cox | delayed_entry_competing_risk |
| 12 | review_to_missed_cost_5 | 0.007734 | cause_specific_cox | delayed_entry_competing_risk |
| 24 | review_to_missed_cost_0.001 | 0.001208 | logistic_regression | complete_eligible |
| 24 | review_to_missed_cost_0.005 | 0.005269 | xgboost | naive_full |
| 24 | review_to_missed_cost_0.01 | 0.008582 | cause_specific_cox | delayed_entry_competing_risk |
| 24 | review_to_missed_cost_0.025 | 0.012696 | xgboost | matched_naive |
| 24 | review_to_missed_cost_0.05 | 0.014305 | xgboost | complete_eligible |
| 24 | review_to_missed_cost_0.1 | 0.014543 | cause_specific_cox | delayed_entry_competing_risk |
| 24 | review_to_missed_cost_0.25 | 0.014543 | cause_specific_cox | delayed_entry_competing_risk |
| 24 | review_to_missed_cost_0.5 | 0.014543 | cause_specific_cox | delayed_entry_competing_risk |
| 24 | review_to_missed_cost_1 | 0.014543 | cause_specific_cox | delayed_entry_competing_risk |
| 24 | review_to_missed_cost_10 | 0.014543 | cause_specific_cox | delayed_entry_competing_risk |
| 24 | review_to_missed_cost_100 | 0.014543 | cause_specific_cox | delayed_entry_competing_risk |
| 24 | review_to_missed_cost_2 | 0.014543 | cause_specific_cox | delayed_entry_competing_risk |
| 24 | review_to_missed_cost_25 | 0.014543 | cause_specific_cox | delayed_entry_competing_risk |
| 24 | review_to_missed_cost_5 | 0.014543 | cause_specific_cox | delayed_entry_competing_risk |
| 36 | review_to_missed_cost_0.001 | 0.001208 | xgboost | naive_full |
| 36 | review_to_missed_cost_0.005 | 0.005188 | xgboost | complete_eligible |
| 36 | review_to_missed_cost_0.01 | 0.008837 | xgboost | complete_eligible |
| 36 | review_to_missed_cost_0.025 | 0.014775 | xgboost | complete_eligible |
| 36 | review_to_missed_cost_0.05 | 0.018960 | xgboost | complete_eligible |
| 36 | review_to_missed_cost_0.1 | 0.020523 | logistic_regression | complete_eligible |
| 36 | review_to_missed_cost_0.25 | 0.020598 | cause_specific_cox | delayed_entry_competing_risk |
| 36 | review_to_missed_cost_0.5 | 0.020598 | cause_specific_cox | delayed_entry_competing_risk |
| 36 | review_to_missed_cost_1 | 0.020598 | cause_specific_cox | delayed_entry_competing_risk |
| 36 | review_to_missed_cost_10 | 0.020598 | cause_specific_cox | delayed_entry_competing_risk |
| 36 | review_to_missed_cost_100 | 0.020598 | cause_specific_cox | delayed_entry_competing_risk |
| 36 | review_to_missed_cost_2 | 0.020598 | cause_specific_cox | delayed_entry_competing_risk |
| 36 | review_to_missed_cost_25 | 0.020598 | cause_specific_cox | delayed_entry_competing_risk |
| 36 | review_to_missed_cost_5 | 0.020598 | cause_specific_cox | delayed_entry_competing_risk |

## Paired uncertainty and segments

Paired, vintage-stratified bootstrap intervals use 1,000 replicates. Intervals are exploratory and unadjusted; inference is conditional on the 2020–2022 vintage mix.
Intervals excluding zero by metric: {"brier_score": 18, "decision_cost": 21, "pr_auc": 14, "roc_auc": 14}.
Vintage/FICO/occupancy segment summaries are in `segment_metrics.csv`; paired comparison intervals are in `paired_bootstrap.csv`.

These descriptive later-vintage results do not support confirmatory inference: the vintages' outcomes were previously inspected, and the tested cost scenarios are hypothetical.
