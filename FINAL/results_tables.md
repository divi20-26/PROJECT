# FINAL — Consolidated research results

This summary is generated from existing result files. Every row in `consolidated_results.csv` includes its source path. `ppt_results.csv` is a compact presentation table. Existing results, scripts, models, and predictions were not changed.

> **Interpretation:** 2017–2019 and 2020–2022 have previously been examined. All reported performance results from those vintages are exploratory, not untouched confirmatory evidence.

## 1. Training label strategies

| Horizon | Strategy | Training loans | Defaults | Training default prevalence | Naive false negatives |
| --- | --- | --- | --- | --- | --- |
| 12m | naive_full | 491,538 | 2,202 | 0.0045 | 45 |
| 12m | complete_eligible | 427,298 | 2,129 | 0.0050 | — |
| 12m | matched_naive | 427,298 | 2,085 | 0.0049 | — |
| 24m | naive_full | 491,538 | 7,342 | 0.0149 | 285 |
| 24m | complete_eligible | 382,501 | 7,013 | 0.0183 | — |
| 24m | matched_naive | 382,501 | 6,796 | 0.0178 | — |
| 36m | naive_full | 491,538 | 13,086 | 0.0266 | 657 |
| 36m | complete_eligible | 341,780 | 12,390 | 0.0363 | — |
| 36m | matched_naive | 341,780 | 11,884 | 0.0348 | — |

Naive false negatives are reported from the retrospective label comparison; they are not the test-set model false negatives. The common mature test population has zero label disagreements by the comparison file.

## 2. Binary models — common mature 2017–2019 test cohort

| Horizon | Model | Training-label strategy | ROC-AUC | PR-AUC | Brier | ECE | Calibration slope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12m | Logistic Regression | naive_full | 0.6468 | 0.0176 | 0.0098 | 0.0081 | 0.3698 |
| 12m | Logistic Regression | complete_eligible | 0.6501 | 0.0185 | 0.0098 | 0.0080 | 0.3950 |
| 12m | Logistic Regression | matched_naive | 0.6508 | 0.0181 | 0.0098 | 0.0078 | 0.3806 |
| 12m | XGBoost | naive_full | 0.6682 | 0.0176 | 0.0098 | 0.0083 | 0.5481 |
| 12m | XGBoost | complete_eligible | 0.6623 | 0.0185 | 0.0098 | 0.0076 | 0.5239 |
| 12m | XGBoost | matched_naive | 0.6686 | 0.0184 | 0.0098 | 0.0078 | 0.5512 |
| 24m | Logistic Regression | naive_full | 0.7195 | 0.0709 | 0.0261 | 0.0214 | 0.6078 |
| 24m | Logistic Regression | complete_eligible | 0.7220 | 0.0698 | 0.0260 | 0.0201 | 0.6405 |
| 24m | Logistic Regression | matched_naive | 0.7196 | 0.0694 | 0.0260 | 0.0206 | 0.6275 |
| 24m | XGBoost | naive_full | 0.7250 | 0.0696 | 0.0262 | 0.0223 | 0.7231 |
| 24m | XGBoost | complete_eligible | 0.7318 | 0.0726 | 0.0259 | 0.0187 | 0.8102 |
| 24m | XGBoost | matched_naive | 0.7295 | 0.0719 | 0.0260 | 0.0200 | 0.8057 |
| 36m | Logistic Regression | naive_full | 0.7043 | 0.1003 | 0.0418 | 0.0341 | 0.5667 |
| 36m | Logistic Regression | complete_eligible | 0.7024 | 0.0979 | 0.0414 | 0.0297 | 0.5712 |
| 36m | Logistic Regression | matched_naive | 0.6999 | 0.0972 | 0.0417 | 0.0326 | 0.5630 |
| 36m | XGBoost | naive_full | 0.7059 | 0.0946 | 0.0421 | 0.0359 | 0.6558 |
| 36m | XGBoost | complete_eligible | 0.7227 | 0.1054 | 0.0411 | 0.0264 | 0.8367 |
| 36m | XGBoost | matched_naive | 0.7183 | 0.1020 | 0.0416 | 0.0316 | 0.8045 |

All model/strategy rows at a horizon share the same test cohort and labels. The target is default by horizon; voluntary payoff is a non-default competing outcome and early other termination is excluded. Calibration intercept and slope are in the consolidated CSV. Lower Brier/ECE is better; higher AUC is better.

## 3. Survival analysis

### Marginal-IPCW cumulative/dynamic metrics

| Exploratory cohort | Horizon | Observed N | Defaults | IPCW ROC-AUC | IPCW PR-AUC | IPCW Brier |
| --- | --- | --- | --- | --- | --- | --- |
| 2017-2019 | 12m | 144,083 | 1,420 | 0.6755 | 0.0208 | 0.0098 |
| 2017-2019 | 24m | 144,059 | 3,840 | 0.7161 | 0.0677 | 0.0258 |
| 2017-2019 | 36m | 144,052 | 6,230 | 0.7034 | 0.0986 | 0.0409 |
| 2020-2022 | 12m | 148,202 | 1,147 | 0.7017 | 0.0176 | 0.0077 |
| 2020-2022 | 24m | 148,148 | 2,156 | 0.7401 | 0.0397 | 0.0142 |
| 2020-2022 | 36m | 148,124 | 3,053 | 0.7538 | 0.0605 | 0.0199 |

These censor-aware metrics use a marginal independent-censoring assumption; voluntary payoff is treated as competing non-default. They are separate from the complete-outcome model comparison above.

### Cause-specific concordance

| Exploratory cohort | 36m concordance | N | Defaults | Payoff handling |
| --- | --- | --- | --- | --- |
| 2017-2019 | 0.6964 | 144,306 | 8,127 | Payoff treated as censored |
| 2020-2022 | 0.7481 | 148,543 | 3,847 | Payoff treated as censored |

This cause-specific Harrell concordance treats payoff as censored and is not a competing-risk concordance statistic.

## 4. SHAP explainability and cross-horizon stability

| Horizon | Top feature family | Mean |SHAP| | Mean signed SHAP | Scale |
| --- | --- | --- | --- | --- |
| 12m | property_state | 1.5034 | 1.5034 | raw log-odds margin |
| 12m | original_interest_rate | 0.9159 | -0.8958 | raw log-odds margin |
| 12m | classic_fico | 0.5558 | -0.3113 | raw log-odds margin |
| 12m | original_ltv | 0.2707 | 0.0170 | raw log-odds margin |
| 12m | channel | 0.2681 | 0.2681 | raw log-odds margin |
| 24m | original_interest_rate | 0.9609 | -0.9565 | raw log-odds margin |
| 24m | property_state | 0.8389 | 0.8389 | raw log-odds margin |
| 24m | classic_fico | 0.6804 | -0.3015 | raw log-odds margin |
| 24m | number_of_borrowers | 0.2749 | 0.2749 | raw log-odds margin |
| 24m | original_cltv | 0.2228 | -0.0333 | raw log-odds margin |
| 36m | original_interest_rate | 1.0608 | -1.0567 | raw log-odds margin |
| 36m | property_state | 0.7486 | 0.7486 | raw log-odds margin |
| 36m | classic_fico | 0.5893 | -0.2925 | raw log-odds margin |
| 36m | original_cltv | 0.2824 | -0.0288 | raw log-odds margin |
| 36m | number_of_borrowers | 0.2165 | 0.2165 | raw log-odds margin |

The SHAP values explain model outputs, not causal effects. Signed values are raw-margin (log-odds) contributions. Results are based on a fixed common sample of 500 mature loans per evaluation split.

| Horizon pair | Spearman rank correlation | Feature families |
| --- | --- | --- |
| 12m vs 24m | 0.930 | 23 |
| 12m vs 36m | 0.868 | 23 |
| 24m vs 36m | 0.955 | 23 |

Observed rank correlation range: **0.868–0.955**.

## 5. SHAP explanation clusters

| Horizon | Selected k | Validation silhouette | Test cluster sizes and mean risk | Validation seed ARI |
| --- | --- | --- | --- | --- |
| 12m | 2 | 0.4428 | C1: 332 (mean risk 0.0011), C2: 168 (mean risk 0.0042) | 1.000 |
| 24m | 2 | 0.4792 | C1: 251 (mean risk 0.0035), C2: 249 (mean risk 0.0122) | 1.000 |
| 36m | 2 | 0.4290 | C1: 276 (mean risk 0.0083), C2: 224 (mean risk 0.0280) | 1.000 |

Candidate k values 2–6 were compared using validation explanations only. Cluster sizes/risk are descriptive of the sampled exploratory test cohort. Cluster IDs are horizon-specific explanation patterns, not borrower types.

## 6. Macroeconomic augmentation

| Horizon | Metric | Baseline | Augmented | Augmented − baseline | Paired 95% interval |
| --- | --- | --- | --- | --- | --- |
| 12m | roc_auc | 0.6623 | 0.6569 | -0.0054 | [-0.0120, 0.0004] |
| 12m | pr_auc | 0.0185 | 0.0190 | 0.0005 | [-0.0009, 0.0021] |
| 12m | brier_score | 0.0098 | 0.0107 | 0.0009 | [0.0009, 0.0010] |
| 24m | roc_auc | 0.7318 | 0.7163 | -0.0155 | [-0.0182, -0.0130] |
| 24m | pr_auc | 0.0726 | 0.0635 | -0.0091 | [-0.0121, -0.0067] |
| 24m | brier_score | 0.0259 | 0.0266 | 0.0007 | [0.0006, 0.0009] |
| 36m | roc_auc | 0.7227 | 0.7207 | -0.0020 | [-0.0036, -0.0006] |
| 36m | pr_auc | 0.1054 | 0.0996 | -0.0058 | [-0.0083, -0.0036] |
| 36m | brier_score | 0.0411 | 0.0411 | -0.0001 | [-0.0003, 0.0001] |

Indicators: UNRATE (U.S. Bureau of Labor Statistics via FRED), GDPC1 (U.S. Bureau of Economic Analysis via FRED), USSTHPI (U.S. Federal Housing Finance Agency via FRED).

Origination-date proxy: Release 47 has first-payment month but no exact origination date. The macro experiment uses first_payment_month minus two calendar months as an origination-decision-date proxy; this is approximate and not an exact real-time reconstruction.

FRED revision limitation: FRED CSV provides current revised observations, not historical real-time data vintages. Historical GDP and other revised series may therefore contain information revised after the modeled prediction date.

Date alignment rule: first-payment month minus two months; fixed modeled release lags; backward as-of join; per-series maximum age; no future available_date

On this experiment and these inspected vintages, augmentation did not consistently improve the reported metrics. That is not evidence that macroeconomic features are universally ineffective: it is specific to these national series, this date proxy, the revised FRED snapshot, model settings, and exploratory cohorts.

## Interpretation and limitations

- Baseline and macro experiment results are distinct; macro metrics use the common macro-available cohort and must not be mixed with other score tables without noting their cohort definitions.
- The macro comparison uses national indicators and an approximate first-payment-month-minus-two-month decision-date proxy. Fixed availability lags are not exact historical release timestamps; FRED's current-vintage observations may include later revisions.
- SHAP reports model attribution, not cause or effect. Cluster quality scores are validation diagnostics and do not establish naturally occurring borrower groups.
- None of the previously inspected test vintages can support untouched confirmatory claims.
- Values marked **Unavailable** in the consolidated CSV were not provided by the corresponding source artifact; unavailable entries are not imputed.

## Source files

- `results/tables/pilot_label_strategy_comparison.csv`
- `results/tables/binary_training_cohort_summary.csv`
- `results/tables/pilot_metrics.csv`
- `results/survival_censor_aware/ipcw_time_dependent_metrics.csv`
- `results/survival_censor_aware/cause_specific_concordance.csv`
- `results/explainability/global_shap_feature_importance.csv`
- `results/explainability/horizon_rank_consistency.csv`
- `results/explainability/cluster_quality_diagnostics.csv`
- `results/explainability/cluster_risk_profile_summary.csv`
- `results/macro_experiment/baseline_vs_macro_metrics.csv`
- `results/macro_experiment/paired_performance_differences.csv`
- `data/macro/macro_data_manifest.json`
- `results/macro_experiment/macro_experiment_metadata.json`
