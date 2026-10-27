# Methodology audit and final-experiment gate

Audit date: 2026-10-25

## Decision costs

For binary label \(y\), predicted risk \(p\), threshold \(t\), missed-default cost \(c_m\), and review cost \(c_r\), the implemented mean cost is

```text
mean(c_m * 1[y=1 and p<t] + c_r * 1[p>=t])
```

Thus review cost is charged to every flagged loan, including flagged defaults; correct negatives incur no cost. Decision boundaries use `p >= t`. Threshold search considers every distinct validation score and one value immediately above the maximum score, representing zero reviews. Ties select the largest threshold, minimizing reviews among equal-cost actions. Tests cover the cost arithmetic, exact-score boundary, no-review action, invalid inputs, and near ties.

Some cost scenarios choose no review because a validation score boundary with no flagged loans minimizes the assumed normalized cost: at those scenario ratios, reviewing even a small predicted-risk tail costs more than the validation-estimated missed defaults saved. In the expanded 14-ratio grid, 194 of 294 model/strategy/horizon/scenario thresholds flagged no validation loans; 119 of 294 corresponding test decisions flagged none. This is an implication of the assumptions and validation data, not evidence that no review is operationally optimal.

The originally analyzed review-to-missed-default ratios were 0.01–5.0. The expanded sensitivity grid in `research_protocol_v1.json` is 0.001–100.0. No lender-specific empirical utility evidence or observed monetary loss estimates justify the values; all ratios remain explicitly hypothetical and normalized.

## Temporal integrity

- The feature whitelist consists of origination-time fields and excludes loan IDs, vintage, eligibility fields, and outcomes.
- Active-feature selection and preprocessing are fit separately within each training cohort. Validation and test rows are transformed using the fitted preprocessor.
- Hyperparameters are fixed pilot settings; no test-cohort tuning was performed.
- Calibration intercept/slope and reliability curves are descriptive test-set evaluations only; no recalibration was applied from those results.
- Decision thresholds are selected on 2015–2016 validation predictions only.
- 2017–2019 results were previously inspected and are exploratory. 2020–2022 results were also previously inspected and are exploratory.

## Survival analysis

The Cox models use entry time `max(entry_month - 1, 0)`, exit at the observed event/censor month, default event code 1, payoff event code 2, and code 0 for other termination/right censoring in each cause-specific fit. Both default and payoff models are required. Predicted cumulative incidence combines piecewise-constant cause-specific hazards and is checked for finite values, probability bounds, and monotonicity. Boundary and event-code tests are in the test suite.

The separate censor-aware evaluation at `results/survival_censor_aware/` uses entry-month-1 evaluation cohorts, frozen 36-month Cox cumulative-incidence predictions, and a discrete marginal reverse-Kaplan-Meier estimate of censoring survival. Voluntary payoffs are known non-default competing outcomes; other terminations and right-censored loans are treated as censoring for the IPCW metrics. At horizon \(h\), known events by \(h\) receive inverse censoring-survival weights at their event month, and known controls receive the inverse censoring-survival weight at \(h\). This targets a marginal independent-censoring estimand; it does not adjust for covariate-dependent censoring. Censoring is rare in these observed cohorts, so the reported weights are close to one.

| Evaluation cohort (exploratory) | Horizon | Observed for horizon | Defaults | Censored before horizon | IPCW ROC-AUC | IPCW PR-AUC | IPCW Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2017–2019 | 12m | 144,083 | 1,420 | 223 | 0.6755 | 0.0208 | 0.009766 |
| 2017–2019 | 24m | 144,059 | 3,840 | 247 | 0.7161 | 0.0677 | 0.025834 |
| 2017–2019 | 36m | 144,052 | 6,230 | 254 | 0.7034 | 0.0986 | 0.040931 |
| 2020–2022 | 12m | 148,202 | 1,147 | 341 | 0.7017 | 0.0176 | 0.007667 |
| 2020–2022 | 24m | 148,148 | 2,156 | 395 | 0.7401 | 0.0397 | 0.014203 |
| 2020–2022 | 36m | 148,124 | 3,053 | 419 | 0.7538 | 0.0605 | 0.019874 |

Cause-specific Harrell concordance at 36 months was 0.6964 (n=144,306; 8,127 defaults) for 2017–2019 and 0.7481 (n=148,543; 3,847 defaults) for 2020–2022. Payoffs are censored in this cause-specific concordance calculation, so it is not a competing-risk concordance measure. These performance results are exploratory because both evaluation periods were inspected previously; none support confirmatory claims.

## Bootstrap, matching, calibration, and segments

- The paired bootstrap reports observed full-sample candidate-minus-reference differences and percentile intervals from 1,000 paired loan resamples within vintage. It preserves the evaluated vintage counts, but not within-vintage clustering or calendar-time dependence; inference is conditional on the observed vintage mix and intervals are unadjusted.
- The five-seed matched-naive experiment exactly preserves vintage/FICO/LTV/occupancy stratum counts and reports broader feature-balance diagnostics. It is a sampling sensitivity analysis, not a causal design.
- Reliability plots, calibration intercept/slope, and expected calibration error are available at 12, 24, and 36 months for both previously inspected evaluation periods.
- Segment tables report vintage, FICO band, and occupancy. Unsupported small segments are retained with discrimination metrics withheld.
- Sensitivity to alternate censoring estimators, outcome definitions, cohort exclusions, rare-level handling, and model settings remains exploratory work; do not use previously inspected test outcomes to choose among those variants.

## Holdout availability and stop decision

No eligible untouched 36-month holdout is available in the repository:

| Vintage range | Prior exposure / 36-month maturity | Decision |
|---|---|---|
| 2017–2019 | Pilot metrics and reports already inspected | Exploratory only |
| 2020–2022 | Outcome-coverage and default-rate reports were already generated; later model evaluation also inspected | Exploratory only |
| 2023 | Outcome-coverage report already generated; 19,366/50,000 (38.7%) loans are mature-eligible at 36 months as of 2026-03-31; 30,066/50,000 are right-censored before 36 months | Not adequate and not untouched |
| 2024 | Outcome-coverage report already generated; 9,692/50,000 (19.4%) are mature-eligible at 36 months; 40,084/50,000 are right-censored before 36 months | Not adequate and not untouched |

The available coverage table includes vintages only through 2024. Therefore final/confirmatory model fitting and evaluation are **blocked**. Do not fit a model or tune any decision on an existing cohort and then describe its score as final confirmatory performance. The next confirmatory run requires a future vintage range with adequate 36-month follow-up whose outcomes have not informed modeling, feature, threshold, cost, or protocol choices.

The prospective protocol is frozen in `research_protocol_v1.json`; it was written after exploratory results were examined and is not a preregistration of those results.
