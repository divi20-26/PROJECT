# Research extension implementation and results

## Status

The three requested extensions are implemented in isolated workflows. The
frozen pilot scripts, model artifacts, predictions, and original results were
not overwritten. The new analyses were run on the repository's real Freddie
Mac cohorts and downloaded FRED data; synthetic fixtures were used only by
automated software tests.

These are exploratory results. The 2017–2019 and 2020–2022 vintages were
previously inspected and are not untouched confirmatory holdouts.

## Existing baseline retained

The project retains its Release 47 extraction, origination-only feature
pipeline, cutoff-aware naive/complete/matched-naive labels, 2005–2014
training, 2015–2016 validation, 2017–2019 test, 12/24/36-month outcomes,
Logistic Regression and XGBoost models, competing-risk survival model,
evaluation, decision-cost, robustness, and survival-censor-aware analyses.
The primary saved pilot results and limitations remain documented in
`pilot_results.md` and `methodology_audit.md`.

## New implementation

| File | Purpose |
|---|---|
| `research_extensions.py` | Shared date-safe macro joins, SHAP feature-family mapping, rank comparisons, and validation-selected clustering. |
| `16_explainability_and_profiles.py` | Explains saved `complete_eligible` XGBoost models; compares horizons; fits and evaluates validation-selected SHAP clusters. |
| `17_prepare_macro_data.py` | Downloads or reads official FRED CSV series, prepares availability dates, and records the data manifest. |
| `18_macro_augmented_comparison.py` | Fits isolated baseline/macro XGBoost comparisons; evaluates common cohorts, paired intervals, and augmented-model SHAP. |
| `tests/test_research_extensions.py` | Synthetic-fixture tests for time alignment, row-grain checks, SHAP mapping, clustering, rankings, and paired metrics. |
| `requirements.txt` | Adds SHAP as a required dependency. |
| `README.md` | Documents extension design, setup, execution order, outputs, and limitations. |

## SHAP risk profiling and cross-horizon results

`16_explainability_and_profiles.py` used the saved pilot XGBoost models at all
three horizons. For each split, the 500 sampled loans were drawn from the
intersection of loans with mature labels at all three horizons, so the
cross-horizon comparisons use the same loan IDs. The 2017–2019 test cohort
remains exploratory. SHAP values are TreeExplainer attributions on raw model
margin (log-odds), computed using the fitted transformed inputs. The maximum
local additivity error was below `0.000005` at every horizon/split.

The three most important original feature families were broadly stable:
property state, original interest rate, and classic FICO. Their ranks were
1/2/3 at 12 months, 2/1/3 at 24 months, and 2/1/3 at 36 months. Spearman
rank correlations were 0.930 (12 vs 24 months), 0.868 (12 vs 36 months), and
0.955 (24 vs 36 months). Original CLTV and number of borrowers moved upward
in relative rank at longer horizons, while occupancy and channel importance
declined. Mean signed SHAP values show the direction of model contributions
on the raw-margin scale; they are not causal effects. Raw and normalized
importance are both reported because model scales differ by horizon.

K-means was run independently at each horizon on original-feature-family
SHAP vectors. The selected `k=2` had the best validation silhouette at each
horizon (0.443, 0.479, and 0.429 at 12, 24, and 36 months). Mean pairwise
adjusted Rand index across five validation-only seeds was 1.0 at every
horizon. The selected profiles separate lower- and higher-predicted-risk
groups in the sampled test loans; for example, 36-month mean predicted risk
was 0.0083 versus 0.0280. Sample default counts are small and cluster IDs are
not comparable across horizons. These are explanation-pattern clusters, not
borrower types, lending categories, or treatment recommendations.

Outputs are under `results/explainability/`: local transformed and
feature-family SHAP contributions, global transformed and family importance,
`horizon_feature_importance.csv`, horizon rankings and Spearman consistency,
`cluster_assignments.csv`, quality diagnostics, cluster summaries,
representative illustrative loans, figures, and execution metadata.

## Macroeconomic augmentation

The macro manifest records three official FRED series: BLS `UNRATE`, BEA
`GDPC1`, and FHFA `USSTHPI`. Source CSVs were downloaded successfully and
stored under `data/macro/source/`; prepared observations, checksums, source
URLs, units, frequency, geography, and timing policy are recorded in
`data/macro/macro_data_manifest.json`.

The model comparison used complete-eligible labels and the existing temporal
splits. Both baseline-only and macro-augmented XGBoost models were trained
with the same settings, seed, and macro-available rows. No rows were excluded
for missing macro indicators, no future availability-date joins were found,
and row-level joins preserve the loan key. The evaluation used 144,184,
144,160, and 144,153 common test loans at 12, 24, and 36 months respectively.
Paired 95% intervals use 1,000 vintage-stratified loan bootstrap replicates.

| Horizon | Metric | Baseline | Macro augmented | Macro minus baseline (95% interval) |
|---:|---|---:|---:|---:|
| 12m | ROC-AUC | 0.6623 | 0.6569 | -0.0054 [-0.0120, 0.0004] |
| 12m | PR-AUC | 0.0185 | 0.0190 | +0.0005 [-0.0009, 0.0021] |
| 12m | Brier | 0.00979 | 0.01071 | +0.00093 [+0.00086, +0.00100] |
| 24m | ROC-AUC | 0.7318 | 0.7163 | -0.0155 [-0.0182, -0.0130] |
| 24m | PR-AUC | 0.0726 | 0.0635 | -0.0091 [-0.0121, -0.0067] |
| 24m | Brier | 0.02590 | 0.02664 | +0.00073 [+0.00059, +0.00088] |
| 36m | ROC-AUC | 0.7227 | 0.7207 | -0.0020 [-0.0036, -0.0006] |
| 36m | PR-AUC | 0.1054 | 0.0996 | -0.0058 [-0.0083, -0.0036] |
| 36m | Brier | 0.04114 | 0.04107 | -0.00007 [-0.00025, +0.00011] |

The macro indicators did **not** produce a consistent performance improvement:
ROC-AUC and PR-AUC were lower for the augmented model at 24 and 36 months;
12-month discrimination differences were small and their intervals included
zero. Brier score worsened at 12 and 24 months and was effectively unchanged
at 36 months. Calibration metrics are also available in the metrics CSV and
vary by horizon. The 500-loan augmented-model SHAP sample gives the greatest
mean absolute macro attribution to real GDP at 12 and 24 months and to the
national house-price index at 36 months; this measures model use, not economic
causality.

Important macro timing limitations remain: Release 47 has no exact origination
date, so first-payment month minus two calendar months is only a proxy. Fixed
release lags approximate availability and are not archived release timestamps.
FRED's current values are revised historical values rather than true
real-time vintages. Indicators are national, not state-level.

Outputs and six isolated model artifacts are under `results/macro_experiment/`
and `models/macro_experiment/`. The report, paired predictions, metrics,
matched-cohort report, date-alignment audit, paired differences, SHAP table,
plot, and metadata are all generated.

## Reproduction commands

From the project root, after the original feature/outcome/label pipeline and
pilot XGBoost models are available:

```bash
python -m pip install -r requirements.txt
python 16_explainability_and_profiles.py --max-sample-per-split 500
python 17_prepare_macro_data.py --download
python 18_macro_augmented_comparison.py
python -m unittest tests.test_censor_aware_survival_metrics tests.test_late_vintage_evaluation tests.test_outcomes_and_labels tests.test_pilot_models tests.test_research_extensions -v
```

For a rerun after macro models and predictions already exist:

```bash
python 18_macro_augmented_comparison.py --reuse-existing
```

Use `python 17_prepare_macro_data.py` without `--download` to reuse the
downloaded source CSVs. If network access is unavailable, place official
series CSVs in `data/macro/source/` with the filenames and headers described
in `README.md`.

## Validation and remaining blockers

- **Passed:** 55 baseline and extension unit tests in the focused command
  above. Tests include synthetic macro timing and merge fixtures, transformed
  feature/SHAP shape checks, deterministic clustering, rank calculations, and
  paired weighted metrics checked against scikit-learn.
- **Passed:** Python syntax compilation for all four new modules and the new
  test module.
- **Real-data workflows completed:** 500-loan-per-split SHAP profiling,
  cross-horizon analysis, macro series preparation, all six macro/baseline
  model fits, matched-cohort scoring, 1,000-replicate paired intervals, and
  macro-augmented SHAP analysis.
- **Not rerun:** The full default discovery suite's archive-wide Release 47
  ZIP-schema test scans the large source archives and had previously remained
  in progress for several minutes. The focused 55-test command does not
  include that archive scan. Run `python -m unittest discover -s tests -v`
  separately when a longer validation window is available.
- **Still blocked:** Confirmatory model claims require a future, untouched
  vintage range with adequate 36-month follow-up. The current evaluation
  periods are exploratory. Exact origination timestamps and real-time macro
  data vintages would be needed for a strict historical as-of macro backtest.
