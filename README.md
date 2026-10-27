# Freddie Mac SFLLD Label-Distortion Research

This repository contains a reproducible analysis of how incomplete follow-up and
label construction affect mortgage default prediction. Freddie Mac Release 47
field mappings, outcomes, origination-time features, and historical label
strategies are implemented in the existing data pipeline; the extraction and
data-quality stages are preserved.

## Research pipeline

1. `01_inspect_files.py`–`03_data_quality.py` inspect, extract, and validate the
   downloaded sample vintages.
2. `04_create_outcomes.py`–`06_create_label_strategies.py` map documented
   outcomes, build the origination feature set, and create cutoff-aware labels.
3. `07_train_binary_models.py` fits Logistic Regression and XGBoost under
   naive, complete-horizon, and matched-naive strategies.
4. `08_train_survival_models.py` fits delayed-entry cause-specific Cox models
   for default and voluntary payoff and predicts default cumulative incidence.
5. `09_evaluate_models.py` evaluates models on common, fully ascertainable
   validation/test cohorts and reports discrimination and calibration.
6. `10_decision_costs.py` selects thresholds using validation data only and
   evaluates predeclared hypothetical cost ratios on test data.
7. `11_robustness.py` calculates paired, vintage-stratified bootstrap intervals
   and segment results. `13_matched_sampling_sensitivity.py` assesses matched
   sample variability across fixed seeds and covariate-balance diagnostics.
8. `12_make_paper_figures.py` creates ROC/PR and calibration figures from saved
   predictions.
9. `14_exploratory_late_vintage_evaluation.py` applies the frozen pilot models
   and validation-selected thresholds to 2020–2022, with paired bootstrap
   intervals and vintage/FICO/occupancy segments in a separate output folder.
   These vintages were previously inspected in outcome-coverage reports, so
   this is an exploratory temporal evaluation, not a confirmatory holdout.
10. `15_survival_censor_aware_metrics.py` computes exploratory IPCW cumulative/
    dynamic discrimination and Brier metrics plus cause-specific concordance
    for the inspected 2017–2019 and 2020–2022 periods.

## Controlled experiment

The frozen time split is 2005–2014 training (cutoff 2014-12-31), 2015–2016
validation, and 2017–2019 test. Horizons are 12, 24, and 36 months. The primary
test comparisons use the same horizon-mature loans and labels for every model.
Preprocessing is fitted on training data; validation is used for threshold
selection. The 2017–2019 test cohort has already been used for pilot reporting,
so its results are exploratory rather than confirmatory.

Decision costs are normalized hypothetical scenarios, not observed dollar
losses or estimated lender utilities. Ratios of review cost to missed-default
cost range from 0.001 to 100.0 in the expanded sensitivity analysis. A selected
threshold minimizes validation cost over exact score boundaries and includes a
no-review action. Results are sensitivity evidence, not financial-loss
estimates.

The existing pilot survival table evaluates horizon-specific default cumulative
incidence on the common, fully mature cohort. A separate script now also reports
marginal-IPCW cumulative/dynamic ROC-AUC, PR-AUC, and Brier scores; voluntary
payoff is a competing non-default event and other termination/right censoring
are handled with a reverse-Kaplan-Meier censoring estimate. It also reports a
cause-specific Harrell concordance, treating payoff as censored. These survival
metrics assume marginal independent censoring and are exploratory, not
confirmatory.
Performance history begins at Freddie Mac acquisition, so delayed entry does
not recover unobserved pre-acquisition loan performance.

## Reproduction

Install dependencies from `requirements.txt`, then run from the project root:

```bash
python 07_train_binary_models.py
python 08_train_survival_models.py
python 09_evaluate_models.py
python 10_decision_costs.py
python 11_robustness.py
python 13_matched_sampling_sensitivity.py
python 12_make_paper_figures.py
python 14_exploratory_late_vintage_evaluation.py
python 15_survival_censor_aware_metrics.py
python -m unittest discover -s tests -v
```

`models/pilot/pilot_settings.json` records the split, feature whitelist,
transformations, model settings, cost scenarios, bootstrap protocol, and
installed package versions. Model artifacts and predictions are under
`models/pilot/` and `results/predictions/`; tables and figures are under
`results/tables/` and `results/figures/`. Source outcomes and features remain
under `data/outcomes/` and `data/features/`.
Late-vintage predictions and summaries are isolated under
`results/exploratory_late_vintages/`; they must not be presented as an untouched
holdout because 2020–2022 outcome coverage and default-rate summaries were
already generated.

The current analyses are research baselines, not a deployment recommendation.
See `results/pilot_results.md` for observed results and remaining limitations.

## Research extensions (isolated from the frozen baseline)

The extensions below use separate output and model directories. They do not
overwrite the pilot predictions, saved pilot models, tables, or figures.
Install the updated dependencies before running the explainability step:

```bash
python -m pip install -r requirements.txt
```

Run the following after the original feature, outcome, and label preparation
steps have completed and the frozen pilot XGBoost artifacts exist:

```bash
python 16_explainability_and_profiles.py --max-sample-per-split 500
python 17_prepare_macro_data.py --download
python 18_macro_augmented_comparison.py
python 19_build_final_results_summary.py
python -m unittest discover -s tests -v
```

If macro models and predictions already exist under the isolated extension
directories, their metrics and reports can be regenerated without refitting:

```bash
python 18_macro_augmented_comparison.py --reuse-existing
python 19_build_final_results_summary.py
```

`19_build_final_results_summary.py` reads the existing baseline and extension
tables and writes source-traceable consolidated CSV, Markdown, and PPT-ready
tables under `results/FINAL/`. It does not retrain models or modify source
results.

`16_explainability_and_profiles.py` explains the saved XGBoost
`complete_eligible` models at 12, 24, and 36 months. It selects identical
loan IDs across the three horizon-mature validation cohorts and, separately,
across the three test cohorts. SHAP TreeExplainer values are calculated on
the exact one-hot-transformed model inputs in raw model-margin (log-odds)
units; encoded contributions are retained and also summed within their
original feature families. SHAP means model attribution, not causality.
Clustering uses family-level SHAP vectors, chooses the cluster count by
validation-sample silhouette score, and applies that validation-fitted
clustering to the test sample. Cluster IDs are independently fit at each
horizon and must not be interpreted as equivalent across horizons.

Outputs are stored under `results/explainability/`, including local
contributions, global transformed/original-family importance, horizon ranks
and Spearman consistency, cluster assignments and summaries, cluster-quality
diagnostics, representative illustrative loans, figures, and a metadata JSON.
The default cap is 500 random loans per split; change it only before running
and record the sample size when interpreting results.

`17_prepare_macro_data.py` downloads CSVs without an API key from FRED. The
indicators are UNRATE (BLS monthly national unemployment, percent, seasonally
adjusted), GDPC1 (BEA quarterly real GDP, billions of chained 2017 dollars,
seasonally adjusted annual rate), and USSTHPI (FHFA quarterly national
all-transactions house-price index, not seasonally adjusted). Source pages:
[UNRATE](https://fred.stlouisfed.org/series/UNRATE),
[GDPC1](https://fred.stlouisfed.org/series/GDPC1), and
[USSTHPI](https://fred.stlouisfed.org/series/USSTHPI). Raw CSVs and a source
manifest are saved in `data/macro/`. If network access is unavailable, place
the official CSV exports in `data/macro/source/` as `UNRATE.csv`,
`GDPC1.csv`, and `USSTHPI.csv`, preserving the FRED column headers, then run
`python 17_prepare_macro_data.py`.

Macro alignment is a backward as-of join with modeled availability delays:
month-end + 7 days for UNRATE, quarter-end + 45 days for GDP, and quarter-end +
90 days for FHFA HPI, with per-series maximum data-age limits. Release 47
contains first-payment month but no exact loan-origination date; this branch
uses first-payment month minus two calendar months as a documented decision-
date proxy. FRED CSV contains current revised values rather than archived
real-time vintages. Thus this is a timing-aware exploratory comparison, not a
strict historical real-time backtest. All chosen indicators are national,
not state-level measures.

`18_macro_augmented_comparison.py` refits both baseline-only and macro-
augmented XGBoost models in `models/macro_experiment/`. Both use the frozen
complete-eligible label strategy, existing temporal splits and XGBoost
settings, and the same macro-available loan cohort per horizon. It reports
matched-cohort metrics, paired vintage-stratified bootstrap intervals,
macro-augmented SHAP importance, and a separate report in
`results/macro_experiment/`. These models and outputs are additional
experiments, not replacements for `models/pilot/` or the original pilot
results. Neither 2017–2019 nor 2020–2022 is an untouched confirmatory holdout.

The extension software tests use small synthetic fixtures only. They validate
date availability, duplicate-key/row-grain checks, transformed-feature
mapping, SHAP dimensional checks, clustering reproducibility, horizon ranks,
and isolated output paths; synthetic fixtures are not research results.
