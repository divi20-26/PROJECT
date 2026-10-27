"""Generate publication figures from saved common-cohort pilot predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

import pilot_model_utils as utils


MODEL_ORDER = [
    ("logistic_regression", "naive_full"),
    ("logistic_regression", "complete_eligible"),
    ("logistic_regression", "matched_naive"),
    ("xgboost", "naive_full"),
    ("xgboost", "complete_eligible"),
    ("xgboost", "matched_naive"),
    ("cause_specific_cox", "delayed_entry_competing_risk"),
]


def load_test_predictions() -> pd.DataFrame:
    paths = [
        utils.PREDICTION_DIR / "binary_pilot_predictions.parquet",
        utils.PREDICTION_DIR / "survival_pilot_predictions.parquet",
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pilot prediction files: {missing}")
    predictions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    test = predictions.loc[predictions["split"].eq("test")].copy()
    if test.empty:
        raise ValueError("No saved test predictions are available for figures.")
    if not np.isfinite(test["probability"]).all():
        raise ValueError("Test predictions contain nonfinite probabilities.")
    if not test["probability"].between(0, 1).all():
        raise ValueError("Test predictions must be in [0, 1].")
    return test


def _label(model: str, strategy: str) -> str:
    return f"{model.replace('_', ' ')} / {strategy.replace('_', ' ')}"


def _save_figure(figure: plt.Figure, stem: str) -> None:
    figure_dir: Path = utils.config.FIGURES_DIR
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def make_roc_pr_figure(test: pd.DataFrame) -> None:
    horizons = utils.HORIZONS
    figure, axes = plt.subplots(len(horizons), 2, figsize=(15, 5 * len(horizons)))
    for row, horizon in enumerate(horizons):
        cohort = test.loc[test["horizon_months"].eq(horizon)]
        for model, strategy in MODEL_ORDER:
            group = cohort.loc[
                cohort["model"].eq(model) & cohort["strategy"].eq(strategy)
            ]
            if group.empty:
                raise ValueError(f"Missing predictions for {model}/{strategy}/{horizon}m.")
            y_true = group["y_true"].to_numpy(dtype="int8")
            probability = group["probability"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y_true, probability)
            precision, recall, _ = precision_recall_curve(y_true, probability)
            label = _label(model, strategy)
            axes[row, 0].plot(fpr, tpr, label=label, linewidth=1.25)
            axes[row, 1].plot(recall, precision, label=label, linewidth=1.25)
        axes[row, 0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
        axes[row, 0].set(title=f"{horizon}-month ROC", xlabel="False positive rate", ylabel="True positive rate")
        axes[row, 1].set(
            title=f"{horizon}-month precision-recall",
            xlabel="Recall",
            ylabel="Precision",
            ylim=(0, 1),
        )
        axes[row, 1].axhline(
            float(cohort.loc[cohort["model"].eq("logistic_regression")
                             & cohort["strategy"].eq("naive_full"), "y_true"].mean()),
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="Test prevalence",
        )
    axes[0, 0].legend(fontsize=7, loc="lower right")
    axes[0, 1].legend(fontsize=7, loc="upper right")
    figure.suptitle("Common mature test cohorts (2017–2019)", y=1.01)
    figure.tight_layout()
    _save_figure(figure, "figure_04_roc_pr_curves")


def make_calibration_figure(test: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, len(utils.HORIZONS), figsize=(6 * len(utils.HORIZONS), 5))
    for axis, horizon in zip(np.atleast_1d(axes), utils.HORIZONS):
        cohort = test.loc[test["horizon_months"].eq(horizon)]
        calibration_lines = []
        axis_limit = 0.0
        for model, strategy in MODEL_ORDER:
            group = cohort.loc[
                cohort["model"].eq(model) & cohort["strategy"].eq(strategy)
            ]
            if group.empty:
                raise ValueError(f"Missing predictions for {model}/{strategy}/{horizon}m.")
            observed, predicted = calibration_curve(
                group["y_true"].to_numpy(dtype="int8"),
                group["probability"].to_numpy(dtype=float),
                n_bins=10,
                strategy="quantile",
            )
            calibration_lines.append((model, strategy, predicted, observed))
            axis_limit = max(axis_limit, float(predicted.max()), float(observed.max()))
        axis_limit = min(1.0, max(axis_limit * 1.15, 0.01))
        for model, strategy, predicted, observed in calibration_lines:
            axis.plot(
                predicted,
                observed,
                marker="o",
                markersize=3,
                linewidth=1.1,
                label=_label(model, strategy),
            )
        axis.plot([0, axis_limit], [0, axis_limit], color="black", linestyle="--", linewidth=0.8)
        axis.set(
            title=f"{horizon}-month calibration",
            xlabel="Mean predicted default incidence",
            ylabel="Observed default frequency",
            xlim=(0, axis_limit),
            ylim=(0, axis_limit),
        )
    axes[0].legend(fontsize=7, loc="upper left")
    figure.suptitle("Reliability curves on common mature test cohorts", y=1.02)
    figure.tight_layout()
    _save_figure(figure, "figure_05_calibration_curves")


def make_publication_figures() -> list[Path]:
    test = load_test_predictions()
    make_roc_pr_figure(test)
    make_calibration_figure(test)
    output_dir = utils.config.FIGURES_DIR
    return [
        output_dir / "figure_04_roc_pr_curves.png",
        output_dir / "figure_04_roc_pr_curves.pdf",
        output_dir / "figure_05_calibration_curves.png",
        output_dir / "figure_05_calibration_curves.pdf",
    ]


if __name__ == "__main__":
    for output in make_publication_figures():
        print(output)
