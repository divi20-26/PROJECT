"""Explain saved pilot XGBoost models and profile validation-selected SHAP clusters."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pilot_model_utils as utils
from research_extensions import (
    aggregate_shap_by_family,
    choose_explanation_clusters,
    horizon_importance_table,
    require_existing_inputs,
    transformed_feature_families,
)


OUTPUT_DIR = utils.config.RESULTS_DIR / "explainability"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_STRATEGY = "complete_eligible"
SPLITS = {
    "validation": utils.VALIDATION_YEARS,
    "test": utils.TEST_YEARS,
}
CHARACTERISTICS = [
    "classic_fico",
    "original_ltv",
    "original_cltv",
    "original_dti",
    "original_interest_rate",
]


def _common_sample(maximum: int, seed: int) -> dict[str, pd.DataFrame]:
    """Use identical loan IDs across horizons within each evaluation split."""
    cohorts: dict[str, dict[int, pd.DataFrame]] = {}
    for split, years in SPLITS.items():
        cohorts[split] = {
            horizon: utils.load_mature_cohort(years, horizon)
            for horizon in utils.HORIZONS
        }
    samples = {}
    for split, by_horizon in cohorts.items():
        key_sets = [
            set(zip(frame["vintage"], frame["loan_identifier"]))
            for frame in by_horizon.values()
        ]
        common_keys = set.intersection(*key_sets)
        if not common_keys:
            raise ValueError(f"No common horizon-eligible loans in {split}.")
        reference = by_horizon[utils.HORIZONS[0]]
        mask = [
            (int(vintage), identifier) in common_keys
            for vintage, identifier in zip(reference["vintage"], reference["loan_identifier"])
        ]
        common = reference.loc[mask, ["vintage", "loan_identifier"]].copy()
        n = min(maximum, len(common))
        chosen = common.sample(n=n, random_state=seed).sort_values(
            ["vintage", "loan_identifier"], kind="stable"
        )
        selected_keys = set(zip(chosen["vintage"], chosen["loan_identifier"]))
        for horizon, frame in by_horizon.items():
            selected = frame.loc[
                [
                    (int(vintage), identifier) in selected_keys
                    for vintage, identifier in zip(frame["vintage"], frame["loan_identifier"])
                ]
            ].copy()
            if len(selected) != n:
                raise ValueError(f"Common {split} sample diverged at {horizon} months.")
            samples[f"{split}:{horizon}"] = selected.reset_index(drop=True)
    return samples


def _load_shap():
    try:
        import shap
    except ImportError as error:
        raise RuntimeError(
            "SHAP is required for this extension. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from error
    return shap


def _explain(estimator, frame: pd.DataFrame, shap_module):
    preprocessor = estimator.named_steps["preprocessor"]
    model = estimator.named_steps["model"]
    feature_columns = list(preprocessor.feature_names_in_)
    transformed_names, families = transformed_feature_families(
        preprocessor, feature_columns
    )
    matrix = preprocessor.transform(frame[feature_columns])
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(transformed_names):
        raise ValueError("Transformed feature matrix does not match fitted feature names.")

    explainer = shap_module.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    values = np.asarray(explainer.shap_values(matrix), dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.shape != matrix.shape:
        raise ValueError(
            f"SHAP output shape {values.shape} does not match transformed inputs {matrix.shape}."
        )
    expected_value = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if expected_value.size != 1:
        raise ValueError("Expected one SHAP base value for binary XGBoost raw-margin output.")
    margins = model.get_booster().predict(
        __import__("xgboost").DMatrix(matrix), output_margin=True
    )
    max_additivity_error = float(
        np.max(np.abs(values.sum(axis=1) + expected_value[0] - margins))
    )
    if max_additivity_error > 1e-3:
        raise ValueError(
            f"SHAP additivity check failed: maximum absolute raw-margin error "
            f"{max_additivity_error:.6g}."
        )
    family_names = list(feature_columns)
    family_values = aggregate_shap_by_family(values, families, family_names)
    probability = estimator.predict_proba(frame[feature_columns])[:, 1]
    return (
        values,
        family_values,
        transformed_names,
        families,
        probability,
        max_additivity_error,
        matrix,
    )


def _wide_local_frame(
    frame: pd.DataFrame,
    split: str,
    horizon: int,
    probability: np.ndarray,
    values: np.ndarray,
    columns: list[str],
    value_prefix: str,
) -> pd.DataFrame:
    output = frame[
        [
            "vintage",
            "loan_identifier",
            "y_true",
            "classic_fico",
            "original_ltv",
            "original_cltv",
            "original_dti",
            "original_interest_rate",
            "occupancy_status",
            "loan_purpose",
            "property_state",
        ]
    ].copy()
    output.insert(0, "split", split)
    output.insert(1, "horizon_months", horizon)
    output["predicted_default_probability"] = probability
    contributions = pd.DataFrame(
        {f"{value_prefix}{name}": values[:, index] for index, name in enumerate(columns)}
    )
    return pd.concat(
        [output.reset_index(drop=True), contributions.reset_index(drop=True)], axis=1
    )


def _importance(frame: pd.DataFrame, values: np.ndarray, columns: list[str], horizon: int):
    return pd.DataFrame(
        {
            "horizon_months": horizon,
            "feature": columns,
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "mean_signed_shap": np.mean(values, axis=0),
            "fraction_positive_shap": np.mean(values > 0, axis=0),
            "evaluation_split": "common_mature_test_sample",
            "sample_size": len(frame),
            "shap_output_scale": "raw XGBoost margin (log-odds)",
        }
    )


def _risk_profile_summary(
    frame: pd.DataFrame, labels: np.ndarray, shap_values: np.ndarray, features: list[str],
    horizon: int, split: str
) -> list[dict]:
    rows = []
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        values = shap_values[mask]
        mean_shap = values.mean(axis=0)
        positive = np.argsort(mean_shap)[::-1][:3]
        negative = np.argsort(mean_shap)[:3]
        row = {
            "split": split,
            "horizon_months": horizon,
            "cluster": int(cluster),
            "cluster_size": int(mask.sum()),
            "mean_predicted_default_probability": float(
                frame.loc[mask, "predicted_default_probability"].mean()
            ),
            "median_predicted_default_probability": float(
                frame.loc[mask, "predicted_default_probability"].median()
            ),
            "observed_defaults": int(frame.loc[mask, "y_true"].sum()),
            "observed_default_rate": float(frame.loc[mask, "y_true"].mean()),
            "top_positive_mean_shap_features": json.dumps(
                [{"feature": features[i], "mean_shap": float(mean_shap[i])} for i in positive]
            ),
            "top_negative_mean_shap_features": json.dumps(
                [{"feature": features[i], "mean_shap": float(mean_shap[i])} for i in negative]
            ),
        }
        for characteristic in CHARACTERISTICS:
            row[f"mean_{characteristic}"] = pd.to_numeric(
                frame.loc[mask, characteristic], errors="coerce"
            ).mean()
        for characteristic in ("occupancy_status", "loan_purpose", "property_state"):
            modes = frame.loc[mask, characteristic].astype("string").mode(dropna=True)
            row[f"modal_{characteristic}"] = modes.iloc[0] if not modes.empty else None
        rows.append(row)
    return rows


def run(maximum: int = 500, seed: int | None = None) -> None:
    seed = utils.SEED if seed is None else seed
    model_paths = [
        utils.MODEL_DIR / f"xgboost_{MODEL_STRATEGY}_{horizon}m.joblib"
        for horizon in utils.HORIZONS
    ]
    require_existing_inputs(model_paths)
    shap_module = _load_shap()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    samples = _common_sample(maximum, seed)
    family_locals = []
    transformed_locals = []
    importance_frames = []
    transformed_importance_frames = []
    assignment_frames = []
    diagnostics = []
    cluster_summaries = []
    representative_rows = []
    additivity = {}

    for horizon in utils.HORIZONS:
        estimator = joblib.load(
            utils.MODEL_DIR / f"xgboost_{MODEL_STRATEGY}_{horizon}m.joblib"
        )
        split_results = {}
        for split in SPLITS:
            frame = samples[f"{split}:{horizon}"]
            (
                transformed,
                family,
                transformed_names,
                families,
                probability,
                error,
                transformed_matrix,
            ) = _explain(estimator, frame, shap_module)
            additivity[f"{split}:{horizon}"] = error
            family_columns = list(estimator.named_steps["preprocessor"].feature_names_in_)
            family_local = _wide_local_frame(
                frame, split, horizon, probability, family, family_columns, "shap_family__"
            )
            transformed_local = _wide_local_frame(
                frame, split, horizon, probability, transformed, transformed_names, "shap_transformed__"
            )
            family_locals.append(family_local)
            transformed_locals.append(transformed_local)
            split_results[split] = (
                frame,
                family,
                family_local,
                transformed,
                transformed_matrix,
            )
            if split == "test":
                importance_frames.append(_importance(frame, family, family_columns, horizon))
                transformed_importance_frames.append(
                    _importance(frame, transformed, transformed_names, horizon)
                )

        validation_frame, validation_values, validation_local = split_results["validation"][:3]
        test_frame, test_values, test_local = split_results["test"][:3]
        validation_matrix = validation_values
        cluster_diagnostics, cluster_model, stability = choose_explanation_clusters(
            validation_matrix,
            seed=seed + horizon,
        )
        cluster_diagnostics["horizon_months"] = horizon
        diagnostics.append(cluster_diagnostics)

        validation_labels = cluster_model.predict(validation_matrix)
        test_labels = cluster_model.predict(test_values)
        for split, frame, values, local, labels in (
            ("validation", validation_frame, validation_values, validation_local, validation_labels),
            ("test", test_frame, test_values, test_local, test_labels),
        ):
            local["cluster"] = labels
            assignment_frames.append(
                local[
                    [
                        "split",
                        "horizon_months",
                        "vintage",
                        "loan_identifier",
                        "cluster",
                        "predicted_default_probability",
                        "y_true",
                    ]
                ].copy()
            )
            cluster_summaries.extend(
                _risk_profile_summary(
                    local,
                    labels,
                    values,
                    list(estimator.named_steps["preprocessor"].feature_names_in_),
                    horizon,
                    split,
                )
            )
        best_k = int(cluster_diagnostics.loc[cluster_diagnostics["selected"], "n_clusters"].iloc[0])
        for cluster in range(best_k):
            candidates = np.flatnonzero(test_labels == cluster)
            if not len(candidates):
                continue
            distances = np.linalg.norm(
                test_values[candidates] - cluster_model.cluster_centers_[cluster], axis=1
            )
            selected_index = int(candidates[np.argmin(distances)])
            row = test_local.iloc[selected_index]
            representative_values = test_values[selected_index]
            feature_names = list(estimator.named_steps["preprocessor"].feature_names_in_)
            positive_indices = np.argsort(representative_values)[::-1][:5]
            negative_indices = np.argsort(representative_values)[:5]
            representative_rows.append(
                {
                    "horizon_months": horizon,
                    "cluster": cluster,
                    "vintage": int(row["vintage"]),
                    "loan_identifier": row["loan_identifier"],
                    "predicted_default_probability": float(
                        row["predicted_default_probability"]
                    ),
                    "representative_rule": "test-sample loan nearest validation-fitted SHAP cluster center",
                    "illustrative_not_population_estimate": True,
                    "top_positive_shap_contributors": json.dumps(
                        [
                            {
                                "feature": feature_names[index],
                                "shap_raw_margin": float(representative_values[index]),
                            }
                            for index in positive_indices
                        ]
                    ),
                    "top_negative_shap_contributors": json.dumps(
                        [
                            {
                                "feature": feature_names[index],
                                "shap_raw_margin": float(representative_values[index]),
                            }
                            for index in negative_indices
                        ]
                    ),
                }
            )

        figure, axis = plt.subplots(figsize=(10, 6))
        shap_module.summary_plot(
            split_results["test"][3],
            features=split_results["test"][4],
            plot_type="dot",
            feature_names=transformed_names,
            max_display=18,
            show=False,
        )
        plt.title(
            f"{horizon}-month XGBoost SHAP summary "
            f"(raw margin; n={len(test_frame):,}, 2017–2019 exploratory)"
        )
        plt.tight_layout()
        figure.savefig(FIGURE_DIR / f"shap_summary_{horizon}m.png", dpi=220, bbox_inches="tight")
        plt.close(figure)

        profile = pd.DataFrame(
            [row for row in cluster_summaries if row["horizon_months"] == horizon and row["split"] == "test"]
        )
        if not profile.empty:
            figure, axis = plt.subplots(figsize=(8, 5))
            profile.set_index("cluster")[
                ["mean_predicted_default_probability", "observed_default_rate"]
            ].plot(kind="bar", ax=axis)
            axis.set_ylabel("Rate")
            axis.set_xlabel("Validation-fitted SHAP cluster")
            axis.set_title(
                f"{horizon}-month risk by explanation cluster (n={int(profile['cluster_size'].sum()):,})"
            )
            axis.legend(["Mean predicted default risk", "Observed default rate"])
            figure.tight_layout()
            figure.savefig(
                FIGURE_DIR / f"cluster_risk_profiles_{horizon}m.png",
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(figure)

        print(
            f"Explained {horizon}m XGBoost: validation={len(validation_frame):,}, "
            f"test={len(test_frame):,}, selected k={best_k}, "
            f"validation silhouette="
            f"{cluster_diagnostics.loc[cluster_diagnostics['selected'], 'silhouette_score'].iloc[0]:.4f}, "
            f"mean seed ARI={stability:.4f}",
            flush=True,
        )

    pd.concat(family_locals, ignore_index=True).to_csv(
        OUTPUT_DIR / "local_shap_family_contributions.csv", index=False
    )
    pd.concat(transformed_locals, ignore_index=True).to_csv(
        OUTPUT_DIR / "local_shap_transformed_contributions.csv", index=False
    )
    pd.concat(assignment_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "cluster_assignments.csv", index=False
    )
    importance = pd.concat(importance_frames, ignore_index=True)
    importance.to_csv(OUTPUT_DIR / "global_shap_feature_importance.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "horizon_feature_importance.csv", index=False)
    pd.concat(transformed_importance_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "global_shap_transformed_importance.csv", index=False
    )
    rankings, consistency = horizon_importance_table(importance_frames)
    rankings.to_csv(OUTPUT_DIR / "horizon_feature_rankings.csv", index=False)
    consistency.to_csv(OUTPUT_DIR / "horizon_rank_consistency.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        OUTPUT_DIR / "cluster_quality_diagnostics.csv", index=False
    )
    pd.DataFrame(cluster_summaries).to_csv(
        OUTPUT_DIR / "cluster_risk_profile_summary.csv", index=False
    )
    pd.DataFrame(representative_rows).to_csv(
        OUTPUT_DIR / "representative_explanations.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(11, 7))
    top_features = (
        importance.groupby("feature")["mean_abs_shap"].mean().nlargest(15).index
    )
    plotting = rankings.loc[rankings["feature"].isin(top_features)]
    for horizon in utils.HORIZONS:
        axis.plot(
            plotting[f"rank_{horizon}m"],
            plotting["feature"],
            marker="o",
            label=f"{horizon} months",
        )
    axis.invert_xaxis()
    axis.set_xlabel("Importance rank (1 = highest)")
    axis.set_ylabel("Origination feature family")
    axis.set_title("XGBoost feature-importance ranks by prediction horizon")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "horizon_importance_ranks.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "status": "completed",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "saved XGBoost complete_eligible pilot pipeline",
        "model_settings": {
            key: estimator.named_steps["model"].get_params()[key]
            for key in (
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
                "reg_lambda",
                "tree_method",
            )
        },
        "target": "default by horizon using existing complete-horizon labels",
        "model_artifacts": [str(path) for path in model_paths],
        "horizons_months": utils.HORIZONS,
        "cohorts": {
            "validation": utils.VALIDATION_YEARS,
            "test": utils.TEST_YEARS,
            "test_interpretation": "previously inspected exploratory cohort, not untouched confirmatory evidence",
            "sampling": "fixed-seed random sample from loan-ID intersection of all three horizon-mature cohorts",
            "maximum_per_split": maximum,
        },
        "label_strategy": MODEL_STRATEGY,
        "feature_mapping": "SHAP encoded-input contributions summed within original feature family; exact transformed names also retained separately",
        "shap": {
            "method": "TreeExplainer with tree_path_dependent perturbation and raw XGBoost margin output",
            "scale": "raw model margin (log-odds), not probability points",
            "maximum_additivity_error": additivity,
        },
        "clustering": {
            "algorithm": "KMeans on original-feature-family SHAP vectors",
            "candidate_k": [2, 3, 4, 5, 6],
            "selection": "highest silhouette on the validation explanation sample only",
            "distance": "Euclidean distance on unscaled original-feature-family SHAP vectors",
            "seed": seed,
            "n_init": 10,
            "stability": "mean pairwise adjusted Rand index across five validation-only initial seeds",
        },
        "versions": {
            "python": platform.python_version(),
            "shap": importlib.metadata.version("shap"),
            "xgboost": importlib.metadata.version("xgboost"),
        },
        "limitations": [
            "SHAP describes model associations, not causal effects.",
            "Clusters are explanation-pattern groups, not borrower types or treatment recommendations.",
            "Cluster IDs are fit separately per horizon and are not directly comparable across horizons.",
            "Observed evaluation periods have been inspected previously.",
        ],
    }
    (OUTPUT_DIR / "explainability_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Explainability outputs saved under {OUTPUT_DIR}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sample-per-split", type=int, default=500)
    parser.add_argument("--seed", type=int, default=utils.SEED)
    args = parser.parse_args()
    if args.max_sample_per_split < 20:
        raise ValueError("--max-sample-per-split must be at least 20.")
    run(args.max_sample_per_split, args.seed)


if __name__ == "__main__":
    main()
