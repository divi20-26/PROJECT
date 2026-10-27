from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "censor_aware_survival_metrics", ROOT / "15_survival_censor_aware_metrics.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load censor-aware survival metrics.")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class CensorAwareSurvivalMetricTests(unittest.TestCase):
    def test_censoring_survival_uses_left_limit_at_month_boundary(self) -> None:
        survival = metrics._censoring_survival_before(
            np.array([1, 2, 2, 3]),
            np.array([False, False, True, True]),
            np.array([1, 2, 3]),
        )
        np.testing.assert_allclose(survival, [1.0, 1.0, 2 / 3])

    def test_ipcw_metrics_keep_payoff_as_nondefault_and_mask_early_censor(self) -> None:
        result = metrics._ipcw_horizon_metrics(
            np.array(
                [
                    "default",
                    "voluntary_payoff",
                    "other_termination",
                    "right_censored",
                    "default",
                ]
            ),
            np.array([1, 2, 2, 3, 4]),
            horizon=2,
            probability=np.array([0.8, 0.1, 0.2, 0.1, 0.05]),
        )
        self.assertEqual(result["n_observed_for_horizon"], 4)
        self.assertEqual(result["defaults"], 1)
        self.assertEqual(result["observed_censored_before_horizon"], 1)
        self.assertAlmostEqual(result["roc_auc_ipcw"], 1.0)
        self.assertAlmostEqual(result["pr_auc_ipcw"], 1.0)
        self.assertAlmostEqual(result["brier_score_ipcw"], 0.015625)

    def test_ipcw_weights_reflect_early_censoring(self) -> None:
        result = metrics._ipcw_horizon_metrics(
            np.array(["other_termination", "default", "right_censored"]),
            np.array([1, 2, 3]),
            horizon=2,
            probability=np.array([0.1, 0.8, 0.1]),
        )
        self.assertAlmostEqual(
            result["censoring_survival_at_horizon_left_limit"], 2 / 3
        )
        self.assertAlmostEqual(result["max_ipcw_weight"], 1.5)


if __name__ == "__main__":
    unittest.main()
