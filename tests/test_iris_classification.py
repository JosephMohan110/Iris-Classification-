from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from iris_classification import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    evaluate_models,
    load_dataset,
    run_benchmark,
    split_dataset,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class IrisClassificationTests(unittest.TestCase):
    def test_dataset_schema_and_class_balance(self) -> None:
        frame = load_dataset(PROJECT_DIR / "data.csv")

        self.assertEqual(len(frame), 150)
        self.assertEqual(list(FEATURE_COLUMNS), list(frame.columns[:4]))
        self.assertEqual(frame[TARGET_COLUMN].value_counts().to_dict(), {
            "setosa": 50,
            "versicolor": 50,
            "virginica": 50,
        })
        self.assertEqual(int(frame.isna().sum().sum()), 0)

    def test_split_is_stratified_and_preserves_feature_columns(self) -> None:
        frame = load_dataset(PROJECT_DIR / "data.csv")
        X_train, X_test, y_train, y_test = split_dataset(frame)

        self.assertEqual(len(X_train), 120)
        self.assertEqual(len(X_test), 30)
        self.assertEqual(list(X_train.columns), list(FEATURE_COLUMNS))
        self.assertEqual(y_train.value_counts().to_dict(), {
            "setosa": 40,
            "versicolor": 40,
            "virginica": 40,
        })
        self.assertEqual(y_test.value_counts().to_dict(), {
            "setosa": 10,
            "versicolor": 10,
            "virginica": 10,
        })

    def test_models_produce_finite_multiclass_metrics(self) -> None:
        frame = load_dataset(PROJECT_DIR / "data.csv")
        X_train, X_test, y_train, y_test = split_dataset(frame)
        results, summary = evaluate_models(X_train, X_test, y_train, y_test)

        self.assertEqual(set(results), {"Logistic Regression", "Decision Tree"})
        self.assertEqual(len(summary), 2)
        for row in summary.to_dict(orient="records"):
            self.assertTrue(0.0 <= row["accuracy"] <= 1.0)
            self.assertTrue(0.0 <= row["f1_macro"] <= 1.0)
            self.assertTrue(0.0 <= row["roc_auc_ovr_macro"] <= 1.0)

    def test_full_run_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            run = run_benchmark(PROJECT_DIR / "data.csv", output_dir)

            expected = {
                "classification_reports.json",
                "benchmark_metadata.json",
                "performance_summary.csv",
                "confusion_matrices.png",
                "roc_curves.png",
                "performance_summary.png",
                "decision_boundaries.png",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                expected,
            )
            self.assertEqual(len(run["summary"]), 2)
            self.assertTrue(all(path.is_file() for path in run["artifacts"].values()))


if __name__ == "__main__":
    unittest.main()
