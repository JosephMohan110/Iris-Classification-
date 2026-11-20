"""Reproducible Iris classification benchmark and evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
FEATURE_COLUMNS = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)
TARGET_COLUMN = "species"
RANDOM_STATE = 42
TEST_SIZE = 0.20

sns.set_theme(style="whitegrid", context="notebook")


def load_dataset(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the expected Iris CSV without discarding observations."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Iris dataset was not found: {data_path}")

    frame = pd.read_csv(data_path)
    expected_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Dataset contains no rows.")
    if frame[expected_columns].isna().any().any():
        raise ValueError("Dataset contains missing feature or target values.")

    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Feature values must all be finite numbers.")

    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("string").str.strip()
    if (frame[TARGET_COLUMN] == "").any():
        raise ValueError("Target labels must not be empty.")
    if frame[TARGET_COLUMN].nunique() < 3:
        raise ValueError("This benchmark expects a multiclass target with at least 3 classes.")
    return frame


def split_dataset(
    frame: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and labels with stratification and a fixed random seed."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset must contain the target column {TARGET_COLUMN!r}.")
    X = frame.loc[:, FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_models() -> dict[str, Pipeline]:
    """Build comparable, leakage-safe model pipelines."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Fit both pipelines and calculate holdout metrics, reports, and probabilities."""
    results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, float | str]] = []

    for model_name, estimator in build_models().items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)
        classes = estimator.classes_

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, predictions, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                y_test, predictions, average="weighted", zero_division=0
            )
        )
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "roc_auc_ovr_macro": float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                )
            ),
        }
        results[model_name] = {
            "estimator": estimator,
            "predictions": predictions,
            "probabilities": probabilities,
            "classes": classes,
            "metrics": metrics,
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=classes,
                output_dict=True,
                zero_division=0,
            ),
        }
        summary_rows.append({"model": model_name, **metrics})

    summary = pd.DataFrame(summary_rows).sort_values(
        "f1_macro", ascending=False, ignore_index=True
    )
    return results, summary


def plot_confusion_matrices(
    results: dict[str, dict[str, Any]],
    y_test: pd.Series,
    output_dir: Path,
) -> Path:
    """Save side-by-side confusion matrices using the shared class order."""
    class_names = sorted(y_test.unique().tolist())
    fig, axes = plt.subplots(1, len(results), figsize=(12, 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, (model_name, result) in zip(axes, results.items()):
        matrix = confusion_matrix(y_test, result["predictions"], labels=class_names)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axis,
        )
        axis.set_title(model_name)
        axis.set_xlabel("Predicted species")
        axis.set_ylabel("Actual species")

    fig.suptitle("Holdout confusion matrices (20% test split)", fontsize=15, weight="bold")
    return _save_figure(fig, output_dir / "confusion_matrices.png")


def plot_roc_curves(
    results: dict[str, dict[str, Any]],
    y_test: pd.Series,
    output_dir: Path,
) -> Path:
    """Save one-vs-rest ROC curves for each species and model."""
    class_names = sorted(y_test.unique().tolist())
    binary_targets = label_binarize(y_test, classes=class_names)
    class_colors = sns.color_palette("colorblind", n_colors=len(class_names))
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)

    for model_name, result in results.items():
        model_auc = result["metrics"]["roc_auc_ovr_macro"]
        for class_index, (class_name, color) in enumerate(zip(class_names, class_colors)):
            false_positive_rate, true_positive_rate, _ = roc_curve(
                binary_targets[:, class_index],
                result["probabilities"][:, class_index],
            )
            axis.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                linestyle="-" if model_name == "Logistic Regression" else "--",
                linewidth=2,
                label=f"{model_name} — {class_name} (macro AUC {model_auc:.3f})",
            )

    axis.plot([0, 1], [0, 1], color="black", linestyle=":", label="Random baseline")
    axis.set(
        title="Multiclass ROC curves (one-vs-rest)",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right", fontsize=8, frameon=True)
    return _save_figure(fig, output_dir / "roc_curves.png")


def plot_performance_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    """Save a compact comparison of the four primary classification metrics."""
    metric_columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    plot_data = summary.set_index("model")[metric_columns].rename(
        columns={
            "accuracy": "Accuracy",
            "precision_macro": "Macro precision",
            "recall_macro": "Macro recall",
            "f1_macro": "Macro F1",
        }
    )
    axis = plot_data.plot(kind="bar", figsize=(10, 6), ylim=(0, 1.05), rot=0)
    axis.set_title("Model performance on the stratified holdout set", weight="bold")
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.legend(title="Metric", loc="lower right")
    fig = axis.get_figure()
    fig.tight_layout()
    return _save_figure(fig, output_dir / "performance_summary.png")


def plot_decision_boundaries(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> Path:
    """Plot illustrative two-feature boundaries; these models are not benchmarked."""
    feature_names = list(X_train.columns[:2])
    class_names = sorted(y_train.unique().tolist())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    palette = sns.color_palette("Set2", n_colors=len(class_names))
    color_map = ListedColormap(palette)

    x_min = min(X_train[feature_names[0]].min(), X_test[feature_names[0]].min()) - 0.5
    x_max = max(X_train[feature_names[0]].max(), X_test[feature_names[0]].max()) + 0.5
    y_min = min(X_train[feature_names[1]].min(), X_test[feature_names[1]].min()) - 0.5
    y_max = max(X_train[feature_names[1]].max(), X_test[feature_names[1]].max()) + 0.5
    mesh_x, mesh_y = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02),
    )
    grid = pd.DataFrame(
        np.c_[mesh_x.ravel(), mesh_y.ravel()],
        columns=feature_names,
    )
    models = build_models()
    fig, axes = plt.subplots(1, len(models), figsize=(13, 5), constrained_layout=True)

    for axis, (model_name, estimator) in zip(np.atleast_1d(axes), models.items()):
        estimator.fit(X_train[feature_names], y_train)
        predicted_indices = np.array(
            [class_to_index[label] for label in estimator.predict(grid)]
        ).reshape(mesh_x.shape)
        axis.contourf(
            mesh_x,
            mesh_y,
            predicted_indices,
            levels=np.arange(len(class_names) + 1) - 0.5,
            cmap=color_map,
            alpha=0.28,
        )
        for class_name, color in zip(class_names, palette):
            mask = y_test.to_numpy() == class_name
            axis.scatter(
                X_test.loc[mask, feature_names[0]],
                X_test.loc[mask, feature_names[1]],
                color=[color],
                edgecolor="black",
                linewidth=0.6,
                s=48,
                label=class_name,
            )
        test_predictions = estimator.predict(X_test[feature_names])
        errors = test_predictions != y_test.to_numpy()
        if errors.any():
            axis.scatter(
                X_test.loc[errors, feature_names[0]],
                X_test.loc[errors, feature_names[1]],
                facecolors="none",
                edgecolors="black",
                linewidths=1.8,
                s=105,
                label="Misclassified",
            )
        axis.set_title(model_name)
        axis.set_xlabel(feature_names[0].replace("_", " ").title())
        axis.set_ylabel(feature_names[1].replace("_", " ").title())
        axis.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Illustrative decision boundaries using two features only",
        fontsize=14,
        weight="bold",
    )
    return _save_figure(fig, output_dir / "decision_boundaries.png")


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run_benchmark(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the full benchmark and write machine-readable metrics and charts."""
    frame = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_dataset(frame)
    results, summary = evaluate_models(X_train, X_test, y_train, y_test)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = {
        model_name: result["classification_report"]
        for model_name, result in results.items()
    }
    (output_dir / "classification_reports.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "dataset": Path(data_path).name,
                "rows": int(len(frame)),
                "features": list(FEATURE_COLUMNS),
                "target": TARGET_COLUMN,
                "classes": sorted(frame[TARGET_COLUMN].unique().tolist()),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "stratified": True,
                "duplicate_rows_retained": int(frame.duplicated().sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary.to_csv(output_dir / "performance_summary.csv", index=False)
    artifacts = {
        "confusion_matrices": plot_confusion_matrices(results, y_test, output_dir),
        "roc_curves": plot_roc_curves(results, y_test, output_dir),
        "performance_summary": plot_performance_summary(summary, output_dir),
        "decision_boundaries": plot_decision_boundaries(
            X_train, X_test, y_train, y_test, output_dir
        ),
    }
    return {
        "data": frame,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results": results,
        "summary": summary,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Logistic Regression and Decision Tree on Iris."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the Iris CSV (default: data.csv next to this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, JSON, and PNG evaluation artifacts.",
    )
    args = parser.parse_args()

    run = run_benchmark(args.data, args.output_dir)
    print("Holdout performance (sorted by macro F1):")
    print(run["summary"].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nEvaluation artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
