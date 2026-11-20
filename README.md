# Iris Classification Benchmark

A reproducible, end-to-end multiclass classification project using the local Iris dataset. It compares Logistic Regression and Decision Tree models with an 80/20 stratified holdout split, leakage-safe feature scaling, classification metrics, confusion matrices, and one-vs-rest ROC curves.

## Project contents

```text
Iris/
├── data.csv
├── iris_classification.py
├── notebooks/
│   └── iris_classification_benchmark.ipynb
├── outputs/
│   ├── benchmark_metadata.json
│   ├── classification_reports.json
│   ├── confusion_matrices.png
│   ├── decision_boundaries.png
│   ├── eda_pairplot.png
│   ├── feature_correlation.png
│   ├── performance_summary.csv
│   ├── performance_summary.png
│   └── roc_curves.png
├── requirements.txt
└── tests/
    └── test_iris_classification.py
```

## Setup on Windows

From PowerShell, create a dedicated environment in the project folder:

```powershell
cd "C:\path\to\Iris"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If reusing the environment in the parent folder of this project, activate it with:

```powershell
..\venv\Scripts\Activate.ps1
```

## Run the notebook

From the `Iris` directory:

```powershell
jupyter lab
```

Open `notebooks/iris_classification_benchmark.ipynb` and run all cells. The notebook documents the dataset, exploratory analysis, train/test design, benchmark metrics, per-class results, confusion matrices, ROC curves, performance comparison, and an illustrative decision-boundary plot.

## Run as a Python script

From the `Iris` directory:

```powershell
python iris_classification.py
```

Optional dataset/output locations:

```powershell
python iris_classification.py --data data.csv --output-dir outputs
```

The script writes CSV/JSON metrics and PNG charts to `outputs/`. It does not make network or API calls.

## Methodology

- The local CSV contains 150 rows, four numeric measurements, and three evenly represented species.
- The dataset has no missing feature or target values. One exact duplicate row is present and retained; duplicate removal is not part of this benchmark.
- A fixed random seed (`42`) and stratified 80/20 split produce 120 training and 30 test examples, with 40/10 examples per species.
- `StandardScaler` is fitted inside each scikit-learn `Pipeline`, so its statistics are learned from training data only. Scaling is important for Logistic Regression; it does not materially change Decision Tree splits, but keeping it in the pipeline makes preprocessing explicit and the comparison consistent.
- Both models are evaluated on the same untouched holdout set. The summary includes accuracy, macro and weighted precision/recall/F1, and macro one-vs-rest ROC AUC. Per-species precision, recall, and F1 are also saved.
- The decision-boundary figure fits separate illustrative models using only sepal length and sepal width. These two-feature models are for visualization only; their scores are not included in the four-feature benchmark.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover dataset assumptions, split sizes and class balance, multiclass metrics, and end-to-end artifact generation.

## Holdout results

Results from the checked-in run with `random_state=42`:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Macro OvR ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 | 0.997 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 | 0.950 |

Both models made two errors on the 30-row holdout. These numbers describe one small split only; they are not a reliable estimate of production performance or proof that one model is universally better.

## Interpreting the results

This is a small educational benchmark, not evidence of production readiness. The holdout set has only 30 samples, so a single prediction changes accuracy by about 3.3 percentage points. Treat differences between models cautiously; use cross-validation and representative external data before relying on a model in practice.
