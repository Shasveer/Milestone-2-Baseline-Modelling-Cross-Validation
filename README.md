# Khetha Milestone 2: Baseline Modelling and Cross-Validation

This repository establishes interpretable baseline performance for Khetha claims
data before advanced model tuning. The implementation is in
`khetha_baseline_models.py` and is designed to run as a standalone script.

## Repository contents

- `khetha_baseline_models.py`: baseline engine and executable workflow.
- `khetha_data_pipeline.py`: Milestone 1 pipeline module. Keep this file in the
	repository root beside the datasets so the baseline script can import it.
- `claims_data.csv`: claim-level modelling data, including `fraud_label` and
	`claim_amount` targets.
- `fraud_labels.csv`, `vehicles.csv`, `policyholders.csv`, and
	`geographic_risk.csv`: supporting Milestone 1 data sources.
- `config.json`: Khetha thresholds and South African domain configuration.

The baseline module imports `KhethaDataPipeline` from Milestone 1 and calls its
`run(target_col, key_features)` method for both modelling tasks. If the pipeline
module is not present in a local checkout, the script uses a claims-only
preparation fallback so that local execution remains possible. For submission,
include the Milestone 1 pipeline file in the repository root.

## Models and validation

`KhethaBaselineEngine` supports:

- `DecisionTreeClassifier` for fraud classification.
- `DecisionTreeRegressor` for claim-cost regression.
- `LogisticRegression` with `solver="lbfgs"`, `max_iter=1000`, and balanced
	class weights. Multinomial routing is enabled with `multi_class=True`.
- `StratifiedKFold` with accuracy scoring for classification.
- `KFold` with $R^2$ scoring for regression.
- Fold-local preprocessing pipelines that fit categorical encoders inside each
	cross-validation training fold to prevent preprocessing leakage.
- Cost-complexity pruning through `cost_complexity_pruning_path()`.
- Fit diagnostics that log overfitting when the train score exceeds the
	validation score by more than 15%, and underfitting when both scores are
	below 0.7.
- Decision-tree visualisation of the top three levels using `plot_tree`,
	feature names, and filled nodes.

Cross-validation methods return a consistent dictionary:

```python
{
		"mean": 0.0,
		"std": 0.0,
		"scores": numpy_array,
}
```

## Installation

Install the runtime dependencies in the active Python environment:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

The Milestone 1 pipeline has the same core scientific Python dependencies.

## Running the baseline workflow

From the repository root, run:

```bash
python khetha_baseline_models.py
```

The script will:

1. Load and split fraud data through the Milestone 1 pipeline.
2. Train and cross-validate a constrained fraud decision tree.
3. Generate `khetha_tree_splits.png` when matplotlib is available.
4. Load and split claim-cost data through the Milestone 1 pipeline.
5. Compare linear regression and a decision-tree regressor.
6. Tune a pruned tree and emit fit diagnostics through the logger.

The generated PNG is a review artifact and is not required for importing the
engine in another Python module.

## Programmatic use

```python
from khetha_baseline_models import KhethaBaselineEngine

engine = KhethaBaselineEngine(X_train, y_train, X_test, y_test, "classification")
tree = engine.build_tree_baseline(max_depth=5, ccp_alpha=0.0)
scores = engine.run_cross_validation(tree, cv_folds=5)
diagnostics = engine.diagnose_fit(tree)
```

For regression, pass `task_type="regression"`. `control_tree_complexity()`
returns `(best_ccp_alpha, best_model)` after evaluating pruning candidates.

## Verification

The expected smoke checks are:

```bash
python khetha_baseline_models.py
python -m py_compile khetha_baseline_models.py
```

The script should print cross-validation summaries for fraud and claim cost,
and warning logs when the score comparison indicates overfitting or underfitting.