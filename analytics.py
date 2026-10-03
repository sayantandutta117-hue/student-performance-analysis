"""
Machine learning regression module for student marks prediction.

Uses scikit-learn to compare multiple regression models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ML configuration
FEATURE_COLUMNS = [
    "attendance",
    "study_hours",
    "assignment_score",
    "midterm_marks",
    "previous_marks",
]
TARGET_COLUMN = "marks"
MIN_TOTAL_SAMPLES = 5
RANDOM_STATE = 42
TEST_SIZE = 0.3
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Data preparation (shared by all pipelines)
# ---------------------------------------------------------------------------

def prepare_regression_data(student):
    """
    Validate and prepare data for regression.

    Missing feature values are reported but excluded from training.
    If too many records are incomplete, returns None.

    Returns:
        (X, y) or (None, None) if data is invalid.
    """
    if student.empty:
        print("Cannot train model: dataset is empty.")
        return None, None

    missing_features = [col for col in FEATURE_COLUMNS if col not in student.columns]
    if missing_features:
        print(f"Cannot train model: missing feature columns: {', '.join(missing_features)}")
        return None, None

    if TARGET_COLUMN not in student.columns:
        print(f"Cannot train model: missing target column '{TARGET_COLUMN}'.")
        return None, None

    X = student[FEATURE_COLUMNS].copy()
    y = student[TARGET_COLUMN].copy()

    # Ensure numeric types
    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # Detect missing values before dropping
    missing_mask = X.isna().any(axis=1) | y.isna()
    missing_count = int(missing_mask.sum())
    if missing_count > 0:
        print(
            f"Warning: {missing_count} record(s) have missing or non-numeric ML features "
            f"and will be excluded from training."
        )

    # Drop incomplete rows
    X = X[~missing_mask].reset_index(drop=True)
    y = y[~missing_mask].reset_index(drop=True)

    if len(X) < MIN_TOTAL_SAMPLES:
        print(
            f"Cannot train model: need at least {MIN_TOTAL_SAMPLES} complete student records "
            f"for a meaningful train/test split. Current complete records: {len(X)}."
        )
        return None, None

    return X, y


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_regression_models():
    """
    Return a dict of model name -> sklearn regressor.

    All applicable models use random_state=42 for reproducibility.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_single_model(model, X_train, y_train, X_test, y_test):
    """Fit a model and return (model, y_pred, metrics_dict)."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_regression_model(y_test, y_pred)
    return model, y_pred, metrics


def train_regression_model(student):
    """
    Train a Linear Regression model to predict marks from academic features.

    Returns:
        dict with keys:
            model, X_train, X_test, y_train, y_test,
            y_pred, metrics, features, target,
            used_rows, total_rows
        or None if training fails.
    """
    X, y = prepare_regression_data(student)
    if X is None:
        return None

    total_rows = len(student)
    used_rows = len(X)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    except ValueError as e:
        print(f"Cannot perform train/test split: {e}")
        return None

    model = LinearRegression()
    _, y_pred, metrics = _train_single_model(model, X_train, y_train, X_test, y_test)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "used_rows": used_rows,
        "total_rows": total_rows,
    }


def predict_marks(model, feature_values):
    """
    Predict marks for given feature values.

    Args:
        model: trained LinearRegression model
        feature_values: list or array of 5 numeric values in the order:
            [attendance, study_hours, assignment_score, midterm_marks, previous_marks]

    Returns:
        predicted marks (float) or None on error.
    """
    if not hasattr(feature_values, "__iter__") or len(feature_values) != len(FEATURE_COLUMNS):
        print(
            f"Invalid input: expected {len(FEATURE_COLUMNS)} feature values "
            f"({', '.join(FEATURE_COLUMNS)})."
        )
        return None

    try:
        values = [float(v) for v in feature_values]
    except (TypeError, ValueError):
        print("Invalid feature values. Please enter numeric values.")
        return None

    prediction = model.predict([values])
    return float(prediction[0])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_regression_model(y_true, y_pred):
    """
    Calculate regression evaluation metrics.

    Returns:
        dict with MAE, MSE, RMSE, R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_regression_models(student):
    """
    Train and evaluate multiple regression models on the same data.

    Returns:
        pandas.DataFrame with columns:
            Model, MAE, MSE, RMSE, R²
        or None if training fails.
    """
    X, y = prepare_regression_data(student)
    if X is None:
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    except ValueError as e:
        print(f"Cannot perform train/test split: {e}")
        return None

    models = get_regression_models()
    rows = []

    for name, model in models.items():
        _, y_pred, metrics = _train_single_model(model, X_train, y_train, X_test, y_test)
        rows.append({
            "Model": name,
            "MAE": metrics["mae"],
            "MSE": metrics["mse"],
            "RMSE": metrics["rmse"],
            "R²": metrics["r2"],
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df[["Model", "MAE", "MSE", "RMSE", "R²"]]
    return result_df


def _safe_cv_folds(n_samples):
    """Return a safe number of CV folds, never exceeding n_samples."""
    if n_samples < 2:
        return 0
    return min(CV_FOLDS, n_samples)


def cross_validate_regression_models(student):
    """
    Run k-fold cross-validation for all regression models.

    Uses KFold with shuffle=True and random_state=42.

    Returns:
        pandas.DataFrame with columns:
            Model, RMSE Mean, RMSE Std, R² Mean, R² Std
        or None if cross-validation fails.
    """
    X, y = prepare_regression_data(student)
    if X is None:
        return None

    n_samples = len(X)
    k = _safe_cv_folds(n_samples)

    if k < 2:
        print(
            f"Cannot run cross-validation: need at least 2 complete records. "
            f"Current complete records: {n_samples}."
        )
        return None

    if k != CV_FOLDS:
        print(
            f"Note: dataset is small ({n_samples} records). "
            f"Using {k}-fold cross-validation instead of {CV_FOLDS}-fold."
        )

    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    models = get_regression_models()
    rows = []

    for name, model in models.items():
        fold_rmse_scores = []
        fold_r2_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model_clone = _clone_model(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_val)

            fold_rmse_scores.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
            fold_r2_scores.append(float(r2_score(y_val, y_pred)))

        rows.append({
            "Model": name,
            "RMSE Mean": np.mean(fold_rmse_scores),
            "RMSE Std": np.std(fold_rmse_scores, ddof=1),
            "R² Mean": np.mean(fold_r2_scores),
            "R² Std": np.std(fold_r2_scores, ddof=1),
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df[["Model", "RMSE Mean", "RMSE Std", "R² Mean", "R² Std"]]
    return result_df


def _clone_model(model):
    """Create a fresh unfitted copy of a sklearn model."""
    return type(model)(**model.get_params())
