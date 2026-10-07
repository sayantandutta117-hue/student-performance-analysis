import pandas as pd
import numpy as np
import analytics
from sklearn.model_selection import train_test_split


def train_specific_regression_model(student, model_name):
    X, y = analytics.prepare_regression_data(student)
    if X is None:
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=analytics.TEST_SIZE, random_state=analytics.RANDOM_STATE
        )
    except ValueError:
        return None

    models = analytics.get_regression_models()
    if model_name not in models:
        return None

    model = models[model_name]
    model, y_pred, metrics = analytics._train_single_model(model, X_train, y_train, X_test, y_test)

    return {
        "model": model,
        "features": list(analytics.FEATURE_COLUMNS),
        "target": analytics.REGRESSION_TARGET_COLUMN,
        "used_rows": len(X),
        "total_rows": len(student),
        "metrics": metrics,
    }


def train_specific_classification_model(student, model_name):
    X, y, label_map = analytics.prepare_classification_data(student)
    if X is None:
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=analytics.TEST_SIZE, random_state=analytics.RANDOM_STATE, stratify=y
        )
    except ValueError:
        return None

    models = analytics.get_classification_models()
    if model_name not in models:
        return None

    model = models[model_name]
    model, y_pred, metrics = analytics._train_single_classifier(model, X_train, y_train, X_test, y_test)

    return {
        "model": model,
        "features": list(analytics.FEATURE_COLUMNS),
        "target": analytics.CLASSIFICATION_TARGET_COLUMN,
        "used_rows": len(X),
        "total_rows": len(student),
        "metrics": metrics,
        "label_map": label_map,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def get_model_explanation(model, model_name, model_type="regression"):
    features = list(analytics.FEATURE_COLUMNS)

    if model_name in ["Linear Regression", "Logistic Regression"]:
        if not hasattr(model, "coef_"):
            return None

        coef = model.coef_
        if hasattr(coef, "flatten"):
            coef_values = [float(v) for v in coef.flatten()]
        else:
            coef_values = [float(coef)]

        if len(coef_values) != len(features):
            return None

        return {
            "type": "coefficients",
            "features": features,
            "values": coef_values,
            "model_type": model_type,
        }
    elif model_name in ["Decision Tree", "Random Forest"]:
        if not hasattr(model, "feature_importances_"):
            return None

        importances = [float(v) for v in model.feature_importances_]

        if len(importances) != len(features):
            return None

        return {
            "type": "feature_importance",
            "features": features,
            "values": importances,
            "model_type": model_type,
        }

    return None


def validate_prediction_inputs(feature_values):
    if feature_values is None:
        return False, "Feature values are required."
    if not hasattr(feature_values, "__iter__"):
        return False, "Invalid feature values."
    if len(feature_values) != len(analytics.FEATURE_COLUMNS):
        return False, f"Expected {len(analytics.FEATURE_COLUMNS)} feature values."

    try:
        values = [float(v) for v in feature_values]
    except (TypeError, ValueError):
        return False, "All feature values must be numeric."

    return True, values
