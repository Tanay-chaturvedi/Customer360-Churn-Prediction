"""Tests for MLflow experiment tracking integration in Customer360."""

from pathlib import Path
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import mlflow

from src.ml.tracking import (
    get_tracking_uri,
    init_mlflow,
    extract_estimator_params,
    generate_run_plots,
    log_training_run,
    log_champion_results,
    EXPERIMENT_NAME,
)


def test_get_tracking_uri():
    uri = get_tracking_uri()
    assert "sqlite:///" in uri
    assert "mlflow.db" in uri


def test_init_mlflow():
    uri = init_mlflow()
    assert mlflow.get_tracking_uri() == uri
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert exp is not None
    assert exp.name == EXPERIMENT_NAME


def test_extract_estimator_params_lr():
    clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42)
    params = extract_estimator_params(clf, "Logistic_Regression", "Class_Weights", 42)
    assert params["model_name"] == "Logistic_Regression"
    assert params["method"] == "Class_Weights"
    assert params["random_seed"] == 42
    assert params["C"] == 0.1
    assert params["max_iter"] == 1000
    assert params["class_weight"] == "balanced"


def test_extract_estimator_params_xgb():
    clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=2.5, random_state=42)
    params = extract_estimator_params(clf, "XGBoost", "Class_Weights", 42)
    assert params["model_name"] == "XGBoost"
    assert params["n_estimators"] == 100
    assert params["max_depth"] == 3
    assert params["learning_rate"] == 0.05
    assert params["scale_pos_weight"] == 2.5


def test_generate_run_plots(tmp_path):
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.9, 0.15, 0.85])

    plots = generate_run_plots("Test_Model", y_true, y_pred, y_prob, tmp_path)
    assert "confusion_matrix" in plots
    assert "roc_curve" in plots
    assert "pr_curve" in plots
    assert plots["confusion_matrix"].exists()
    assert plots["roc_curve"].exists()
    assert plots["pr_curve"].exists()
