"""MLflow experiment tracking integration for Customer360 Churn Prediction.

Provides centralized MLflow tracking configuration and logging helpers:
- SQLite tracking store at mlflow/mlflow.db
- Artifact repository at mlflow/artifacts
- Consistent experiment: Customer360_Churn_Prediction
- Parameter logging directly from estimator configurations
- Metric logging for validation benchmarks and champion test evaluation
- Artifact logging for models, preprocessor, feature names, plots, and markdown reports
- Tagging for candidate and champion models
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import mlflow

logger = logging.getLogger("Customer360.Tracking")

# Silence noisy agent hints if present in MLflow 3.x
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MLFLOW_DIR = _PROJECT_ROOT / "mlflow"
_DB_PATH = _MLFLOW_DIR / "mlflow.db"
_ARTIFACTS_DIR = _MLFLOW_DIR / "artifacts"
EXPERIMENT_NAME = "Customer360_Churn_Prediction"

# Registry for run IDs within an execution session
_SESSION_RUN_IDS: Dict[str, str] = {}


def get_tracking_uri() -> str:
    """Return the absolute SQLite tracking URI for the local store."""
    return f"sqlite:///{_DB_PATH.as_posix()}"


def init_mlflow(experiment_name: str = EXPERIMENT_NAME) -> str:
    """Configure MLflow with SQLite tracking store and ensure experiment exists.

    Returns:
        The tracking URI string.
    """
    _MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"Configured MLflow tracking URI: {tracking_uri}")

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.info(f"Creating new MLflow experiment: {experiment_name}")
        client.create_experiment(
            name=experiment_name,
            artifact_location=_ARTIFACTS_DIR.as_uri()
        )
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def extract_estimator_params(clf: Any, model_name: str, method: str, random_state: int) -> Dict[str, Any]:
    """Read actual hyperparameters from the fitted or configured estimator object."""
    params: Dict[str, Any] = {
        "model_name": model_name,
        "method": method,
        "random_seed": random_state,
        "estimator_class": type(clf).__name__,
        "train_split": 0.70,
        "val_split": 0.15,
        "test_split": 0.15,
    }

    # Extract all native estimator parameters
    if hasattr(clf, "get_params"):
        native_params = clf.get_params()
        for k, v in native_params.items():
            if v is not None and isinstance(v, (int, float, str, bool)):
                params[f"estimator_{k}"] = v
            elif isinstance(v, (int, float, str, bool)):
                params[f"estimator_{k}"] = v

    # Add specific model-family parameters for easy top-level comparison
    if hasattr(clf, "C"):
        params["C"] = clf.C
    if hasattr(clf, "max_iter"):
        params["max_iter"] = clf.max_iter
    if hasattr(clf, "solver"):
        params["solver"] = clf.solver
    if hasattr(clf, "n_estimators"):
        params["n_estimators"] = clf.n_estimators
    if hasattr(clf, "max_depth"):
        params["max_depth"] = clf.max_depth
    if hasattr(clf, "min_samples_split"):
        params["min_samples_split"] = clf.min_samples_split
    if hasattr(clf, "min_samples_leaf"):
        params["min_samples_leaf"] = clf.min_samples_leaf
    if hasattr(clf, "learning_rate"):
        params["learning_rate"] = clf.learning_rate
    if hasattr(clf, "subsample"):
        params["subsample"] = clf.subsample
    if hasattr(clf, "colsample_bytree"):
        params["colsample_bytree"] = clf.colsample_bytree
    if hasattr(clf, "scale_pos_weight") and clf.scale_pos_weight is not None:
        try:
            params["scale_pos_weight"] = round(float(clf.scale_pos_weight), 4)
        except (ValueError, TypeError):
            params["scale_pos_weight"] = str(clf.scale_pos_weight)
    if hasattr(clf, "class_weight") and clf.class_weight is not None:
        params["class_weight"] = str(clf.class_weight)

    return params


def generate_run_plots(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path
) -> Dict[str, Path]:
    """Generate per-run visual diagnostic figures (Confusion Matrix, ROC, PR curve)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}

    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = np.array([
        [f"{cm[0,0]:,}\n({cm_norm[0,0]*100:.1f}%)", f"{cm[0,1]:,}\n({cm_norm[0,1]*100:.1f}%)"],
        [f"{cm[1,0]:,}\n({cm_norm[1,0]*100:.1f}%)", f"{cm[1,1]:,}\n({cm_norm[1,1]*100:.1f}%)"]
    ])
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
        linewidths=1.0, linecolor="white", annot_kws={"size": 10, "weight": "bold"}
    )
    ax.set_title(f"Confusion Matrix: {model_name} (Val)", weight="bold", fontsize=10)
    ax.set_xlabel("Predicted Label", weight="semibold")
    ax.set_ylabel("True Label", weight="semibold")
    ax.set_xticklabels(["Retained (0)", "Churned (1)"])
    ax.set_yticklabels(["Retained (0)", "Churned (1)"])
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    plots["confusion_matrix"] = cm_path

    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=100)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, color="#2563EB", linewidth=2.0, label=f"ROC (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_title(f"ROC Curve: {model_name} (Val)", weight="bold", fontsize=10)
    ax.set_xlabel("False Positive Rate", weight="semibold")
    ax.set_ylabel("True Positive Rate (Recall)", weight="semibold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    plots["roc_curve"] = roc_path

    # 3. PR Curve
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=100)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_score = average_precision_score(y_true, y_prob)
    baseline = float(np.mean(y_true))
    ax.plot(recall, precision, color="#10B981", linewidth=2.0, label=f"PR (AUC = {pr_auc_score:.4f})")
    ax.axhline(y=baseline, color="black", linestyle=":", alpha=0.6, label=f"Baseline ({baseline*100:.1f}%)")
    ax.set_title(f"PR Curve: {model_name} (Val)", weight="bold", fontsize=10)
    ax.set_xlabel("Recall", weight="semibold")
    ax.set_ylabel("Precision", weight="semibold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    pr_path = output_dir / "pr_curve.png"
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close(fig)
    plots["pr_curve"] = pr_path

    return plots


def log_training_run(
    clf: Any,
    name: str,
    method: str,
    metrics: Dict[str, float],
    y_val: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    random_state: int,
    model_save_path: Path,
    models_dir: Path,
    smote_obj: Optional[Any] = None,
    imbalance_ratio: Optional[float] = None
) -> str:
    """Log a complete candidate model training run to MLflow."""
    init_mlflow()

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        _SESSION_RUN_IDS[name] = run_id

        # 1. Tags
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("method", method)
        mlflow.set_tag("estimator_type", type(clf).__name__)
        mlflow.set_tag("model_role", "candidate")

        # 2. Extract and log parameters
        params = extract_estimator_params(clf, name, method, random_state)
        if imbalance_ratio is not None:
            params["imbalance_ratio"] = round(imbalance_ratio, 4)
        if smote_obj is not None:
            params["smote_sampling_strategy"] = str(getattr(smote_obj, "sampling_strategy", "auto"))
            params["smote_k_neighbors"] = getattr(smote_obj, "k_neighbors", 5)
            params["smote_random_state"] = getattr(smote_obj, "random_state", random_state)

        # Sanitize parameters (convert all values to string or numeric)
        clean_params = {str(k): (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()}
        mlflow.log_params(clean_params)

        # 3. Log metrics
        metrics_dict = {
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "ROC-AUC": metrics["ROC-AUC"],
            "PR-AUC": metrics["PR-AUC"],
            "Train_Time_Sec": metrics["Train_Time_Sec"],
            "val_accuracy": metrics["Accuracy"],
            "val_precision": metrics["Precision"],
            "val_recall": metrics["Recall"],
            "val_f1_score": metrics["F1-Score"],
            "val_roc_auc": metrics["ROC-AUC"],
            "val_pr_auc": metrics["PR-AUC"],
            "train_time_sec": metrics["Train_Time_Sec"],
        }
        mlflow.log_metrics(metrics_dict)

        # 4. Log model artifact (joblib)
        if model_save_path.exists():
            mlflow.log_artifact(str(model_save_path), artifact_path="model")

        # 5. Log preprocessor and feature names if available
        preprocessor_path = models_dir / "preprocessor.joblib"
        if preprocessor_path.exists():
            mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

        features_path = models_dir / "feature_names.joblib"
        if features_path.exists():
            mlflow.log_artifact(str(features_path), artifact_path="preprocessor")

        # 6. Generate and log visual plots
        run_plots_dir = _MLFLOW_DIR / "temp_plots" / name
        try:
            plots = generate_run_plots(name, y_val, y_pred, y_prob, run_plots_dir)
            for plot_path in plots.values():
                mlflow.log_artifact(str(plot_path), artifact_path="evaluation_plots")
        except Exception as e:
            logger.warning(f"Could not generate per-run plots for {name}: {e}")

        # 7. Log individual evaluation summary
        summary_text = (
            f"# Run Evaluation Summary: {name}\n\n"
            f"- **Method**: {method}\n"
            f"- **Validation PR-AUC**: {metrics['PR-AUC']:.4f}\n"
            f"- **Validation ROC-AUC**: {metrics['ROC-AUC']:.4f}\n"
            f"- **Validation Recall**: {metrics['Recall']*100:.2f}%\n"
            f"- **Validation Precision**: {metrics['Precision']*100:.2f}%\n"
            f"- **Validation F1-Score**: {metrics['F1-Score']:.4f}\n"
            f"- **Validation Accuracy**: {metrics['Accuracy']*100:.2f}%\n"
            f"- **Training Duration**: {metrics['Train_Time_Sec']:.2f}s\n"
        )
        summary_file = run_plots_dir / "run_summary.md"
        summary_file.write_text(summary_text, encoding="utf-8")
        mlflow.log_artifact(str(summary_file), artifact_path="reports")

        logger.info(f"MLflow logged candidate run: {name} (Run ID: {run_id})")
        return run_id


def log_champion_results(
    best_model_name: str,
    test_metrics: Dict[str, float],
    reports_dir: Path
) -> None:
    """Update Champion model run with champion tag, test metrics, and comprehensive reports."""
    init_mlflow()
    champion_run_id = _SESSION_RUN_IDS.get(best_model_name)

    if not champion_run_id:
        # Search for run in the experiment by name if not in current session memory
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"tags.model_name = '{best_model_name}'",
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs:
                champion_run_id = runs[0].info.run_id

    if not champion_run_id:
        logger.warning(f"Could not locate active MLflow run for champion: {best_model_name}")
        return

    logger.info(f"Updating Champion MLflow run: {best_model_name} (Run ID: {champion_run_id})")

    with mlflow.start_run(run_id=champion_run_id):
        # Update tag to champion
        mlflow.set_tag("model_role", "champion")
        mlflow.set_tag("champion_selected", "true")

        # Log test metrics
        test_metric_dict = {
            "test_Accuracy": test_metrics.get("Accuracy", 0.0),
            "test_Precision": test_metrics.get("Precision", 0.0),
            "test_Recall": test_metrics.get("Recall", 0.0),
            "test_F1-Score": test_metrics.get("F1-Score", 0.0),
            "test_ROC-AUC": test_metrics.get("ROC-AUC", 0.0),
            "test_PR-AUC": test_metrics.get("PR-AUC", 0.0),
            "test_accuracy": test_metrics.get("Accuracy", 0.0),
            "test_precision": test_metrics.get("Precision", 0.0),
            "test_recall": test_metrics.get("Recall", 0.0),
            "test_f1_score": test_metrics.get("F1-Score", 0.0),
            "test_roc_auc": test_metrics.get("ROC-AUC", 0.0),
            "test_pr_auc": test_metrics.get("PR-AUC", 0.0),
        }
        mlflow.log_metrics(test_metric_dict)

        # Log stage 2 comprehensive artifacts to champion run
        report_files = [
            reports_dir / "model_comparison.csv",
            reports_dir / "evaluation_summary.md",
            reports_dir / "confusion_matrices.png",
            reports_dir / "roc_curves.png",
            reports_dir / "pr_curves.png",
        ]
        for rf in report_files:
            if rf.exists():
                mlflow.log_artifact(str(rf), artifact_path="reports")

    # Also log shared reports (model_comparison.csv & evaluation_summary.md) to candidate runs
    for model_name, run_id in _SESSION_RUN_IDS.items():
        if model_name != best_model_name:
            try:
                with mlflow.start_run(run_id=run_id):
                    mlflow.set_tag("model_role", "candidate")
                    cmp_csv = reports_dir / "model_comparison.csv"
                    eval_md = reports_dir / "evaluation_summary.md"
                    if cmp_csv.exists():
                        mlflow.log_artifact(str(cmp_csv), artifact_path="reports")
                    if eval_md.exists():
                        mlflow.log_artifact(str(eval_md), artifact_path="reports")
            except Exception as e:
                logger.debug(f"Could not attach shared reports to candidate run {model_name}: {e}")

    logger.info(f"Champion {best_model_name} successfully tagged and logged in MLflow.")


def get_session_run_ids() -> Dict[str, str]:
    """Return dictionary of {model_name: run_id} for the current execution."""
    return dict(_SESSION_RUN_IDS)
