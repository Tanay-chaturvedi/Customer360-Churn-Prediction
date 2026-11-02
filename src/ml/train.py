"""Model training and class imbalance handling for Customer360.

Trains:
1. Logistic Regression (Class-weighted)
2. Random Forest (Class-weighted)
3. XGBoost (scale_pos_weight adjusted)
4. SMOTE Experiment Variants (Trained on synthetically balanced training data)

Validates models on the holdout validation set and serializes artifacts to models/.
"""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import joblib
from src.ml.tracking import log_training_run

logger = logging.getLogger("Customer360.Train")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute all standard evaluation metrics for binary classification."""
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1-Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "PR-AUC": round(float(average_precision_score(y_true, y_prob)), 4)
    }


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    evaluate_smote: bool = True,
    models_dir: Path = None,
    random_state: int = 42,
    track_mlflow: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """Train class-weighted classifiers and optional SMOTE experiment models.

    Parameters:
        X_train: Processed training feature matrix
        y_train: Training labels (0/1)
        X_val: Processed validation feature matrix
        y_val: Validation labels (0/1)
        evaluate_smote: Whether to run SMOTE experimental branch
        models_dir: Directory to save serialized model files
        random_state: Fixed random seed for reproducibility

    Returns:
        (trained_models_dict, val_comparison_df, best_model_name)
    """
    if models_dir is None:
        models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Calculate class imbalance ratio for XGBoost scale_pos_weight
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    imbalance_ratio = neg_count / pos_count
    logger.info(f"Training set class distribution: 0={neg_count:,}, 1={pos_count:,} (Ratio: {imbalance_ratio:.2f}:1)")

    # Define core class-weighted models
    model_definitions = {
        "Logistic_Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            solver="lbfgs",
            random_state=random_state
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=imbalance_ratio,
            eval_metric="logloss",
            random_state=random_state
        )
    }

    trained_models = {}
    val_results = []

    # 1. Train and evaluate class-weighted models
    logger.info("=== Phase 1: Training Class-Weighted Models ===")
    for name, clf in model_definitions.items():
        t0 = time.time()
        logger.info(f"Training {name} with algorithmic class weighting...")
        clf.fit(X_train, y_train)
        duration = time.time() - t0

        # Predict on validation set
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, y_pred, y_prob)
        metrics["Model"] = name
        metrics["Method"] = "Class_Weights"
        metrics["Train_Time_Sec"] = round(duration, 2)
        val_results.append(metrics)
        trained_models[name] = clf

        # Save individual model
        save_path = models_dir / f"{name.lower()}.joblib"
        joblib.dump(clf, save_path)
        logger.info(f"Saved {name} to {save_path} | Val ROC-AUC: {metrics['ROC-AUC']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f}")

        if track_mlflow:
            try:
                log_training_run(
                    clf=clf,
                    name=name,
                    method="Class_Weights",
                    metrics=metrics,
                    y_val=y_val,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    random_state=random_state,
                    model_save_path=save_path,
                    models_dir=models_dir,
                    imbalance_ratio=imbalance_ratio
                )
            except Exception as e:
                logger.warning(f"Could not log {name} to MLflow: {e}")

    # 2. Evaluate SMOTE experiment (applying SMOTE strictly to training data only)
    if evaluate_smote:
        logger.info("=== Phase 2: Evaluating SMOTE Experiment (Training Data Only) ===")
        smote = SMOTE(random_state=random_state)
        logger.info("Applying SMOTE oversampling strictly to X_train...")
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        logger.info(f"Resampled training set shape: {X_train_smote.shape} | New class balance: 50/50")

        smote_definitions = {
            "Logistic_Regression_SMOTE": LogisticRegression(
                max_iter=1000,
                C=0.1,
                solver="lbfgs",
                random_state=random_state
            ),
            "Random_Forest_SMOTE": RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=random_state
            ),
            "XGBoost_SMOTE": XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=random_state
            )
        }

        for name, clf in smote_definitions.items():
            t0 = time.time()
            logger.info(f"Training {name} on SMOTE resampled data...")
            clf.fit(X_train_smote, y_train_smote)
            duration = time.time() - t0

            # Evaluate on un-resampled, real validation distribution
            y_pred = clf.predict(X_val)
            y_prob = clf.predict_proba(X_val)[:, 1]

            metrics = compute_metrics(y_val, y_pred, y_prob)
            metrics["Model"] = name
            metrics["Method"] = "SMOTE_Oversampling"
            metrics["Train_Time_Sec"] = round(duration, 2)
            val_results.append(metrics)
            trained_models[name] = clf

            save_path = models_dir / f"{name.lower()}.joblib"
            joblib.dump(clf, save_path)
            logger.info(f"Saved {name} to {save_path} | Val ROC-AUC: {metrics['ROC-AUC']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f}")

            if track_mlflow:
                try:
                    log_training_run(
                        clf=clf,
                        name=name,
                        method="SMOTE_Oversampling",
                        metrics=metrics,
                        y_val=y_val,
                        y_pred=y_pred,
                        y_prob=y_prob,
                        random_state=random_state,
                        model_save_path=save_path,
                        models_dir=models_dir,
                        smote_obj=smote
                    )
                except Exception as e:
                    logger.warning(f"Could not log {name} to MLflow: {e}")

    # Build comparison summary table
    val_comparison_df = pd.DataFrame(val_results)
    # Order columns
    cols_order = ["Model", "Method", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "Train_Time_Sec"]
    val_comparison_df = val_comparison_df[cols_order].sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)

    # Determine best model based on PR-AUC (primary metric for imbalanced churn) and F1-Score
    best_model_name = val_comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_save_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, best_save_path)
    logger.info(f"Selected Best Model on Validation Set: {best_model_name} (Saved to {best_save_path})")

    return trained_models, val_comparison_df, best_model_name


if __name__ == "__main__":
    from src.ml.data_loader import load_raw_dataset, split_data
    from src.ml.preprocessor import fit_and_transform_splits

    df = load_raw_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_p, X_val_p, X_test_p, _, _ = fit_and_transform_splits(X_train, X_val, X_test)
    models, val_df, best_name = train_models(X_train_p, y_train.values, X_val_p, y_val.values)
    print("\nValidation Comparison Table:")
    print(val_df.to_string(index=False))
