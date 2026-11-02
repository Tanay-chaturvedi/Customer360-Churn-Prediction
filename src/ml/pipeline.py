"""Master Machine Learning Pipeline Orchestrator for Customer360.

Executes the end-to-end reproducible ML training, imbalance handling,
validation selection, and final test evaluation in a single command:

    python -m src.ml.pipeline
"""

import sys
import time
import logging
from pathlib import Path
from src.ml.data_loader import load_raw_dataset, split_data
from src.ml.preprocessor import fit_and_transform_splits
from src.ml.train import train_models
from src.ml.evaluate import evaluate_pipeline

# Configure structured pipeline logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Customer360.MLPipeline")


def run_pipeline() -> None:
    """Execute the end-to-end Machine Learning pipeline adhering to strict validation selection."""
    pipeline_start = time.time()
    logger.info("=" * 80)
    logger.info("STARTING CUSTOMER360 MACHINE LEARNING PIPELINE (AUDITED)")
    logger.info("=" * 80)

    # 1. Load Data
    logger.info("\n[STEP 1/4] Ingesting and Splitting Dataset (70% Train, 15% Val, 15% Test)...")
    df = load_raw_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size=0.15, val_size=0.15, random_state=42
    )

    # 2. Preprocess Data
    logger.info("\n[STEP 2/4] Fitting Preprocessor (Leakage Prevention - Fit strictly on Train)...")
    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train_proc, X_val_proc, X_test_proc, preprocessor, feature_names = fit_and_transform_splits(
        X_train, X_val, X_test, save_path=models_dir / "preprocessor.joblib"
    )

    # 3. Model Training & Validation Set Benchmark (Model Selection Phase)
    logger.info("\n[STEP 3/4] Training Models & Selecting Champion on Validation Set...")
    models, val_comparison_df, best_model_name = train_models(
        X_train=X_train_proc,
        y_train=y_train.values,
        X_val=X_val_proc,
        y_val=y_val.values,
        evaluate_smote=True,
        models_dir=models_dir,
        random_state=42
    )

    print("\n" + "=" * 80)
    print("STAGE 1: VALIDATION SET BENCHMARK RESULTS (MODEL SELECTION)")
    print("=" * 80)
    print(val_comparison_df.to_string(index=False))

    # 4. Final Evaluation on Held-Out Test Set (Selected Champion Only)
    logger.info(f"\n[STEP 4/4] Evaluating Selected Champion '{best_model_name}' on Held-Out Test Set...")
    unified_df, test_metrics = evaluate_pipeline(
        models=models,
        best_model_name=best_model_name,
        X_val=X_val_proc,
        y_val=y_val.values,
        X_test=X_test_proc,
        y_test=y_test.values,
        reports_dir=reports_dir
    )

    print("\n" + "=" * 80)
    print("STAGE 2: AUDITED MODEL COMPARISON & FINAL TEST BENCHMARK")
    print("=" * 80)
    print(unified_df.to_string(index=False))

    total_time = time.time() - pipeline_start
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
    print(f"Champion Model Selected : {best_model_name} (from Validation benchmark)")
    print(f"Final Test PR-AUC       : {test_metrics['PR-AUC']:.4f}")
    print(f"Final Test ROC-AUC      : {test_metrics['ROC-AUC']:.4f}")
    print(f"Final Test Recall       : {test_metrics['Recall']*100:.2f}%")
    print(f"MLflow Tracking URI     : sqlite:///mlflow/mlflow.db")
    print(f"MLflow Experiment       : Customer360_Churn_Prediction")
    print(f"MLflow UI Command       : mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 127.0.0.1 --port 5000")
    print(f"Artifacts Saved to      : {models_dir.resolve()}")
    print(f"Reports & Plots Saved to: {reports_dir.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)
