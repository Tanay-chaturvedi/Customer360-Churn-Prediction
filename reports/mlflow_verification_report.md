# Customer360 – MLflow Verification & Audit Report

**Date of Audit**: 2026-11-03
**Auditor**: Automated Verification Suite & Agentic Diagnostics
**Tracking URI**: `sqlite:///mlflow/mlflow.db`
**Overall Verification Status**: **PASS**

---

## 1. Executive Summary

| Verification Area | Status | Evidence / Summary |
| :--- | :---: | :--- |
| **MLflow UI Accessibility** | **PASS** | Server responds HTTP 200 at `http://127.0.0.1:5000/`. Accessible via PowerShell command with SQLite backend. |
| **SQLite Database Health** | **PASS** | `mlflow/mlflow.db` schema intact with 64 system/tracking tables. Zero data corruption. |
| **Experiment Configuration** | **PASS** | Experiment `Customer360_Churn_Prediction` active with artifact repository at `mlflow/artifacts/`. |
| **Experiment Runs Count** | **PASS** | Exactly 12 runs tracked (2 complete executions of the 6 core model architectures). No missing runs. |
| **Parameters Verification** | **PASS** | Verified dynamically extracted hyperparameters from estimator objects matching training definitions. |
| **Metrics Verification** | **PASS** | All standard validation metrics numeric and identical to `reports/model_comparison.csv`. |
| **Champion Model Tagging** | **PASS** | XGBoost (Class_Weights) tagged `model_role=champion`. Candidates tagged `model_role=candidate`. |
| **Artifact Availability & Integrity** | **PASS** | All 12 runs verified: model `.joblib`, `preprocessor.joblib`, `feature_names.joblib`, PNG plots, and reports load cleanly. |
| **FastAPI & Service Integration** | **PASS** | `/health`, `/predict`, `/predict/batch`, and full dataset scoring (7,043 records in 0.06s) operational. |
| **Pytest Suite** | **PASS** | 33 passed tests (28 API endpoint tests + 5 MLflow tracking unit tests). |
| **Browser Visual Capture** | **NOT VERIFIED** | Playwright driver binary unavailable on host (`404 Not Found`). Stated transparently per audit guidelines. |

---

## 2. MLflow UI Verification

- **Status**: **PASS**
- **Endpoint**: `http://127.0.0.1:5000`
- **Execution Command**:
  ```powershell
  mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 127.0.0.1 --port 5000
  ```
- **Evidence**:
  - Live probe via HTTP request to `http://127.0.0.1:5000/` returned status code **`200 OK`**.
  - Tracking server processes running under Python 3.13 (`uvicorn` ASGI server with SQLite connection pooling).
  - Experiment `Customer360_Churn_Prediction` (ID: 1) is active and bound to the shared SQLite database.

---

## 3. SQLite Tracking Database Verification

- **Status**: **PASS**
- **Database File**: [`mlflow/mlflow.db`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/mlflow/mlflow.db)
- **Database Size**: 1,118,208 bytes (1.12 MB)
- **Key Tables Audited**:
  - `experiments`: 2 records (ID 0: `Default`, ID 1: `Customer360_Churn_Prediction`).
  - `runs`: 12 records (all status `FINISHED`, lifecycle stage `active`).
  - `params`: 316 total entries (56 distinct keys recorded across model families).
  - `metrics`: 192 total metric measurements (26 distinct metric names across validation and test phases).
  - `tags`: Run-level tags for `model_role`, `model_name`, `method`, `estimator_type`, and source tracking.
- **Credential Safety**: No credentials, passwords, or secret connection strings are logged in the database or reports.

---

## 4. Experiment Runs Verification

- **Status**: **PASS**
- **Actual Experiment Count**: **2** (`Default` [ID 0], `Customer360_Churn_Prediction` [ID 1])
- **Actual Run Count**: **12 runs** (Two complete training pipeline executions; run history preserved without overwriting)

### Breakdown of Existing Model Runs

| # | Run UUID | Model Name | Method / Balancing | Lifecycle | Status | Tag: `model_role` |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `8d6d5d3b...` | `Logistic_Regression` | Class_Weights | active | FINISHED | `candidate` |
| 2 | `877c218f...` | `Random_Forest` | Class_Weights | active | FINISHED | `candidate` |
| 3 | `eae34356...` | `XGBoost` | Class_Weights | active | FINISHED | `champion` |
| 4 | `966c4d3c...` | `Logistic_Regression_SMOTE` | SMOTE_Oversampling | active | FINISHED | `candidate` |
| 5 | `e653c27b...` | `Random_Forest_SMOTE` | SMOTE_Oversampling | active | FINISHED | `candidate` |
| 6 | `9730a311...` | `XGBoost_SMOTE` | SMOTE_Oversampling | active | FINISHED | `candidate` |
| 7 | `11acfcf5...` | `Logistic_Regression` | Class_Weights | active | FINISHED | `candidate` |
| 8 | `f96d14fc...` | `Random_Forest` | Class_Weights | active | FINISHED | `candidate` |
| 9 | `e57e34aa...` | `XGBoost` | Class_Weights | active | FINISHED | `champion` |
| 10 | `fecf8b89...` | `Logistic_Regression_SMOTE` | SMOTE_Oversampling | active | FINISHED | `candidate` |
| 11 | `b428ee41...` | `Random_Forest_SMOTE` | SMOTE_Oversampling | active | FINISHED | `candidate` |
| 12 | `426d1bf6...` | `XGBoost_SMOTE` | SMOTE_Oversampling | active | FINISHED | `candidate` |

*Note: Runs 1–6 represent the first execution batch; Runs 7–12 represent the second execution batch verifying repeatability. All 6 required model configurations are tracked in both batches.*

---

## 5. Parameters Verification

- **Status**: **PASS**
- **Extraction Source**: Directly extracted from fitted scikit-learn / XGBoost estimator instances via `get_params()`, alongside verified split configurations.
- **Evidence of Recorded Values**:

### 1. `Logistic_Regression` (Class Weights)
- `model_name`: `Logistic_Regression`
- `method`: `Class_Weights`
- `random_seed`: `42`
- `train_split`: `0.70`, `val_split`: `0.15`, `test_split`: `0.15`
- `class_weight`: `balanced`
- `imbalance_ratio`: `2.7683`
- `C`: `0.1`, `max_iter`: `1000`, `solver`: `lbfgs`, `penalty`: `l2`

### 2. `Random_Forest` (Class Weights)
- `model_name`: `Random_Forest`
- `method`: `Class_Weights`
- `random_seed`: `42`
- `class_weight`: `balanced`
- `n_estimators`: `300`, `max_depth`: `10`, `min_samples_split`: `10`, `min_samples_leaf`: `4`, `n_jobs`: `-1`

### 3. `XGBoost` (Class Weights - Champion)
- `model_name`: `XGBoost`
- `method`: `Class_Weights`
- `random_seed`: `42`
- `scale_pos_weight`: `2.7683`
- `imbalance_ratio`: `2.7683`
- `n_estimators`: `250`, `max_depth`: `4`, `learning_rate`: `0.03`, `subsample`: `0.8`, `colsample_bytree`: `0.8`, `eval_metric`: `logloss`

### 4. `Logistic_Regression_SMOTE`
- `model_name`: `Logistic_Regression_SMOTE`
- `method`: `SMOTE_Oversampling`
- `random_seed`: `42`
- `smote_sampling_strategy`: `auto`
- `smote_k_neighbors`: `5`, `smote_random_state`: `42`
- `C`: `0.1`, `max_iter`: `1000`, `solver`: `lbfgs`

### 5. `Random_Forest_SMOTE`
- `model_name`: `Random_Forest_SMOTE`
- `method`: `SMOTE_Oversampling`
- `random_seed`: `42`
- `smote_sampling_strategy`: `auto`, `smote_k_neighbors`: `5`, `smote_random_state`: `42`
- `n_estimators`: `300`, `max_depth`: `10`, `min_samples_split`: `10`, `min_samples_leaf`: `4`, `n_jobs`: `-1`

### 6. `XGBoost_SMOTE`
- `model_name`: `XGBoost_SMOTE`
- `method`: `SMOTE_Oversampling`
- `random_seed`: `42`
- `smote_sampling_strategy`: `auto`, `smote_k_neighbors`: `5`, `smote_random_state`: `42`
- `n_estimators`: `250`, `max_depth`: `4`, `learning_rate`: `0.03`, `subsample`: `0.8`, `colsample_bytree`: `0.8`

---

## 6. Metrics Verification

- **Status**: **PASS**
- **Evaluation Methodology & Test Set Disclosure**: Model selection in the corrected pipeline uses validation PR-AUC and F1-score only. However, during an earlier development iteration, multiple candidate models were evaluated on the test set before the evaluation workflow was corrected. Therefore, the test set was not completely untouched throughout development. Final test metrics should be interpreted with this limitation in mind.
- **Cross-Verification with [`reports/model_comparison.csv`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/reports/model_comparison.csv)**:

| Run Name | Split | MLflow PR-AUC | CSV PR-AUC | MLflow ROC-AUC | CSV ROC-AUC | MLflow Recall | CSV Recall | MLflow F1 | CSV F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** (Champion) | Validation | **0.6435** | **0.6435** | **0.8383** | **0.8383** | **76.51%** | **76.51%** | **0.6250** | **0.6250** | **MATCH** |
| **XGBoost_SMOTE** | Validation | **0.6373** | **0.6373** | **0.8392** | **0.8392** | **65.84%** | **65.84%** | **0.6293** | **0.6293** | **MATCH** |
| **Random_Forest** | Validation | **0.6360** | **0.6360** | **0.8351** | **0.8351** | **75.80%** | **75.80%** | **0.6256** | **0.6256** | **MATCH** |
| **Logistic_Regression_SMOTE** | Validation | **0.6280** | **0.6280** | **0.8290** | **0.8290** | **76.16%** | **76.16%** | **0.6203** | **0.6203** | **MATCH** |
| **Logistic_Regression** | Validation | **0.6246** | **0.6246** | **0.8286** | **0.8286** | **76.51%** | **76.51%** | **0.6152** | **0.6152** | **MATCH** |
| **Random_Forest_SMOTE** | Validation | **0.6188** | **0.6188** | **0.8322** | **0.8322** | **68.33%** | **68.33%** | **0.6275** | **0.6275** | **MATCH** |
| **XGBoost** (Final Test) | Held-Out Test | **0.6651** | **0.6651** | **0.8525** | **0.8525** | **80.36%** | **80.36%** | **0.6383** | **0.6383** | **MATCH** |

---

## 7. Champion Model Verification

- **Status**: **PASS**
- **Expected Champion**: `XGBoost` (Class_Weights)
- **Active Champion Run ID**: `e57e34aaa8734b50955793efc43a1299` (Previous batch champion: `eae34356dcf54fd2a6d928f9f1759a00`)
- **Tags Verified**:
  - `model_role`: `champion` (All other 5 candidate runs have `model_role=candidate`)
  - `champion_selected`: `true`
  - `model_name`: `XGBoost`
  - `method`: `Class_Weights`
- **Artifact Compatibility**:
  - Champion model artifact saved at [`models/best_model.joblib`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/models/best_model.joblib).
  - Loaded model type: `xgboost.sklearn.XGBClassifier`.
  - Exactly identical to the champion artifact archived in MLflow.

---

## 8. Artifact Availability & Integrity Verification

- **Status**: **PASS**
- **Test Methodology**: Every artifact in `mlflow/artifacts/<run_uuid>/artifacts/` was opened, deserialized via `joblib.load()`, or verified with `PIL.Image.verify()`.

| Artifact Category | Verified Path in MLflow Store | Loading Check Result |
| :--- | :--- | :---: |
| **Trained Estimator** | `model/*.joblib` | **PASS** – Loaded directly into `LogisticRegression`, `RandomForestClassifier`, or `XGBClassifier` without skops dependencies. |
| **Preprocessor** | `preprocessor/preprocessor.joblib` | **PASS** – Loaded directly as scikit-learn `Pipeline` (steps: `cleaner`, `transformer`). |
| **Feature Names** | `preprocessor/feature_names.joblib` | **PASS** – Loaded as Python list of 46 feature name strings. |
| **Confusion Matrix** | `evaluation_plots/confusion_matrix.png` | **PASS** – Verified PNG image header & byte integrity across all 12 runs. |
| **ROC Curve** | `evaluation_plots/roc_curve.png` | **PASS** – Verified PNG image header & byte integrity across all 12 runs. |
| **PR Curve** | `evaluation_plots/pr_curve.png` | **PASS** – Verified PNG image header & byte integrity across all 12 runs. |
| **Run Summary** | `reports/run_summary.md` | **PASS** – Verified readable markdown file with per-run validation metrics. |
| **Model Comparison CSV** | `reports/model_comparison.csv` | **PASS** – Verified readable CSV with full candidate & test benchmarks. |
| **Evaluation Summary MD** | `reports/evaluation_summary.md` | **PASS** – Verified audited markdown summary report intact. |
| **Overall Comparison Curves** | `reports/confusion_matrices.png`, `reports/roc_curves.png`, `reports/pr_curves.png` | **PASS** – Verified on champion run. |

---

## 9. API & System Integration Verification

- **Status**: **PASS**
- **FastAPI Endpoints Tested**:
  1. `GET /health`: Returned HTTP 200, `{"status": "ok", "model_loaded": true, "model_name": "XGBClassifier"}`.
  2. `POST /predict`: Returned HTTP 200, scored test customer with `churn_probability = 0.8499`, `predicted_churn = "Yes"`, `risk_category = "High Risk"`.
  3. `POST /predict/batch`: Returned HTTP 200, scored multiple customers with preserved IDs.
- **Batch Dataset Simulation (Streamlit Backend Logic)**:
  - Scored all **7,043 customer records** from `data/raw/customer_churn.csv` in **0.06 seconds** using the champion model and preprocessor.
  - Risk distribution: 1,658 High Risk (23.5%), 1,644 Medium Risk (23.3%), 3,741 Low Risk (53.1%).
- **Pytest Suite**:
  - `tests/test_api.py`: 28 passed.
  - `tests/test_tracking.py`: 5 passed.
  - **Total**: 33 passed, 0 failed.

---

## 10. Browser-Level UI Inspection Status

- **Status**: **NOT VERIFIED (via Headless Browser)** / **PASS (via Direct HTTP & SQLite API)**
- **Finding**: Headless browser automation via Playwright was unable to launch due to Microsoft Azureedge returning `404 Not Found` for the specific Windows Playwright driver build (`playwright-1.57.0-win32_x64.zip`).
- **Transparency Statement**: As required by Section 9 ("*If browser inspection or screenshots are unavailable, clearly state that instead of claiming visual verification*"), browser screenshots are omitted.
- **Manual Verification Instructions**:
  The user can view the MLflow UI in any local browser:
  1. Open [http://127.0.0.1:5000](http://127.0.0.1:5000)
  2. Click experiment **Customer360_Churn_Prediction**
  3. Inspect the runs table and click the top **XGBoost** run (tagged `model_role=champion`)
  4. View Parameters, Metrics, and Artifacts tabs.

---

## 11. Issues Discovered and Fixes Performed

1. **Issue Discovered**: In `src/ml/tracking.py`, when extracting hyperparameters for `XGBoost_SMOTE`, `clf.scale_pos_weight` was `None`, causing `float(clf.scale_pos_weight)` to raise a `TypeError`.
   - **Fix Performed**: Added a check in `src/ml/tracking.py` to ensure `scale_pos_weight is not None` before casting to float, falling back safely to string or omitting it.
2. **Issue Discovered**: Deserialization of `preprocessor.joblib` outside the project root required `src` to be in `sys.path` due to custom transformer `DataCleaner`.
   - **Fix Performed**: Verified `sys.path` handling in test scripts and confirmed that FastAPI and Streamlit already import with project root in `sys.path`.
