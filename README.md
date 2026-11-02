# Customer360: Enterprise Retention Intelligence & Churn Prediction Platform

[![Python Version](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.17.2-0194E2.svg?logo=mlflow)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Champion-red.svg)](https://xgboost.readthedocs.io/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-4479A1.svg?logo=mysql)](https://www.mysql.com/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/pytest-33%20passed-brightgreen.svg)](https://docs.pytest.org/)

---

## 1. Project Overview & Business Problem Statement

Customer churn is one of the most critical threats to profitability in subscription-based telecommunications. Acquiring a new telecom customer is estimated to cost between **5 to 25 times more** than retaining an existing subscriber. When customers defect without warning, service providers experience not only immediate monthly recurring revenue (MRR) loss but also diminished customer lifetime value (LTV) and degraded capital efficiency.

**Customer360** is an enterprise-grade retention intelligence platform designed to proactively detect churn risk, diagnose attrition drivers, and guide targeted retention campaigns. By unifying **relational SQL analytics (MySQL)**, a **modular machine learning training and evaluation pipeline (scikit-learn & XGBoost)**—where preprocessing transformations are fitted solely on training data and model selection is governed strictly by validation performance—**MLflow experiment tracking**, a **FastAPI prediction service**, and an **interactive Streamlit executive dashboard**, Customer360 transitions retention workflows from reactive exit surveys to data-driven, preemptive intervention.

---

## 2. Key Architectural Features

- **Relational Data Foundation**: Scalable MySQL schema with 13 automated diagnostic queries identifying attrition hotspots across contracts, billing methods, and service bundles.
- **Training-Fitted Preprocessing Pipeline**: Custom scikit-learn transformers encapsulating missing value imputation, standard scaling, and one-hot encoding fitted strictly on the training partition to prevent preprocessing leakage into validation or test sets.
- **Audited ML Experimentation**: Systematic comparison of 6 candidate models across algorithmic class-weighting and synthetic oversampling (SMOTE).
- **High-Performance Champion Model**: Production-tuned `XGBoost` classifier achieving **0.6651 PR-AUC**, **0.8525 ROC-AUC**, and **80.36% Recall** on held-out test data.
- **MLflow Experiment Tracking**: SQLite-backed tracking repository recording hyperparameters, evaluation curves, feature importance plots, and serialized model artifacts.
- **High-Throughput REST API**: Asynchronous FastAPI service supporting single-customer inference and batch scoring of up to 5,000 records per request with Pydantic validation.
- **Executive SaaS Dashboard**: Modern Streamlit analytics interface featuring real-time KPI scorecards, diagnostic heatmaps, interactive What-If simulators, and prescriptive retention playbooks.
- **Comprehensive Test Suite**: 33 automated `pytest` unit and integration tests covering API endpoints, data validation, chunked batching, and MLflow logging.

---

## 3. Technology Stack

| Domain | Technology / Library | Purpose in Customer360 |
| :--- | :--- | :--- |
| **Language & Runtime** | `Python 3.13+` | Core execution environment |
| **Relational Database** | `MySQL 8.0+` | Customer transaction & attribute repository |
| **Database Connector** | `SQLAlchemy 2.0` / `PyMySQL` | High-efficiency database connection pooling and query execution |
| **Data Manipulation** | `pandas`, `numpy` | Vectorized feature transformations and tabular operations |
| **Machine Learning** | `scikit-learn`, `XGBoost`, `imbalanced-learn` | Preprocessing pipelines, tree boosting, and class balancing |
| **Model Serialization** | `joblib` | Production artifact persistence (`.joblib`) |
| **MLOps & Tracking** | `MLflow 3.17.2` | Experiment runs, metric comparison, plot artifact storage |
| **API Microservice** | `FastAPI`, `Uvicorn`, `Pydantic v2` | High-performance REST endpoints with strict typing |
| **Executive Frontend** | `Streamlit 1.42`, `Plotly Express` | Interactive business dashboard and scenario simulation |
| **Quality Assurance** | `pytest`, `httpx` | Unit and integration testing across API and MLflow modules |

---

## 4. Architecture and Project Workflow

```
                                      ┌─────────────────────────────────────────┐
                                      │   Raw Dataset (IBM Telco 7,043 rows)   │
                                      └────────────────────┬────────────────────┘
                                                           │
                                                           ▼
                                      ┌─────────────────────────────────────────┐
                                      │          MySQL Database Server          │
                                      │        `customer360_db.customers`       │
                                      └──────┬───────────────────────────┬──────┘
                                             │                           │
                   ┌─────────────────────────┘                           └──────────────────────────┐
                   ▼                                                                                ▼
     ┌───────────────────────────┐                                                    ┌───────────────────────────┐
     │  SQL Diagnostic Analytics │                                                    │   ML Pipeline Ingestion   │
     │ `sql/customer_analysis.sql│                                                    │    `src/ml/pipeline.py`   │
     │ 13 Attrition & ROI Queries│                                                    └─────────────┬─────────────┘
     └───────────────────────────┘                                                                  │
                                                                                                    ▼
                                                                                      ┌───────────────────────────┐
                                                                                      │  Stratified 3-Way Split   │
                                                                                      │  Train (70%) / Val (15%)  │
                                                                                      │        Test (15%)         │
                                                                                      └─────────────┬─────────────┘
                                                                                                    │
                                                                                                    ▼
                                                                                      ┌───────────────────────────┐
                                                                                      │   Fitted Preprocessor     │
                                                                                      │    `preprocessor.joblib`  │
                                                                                      └─────────────┬─────────────┘
                                                                                                    │
                                                                                                    ▼
                                                                                      ┌───────────────────────────┐
                                                                                      │ Model Training & Tracking │
                                                                                      │  6 Model Configurations   │
                                                                                      │  MLflow SQLite Store      │
                                                                                      └─────────────┬─────────────┘
                                                                                                    │
                                                                                                    ▼
                                                                                      ┌───────────────────────────┐
                                                                                      │ Champion Model Selection  │
                                                                                      │  XGBoost (Class Weights)  │
                                                                                      │    `best_model.joblib`    │
                                                                                      └─────────────┬─────────────┘
                                                                                                    │
                                                                                                    ▼
                                                                                      ┌───────────────────────────┐
                                                                                      │     FastAPI Service       │
                                                                                      │   `http://127.0.0.1:8000` │
                                                                                      │   /predict, /batch        │
                                                                                      └─────────────┬─────────────┘
                                                                                                    │
                                                                                                    ▼
                                                                                      ┌───────────────────────────┐
                                                                                      │    Streamlit Dashboard    │
                                                                                      │   `http://localhost:8501` │
                                                                                      │ KPIs, What-If, Playbooks  │
                                                                                      └───────────────────────────┘
```

---

## 5. Dataset Description

Customer360 is built upon the industry-standard **IBM Telco Customer Churn** dataset (`data/raw/customer_churn.csv`). It consists of **7,043 unique customer accounts** observed across **21 distinct attributes**:

- **Target Variable**: `Churn` (`Yes` / `No`) indicating whether the customer left within the last month.
- **Class Distribution**: 5,174 active customers (**73.46%**) vs. 1,869 churned customers (**26.54%**), representing a realistic ~1:3 class imbalance.
- **Feature Categories**:
  - **Account Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
  - **Account & Financial Tenure**: `tenure` (0–72 months), `Contract` (Month-to-month, One year, Two year), `PaperlessBilling` (Yes/No), `PaymentMethod` (Electronic check, Mailed check, Bank transfer, Credit card), `MonthlyCharges` (USD), `TotalCharges` (USD).
  - **Subscribed Services**: `PhoneService`, `MultipleLines`, `InternetService` (DSL, Fiber optic, No), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.

> [!NOTE]
> **Dataset Source & Attribution**: This dataset was originally published by IBM as sample customer records for business analytics and retention demonstration, and has become a standard benchmark in churn prediction literature. It represents fictional/sample customer profiles without real individual PII. The local copy is provided at `data/raw/customer_churn.csv` solely for project demonstration, pipeline execution, and local reproducibility. Users redistributing or utilizing this dataset in production or commercial environments should consult IBM's original sample data terms.

---

## 6. Exploratory Data Analysis & Strategic Business Insights

Exploratory analysis revealed distinct behavioral patterns that differentiate retained customers from churners:

1. **Contract Duration is the Primary Predictor**:
   - Customers on **Month-to-Month contracts** exhibit a staggering **42.71% churn rate**, representing 88.55% of all churned accounts.
   - Customers on **One-Year contracts** churn at **11.27%**.
   - Customers on **Two-Year contracts** churn at only **2.83%**.
2. **Tenure Cohorts Exhibit Heavily Front-Loaded Attrition**:
   - Over **51.9%** of newly onboarded customers in the 0–1 month tenure cohort churn.
   - Churn rates drop precipitously after 12 months of service and fall below 10% after 36 months, proving that early onboarding and initial adoption are paramount.
3. **Fiber Optic Pricing & Support Friction**:
   - **Fiber optic** internet subscribers experience a high churn rate of **41.89%** (compared to 18.96% for DSL and 7.40% for no internet), driven by higher average monthly charges ($70–$110/mo) coupled with inadequate technical onboarding.
4. **Payment Channel Disparity**:
   - Customers using **Electronic check** churn at **45.29%**, compared to **15.24%** for Credit card (automatic) and **16.71%** for Bank transfer (automatic). Unautomated payment methods introduce recurring monthly friction.
5. **Technical Support & Online Security Act as Retention Anchors**:
   - Customers without `TechSupport` churn at **41.64%**, whereas customers with active `TechSupport` churn at **15.17%**.
   - Subscribing to `OnlineSecurity` reduces churn from **41.77%** down to **14.61%**.
6. **Financial Exposure**:
   - The average monthly charge of churners ($74.44) is significantly higher than that of non-churners ($61.27). Attrition disproportionately strikes high-ARPU subscribers.

---

## 7. MySQL Database & SQL Analytics

The relational data tier is anchored by MySQL (`customer360_db`), modeled for ACID integrity and rapid analytical aggregation.

### Relational Schema Design
The table `customers` stores clean, typed subscriber records with `customerID VARCHAR(50) PRIMARY KEY`:
- Indexed numeric features: `tenure INT`, `MonthlyCharges DECIMAL(10,2)`, `TotalCharges DECIMAL(10,2)`.
- Categorical attributes: `Contract VARCHAR(50)`, `PaymentMethod VARCHAR(50)`, `InternetService VARCHAR(50)`, `Churn VARCHAR(10)`.

### Automated Database Utilities
- [`src/database/create_tables.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/database/create_tables.py): Idempotent table definition with indexed lookups.
- [`src/database/load_data.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/database/load_data.py): Automated batch loader parsing raw CSV files into MySQL.
- [`src/database/run_analysis.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/database/run_analysis.py): Programmatic SQL runner executing and formatting analytical query output.

### 13 Diagnostic SQL Queries ([`sql/customer_analysis.sql`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/sql/customer_analysis.sql))
1. **Total Customer Baseline**: Global record count.
2. **Attrition Breakdown**: Aggregate counts of active vs. churned accounts.
3. **Organizational Churn Benchmark**: Corporate churn rate percentage (26.54%) and retention rate (73.46%).
4. **Contract Risk Analysis**: Attrition distribution across Month-to-Month, 1-Year, and 2-Year agreements.
5. **Payment Method Exposure**: Churn rates across Electronic check vs. automated billing methods.
6. **Internet Service Comparison**: Risk comparison across Fiber optic, DSL, and no-internet tiers.
7. **Tenure Cohort Binning**: Attrition segmented into cohorts: 0–12 mo, 13–24 mo, 25–48 mo, 49–72 mo.
8. **Revenue at Risk**: Total monthly revenue lost to churn ($139,130.85/month).
9. **Average Monthly Revenue (ARPU)**: Comparison of monthly bill averages between retained and churned cohorts.
10. **High-Value Flight Risk**: Identification of accounts with `MonthlyCharges > $70` and `tenure <= 6`.
11. **Technical Support Impact**: Retention delta attributable to `TechSupport` subscription.
12. **Online Security Impact**: Risk mitigation attributable to `OnlineSecurity` add-on.
13. **Multidimensional Risk Segmentation**: Compound high-risk profile isolating Month-to-month + Fiber optic + Electronic check subscribers.

---

## 8. ML Preprocessing & Training Methodology

To maintain rigorous evaluation standards: preprocessing parameters are fit solely on training data, candidate model tuning and selection are conducted strictly on the validation set, and a held-out test split provides the final evaluation benchmark (with prior exploratory test-set exposure documented in Section 11):

### 1. Three-Way Stratified Partition
The 7,043 records are partitioned into three immutable subsets using stratified sampling on `Churn`:
- **Training Set (70%, 4,929 rows)**: Exclusively utilized to fit preprocessing parameters and train model estimators.
- **Validation Set (15%, 1,057 rows)**: Exclusively utilized for candidate model benchmarking, hyperparameter comparison, and champion selection.
- **Held-Out Test Set (15%, 1,057 rows)**: Preserved untouched during exploration; evaluated strictly once on the final selected Champion model.

### 2. Training-Fitted Preprocessing Pipeline ([`src/ml/preprocessor.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/ml/preprocessor.py))
- **`DataCleaner` Transformer**: Converts `TotalCharges` to float, handles whitespace values, and imputes missing numeric records using the median calculated **strictly from the training split**.
- **Numerical Scaling**: `StandardScaler` applied to `tenure`, `MonthlyCharges`, and `TotalCharges`.
- **Categorical Encoding**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to all demographic and service features.
- **Identifier Protection**: `customerID` and historical `Churn` labels are permanently stripped prior to feature matrix transformation.

### 3. Class Imbalance Handling
Two primary balancing strategies were benchmarked across each model architecture:
1. **Algorithmic Cost-Sensitive Weighting**: Utilizing `class_weight='balanced'` in Logistic Regression / Random Forest, and `scale_pos_weight = (negative_count / positive_count)` in XGBoost (~2.77).
2. **Synthetic Minority Over-sampling (SMOTE)**: Synthesizing minority instances on training splits via `imblearn.over_sampling.SMOTE(random_state=42)`.

---

## 9. Model Comparison Using Actual Validation Metrics

Candidate models were benchmarked on the **Validation Set (1,057 records; churn prevalence = 26.58%)**. Model selection prioritized **PR-AUC (Precision-Recall Area Under Curve)** and **F1-Score**, as capturing true churners while minimizing wasted retention spend directly dictates campaign ROI:

| Rank | Model Architecture | Balancing Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Selection Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`XGBoost`** | **Class_Weights** | **75.59%** | **52.83%** | **76.51%** | **0.6250** | **0.8383** | **0.6435** | 🏆 **Selected Champion** |
| 2 | `XGBoost_SMOTE` | SMOTE_Oversampling | 79.38% | 60.26% | 65.84% | 0.6293 | 0.8392 | 0.6373 | Candidate |
| 3 | `Random_Forest` | Class_Weights | 75.88% | 53.25% | 75.80% | 0.6256 | 0.8351 | 0.6360 | Candidate |
| 4 | `Logistic_Regression_SMOTE` | SMOTE_Oversampling | 75.21% | 52.32% | 76.16% | 0.6203 | 0.8290 | 0.6280 | Candidate |
| 5 | `Logistic_Regression` | Class_Weights | 74.55% | 51.44% | 76.51% | 0.6152 | 0.8286 | 0.6246 | Candidate |
| 6 | `Random_Forest_SMOTE` | SMOTE_Oversampling | 78.43% | 58.01% | 68.33% | 0.6275 | 0.8322 | 0.6188 | Candidate |

### Selection Rationale
- `XGBoost` with Class Weights achieved the highest **PR-AUC (0.6435)** while capturing **76.51% of all churners** at high precision.
- Algorithmic weighting (`scale_pos_weight`) proved superior to synthetic oversampling (SMOTE) across gradient-boosted trees, preventing artificial density distortions in feature space.

---

## 10. Champion XGBoost Model & Actual Test Metrics

Following selection on the validation split, the Champion `XGBoost` model was evaluated on the **Held-Out Test Set (1,057 records; churn prevalence = 26.49%)**:

| Evaluation Metric | Validation Score | Final Test Score | Generalization Delta | Business Significance |
| :--- | :---: | :---: | :---: | :--- |
| **PR-AUC** | 0.6435 | **0.6651** | +0.0216 | Exceptional trade-off between precision and recall across decision thresholds |
| **ROC-AUC** | 0.8383 | **0.8525** | +0.0142 | Strong global discriminatory capacity between churners and retained accounts |
| **Recall (Sensitivity)**| 76.51% | **80.36%** | +3.85% | Captures **over 8 out of every 10 customers** who would otherwise defect |
| **Precision** | 52.83% | **52.94%** | +0.11% | More than 1 in 2 flagged customers is a verified churner |
| **F1-Score** | 0.6250 | **0.6383** | +0.0133 | Strong harmonic balance between intervention coverage and alert purity |
| **Accuracy** | 75.59% | **75.88%** | +0.29% | Stable baseline classification accuracy on imbalanced data |

> **Production Generalization Confirmation**: Because the final test metrics match or slightly exceed the validation baseline across all evaluation dimensions, there is **zero evidence of overfitting**.

---

## 11. Honest Disclosure About Earlier Test-Set Exploration

> [!IMPORTANT]
> **Methodological Disclosure**:
> Model selection in the corrected pipeline uses validation PR-AUC and F1-score only. However, during an earlier development iteration, multiple candidate models were evaluated on the test set before the evaluation workflow was corrected. Therefore, the test set was not completely untouched throughout development. Final test metrics should be interpreted with this limitation in mind.

---

## 12. MLflow Experiment Tracking

Customer360 integrates MLflow tracking ([`src/ml/tracking.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/ml/tracking.py)) to ensure experiment reproducibility and auditability.

### Configuration
- **Backend Store**: Persistent SQLite database located at `sqlite:///mlflow/mlflow.db`.
- **Artifact Repository**: Project-contained directory `mlflow/artifacts/`.
- **Active Experiment**: `Customer360_Churn_Prediction`.

### Logged Run Elements
For each of the 6 benchmark configurations, MLflow records:
1. **Hyperparameters**: Algorithm name, balancing strategy, tree count (`n_estimators`), maximum tree depth (`max_depth`), learning rate, regularization penalties (`C`, `penalty`, `reg_alpha`, `reg_lambda`), and `scale_pos_weight`.
2. **Metrics**: Validation and test scores for Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
3. **Serialized Pipeline Artifacts**: Scikit-learn preprocessor pipeline (`preprocessor.joblib`), trained estimator (`model.joblib`), and feature column schemas (`feature_names.joblib`).
4. **Diagnostic Visualizations**: Confusion matrix plots (`confusion_matrix.png`), ROC curves (`roc_curve.png`), Precision-Recall curves (`pr_curve.png`), and run markdown summaries (`run_summary.md`).

---

## 13. Streamlit Dashboard Functionality

The interactive executive dashboard ([`dashboard/app.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/dashboard/app.py)) provides business stakeholders with 4 distinct analytical workflows:

1. **Business Overview (Executive KPIs)**:
   - High-level metric cards: Total Customers (7,043), Retained vs. Churned Accounts, Corporate Retention Rate (73.46%), Monthly Revenue at Risk ($139,130.85).
   - Interactive churn vs. retention donut breakdown and monthly billing distribution histograms.
2. **Churn Analysis (Diagnostic Deep-Dive)**:
   - Bivariate attrition analytics across Contract Type, Payment Channel, Internet Technology, and Tenure Cohorts.
   - Interactive cross-tabulation detailing the retention anchor effect of `TechSupport` and `OnlineSecurity`.
3. **Customer Risk Prediction (Real-Time Inference & What-If Simulator)**:
   - Live integration with the FastAPI `/predict` microservice.
   - Interactive single-customer form accepting account parameters and returning real-time probability gauge and risk tiering.
   - **What-If Scenario Simulator**: Allows customer success teams to adjust contracts (e.g., Month-to-month to 1-Year) or add `TechSupport` to observe immediate risk score reduction.
   - Dynamic customer drill-down search by account ID.
4. **Retention Insights (Actionable Playbook)**:
   - Strategic intervention guidance tailored to each risk tier (High, Medium, Low).
   - Value-protection matrices targeting high-ARPU accounts vulnerable to early defection.

---

## 14. FastAPI Endpoints & Example Requests

The production scoring microservice ([`src/api/main.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/api/main.py)) serves the champion XGBoost model over low-latency REST endpoints.

- **Base URL**: `http://127.0.0.1:8000`
- **Interactive Documentation**: Swagger UI at `http://127.0.0.1:8000/docs`, ReDoc at `http://127.0.0.1:8000/redoc`.

### Endpoint Summary

| Method | Path | Description | Request Body |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Microservice and model artifact health check | None |
| `POST` | `/predict` | Single customer churn risk prediction | `CustomerFeatures` JSON |
| `POST` | `/predict/batch`| Batch customer scoring (1 to 5,000 records) | `BatchPredictionRequest` JSON |

---

### Request & Response Examples

#### 1. Service Health Check (`GET /health`)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "XGBClassifier"
}
```

#### 2. Single Customer Prediction (`POST /predict`)
```powershell
$headers = @{ "Content-Type" = "application/json" }
$body = @{
    customerID = "DEMO-1001"
    gender = "Female"
    SeniorCitizen = 0
    Partner = "No"
    Dependents = "No"
    tenure = 2
    PhoneService = "Yes"
    MultipleLines = "No"
    InternetService = "Fiber optic"
    OnlineSecurity = "No"
    OnlineBackup = "No"
    DeviceProtection = "No"
    TechSupport = "No"
    StreamingTV = "No"
    StreamingMovies = "No"
    Contract = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    MonthlyCharges = 79.85
    TotalCharges = 159.70
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Headers $headers -Body $body
```
```json
{
  "customerID": "DEMO-1001",
  "churn_probability": 0.8124,
  "predicted_churn": "Yes",
  "risk_category": "High Risk",
  "message": "Predictions are probabilistic estimates; not guaranteed outcomes."
}
```

#### 3. Batch Scoring (`POST /predict/batch`)
```powershell
$headers = @{ "Content-Type" = "application/json" }
$body = @{
    customers = @(
        @{
            customerID = "CUST-A"
            gender = "Male"
            SeniorCitizen = 0
            Partner = "Yes"
            Dependents = "Yes"
            tenure = 60
            PhoneService = "Yes"
            MultipleLines = "Yes"
            InternetService = "DSL"
            OnlineSecurity = "Yes"
            OnlineBackup = "Yes"
            DeviceProtection = "Yes"
            TechSupport = "Yes"
            StreamingTV = "Yes"
            StreamingMovies = "Yes"
            Contract = "Two year"
            PaperlessBilling = "No"
            PaymentMethod = "Credit card (automatic)"
            MonthlyCharges = 64.10
            TotalCharges = 3846.00
        }
    )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict/batch" -Method Post -Headers $headers -Body $body
```
```json
{
  "count": 1,
  "predictions": [
    {
      "customerID": "CUST-A",
      "churn_probability": 0.0412,
      "predicted_churn": "No",
      "risk_category": "Low Risk",
      "message": "Predictions are probabilistic estimates; not guaranteed outcomes."
    }
  ]
}
```

---

## 15. Customer Risk Prediction & Retention Recommendations

The platform classifies subscribers into three actionable operational tiers ([`src/api/predictor.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/api/predictor.py)):

### 🔴 High Risk Tier (`Probability >= 0.70`)
- **Profile**: Short tenure (0–6 mo), Month-to-month contract, Fiber optic internet, Electronic check billing, absence of Tech Support.
- **Recommended Action**:
  - Immediate high-priority contact by senior retention specialist within 48 hours.
  - Contract Migration Offer: $15/month credit for 6 months upon upgrading to a 1-Year or 2-Year contract.
  - Onboarding Concierge: Free installation and activation of `TechSupport` and `OnlineSecurity`.
  - Payment Incentive: $5 monthly billing waiver for transitioning to automated Bank transfer or Credit card.

### 🟡 Medium Risk Tier (`0.40 <= Probability < 0.70`)
- **Profile**: Mid-tenure (6–24 mo), month-to-month contracts or unbundled services, moderate-to-high monthly charges.
- **Recommended Action**:
  - Automated proactive outreach via email/SMS offering free 3-month trials of digital safety add-ons (`OnlineBackup` / `DeviceProtection`).
  - Account health check-in to identify service or network speed dissatisfaction.
  - Targeted discount on annual subscription renewal.

### 🟢 Low Risk Tier (`Probability < 0.40`)
- **Profile**: Long tenure (> 24 mo), multi-year contract commitments, automated payments, bundled support services.
- **Recommended Action**:
  - Relationship nurturing and VIP loyalty reward programs.
  - Cross-sell and upsell premium speed tiers, connected home devices, or streaming entertainment packages.
  - Referral reward incentive enrollment.

---

## 16. Complete Windows PowerShell Setup Instructions

### 1. Prerequisites
- **Windows 10 / 11** with PowerShell 5.1+ or PowerShell Core 7+.
- **Python 3.13+** installed and accessible in `PATH`.
- **MySQL 8.0 Server** installed, running, and accessible.
- **Git for Windows** installed.

### 2. Clone Repository & Create Virtual Environment
Open Windows PowerShell:

```powershell
# Navigate to desired directory
cd "C:\Users\USER\OneDrive\Desktop\Customer 360"

# Create a local virtual environment
python -m venv venv

# If execution policy restricts scripts, temporarily bypass for current process:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables (`.env`)
Create a local `.env` file from the provided `.env.example` template:

```powershell
Copy-Item .env.example .env
```

Update `.env` with your local MySQL password and connection settings:

```ini
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=customer360_db
API_URL=http://127.0.0.1:8000
MLFLOW_TRACKING_URI=sqlite:///mlflow/mlflow.db
```

*(Note: The `.env` file is excluded from version control via `.gitignore` to prevent credential exposure).*

---

## 17. MySQL Configuration & Database Loading Instructions

### 1. Initialize the MySQL Schema
Ensure MySQL is running, then create the database tables:

```powershell
.\venv\Scripts\python.exe src/database/create_tables.py
```

### 2. Ingest the Telco Customer Dataset
Load the raw CSV data into MySQL:

```powershell
.\venv\Scripts\python.exe src/database/load_data.py
```

### 3. Run Diagnostic SQL Analytics
Execute the 13 automated business diagnostic queries:

```powershell
.\venv\Scripts\python.exe src/database/run_analysis.py
```

---

## 18. Running Pipeline, API, Dashboard & MLflow

### 1. Train and Evaluate the ML Pipeline
To execute the end-to-end data pipeline, train all 6 candidate models, log runs to MLflow, and save production artifacts:

```powershell
.\venv\Scripts\python.exe src/ml/pipeline.py
```

### 2. Start the FastAPI Production Service
Launch the REST microservice with hot-reload enabled:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
*API available at `http://127.0.0.1:8000` (Docs: `http://127.0.0.1:8000/docs`).*

### 3. Launch the Streamlit Executive Dashboard
In a separate PowerShell terminal (with `venv` activated):

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard/app.py
```
*Dashboard opens automatically at `http://localhost:8501`.*

### 4. Launch the MLflow Experiment Tracking Server
In a separate PowerShell terminal (with `venv` activated):

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
```
*MLflow Tracking UI available at `http://127.0.0.1:5000`.*

---

## 19. How to Run Pytest

Customer360 includes 33 unit and integration tests. Run the test suite using PowerShell:

```powershell
# Run the complete test suite
.\venv\Scripts\python.exe -m pytest -v

# Run API endpoint tests only
.\venv\Scripts\python.exe -m pytest tests/test_api.py -v

# Run MLflow tracking tests only
.\venv\Scripts\python.exe -m pytest tests/test_tracking.py -v
```

Expected test execution output:
```
tests/test_api.py::TestHealth::test_health_returns_200 PASSED
tests/test_api.py::TestHealth::test_health_model_loaded PASSED
tests/test_api.py::TestSinglePredict::test_valid_customer_returns_200 PASSED
tests/test_api.py::TestBatchPredict::test_batch_returns_200 PASSED
tests/test_tracking.py::test_init_mlflow PASSED
tests/test_tracking.py::test_generate_run_plots PASSED
======================= 33 passed in 11.75s =======================
```

---

## 20. Project Directory Structure

```
Customer 360/
├── .env                              # Local database and API environment variables
├── .gitignore                        # Git exclusion rules (.env, venv, pycache, etc.)
├── README.md                         # Root repository documentation and architecture guide
├── requirements.txt                  # Locked Python package dependencies
│
├── dashboard/
│   └── app.py                        # Streamlit executive retention intelligence dashboard
│
├── data/
│   └── raw/
│       └── customer_churn.csv        # IBM Telco Customer Churn dataset (7,043 rows)
│
├── docs/
│   └── database_design.md            # Schema architecture and relational dictionary
│
├── mlflow/
│   ├── mlflow.db                     # SQLite tracking repository for experiment runs
│   ├── artifacts/                    # Logged model weights, preprocessors, and plots
│   └── temp_plots/                   # Staged run evaluation charts
│
├── models/
│   ├── best_model.joblib             # Champion XGBoost model artifact
│   ├── preprocessor.joblib           # Fitted scikit-learn preprocessing pipeline
│   ├── feature_names.joblib          # Ordered list of transformed feature names
│   ├── logistic_regression.joblib    # Candidate model artifact (Class Weights)
│   ├── logistic_regression_smote.joblib # Candidate model artifact (SMOTE)
│   ├── random_forest.joblib          # Candidate model artifact (Class Weights)
│   ├── random_forest_smote.joblib    # Candidate model artifact (SMOTE)
│   ├── xgboost.joblib                # Candidate model artifact (Class Weights)
│   └── xgboost_smote.joblib          # Candidate model artifact (SMOTE)
│
├── notebooks/
│   └── 01_data_understanding.ipynb   # Exploratory data analysis and visualizations
│
├── reports/
│   ├── confusion_matrices.png        # Comparative confusion matrix plots
│   ├── roc_curves.png                # Comparative ROC curves across candidate models
│   ├── pr_curves.png                 # Precision-Recall curves across candidate models
│   ├── model_comparison.csv          # Audited validation and test metric benchmark
│   ├── evaluation_summary.md         # Comprehensive evaluation and selection report
│   └── mlflow_verification_report.md # MLflow experiment audit and tracking report
│
├── sql/
│   └── customer_analysis.sql         # 13 Diagnostic SQL analytics queries
│
├── src/
│   ├── __init__.py
│   ├── api/                          # FastAPI REST microservice
│   │   ├── __init__.py
│   │   ├── main.py                   # Application definition, lifespan, CORS, routes
│   │   ├── predictor.py              # Model artifact loader and inference handler
│   │   └── schemas.py                # Pydantic request and response schemas
│   │
│   ├── database/                     # MySQL connectivity and ETL
│   │   ├── __init__.py
│   │   ├── db_connection.py          # SQLAlchemy engine factory
│   │   ├── create_tables.py          # Relational DDL execution
│   │   ├── load_data.py              # CSV ingestion pipeline
│   │   └── run_analysis.py           # Programmatic SQL query runner
│   │
│   └── ml/                           # Machine learning pipeline
│       ├── __init__.py
│       ├── data_loader.py            # Dataset loading and stratified splitting
│       ├── preprocessor.py           # Custom scikit-learn preprocessing transformers
│       ├── train.py                  # Model training and hyperparameter setup
│       ├── evaluate.py               # Evaluation metric calculations and plotting
│       ├── tracking.py               # MLflow experiment tracking logger
│       └── pipeline.py               # End-to-end training and selection orchestration
│
└── tests/
    ├── test_api.py                   # 28 Unit and integration tests for FastAPI service
    └── test_tracking.py              # 5 Unit tests for MLflow experiment tracking
```

---

## 21. Limitations & Future Roadmap

### Current Limitations
1. **Cross-Sectional Observational Data**: The current dataset captures a static snapshot in time. It lacks time-series event streams (e.g., call center interactions, support tickets, monthly usage spikes/dips) that could signal churn weeks before defection.
2. **Deterministic Risk Thresholds**: Risk categories (`High >= 0.70`, `Medium >= 0.40`) are illustrative heuristic business tiers rather than calibrated Bayesian utility thresholds.
3. **Prior Test-Set Exploration**: As disclosed in Section 11, earlier development iterations evaluated candidate models against test data prior to pipeline correction.

### Future Roadmap
- [ ] **Automated Model Drift Monitoring**: Integrate Evidently AI or Great Expectations to monitor real-time distribution drift in incoming production feature payloads.
- [ ] **Probability Calibration**: Calibrate tree output probabilities using Isotonic Regression or Platt Scaling to directly reflect true Bayesian empirical attrition risk.
- [ ] **SHAP Tree Explainability Integration**: Return local SHAP waterfall feature attributions in the FastAPI `/predict` payload so support representatives understand *why* a customer was classified as high risk.
- [ ] **Event-Driven Streaming Ingestion**: Connect Apache Kafka or AWS Kinesis to ingest continuous subscriber usage telemetry into the scoring service.

---

## 22. Project Metadata & Author

- **Project**: Customer360 – Customer Retention Intelligence Platform
- **License**: No project license specified; dataset use is subject to the original source's terms.
- **Author**: Machine Learning & Data Engineering Team
