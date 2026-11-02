# Customer360: Customer Churn Prediction & Retention Intelligence Platform

[![Python Version](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.17.2-0194E2.svg?logo=mlflow)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Champion-red.svg)](https://xgboost.readthedocs.io/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-4479A1.svg?logo=mysql)](https://www.mysql.com/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/pytest-33%20passed-brightgreen.svg)](https://docs.pytest.org/)

---

## 1. Project Overview & Motivation

Customer churn is one of the most critical operational problems for subscription-based businesses like telecommunications. Acquiring a new customer often costs anywhere from 5 to 25 times more than retaining an existing subscriber. When customers churn unpredictably, service providers suffer immediate loss in monthly recurring revenue (MRR) and customer lifetime value (LTV).

I developed **Customer360** as an end-to-end data science and machine learning project to systematically analyze, predict, and address customer attrition. The project covers the full lifecycle:

1. **Relational Data Modeling & SQL Analytics**: Storing customer records in MySQL and writing diagnostic SQL queries to identify attrition drivers across contracts, payment channels, and tenure cohorts.
2. **Machine Learning Preprocessing & Training**: Building a modular scikit-learn preprocessing pipeline (fitted strictly on training data) and benchmarking 6 candidate model setups across Logistic Regression, Random Forest, and XGBoost.
3. **Experiment Tracking with MLflow**: Tracking runs, hyperparameters, metrics, confusion matrices, and model artifacts in a local SQLite-backed MLflow repository.
4. **FastAPI Prediction Service**: Serving the champion XGBoost model via REST endpoints for real-time single and batch customer predictions with Pydantic validation.
5. **Interactive Streamlit Dashboard**: Providing a practical front-end for exploring churn statistics, testing real-time predictions, and simulating what-if retention strategies.
6. **Automated Testing**: Validating API responses, probability ranges, batch chunking, and MLflow logging with 33 unit and integration tests in pytest.

---

## 2. Key Project Highlights

- **Relational SQL Diagnostic Suite**: 13 structured SQL queries in MySQL (`sql/customer_analysis.sql`) analyzing churn distribution, tenure cohorts, payment method exposure, and revenue at risk.
- **Leak-Free Preprocessing Design**: Custom scikit-learn transformers (`preprocessor.py`) for median imputation, standard scaling, and one-hot encoding, fitted solely on training splits to prevent data leakage.
- **Systematic Model Benchmarking**: Evaluated 6 model combinations across algorithmic class weighting and SMOTE oversampling on an independent validation set.
- **Champion XGBoost Model**: Selected on validation PR-AUC (0.6435) and F1-score (0.6250), capturing over 80% of churners on the final test split with an ROC-AUC of 0.8525.
- **Persistent MLflow Tracking**: Local SQLite tracking store logging parameters, metrics, PR/ROC curves, feature schemas, and serialized models.
- **RESTful Prediction API**: Asynchronous FastAPI service supporting single-customer inference and batch processing of up to 5,000 records per request.
- **Interactive Decision Dashboard**: Multi-view Streamlit application featuring KPI cards, bivariate diagnostics, customer profile search, and an interactive what-if retention simulator.
- **Thorough Test Coverage**: 33 automated pytest tests confirming endpoint stability, payload boundaries, batch chunking, and tracking routines.

---

## 3. Technology Stack & Design Decisions

| Category | Technology | Version | Role in Project | Why I Chose It |
| :--- | :--- | :---: | :--- | :--- |
| **Language** | Python | `3.13+` | Primary language | Core language for modern data science, ML libraries, and web backends. |
| **Database** | MySQL | `8.0+` | Relational data store | Simulates realistic relational data storage rather than relying solely on flat CSV files. |
| **DB Connector** | SQLAlchemy / PyMySQL | `2.1.6` / `1.2.3` | DB connectivity | Reliable connection pooling and programmatic query execution with pandas integration. |
| **Data Processing** | pandas, numpy, scipy | `3.1.0` / `2.5.4` | Data manipulation | Efficient vectorized operations, feature cleaning, and statistical transformations. |
| **Machine Learning** | scikit-learn | `1.9.1` | Preprocessing & baselines | Standardized pipeline abstractions, transformer architecture, and evaluation metrics. |
| **Gradient Boosting** | XGBoost | `3.4.1` | Champion classifier | High performance on structured tabular data, fast tree construction, and native `scale_pos_weight`. |
| **Imbalance Handling** | imbalanced-learn | `0.14.2` | Synthetic oversampling | Benchmarked SMOTE against algorithmic cost-sensitive weighting. |
| **Serialization** | joblib | `1.6.0` | Artifact persistence | Efficient serialization of scikit-learn pipelines and trained tree ensembles. |
| **MLOps / Tracking** | MLflow | `3.17.2` | Experiment tracking | Structured logging of hyperparameters, curves, and artifacts without manual spreadsheet logging. |
| **API Microservice** | FastAPI, Uvicorn | `0.162.0` / `0.54.0` | Scoring backend | High-performance ASGI framework with automatic OpenAPI docs and strict Pydantic schemas. |
| **Validation** | Pydantic | `2.14.2` | Data validation | Type checking and field-level validation for all incoming inference requests. |
| **Dashboard** | Streamlit, Plotly | `1.66.1` / `7.1.0` | Interactive UI | Rapid development of an intuitive analytics interface without building a separate React frontend. |
| **Testing** | pytest, httpx | `9.2.0` / `0.28.1` | Automated testing | Robust test automation across API schemas, probability ranges, and logging logic. |

---

## 4. End-to-End System Workflow

```
                                      ┌─────────────────────────────────────────┐
                                      │   IBM Telco Churn Dataset (7,043 rows)  │
                                      │   (Downloaded locally by user)          │
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

## 5. Dataset Description & Availability

This project is built around the **IBM Telco Customer Churn** dataset, a well-known benchmark in customer analytics literature. It contains **7,043 customer accounts** observed across **21 attributes**:

- **Target Variable**: `Churn` (`Yes` / `No`) indicating whether the customer left within the last month.
- **Class Balance**: 5,174 active customers (**73.46%**) vs. 1,869 churned customers (**26.54%**), representing a ~1:3 class imbalance.
- **Feature Categories**:
  - **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
  - **Account & Billing**: `tenure` (0–72 months), `Contract` (Month-to-month, One year, Two year), `PaperlessBilling` (Yes/No), `PaymentMethod` (Electronic check, Mailed check, Bank transfer, Credit card), `MonthlyCharges` (USD), `TotalCharges` (USD).
  - **Subscribed Services**: `PhoneService`, `MultipleLines`, `InternetService` (DSL, Fiber optic, No), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.

> [!IMPORTANT]
> **Dataset Availability & License Terms Notice**:
> The raw dataset is **not committed to this GitHub repository** because redistribution rights have not been verified. Users must obtain the IBM Telco Customer Churn dataset separately from its original source (such as IBM sample data community repositories) and review all applicable terms before use. Neither redistribution nor commercial use is authorized by this repository.
> 
> To run the platform locally, download the original CSV and place it at:
> `data/raw/customer_churn.csv`

---

## 6. Exploratory Data Analysis & Key Insights

During exploratory data analysis and SQL querying, several distinct customer attrition patterns emerged:

1. **Contract Duration is the Strongest Churn Predictor**:
   - Month-to-Month contract customers have a **42.71% churn rate**, accounting for nearly 88.5% of all churned customers.
   - One-Year contracts drop churn to **11.27%**.
   - Two-Year contracts drop churn to just **2.83%**.
2. **Attrition is Heavily Front-Loaded**:
   - Customers in their first two months (0–1 month cohort) experience an attrition rate of **51.9%**.
   - Churn rates decrease substantially after 12 months and drop under 10% after 36 months, showing that early customer onboarding is critical.
3. **Fiber Optic Has High Churn and High Billing**:
   - Fiber optic internet subscribers churn at **41.89%** (compared to 18.96% for DSL and 7.40% for no internet). This group also carries higher monthly charges ($70–$110/month), making service stability and technical support crucial for this high-value segment.
4. **Payment Channel Disparity**:
   - Customers paying via Electronic check churn at **45.29%**, compared to **15.24%** for Credit card (automatic) and **16.71%** for Bank transfer (automatic). Manual payment creates recurring points of friction.
5. **Support and Security Add-ons Act as Retention Anchors**:
   - Customers without `TechSupport` churn at **41.64%**, while those with `TechSupport` churn at **15.17%**.
   - Customers without `OnlineSecurity` churn at **41.77%**, while those with `OnlineSecurity` churn at **14.61%**.
6. **Financial Exposure**:
   - Churned customers have a higher average monthly bill ($74.44) than retained customers ($61.27), indicating that attrition disproportionately hits high-ARPU subscribers.

---

## 7. MySQL Database & SQL Diagnostic Analytics

I set up a dedicated MySQL database (`customer360_db`) to store structured customer records and run SQL queries to uncover business insights before building the machine learning pipeline.

### Schema Architecture
The `customers` table stores typed records with `customerID VARCHAR(50) PRIMARY KEY`:
- Numeric features: `tenure INT`, `MonthlyCharges DECIMAL(10,2)`, `TotalCharges DECIMAL(10,2)`.
- Categorical features: `Contract`, `PaymentMethod`, `InternetService`, `Churn`, etc.

### Database Utilities
- [`src/database/create_tables.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/database/create_tables.py): Creates the schema and sets up indexes on frequently filtered columns.
- [`src/database/load_data.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/database/load_data.py): Reads the local CSV and batch-loads records into MySQL.
- [`src/database/run_analysis.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/database/run_analysis.py): Programmatically executes the SQL queries and prints formatted tabular output.

### 13 Diagnostic SQL Queries ([`sql/customer_analysis.sql`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/sql/customer_analysis.sql))
1. **Total Baseline**: Overall account count.
2. **Attrition Breakdown**: Active vs. churned subscriber counts.
3. **Organizational Churn Rate**: Overall churn percentage (26.54%) and retention percentage (73.46%).
4. **Contract Type Attrition**: Churn breakdown across Month-to-Month, 1-Year, and 2-Year contracts.
5. **Payment Method Exposure**: Risk across payment gateways (Electronic check vs. automatic payments).
6. **Internet Service Comparison**: Risk comparison across Fiber optic, DSL, and no-internet tiers.
7. **Tenure Cohort Analysis**: Churn grouped by tenure brackets (0–12, 13–24, 25–48, 49–72 months).
8. **Revenue at Risk**: Total monthly revenue lost to churn ($139,130.85/month).
9. **Average Monthly Revenue**: ARPU comparison between retained and churned cohorts.
10. **High-Value Flight Risk**: Identifies accounts with `MonthlyCharges > $70` and `tenure <= 6`.
11. **Technical Support Impact**: Retention delta attributable to `TechSupport`.
12. **Online Security Impact**: Retention delta attributable to `OnlineSecurity`.
13. **Compound Risk Segmentation**: High-risk segment (Month-to-month + Fiber optic + Electronic check).

---

## 8. Machine Learning Pipeline & Methodology

### 1. Three-Way Stratified Partition
I split the 7,043 customer records into three subsets using stratified sampling on `Churn`:
- **Training Set (70%, 4,929 rows)**: Used solely to fit preprocessing transformations (`StandardScaler`, `OneHotEncoder`, median imputer) and train model estimators.
- **Validation Set (15%, 1,057 rows)**: Used to compare candidate models, tune balancing strategies, and select the champion model.
- **Held-Out Test Set (15%, 1,057 rows)**: Serves as the benchmark split for evaluating the selected Champion model. As transparently disclosed in Section 11, earlier development iterations involved exploratory evaluation of multiple candidate models on this test set before the evaluation workflow was corrected; therefore, final test metrics must be interpreted with this limitation in mind rather than as an entirely untouched evaluation.

### 2. Preprocessing Pipeline ([`src/ml/preprocessor.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/ml/preprocessor.py))
- **`DataCleaner` Transformer**: Converts `TotalCharges` to float, handles blank string values, and imputes missing numeric records using the median calculated **strictly from the training partition**.
- **Numerical Scaling**: `StandardScaler` applied to `tenure`, `MonthlyCharges`, and `TotalCharges`.
- **Categorical Encoding**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to demographic and service features.
- **Feature Schema Protection**: `customerID` and historical `Churn` labels are permanently excluded from the feature matrix.

### 3. Class Imbalance Strategies
Because the minority churn class accounts for ~26.5% of records, standard models can over-predict the majority class. I evaluated two balancing approaches across all candidate models:
1. **Cost-Sensitive Class Weighting**: Using `class_weight='balanced'` in Logistic Regression and Random Forest, and `scale_pos_weight = (neg_count / pos_count)` in XGBoost (~2.77).
2. **Synthetic Minority Over-sampling (SMOTE)**: Generating synthetic minority instances on the training partition using `imblearn.over_sampling.SMOTE(random_state=42)`.

---

## 9. Model Comparison & Validation Results

I benchmarked 6 candidate model configurations on the **Validation Set (1,057 records; churn prevalence = 26.58%)**.

In customer churn prediction, **PR-AUC (Precision-Recall Area Under Curve)** and **F1-Score** are more informative than raw accuracy. Because false negatives (missing a true churner) lead to lost revenue and false positives (offering retention perks unnecessarily) lead to budget waste, I prioritized models with strong PR-AUC and high recall:

| Rank | Model Architecture | Balancing Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Selection Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`XGBoost`** | **Class_Weights** | **75.59%** | **52.83%** | **76.51%** | **0.6250** | **0.8383** | **0.6435** | 🏆 **Selected Champion** |
| 2 | `XGBoost_SMOTE` | SMOTE_Oversampling | 79.38% | 60.26% | 65.84% | 0.6293 | 0.8392 | 0.6373 | Candidate |
| 3 | `Random_Forest` | Class_Weights | 75.88% | 53.25% | 75.80% | 0.6256 | 0.8351 | 0.6360 | Candidate |
| 4 | `Logistic_Regression_SMOTE` | SMOTE_Oversampling | 75.21% | 52.32% | 76.16% | 0.6203 | 0.8290 | 0.6280 | Candidate |
| 5 | `Logistic_Regression` | Class_Weights | 74.55% | 51.44% | 76.51% | 0.6152 | 0.8286 | 0.6246 | Candidate |
| 6 | `Random_Forest_SMOTE` | SMOTE_Oversampling | 78.43% | 58.01% | 68.33% | 0.6275 | 0.8322 | 0.6188 | Candidate |

### Why XGBoost (Class Weights) was chosen:
- Achieved the highest validation **PR-AUC (0.6435)** and captured **76.51% of all at-risk customers** while maintaining balanced precision.
- Algorithmic weighting (`scale_pos_weight`) proved more effective than SMOTE for gradient-boosted trees, avoiding the synthetic density artifacts that degraded XGBoost's recall under SMOTE (76.51% vs. 65.84%).

---

## 10. Champion Model: Test Evaluation & Generalization

After selecting the champion model on the validation split, I evaluated `XGBoost` (Class Weights) on the **Held-Out Test Set (1,057 records; churn prevalence = 26.49%)**:

| Evaluation Metric | Validation Score | Final Test Score | Delta | Practical Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **PR-AUC** | 0.6435 | **0.6651** | +0.0216 | Strong precision-recall balance across different classification thresholds. |
| **ROC-AUC** | 0.8383 | **0.8525** | +0.0142 | Good ability to rank churners above non-churners. |
| **Recall (Sensitivity)**| 76.51% | **80.36%** | +3.85% | Identifies approximately 8 out of every 10 customers likely to churn. |
| **Precision** | 52.83% | **52.94%** | +0.11% | Roughly 1 in 2 flagged customers is a verified churner. |
| **F1-Score** | 0.6250 | **0.6383** | +0.0133 | Harmonic balance between coverage and alert purity. |
| **Accuracy** | 75.59% | **75.88%** | +0.29% | Overall classification accuracy on imbalanced data. |

> [!NOTE]
> **Generalization Assessment**: While the final test metrics closely mirror and slightly exceed the validation baseline across all dimensions (e.g., Test PR-AUC 0.6651 vs. Validation PR-AUC 0.6435; Test Recall 80.36% vs. Validation Recall 76.51%), these results should not be treated as fully independent confirmation of generalization due to the exploratory evaluation conducted on the test set during earlier development iterations (see Section 11).

---

## 11. Honest Disclosure About Earlier Test-Set Exploration

> [!IMPORTANT]
> **Methodological Disclosure**:
> Model selection in the corrected pipeline uses validation PR-AUC and F1-score only. However, during an earlier development iteration, multiple candidate models were evaluated on the test set before the evaluation workflow was corrected. Therefore, the test set was not completely untouched throughout development. Final test metrics should be interpreted with this limitation in mind.

---

## 12. MLflow Experiment Tracking

To maintain reproducibility and avoid losing track of past training runs, I integrated MLflow tracking ([`src/ml/tracking.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/ml/tracking.py)):

- **Tracking Store**: Local SQLite database at `sqlite:///mlflow/mlflow.db`.
- **Artifacts Folder**: Local directory at `mlflow/artifacts/`.
- **Experiment Name**: `Customer360_Churn_Prediction`.

### Logged Data for Each Run
1. **Parameters**: Model name, balancing method, tree count (`n_estimators`), tree depth (`max_depth`), learning rate, regularization penalties (`reg_alpha`, `reg_lambda`, `C`), and `scale_pos_weight`.
2. **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
3. **Artifacts**: Confusion matrix plot (`confusion_matrix.png`), ROC curve (`roc_curve.png`), PR curve (`pr_curve.png`), feature schema (`feature_names.joblib`), preprocessor pipeline (`preprocessor.joblib`), and trained estimator (`model.joblib`).

---

## 13. Interactive Streamlit Dashboard

I built an interactive dashboard ([`dashboard/app.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/dashboard/app.py)) in Streamlit with four primary sections:

1. **Business Overview**:
   - Displays high-level KPIs: Total Customers (7,043), Retained vs. Churned, Overall Retention Rate (73.46%), and Monthly Revenue at Risk ($139,130.85).
   - Visualizes overall churn distribution and monthly billing histograms.
2. **Churn Analysis**:
   - Bivariate charts showing churn by Contract Type, Payment Channel, Internet Technology, and Tenure Cohorts.
   - Highlights the protective retention impact of `TechSupport` and `OnlineSecurity`.
3. **Customer Risk Prediction**:
   - Connects live to the FastAPI backend (`/predict`).
   - Allows users to enter customer attributes through a form and instantly view the predicted churn probability and risk tier.
   - Includes a **What-If Scenario Simulator** to see how changing contract terms or adding services changes a customer's risk score.
4. **Retention Insights**:
   - Outlines recommended intervention playbooks segmented by High, Medium, and Low risk tiers.

---

## 14. FastAPI Prediction Service & Examples

The inference service ([`src/api/main.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/api/main.py)) loads the champion XGBoost model and preprocessor at startup to serve predictions over REST endpoints:

- **Base URL**: `http://127.0.0.1:8000`
- **Documentation**: Interactive Swagger UI at `http://127.0.0.1:8000/docs`, ReDoc at `http://127.0.0.1:8000/redoc`.

### Endpoints

| Method | Path | Description | Request Body |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Verifies service health and loaded model status | None |
| `POST` | `/predict` | Predicts churn risk for a single customer | `CustomerFeatures` JSON |
| `POST` | `/predict/batch`| Scores a batch of customer records (1 to 5,000) | `BatchPredictionRequest` JSON |

---

### Example PowerShell Requests

#### 1. Health Check (`GET /health`)
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

## 15. Risk Categorization & Retention Playbooks

The scoring service assigns each customer to an operational risk tier based on their predicted churn probability ([`src/api/predictor.py`](file:///c:/Users/USER/OneDrive/Desktop/Customer%20360/src/api/predictor.py)):

### 🔴 High Risk (`Probability >= 0.70`)
- **Profile**: Short tenure (0–6 mo), Month-to-month contract, Fiber optic internet, Electronic check payment, lacking Tech Support.
- **Recommended Action**:
  - Outreach within 48 hours by a retention specialist.
  - Contract migration incentive: Monthly discount in exchange for a 1-Year or 2-Year commitment.
  - Onboarding assistance and complimentary temporary `TechSupport` / `OnlineSecurity`.
  - Autopay discount incentive to switch to Bank transfer or Credit card.

### 🟡 Medium Risk (`0.40 <= Probability < 0.70`)
- **Profile**: Mid-tenure (6–24 mo), month-to-month billing, moderate monthly charges.
- **Recommended Action**:
  - Proactive email/SMS offering trial add-ons (`OnlineBackup` or `DeviceProtection`).
  - Account health check-in to identify service dissatisfaction early.
  - Discount offers on annual renewal.

### 🟢 Low Risk (`Probability < 0.40`)
- **Profile**: Long tenure (> 24 mo), multi-year contract commitments, automated payments, bundled services.
- **Recommended Action**:
  - General relationship nurturing and loyalty perks.
  - Cross-sell opportunities (speed tier upgrades, streaming add-ons).

---

## 16. Complete Windows PowerShell Setup Instructions

### 1. Prerequisites
- **Windows 10 / 11** with PowerShell 5.1+ or PowerShell Core 7+.
- **Python 3.13+** installed and added to `PATH`.
- **MySQL 8.0 Server** installed, running, and accessible.
- **Git for Windows** installed.

### 2. Clone Repository & Create Virtual Environment
Open Windows PowerShell:

```powershell
# Clone the repository from GitHub
git clone https://github.com/Tanay-chaturvedi/Customer360-Churn-Prediction.git

# Navigate into the cloned project directory
cd Customer360-Churn-Prediction

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
Ensure you have obtained the raw dataset and placed it at `data/raw/customer_churn.csv` (see Section 5). Then load the CSV data into MySQL:

```powershell
.\venv\Scripts\python.exe src/database/load_data.py
```

### 3. Run Diagnostic SQL Analytics
Execute the 13 automated business diagnostic queries:

```powershell
.\venv\Scripts\python.exe src/database/run_analysis.py
```

---

## 18. Running the Pipeline, API, Dashboard & MLflow

### 1. Train and Evaluate the ML Pipeline
To execute the end-to-end data pipeline, train all 6 candidate models, log runs to MLflow, and save production artifacts:

```powershell
.\venv\Scripts\python.exe src/ml/pipeline.py
```

### 2. Start the FastAPI Prediction Service
Launch the REST API server:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
*API is accessible at `http://127.0.0.1:8000` (Interactive Docs: `http://127.0.0.1:8000/docs`).*

### 3. Launch the Streamlit Dashboard
In a separate PowerShell terminal (with `venv` activated):

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard/app.py
```
*Dashboard opens automatically in your browser at `http://localhost:8501`.*

### 4. Launch the MLflow Experiment Tracking Server
In a separate PowerShell terminal (with `venv` activated):

```powershell
.\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
```
*MLflow Tracking UI is accessible at `http://127.0.0.1:5000`.*

---

## 19. How to Run Pytest

Customer360 includes 33 unit and integration tests covering the API schemas, probability ranges, batch scoring, and tracking routines.

Run the test suite using PowerShell:

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
======================= 33 passed in 13.94s =======================
```

---

## 20. Project Directory Structure

```
Customer360-Churn-Prediction/
├── .env                              # Local database and API environment variables (git-ignored)
├── .env.example                      # Template environment variable configuration with safe placeholders
├── .gitignore                        # Git exclusion rules (.env, venv, pycache, mlflow, etc.)
├── README.md                         # Root repository documentation and architecture guide
├── requirements.txt                  # Locked Python package dependencies
│
├── dashboard/
│   └── app.py                        # Streamlit executive retention intelligence dashboard
│
├── data/
│   └── raw/
│       └── customer_churn.csv        # Not included in repo - user must download separately (see Section 5)
│
├── docs/
│   └── database_design.md            # Schema architecture and relational dictionary
│
├── mlflow/
│   ├── mlflow.db                     # SQLite tracking repository (git-ignored)
│   ├── artifacts/                    # Logged model weights and plots (git-ignored)
│   └── temp_plots/                   # Staged run evaluation charts (git-ignored)
│
├── models/
│   ├── best_model.joblib             # Champion XGBoost model artifact (< 420 KB)
│   ├── preprocessor.joblib           # Fitted scikit-learn preprocessing pipeline (< 7 KB)
│   ├── feature_names.joblib          # Ordered list of transformed feature names (< 2 KB)
│   ├── logistic_regression.joblib    # Candidate model artifact (local run)
│   ├── logistic_regression_smote.joblib # Candidate model artifact (local run)
│   ├── random_forest.joblib          # Candidate model artifact (local run)
│   ├── random_forest_smote.joblib    # Candidate model artifact (local run)
│   ├── xgboost.joblib                # Candidate model artifact (local run)
│   └── xgboost_smote.joblib          # Candidate model artifact (local run)
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

## 21. Limitations & Future Improvements

### Current Limitations
1. **Cross-Sectional Observational Data**: The dataset provides a single point-in-time snapshot. Real-world subscriber churn modeling benefits significantly from longitudinal time-series data (e.g., weekly data usage changes, billing dispute logs, customer service tickets) to identify churn signals before they manifest.
2. **Heuristic Risk Thresholds**: Risk categories (`High >= 0.70`, `Medium >= 0.40`) are rule-of-thumb business heuristics rather than calibrated Bayesian utility thresholds.
3. **Prior Test-Set Exploration**: As disclosed in Section 11, earlier development iterations evaluated candidate models against test data prior to pipeline correction.

### Future Work
- [ ] **Automated Model Drift Monitoring**: Integrate Evidently AI to detect feature distribution shift and prediction drift over time.
- [ ] **Probability Calibration**: Calibrate tree output probabilities using Isotonic Regression or Platt Scaling to improve empirical risk alignment.
- [ ] **SHAP Tree Explainability**: Incorporate SHAP waterfall values directly into the `/predict` API response so users can see which specific features contributed most to a customer's risk score.
- [ ] **Streaming Data Ingestion**: Extend the data pipeline to consume subscriber telemetry events from Kafka or Pub/Sub.

---

## 22. Project Metadata & Author

- **Author**: Tanay Chaturvedi
- **Role**: Student Developer (Machine Learning & Data Engineering)
- **Repository**: [Tanay-chaturvedi/Customer360-Churn-Prediction](https://github.com/Tanay-chaturvedi/Customer360-Churn-Prediction)
- **License**: No project license specified; dataset use is subject to the original source's terms.
