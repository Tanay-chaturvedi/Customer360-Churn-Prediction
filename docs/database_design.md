# 🏛️ Customer360 Database Design & SQL Analytics Architecture

This document outlines the relational database architecture, table schema, data ingestion pipelines, and SQL analytical queries developed for the **Customer360 – Customer Churn Prediction & Retention Intelligence Platform**.

---

## 1. Database Architecture & Design Rationale

### 1.1 Technology Selection
- **Database Engine**: MySQL 8.0 (InnoDB Storage Engine)
- **Database Name**: `customer360_db`
- **Character Set & Collation**: `utf8mb4` with `utf8mb4_unicode_ci` (Full Unicode support for internationalization and robust string comparison).
- **Driver & ORM Stack**: SQLAlchemy 2.x with PyMySQL pure-Python connector for maximum portability and connection pooling.

### 1.2 Architectural Principles
1. **Primary Entity Granularity**: Each row represents exactly one customer account uniquely identified by `customerID`.
2. **ACID Compliance**: Data ingestion utilizes InnoDB transactions to guarantee atomic loads (either all 7,043 rows insert cleanly or the transaction aborts with zero partial states).
3. **Data Type Precision**: Numerical currencies (`MonthlyCharges`, `TotalCharges`) utilize fixed-point `DECIMAL(10, 2)` to eliminate IEEE 754 floating-point rounding errors common in monetary aggregation.
4. **Targeted Analytical Indexing**: B-tree indexes are defined on high-frequency grouping and filtering dimensions (`Churn`, `Contract`, `InternetService`, `PaymentMethod`, `tenure`, `MonthlyCharges`) to accelerate reporting and analytical query execution.
5. **Separation of Operational Keys from Feature Sets**: `customerID` serves as the primary database entity key for customer tracking and dashboard retrieval, but is strictly isolated from model training matrices.

---

## 2. Table Schema (`customers`)

```sql
CREATE TABLE IF NOT EXISTS `customers` (
    `customerID` VARCHAR(50) NOT NULL,
    `gender` VARCHAR(20) NOT NULL,
    `SeniorCitizen` TINYINT(1) NOT NULL,
    `Partner` VARCHAR(10) NOT NULL,
    `Dependents` VARCHAR(10) NOT NULL,
    `tenure` INT NOT NULL,
    `PhoneService` VARCHAR(10) NOT NULL,
    `MultipleLines` VARCHAR(30) NOT NULL,
    `InternetService` VARCHAR(30) NOT NULL,
    `OnlineSecurity` VARCHAR(30) NOT NULL,
    `OnlineBackup` VARCHAR(30) NOT NULL,
    `DeviceProtection` VARCHAR(30) NOT NULL,
    `TechSupport` VARCHAR(30) NOT NULL,
    `StreamingTV` VARCHAR(30) NOT NULL,
    `StreamingMovies` VARCHAR(30) NOT NULL,
    `Contract` VARCHAR(30) NOT NULL,
    `PaperlessBilling` VARCHAR(10) NOT NULL,
    `PaymentMethod` VARCHAR(50) NOT NULL,
    `MonthlyCharges` DECIMAL(10, 2) NOT NULL,
    `TotalCharges` DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
    `Churn` VARCHAR(10) NOT NULL,
    PRIMARY KEY (`customerID`),
    INDEX `idx_customers_churn` (`Churn`),
    INDEX `idx_customers_contract` (`Contract`),
    INDEX `idx_customers_internetservice` (`InternetService`),
    INDEX `idx_customers_paymentmethod` (`PaymentMethod`),
    INDEX `idx_customers_tenure` (`tenure`),
    INDEX `idx_customers_monthlycharges` (`MonthlyCharges`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.1 Column Definitions & Data Type Justifications

| Column Name | MySQL Data Type | Constraints | Description & Justification |
| :--- | :--- | :--- | :--- |
| `customerID` | `VARCHAR(50)` | `PRIMARY KEY` | Unique customer alphanumeric token (e.g., `'7590-VHVEG'`). Ensures entity uniqueness and prevents duplicate records. |
| `gender` | `VARCHAR(20)` | `NOT NULL` | Customer biological gender (`'Female'`, `'Male'`). |
| `SeniorCitizen` | `TINYINT(1)` | `NOT NULL` | Binary demographic indicator (`0` = Under 65, `1` = Senior Citizen 65+). Storage-efficient alternative to full integer. |
| `Partner` | `VARCHAR(10)` | `NOT NULL` | Whether customer has a domestic partner (`'Yes'`, `'No'`). |
| `Dependents` | `VARCHAR(10)` | `NOT NULL` | Whether customer has financial dependents (`'Yes'`, `'No'`). |
| `tenure` | `INT` | `NOT NULL` | Total number of months customer has remained subscribed (range: `0` to `72`). |
| `PhoneService` | `VARCHAR(10)` | `NOT NULL` | Whether customer subscribes to home phone service (`'Yes'`, `'No'`). |
| `MultipleLines` | `VARCHAR(30)` | `NOT NULL` | Status of multiple phone lines (`'No'`, `'Yes'`, `'No phone service'`). |
| `InternetService`| `VARCHAR(30)` | `NOT NULL` | Primary internet architecture (`'DSL'`, `'Fiber optic'`, `'No'`). Top predictive signal. |
| `OnlineSecurity` | `VARCHAR(30)` | `NOT NULL` | Security add-on subscription (`'No'`, `'Yes'`, `'No internet service'`). |
| `OnlineBackup` | `VARCHAR(30)` | `NOT NULL` | Cloud storage backup service (`'No'`, `'Yes'`, `'No internet service'`). |
| `DeviceProtection` | `VARCHAR(30)` | `NOT NULL` | Hardware warranty/protection (`'No'`, `'Yes'`, `'No internet service'`). |
| `TechSupport` | `VARCHAR(30)` | `NOT NULL` | Dedicated technical support tier (`'No'`, `'Yes'`, `'No internet service'`). |
| `StreamingTV` | `VARCHAR(30)` | `NOT NULL` | Premium television streaming add-on (`'No'`, `'Yes'`, `'No internet service'`). |
| `StreamingMovies`| `VARCHAR(30)` | `NOT NULL` | Premium on-demand movie streaming (`'No'`, `'Yes'`, `'No internet service'`). |
| `Contract` | `VARCHAR(30)` | `NOT NULL` | Service agreement term (`'Month-to-month'`, `'One year'`, `'Two year'`). Core churn indicator. |
| `PaperlessBilling` | `VARCHAR(10)` | `NOT NULL` | Electronic invoice preference (`'Yes'`, `'No'`). |
| `PaymentMethod` | `VARCHAR(50)` | `NOT NULL` | Billing channel (`'Electronic check'`, `'Mailed check'`, `'Bank transfer (automatic)'`, `'Credit card (automatic)'`). |
| `MonthlyCharges` | `DECIMAL(10, 2)` | `NOT NULL` | Current recurring monthly bill amount in USD. Fixed-point precision avoids float rounding errors. |
| `TotalCharges` | `DECIMAL(10, 2)` | `NOT NULL DEFAULT 0.00` | Cumulative lifetime charges billed to date. 11 raw blank whitespace values are safely coerced to `$0.00` (tenure=0 new accounts). |
| `Churn` | `VARCHAR(10)` | `NOT NULL` | Ground-truth target status (`'Yes'`, `'No'`). |

---

## 3. Data Pipeline & Ingestion Architecture

```
[data/raw/customer_churn.csv]
               │
               ▼
[src/database/load_data.py]
  ├── 1. Whitespace Trimming on String Attributes
  ├── 2. Customer ID Uniqueness & Duplicate Audit
  ├── 3. TotalCharges Sanitization (Blank strings -> 0.00 for tenure=0)
  ├── 4. Numeric Type Coercion & Precision Enforcement
  └── 5. Transactional Atomic Load via SQLAlchemy (InnoDB Engine)
               │
               ▼
   [MySQL: customer360_db.customers]
```

---

## 4. SQL Analytics Suite (15 Strategic Queries)

All queries are consolidated in `sql/customer_analysis.sql` and executable via `python -m src.database.run_analysis`.

### Query 1: Total Customer Volume
- **SQL**: `SELECT COUNT(*) AS total_customers FROM customers;`
- **Purpose**: Establishes baseline scale (`7,043` accounts).

### Query 2: Total Churned vs. Retained Customers
- **SQL**: `SELECT Churn AS churn_status, COUNT(*) AS customer_count FROM customers GROUP BY Churn;`
- **Purpose**: Counts active retained (`5,174`) vs. churned (`1,869`) accounts.

### Query 3: Overall Churn & Retention Rates
- **SQL**: Computes proportion of churned vs. retained subscribers.
- **Key Finding**: Baseline churn is **26.54%**; retention rate is **73.46%**.

### Query 4: Churn Analysis by Contract Type
- **SQL**: Groups by `Contract`, calculating customer count and churn percentage.
- **Key Finding**: Month-to-month contracts have **42.71% churn**, while 1-year is **11.27%** and 2-year is **2.83%**. Over 88% of all churn occurs on month-to-month plans.

### Query 5: Churn Analysis by Payment Method
- **SQL**: Evaluates churn rate and average monthly bill by payment gateway.
- **Key Finding**: **Electronic check** users exhibit a **45.29% churn rate**, compared to **15.24%** for credit card auto-pay and **16.71%** for bank transfer auto-pay.

### Query 6: Churn Analysis by Internet Service Architecture
- **SQL**: Analyzes DSL, Fiber Optic, and No Internet cohorts.
- **Key Finding**: **Fiber optic** accounts churn at **41.89%** despite having the highest average bill ($91.50/mo), indicating service delivery friction or expectation mismatch.

### Query 7: Churn by Senior Citizen Demographic
- **SQL**: Segments customers by `SeniorCitizen` flag.
- **Key Finding**: Senior citizens (65+) churn at **41.68%**, almost double the **23.61%** rate among non-seniors.

### Query 8: Average Monthly Charges by Churn Status
- **SQL**: Computes mean, min, max, and standard deviation of `MonthlyCharges` by churn.
- **Key Finding**: Churned customers pay an average of **$74.44/month**, compared to **$61.27/month** for retained customers (a $13.17/mo premium).

### Query 9: Average Tenure by Churn Status
- **SQL**: Aggregates customer lifecycle metrics by churn status.
- **Key Finding**: Retained customers stay an average of **37.6 months**, whereas churned customers stay only **18.0 months** (with median tenure at 10 months).

### Query 10: Top Compound Risk Micro-Segments
- **SQL**: Groups by `(Contract, InternetService, PaymentMethod)` with `HAVING COUNT(*) >= 50`.
- **Key Finding**: Customers on **Month-to-month + Fiber optic + Electronic check** have an extraordinary **54.6% churn rate**, representing the single highest-risk cohort in the business.

### Query 11: High-Spend Month-to-Month Exposure
- **SQL**: Isolates subscribers paying $\ge \$70.00$/month on Month-to-month contracts.
- **Key Finding**: Identifies high-risk revenue concentration where over 50% of monthly run-rate in this bracket is lost to attrition.

### Query 12: Direct Revenue Exposure Among Churned Customers
- **SQL**: Sums `MonthlyCharges` (lost MRR) and `TotalCharges` (historical LTV lost).
- **Key Finding**: Total lost Monthly Recurring Revenue (MRR) from churned accounts is **$139,130.85/month**, totaling over **$2.86 Million** in historical lost billing.

### Query 13: Early Tenure Risk Exposure (Tenure $\le$ 6 Months)
- **SQL**: Bins customers into Onboarding Cohorts: `Month 0-1`, `Month 2-3`, `Month 4-6`, and `>6 Months`.
- **Key Finding**: Subscribers in **Months 0-1 experience a 51.9% churn rate**; months 2-3 experience 40.5% churn. Churn dramatically drops after month 6.

### Query 14: Churn Analysis by Paperless Billing Adoption
- **SQL**: Compares paperless billing vs. paper invoices.
- **Key Finding**: Paperless billing subscribers churn at **33.57%** vs. **16.33%** for paper billing. (Paperless billing correlates strongly with Electronic check and Fiber optic subscriptions).

### Query 15: Strategic 9-Quadrant Customer Segmentation Matrix
- **SQL**: Cross-tabulates 3 tenure tiers ($\le 12$m, $13-36$m, $>36$m) with 3 spend tiers ($<\$35$, $\$35-\$70$, $>\$70$).
- **Key Finding**: **Short Tenure + High Spend** accounts have the highest churn rate (**~53.8%**), pinpointing the primary target segment for VIP retention intervention.

---

## 5. Security & Secret Management

- **Environment Isolation**: Database credentials are strictly externalized into `.env`.
- **Git Protection**: `.env` is registered in `.gitignore` to prevent secret leaks to GitHub.
- **Template Provisioning**: `.env.example` provides a documented template for deployment across local, staging, and production environments.
