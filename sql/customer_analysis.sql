-- ============================================================================
-- Customer360: Customer Churn Prediction & Retention Intelligence Platform
-- SQL Analytics & Business Diagnostic Queries
-- Database: customer360_db | Table: customers
-- ============================================================================

USE customer360_db;

-- ----------------------------------------------------------------------------
-- QUERY 1: Total Customer Count
-- Purpose: Establishes the baseline volume of customer accounts in the database.
-- ----------------------------------------------------------------------------
SELECT
    COUNT(*) AS total_customers
FROM customers;


-- ----------------------------------------------------------------------------
-- QUERY 2: Total Churned vs. Retained Customers
-- Purpose: Counts total churned vs. active retained customers.
-- ----------------------------------------------------------------------------
SELECT
    Churn AS churn_status,
    COUNT(*) AS customer_count
FROM customers
GROUP BY Churn;


-- ----------------------------------------------------------------------------
-- QUERY 3: Overall Churn Percentage & Retention Rate
-- Purpose: Calculates high-level organizational churn benchmark.
-- ----------------------------------------------------------------------------
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS total_churned,
    SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS total_retained,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS churn_rate_pct,
    ROUND((SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS retention_rate_pct
FROM customers;


-- ----------------------------------------------------------------------------
-- QUERY 4: Churn Analysis by Contract Type
-- Purpose: Evaluates customer attrition across commitment durations.
-- ----------------------------------------------------------------------------
SELECT
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS retained_customers,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY 5: Churn Analysis by Payment Method
-- Purpose: Identifies high-risk payment gateways and billing channels.
-- ----------------------------------------------------------------------------
SELECT
    PaymentMethod,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY 6: Churn Analysis by Internet Service Architecture
-- Purpose: Assesses customer attrition across network service tiers.
-- ----------------------------------------------------------------------------
SELECT
    InternetService,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY 7: Churn by Senior Citizen Demographic
-- Purpose: Compares attrition risk between senior citizens (65+) and non-seniors.
-- ----------------------------------------------------------------------------
SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior Citizen (65+)' ELSE 'Non-Senior (<65)' END AS demographic_segment,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY SeniorCitizen
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY 8: Average Monthly Charges by Churn Status
-- Purpose: Quantifies price disparity between churned and retained customers.
-- ----------------------------------------------------------------------------
SELECT
    Churn AS churn_status,
    COUNT(*) AS customer_count,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(MIN(MonthlyCharges), 2) AS min_monthly_charges,
    ROUND(MAX(MonthlyCharges), 2) AS max_monthly_charges,
    ROUND(STDDEV(MonthlyCharges), 2) AS std_monthly_charges
FROM customers
GROUP BY Churn;


-- ----------------------------------------------------------------------------
-- QUERY 9: Average Tenure by Churn Status
-- Purpose: Demonstrates customer lifecycle duration disparity between segments.
-- ----------------------------------------------------------------------------
SELECT
    Churn AS churn_status,
    COUNT(*) AS customer_count,
    ROUND(AVG(tenure), 1) AS avg_tenure_months,
    MIN(tenure) AS min_tenure_months,
    MAX(tenure) AS max_tenure_months,
    ROUND(STDDEV(tenure), 1) AS std_tenure_months
FROM customers
GROUP BY Churn;


-- ----------------------------------------------------------------------------
-- QUERY 10: Top High-Risk Customer Micro-Segments (Compound Risk Analysis)
-- Purpose: Discovers multi-attribute intersections with severe churn concentrations.
-- ----------------------------------------------------------------------------
SELECT
    Contract,
    InternetService,
    PaymentMethod,
    COUNT(*) AS segment_size,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_count,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_at_risk
FROM customers
GROUP BY Contract, InternetService, PaymentMethod
HAVING COUNT(*) >= 50
ORDER BY churn_rate_pct DESC
LIMIT 10;


-- ----------------------------------------------------------------------------
-- QUERY 11: High Monthly Charges & Month-to-Month Contract Vulnerability
-- Purpose: Identifies the most financially volatile cohort (Paying >$70/mo on month-to-month).
-- ----------------------------------------------------------------------------
SELECT
    COUNT(*) AS total_high_paying_m2m,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_count,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_runrate,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost
FROM customers
WHERE Contract = 'Month-to-month' AND MonthlyCharges >= 70.00;


-- ----------------------------------------------------------------------------
-- QUERY 12: Direct Revenue Exposure Among Churned Customers
-- Purpose: Calculates total Monthly Recurring Revenue (MRR) and cumulative LTV lost.
-- ----------------------------------------------------------------------------
SELECT
    COUNT(*) AS churned_customers,
    ROUND(SUM(MonthlyCharges), 2) AS lost_mrr,
    ROUND(AVG(MonthlyCharges), 2) AS avg_churned_monthly_spend,
    ROUND(SUM(TotalCharges), 2) AS cumulative_historical_revenue_lost,
    ROUND(AVG(TotalCharges), 2) AS avg_lifetime_value_lost
FROM customers
WHERE Churn = 'Yes';


-- ----------------------------------------------------------------------------
-- QUERY 13: Early Tenure Risk Exposure (Customers with Tenure <= 6 Months)
-- Purpose: Quantifies customer attrition during the critical onboarding window.
-- ----------------------------------------------------------------------------
SELECT
    CASE
        WHEN tenure <= 1 THEN 'Month 0 - 1 (New Subscriber)'
        WHEN tenure <= 3 THEN 'Month 2 - 3 (Early Onboarding)'
        WHEN tenure <= 6 THEN 'Month 4 - 6 (Mid Onboarding)'
        ELSE 'Tenure > 6 Months (Established)'
    END AS onboarding_cohort,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS cohort_churn_rate_pct
FROM customers
GROUP BY onboarding_cohort
ORDER BY MIN(tenure) ASC;


-- ----------------------------------------------------------------------------
-- QUERY 14: Churn Analysis by Paperless Billing Adoption
-- Purpose: Evaluates correlation between digital invoicing and customer attrition.
-- ----------------------------------------------------------------------------
SELECT
    PaperlessBilling,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY PaperlessBilling
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY 15: Strategic Customer Segmentation Matrix (Tenure vs. Monthly Spend)
-- Purpose: Partitions customer base into 9 actionable quadrants for retention targeting.
-- ----------------------------------------------------------------------------
SELECT
    CASE
        WHEN tenure <= 12 THEN 'Tier 1: Short Tenure (<=12m)'
        WHEN tenure <= 36 THEN 'Tier 2: Mid Tenure (13-36m)'
        ELSE 'Tier 3: Loyal (>36m)'
    END AS tenure_tier,
    CASE
        WHEN MonthlyCharges < 35 THEN 'Low Spend (<$35)'
        WHEN MonthlyCharges <= 70 THEN 'Medium Spend ($35-$70)'
        ELSE 'High Spend (>$70)'
    END AS spend_tier,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS segment_churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS segment_mrr
FROM customers
GROUP BY tenure_tier, spend_tier
ORDER BY tenure_tier ASC, spend_tier DESC;
