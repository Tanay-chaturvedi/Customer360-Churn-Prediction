"""Data loading script for Customer360.

Ingests raw data from `data/raw/customer_churn.csv` into the MySQL `customers` table,
performing data cleaning, missing value resolution for TotalCharges, duplicate prevention,
and atomic transaction handling.
"""

import sys
import time
import logging
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from src.database.db_connection import get_engine
from src.database.create_tables import create_customers_table

logger = logging.getLogger("Customer360.LoadData")


def get_dataset_path() -> Path:
    """Locate customer_churn.csv across expected relative paths."""
    candidate_paths = [
        Path("data/raw/customer_churn.csv"),
        Path("../data/raw/customer_churn.csv"),
        Path(__file__).resolve().parents[2] / "data" / "raw" / "customer_churn.csv"
    ]
    for p in candidate_paths:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("Could not find 'data/raw/customer_churn.csv'. Please check directory structure.")


def prepare_data(csv_path: Path) -> pd.DataFrame:
    """Read and sanitize the dataset before database insertion."""
    logger.info(f"Reading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    initial_rows = len(df)
    logger.info(f"Loaded {initial_rows:,} raw records with {df.shape[1]} attributes.")

    # 1. Strip whitespace from all string columns
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # 2. Check for duplicate customer IDs
    duplicates = df['customerID'].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate customer IDs! Deduplicating keeping first...")
        df = df.drop_duplicates(subset=['customerID'], keep='first')
    else:
        logger.info("Verified 100% customerID uniqueness (0 duplicates).")

    # 3. Handle TotalCharges anomaly (blank strings -> numeric 0.00)
    blank_tc_count = (df['TotalCharges'] == '').sum()
    if blank_tc_count > 0:
        logger.info(f"Found {blank_tc_count} rows with blank 'TotalCharges'. Coercing to 0.00 (tenure=0 new accounts).")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.00).round(2)

    # 4. Ensure MonthlyCharges is numeric decimal
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').round(2)

    # 5. Ensure SeniorCitizen is integer
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

    return df


def load_data_to_mysql(truncate_first: bool = True) -> int:
    """Load sanitized customer data into the MySQL `customers` table."""
    start_time = time.time()
    csv_path = get_dataset_path()
    df = prepare_data(csv_path)

    # Ensure table exists
    create_customers_table(drop_existing=False)

    engine = get_engine(include_db=True)

    with engine.begin() as conn:
        if truncate_first:
            logger.info("Truncating 'customers' table before loading to prevent duplicates...")
            conn.execute(text("TRUNCATE TABLE `customers`;"))

        logger.info(f"Inserting {len(df):,} records into MySQL 'customers' table...")
        df.to_sql(
            name='customers',
            con=conn,
            if_exists='append',
            index=False,
            chunksize=1000
        )

    duration = time.time() - start_time
    logger.info(f"Successfully loaded {len(df):,} records into MySQL in {duration:.2f} seconds.")

    # Verification queries
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM `customers`;")).scalar()
        sample = conn.execute(text("SELECT customerID, tenure, MonthlyCharges, TotalCharges, Churn FROM `customers` LIMIT 3;")).fetchall()
        tc_zero_count = conn.execute(text("SELECT COUNT(*) FROM `customers` WHERE TotalCharges = 0;")).scalar()

    logger.info("--- Post-Load Validation Metrics ---")
    logger.info(f"Total Rows in MySQL Table: {count:,} (Expected: {len(df):,})")
    logger.info(f"Accounts with TotalCharges = $0.00 (tenure=0): {tc_zero_count}")
    logger.info("Sample Records:")
    for row in sample:
        logger.info(f"  {row}")

    if count != len(df):
        raise ValueError(f"Data verification failed! Expected {len(df)} rows, found {count} in database.")

    return count


if __name__ == "__main__":
    try:
        load_data_to_mysql(truncate_first=True)
    except Exception as e:
        logger.error(f"Data loading failed: {e}", exc_info=True)
        sys.exit(1)
