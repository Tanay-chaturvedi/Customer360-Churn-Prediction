"""Database and table initialization for Customer360.

Creates the database `customer360_db` if not exists and provisions the
`customers` table with proper data types, primary keys, and analytic indexes.
"""

import sys
import argparse
import logging
from sqlalchemy import text
from src.database.db_connection import get_engine, get_db_config

logger = logging.getLogger("Customer360.CreateTables")


def create_database_if_not_exists() -> None:
    """Ensure the target database exists on the MySQL server."""
    cfg = get_db_config()
    db_name = cfg["database"]
    logger.info(f"Checking database '{db_name}' existence on MySQL server...")

    engine_server = get_engine(include_db=False)
    with engine_server.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        create_db_sql = text(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        conn.execute(create_db_sql)
        logger.info(f"Database '{db_name}' verified/created successfully.")


def create_customers_table(drop_existing: bool = False) -> None:
    """Create the customers table with optimized schema and indexes."""
    create_database_if_not_exists()

    engine = get_engine(include_db=True)
    with engine.begin() as conn:
        if drop_existing:
            logger.warning("Dropping existing 'customers' table...")
            conn.execute(text("DROP TABLE IF EXISTS `customers`;"))

        ddl = """
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
        """
        conn.execute(text(ddl))
        logger.info("Table 'customers' created or verified successfully.")


def verify_table_structure() -> None:
    """Print the created table structure for validation."""
    engine = get_engine(include_db=True)
    with engine.connect() as conn:
        result = conn.execute(text("DESCRIBE `customers`;")).fetchall()
        logger.info("Table `customers` schema verification:")
        for row in result:
            field, col_type, null, key, default, extra = row
            logger.info(f"  - {field:<18} {col_type:<18} Null:{null:<4} Key:{key:<4} Default:{str(default)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create database and tables for Customer360.")
    parser.add_argument("--drop", action="store_true", help="Drop existing customers table before creation.")
    args = parser.parse_args()

    try:
        create_customers_table(drop_existing=args.drop)
        verify_table_structure()
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)
        sys.exit(1)
