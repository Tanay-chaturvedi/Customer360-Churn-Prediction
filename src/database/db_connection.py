"""Database connection management for Customer360.

Handles MySQL connection pooling, credentials loading via environment
variables, and engine initialization using SQLAlchemy and PyMySQL.
"""

import os
import sys
import logging
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pymysql

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Customer360.DB")

# Locate and load .env file from project root
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
    logger.info(f"Loaded environment variables from: {env_path}")
else:
    fallback_env = Path(__file__).resolve().parents[2] / ".env"
    if fallback_env.exists():
        load_dotenv(fallback_env)
        logger.info(f"Loaded fallback environment from: {fallback_env}")
    else:
        logger.warning("No .env file found. Relying on system environment variables.")


def get_db_config() -> dict:
    """Retrieve database connection parameters from environment variables.
    Supports both DB_* and MYSQL_* naming conventions.
    """
    host = os.getenv("DB_HOST") or os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("DB_PORT") or os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("DB_USER") or os.getenv("MYSQL_USER", "root")
    password = os.getenv("DB_PASSWORD") or os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("DB_NAME") or "customer360_db"

    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database
    }


def get_connection_url(include_db: bool = True) -> str:
    """Construct a safe SQLAlchemy connection URL with URL-encoded credentials."""
    cfg = get_db_config()
    safe_user = urllib.parse.quote_plus(cfg["user"])
    safe_pwd = urllib.parse.quote_plus(cfg["password"]) if cfg["password"] else ""
    auth = f"{safe_user}:{safe_pwd}" if safe_pwd else safe_user
    db_target = f"/{cfg['database']}" if include_db else ""

    return f"mysql+pymysql://{auth}@{cfg['host']}:{cfg['port']}{db_target}?charset=utf8mb4"


def get_engine(include_db: bool = True, pool_size: int = 5, pool_recycle: int = 3600) -> Engine:
    """Create and return an SQLAlchemy Engine instance."""
    url = get_connection_url(include_db=include_db)
    return create_engine(
        url,
        pool_size=pool_size,
        pool_recycle=pool_recycle,
        echo=False
    )


def get_raw_connection(include_db: bool = True) -> pymysql.Connection:
    """Obtain a direct PyMySQL raw connection for low-level transactions and administrative commands."""
    cfg = get_db_config()
    return pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"] if include_db else None,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )


def test_connection() -> bool:
    """Verify MySQL connectivity and log diagnostic details."""
    cfg = get_db_config()
    logger.info(f"Attempting connection to MySQL at {cfg['host']}:{cfg['port']} as user '{cfg['user']}'...")

    try:
        # First verify server availability without requiring target database to exist
        engine_server = get_engine(include_db=False)
        with engine_server.connect() as conn:
            version = conn.execute(text("SELECT VERSION()")).scalar()
            logger.info(f"Connected to MySQL Server successfully! Version: {version}")

        # Check if target database exists
        with engine_server.connect() as conn:
            dbs = [row[0] for row in conn.execute(text("SHOW DATABASES")).fetchall()]
            if cfg["database"] in dbs:
                logger.info(f"Target database '{cfg['database']}' exists and is accessible.")
            else:
                logger.info(f"Target database '{cfg['database']}' does not exist yet (will be created automatically).")

        return True
    except pymysql.err.OperationalError as e:
        logger.error(f"MySQL Operational Error ({e.args[0]}): {e.args[1]}")
        if e.args[0] == 1045:
            logger.error("Authentication failed! Please check DB_USER and DB_PASSWORD in your .env file.")
        elif e.args[0] == 2003:
            logger.error(f"Cannot connect to MySQL server on '{cfg['host']}:{cfg['port']}'. Is the MySQL service running?")
        return False
    except Exception as e:
        logger.error(f"Failed to connect to MySQL: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
