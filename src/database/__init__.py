"""Database module for Customer360.
Provides database connection, table initialization, and ETL data ingestion.
"""

from .db_connection import get_engine, get_raw_connection, test_connection

__all__ = ["get_engine", "get_raw_connection", "test_connection"]
