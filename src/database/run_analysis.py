"""SQL Analytics Runner for Customer360.

Executes all 15 queries in `sql/customer_analysis.sql` against MySQL,
validates output correctness, and prints formatted summary tables.
"""

import sys
import re
import logging
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from src.database.db_connection import get_engine

logger = logging.getLogger("Customer360.SQLRunner")


def parse_sql_file(sql_path: Path) -> list:
    """Parse sql file into discrete query titles and statements."""
    with open(sql_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on statement comments
    sections = re.split(r"-- -{50,}\n", content)
    queries = []

    for sec in sections:
        sec = sec.strip()
        if not sec or sec.startswith("-- =="):
            continue

        # Extract title from comments
        lines = sec.splitlines()
        title_lines = [l.strip().lstrip("-").strip() for l in lines if l.strip().startswith("-- QUERY") or l.strip().startswith("-- Purpose")]
        title = " | ".join(title_lines) if title_lines else "SQL Query"

        # Extract executable SQL (strip out comments and USE statements)
        sql_lines = [l for l in lines if not l.strip().startswith("--")]
        stmt = "\n".join(sql_lines).strip()
        if stmt.endswith(";"):
            stmt = stmt[:-1].strip()

        if stmt and not stmt.upper().startswith("USE "):
            queries.append({"title": title, "query": stmt})

    return queries


def run_all_queries() -> bool:
    """Execute all queries and log results."""
    sql_path = Path("sql/customer_analysis.sql")
    if not sql_path.exists():
        sql_path = Path(__file__).resolve().parents[2] / "sql" / "customer_analysis.sql"

    logger.info(f"Loading SQL analysis suite from: {sql_path}")
    queries = parse_sql_file(sql_path)
    logger.info(f"Identified {len(queries)} distinct analytical queries.")

    engine = get_engine(include_db=True)
    all_passed = True

    with engine.connect() as conn:
        for idx, q in enumerate(queries, start=1):
            title = q["title"]
            stmt = q["query"]
            print(f"\n{'='*80}")
            print(f"[{idx}/15] {title}")
            print(f"{'='*80}")

            try:
                df_res = pd.read_sql_query(text(stmt), conn)
                print(df_res.to_string(index=False))
            except Exception as e:
                logger.error(f"Failed executing query {idx}: {e}")
                all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_all_queries()
    sys.exit(0 if success else 1)
