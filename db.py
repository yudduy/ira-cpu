"""
Database module for CPU Index Builder

Handles all SQLite operations:
- Storing article metadata
- Tracking collection progress (for resume)
- Storing LLM classifications
- Storing final index values
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import config


# TODO: Refactor to use context manager pattern for better resource management
# Current pattern opens/closes connection on each operation. Best practice is:
#
#   from contextlib import closing
#   with closing(sqlite3.connect(DB_PATH)) as conn:
#       with conn:  # handles transaction commit/rollback
#           cursor = conn.cursor()
#           ...
#
# This ensures connections are always closed, even on exceptions.
# See: https://pyneng.readthedocs.io/en/latest/book/25_db/sqlite3_context_manager.html
# See: https://blog.rtwilson.com/a-python-sqlite3-context-manager-gotcha/
def get_connection() -> sqlite3.Connection:
    """Get database connection, creating file if needed."""
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    return conn


def init_db():
    """
    Create all tables if they don't exist.
    Safe to call multiple times.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Articles table - stores article metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            month TEXT NOT NULL,
            title TEXT,
            source TEXT,
            date TEXT,
            snippet TEXT,
            full_text TEXT,
            has_uncertainty_keyword INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Progress table - tracks which months are complete
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            month TEXT NOT NULL,
            query_type TEXT NOT NULL,
            count INTEGER NOT NULL,
            completed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (month, query_type)
        )
    """)

    # Classifications table - LLM validation results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            article_id TEXT PRIMARY KEY,
            is_climate_policy INTEGER,
            has_uncertainty INTEGER,
            reasoning TEXT,
            confidence TEXT,
            classified_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (article_id) REFERENCES articles(id)
        )
    """)

    # Index values table - final CPU index
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS index_values (
            month TEXT PRIMARY KEY,
            denominator INTEGER NOT NULL,
            numerator INTEGER NOT NULL,
            raw_ratio REAL NOT NULL,
            normalized REAL,
            calculated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_month_count(month: str, query_type: str, count: int):
    """
    Save the article count for a month.
    query_type: 'denominator' or 'numerator'
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO progress (month, query_type, count, completed_at)
        VALUES (?, ?, ?, ?)
    """, (month, query_type, count, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def get_month_count(month: str, query_type: str) -> Optional[int]:
    """Get saved count for a month, or None if not collected."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT count FROM progress WHERE month = ? AND query_type = ?
    """, (month, query_type))

    row = cursor.fetchone()
    conn.close()

    return row["count"] if row else None


def get_all_progress() -> list:
    """Get all progress records."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT month, query_type, count, completed_at
        FROM progress
        ORDER BY month, query_type
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_completed_months() -> set:
    """Get set of months where BOTH denominator and numerator are collected."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT month FROM progress
        GROUP BY month
        HAVING COUNT(DISTINCT query_type) = 2
    """)

    months = {row["month"] for row in cursor.fetchall()}
    conn.close()
    return months


def save_article(article: dict):
    """Save a single article to the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO articles
        (id, month, title, source, date, snippet, full_text, has_uncertainty_keyword)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        article["id"],
        article["month"],
        article.get("title"),
        article.get("source"),
        article.get("date"),
        article.get("snippet"),
        article.get("full_text"),
        article.get("has_uncertainty_keyword", 0),
    ))

    conn.commit()
    conn.close()


def save_classification(article_id: str, is_climate_policy: bool,
                        has_uncertainty: bool, reasoning: str, confidence: str):
    """Save LLM classification result."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO classifications
        (article_id, is_climate_policy, has_uncertainty, reasoning, confidence, classified_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        article_id,
        int(is_climate_policy),
        int(has_uncertainty),
        reasoning,
        confidence,
        datetime.now().isoformat(),
    ))

    conn.commit()
    conn.close()


def save_index_value(month: str, denominator: int, numerator: int,
                     raw_ratio: float, normalized: float = None):
    """Save calculated index value for a month."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO index_values
        (month, denominator, numerator, raw_ratio, normalized, calculated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (month, denominator, numerator, raw_ratio, normalized, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def get_all_index_values() -> list:
    """Get all index values for export."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT month, denominator, numerator, raw_ratio, normalized
        FROM index_values
        ORDER BY month
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def export_to_csv(output_path: str):
    """Export index to CSV file."""
    import csv

    values = get_all_index_values()
    if not values:
        raise ValueError("No index values to export. Run 'Build Index' first.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["month", "denominator", "numerator", "raw_ratio", "normalized"])
        writer.writeheader()
        writer.writerows(values)

    return len(values)
