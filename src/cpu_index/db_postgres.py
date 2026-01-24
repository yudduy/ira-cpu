"""
PostgreSQL Database Module for CPU Index Builder

Handles all PostgreSQL operations:
- Connection management with connection pooling
- Article storage and retrieval
- Keyword classifications
- LLM validation results
- Index value caching
- Collection progress tracking

Replaces the SQLite-based db.py for production use.
"""

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_values

from . import config

# Connection pool (initialized lazily)
_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def get_connection_pool() -> pool.ThreadedConnectionPool:
    """Get or create the connection pool."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=config.DATABASE_URL,
        )
    return _connection_pool


@contextmanager
def get_connection():
    """
    Context manager for database connections.
    Automatically returns connection to pool when done.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def close_pool():
    """Close all connections in the pool."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None


def init_db():
    """
    Initialize database schema.
    Creates all tables if they don't exist.
    Safe to call multiple times.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Raw API responses for audit trail
            cur.execute("""
                CREATE TABLE IF NOT EXISTS raw_responses (
                    id SERIAL PRIMARY KEY,
                    query_hash VARCHAR(64) NOT NULL,
                    month VARCHAR(7) NOT NULL,
                    response_json JSONB NOT NULL,
                    fetched_at TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_responses_month
                ON raw_responses(month)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_responses_query_hash
                ON raw_responses(query_hash)
            """)

            # Parsed articles
            cur.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id VARCHAR(255) PRIMARY KEY,
                    title TEXT,
                    date DATE,
                    source VARCHAR(255),
                    snippet TEXT,
                    month VARCHAR(7) NOT NULL,
                    fetched_at TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_month ON articles(month)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(date)
            """)

            # Local keyword classifications
            cur.execute("""
                CREATE TABLE IF NOT EXISTS keyword_classifications (
                    article_id VARCHAR(255) PRIMARY KEY
                        REFERENCES articles(id) ON DELETE CASCADE,
                    has_uncertainty BOOLEAN DEFAULT FALSE,
                    has_reversal_terms BOOLEAN DEFAULT FALSE,
                    has_implementation_terms BOOLEAN DEFAULT FALSE,
                    has_upside_terms BOOLEAN DEFAULT FALSE,
                    has_ira_mention BOOLEAN DEFAULT FALSE,
                    has_obbba_mention BOOLEAN DEFAULT FALSE,
                    matched_terms JSONB,
                    classified_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # LLM validation results
            cur.execute("""
                CREATE TABLE IF NOT EXISTS llm_classifications (
                    article_id VARCHAR(255) PRIMARY KEY
                        REFERENCES articles(id) ON DELETE CASCADE,
                    is_climate_policy BOOLEAN,
                    uncertainty_type VARCHAR(20),
                    certainty_level INTEGER
                        CHECK (certainty_level >= 1 AND certainty_level <= 5),
                    reasoning TEXT,
                    model VARCHAR(50),
                    classified_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Computed indices (cached)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS index_values (
                    month VARCHAR(7) NOT NULL,
                    index_type VARCHAR(50) NOT NULL,
                    outlet VARCHAR(100) DEFAULT '',
                    denominator INTEGER NOT NULL,
                    numerator INTEGER NOT NULL,
                    raw_ratio REAL,
                    normalized REAL,
                    scaled BOOLEAN DEFAULT FALSE,
                    dedup_strategy VARCHAR(50) DEFAULT 'none',
                    computed_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (month, index_type, outlet, scaled, dedup_strategy)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_index_values_type
                ON index_values(index_type)
            """)

            # Collection progress
            cur.execute("""
                CREATE TABLE IF NOT EXISTS collection_progress (
                    month VARCHAR(7) PRIMARY KEY,
                    articles_fetched INTEGER DEFAULT 0,
                    completed_at TIMESTAMP
                )
            """)

            # Monthly counts (count-based collection)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS monthly_counts (
                    month VARCHAR(7) PRIMARY KEY,
                    denominator INTEGER NOT NULL,
                    numerator_cpu INTEGER NOT NULL,
                    numerator_impl INTEGER NOT NULL,
                    numerator_reversal INTEGER NOT NULL,
                    collected_at TIMESTAMP DEFAULT NOW()
                )
            """)


# =============================================================================
# RAW RESPONSE OPERATIONS
# =============================================================================

def save_raw_response(month: str, query: str, response_json: dict):
    """
    Save raw API response for audit trail.

    Args:
        month: Month in YYYY-MM format
        query: The search query used
        response_json: Full API response as dict
    """
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:64]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO raw_responses (query_hash, month, response_json)
                VALUES (%s, %s, %s)
            """, (query_hash, month, json.dumps(response_json)))


def get_raw_responses(month: str) -> list[dict]:
    """Get all raw responses for a month."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM raw_responses WHERE month = %s
                ORDER BY fetched_at
            """, (month,))
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# ARTICLE OPERATIONS
# =============================================================================

def save_article(article: dict):
    """
    Save a single article to the database.
    Uses UPSERT to handle duplicates gracefully.

    Args:
        article: Dict with keys: id, title, date, source, snippet, month
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO articles (id, title, date, source, snippet, month)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    date = EXCLUDED.date,
                    source = EXCLUDED.source,
                    snippet = EXCLUDED.snippet,
                    month = EXCLUDED.month
            """, (
                article["id"],
                article.get("title"),
                article.get("date"),
                article.get("source"),
                article.get("snippet"),
                article["month"],
            ))


def save_articles_batch(articles: list[dict]):
    """
    Save multiple articles efficiently using batch insert.

    Args:
        articles: List of article dicts
    """
    if not articles:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            values = [
                (
                    a["id"],
                    a.get("title"),
                    a.get("date"),
                    a.get("source"),
                    a.get("snippet"),
                    a["month"],
                )
                for a in articles
            ]
            execute_values(
                cur,
                """
                INSERT INTO articles (id, title, date, source, snippet, month)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    date = EXCLUDED.date,
                    source = EXCLUDED.source,
                    snippet = EXCLUDED.snippet,
                    month = EXCLUDED.month
                """,
                values,
            )


def get_article(article_id: str) -> Optional[dict]:
    """Get a single article by ID."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_all_articles(limit: int = None) -> list[dict]:
    """Get all articles from database."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM articles ORDER BY date"
            if limit:
                query += f" LIMIT {limit}"
            cur.execute(query)
            return [dict(row) for row in cur.fetchall()]


def get_articles_by_month(month: str) -> list[dict]:
    """Get all articles for a given month."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM articles WHERE month = %s ORDER BY date
            """, (month,))
            return [dict(row) for row in cur.fetchall()]


def get_articles_by_source(source: str) -> list[dict]:
    """Get all articles from a specific source/outlet."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM articles WHERE source = %s ORDER BY date
            """, (source,))
            return [dict(row) for row in cur.fetchall()]


def get_article_count_by_month() -> dict[str, int]:
    """Get article counts grouped by month."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT month, COUNT(*) as count
                FROM articles
                GROUP BY month
                ORDER BY month
            """)
            return {row[0]: row[1] for row in cur.fetchall()}


def get_article_count_by_source() -> dict[str, int]:
    """Get article counts grouped by source."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source, COUNT(*) as count
                FROM articles
                GROUP BY source
                ORDER BY count DESC
            """)
            return {row[0]: row[1] for row in cur.fetchall()}


def get_total_article_count() -> int:
    """Get total number of articles in database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM articles")
            return cur.fetchone()[0]


def get_random_articles(n: int = 100) -> list[dict]:
    """Get a random sample of articles (for LLM validation)."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM articles
                ORDER BY RANDOM()
                LIMIT %s
            """, (n,))
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# KEYWORD CLASSIFICATION OPERATIONS
# =============================================================================

def save_keyword_classification(
    article_id: str,
    has_uncertainty: bool = False,
    has_reversal_terms: bool = False,
    has_implementation_terms: bool = False,
    has_upside_terms: bool = False,
    has_ira_mention: bool = False,
    has_obbba_mention: bool = False,
    matched_terms: dict = None,
):
    """Save local keyword classification for an article."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO keyword_classifications
                (article_id, has_uncertainty, has_reversal_terms,
                 has_implementation_terms, has_upside_terms,
                 has_ira_mention, has_obbba_mention, matched_terms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (article_id) DO UPDATE SET
                    has_uncertainty = EXCLUDED.has_uncertainty,
                    has_reversal_terms = EXCLUDED.has_reversal_terms,
                    has_implementation_terms = EXCLUDED.has_implementation_terms,
                    has_upside_terms = EXCLUDED.has_upside_terms,
                    has_ira_mention = EXCLUDED.has_ira_mention,
                    has_obbba_mention = EXCLUDED.has_obbba_mention,
                    matched_terms = EXCLUDED.matched_terms,
                    classified_at = NOW()
            """, (
                article_id,
                has_uncertainty,
                has_reversal_terms,
                has_implementation_terms,
                has_upside_terms,
                has_ira_mention,
                has_obbba_mention,
                json.dumps(matched_terms) if matched_terms else None,
            ))


def save_keyword_classifications_batch(classifications: list[dict]):
    """Save multiple keyword classifications efficiently."""
    if not classifications:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            values = [
                (
                    c["article_id"],
                    c.get("has_uncertainty", False),
                    c.get("has_reversal_terms", False),
                    c.get("has_implementation_terms", False),
                    c.get("has_upside_terms", False),
                    c.get("has_ira_mention", False),
                    c.get("has_obbba_mention", False),
                    json.dumps(c.get("matched_terms")) if c.get("matched_terms") else None,
                )
                for c in classifications
            ]
            execute_values(
                cur,
                """
                INSERT INTO keyword_classifications
                (article_id, has_uncertainty, has_reversal_terms,
                 has_implementation_terms, has_upside_terms,
                 has_ira_mention, has_obbba_mention, matched_terms)
                VALUES %s
                ON CONFLICT (article_id) DO UPDATE SET
                    has_uncertainty = EXCLUDED.has_uncertainty,
                    has_reversal_terms = EXCLUDED.has_reversal_terms,
                    has_implementation_terms = EXCLUDED.has_implementation_terms,
                    has_upside_terms = EXCLUDED.has_upside_terms,
                    has_ira_mention = EXCLUDED.has_ira_mention,
                    has_obbba_mention = EXCLUDED.has_obbba_mention,
                    matched_terms = EXCLUDED.matched_terms,
                    classified_at = NOW()
                """,
                values,
            )


def get_classification_counts_by_month() -> list[dict]:
    """
    Get classification counts aggregated by month.
    Returns counts for each category for index calculation.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    a.month,
                    COUNT(*) as total_articles,
                    COUNT(*) FILTER (WHERE kc.has_uncertainty) as uncertainty_count,
                    COUNT(*) FILTER (WHERE kc.has_uncertainty AND kc.has_reversal_terms)
                        as reversal_uncertainty_count,
                    COUNT(*) FILTER (WHERE kc.has_uncertainty AND kc.has_implementation_terms)
                        as implementation_uncertainty_count,
                    COUNT(*) FILTER (WHERE kc.has_uncertainty AND kc.has_upside_terms)
                        as upside_uncertainty_count,
                    COUNT(*) FILTER (WHERE kc.has_ira_mention) as ira_count,
                    COUNT(*) FILTER (WHERE kc.has_obbba_mention) as obbba_count
                FROM articles a
                LEFT JOIN keyword_classifications kc ON a.id = kc.article_id
                GROUP BY a.month
                ORDER BY a.month
            """)
            return [dict(row) for row in cur.fetchall()]


def get_classification_counts_by_outlet(month: str = None) -> list[dict]:
    """
    Get classification counts aggregated by outlet (source).
    Used for BBD-style outlet-level analysis.

    Args:
        month: Optional filter for specific month
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT
                    a.source as outlet,
                    a.month,
                    COUNT(*) as total_articles,
                    COUNT(*) FILTER (WHERE kc.has_uncertainty) as uncertainty_count,
                    COUNT(*) FILTER (WHERE kc.has_uncertainty AND kc.has_reversal_terms)
                        as reversal_uncertainty_count,
                    COUNT(*) FILTER (WHERE kc.has_uncertainty AND kc.has_implementation_terms)
                        as implementation_uncertainty_count
                FROM articles a
                LEFT JOIN keyword_classifications kc ON a.id = kc.article_id
            """
            if month:
                query += " WHERE a.month = %s"
                query += " GROUP BY a.source, a.month ORDER BY a.source, a.month"
                cur.execute(query, (month,))
            else:
                query += " GROUP BY a.source, a.month ORDER BY a.source, a.month"
                cur.execute(query)

            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# LLM CLASSIFICATION OPERATIONS
# =============================================================================

def save_llm_classification(
    article_id: str,
    is_climate_policy: bool,
    uncertainty_type: str,
    certainty_level: int,
    reasoning: str,
    model: str,
):
    """Save LLM validation result for an article."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO llm_classifications
                (article_id, is_climate_policy, uncertainty_type,
                 certainty_level, reasoning, model)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (article_id) DO UPDATE SET
                    is_climate_policy = EXCLUDED.is_climate_policy,
                    uncertainty_type = EXCLUDED.uncertainty_type,
                    certainty_level = EXCLUDED.certainty_level,
                    reasoning = EXCLUDED.reasoning,
                    model = EXCLUDED.model,
                    classified_at = NOW()
            """, (
                article_id,
                is_climate_policy,
                uncertainty_type,
                certainty_level,
                reasoning,
                model,
            ))


def get_llm_classification(article_id: str) -> Optional[dict]:
    """Get LLM classification for a specific article."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM llm_classifications WHERE article_id = %s
            """, (article_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_llm_classification_stats() -> dict:
    """Get statistics on LLM classifications for validation."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_classified,
                    COUNT(*) FILTER (WHERE is_climate_policy) as climate_policy_count,
                    COUNT(*) FILTER (WHERE uncertainty_type = 'reversal')
                        as reversal_count,
                    COUNT(*) FILTER (WHERE uncertainty_type = 'implementation')
                        as implementation_count,
                    COUNT(*) FILTER (WHERE uncertainty_type = 'none')
                        as no_uncertainty_count,
                    AVG(certainty_level) as avg_certainty
                FROM llm_classifications
            """)
            return dict(cur.fetchone())


def get_unclassified_articles_for_llm(limit: int = 100) -> list[dict]:
    """Get articles that haven't been LLM-classified yet."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT a.*
                FROM articles a
                LEFT JOIN llm_classifications lc ON a.id = lc.article_id
                WHERE lc.article_id IS NULL
                ORDER BY RANDOM()
                LIMIT %s
            """, (limit,))
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# INDEX VALUE OPERATIONS
# =============================================================================

def save_index_value(
    month: str,
    index_type: str,
    denominator: int,
    numerator: int,
    raw_ratio: float = None,
    normalized: float = None,
    outlet: str = "",
    scaled: bool = False,
    dedup_strategy: str = "none",
):
    """Save computed index value."""
    if raw_ratio is None and denominator > 0:
        raw_ratio = numerator / denominator

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO index_values
                (month, index_type, outlet, denominator, numerator,
                 raw_ratio, normalized, scaled, dedup_strategy)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (month, index_type, outlet, scaled, dedup_strategy)
                DO UPDATE SET
                    denominator = EXCLUDED.denominator,
                    numerator = EXCLUDED.numerator,
                    raw_ratio = EXCLUDED.raw_ratio,
                    normalized = EXCLUDED.normalized,
                    computed_at = NOW()
            """, (
                month, index_type, outlet, denominator, numerator,
                raw_ratio, normalized, scaled, dedup_strategy,
            ))


def get_index_values(
    index_type: str,
    outlet: str = "",
    scaled: bool = False,
    dedup_strategy: str = "none",
) -> list[dict]:
    """Get all index values for a specific type and configuration."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM index_values
                WHERE index_type = %s
                  AND outlet = %s
                  AND scaled = %s
                  AND dedup_strategy = %s
                ORDER BY month
            """, (index_type, outlet, scaled, dedup_strategy))
            return [dict(row) for row in cur.fetchall()]


def get_all_index_types() -> list[str]:
    """Get list of all computed index types."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT index_type FROM index_values ORDER BY index_type")
            return [row[0] for row in cur.fetchall()]


# =============================================================================
# COLLECTION PROGRESS OPERATIONS
# =============================================================================

def save_collection_progress(month: str, articles_fetched: int):
    """Mark a month as collected with article count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO collection_progress (month, articles_fetched, completed_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (month) DO UPDATE SET
                    articles_fetched = EXCLUDED.articles_fetched,
                    completed_at = NOW()
            """, (month, articles_fetched))


def get_collection_progress() -> dict[str, dict]:
    """Get collection progress for all months."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM collection_progress ORDER BY month
            """)
            return {
                row["month"]: {
                    "articles_fetched": row["articles_fetched"],
                    "completed_at": row["completed_at"],
                }
                for row in cur.fetchall()
            }


def get_completed_months() -> set[str]:
    """Get set of months that have been collected."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT month FROM collection_progress WHERE completed_at IS NOT NULL")
            return {row[0] for row in cur.fetchall()}


def is_month_complete(month: str) -> bool:
    """Check if a specific month has been collected."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM collection_progress
                WHERE month = %s AND completed_at IS NOT NULL
            """, (month,))
            return cur.fetchone() is not None


def mark_month_complete(month: str, articles_fetched: int):
    """
    Mark a month as complete with article count.
    Alias for save_collection_progress for clearer semantics.

    Args:
        month: Month in YYYY-MM format
        articles_fetched: Number of articles fetched and stored
    """
    save_collection_progress(month, articles_fetched)


# =============================================================================
# EXPORT OPERATIONS
# =============================================================================

def export_index_to_csv(
    output_path: str,
    index_types: list[str] = None,
    include_all_configs: bool = False,
) -> int:
    """
    Export index values to CSV file.

    Args:
        output_path: Path to output CSV file
        index_types: List of index types to include (None = all)
        include_all_configs: If True, include all outlet/scaled/dedup combinations

    Returns:
        Number of rows exported
    """
    import csv
    from pathlib import Path

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM index_values"
            conditions = []
            params = []

            if index_types:
                conditions.append("index_type = ANY(%s)")
                params.append(index_types)

            if not include_all_configs:
                conditions.append("outlet = ''")
                conditions.append("scaled = FALSE")
                conditions.append("dedup_strategy = 'none'")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY month, index_type"

            cur.execute(query, params or None)
            rows = cur.fetchall()

    if not rows:
        raise ValueError("No index values to export")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "month", "index_type", "outlet", "denominator", "numerator",
        "raw_ratio", "normalized", "scaled", "dedup_strategy",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    return len(rows)


# =============================================================================
# MONTHLY COUNTS OPERATIONS (Count-based collection)
# =============================================================================

def save_monthly_counts(counts: dict):
    """
    Save monthly counts from count-based collection.

    Args:
        counts: Dict with month, denominator, numerator_cpu, numerator_impl, numerator_reversal
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO monthly_counts
                (month, denominator, numerator_cpu, numerator_impl, numerator_reversal)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (month) DO UPDATE SET
                    denominator = EXCLUDED.denominator,
                    numerator_cpu = EXCLUDED.numerator_cpu,
                    numerator_impl = EXCLUDED.numerator_impl,
                    numerator_reversal = EXCLUDED.numerator_reversal,
                    collected_at = NOW()
            """, (
                counts["month"],
                counts["denominator"],
                counts["numerator_cpu"],
                counts["numerator_impl"],
                counts["numerator_reversal"],
            ))


def get_monthly_counts() -> list[dict]:
    """Get all monthly counts for index calculation."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM monthly_counts ORDER BY month
            """)
            return [dict(row) for row in cur.fetchall()]


def get_completed_count_months() -> set[str]:
    """Get set of months that have count data."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT month FROM monthly_counts")
            return {row[0] for row in cur.fetchall()}
