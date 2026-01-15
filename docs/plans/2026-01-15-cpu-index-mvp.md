# CPU Index MVP Implementation Plan

## Execution Complete - 2026-01-15

**All 10 Tasks:** Completed and verified

**Final Verification:**
- All modules import successfully
- Interactive menu runs and exits cleanly
- Git commits: 10 commits (752946d to c7bea08)

**Files Created:**
- `requirements.txt`, `.env.example`, `.gitignore`
- `config.py`, `db.py`, `api.py`, `collector.py`, `classifier.py`, `indexer.py`, `run.py`
- `README.md` (updated)

---

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modular, researcher-friendly tool to construct a Climate Policy Uncertainty (CPU) index from LexisNexis news articles.

**Architecture:** Six Python modules with clear separation: config (settings), db (SQLite storage), api (LexisNexis client), collector (data pipeline), classifier (GPT-5 Nano via Vercel AI SDK), indexer (CPU calculation). Single interactive entry point (run.py) with menu-driven interface for non-technical users.

**Tech Stack:** Python 3.11+, SQLite, requests, python-dotenv, ai-sdk-python (Vercel AI SDK), pydantic

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Modify: existing `boilerplate.py` (keep as reference)

**Step 1: Create requirements.txt**

```txt
python-dotenv>=1.0.0
requests>=2.31.0
ai-sdk-python>=0.1.0
pydantic>=2.0.0
```

**Step 2: Create .env.example**

```bash
# LexisNexis API Credentials (get from Stanford library)
clientid=your_client_id_here
clientsecret=your_client_secret_here

# OpenAI API Key (for GPT-5 Nano classification)
OPENAI_API_KEY=your_openai_key_here

# Auto-populated by the tool (do not edit manually)
# lntoken=
# lnexpire=
```

**Step 3: Create .gitignore**

```
# Environment
.env
__pycache__/
*.pyc

# Data (too large for git)
data/cpu.db
data/exports/*.csv

# IDE
.vscode/
.idea/
```

**Step 4: Create data directories**

Run: `mkdir -p data/exports`

**Step 5: Commit**

```bash
git add requirements.txt .env.example .gitignore
git commit -m "chore: project setup with dependencies and gitignore"
```

---

## Task 2: Configuration Module

**Files:**
- Create: `config.py`

**Step 1: Create config.py with researcher-friendly documentation**

```python
"""
Configuration for Climate Policy Uncertainty (CPU) Index Builder

HOW TO USE THIS FILE:
--------------------
1. Review the keywords below - do they match your research goals?
2. Set your date range (START_DATE, END_DATE)
3. After getting API access, fill in SOURCE_IDS (see instructions below)

You generally don't need to change anything else.
"""

# =============================================================================
# DATE RANGE
# =============================================================================
# What time period should we analyze?
# Format: "YYYY-MM-DD"

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"  # Set to today's date or leave as future date


# =============================================================================
# NEWS SOURCES
# =============================================================================
# These are the approved high-credibility sources for the index.
#
# IMPORTANT: You need to look up the Source IDs using the LexisNexis WSAPI:
# 1. Go to https://solutions.nexis.com/wsapi
# 2. Login with your Stanford credentials
# 3. Go to Ref. Materials > Sources
# 4. Search for each source and copy its ID
#
# Leave as None if a source is not available in LexisNexis.

SOURCE_IDS = {
    "Financial Times": None,          # Example: "MTA2OTUwNQ"
    "Wall Street Journal": None,
    "New York Times": None,
    "Washington Post": None,
    "Reuters": None,
    "Bloomberg": None,
    "Politico": None,
    "The Economist": None,
}


# =============================================================================
# CLIMATE & ENERGY KEYWORDS
# =============================================================================
# Articles must contain AT LEAST ONE term from this list.
# These identify articles about climate/energy topics.

CLIMATE_TERMS = [
    "climate",
    "climate change",
    "clean energy",
    "renewable",
    "renewable energy",
    "decarbonization",
    "carbon",
    "greenhouse gas",
    "net zero",
    "EV",
    "electric vehicle",
    "hydrogen",
    "solar",
    "wind",
    "battery",
    "grid",
]


# =============================================================================
# POLICY KEYWORDS
# =============================================================================
# Articles must ALSO contain AT LEAST ONE term from this list.
# These identify articles about policy/government action.

POLICY_TERMS = [
    "policy",
    "regulation",
    "tax credit",
    "subsidy",
    "grant",
    "incentive",
    "Congress",
    "DOE",
    "Department of Energy",
    "EPA",
    "Treasury",
    "IRS",
    "White House",
    "legislation",
    "Inflation Reduction Act",
    "IRA",
]


# =============================================================================
# UNCERTAINTY KEYWORDS
# =============================================================================
# For the NUMERATOR: articles that express uncertainty about policy.
# These words suggest doubt, delay, or instability.

UNCERTAINTY_TERMS = [
    "uncertain",
    "uncertainty",
    "unclear",
    "unpredictable",
    "delay",
    "delayed",
    "freeze",
    "frozen",
    "rollback",
    "repeal",
    "reversal",
    "litigation",
    "lawsuit",
    "challenge",
    "suspend",
    "halt",
    "block",
]


# =============================================================================
# LLM VALIDATION SETTINGS
# =============================================================================
# We use GPT-5 Nano to spot-check if our keywords are accurate.
# This is very cheap (~$0.01 per 100 articles).

LLM_MODEL = "gpt-5-nano"      # Cheapest OpenAI model, great for classification
LLM_SAMPLE_SIZE = 100         # Articles to classify per validation run
LLM_TEMPERATURE = 0.0         # Deterministic output (no randomness)


# =============================================================================
# API SETTINGS (Advanced - usually don't need to change)
# =============================================================================

LEXISNEXIS_BASE_URL = "https://services-api.lexisnexis.com/v1"
MAX_RESULTS_PER_QUERY = 50    # LexisNexis limit
REQUEST_DELAY_SECONDS = 0.5   # Pause between API calls (be nice to shared quota)


# =============================================================================
# FILE PATHS (Advanced - usually don't need to change)
# =============================================================================

DB_PATH = "data/cpu.db"
EXPORT_DIR = "data/exports"
```

**Step 2: Verify config loads without errors**

Run: `python -c "import config; print('Config loaded successfully')"`
Expected: `Config loaded successfully`

**Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add config.py with researcher-friendly documentation"
```

---

## Task 3: Database Module

**Files:**
- Create: `db.py`

**Step 1: Create db.py with SQLite schema and operations**

```python
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
```

**Step 2: Test database initialization**

Run: `python -c "import db; db.init_db(); print('Database initialized')"`
Expected: `Database initialized`

**Step 3: Verify database file created**

Run: `ls -la data/`
Expected: Shows `cpu.db` file

**Step 4: Commit**

```bash
git add db.py
git commit -m "feat: add db.py with SQLite schema and operations"
```

---

## Task 4: API Module

**Files:**
- Create: `api.py`

**Step 1: Create api.py building on boilerplate**

```python
"""
LexisNexis API client for CPU Index Builder

Handles:
- Authentication (token management)
- Query building (keywords + date filters)
- Pagination (fetching all results)
- Rate limiting (protecting shared quota)
"""

import os
import time
import requests
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv, find_dotenv, set_key

import config

load_dotenv()


# =============================================================================
# AUTHENTICATION (from boilerplate.py)
# =============================================================================

def get_token() -> str:
    """
    Get a valid API token. Refreshes automatically if expired.
    Returns empty string if credentials not configured.
    """
    if "clientid" not in os.environ or "clientsecret" not in os.environ:
        print("Error: LexisNexis credentials not found in .env file.")
        print("Please add 'clientid' and 'clientsecret' to your .env file.")
        return ""

    clientid = os.environ["clientid"]
    clientsecret = os.environ["clientsecret"]

    # Check if we have a valid cached token
    if "lnexpire" in os.environ:
        token_expire = int(os.environ.get("lnexpire", 0))
        if token_expire > int(time.time()):
            return os.environ.get("lntoken", "")

    # Need to get a new token
    return _refresh_token(clientid, clientsecret)


def _refresh_token(clientid: str, clientsecret: str) -> str:
    """Request a new token from LexisNexis auth API."""
    url = "https://auth-api.lexisnexis.com/oauth/v2/token"
    data = "grant_type=client_credentials&scope=http%3a%2f%2foauth.lexisnexis.com%2fall"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(
        url, data=data, headers=headers,
        auth=HTTPBasicAuth(clientid, clientsecret)
    )

    if response.status_code != 200:
        print(f"Error getting token: {response.status_code}")
        print(response.text)
        return ""

    results = response.json()
    token = results["access_token"]
    expires_in = results["expires_in"]

    # Save to .env file
    dotenv_file = find_dotenv()
    set_key(dotenv_file, "lntoken", token)
    set_key(dotenv_file, "lnexpire", str(int(time.time()) + expires_in))

    # Reload environment
    load_dotenv(override=True)

    return token


# =============================================================================
# QUERY BUILDING
# =============================================================================

def build_search_query(
    climate_terms: list,
    policy_terms: list,
    uncertainty_terms: list = None,
    start_date: str = None,
    end_date: str = None,
) -> str:
    """
    Build a LexisNexis search query string.

    Args:
        climate_terms: List of climate/energy keywords (require at least 1)
        policy_terms: List of policy keywords (require at least 1)
        uncertainty_terms: Optional list of uncertainty keywords (for numerator)
        start_date: Start of date range (YYYY-MM-DD)
        end_date: End of date range (YYYY-MM-DD)

    Returns:
        URL-encoded search query string
    """
    # Build climate terms: (climate OR "clean+energy" OR renewable)
    climate_part = " OR ".join([_format_term(t) for t in climate_terms])

    # Build policy terms: (policy OR regulation OR Congress)
    policy_part = " OR ".join([_format_term(t) for t in policy_terms])

    # Combine: (climate terms) AND (policy terms)
    query = f"({climate_part}) AND ({policy_part})"

    # Add uncertainty terms if provided (for numerator query)
    if uncertainty_terms:
        uncertainty_part = " OR ".join([_format_term(t) for t in uncertainty_terms])
        query += f" AND ({uncertainty_part})"

    # Add date range
    if start_date:
        query += f" AND Date ge {start_date}"
    if end_date:
        query += f" AND Date le {end_date}"

    return query


def _format_term(term: str) -> str:
    """Format a search term for LexisNexis query."""
    if " " in term:
        # Multi-word phrase: use quotes and plus signs
        return '"' + term.replace(" ", "+") + '"'
    return term


def build_month_dates(year: int, month: int) -> tuple:
    """Get start and end dates for a given month."""
    import calendar

    start_date = f"{year}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    end_date = f"{year}-{month:02d}-{last_day:02d}"

    return start_date, end_date


# =============================================================================
# API REQUESTS
# =============================================================================

def fetch_count(query: str, dry_run: bool = False) -> int:
    """
    Get total article count for a query (uses minimal quota).

    Args:
        query: Search query string
        dry_run: If True, return fake count without API call

    Returns:
        Total number of matching articles
    """
    if dry_run:
        return 100  # Fake count for testing

    token = get_token()
    if not token:
        raise RuntimeError("No API token available")

    # Build URL with $top=1 to minimize data transfer
    base_url = f"{config.LEXISNEXIS_BASE_URL}/News"
    params = {
        "$search": query,
        "$top": 1,
        "$orderby": "Date desc",
    }
    url = f"{base_url}?{urlencode(params)}"

    headers = {"Authorization": f"Bearer {token}"}

    time.sleep(config.REQUEST_DELAY_SECONDS)
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    result = response.json()

    # Total count is in @odata.count
    return result.get("@odata.count", 0)


def fetch_metadata(
    query: str,
    max_results: int = None,
    dry_run: bool = False,
) -> list:
    """
    Fetch article metadata (not full text).
    Handles pagination automatically.

    Args:
        query: Search query string
        max_results: Maximum articles to fetch (None = all)
        dry_run: If True, return fake data without API call

    Returns:
        List of article dicts with metadata
    """
    if dry_run:
        return [{"id": f"fake_{i}", "title": f"Fake Article {i}"} for i in range(10)]

    token = get_token()
    if not token:
        raise RuntimeError("No API token available")

    articles = []
    skip = 0

    while True:
        # Build URL
        base_url = f"{config.LEXISNEXIS_BASE_URL}/News"
        params = {
            "$search": query,
            "$top": config.MAX_RESULTS_PER_QUERY,
            "$skip": skip,
            "$orderby": "Date desc",
            "$expand": "Source",  # Include source metadata
        }
        url = f"{base_url}?{urlencode(params)}"

        headers = {"Authorization": f"Bearer {token}"}

        time.sleep(config.REQUEST_DELAY_SECONDS)
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")

        result = response.json()
        batch = result.get("value", [])

        if not batch:
            break

        # Extract relevant fields
        for item in batch:
            articles.append({
                "id": item.get("ResultId", ""),
                "title": item.get("Title", ""),
                "date": item.get("Date", ""),
                "source": item.get("Source", {}).get("Name", ""),
                "snippet": item.get("Overview", ""),
            })

        # Check if we have enough
        if max_results and len(articles) >= max_results:
            articles = articles[:max_results]
            break

        # Check if there are more results
        skip += config.MAX_RESULTS_PER_QUERY
        total = result.get("@odata.count", 0)
        if skip >= total:
            break

    return articles


def fetch_full_text(article_ids: list, dry_run: bool = False) -> list:
    """
    Fetch full text for specific articles via BatchNews endpoint.

    WARNING: This uses the stricter BatchNews quota.
    Use sparingly - only for LLM validation samples.

    Args:
        article_ids: List of article IDs to fetch
        dry_run: If True, return fake data

    Returns:
        List of articles with full_text field
    """
    if dry_run:
        return [{"id": aid, "full_text": "Fake article text..."} for aid in article_ids]

    # TODO: Implement BatchNews full text retrieval
    # For MVP, we can use snippets for LLM classification
    raise NotImplementedError("Full text retrieval not yet implemented - use snippets")
```

**Step 2: Test API token retrieval (will fail without credentials)**

Run: `python -c "import api; print('API module loaded')"`
Expected: `API module loaded`

**Step 3: Test query building**

Run: `python -c "import api; q = api.build_search_query(['climate'], ['policy'], start_date='2021-01-01', end_date='2021-01-31'); print(q)"`
Expected: `(climate) AND (policy) AND Date ge 2021-01-01 AND Date le 2021-01-31`

**Step 4: Commit**

```bash
git add api.py
git commit -m "feat: add api.py with LexisNexis client and query building"
```

---

## Task 5: Collector Module

**Files:**
- Create: `collector.py`

**Step 1: Create collector.py with data pipeline**

```python
"""
Data collection pipeline for CPU Index Builder

Orchestrates monthly data collection with:
- Automatic resume from where you left off
- Progress tracking
- API usage estimation
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta

import config
import db
import api


def generate_months(start_date: str, end_date: str) -> list:
    """Generate list of months between start and end dates."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    months = []
    current = start.replace(day=1)

    while current <= end:
        months.append(current.strftime("%Y-%m"))
        current += relativedelta(months=1)

    return months


def get_incomplete_months() -> list:
    """Get months that haven't been collected yet."""
    all_months = generate_months(config.START_DATE, config.END_DATE)
    completed = db.get_completed_months()
    return [m for m in all_months if m not in completed]


def estimate_api_usage() -> dict:
    """
    Estimate API calls needed before running collection.

    Returns dict with:
    - months: number of months to collect
    - searches: estimated API searches
    - percent_quota: percent of annual quota (24,000)
    """
    incomplete = get_incomplete_months()
    num_months = len(incomplete)

    # 2 searches per month (denominator + numerator)
    searches = num_months * 2

    return {
        "months": num_months,
        "searches": searches,
        "percent_quota": searches / 24000,  # Stanford annual limit
        "incomplete_months": incomplete,
    }


def collect_month(year: int, month: int, dry_run: bool = False) -> dict:
    """
    Collect data for a single month.

    Returns dict with:
    - month: "YYYY-MM"
    - denominator: count of all climate-policy articles
    - numerator: count of articles with uncertainty language
    - raw_ratio: numerator / denominator
    - status: "complete", "skipped", or "error"
    """
    month_str = f"{year:04d}-{month:02d}"

    # Check if already complete
    if month_str in db.get_completed_months():
        return {
            "month": month_str,
            "status": "skipped",
            "reason": "already complete",
        }

    try:
        # Get date range for this month
        start_date, end_date = api.build_month_dates(year, month)

        # Build denominator query (climate + policy)
        denom_query = api.build_search_query(
            climate_terms=config.CLIMATE_TERMS,
            policy_terms=config.POLICY_TERMS,
            start_date=start_date,
            end_date=end_date,
        )

        # Fetch denominator count
        denominator = api.fetch_count(denom_query, dry_run=dry_run)

        # Build numerator query (add uncertainty)
        numer_query = api.build_search_query(
            climate_terms=config.CLIMATE_TERMS,
            policy_terms=config.POLICY_TERMS,
            uncertainty_terms=config.UNCERTAINTY_TERMS,
            start_date=start_date,
            end_date=end_date,
        )

        # Fetch numerator count
        numerator = api.fetch_count(numer_query, dry_run=dry_run)

        # Calculate ratio (handle division by zero)
        raw_ratio = numerator / denominator if denominator > 0 else 0.0

        # Save to database
        if not dry_run:
            db.save_month_count(month_str, "denominator", denominator)
            db.save_month_count(month_str, "numerator", numerator)

        return {
            "month": month_str,
            "denominator": denominator,
            "numerator": numerator,
            "raw_ratio": raw_ratio,
            "status": "complete",
        }

    except Exception as e:
        return {
            "month": month_str,
            "status": "error",
            "error": str(e),
        }


def collect_all(dry_run: bool = False, progress_callback=None) -> dict:
    """
    Collect data for all incomplete months.

    Args:
        dry_run: If True, simulate without API calls
        progress_callback: Optional function(current, total, month) for progress updates

    Returns:
        Summary dict with totals and results
    """
    incomplete = get_incomplete_months()
    total = len(incomplete)

    results = []
    errors = []

    for i, month_str in enumerate(incomplete):
        year, month = int(month_str[:4]), int(month_str[5:7])

        if progress_callback:
            progress_callback(i + 1, total, month_str)

        result = collect_month(year, month, dry_run=dry_run)
        results.append(result)

        if result["status"] == "error":
            errors.append(result)

    return {
        "months_processed": len(results),
        "months_complete": len([r for r in results if r["status"] == "complete"]),
        "months_skipped": len([r for r in results if r["status"] == "skipped"]),
        "months_error": len(errors),
        "errors": errors,
        "results": results,
    }


def get_collection_status() -> dict:
    """Get current collection status for display."""
    all_months = generate_months(config.START_DATE, config.END_DATE)
    completed = db.get_completed_months()
    incomplete = get_incomplete_months()

    progress = db.get_all_progress()

    return {
        "date_range": f"{config.START_DATE} to {config.END_DATE}",
        "total_months": len(all_months),
        "completed_months": len(completed),
        "incomplete_months": len(incomplete),
        "percent_complete": len(completed) / len(all_months) if all_months else 0,
        "next_month": incomplete[0] if incomplete else None,
        "progress_records": progress,
    }
```

**Step 2: Test collector module**

Run: `python -c "import db; db.init_db(); import collector; print(collector.estimate_api_usage())"`
Expected: Shows dict with months, searches, percent_quota

**Step 3: Commit**

```bash
git add collector.py
git commit -m "feat: add collector.py with data pipeline and resume support"
```

---

## Task 6: Classifier Module

**Files:**
- Create: `classifier.py`

**Step 1: Create classifier.py with GPT-5 Nano integration**

```python
"""
LLM Classification for CPU Index Builder

Uses GPT-5 Nano via Vercel AI SDK to validate keyword accuracy.
This is a SAMPLING tool - we don't classify every article.

Cost: ~$0.01 per 100 articles (very cheap!)
"""

import os
import random
from typing import Optional

from pydantic import BaseModel

import config
import db

# Check if AI SDK is available
try:
    from ai_sdk import generate_object, openai
    AI_SDK_AVAILABLE = True
except ImportError:
    AI_SDK_AVAILABLE = False
    print("Warning: ai-sdk-python not installed. LLM classification disabled.")


# =============================================================================
# STRUCTURED OUTPUT SCHEMA
# =============================================================================

class ArticleClassification(BaseModel):
    """Pydantic model for LLM response - ensures valid JSON."""
    is_climate_policy: bool      # Is this about US climate/energy policy?
    has_uncertainty: bool        # Does it express policy uncertainty?
    reasoning: str               # Brief explanation
    confidence: str              # "high", "medium", "low"


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_article(article: dict) -> Optional[ArticleClassification]:
    """
    Classify a single article using GPT-5 Nano.

    Returns ArticleClassification or None if AI SDK not available.
    """
    if not AI_SDK_AVAILABLE:
        return None

    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Cannot classify.")
        return None

    model = openai(config.LLM_MODEL)

    result = generate_object(
        model=model,
        schema=ArticleClassification,
        prompt=_build_prompt(article),
        temperature=config.LLM_TEMPERATURE,
    )

    return result.object


def _build_prompt(article: dict) -> str:
    """Build classification prompt."""
    text = article.get("full_text") or article.get("snippet") or ""

    return f"""
You are classifying news articles for a Climate Policy Uncertainty Index.

ARTICLE:
- Title: {article.get('title', 'Unknown')}
- Date: {article.get('date', 'Unknown')}
- Source: {article.get('source', 'Unknown')}
- Text: {text[:2000]}  # Truncate to save tokens

TASK: Answer these questions:

1. Is this article primarily about U.S. climate or clean energy POLICY?
   - YES if about: legislation, regulation, tax credits, subsidies, government programs
   - NO if about: weather, climate science only, international policy, corporate actions without policy context

2. If YES to #1: Does it express UNCERTAINTY about policy?
   - YES if mentions: delays, legal challenges, rollbacks, unclear guidance, political disputes, pending decisions
   - NO if policy seems stable/certain

Be concise in your reasoning (1-2 sentences max).
Set confidence to "high" if clear-cut, "medium" if borderline, "low" if unsure.
"""


def classify_sample(
    month: str = None,
    sample_size: int = None,
    dry_run: bool = False,
) -> dict:
    """
    Classify a random sample of articles.

    Args:
        month: Specific month to sample from (None = all months)
        sample_size: Number of articles (default from config)
        dry_run: If True, return fake results

    Returns:
        Summary dict with classification results
    """
    if sample_size is None:
        sample_size = config.LLM_SAMPLE_SIZE

    if dry_run:
        return {
            "sample_size": sample_size,
            "classified": sample_size,
            "is_climate_policy_yes": int(sample_size * 0.85),
            "has_uncertainty_yes": int(sample_size * 0.20),
            "dry_run": True,
        }

    # TODO: Implement actual sampling from database
    # For now, return placeholder
    return {
        "sample_size": sample_size,
        "status": "not_implemented",
        "message": "Full implementation requires articles in database. Run collection first.",
    }


def estimate_classification_cost(num_articles: int) -> dict:
    """
    Estimate cost before running classification.

    GPT-5 Nano pricing:
    - Input: $0.05 per 1M tokens
    - Output: $0.40 per 1M tokens
    """
    # Assumptions
    input_tokens_per_article = 500   # Prompt + article text
    output_tokens_per_article = 100  # Classification response

    total_input = num_articles * input_tokens_per_article
    total_output = num_articles * output_tokens_per_article

    input_cost = (total_input / 1_000_000) * 0.05
    output_cost = (total_output / 1_000_000) * 0.40
    total_cost = input_cost + output_cost

    return {
        "articles": num_articles,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_cost_usd": round(total_cost, 4),
        "cost_per_100_articles": round(total_cost / num_articles * 100, 4),
    }


def get_validation_status() -> dict:
    """Get current validation status."""
    # TODO: Query database for classification stats
    return {
        "total_classified": 0,
        "ai_sdk_available": AI_SDK_AVAILABLE,
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
    }
```

**Step 2: Test classifier module loads**

Run: `python -c "import classifier; print(classifier.estimate_classification_cost(100))"`
Expected: Shows dict with cost estimate

**Step 3: Commit**

```bash
git add classifier.py
git commit -m "feat: add classifier.py with GPT-5 Nano via AI SDK"
```

---

## Task 7: Indexer Module

**Files:**
- Create: `indexer.py`

**Step 1: Create indexer.py with CPU calculation**

```python
"""
CPU Index calculation for CPU Index Builder

Calculates the Climate Policy Uncertainty index following
Baker, Bloom & Davis (2016) methodology:

CPU_t = (uncertainty articles) / (all climate-policy articles)

Normalized so mean = 100 for interpretability.
"""

import statistics
from typing import Optional

import db


def calculate_raw_index() -> list:
    """
    Calculate raw CPU ratio for each month from database.

    Returns list of dicts with month, denominator, numerator, raw_ratio
    """
    progress = db.get_all_progress()

    # Group by month
    months_data = {}
    for record in progress:
        month = record["month"]
        if month not in months_data:
            months_data[month] = {}
        months_data[month][record["query_type"]] = record["count"]

    # Calculate ratios
    results = []
    for month in sorted(months_data.keys()):
        data = months_data[month]
        if "denominator" in data and "numerator" in data:
            denom = data["denominator"]
            numer = data["numerator"]
            ratio = numer / denom if denom > 0 else 0.0

            results.append({
                "month": month,
                "denominator": denom,
                "numerator": numer,
                "raw_ratio": ratio,
            })

    return results


def normalize_index(
    raw_values: list,
    base_start: str = None,
    base_end: str = None,
) -> list:
    """
    Normalize index so mean = 100 over base period.

    Args:
        raw_values: List from calculate_raw_index()
        base_start: Start of base period (None = use all)
        base_end: End of base period (None = use all)

    Returns:
        Same list with added 'normalized' field
    """
    if not raw_values:
        return []

    # Filter to base period if specified
    if base_start and base_end:
        base_values = [
            v for v in raw_values
            if base_start <= v["month"] <= base_end
        ]
    else:
        base_values = raw_values

    # Calculate mean of raw ratios in base period
    if not base_values:
        mean_ratio = statistics.mean([v["raw_ratio"] for v in raw_values])
    else:
        mean_ratio = statistics.mean([v["raw_ratio"] for v in base_values])

    # Normalize: (raw / mean) * 100
    for v in raw_values:
        if mean_ratio > 0:
            v["normalized"] = (v["raw_ratio"] / mean_ratio) * 100
        else:
            v["normalized"] = 0.0

    return raw_values


def build_index(base_start: str = None, base_end: str = None) -> dict:
    """
    Build complete CPU index from database.

    Returns dict with metadata and series.
    """
    raw = calculate_raw_index()

    if not raw:
        return {
            "status": "error",
            "message": "No data found. Run data collection first.",
        }

    normalized = normalize_index(raw, base_start, base_end)

    # Calculate statistics
    ratios = [v["raw_ratio"] for v in normalized]
    norm_values = [v["normalized"] for v in normalized]

    # Save to database
    for v in normalized:
        db.save_index_value(
            month=v["month"],
            denominator=v["denominator"],
            numerator=v["numerator"],
            raw_ratio=v["raw_ratio"],
            normalized=v["normalized"],
        )

    return {
        "status": "success",
        "metadata": {
            "period": f"{normalized[0]['month']} to {normalized[-1]['month']}",
            "base_period": f"{base_start or 'all'} to {base_end or 'all'}",
            "num_months": len(normalized),
            "mean_raw_ratio": statistics.mean(ratios),
            "std_raw_ratio": statistics.stdev(ratios) if len(ratios) > 1 else 0,
            "mean_normalized": statistics.mean(norm_values),  # Should be ~100
        },
        "series": normalized,
    }


def validate_against_events() -> dict:
    """
    Sanity check: Compare index to known policy events.
    """
    # Key events to check
    events = [
        {"date": "2022-07", "event": "Manchin withdraws support", "expected": "spike"},
        {"date": "2022-08", "event": "IRA signed into law", "expected": "drop"},
        {"date": "2024-11", "event": "Trump wins election", "expected": "spike"},
        {"date": "2025-01", "event": "Trump executive orders", "expected": "spike"},
    ]

    index_values = db.get_all_index_values()
    if not index_values:
        return {"status": "error", "message": "No index values. Build index first."}

    # Create lookup
    value_by_month = {v["month"]: v for v in index_values}

    results = []
    for event in events:
        month = event["date"]
        if month not in value_by_month:
            results.append({**event, "result": "NO DATA"})
            continue

        current = value_by_month[month]["normalized"]

        # Get prior month
        prior_month = _get_prior_month(month)
        prior = value_by_month.get(prior_month, {}).get("normalized", current)

        change = ((current - prior) / prior * 100) if prior > 0 else 0

        # Check if matches expectation
        if event["expected"] == "spike":
            passed = change > 10  # At least 10% increase
        else:  # drop
            passed = change < -10  # At least 10% decrease

        results.append({
            **event,
            "actual_cpu": round(current, 1),
            "prior_cpu": round(prior, 1),
            "change_percent": round(change, 1),
            "result": "PASS" if passed else "FAIL",
        })

    passed = sum(1 for r in results if r.get("result") == "PASS")
    total = len([r for r in results if r.get("result") != "NO DATA"])

    return {
        "events": results,
        "summary": f"{passed}/{total} events validated",
    }


def _get_prior_month(month: str) -> str:
    """Get the month before the given month."""
    year, mon = int(month[:4]), int(month[5:7])
    if mon == 1:
        return f"{year-1}-12"
    return f"{year}-{mon-1:02d}"


def get_index_summary() -> dict:
    """Get summary statistics for display."""
    values = db.get_all_index_values()

    if not values:
        return {"status": "empty", "message": "No index calculated yet."}

    norm = [v["normalized"] for v in values if v["normalized"]]

    if not norm:
        return {"status": "incomplete", "message": "Index not normalized yet."}

    # Find peaks and troughs
    sorted_by_value = sorted(values, key=lambda x: x["normalized"] or 0, reverse=True)

    return {
        "status": "ready",
        "period": f"{values[0]['month']} to {values[-1]['month']}",
        "num_months": len(values),
        "mean": round(statistics.mean(norm), 1),
        "std": round(statistics.stdev(norm), 1) if len(norm) > 1 else 0,
        "min": round(min(norm), 1),
        "max": round(max(norm), 1),
        "top_3_peaks": [
            {"month": v["month"], "cpu": round(v["normalized"], 1)}
            for v in sorted_by_value[:3]
        ],
        "top_3_troughs": [
            {"month": v["month"], "cpu": round(v["normalized"], 1)}
            for v in sorted_by_value[-3:]
        ],
    }
```

**Step 2: Test indexer module loads**

Run: `python -c "import indexer; print('Indexer loaded')"`
Expected: `Indexer loaded`

**Step 3: Commit**

```bash
git add indexer.py
git commit -m "feat: add indexer.py with CPU calculation and event validation"
```

---

## Task 8: Interactive Runner

**Files:**
- Create: `run.py`

**Step 1: Create run.py with interactive menu**

```python
#!/usr/bin/env python3
"""
CPU Index Builder - Interactive Runner

Just run: python run.py

A menu-driven interface for non-technical researchers.
"""

import sys
import os

# Ensure we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
import config
import collector
import classifier
import indexer


def main():
    """Main entry point."""
    # Initialize database
    db.init_db()

    print_header()

    while True:
        print_menu()
        choice = input("\nEnter choice (1-7): ").strip()

        if choice == "1":
            show_status()
        elif choice == "2":
            estimate_usage()
        elif choice == "3":
            collect_data()
        elif choice == "4":
            validate_sample()
        elif choice == "5":
            build_index()
        elif choice == "6":
            export_csv()
        elif choice == "7":
            print("\nGoodbye!\n")
            break
        else:
            print("\n[!] Invalid choice. Please enter 1-7.\n")

        input("\nPress Enter to continue...")
        print("\n" + "="*60 + "\n")


def print_header():
    """Print welcome header."""
    print("""
+===========================================================+
|     Climate Policy Uncertainty (CPU) Index Builder        |
|                                                           |
|  This tool collects news articles from LexisNexis and     |
|  builds a monthly uncertainty index for climate policy.   |
+===========================================================+
    """)


def print_menu():
    """Print main menu."""
    print("""
+-----------------------------------+
|           MAIN MENU               |
+-----------------------------------+
|  1. Show status & progress        |
|  2. Estimate API usage            |
|  3. Collect data from LexisNexis  |
|  4. Validate keywords (LLM)       |
|  5. Build CPU index               |
|  6. Export index to CSV           |
|  7. Exit                          |
+-----------------------------------+
    """)


def show_status():
    """Show current collection status."""
    print("\n" + "="*50)
    print("STATUS & PROGRESS")
    print("="*50)

    status = collector.get_collection_status()

    print(f"\nDate Range: {status['date_range']}")
    print(f"Total Months: {status['total_months']}")
    print(f"Completed: {status['completed_months']} ({status['percent_complete']:.0%})")
    print(f"Remaining: {status['incomplete_months']}")

    if status['next_month']:
        print(f"\nNext month to collect: {status['next_month']}")
    else:
        print("\n[OK] All months collected!")

    # Show index status
    idx_status = indexer.get_index_summary()
    print(f"\nIndex Status: {idx_status['status']}")
    if idx_status['status'] == 'ready':
        print(f"  Period: {idx_status['period']}")
        print(f"  Mean: {idx_status['mean']}, Std: {idx_status['std']}")

    # Show validation status
    val_status = classifier.get_validation_status()
    print(f"\nLLM Validation:")
    print(f"  AI SDK Available: {'Yes' if val_status['ai_sdk_available'] else 'No'}")
    print(f"  OpenAI Key Set: {'Yes' if val_status['openai_key_set'] else 'No'}")


def estimate_usage():
    """Estimate API usage before collection."""
    print("\n" + "="*50)
    print("API USAGE ESTIMATE")
    print("="*50)

    est = collector.estimate_api_usage()

    print(f"\nMonths to collect: {est['months']}")
    print(f"Estimated API searches: {est['searches']}")
    print(f"Percent of annual quota: {est['percent_quota']:.2%}")
    print(f"  (Stanford limit: 24,000 searches/year shared)")

    if est['months'] > 0:
        print(f"\nFirst 5 months to collect:")
        for m in est['incomplete_months'][:5]:
            print(f"  - {m}")
        if est['months'] > 5:
            print(f"  ... and {est['months'] - 5} more")


def collect_data():
    """Run data collection with confirmation."""
    print("\n" + "="*50)
    print("DATA COLLECTION")
    print("="*50)

    est = collector.estimate_api_usage()

    if est['months'] == 0:
        print("\n[OK] All months already collected!")
        return

    print(f"\nMonths to collect: {est['months']}")
    print(f"Estimated API searches: {est['searches']}")
    print(f"Percent of annual quota: {est['percent_quota']:.2%}")

    print("\n[!] This will use your LexisNexis API quota.")
    print("    Make sure you have tested your queries in WSAPI first.")

    confirm = input("\nProceed with collection? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("\n[X] Collection cancelled.")
        return

    # Ask about dry run
    dry_run_input = input("Do a dry run first (no API calls)? (yes/no): ").strip().lower()
    dry_run = dry_run_input == "yes"

    if dry_run:
        print("\n[DRY RUN] Simulating collection without API calls...")
    else:
        print("\n[LIVE] Starting collection...")

    def progress_callback(current, total, month):
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {current}/{total} - {month}", end="", flush=True)

    result = collector.collect_all(dry_run=dry_run, progress_callback=progress_callback)

    print("\n")
    print(f"\n[OK] Collection complete!")
    print(f"  Processed: {result['months_processed']}")
    print(f"  Completed: {result['months_complete']}")
    print(f"  Skipped: {result['months_skipped']}")
    print(f"  Errors: {result['months_error']}")

    if result['errors']:
        print("\nErrors encountered:")
        for err in result['errors'][:5]:
            print(f"  - {err['month']}: {err.get('error', 'Unknown error')}")


def validate_sample():
    """Run LLM validation on sample."""
    print("\n" + "="*50)
    print("LLM KEYWORD VALIDATION")
    print("="*50)

    val_status = classifier.get_validation_status()

    if not val_status['ai_sdk_available']:
        print("\n[!] AI SDK not installed.")
        print("    Run: pip install ai-sdk-python")
        return

    if not val_status['openai_key_set']:
        print("\n[!] OpenAI API key not set.")
        print("    Add OPENAI_API_KEY to your .env file")
        return

    # Show cost estimate
    sample_size = config.LLM_SAMPLE_SIZE
    cost_est = classifier.estimate_classification_cost(sample_size)

    print(f"\nSample size: {sample_size} articles")
    print(f"Estimated cost: ${cost_est['estimated_cost_usd']:.4f}")
    print(f"  (GPT-5 Nano: $0.05/1M input, $0.40/1M output)")

    confirm = input("\nRun validation? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("\n[X] Validation cancelled.")
        return

    print("\n[!] LLM validation not fully implemented yet.")
    print("    This requires articles in the database.")
    print("    Run data collection first, then try again.")


def build_index():
    """Build CPU index from collected data."""
    print("\n" + "="*50)
    print("BUILD CPU INDEX")
    print("="*50)

    status = collector.get_collection_status()

    if status['completed_months'] == 0:
        print("\n[!] No data collected yet.")
        print("    Run data collection first (option 3).")
        return

    print(f"\nData available: {status['completed_months']} months")

    # Ask about base period
    print("\nNormalization base period:")
    print("  The index will be normalized so the MEAN = 100 over this period.")
    print("  Leave blank to use all available data.")

    base_start = input("\n  Base period start (YYYY-MM or blank): ").strip()
    base_end = input("  Base period end (YYYY-MM or blank): ").strip()

    base_start = base_start if base_start else None
    base_end = base_end if base_end else None

    print("\nBuilding index...")

    result = indexer.build_index(base_start=base_start, base_end=base_end)

    if result['status'] != 'success':
        print(f"\n[!] Error: {result.get('message', 'Unknown error')}")
        return

    meta = result['metadata']
    print(f"\n[OK] Index built successfully!")
    print(f"  Period: {meta['period']}")
    print(f"  Months: {meta['num_months']}")
    print(f"  Mean (normalized): {meta['mean_normalized']:.1f}")
    print(f"  Std Dev (raw): {meta['std_raw_ratio']:.4f}")

    # Run event validation
    print("\nValidating against known events...")
    validation = indexer.validate_against_events()

    print(f"\nEvent Validation: {validation.get('summary', 'N/A')}")
    for event in validation.get('events', []):
        status_icon = "[OK]" if event.get('result') == 'PASS' else "[--]"
        print(f"  {status_icon} {event['event']} ({event['date']}): {event.get('result', 'N/A')}")


def export_csv():
    """Export index to CSV file."""
    print("\n" + "="*50)
    print("EXPORT TO CSV")
    print("="*50)

    idx_status = indexer.get_index_summary()

    if idx_status['status'] != 'ready':
        print("\n[!] No index to export.")
        print("    Build the index first (option 5).")
        return

    output_path = os.path.join(config.EXPORT_DIR, "cpu_index.csv")

    print(f"\nExport destination: {output_path}")

    confirm = input("Proceed with export? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("\n[X] Export cancelled.")
        return

    try:
        num_rows = db.export_to_csv(output_path)
        print(f"\n[OK] Exported {num_rows} months to:")
        print(f"    {output_path}")
    except Exception as e:
        print(f"\n[!] Export failed: {e}")


if __name__ == "__main__":
    main()
```

**Step 2: Test run.py loads**

Run: `python run.py`
Expected: Shows menu, can exit with option 7

**Step 3: Commit**

```bash
git add run.py
git commit -m "feat: add run.py with interactive menu for researchers"
```

---

## Task 9: Add dateutil dependency and final polish

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements.txt with dateutil**

```txt
python-dotenv>=1.0.0
requests>=2.31.0
ai-sdk-python>=0.1.0
pydantic>=2.0.0
python-dateutil>=2.8.0
```

**Step 2: Test full installation**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 3: Final integration test**

Run: `python run.py` and test each menu option briefly

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add python-dateutil dependency"
```

---

## Task 10: Documentation

**Files:**
- Update: `README.md`

**Step 1: Update README.md with usage instructions**

```markdown
# Climate Policy Credibility & Uncertainty Measure (CPU)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   ```bash
   cp .env.example .env
   # Edit .env with your LexisNexis and OpenAI credentials
   ```

3. **Run the tool:**
   ```bash
   python run.py
   ```

4. **Follow the interactive menu:**
   - Option 1: Check status
   - Option 2: Estimate API usage
   - Option 3: Collect data
   - Option 5: Build index
   - Option 6: Export to CSV

## Files

| File | Purpose |
|------|---------|
| `config.py` | All settings (keywords, dates, sources) |
| `run.py` | Interactive menu (main entry point) |
| `db.py` | SQLite database operations |
| `api.py` | LexisNexis API client |
| `collector.py` | Data collection pipeline |
| `classifier.py` | LLM validation (GPT-5 Nano) |
| `indexer.py` | CPU index calculation |

## API Quota

Stanford shares 24,000 searches / 1,200,000 documents per year.
Always run "Estimate API usage" before collecting data.

## Modifying Keywords

Edit the keyword lists in `config.py`:
- `CLIMATE_TERMS`: Climate/energy topics
- `POLICY_TERMS`: Policy/government terms
- `UNCERTAINTY_TERMS`: Uncertainty language
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with usage instructions"
```

---

## Summary

After completing all tasks, you will have:

```
kiev/
├── config.py          # Settings (researcher-editable)
├── db.py              # SQLite storage
├── api.py             # LexisNexis client
├── collector.py       # Data pipeline
├── classifier.py      # GPT-5 Nano validation
├── indexer.py         # CPU calculation
├── run.py             # Interactive menu
├── requirements.txt   # Dependencies
├── .env.example       # Template for credentials
├── .gitignore         # Ignore patterns
├── README.md          # Usage docs
├── data/
│   ├── cpu.db         # SQLite database
│   └── exports/
│       └── cpu_index.csv
└── docs/
    └── plans/
        └── 2026-01-15-cpu-index-mvp.md
```

**Workflow for researchers:**
1. Edit `config.py` to set date range
2. Run `python run.py`
3. Use menu options in order: Status → Estimate → Collect → Build → Export
