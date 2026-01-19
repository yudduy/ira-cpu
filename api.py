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
from urllib.parse import urlencode

import requests
from dotenv import find_dotenv, load_dotenv, set_key
from requests.auth import HTTPBasicAuth

import config

load_dotenv()

# =============================================================================
# REQUEST CONFIGURATION
# =============================================================================
# Best practice: Always set timeouts to prevent indefinite hangs
# See: https://python-requests.org/python-requests-timeout/
# Format: (connect_timeout, read_timeout) in seconds
REQUEST_TIMEOUT = (5, 30)  # 5s to connect, 30s to read


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
        auth=HTTPBasicAuth(clientid, clientsecret),
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        print(f"Error getting token: {response.status_code}")
        print(response.text)
        return ""

    results = response.json()
    token = results["access_token"]
    expires_in = results["expires_in"]

    # TODO: Consider storing tokens in memory or a separate cache file instead of .env
    # Writing to .env at runtime is a security risk if .env is version-controlled.
    # Options: (1) module-level dict cache, (2) .cache/token.json (gitignored)
    # See: https://betterstack.com/community/guides/scaling-python/python-timeouts/
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
    direction_terms: list = None,
) -> str:
    """
    Build a LexisNexis search query string (without date filter).

    Args:
        climate_terms: List of climate/energy keywords (require at least 1)
        policy_terms: List of policy keywords (require at least 1)
        uncertainty_terms: Optional list of uncertainty keywords (for numerator)
        direction_terms: Optional list of directional keywords for asymmetric indices
                        (DOWNSIDE_TERMS for CPU-Down, UPSIDE_TERMS for CPU-Up)

    Returns:
        Search query string (dates should be passed via $filter parameter)

    Query Structure:
        - Denominator: (climate) AND (policy)
        - Numerator (standard): (climate) AND (policy) AND (uncertainty)
        - Numerator Down: (climate) AND (policy) AND (uncertainty) AND (downside)
        - Numerator Up: (climate) AND (policy) AND (uncertainty) AND (upside)

    Based on Segal, Shaliastovich & Yaron (2015) good/bad uncertainty methodology.
    """
    # Build climate terms: (climate OR "clean energy" OR renewable)
    climate_part = " OR ".join([_format_term(t) for t in climate_terms])

    # Build policy terms: (policy OR regulation OR Congress)
    policy_part = " OR ".join([_format_term(t) for t in policy_terms])

    # Combine: (climate terms) AND (policy terms)
    query = f"({climate_part}) AND ({policy_part})"

    # Add uncertainty terms if provided (for numerator query)
    if uncertainty_terms:
        uncertainty_part = " OR ".join([_format_term(t) for t in uncertainty_terms])
        query += f" AND ({uncertainty_part})"

    # Add direction terms if provided (for CPU-Down or CPU-Up)
    if direction_terms:
        direction_part = " OR ".join([_format_term(t) for t in direction_terms])
        query += f" AND ({direction_part})"

    return query


def build_date_filter(start_date: str, end_date: str) -> str:
    """
    Build a date filter for LexisNexis API ($filter parameter).

    Args:
        start_date: Start of date range (YYYY-MM-DD)
        end_date: End of date range (YYYY-MM-DD)

    Returns:
        Filter string for $filter parameter
    """
    return f"Date ge {start_date} and Date le {end_date}"


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

def fetch_count(query: str, date_filter: str = None, dry_run: bool = False) -> int:
    """
    Get total article count for a query (uses minimal quota).

    Args:
        query: Search query string
        date_filter: Date filter string (e.g., "Date ge 2024-01-01 and Date le 2024-01-31")
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

    # Add date filter as separate $filter parameter (not in $search)
    if date_filter:
        params["$filter"] = date_filter

    url = f"{base_url}?{urlencode(params)}"

    headers = {"Authorization": f"Bearer {token}"}

    time.sleep(config.REQUEST_DELAY_SECONDS)
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    result = response.json()

    # Total count is in @odata.count
    return result.get("@odata.count", 0)


def fetch_metadata(
    query: str,
    date_filter: str = None,
    max_results: int = None,
    dry_run: bool = False,
    progress_callback: callable = None,
) -> list[dict]:
    """
    Fetch article metadata (not full text).
    Handles pagination automatically.

    Args:
        query: Search query string
        date_filter: Date filter string (e.g., "Date ge 2024-01-01 and Date le 2024-01-31")
        max_results: Maximum articles to fetch (None = all)
        dry_run: If True, return fake data without API call
        progress_callback: Optional callback function(fetched_count, total_count)

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
    total = None

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

        # Add date filter if provided
        if date_filter:
            params["$filter"] = date_filter

        url = f"{base_url}?{urlencode(params)}"

        headers = {"Authorization": f"Bearer {token}"}

        time.sleep(config.REQUEST_DELAY_SECONDS)
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")

        result = response.json()
        batch = result.get("value", [])

        # Get total count from first response
        if total is None:
            total = result.get("@odata.count", 0)

        if not batch:
            break

        # Extract relevant fields
        for item in batch:
            article = {
                "id": item.get("ResultId", ""),
                "title": item.get("Title", ""),
                "date": item.get("Date", ""),
                "source": item.get("Source", {}).get("Name", ""),
                "snippet": item.get("Overview", ""),
            }
            articles.append(article)

        # Report progress if callback provided
        if progress_callback:
            progress_callback(len(articles), total)

        # Check if we have enough
        if max_results and len(articles) >= max_results:
            articles = articles[:max_results]
            break

        # Check if there are more results
        skip += config.MAX_RESULTS_PER_QUERY
        if skip >= total:
            break

    return articles


def fetch_articles_for_month(
    year: int,
    month: int,
    query: str = None,
    dry_run: bool = False,
    progress_callback: callable = None,
) -> tuple[list[dict], dict]:
    """
    Fetch all articles matching base query for a given month.

    This is the primary fetch function for the "fetch once, classify locally" strategy.
    It fetches articles matching (climate AND policy) and stores them for local classification.

    Args:
        year: Year to fetch
        month: Month to fetch (1-12)
        query: Optional custom query (defaults to climate AND policy)
        dry_run: If True, return fake data without API call
        progress_callback: Optional callback function(fetched_count, total_count)

    Returns:
        Tuple of (articles_list, raw_response_metadata)
        - articles_list: List of article dicts with id, title, date, source, snippet
        - raw_response_metadata: Dict with query_hash, total_count, fetched_at
    """
    import hashlib
    from datetime import datetime, timezone

    # Build date range
    start_date, end_date = build_month_dates(year, month)
    date_filter = build_date_filter(start_date, end_date)

    # Build query (default: climate AND policy - the denominator)
    if query is None:
        query = build_search_query(
            climate_terms=config.CLIMATE_TERMS,
            policy_terms=config.POLICY_TERMS,
        )

    # Create query hash for deduplication
    query_hash = hashlib.sha256(f"{query}|{date_filter}".encode()).hexdigest()[:16]

    # Fetch articles
    articles = fetch_metadata(
        query=query,
        date_filter=date_filter,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )

    # Add month field to each article
    month_str = f"{year}-{month:02d}"
    for article in articles:
        article["month"] = month_str

    # Prepare metadata for raw response storage
    raw_metadata = {
        "query_hash": query_hash,
        "query": query,
        "date_filter": date_filter,
        "month": month_str,
        "total_count": len(articles),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    return articles, raw_metadata


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
