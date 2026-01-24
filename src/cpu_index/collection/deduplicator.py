"""
Article Deduplication Strategies for CPU Index Builder

LexisNexis API doesn't provide deduplication (UI only), so we implement
multiple deduplication strategies for analysis.

Strategies:
1. exact: Exact title match (case-insensitive)
2. fuzzy: Fuzzy title matching using Levenshtein similarity
3. same_day_source: Same source + same day + similar title
4. none: No deduplication (for comparison)

Wire services (AP, Reuters, AFP) create massive duplication as their
stories are picked up by many outlets.
"""

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Optional


# Known wire services that syndicate content
WIRE_SERVICES = {
    "associated press",
    "ap",
    "reuters",
    "afp",
    "agence france-presse",
    "agence france presse",
    "united press international",
    "upi",
}


def _normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    if not title:
        return ""
    return " ".join(title.lower().split())


def _calculate_similarity(title1: str, title2: str) -> float:
    """
    Calculate similarity between two titles using SequenceMatcher.

    Returns value between 0 (completely different) and 1 (identical).
    """
    t1 = _normalize_title(title1)
    t2 = _normalize_title(title2)
    return SequenceMatcher(None, t1, t2).ratio()


def exact_title_dedup(articles: list[dict]) -> list[dict]:
    """
    Remove duplicates based on exact title match (case-insensitive).

    Keeps the first occurrence of each unique title.

    Args:
        articles: List of article dicts with 'title' field

    Returns:
        Deduplicated list of articles
    """
    if not articles:
        return []

    seen_titles = set()
    result = []

    for article in articles:
        title = _normalize_title(article.get("title", ""))
        if title not in seen_titles:
            seen_titles.add(title)
            result.append(article)

    return result


def fuzzy_title_dedup(
    articles: list[dict],
    threshold: float = 0.85,
) -> list[dict]:
    """
    Remove duplicates based on fuzzy title matching.

    Uses Levenshtein-based similarity. If two titles are similar enough,
    the later one is considered a duplicate.

    Args:
        articles: List of article dicts with 'title' field
        threshold: Similarity threshold (0-1). Higher = stricter matching.

    Returns:
        Deduplicated list of articles
    """
    if not articles:
        return []

    result = []

    for article in articles:
        title = article.get("title", "")
        is_duplicate = False

        for kept_article in result:
            kept_title = kept_article.get("title", "")
            similarity = _calculate_similarity(title, kept_title)

            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(article)

    return result


def same_day_source_dedup(
    articles: list[dict],
    similarity_threshold: float = 0.7,
) -> list[dict]:
    """
    Remove duplicates from the same source on the same day.

    Only considers articles duplicates if they:
    1. Are from the same source
    2. Are on the same day
    3. Have similar titles

    Args:
        articles: List of article dicts with 'title', 'date', 'source' fields
        similarity_threshold: Title similarity threshold for same-day-source

    Returns:
        Deduplicated list of articles
    """
    if not articles:
        return []

    # Group by (source, date)
    groups = defaultdict(list)
    for article in articles:
        source = article.get("source", "").lower()
        date = article.get("date", "")[:10]  # Just the date part
        groups[(source, date)].append(article)

    result = []

    for (source, date), group in groups.items():
        # Within each group, use fuzzy dedup
        deduped_group = fuzzy_title_dedup(group, threshold=similarity_threshold)
        result.extend(deduped_group)

    return result


def deduplicate_articles(
    articles: list[dict],
    strategy: str = "exact",
    **kwargs,
) -> list[dict]:
    """
    Main entry point for article deduplication.

    Args:
        articles: List of article dicts
        strategy: Deduplication strategy ("exact", "fuzzy", "same_day_source", "none")
        **kwargs: Additional arguments passed to the strategy function

    Returns:
        Deduplicated list of articles

    Raises:
        ValueError: If unknown strategy is specified
    """
    strategies = {
        "exact": exact_title_dedup,
        "fuzzy": fuzzy_title_dedup,
        "same_day_source": same_day_source_dedup,
        "none": lambda articles, **kw: list(articles),
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown deduplication strategy: {strategy}. "
            f"Must be one of: {list(strategies.keys())}"
        )

    return strategies[strategy](articles, **kwargs)


def get_dedup_stats(
    original: list[dict],
    deduped: list[dict],
) -> dict:
    """
    Calculate deduplication statistics.

    Args:
        original: Original article list
        deduped: Deduplicated article list

    Returns:
        Dict with statistics: original_count, deduped_count, removed_count, removal_rate
    """
    original_count = len(original)
    deduped_count = len(deduped)
    removed_count = original_count - deduped_count

    return {
        "original_count": original_count,
        "deduped_count": deduped_count,
        "removed_count": removed_count,
        "removal_rate": removed_count / original_count if original_count > 0 else 0.0,
    }


def is_wire_service(source: str) -> bool:
    """
    Check if a source is a known wire service.

    Args:
        source: Source name

    Returns:
        True if the source is a wire service
    """
    if not source:
        return False
    return source.lower() in WIRE_SERVICES


def identify_wire_service_duplicates(articles: list[dict]) -> list[dict]:
    """
    Identify articles that are likely wire service duplicates.

    A story appearing in multiple outlets with the same title is likely
    a wire service story being syndicated.

    Args:
        articles: List of article dicts

    Returns:
        List of article dicts that are likely wire service duplicates
    """
    # Group by normalized title
    title_groups = defaultdict(list)
    for article in articles:
        title = _normalize_title(article.get("title", ""))
        if title:
            title_groups[title].append(article)

    # Find titles that appear in multiple sources
    duplicates = []
    for title, group in title_groups.items():
        if len(group) > 1:
            # Check if different sources
            sources = set(a.get("source", "").lower() for a in group)
            if len(sources) > 1:
                # Multiple sources with same title = likely wire story
                duplicates.extend(group)

    return duplicates


def compare_dedup_strategies(articles: list[dict]) -> dict:
    """
    Compare different deduplication strategies on the same dataset.

    Useful for understanding how many duplicates each strategy finds.

    Args:
        articles: List of article dicts

    Returns:
        Dict mapping strategy names to their results and stats
    """
    strategies = ["none", "exact", "fuzzy", "same_day_source"]

    results = {}
    for strategy in strategies:
        deduped = deduplicate_articles(articles, strategy=strategy)
        stats = get_dedup_stats(articles, deduped)
        results[strategy] = {
            "count": len(deduped),
            "stats": stats,
        }

    return results
