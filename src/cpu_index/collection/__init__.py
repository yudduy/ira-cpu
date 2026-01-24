"""Data collection pipeline for CPU Index."""

from .api import fetch_count, fetch_metadata, fetch_articles_for_month
from .collector import ArticleCollector
from .count_collector import collect_month_counts, collect_all_counts, get_collection_status
from .deduplicator import deduplicate_articles
