"""
Tests for deduplicator.py - Article deduplication strategies

Supports multiple deduplication approaches:
1. Exact title match
2. Fuzzy title (Levenshtein distance < threshold)
3. Same source + same day + similar title
"""

import pytest

from cpu_index.collection import deduplicator
from cpu_index.collection.deduplicator import (
    exact_title_dedup,
    fuzzy_title_dedup,
    same_day_source_dedup,
    deduplicate_articles,
    get_dedup_stats,
    identify_wire_service_duplicates,
    is_wire_service,
)


class TestExactTitleDedup:
    """Tests for exact title matching deduplication."""

    def test_exact_dedup_removes_duplicates(self):
        """exact_title_dedup should remove articles with identical titles."""
        articles = [
            {"id": "1", "title": "Climate Policy Update", "date": "2024-01-15"},
            {"id": "2", "title": "Climate Policy Update", "date": "2024-01-16"},
            {"id": "3", "title": "Different Title", "date": "2024-01-15"},
        ]

        result = exact_title_dedup(articles)

        assert len(result) == 2
        titles = [a["title"] for a in result]
        assert "Climate Policy Update" in titles
        assert "Different Title" in titles

    def test_exact_dedup_keeps_first_occurrence(self):
        """exact_title_dedup should keep the first occurrence of duplicates."""
        articles = [
            {"id": "first", "title": "Same Title", "date": "2024-01-01"},
            {"id": "second", "title": "Same Title", "date": "2024-01-15"},
        ]

        result = exact_title_dedup(articles)

        assert len(result) == 1
        assert result[0]["id"] == "first"

    def test_exact_dedup_empty_list(self):
        """exact_title_dedup should handle empty list."""
        result = exact_title_dedup([])

        assert result == []

    def test_exact_dedup_case_sensitive(self):
        """exact_title_dedup should be case-insensitive."""
        articles = [
            {"id": "1", "title": "Climate Policy", "date": "2024-01-15"},
            {"id": "2", "title": "CLIMATE POLICY", "date": "2024-01-16"},
            {"id": "3", "title": "climate policy", "date": "2024-01-17"},
        ]

        result = exact_title_dedup(articles)

        # All three should be considered duplicates (case-insensitive)
        assert len(result) == 1


class TestFuzzyTitleDedup:
    """Tests for fuzzy title matching deduplication."""

    def test_fuzzy_dedup_near_duplicates(self):
        """fuzzy_title_dedup should identify near-duplicate titles."""
        articles = [
            {"id": "1", "title": "Climate Policy Update for 2024", "date": "2024-01-15"},
            {"id": "2", "title": "Climate Policy Update for 2025", "date": "2024-01-16"},
            {"id": "3", "title": "Completely Different Article", "date": "2024-01-17"},
        ]

        result = fuzzy_title_dedup(articles, threshold=0.8)

        # First two are similar enough to be considered duplicates
        assert len(result) == 2

    def test_fuzzy_dedup_respects_threshold(self):
        """fuzzy_title_dedup should respect the similarity threshold."""
        articles = [
            {"id": "1", "title": "Climate Policy 2024", "date": "2024-01-15"},
            {"id": "2", "title": "Climate Policy 2025", "date": "2024-01-16"},
        ]

        # With low threshold, they should be kept as separate
        result_low = fuzzy_title_dedup(articles, threshold=0.99)
        assert len(result_low) == 2

        # With high threshold, they should be merged
        result_high = fuzzy_title_dedup(articles, threshold=0.7)
        assert len(result_high) == 1

    def test_fuzzy_dedup_empty_list(self):
        """fuzzy_title_dedup should handle empty list."""
        result = fuzzy_title_dedup([])

        assert result == []


class TestSameDaySourceDedup:
    """Tests for same-day, same-source deduplication."""

    def test_same_day_source_dedup(self):
        """same_day_source_dedup should remove articles from same source on same day."""
        articles = [
            {"id": "1", "title": "Story A", "date": "2024-01-15", "source": "Reuters"},
            {"id": "2", "title": "Story B", "date": "2024-01-15", "source": "Reuters"},
            {"id": "3", "title": "Story C", "date": "2024-01-15", "source": "AP"},
            {"id": "4", "title": "Story D", "date": "2024-01-16", "source": "Reuters"},
        ]

        result = same_day_source_dedup(articles, similarity_threshold=0.0)

        # Should keep: one from Reuters 01-15, one from AP 01-15, one from Reuters 01-16
        assert len(result) == 3

    def test_same_day_source_dedup_considers_similarity(self):
        """same_day_source_dedup should check title similarity for same source/day."""
        articles = [
            {"id": "1", "title": "Climate Rollback Plan", "date": "2024-01-15", "source": "Reuters"},
            {"id": "2", "title": "Climate Rollback Plan Updated", "date": "2024-01-15", "source": "Reuters"},
            {"id": "3", "title": "Totally Different Story", "date": "2024-01-15", "source": "Reuters"},
        ]

        result = same_day_source_dedup(articles, similarity_threshold=0.7)

        # First two are similar and same day/source - should be deduped
        # Third is different enough to keep
        assert len(result) == 2

    def test_same_day_source_dedup_empty_list(self):
        """same_day_source_dedup should handle empty list."""
        result = same_day_source_dedup([])

        assert result == []


class TestDeduplicationPipeline:
    """Tests for the combined deduplication pipeline."""

    def test_deduplicate_articles_default_strategy(self):
        """deduplicate_articles with default strategy should work."""
        articles = [
            {"id": "1", "title": "Duplicate Title", "date": "2024-01-15", "source": "Reuters"},
            {"id": "2", "title": "Duplicate Title", "date": "2024-01-16", "source": "AP"},
            {"id": "3", "title": "Unique Title", "date": "2024-01-17", "source": "NYT"},
        ]

        result = deduplicate_articles(articles, strategy="exact")

        assert len(result) == 2

    def test_deduplicate_articles_all_strategies(self):
        """deduplicate_articles should support all strategies."""
        articles = [
            {"id": "1", "title": "Test Article", "date": "2024-01-15", "source": "Test"},
            {"id": "2", "title": "Test Article", "date": "2024-01-15", "source": "Test"},
        ]

        for strategy in ["exact", "fuzzy", "same_day_source", "none"]:
            result = deduplicate_articles(articles, strategy=strategy)
            assert isinstance(result, list)

    def test_deduplicate_articles_none_strategy(self):
        """deduplicate_articles with 'none' strategy should return all articles."""
        articles = [
            {"id": "1", "title": "Same", "date": "2024-01-15", "source": "Test"},
            {"id": "2", "title": "Same", "date": "2024-01-15", "source": "Test"},
        ]

        result = deduplicate_articles(articles, strategy="none")

        assert len(result) == 2  # No deduplication

    def test_deduplicate_articles_invalid_strategy(self):
        """deduplicate_articles should raise on invalid strategy."""
        articles = [{"id": "1", "title": "Test", "date": "2024-01-15", "source": "Test"}]

        with pytest.raises(ValueError, match="Unknown deduplication strategy"):
            deduplicate_articles(articles, strategy="invalid")


class TestDeduplicationStats:
    """Tests for deduplication statistics."""

    def test_get_dedup_stats(self):
        """get_dedup_stats should return correct counts."""
        original = [
            {"id": "1", "title": "A", "date": "2024-01-15"},
            {"id": "2", "title": "A", "date": "2024-01-16"},
            {"id": "3", "title": "B", "date": "2024-01-17"},
        ]
        deduped = [
            {"id": "1", "title": "A", "date": "2024-01-15"},
            {"id": "3", "title": "B", "date": "2024-01-17"},
        ]

        stats = get_dedup_stats(original, deduped)

        assert stats["original_count"] == 3
        assert stats["deduped_count"] == 2
        assert stats["removed_count"] == 1
        assert stats["removal_rate"] == pytest.approx(1/3)


class TestWireServiceDetection:
    """Tests for wire service duplicate detection."""

    def test_identify_wire_service_duplicates(self):
        """identify_wire_service_duplicates should find wire service articles."""
        articles = [
            {"id": "1", "title": "Breaking News", "source": "Associated Press"},
            {"id": "2", "title": "Breaking News", "source": "Chicago Tribune"},
            {"id": "3", "title": "Breaking News", "source": "Reuters"},
            {"id": "4", "title": "Local Story", "source": "Chicago Tribune"},
        ]

        duplicates = identify_wire_service_duplicates(articles)

        # Should identify that "Breaking News" is likely a wire story
        assert len(duplicates) > 0

    def test_is_wire_service(self):
        """is_wire_service should identify wire service sources."""
        assert is_wire_service("Associated Press")
        assert is_wire_service("Reuters")
        assert is_wire_service("AFP")
        assert not is_wire_service("New York Times")
        assert not is_wire_service("Chicago Tribune")
