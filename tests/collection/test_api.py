"""
Tests for api.py - LexisNexis API client

Uses mocked requests to prevent real API calls.
Tests query building, authentication, and response handling.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from cpu_index import config
from cpu_index.collection import api


class TestQueryBuilding:
    """Tests for search query construction."""

    def test_build_search_query_basic(self):
        """build_search_query should combine climate and policy terms."""
        query = api.build_search_query(
            climate_terms=["climate", "renewable"],
            policy_terms=["policy", "regulation"]
        )

        assert "climate" in query
        assert "renewable" in query
        assert "policy" in query
        assert "regulation" in query
        assert " AND " in query
        assert " OR " in query

    def test_build_search_query_with_uncertainty(self):
        """build_search_query should include uncertainty terms when provided."""
        query = api.build_search_query(
            climate_terms=["climate"],
            policy_terms=["policy"],
            uncertainty_terms=["uncertain", "delay"]
        )

        assert "uncertain" in query
        assert "delay" in query

    def test_build_search_query_with_direction_terms(self):
        """build_search_query should include direction terms for asymmetric indices."""
        query = api.build_search_query(
            climate_terms=["climate"],
            policy_terms=["policy"],
            uncertainty_terms=["uncertain"],
            direction_terms=["rollback", "repeal"]
        )

        # Should have all four term groups
        assert "climate" in query
        assert "policy" in query
        assert "uncertain" in query
        assert "rollback" in query or "repeal" in query

    def test_build_search_query_multiword_terms(self):
        """build_search_query should quote multi-word terms."""
        query = api.build_search_query(
            climate_terms=["clean energy", "climate change"],
            policy_terms=["tax credit"]
        )

        # Multi-word terms should be quoted with + for spaces
        assert '"clean+energy"' in query
        assert '"climate+change"' in query
        assert '"tax+credit"' in query

    def test_format_term_single_word(self):
        """_format_term should return single words unchanged."""
        result = api._format_term("climate")
        assert result == "climate"

    def test_format_term_multi_word(self):
        """_format_term should format multi-word terms."""
        result = api._format_term("clean energy")
        assert result == '"clean+energy"'


class TestBuildMonthDates:
    """Tests for month date range calculation."""

    def test_build_month_dates_january(self):
        """build_month_dates should handle January correctly."""
        start, end = api.build_month_dates(2024, 1)
        assert start == "2024-01-01"
        assert end == "2024-01-31"

    def test_build_month_dates_february_leap_year(self):
        """build_month_dates should handle February in leap year."""
        start, end = api.build_month_dates(2024, 2)
        assert start == "2024-02-01"
        assert end == "2024-02-29"  # 2024 is a leap year

    def test_build_month_dates_february_non_leap_year(self):
        """build_month_dates should handle February in non-leap year."""
        start, end = api.build_month_dates(2023, 2)
        assert start == "2023-02-01"
        assert end == "2023-02-28"

    def test_build_month_dates_december(self):
        """build_month_dates should handle December correctly."""
        start, end = api.build_month_dates(2024, 12)
        assert start == "2024-12-01"
        assert end == "2024-12-31"

    def test_build_month_dates_thirty_day_month(self):
        """build_month_dates should handle 30-day months."""
        start, end = api.build_month_dates(2024, 4)
        assert start == "2024-04-01"
        assert end == "2024-04-30"


class TestGetToken:
    """Tests for token management."""

    def test_get_token_missing_credentials(self, monkeypatch):
        """get_token should return empty string without credentials."""
        # Clear environment variables
        monkeypatch.delenv("clientid", raising=False)
        monkeypatch.delenv("clientsecret", raising=False)

        result = api.get_token()
        assert result == ""

    def test_get_token_uses_cached_token(self, mock_env_vars):
        """get_token should return cached token if not expired."""
        # The mock_env_vars fixture sets lntoken and lnexpire
        result = api.get_token()
        assert result == "mock_cached_token"

    def test_get_token_refreshes_expired_token(self, monkeypatch):
        """get_token should refresh token if expired."""
        monkeypatch.setenv("clientid", "test_client")
        monkeypatch.setenv("clientsecret", "test_secret")
        monkeypatch.setenv("lntoken", "old_token")
        monkeypatch.setenv("lnexpire", "0")  # Expired

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_fresh_token",
            "expires_in": 3600
        }

        with patch("cpu_index.collection.api.requests.post", return_value=mock_response):
            with patch("cpu_index.collection.api.find_dotenv", return_value="/tmp/.env"):
                with patch("cpu_index.collection.api.set_key"):
                    result = api._refresh_token("test_client", "test_secret")

        assert result == "new_fresh_token"

    def test_refresh_token_handles_error(self, monkeypatch):
        """_refresh_token should return empty string on error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("cpu_index.collection.api.requests.post", return_value=mock_response):
            result = api._refresh_token("bad_client", "bad_secret")

        assert result == ""


class TestFetchCount:
    """Tests for fetch_count API calls."""

    def test_fetch_count_dry_run(self):
        """fetch_count with dry_run should return fake count."""
        result = api.fetch_count("test query", dry_run=True)
        assert result == 100  # Fake count

    def test_fetch_count_makes_api_call(self, mock_env_vars):
        """fetch_count should make proper API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"@odata.count": 150, "value": []}

        with patch("cpu_index.collection.api.requests.get", return_value=mock_response) as mock_get:
            with patch("cpu_index.collection.api.time.sleep"):  # Skip delay in tests
                result = api.fetch_count("climate AND policy")

        assert result == 150
        mock_get.assert_called_once()

        # Verify the URL contains expected parameters
        call_url = mock_get.call_args[0][0]
        assert "News" in call_url
        # URL-encoded: $top=1 becomes %24top=1
        assert "top=1" in call_url

    def test_fetch_count_no_token_raises(self, monkeypatch):
        """fetch_count should raise if no token available."""
        monkeypatch.delenv("clientid", raising=False)
        monkeypatch.delenv("clientsecret", raising=False)

        with pytest.raises(RuntimeError, match="No API token"):
            api.fetch_count("test query")

    def test_fetch_count_api_error_raises(self, mock_env_vars):
        """fetch_count should raise on API error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("cpu_index.collection.api.requests.get", return_value=mock_response):
            with patch("cpu_index.collection.api.time.sleep"):
                with pytest.raises(RuntimeError, match="API error 500"):
                    api.fetch_count("test query")


class TestFetchMetadata:
    """Tests for fetch_metadata API calls."""

    def test_fetch_metadata_dry_run(self):
        """fetch_metadata with dry_run should return fake articles."""
        result = api.fetch_metadata("test query", dry_run=True)

        assert len(result) == 10
        assert result[0]["id"] == "fake_0"

    def test_fetch_metadata_extracts_fields(self, mock_env_vars, mock_api_metadata_response):
        """fetch_metadata should extract correct fields from response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_metadata_response

        with patch("cpu_index.collection.api.requests.get", return_value=mock_response):
            with patch("cpu_index.collection.api.time.sleep"):
                result = api.fetch_metadata("test query", max_results=3)

        assert len(result) == 3
        assert result[0]["id"] == "article_001"
        assert result[0]["title"] == "Climate Policy Update"
        assert result[0]["source"] == "Reuters"

    def test_fetch_metadata_respects_max_results(self, mock_env_vars):
        """fetch_metadata should respect max_results parameter."""
        # Create response with more items than max_results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "@odata.count": 100,
            "value": [
                {"ResultId": f"id_{i}", "Title": f"Article {i}",
                 "Date": "2024-01-01", "Source": {"Name": "Test"}, "Overview": "..."}
                for i in range(50)
            ]
        }

        with patch("cpu_index.collection.api.requests.get", return_value=mock_response):
            with patch("cpu_index.collection.api.time.sleep"):
                result = api.fetch_metadata("test", max_results=5)

        assert len(result) == 5

    def test_fetch_metadata_handles_empty_response(self, mock_env_vars):
        """fetch_metadata should handle empty results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"@odata.count": 0, "value": []}

        with patch("cpu_index.collection.api.requests.get", return_value=mock_response):
            with patch("cpu_index.collection.api.time.sleep"):
                result = api.fetch_metadata("obscure query")

        assert result == []


class TestFetchFullText:
    """Tests for fetch_full_text API calls."""

    def test_fetch_full_text_dry_run(self):
        """fetch_full_text with dry_run should return fake text."""
        result = api.fetch_full_text(["id_1", "id_2"], dry_run=True)

        assert len(result) == 2
        assert result[0]["id"] == "id_1"
        assert "full_text" in result[0]

    def test_fetch_full_text_not_implemented(self):
        """fetch_full_text should raise NotImplementedError for real calls."""
        with pytest.raises(NotImplementedError):
            api.fetch_full_text(["id_1"])


class TestFetchMetadataWithDateFilter:
    """Tests for fetch_metadata with date filtering."""

    def test_fetch_metadata_with_date_filter(self, mock_env_vars):
        """fetch_metadata should include date filter in API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"@odata.count": 5, "value": []}

        with patch("cpu_index.collection.api.requests.get", return_value=mock_response) as mock_get:
            with patch("cpu_index.collection.api.time.sleep"):
                api.fetch_metadata(
                    "climate AND policy",
                    date_filter="Date ge 2024-01-01 and Date le 2024-01-31",
                )

        # Verify the URL contains the date filter
        call_url = mock_get.call_args[0][0]
        assert "filter=" in call_url

    def test_fetch_metadata_progress_callback(self, mock_env_vars):
        """fetch_metadata should call progress callback."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "@odata.count": 10,
            "value": [
                {"ResultId": f"id_{i}", "Title": f"Article {i}",
                 "Date": "2024-01-01", "Source": {"Name": "Test"}, "Overview": "..."}
                for i in range(10)
            ]
        }

        progress_calls = []
        def track_progress(fetched, total):
            progress_calls.append((fetched, total))

        with patch("cpu_index.collection.api.requests.get", return_value=mock_response):
            with patch("cpu_index.collection.api.time.sleep"):
                api.fetch_metadata(
                    "test query",
                    progress_callback=track_progress,
                )

        # Should have called progress callback
        assert len(progress_calls) > 0
        assert progress_calls[-1][1] == 10  # Total should be 10


class TestFetchArticlesForMonth:
    """Tests for the fetch_articles_for_month convenience function."""

    def test_fetch_articles_for_month_dry_run(self):
        """fetch_articles_for_month with dry_run should return fake articles."""
        articles, metadata = api.fetch_articles_for_month(
            year=2024,
            month=1,
            dry_run=True,
        )

        assert len(articles) == 10
        assert all(a.get("month") == "2024-01" for a in articles)
        assert "query_hash" in metadata
        assert metadata["month"] == "2024-01"

    def test_fetch_articles_for_month_default_query(self, mock_env_vars):
        """fetch_articles_for_month should use climate AND policy query by default."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "@odata.count": 3,
            "value": [
                {"ResultId": f"id_{i}", "Title": f"Article {i}",
                 "Date": "2024-01-15", "Source": {"Name": "Test"}, "Overview": "..."}
                for i in range(3)
            ]
        }

        with patch("cpu_index.collection.api.requests.get", return_value=mock_response) as mock_get:
            with patch("cpu_index.collection.api.time.sleep"):
                articles, metadata = api.fetch_articles_for_month(
                    year=2024,
                    month=1,
                )

        # Verify climate and policy terms are in the query
        call_url = mock_get.call_args[0][0]
        # Check that at least one climate term is in the query
        assert any(term.lower() in call_url.lower() for term in config.CLIMATE_TERMS[:3])

    def test_fetch_articles_for_month_adds_month_field(self):
        """fetch_articles_for_month should add month field to all articles."""
        articles, metadata = api.fetch_articles_for_month(
            year=2024,
            month=6,
            dry_run=True,
        )

        # All articles should have the correct month
        for article in articles:
            assert article["month"] == "2024-06"

    def test_fetch_articles_for_month_metadata_includes_query_info(self):
        """fetch_articles_for_month should return proper metadata."""
        articles, metadata = api.fetch_articles_for_month(
            year=2024,
            month=1,
            dry_run=True,
        )

        assert "query_hash" in metadata
        assert len(metadata["query_hash"]) == 16  # Truncated hash
        assert "query" in metadata
        assert "date_filter" in metadata
        assert "fetched_at" in metadata
        assert metadata["total_count"] == len(articles)

    def test_fetch_articles_for_month_with_custom_query(self):
        """fetch_articles_for_month should accept custom query."""
        custom_query = "trade AND tariff"
        articles, metadata = api.fetch_articles_for_month(
            year=2024,
            month=1,
            query=custom_query,
            dry_run=True,
        )

        assert metadata["query"] == custom_query


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_fetch_count_respects_delay(self, mock_env_vars):
        """fetch_count should call time.sleep for rate limiting."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"@odata.count": 100, "value": []}

        with patch("cpu_index.collection.api.requests.get", return_value=mock_response):
            with patch("cpu_index.collection.api.time.sleep") as mock_sleep:
                api.fetch_count("test query")

        mock_sleep.assert_called_once()
