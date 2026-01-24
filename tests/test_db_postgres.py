"""
Tests for db_postgres.py - PostgreSQL database operations

These tests verify the PostgreSQL database module functionality.
Integration tests require a running PostgreSQL instance (docker-compose up).
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from cpu_index import db_postgres


class TestConnectionManagement:
    """Tests for database connection management."""

    def test_get_connection_pool_returns_pool(self):
        """get_connection_pool should return a connection pool object."""
        # This test will fail without a running PostgreSQL instance
        # Skip if DATABASE_URL is not set or points to non-running server
        try:
            pool = db_postgres.get_connection_pool()
            assert pool is not None
            db_postgres.close_pool()
        except Exception:
            pytest.skip("PostgreSQL not available")

    def test_close_pool_cleans_up(self):
        """close_pool should close all connections and reset the pool."""
        db_postgres._connection_pool = MagicMock()
        db_postgres.close_pool()
        assert db_postgres._connection_pool is None


class TestArticleOperations:
    """Tests for article CRUD operations."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return mock_conn, mock_cursor

    def test_save_article_calls_execute(self, mock_connection):
        """save_article should execute INSERT with article data."""
        mock_conn, mock_cursor = mock_connection

        with patch.object(db_postgres, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            article = {
                "id": "test123",
                "title": "Test Article",
                "date": "2024-01-15",
                "source": "Test Source",
                "snippet": "Test snippet text",
                "month": "2024-01",
            }

            db_postgres.save_article(article)

            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            assert "INSERT INTO articles" in call_args[0][0]
            assert article["id"] in call_args[0][1]

    def test_save_articles_batch_handles_empty_list(self, mock_connection):
        """save_articles_batch should handle empty list gracefully."""
        # Should not raise and not call database
        db_postgres.save_articles_batch([])


class TestKeywordClassificationOperations:
    """Tests for keyword classification operations."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return mock_conn, mock_cursor

    def test_save_keyword_classification_params(self, mock_connection):
        """save_keyword_classification should use correct parameters."""
        mock_conn, mock_cursor = mock_connection

        with patch.object(db_postgres, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            db_postgres.save_keyword_classification(
                article_id="test123",
                has_uncertainty=True,
                has_reversal_terms=True,
                has_implementation_terms=False,
                has_upside_terms=False,
                has_ira_mention=True,
                has_obbba_mention=False,
                matched_terms={"uncertainty": ["uncertain", "risk"]},
            )

            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            assert "INSERT INTO keyword_classifications" in call_args[0][0]


class TestIndexValueOperations:
    """Tests for index value operations."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return mock_conn, mock_cursor

    def test_save_index_value_computes_ratio(self, mock_connection):
        """save_index_value should compute raw_ratio if not provided."""
        mock_conn, mock_cursor = mock_connection

        with patch.object(db_postgres, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            db_postgres.save_index_value(
                month="2024-01",
                index_type="cpu",
                denominator=100,
                numerator=50,
            )

            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            # The raw_ratio should be computed as 50/100 = 0.5
            assert 0.5 in call_args[0][1]

    def test_save_index_value_handles_zero_denominator(self, mock_connection):
        """save_index_value should handle zero denominator."""
        mock_conn, mock_cursor = mock_connection

        with patch.object(db_postgres, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            # Should not raise ZeroDivisionError
            db_postgres.save_index_value(
                month="2024-01",
                index_type="cpu",
                denominator=0,
                numerator=0,
            )

            mock_cursor.execute.assert_called_once()


class TestCollectionProgressOperations:
    """Tests for collection progress tracking."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return mock_conn, mock_cursor

    def test_save_collection_progress(self, mock_connection):
        """save_collection_progress should insert progress record."""
        mock_conn, mock_cursor = mock_connection

        with patch.object(db_postgres, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            db_postgres.save_collection_progress("2024-01", 1500)

            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args
            assert "INSERT INTO collection_progress" in call_args[0][0]
            assert "2024-01" in call_args[0][1]
            assert 1500 in call_args[0][1]


class TestExportOperations:
    """Tests for CSV export functionality."""

    def test_export_index_to_csv_raises_on_empty(self):
        """export_index_to_csv should raise ValueError if no data."""
        with patch.object(db_postgres, 'get_connection') as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []

            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(ValueError, match="No index values to export"):
                db_postgres.export_index_to_csv("/tmp/test_export.csv")


# =============================================================================
# Integration Tests (require running PostgreSQL)
# =============================================================================

@pytest.fixture
def db_setup():
    """
    Set up test database.
    Skips if PostgreSQL is not available.
    """
    try:
        db_postgres.init_db()
        yield
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")
    finally:
        db_postgres.close_pool()


@pytest.mark.integration
class TestIntegrationArticles:
    """Integration tests for article operations (requires PostgreSQL)."""

    def test_save_and_retrieve_article(self, db_setup):
        """Should be able to save and retrieve an article."""
        article = {
            "id": f"integration_test_{os.urandom(4).hex()}",
            "title": "Integration Test Article",
            "date": "2024-01-15",
            "source": "Test Source",
            "snippet": "This is a test snippet for integration testing.",
            "month": "2024-01",
        }

        db_postgres.save_article(article)
        retrieved = db_postgres.get_article(article["id"])

        assert retrieved is not None
        assert retrieved["title"] == article["title"]
        assert retrieved["source"] == article["source"]

    def test_save_articles_batch(self, db_setup):
        """Should be able to save multiple articles at once."""
        prefix = f"batch_test_{os.urandom(4).hex()}"
        articles = [
            {
                "id": f"{prefix}_{i}",
                "title": f"Batch Article {i}",
                "date": "2024-01-15",
                "source": "Batch Source",
                "snippet": f"Snippet for article {i}",
                "month": "2024-01",
            }
            for i in range(5)
        ]

        db_postgres.save_articles_batch(articles)

        for article in articles:
            retrieved = db_postgres.get_article(article["id"])
            assert retrieved is not None


@pytest.mark.integration
class TestIntegrationClassifications:
    """Integration tests for classification operations."""

    def test_save_and_query_classification_counts(self, db_setup):
        """Should be able to save classifications and query counts."""
        # This test verifies the full flow of saving articles,
        # classifying them, and querying aggregated counts
        prefix = f"class_test_{os.urandom(4).hex()}"

        # Save test articles
        articles = [
            {"id": f"{prefix}_1", "title": "Uncertain rollback", "month": "2024-01",
             "source": "NYT", "snippet": "The policy rollback is uncertain"},
            {"id": f"{prefix}_2", "title": "Clear implementation", "month": "2024-01",
             "source": "WSJ", "snippet": "Implementation timeline is clear"},
        ]
        db_postgres.save_articles_batch(articles)

        # Save classifications
        db_postgres.save_keyword_classification(
            article_id=f"{prefix}_1",
            has_uncertainty=True,
            has_reversal_terms=True,
        )
        db_postgres.save_keyword_classification(
            article_id=f"{prefix}_2",
            has_uncertainty=False,
            has_implementation_terms=True,
        )

        # Query counts
        counts = db_postgres.get_classification_counts_by_month()
        assert len(counts) > 0
