"""
Tests for db.py - Database operations

Uses temporary database files to ensure test isolation.
Never touches the real production database.
"""

import csv
import os
import sqlite3
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDatabaseConnection:
    """Tests for database connection and initialization."""

    def test_get_connection_creates_directory(self, temp_db_config, tmp_path):
        """get_connection should create parent directories if needed."""
        import db

        nested_path = tmp_path / "nested" / "deep" / "test.db"
        import config
        config.DB_PATH = str(nested_path)

        conn = db.get_connection()
        assert nested_path.parent.exists()
        conn.close()

    def test_get_connection_returns_sqlite_connection(self, temp_db_config):
        """get_connection should return a valid SQLite connection."""
        import db

        conn = db.get_connection()
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_get_connection_enables_row_factory(self, temp_db_config):
        """get_connection should enable dict-like row access."""
        import db

        conn = db.get_connection()
        assert conn.row_factory == sqlite3.Row
        conn.close()


class TestInitDb:
    """Tests for database initialization."""

    def test_init_db_creates_tables(self, temp_db_config):
        """init_db should create all required tables."""
        import db

        db.init_db()

        conn = db.get_connection()
        cursor = conn.cursor()

        # Check articles table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
        )
        assert cursor.fetchone() is not None

        # Check progress table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='progress'"
        )
        assert cursor.fetchone() is not None

        # Check classifications table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='classifications'"
        )
        assert cursor.fetchone() is not None

        # Check index_values table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='index_values'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_init_db_is_idempotent(self, temp_db_config):
        """init_db should be safe to call multiple times."""
        import db

        # Should not raise any errors
        db.init_db()
        db.init_db()
        db.init_db()

        # Tables should still exist
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
        count = cursor.fetchone()[0]
        assert count >= 4  # At least 4 tables
        conn.close()


class TestMonthCountOperations:
    """Tests for save_month_count and get_month_count."""

    def test_save_month_count_inserts_new_record(self, initialized_db):
        """save_month_count should insert a new record."""
        import db

        db.save_month_count("2024-06", "denominator", 200)

        result = db.get_month_count("2024-06", "denominator")
        assert result == 200

    def test_save_month_count_updates_existing_record(self, initialized_db):
        """save_month_count should update existing records."""
        import db

        db.save_month_count("2024-06", "denominator", 200)
        db.save_month_count("2024-06", "denominator", 250)

        result = db.get_month_count("2024-06", "denominator")
        assert result == 250

    def test_get_month_count_returns_none_for_missing(self, initialized_db):
        """get_month_count should return None for missing records."""
        import db

        result = db.get_month_count("1999-01", "denominator")
        assert result is None

    def test_save_both_denominator_and_numerator(self, initialized_db):
        """Should handle both query types for same month."""
        import db

        db.save_month_count("2024-07", "denominator", 180)
        db.save_month_count("2024-07", "numerator", 36)

        assert db.get_month_count("2024-07", "denominator") == 180
        assert db.get_month_count("2024-07", "numerator") == 36


class TestProgressTracking:
    """Tests for progress tracking functions."""

    def test_get_all_progress_returns_list(self, initialized_db):
        """get_all_progress should return a list."""
        import db

        result = db.get_all_progress()
        assert isinstance(result, list)

    def test_get_all_progress_returns_dict_items(self, populated_db):
        """get_all_progress should return dicts with expected keys."""
        import db

        result = db.get_all_progress()
        assert len(result) > 0

        for item in result:
            assert "month" in item
            assert "query_type" in item
            assert "count" in item

    def test_get_completed_months_returns_set(self, initialized_db):
        """get_completed_months should return a set."""
        import db

        result = db.get_completed_months()
        assert isinstance(result, set)

    def test_get_completed_months_only_complete_months(self, initialized_db):
        """get_completed_months should only include months with both counts."""
        import db

        # Add complete month (both denominator and numerator)
        db.save_month_count("2024-08", "denominator", 150)
        db.save_month_count("2024-08", "numerator", 30)

        # Add incomplete month (only denominator)
        db.save_month_count("2024-09", "denominator", 160)

        completed = db.get_completed_months()
        assert "2024-08" in completed
        assert "2024-09" not in completed


class TestArticleOperations:
    """Tests for article save operations."""

    def test_save_article_inserts_record(self, initialized_db, sample_article):
        """save_article should insert an article record."""
        import db

        db.save_article(sample_article)

        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM articles WHERE id = ?", (sample_article["id"],))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row["title"] == sample_article["title"]

    def test_save_article_ignores_duplicates(self, initialized_db, sample_article):
        """save_article should ignore duplicate IDs."""
        import db

        db.save_article(sample_article)

        # Modify title and try to insert again
        modified = sample_article.copy()
        modified["title"] = "Modified Title"
        db.save_article(modified)

        # Should still have original title (INSERT OR IGNORE)
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT title FROM articles WHERE id = ?", (sample_article["id"],))
        row = cursor.fetchone()
        conn.close()

        assert row["title"] == sample_article["title"]


class TestClassificationOperations:
    """Tests for classification save operations."""

    def test_save_classification_inserts_record(self, initialized_db):
        """save_classification should insert a classification record."""
        import db

        db.save_classification(
            article_id="test_article_001",
            is_climate_policy=True,
            has_uncertainty=True,
            reasoning="Article discusses IRA tax credit delays",
            confidence="high"
        )

        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM classifications WHERE article_id = ?",
            ("test_article_001",)
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row["is_climate_policy"] == 1
        assert row["has_uncertainty"] == 1
        assert row["confidence"] == "high"

    def test_save_classification_updates_existing(self, initialized_db):
        """save_classification should update existing records."""
        import db

        db.save_classification("test_001", True, True, "First reasoning", "high")
        db.save_classification("test_001", False, False, "Updated reasoning", "low")

        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM classifications WHERE article_id = ?",
            ("test_001",)
        )
        row = cursor.fetchone()
        conn.close()

        assert row["is_climate_policy"] == 0
        assert row["reasoning"] == "Updated reasoning"


class TestIndexValueOperations:
    """Tests for index value operations."""

    def test_save_index_value_inserts_record(self, initialized_db):
        """save_index_value should insert an index record."""
        import db

        db.save_index_value(
            month="2024-10",
            denominator=200,
            numerator=50,
            raw_ratio=0.25,
            normalized=125.0
        )

        values = db.get_all_index_values()
        assert len(values) == 1
        assert values[0]["month"] == "2024-10"
        assert values[0]["normalized"] == 125.0

    def test_get_all_index_values_ordered_by_month(self, initialized_db):
        """get_all_index_values should return results ordered by month."""
        import db

        db.save_index_value("2024-03", 160, 32, 0.20, 100.0)
        db.save_index_value("2024-01", 150, 30, 0.20, 100.0)
        db.save_index_value("2024-02", 180, 45, 0.25, 125.0)

        values = db.get_all_index_values()
        months = [v["month"] for v in values]
        assert months == sorted(months)


class TestSchemaMigration:
    """Tests for database schema migration on init_db()."""

    def test_init_db_migrates_old_index_values_schema(self, temp_db_config):
        """init_db should add missing columns to existing index_values table."""
        import db

        # Step 1: Create database with OLD schema (5 columns, no directional)
        conn = db.get_connection()
        cursor = conn.cursor()

        # Create old-style index_values table without directional columns
        cursor.execute("""
            CREATE TABLE index_values (
                month TEXT PRIMARY KEY,
                denominator INTEGER NOT NULL,
                numerator INTEGER NOT NULL,
                raw_ratio REAL NOT NULL,
                normalized REAL,
                calculated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert some data in old format
        cursor.execute("""
            INSERT INTO index_values (month, denominator, numerator, raw_ratio, normalized)
            VALUES ('2024-01', 100, 20, 0.2, 100.0)
        """)
        conn.commit()
        conn.close()

        # Step 2: Run init_db() - should migrate the table
        db.init_db()

        # Step 3: Verify new columns exist
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(index_values)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        # Should have all new directional columns
        expected_new_columns = {
            "numerator_down", "raw_ratio_down", "normalized_down",
            "numerator_up", "raw_ratio_up", "normalized_up",
            "cpu_direction"
        }
        for col in expected_new_columns:
            assert col in columns, f"Missing column after migration: {col}"

    def test_init_db_preserves_existing_data_after_migration(self, temp_db_config):
        """init_db migration should not lose existing data."""
        import db

        # Create old schema with data
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE index_values (
                month TEXT PRIMARY KEY,
                denominator INTEGER NOT NULL,
                numerator INTEGER NOT NULL,
                raw_ratio REAL NOT NULL,
                normalized REAL
            )
        """)
        cursor.execute("""
            INSERT INTO index_values (month, denominator, numerator, raw_ratio, normalized)
            VALUES ('2024-01', 150, 30, 0.2, 100.0)
        """)
        conn.commit()
        conn.close()

        # Run migration
        db.init_db()

        # Verify data preserved
        values = db.get_all_index_values()
        assert len(values) == 1
        assert values[0]["month"] == "2024-01"
        assert values[0]["denominator"] == 150
        assert values[0]["numerator"] == 30

    def test_init_db_migration_is_idempotent(self, temp_db_config):
        """Running init_db multiple times should not cause errors."""
        import db

        # Create old schema
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE index_values (
                month TEXT PRIMARY KEY,
                denominator INTEGER NOT NULL,
                numerator INTEGER NOT NULL,
                raw_ratio REAL NOT NULL,
                normalized REAL
            )
        """)
        conn.commit()
        conn.close()

        # Run init_db multiple times - should not raise
        db.init_db()
        db.init_db()
        db.init_db()

        # Should still work
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(index_values)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "cpu_direction" in columns


class TestExportToCsv:
    """Tests for CSV export functionality."""

    def test_export_to_csv_creates_file(self, populated_db, tmp_path):
        """export_to_csv should create a CSV file."""
        import db

        output_path = tmp_path / "exports" / "test_index.csv"
        count = db.export_to_csv(str(output_path))

        assert output_path.exists()
        assert count > 0

    def test_export_to_csv_has_correct_headers(self, populated_db, tmp_path):
        """export_to_csv should include correct headers."""
        import db

        output_path = tmp_path / "test_export.csv"
        db.export_to_csv(str(output_path))

        with open(output_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)

        # Now includes directional indices (CPU-Down, CPU-Up)
        expected_headers = [
            "month", "denominator",
            "numerator", "raw_ratio", "normalized",
            "numerator_down", "raw_ratio_down", "normalized_down",
            "numerator_up", "raw_ratio_up", "normalized_up",
            "cpu_direction",
        ]
        assert headers == expected_headers

    def test_export_to_csv_raises_on_empty_db(self, initialized_db, tmp_path):
        """export_to_csv should raise ValueError if no data."""
        import db

        output_path = tmp_path / "empty_export.csv"

        with pytest.raises(ValueError, match="No index values"):
            db.export_to_csv(str(output_path))

    def test_export_to_csv_creates_parent_dirs(self, populated_db, tmp_path):
        """export_to_csv should create parent directories if needed."""
        import db

        output_path = tmp_path / "nested" / "dirs" / "export.csv"
        db.export_to_csv(str(output_path))

        assert output_path.exists()
