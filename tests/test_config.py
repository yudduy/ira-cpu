"""
Tests for config.py - Configuration validation

Verifies all required constants exist and have correct types.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class TestDateConfiguration:
    """Tests for date range configuration."""

    def test_start_date_exists(self):
        """START_DATE constant should be defined."""
        assert hasattr(config, "START_DATE")

    def test_start_date_format(self):
        """START_DATE should be in YYYY-MM-DD format."""
        from datetime import datetime
        # Should not raise ValueError
        datetime.strptime(config.START_DATE, "%Y-%m-%d")

    def test_end_date_exists(self):
        """END_DATE constant should be defined."""
        assert hasattr(config, "END_DATE")

    def test_end_date_format(self):
        """END_DATE should be in YYYY-MM-DD format."""
        from datetime import datetime
        datetime.strptime(config.END_DATE, "%Y-%m-%d")

    def test_date_range_valid(self):
        """END_DATE should be after or equal to START_DATE."""
        from datetime import datetime
        start = datetime.strptime(config.START_DATE, "%Y-%m-%d")
        end = datetime.strptime(config.END_DATE, "%Y-%m-%d")
        assert end >= start, "END_DATE must be >= START_DATE"


class TestSourceConfiguration:
    """Tests for news source configuration."""

    def test_source_ids_exists(self):
        """SOURCE_IDS constant should be defined."""
        assert hasattr(config, "SOURCE_IDS")

    def test_source_ids_is_dict(self):
        """SOURCE_IDS should be a dictionary."""
        assert isinstance(config.SOURCE_IDS, dict)

    def test_source_ids_has_expected_keys(self):
        """SOURCE_IDS should contain expected news sources."""
        expected_sources = [
            "Financial Times",
            "Wall Street Journal",
            "New York Times",
            "Washington Post",
            "Reuters",
            "Bloomberg",
        ]
        for source in expected_sources:
            assert source in config.SOURCE_IDS, f"Missing source: {source}"


class TestKeywordConfiguration:
    """Tests for keyword lists configuration."""

    def test_climate_terms_exists(self):
        """CLIMATE_TERMS constant should be defined."""
        assert hasattr(config, "CLIMATE_TERMS")

    def test_climate_terms_is_list(self):
        """CLIMATE_TERMS should be a list."""
        assert isinstance(config.CLIMATE_TERMS, list)

    def test_climate_terms_not_empty(self):
        """CLIMATE_TERMS should not be empty."""
        assert len(config.CLIMATE_TERMS) > 0

    def test_climate_terms_contains_strings(self):
        """All CLIMATE_TERMS should be strings."""
        for term in config.CLIMATE_TERMS:
            assert isinstance(term, str), f"Expected string, got {type(term)}"

    def test_climate_terms_has_key_words(self):
        """CLIMATE_TERMS should contain essential keywords."""
        essential = ["climate", "renewable", "carbon"]
        for keyword in essential:
            assert any(keyword in term.lower() for term in config.CLIMATE_TERMS), \
                f"Missing essential keyword: {keyword}"

    def test_policy_terms_exists(self):
        """POLICY_TERMS constant should be defined."""
        assert hasattr(config, "POLICY_TERMS")

    def test_policy_terms_is_list(self):
        """POLICY_TERMS should be a list."""
        assert isinstance(config.POLICY_TERMS, list)

    def test_policy_terms_not_empty(self):
        """POLICY_TERMS should not be empty."""
        assert len(config.POLICY_TERMS) > 0

    def test_policy_terms_has_key_words(self):
        """POLICY_TERMS should contain essential keywords."""
        essential = ["policy", "regulation", "Congress"]
        for keyword in essential:
            assert keyword in config.POLICY_TERMS, f"Missing: {keyword}"

    def test_uncertainty_terms_exists(self):
        """UNCERTAINTY_TERMS constant should be defined."""
        assert hasattr(config, "UNCERTAINTY_TERMS")

    def test_uncertainty_terms_is_list(self):
        """UNCERTAINTY_TERMS should be a list."""
        assert isinstance(config.UNCERTAINTY_TERMS, list)

    def test_uncertainty_terms_not_empty(self):
        """UNCERTAINTY_TERMS should not be empty."""
        assert len(config.UNCERTAINTY_TERMS) > 0

    def test_uncertainty_terms_has_key_words(self):
        """UNCERTAINTY_TERMS should contain core uncertainty indicators."""
        # Core uncertainty terms (direction-neutral)
        expected = ["uncertain", "risk", "unclear"]
        for keyword in expected:
            assert any(keyword in term.lower() for term in config.UNCERTAINTY_TERMS), \
                f"Missing uncertainty term: {keyword}"

    def test_downside_terms_exists(self):
        """DOWNSIDE_TERMS constant should be defined for CPU-Down index."""
        assert hasattr(config, "DOWNSIDE_TERMS")
        assert isinstance(config.DOWNSIDE_TERMS, list)
        assert len(config.DOWNSIDE_TERMS) > 0

    def test_downside_terms_has_key_words(self):
        """DOWNSIDE_TERMS should contain rollback/negative indicators."""
        expected = ["rollback", "repeal", "delay", "cut"]
        for keyword in expected:
            assert any(keyword in term.lower() for term in config.DOWNSIDE_TERMS), \
                f"Missing downside term: {keyword}"

    def test_upside_terms_exists(self):
        """UPSIDE_TERMS constant should be defined for CPU-Up index."""
        assert hasattr(config, "UPSIDE_TERMS")
        assert isinstance(config.UPSIDE_TERMS, list)
        assert len(config.UPSIDE_TERMS) > 0

    def test_upside_terms_has_key_words(self):
        """UPSIDE_TERMS should contain expansion/positive indicators."""
        expected = ["expand", "invest", "strengthen"]
        for keyword in expected:
            assert any(keyword in term.lower() for term in config.UPSIDE_TERMS), \
                f"Missing upside term: {keyword}"


class TestLLMConfiguration:
    """Tests for LLM/AI configuration."""

    def test_llm_model_exists(self):
        """LLM_MODEL constant should be defined."""
        assert hasattr(config, "LLM_MODEL")

    def test_llm_model_is_string(self):
        """LLM_MODEL should be a string."""
        assert isinstance(config.LLM_MODEL, str)

    def test_llm_sample_size_exists(self):
        """LLM_SAMPLE_SIZE constant should be defined."""
        assert hasattr(config, "LLM_SAMPLE_SIZE")

    def test_llm_sample_size_is_positive_int(self):
        """LLM_SAMPLE_SIZE should be a positive integer."""
        assert isinstance(config.LLM_SAMPLE_SIZE, int)
        assert config.LLM_SAMPLE_SIZE > 0

    def test_llm_temperature_exists(self):
        """LLM_TEMPERATURE constant should be defined."""
        assert hasattr(config, "LLM_TEMPERATURE")

    def test_llm_temperature_in_valid_range(self):
        """LLM_TEMPERATURE should be between 0 and 2."""
        assert 0.0 <= config.LLM_TEMPERATURE <= 2.0


class TestAPIConfiguration:
    """Tests for API settings configuration."""

    def test_base_url_exists(self):
        """LEXISNEXIS_BASE_URL constant should be defined."""
        assert hasattr(config, "LEXISNEXIS_BASE_URL")

    def test_base_url_is_https(self):
        """LEXISNEXIS_BASE_URL should use HTTPS."""
        assert config.LEXISNEXIS_BASE_URL.startswith("https://")

    def test_max_results_exists(self):
        """MAX_RESULTS_PER_QUERY constant should be defined."""
        assert hasattr(config, "MAX_RESULTS_PER_QUERY")

    def test_max_results_is_positive(self):
        """MAX_RESULTS_PER_QUERY should be positive."""
        assert config.MAX_RESULTS_PER_QUERY > 0

    def test_request_delay_exists(self):
        """REQUEST_DELAY_SECONDS constant should be defined."""
        assert hasattr(config, "REQUEST_DELAY_SECONDS")

    def test_request_delay_is_non_negative(self):
        """REQUEST_DELAY_SECONDS should be non-negative."""
        assert config.REQUEST_DELAY_SECONDS >= 0


class TestFilePathConfiguration:
    """Tests for file path configuration."""

    def test_db_path_exists(self):
        """DB_PATH constant should be defined."""
        assert hasattr(config, "DB_PATH")

    def test_db_path_is_string(self):
        """DB_PATH should be a string."""
        assert isinstance(config.DB_PATH, str)

    def test_db_path_ends_with_db(self):
        """DB_PATH should end with .db extension."""
        assert config.DB_PATH.endswith(".db")

    def test_export_dir_exists(self):
        """EXPORT_DIR constant should be defined."""
        assert hasattr(config, "EXPORT_DIR")

    def test_export_dir_is_string(self):
        """EXPORT_DIR should be a string."""
        assert isinstance(config.EXPORT_DIR, str)
