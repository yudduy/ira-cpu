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

    def test_database_url_exists(self):
        """DATABASE_URL constant should be defined."""
        assert hasattr(config, "DATABASE_URL")

    def test_database_url_is_string(self):
        """DATABASE_URL should be a string."""
        assert isinstance(config.DATABASE_URL, str)

    def test_database_url_is_postgresql(self):
        """DATABASE_URL should be a PostgreSQL connection string."""
        assert config.DATABASE_URL.startswith("postgresql://")

    def test_export_dir_exists(self):
        """EXPORT_DIR constant should be defined."""
        assert hasattr(config, "EXPORT_DIR")

    def test_export_dir_is_string(self):
        """EXPORT_DIR should be a string."""
        assert isinstance(config.EXPORT_DIR, str)


class TestNewKeywordCategories:
    """Tests for new RA1-specific keyword categories."""

    def test_implementation_terms_exists(self):
        """IMPLEMENTATION_TERMS should be defined for CPU_impl index."""
        assert hasattr(config, "IMPLEMENTATION_TERMS")
        assert isinstance(config.IMPLEMENTATION_TERMS, list)
        assert len(config.IMPLEMENTATION_TERMS) > 0

    def test_implementation_terms_has_key_words(self):
        """IMPLEMENTATION_TERMS should contain implementation-related keywords."""
        expected = ["delay", "guidance", "approval", "timeline"]
        for keyword in expected:
            assert any(keyword in term.lower() for term in config.IMPLEMENTATION_TERMS), \
                f"Missing implementation term: {keyword}"

    def test_reversal_terms_exists(self):
        """REVERSAL_TERMS should be defined for CPU_reversal index."""
        assert hasattr(config, "REVERSAL_TERMS")
        assert isinstance(config.REVERSAL_TERMS, list)
        assert len(config.REVERSAL_TERMS) > 0

    def test_reversal_terms_has_key_words(self):
        """REVERSAL_TERMS should contain reversal/rollback keywords."""
        expected = ["rollback", "repeal", "overturn", "terminate"]
        for keyword in expected:
            assert any(keyword in term.lower() for term in config.REVERSAL_TERMS), \
                f"Missing reversal term: {keyword}"

    def test_regime_ira_terms_exists(self):
        """REGIME_IRA_TERMS should be defined for Policy Regime Salience."""
        assert hasattr(config, "REGIME_IRA_TERMS")
        assert isinstance(config.REGIME_IRA_TERMS, list)
        assert len(config.REGIME_IRA_TERMS) > 0

    def test_regime_ira_terms_has_key_words(self):
        """REGIME_IRA_TERMS should contain IRA-specific terms."""
        # Check for main IRA reference
        assert any("Inflation Reduction Act" in term for term in config.REGIME_IRA_TERMS)

    def test_regime_obbba_terms_exists(self):
        """REGIME_OBBBA_TERMS should be defined for Policy Regime Salience."""
        assert hasattr(config, "REGIME_OBBBA_TERMS")
        assert isinstance(config.REGIME_OBBBA_TERMS, list)
        assert len(config.REGIME_OBBBA_TERMS) > 0

    def test_regime_obbba_terms_has_key_words(self):
        """REGIME_OBBBA_TERMS should contain OBBBA-specific terms."""
        assert any("OBBBA" in term or "Beautiful Bill" in term
                   for term in config.REGIME_OBBBA_TERMS)


class TestPlaceboKeywords:
    """Tests for placebo/control index keywords."""

    def test_trade_terms_exists(self):
        """TRADE_TERMS should be defined for trade policy placebo."""
        assert hasattr(config, "TRADE_TERMS")
        assert isinstance(config.TRADE_TERMS, list)
        assert len(config.TRADE_TERMS) > 0

    def test_trade_terms_has_key_words(self):
        """TRADE_TERMS should contain trade policy keywords."""
        expected = ["trade", "tariff"]
        for keyword in expected:
            assert any(keyword in term.lower() for term in config.TRADE_TERMS), \
                f"Missing trade term: {keyword}"

    def test_monetary_terms_exists(self):
        """MONETARY_TERMS should be defined for monetary policy placebo."""
        assert hasattr(config, "MONETARY_TERMS")
        assert isinstance(config.MONETARY_TERMS, list)
        assert len(config.MONETARY_TERMS) > 0

    def test_monetary_terms_has_key_words(self):
        """MONETARY_TERMS should contain monetary policy keywords."""
        expected = ["Federal Reserve", "interest rate"]
        for keyword in expected:
            assert any(keyword in term for term in config.MONETARY_TERMS), \
                f"Missing monetary term: {keyword}"


class TestBBDOutlets:
    """Tests for BBD newspaper outlet configuration."""

    def test_bbd_outlets_exists(self):
        """BBD_OUTLETS should be defined for outlet-level analysis."""
        assert hasattr(config, "BBD_OUTLETS")
        assert isinstance(config.BBD_OUTLETS, list)

    def test_bbd_outlets_has_eight_newspapers(self):
        """BBD_OUTLETS should contain exactly 8 newspapers (per BBD methodology)."""
        assert len(config.BBD_OUTLETS) == 8

    def test_bbd_outlets_has_key_papers(self):
        """BBD_OUTLETS should contain major BBD newspapers."""
        expected = ["New York Times", "Wall Street Journal", "USA Today"]
        for paper in expected:
            assert paper in config.BBD_OUTLETS, f"Missing BBD outlet: {paper}"


class TestDatabaseConfiguration:
    """Tests for database configuration."""

    def test_database_url_exists(self):
        """DATABASE_URL should be defined for PostgreSQL connection."""
        assert hasattr(config, "DATABASE_URL")
        assert isinstance(config.DATABASE_URL, str)

    def test_database_url_is_postgresql(self):
        """DATABASE_URL should be a PostgreSQL connection string."""
        assert config.DATABASE_URL.startswith("postgresql://")

    def test_llm_accuracy_threshold_exists(self):
        """LLM_ACCURACY_THRESHOLD should be defined."""
        assert hasattr(config, "LLM_ACCURACY_THRESHOLD")
        assert isinstance(config.LLM_ACCURACY_THRESHOLD, float)
        assert 0.0 < config.LLM_ACCURACY_THRESHOLD <= 1.0
