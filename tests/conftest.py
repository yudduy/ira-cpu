"""
Shared pytest fixtures for CPU Index Builder tests.

This module provides common fixtures used across test files:
- Mock API responses
- Sample data generators
- Environment mocks
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for package imports
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as requiring external services (PostgreSQL, APIs)"
    )


# =============================================================================
# API RESPONSE FIXTURES
# =============================================================================

@pytest.fixture
def mock_api_count_response():
    """Mock response for fetch_count API call."""
    return {
        "@odata.count": 150,
        "value": [{"ResultId": "test_1", "Title": "Test Article"}]
    }


@pytest.fixture
def mock_api_metadata_response():
    """Mock response for fetch_metadata API call."""
    return {
        "@odata.count": 3,
        "value": [
            {
                "ResultId": "article_001",
                "Title": "Climate Policy Update",
                "Date": "2024-01-15",
                "Source": {"Name": "Reuters"},
                "Overview": "Article about climate policy changes..."
            },
            {
                "ResultId": "article_002",
                "Title": "Energy Regulation News",
                "Date": "2024-01-16",
                "Source": {"Name": "Bloomberg"},
                "Overview": "New energy regulations announced..."
            },
            {
                "ResultId": "article_003",
                "Title": "IRA Implementation Uncertainty",
                "Date": "2024-01-17",
                "Source": {"Name": "Politico"},
                "Overview": "Uncertainty surrounds IRA guidance..."
            },
        ]
    }


@pytest.fixture
def mock_token():
    """Mock API token."""
    return "mock_test_token_12345"


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_article():
    """Sample article for classification tests."""
    return {
        "id": "test_article_001",
        "month": "2024-01",
        "title": "Climate Policy Uncertainty Rises Amid Political Debate",
        "source": "Reuters",
        "date": "2024-01-15",
        "snippet": "Lawmakers debate the future of IRA tax credits...",
        "full_text": "Full article text about climate policy uncertainty..."
    }


@pytest.fixture
def sample_progress_data():
    """Sample progress records for testing."""
    return [
        {"month": "2024-01", "query_type": "denominator", "count": 150},
        {"month": "2024-01", "query_type": "numerator", "count": 30},
        {"month": "2024-02", "query_type": "denominator", "count": 180},
        {"month": "2024-02", "query_type": "numerator", "count": 45},
    ]


@pytest.fixture
def sample_index_values():
    """Sample index values for testing calculations."""
    return [
        {"month": "2024-01", "denominator": 150, "numerator": 30, "raw_ratio": 0.20},
        {"month": "2024-02", "denominator": 180, "numerator": 45, "raw_ratio": 0.25},
        {"month": "2024-03", "denominator": 160, "numerator": 32, "raw_ratio": 0.20},
        {"month": "2024-04", "denominator": 200, "numerator": 60, "raw_ratio": 0.30},
    ]


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for API tests."""
    monkeypatch.setenv("clientid", "test_client_id")
    monkeypatch.setenv("clientsecret", "test_client_secret")
    monkeypatch.setenv("lntoken", "mock_cached_token")
    monkeypatch.setenv("lnexpire", str(int(__import__("time").time()) + 3600))


@pytest.fixture
def mock_openai_env(monkeypatch):
    """Set up mock OpenAI environment for classifier tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")


# =============================================================================
# MOCK HELPERS
# =============================================================================

@pytest.fixture
def mock_requests_get():
    """Create a mock for requests.get with configurable responses."""
    with patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_requests_post():
    """Create a mock for requests.post with configurable responses."""
    with patch("requests.post") as mock_post:
        yield mock_post
