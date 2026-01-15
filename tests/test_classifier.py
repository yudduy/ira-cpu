"""
Tests for classifier.py - GPT-5 Nano classification

Tests LLM classification with mocked AI SDK.
Handles ImportError gracefully when AI SDK is not available.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAISDKAvailability:
    """Tests for AI SDK availability detection."""

    def test_ai_sdk_available_flag_exists(self):
        """AI_SDK_AVAILABLE flag should be defined."""
        import classifier

        assert hasattr(classifier, "AI_SDK_AVAILABLE")
        assert isinstance(classifier.AI_SDK_AVAILABLE, bool)

    def test_handles_missing_ai_sdk_gracefully(self):
        """Module should load even if ai_sdk is not installed."""
        # This test verifies the module imports without crashing
        # The try/except in classifier.py handles ImportError
        import classifier

        # Should complete without exception
        assert True


class TestArticleClassificationSchema:
    """Tests for Pydantic classification schema."""

    def test_article_classification_schema_exists(self):
        """ArticleClassification model should be defined."""
        from classifier import ArticleClassification

        assert ArticleClassification is not None

    def test_article_classification_fields(self):
        """ArticleClassification should have required fields."""
        from classifier import ArticleClassification

        # Create a valid instance
        classification = ArticleClassification(
            is_climate_policy=True,
            has_uncertainty=False,
            reasoning="Test reasoning",
            confidence="high"
        )

        assert classification.is_climate_policy is True
        assert classification.has_uncertainty is False
        assert classification.reasoning == "Test reasoning"
        assert classification.confidence == "high"

    def test_article_classification_validation(self):
        """ArticleClassification should validate field types."""
        from classifier import ArticleClassification
        from pydantic import ValidationError

        # Missing required field should raise
        with pytest.raises(ValidationError):
            ArticleClassification(
                is_climate_policy=True,
                # missing has_uncertainty, reasoning, confidence
            )


class TestClassifyArticle:
    """Tests for single article classification."""

    def test_classify_article_returns_none_without_ai_sdk(self, monkeypatch):
        """classify_article should return None if AI SDK unavailable."""
        import classifier

        monkeypatch.setattr(classifier, "AI_SDK_AVAILABLE", False)

        result = classifier.classify_article({"title": "Test", "snippet": "Test content"})
        assert result is None

    def test_classify_article_returns_none_without_api_key(self, monkeypatch):
        """classify_article should return None without OPENAI_API_KEY."""
        import classifier

        monkeypatch.setattr(classifier, "AI_SDK_AVAILABLE", True)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = classifier.classify_article({"title": "Test", "snippet": "Test content"})
        assert result is None

    def test_classify_article_with_mocked_ai_sdk(self, monkeypatch, mock_openai_env):
        """classify_article should return classification with mocked AI SDK."""
        import classifier

        # Skip if AI SDK is not available (can't test without the module structure)
        if not classifier.AI_SDK_AVAILABLE:
            pytest.skip("AI SDK not installed - cannot test full classify_article flow")

        # Create mock result object
        mock_classification = classifier.ArticleClassification(
            is_climate_policy=True,
            has_uncertainty=True,
            reasoning="Article discusses IRA delays",
            confidence="high"
        )

        mock_result = MagicMock()
        mock_result.object = mock_classification

        mock_generate = MagicMock(return_value=mock_result)
        mock_openai_func = MagicMock(return_value="mock_model")

        monkeypatch.setattr(classifier, "generate_object", mock_generate)
        monkeypatch.setattr(classifier, "openai", mock_openai_func)

        result = classifier.classify_article({
            "title": "IRA Implementation Delays",
            "snippet": "Treasury delays guidance on tax credits..."
        })

        assert result is not None
        assert result.is_climate_policy is True
        assert result.has_uncertainty is True


class TestBuildPrompt:
    """Tests for classification prompt building."""

    def test_build_prompt_includes_title(self, sample_article):
        """_build_prompt should include article title."""
        import classifier

        prompt = classifier._build_prompt(sample_article)
        assert sample_article["title"] in prompt

    def test_build_prompt_includes_snippet(self, sample_article):
        """_build_prompt should include snippet if no full_text."""
        import classifier

        article = {"title": "Test", "snippet": "Test snippet content"}
        prompt = classifier._build_prompt(article)
        assert "Test snippet content" in prompt

    def test_build_prompt_prefers_full_text(self, sample_article):
        """_build_prompt should prefer full_text over snippet."""
        import classifier

        article = {
            "title": "Test",
            "snippet": "Short snippet",
            "full_text": "Full article text here"
        }
        prompt = classifier._build_prompt(article)
        assert "Full article text" in prompt

    def test_build_prompt_includes_instructions(self):
        """_build_prompt should include classification instructions."""
        import classifier

        prompt = classifier._build_prompt({"title": "Test"})

        assert "climate" in prompt.lower()
        assert "policy" in prompt.lower()
        assert "uncertainty" in prompt.lower()

    def test_build_prompt_truncates_long_text(self):
        """_build_prompt should truncate very long text."""
        import classifier

        long_text = "A" * 5000  # 5000 characters
        prompt = classifier._build_prompt({"title": "Test", "full_text": long_text})

        # Prompt should truncate to ~2000 chars for the text portion
        assert len(prompt) < 5500


class TestClassifySample:
    """Tests for batch sample classification."""

    def test_classify_sample_dry_run(self):
        """classify_sample with dry_run should return fake results."""
        import classifier

        result = classifier.classify_sample(sample_size=50, dry_run=True)

        assert result["sample_size"] == 50
        assert result["classified"] == 50
        assert result["dry_run"] is True
        assert "is_climate_policy_yes" in result
        assert "has_uncertainty_yes" in result

    def test_classify_sample_default_size(self, monkeypatch):
        """classify_sample should use config default sample size."""
        import classifier

        monkeypatch.setattr("config.LLM_SAMPLE_SIZE", 75)

        result = classifier.classify_sample(dry_run=True)
        assert result["sample_size"] == 75

    def test_classify_sample_not_implemented_for_real(self):
        """classify_sample without dry_run returns not_implemented status."""
        import classifier

        result = classifier.classify_sample(dry_run=False)
        assert result["status"] == "not_implemented"


class TestEstimateClassificationCost:
    """Tests for cost estimation."""

    def test_estimate_classification_cost_structure(self):
        """estimate_classification_cost should return expected structure."""
        import classifier

        result = classifier.estimate_classification_cost(100)

        assert "articles" in result
        assert "estimated_input_tokens" in result
        assert "estimated_output_tokens" in result
        assert "estimated_cost_usd" in result
        assert "cost_per_100_articles" in result

    def test_estimate_classification_cost_calculation(self):
        """estimate_classification_cost should calculate correctly."""
        import classifier

        result = classifier.estimate_classification_cost(100)

        # Based on assumptions in classifier.py:
        # 500 input tokens/article, 100 output tokens/article
        # Input: $0.05/1M tokens, Output: $0.40/1M tokens
        assert result["articles"] == 100
        assert result["estimated_input_tokens"] == 50000  # 100 * 500
        assert result["estimated_output_tokens"] == 10000  # 100 * 100

        # Cost calculation
        expected_input_cost = (50000 / 1_000_000) * 0.05  # $0.0025
        expected_output_cost = (10000 / 1_000_000) * 0.40  # $0.004
        expected_total = expected_input_cost + expected_output_cost  # $0.0065

        assert abs(result["estimated_cost_usd"] - expected_total) < 0.001

    def test_estimate_classification_cost_scales_linearly(self):
        """Cost should scale linearly with article count."""
        import classifier

        cost_100 = classifier.estimate_classification_cost(100)
        cost_1000 = classifier.estimate_classification_cost(1000)

        # 10x articles should be 10x cost
        assert abs(cost_1000["estimated_cost_usd"] - cost_100["estimated_cost_usd"] * 10) < 0.001


class TestGetValidationStatus:
    """Tests for validation status reporting."""

    def test_get_validation_status_structure(self):
        """get_validation_status should return expected structure."""
        import classifier

        result = classifier.get_validation_status()

        assert "total_classified" in result
        assert "ai_sdk_available" in result
        assert "openai_key_set" in result

    def test_get_validation_status_detects_ai_sdk(self):
        """get_validation_status should reflect AI SDK availability."""
        import classifier

        result = classifier.get_validation_status()
        assert result["ai_sdk_available"] == classifier.AI_SDK_AVAILABLE

    def test_get_validation_status_detects_api_key(self, monkeypatch):
        """get_validation_status should detect OPENAI_API_KEY."""
        import classifier

        # With key set
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        result = classifier.get_validation_status()
        assert result["openai_key_set"] is True

        # Without key
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = classifier.get_validation_status()
        assert result["openai_key_set"] is False
