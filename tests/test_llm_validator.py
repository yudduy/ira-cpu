"""
Tests for llm_validator.py - LLM-based article validation pipeline

Validates keyword classification accuracy using GPT-5-nano with adaptive sampling.
Initial sample: 1000 articles, expand if accuracy < 85%.

Classification schema:
- is_climate_policy: bool - Does article discuss climate policy?
- uncertainty_type: "implementation" | "reversal" | "none" - Type of uncertainty
- certainty_level: 1-5 - LLM's confidence in classification
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSampleSelection:
    """Tests for adaptive sampling logic."""

    def test_select_initial_sample_size(self):
        """select_validation_sample should return up to 1000 articles initially."""
        from llm_validator import select_validation_sample

        # Create 2000 mock articles
        articles = [{"id": str(i), "title": f"Article {i}"} for i in range(2000)]

        sample = select_validation_sample(articles, initial_sample_size=1000)

        assert len(sample) == 1000

    def test_select_sample_smaller_than_requested(self):
        """select_validation_sample should return all if fewer than sample size."""
        from llm_validator import select_validation_sample

        articles = [{"id": str(i), "title": f"Article {i}"} for i in range(500)]

        sample = select_validation_sample(articles, initial_sample_size=1000)

        assert len(sample) == 500

    def test_select_sample_is_random(self):
        """select_validation_sample should return different samples on different calls."""
        from llm_validator import select_validation_sample

        articles = [{"id": str(i), "title": f"Article {i}"} for i in range(5000)]

        sample1 = select_validation_sample(articles, initial_sample_size=100, seed=42)
        sample2 = select_validation_sample(articles, initial_sample_size=100, seed=123)

        # Different seeds should give different samples
        ids1 = {a["id"] for a in sample1}
        ids2 = {a["id"] for a in sample2}
        assert ids1 != ids2

    def test_select_sample_reproducible_with_seed(self):
        """select_validation_sample should be reproducible with same seed."""
        from llm_validator import select_validation_sample

        articles = [{"id": str(i), "title": f"Article {i}"} for i in range(5000)]

        sample1 = select_validation_sample(articles, initial_sample_size=100, seed=42)
        sample2 = select_validation_sample(articles, initial_sample_size=100, seed=42)

        ids1 = [a["id"] for a in sample1]
        ids2 = [a["id"] for a in sample2]
        assert ids1 == ids2

    def test_stratified_sampling_by_category(self):
        """select_validation_sample should stratify by keyword category if available."""
        from llm_validator import select_validation_sample

        articles = [
            {"id": "1", "title": "A", "keyword_category": "implementation"},
            {"id": "2", "title": "B", "keyword_category": "implementation"},
            {"id": "3", "title": "C", "keyword_category": "reversal"},
            {"id": "4", "title": "D", "keyword_category": "reversal"},
            {"id": "5", "title": "E", "keyword_category": "implementation"},
            {"id": "6", "title": "F", "keyword_category": "reversal"},
        ]

        sample = select_validation_sample(
            articles, initial_sample_size=4, stratify_by="keyword_category", seed=42
        )

        # Should have mix of both categories
        categories = [a["keyword_category"] for a in sample]
        assert "implementation" in categories
        assert "reversal" in categories


class TestClassificationSchema:
    """Tests for the classification response schema."""

    def test_parse_valid_classification(self):
        """parse_classification should extract structured fields."""
        from llm_validator import parse_classification

        response = {
            "is_climate_policy": True,
            "uncertainty_type": "implementation",
            "certainty_level": 4,
            "reasoning": "Article discusses policy implementation uncertainty",
        }

        result = parse_classification(response)

        assert result["is_climate_policy"] is True
        assert result["uncertainty_type"] == "implementation"
        assert result["certainty_level"] == 4
        assert "reasoning" in result

    def test_parse_classification_validates_uncertainty_type(self):
        """parse_classification should validate uncertainty_type values."""
        from llm_validator import parse_classification

        response = {
            "is_climate_policy": True,
            "uncertainty_type": "invalid_type",
            "certainty_level": 3,
        }

        with pytest.raises(ValueError, match="Invalid uncertainty_type"):
            parse_classification(response)

    def test_parse_classification_validates_certainty_range(self):
        """parse_classification should validate certainty_level is 1-5."""
        from llm_validator import parse_classification

        response = {
            "is_climate_policy": True,
            "uncertainty_type": "reversal",
            "certainty_level": 10,  # Invalid
        }

        with pytest.raises(ValueError, match="certainty_level must be 1-5"):
            parse_classification(response)

    def test_valid_uncertainty_types(self):
        """Valid uncertainty types should be accepted."""
        from llm_validator import parse_classification

        for utype in ["implementation", "reversal", "none"]:
            response = {
                "is_climate_policy": True,
                "uncertainty_type": utype,
                "certainty_level": 3,
            }
            result = parse_classification(response)
            assert result["uncertainty_type"] == utype


class TestLLMClassification:
    """Tests for LLM-based classification."""

    @patch("llm_validator.openai")
    def test_classify_article_calls_openai(self, mock_openai):
        """classify_article should call OpenAI API with correct prompt."""
        from llm_validator import classify_article

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"is_climate_policy": true, "uncertainty_type": "implementation", "certainty_level": 4, "reasoning": "test"}'
                )
            )
        ]
        mock_openai.chat.completions.create.return_value = mock_response

        article = {
            "title": "Biden Administration Announces New Climate Policy",
            "content": "The administration unveiled plans...",
        }

        result = classify_article(article)

        # Verify OpenAI was called
        mock_openai.chat.completions.create.assert_called_once()
        call_args = mock_openai.chat.completions.create.call_args

        # Check model is from config (gpt-5-nano)
        import config
        assert call_args.kwargs["model"] == config.LLM_MODEL

        # Check response_format for JSON
        assert call_args.kwargs.get("response_format", {}).get("type") == "json_object"

    @patch("llm_validator.openai")
    def test_classify_article_returns_parsed_result(self, mock_openai):
        """classify_article should return parsed classification."""
        from llm_validator import classify_article

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"is_climate_policy": true, "uncertainty_type": "reversal", "certainty_level": 5, "reasoning": "Policy reversal discussed"}'
                )
            )
        ]
        mock_openai.chat.completions.create.return_value = mock_response

        article = {"title": "Trump Plans to Rollback Climate Rules", "content": "..."}

        result = classify_article(article)

        assert result["is_climate_policy"] is True
        assert result["uncertainty_type"] == "reversal"
        assert result["certainty_level"] == 5

    @patch("llm_validator.openai")
    def test_classify_article_handles_api_error(self, mock_openai):
        """classify_article should handle API errors gracefully."""
        from llm_validator import classify_article, ClassificationError

        mock_openai.chat.completions.create.side_effect = Exception("API Error")

        article = {"title": "Test", "content": "..."}

        with pytest.raises(ClassificationError):
            classify_article(article)

    @patch("llm_validator.openai")
    def test_classify_batch_processes_multiple(self, mock_openai):
        """classify_batch should process multiple articles."""
        from llm_validator import classify_batch

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"is_climate_policy": true, "uncertainty_type": "none", "certainty_level": 3, "reasoning": "test"}'
                )
            )
        ]
        mock_openai.chat.completions.create.return_value = mock_response

        articles = [
            {"id": "1", "title": "Article 1", "content": "..."},
            {"id": "2", "title": "Article 2", "content": "..."},
            {"id": "3", "title": "Article 3", "content": "..."},
        ]

        results = classify_batch(articles)

        assert len(results) == 3
        assert mock_openai.chat.completions.create.call_count == 3


class TestAccuracyCalculation:
    """Tests for accuracy metrics calculation."""

    def test_calculate_accuracy_perfect_match(self):
        """calculate_accuracy should return 1.0 for perfect keyword match."""
        from llm_validator import calculate_accuracy

        keyword_classifications = [
            {"id": "1", "is_climate_policy": True, "uncertainty_type": "implementation"},
            {"id": "2", "is_climate_policy": True, "uncertainty_type": "reversal"},
        ]

        llm_classifications = [
            {"id": "1", "is_climate_policy": True, "uncertainty_type": "implementation"},
            {"id": "2", "is_climate_policy": True, "uncertainty_type": "reversal"},
        ]

        accuracy = calculate_accuracy(keyword_classifications, llm_classifications)

        assert accuracy["overall"] == 1.0
        assert accuracy["climate_policy_accuracy"] == 1.0
        assert accuracy["uncertainty_type_accuracy"] == 1.0

    def test_calculate_accuracy_partial_match(self):
        """calculate_accuracy should handle partial matches."""
        from llm_validator import calculate_accuracy

        keyword_classifications = [
            {"id": "1", "is_climate_policy": True, "uncertainty_type": "implementation"},
            {"id": "2", "is_climate_policy": True, "uncertainty_type": "reversal"},
        ]

        llm_classifications = [
            {"id": "1", "is_climate_policy": True, "uncertainty_type": "implementation"},
            {"id": "2", "is_climate_policy": False, "uncertainty_type": "none"},  # Mismatch
        ]

        accuracy = calculate_accuracy(keyword_classifications, llm_classifications)

        assert accuracy["climate_policy_accuracy"] == 0.5
        assert accuracy["overall"] < 1.0

    def test_calculate_accuracy_by_category(self):
        """calculate_accuracy should break down by uncertainty type."""
        from llm_validator import calculate_accuracy

        keyword_classifications = [
            {"id": "1", "is_climate_policy": True, "uncertainty_type": "implementation"},
            {"id": "2", "is_climate_policy": True, "uncertainty_type": "implementation"},
            {"id": "3", "is_climate_policy": True, "uncertainty_type": "reversal"},
        ]

        llm_classifications = [
            {"id": "1", "is_climate_policy": True, "uncertainty_type": "implementation"},
            {"id": "2", "is_climate_policy": True, "uncertainty_type": "none"},  # Wrong
            {"id": "3", "is_climate_policy": True, "uncertainty_type": "reversal"},
        ]

        accuracy = calculate_accuracy(keyword_classifications, llm_classifications)

        assert "implementation_accuracy" in accuracy
        assert "reversal_accuracy" in accuracy


class TestAdaptiveSampling:
    """Tests for adaptive sample expansion."""

    def test_should_expand_sample_below_threshold(self):
        """should_expand_sample should return True if accuracy < 85%."""
        from llm_validator import should_expand_sample

        accuracy_metrics = {"overall": 0.80}

        result = should_expand_sample(accuracy_metrics, threshold=0.85)

        assert result is True

    def test_should_expand_sample_above_threshold(self):
        """should_expand_sample should return False if accuracy >= 85%."""
        from llm_validator import should_expand_sample

        accuracy_metrics = {"overall": 0.90}

        result = should_expand_sample(accuracy_metrics, threshold=0.85)

        assert result is False

    def test_should_expand_sample_at_threshold(self):
        """should_expand_sample should return False if accuracy == threshold."""
        from llm_validator import should_expand_sample

        accuracy_metrics = {"overall": 0.85}

        result = should_expand_sample(accuracy_metrics, threshold=0.85)

        assert result is False

    def test_expand_sample_adds_more_articles(self):
        """expand_sample should add more articles to existing sample."""
        from llm_validator import expand_sample

        all_articles = [{"id": str(i)} for i in range(1000)]
        current_sample = [{"id": str(i)} for i in range(100)]

        expanded = expand_sample(
            all_articles=all_articles,
            current_sample=current_sample,
            additional_count=100,
            seed=42,
        )

        assert len(expanded) == 200
        # Original sample should be preserved
        original_ids = {a["id"] for a in current_sample}
        expanded_ids = {a["id"] for a in expanded}
        assert original_ids.issubset(expanded_ids)


class TestValidationPipeline:
    """Tests for the full validation pipeline."""

    @patch("llm_validator.classify_batch")
    def test_run_validation_pipeline(self, mock_classify):
        """run_validation should execute full pipeline."""
        from llm_validator import run_validation

        mock_classify.return_value = [
            {"id": "1", "is_climate_policy": True, "uncertainty_type": "implementation", "certainty_level": 4},
            {"id": "2", "is_climate_policy": True, "uncertainty_type": "reversal", "certainty_level": 5},
        ]

        articles = [
            {"id": "1", "title": "A", "content": "...", "keyword_category": "implementation"},
            {"id": "2", "title": "B", "content": "...", "keyword_category": "reversal"},
        ]

        result = run_validation(
            articles=articles,
            initial_sample_size=2,
            accuracy_threshold=0.85,
        )

        assert "accuracy" in result
        assert "sample_size" in result
        assert "classifications" in result

    @patch("llm_validator.classify_batch")
    def test_run_validation_expands_if_needed(self, mock_classify):
        """run_validation should expand sample if accuracy below threshold."""
        from llm_validator import run_validation

        # First batch returns low accuracy results
        call_count = [0]

        def classify_side_effect(articles, progress_callback=None):
            call_count[0] += 1
            # Return classifications that will give low accuracy
            return [
                {
                    "id": a["id"],
                    "is_climate_policy": call_count[0] > 1,  # Wrong first time
                    "uncertainty_type": "none",
                    "certainty_level": 3,
                }
                for a in articles
            ]

        mock_classify.side_effect = classify_side_effect

        articles = [
            {"id": str(i), "title": f"Article {i}", "content": "...", "keyword_category": "implementation"}
            for i in range(100)
        ]

        result = run_validation(
            articles=articles,
            initial_sample_size=10,
            accuracy_threshold=0.85,
            max_expansions=2,
            expansion_size=10,
        )

        # Should have expanded at least once
        assert result["sample_size"] > 10 or result["expansions"] >= 1


class TestValidationReport:
    """Tests for validation report generation."""

    def test_generate_validation_report(self):
        """generate_report should create structured validation report."""
        from llm_validator import generate_report

        validation_result = {
            "accuracy": {
                "overall": 0.92,
                "climate_policy_accuracy": 0.95,
                "uncertainty_type_accuracy": 0.88,
            },
            "sample_size": 1000,
            "expansions": 0,
            "classifications": [
                {"id": "1", "is_climate_policy": True, "uncertainty_type": "implementation"},
            ],
        }

        report = generate_report(validation_result)

        assert "summary" in report
        assert "accuracy_metrics" in report
        assert "recommendations" in report
        assert report["summary"]["passed"] is True  # 92% > 85%

    def test_generate_report_recommends_expansion(self):
        """generate_report should recommend expansion if accuracy low."""
        from llm_validator import generate_report

        validation_result = {
            "accuracy": {"overall": 0.75},
            "sample_size": 1000,
            "expansions": 0,
            "classifications": [],
        }

        report = generate_report(validation_result)

        assert report["summary"]["passed"] is False
        assert "expand" in report["recommendations"][0].lower() or "review" in report["recommendations"][0].lower()


class TestPromptGeneration:
    """Tests for classification prompt generation."""

    def test_build_classification_prompt(self):
        """build_classification_prompt should include article content."""
        from llm_validator import build_classification_prompt

        article = {
            "title": "Biden Administration Announces Climate Policy Changes",
            "content": "The administration today announced...",
            "source": "New York Times",
            "date": "2024-01-15",
        }

        prompt = build_classification_prompt(article)

        assert "Biden Administration" in prompt
        assert "climate" in prompt.lower()
        assert "policy" in prompt.lower()
        # Should include classification instructions
        assert "is_climate_policy" in prompt
        assert "uncertainty_type" in prompt

    def test_prompt_includes_uncertainty_definitions(self):
        """build_classification_prompt should define uncertainty types."""
        from llm_validator import build_classification_prompt

        article = {"title": "Test", "content": "Test content"}

        prompt = build_classification_prompt(article)

        assert "implementation" in prompt.lower()
        assert "reversal" in prompt.lower()
