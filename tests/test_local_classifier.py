"""
Tests for local_classifier.py - Local keyword classification

Verifies that the local classifier correctly identifies:
- Uncertainty terms
- Reversal/implementation/upside direction terms
- Regime salience (IRA/OBBBA mentions)
- Correct CPU classification logic (uncertainty REQUIRED)
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import local_classifier


class TestNormalizeText:
    """Tests for text normalization."""

    def test_normalize_lowercase(self):
        """Should convert to lowercase."""
        result = local_classifier._normalize_text("CLIMATE POLICY")
        assert result == "climate policy"

    def test_normalize_whitespace(self):
        """Should normalize multiple spaces."""
        result = local_classifier._normalize_text("climate   policy\n\tuncertainty")
        assert result == "climate policy uncertainty"

    def test_normalize_empty(self):
        """Should handle empty string."""
        result = local_classifier._normalize_text("")
        assert result == ""

    def test_normalize_none(self):
        """Should handle None."""
        result = local_classifier._normalize_text(None)
        assert result == ""


class TestFindMatches:
    """Tests for term matching."""

    def test_find_single_word_match(self):
        """Should find single word matches."""
        text = "the policy is uncertain"
        matches = local_classifier._find_matches(text, ["uncertain", "risk"])
        assert "uncertain" in matches
        assert "risk" not in matches

    def test_find_phrase_match(self):
        """Should find multi-word phrase matches."""
        text = "clean energy policy changes"
        matches = local_classifier._find_matches(text, ["clean energy", "solar"])
        assert "clean energy" in matches
        assert "solar" not in matches

    def test_word_boundary_matching(self):
        """Should respect word boundaries for single words."""
        text = "the brisk market"
        # "risk" should NOT match "brisk"
        matches = local_classifier._find_matches(text, ["risk"])
        assert len(matches) == 0

    def test_case_insensitive(self):
        """Matching should be case insensitive."""
        text = "UNCERTAIN policy ROLLBACK"
        matches = local_classifier._find_matches(text, ["uncertain", "rollback"])
        assert "uncertain" in matches
        assert "rollback" in matches


class TestClassifyArticle:
    """Tests for article classification."""

    def test_classify_uncertainty_only(self):
        """Article with only uncertainty terms."""
        result = local_classifier.classify_article(
            title="Climate policy uncertainty",
            snippet="The future of the policy is unclear and risky.",
        )
        assert result["has_uncertainty"] is True
        assert result["has_reversal_terms"] is False
        assert result["has_implementation_terms"] is False

    def test_classify_reversal_only(self):
        """Article with only reversal terms (no uncertainty)."""
        result = local_classifier.classify_article(
            title="Trump to repeal IRA",
            snippet="The administration plans to rollback climate provisions.",
        )
        assert result["has_uncertainty"] is False  # No uncertainty term!
        assert result["has_reversal_terms"] is True

    def test_classify_uncertainty_and_reversal(self):
        """Article with both uncertainty AND reversal terms."""
        result = local_classifier.classify_article(
            title="Uncertain future for climate policy",
            snippet="There is risk that the IRA will be repealed.",
        )
        assert result["has_uncertainty"] is True
        assert result["has_reversal_terms"] is True

    def test_classify_implementation(self):
        """Article with implementation terms."""
        result = local_classifier.classify_article(
            title="IRA implementation delays",
            snippet="Unclear guidance creates timeline uncertainty for approvals.",
        )
        assert result["has_uncertainty"] is True
        assert result["has_implementation_terms"] is True

    def test_classify_regime_ira(self):
        """Article mentioning IRA specifically."""
        result = local_classifier.classify_article(
            title="Inflation Reduction Act funding",
            snippet="The 45X tax credit provisions under the IRA...",
        )
        assert result["has_ira_mention"] is True

    def test_classify_regime_obbba(self):
        """Article mentioning OBBBA."""
        result = local_classifier.classify_article(
            title="One Big Beautiful Bill",
            snippet="The reconciliation bill may phase out IRA credits.",
        )
        assert result["has_obbba_mention"] is True

    def test_classify_includes_matched_terms(self):
        """Should include matched terms when requested."""
        result = local_classifier.classify_article(
            title="Climate uncertainty",
            snippet="Policy risk and unclear guidance",
            include_matched_terms=True,
        )
        assert "matched_terms" in result
        assert "uncertain" in result["matched_terms"]["uncertainty"] or \
               "unclear" in result["matched_terms"]["uncertainty"]


class TestComputeCPUClassification:
    """Tests for CPU index classification logic - THE CRITICAL PART."""

    def test_cpu_requires_uncertainty(self):
        """CPU should require uncertainty term."""
        # Article with reversal but NO uncertainty
        classification = {
            "has_uncertainty": False,
            "has_reversal_terms": True,
            "has_implementation_terms": False,
            "has_upside_terms": False,
            "has_ira_mention": False,
            "has_obbba_mention": False,
        }
        result = local_classifier.compute_cpu_classification(classification)

        # Should NOT count toward any CPU index!
        assert result["is_cpu"] is False
        assert result["is_cpu_reversal"] is False

    def test_cpu_reversal_requires_both(self):
        """CPU_reversal requires BOTH uncertainty AND reversal."""
        # Uncertainty only
        classification1 = {
            "has_uncertainty": True,
            "has_reversal_terms": False,
            "has_implementation_terms": False,
            "has_upside_terms": False,
            "has_ira_mention": False,
            "has_obbba_mention": False,
        }
        result1 = local_classifier.compute_cpu_classification(classification1)
        assert result1["is_cpu"] is True
        assert result1["is_cpu_reversal"] is False

        # Both uncertainty AND reversal
        classification2 = {
            "has_uncertainty": True,
            "has_reversal_terms": True,
            "has_implementation_terms": False,
            "has_upside_terms": False,
            "has_ira_mention": False,
            "has_obbba_mention": False,
        }
        result2 = local_classifier.compute_cpu_classification(classification2)
        assert result2["is_cpu"] is True
        assert result2["is_cpu_reversal"] is True

    def test_cpu_impl_requires_both(self):
        """CPU_impl requires BOTH uncertainty AND implementation."""
        classification = {
            "has_uncertainty": True,
            "has_reversal_terms": False,
            "has_implementation_terms": True,
            "has_upside_terms": False,
            "has_ira_mention": False,
            "has_obbba_mention": False,
        }
        result = local_classifier.compute_cpu_classification(classification)
        assert result["is_cpu"] is True
        assert result["is_cpu_impl"] is True

    def test_regime_salience_no_uncertainty_required(self):
        """Regime salience does NOT require uncertainty."""
        classification = {
            "has_uncertainty": False,
            "has_reversal_terms": False,
            "has_implementation_terms": False,
            "has_upside_terms": False,
            "has_ira_mention": True,
            "has_obbba_mention": False,
        }
        result = local_classifier.compute_cpu_classification(classification)
        assert result["is_cpu"] is False  # No uncertainty
        assert result["is_regime_ira"] is True  # But IRA mention counts


class TestClassifyArticlesBatch:
    """Tests for batch classification."""

    def test_batch_classification(self):
        """Should classify multiple articles."""
        articles = [
            {"id": "1", "title": "Uncertain policy", "snippet": "Risk of rollback"},
            {"id": "2", "title": "Clear expansion", "snippet": "Investment announced"},
        ]
        results = local_classifier.classify_articles_batch(articles)

        assert len(results) == 2
        assert results[0]["article_id"] == "1"
        assert results[0]["has_uncertainty"] is True
        assert results[1]["article_id"] == "2"

    def test_batch_handles_empty_fields(self):
        """Should handle missing title/snippet."""
        articles = [
            {"id": "1", "title": None, "snippet": "Some uncertainty"},
            {"id": "2", "title": "Some title", "snippet": None},
        ]
        results = local_classifier.classify_articles_batch(articles)
        assert len(results) == 2


class TestAggregateClassifications:
    """Tests for aggregating classifications."""

    def test_aggregate_counts(self):
        """Should correctly count each category."""
        classifications = [
            {"has_uncertainty": True, "has_reversal_terms": True,
             "has_implementation_terms": False, "has_upside_terms": False,
             "has_ira_mention": True, "has_obbba_mention": False},
            {"has_uncertainty": True, "has_reversal_terms": False,
             "has_implementation_terms": True, "has_upside_terms": False,
             "has_ira_mention": False, "has_obbba_mention": False},
            {"has_uncertainty": False, "has_reversal_terms": True,
             "has_implementation_terms": False, "has_upside_terms": False,
             "has_ira_mention": True, "has_obbba_mention": True},
        ]
        result = local_classifier.aggregate_classifications(
            classifications, month="2024-01"
        )

        assert result["month"] == "2024-01"
        assert result["total_articles"] == 3
        assert result["cpu_count"] == 2  # Only 2 have uncertainty
        assert result["cpu_reversal_count"] == 1  # Only 1 has uncertainty + reversal
        assert result["cpu_impl_count"] == 1  # Only 1 has uncertainty + impl
        assert result["regime_ira_count"] == 2  # 2 mention IRA
        assert result["regime_obbba_count"] == 1  # 1 mentions OBBBA


class TestPlaceboClassification:
    """Tests for placebo index classification."""

    def test_trade_placebo(self):
        """Should classify trade policy articles."""
        result = local_classifier.classify_for_placebo(
            title="Trade policy uncertainty",
            snippet="Tariff risk creates unclear trade situation",
            placebo_type="trade",
        )
        assert result["has_domain_terms"] is True
        assert result["has_uncertainty"] is True
        assert result["is_placebo_cpu"] is True

    def test_monetary_placebo(self):
        """Should classify monetary policy articles."""
        result = local_classifier.classify_for_placebo(
            title="Federal Reserve policy",
            snippet="Interest rate uncertainty ahead of FOMC",
            placebo_type="monetary",
        )
        assert result["has_domain_terms"] is True
        assert result["has_uncertainty"] is True

    def test_invalid_placebo_type(self):
        """Should raise error for invalid placebo type."""
        with pytest.raises(ValueError, match="Unknown placebo type"):
            local_classifier.classify_for_placebo(
                title="test", snippet="test", placebo_type="invalid"
            )


class TestStevesCritique:
    """
    Tests specifically addressing Steve's critique:
    "Rollback alone doesn't mean uncertainty - it could mean CERTAIN rollback."
    """

    def test_certain_rollback_not_counted(self):
        """
        Article: "Trump WILL repeal IRA" (certain, not uncertain)
        Should NOT count toward CPU indices.
        """
        result = local_classifier.classify_article(
            title="Trump to repeal IRA",
            snippet="The president announced he will definitely rollback the act.",
        )
        cpu = local_classifier.compute_cpu_classification(result)

        # No uncertainty term, so shouldn't count
        assert cpu["is_cpu"] is False
        assert cpu["is_cpu_reversal"] is False

    def test_uncertain_rollback_is_counted(self):
        """
        Article: "Trump MIGHT repeal IRA" (uncertain)
        Should count toward CPU_reversal.
        """
        result = local_classifier.classify_article(
            title="Uncertainty about IRA repeal",
            snippet="There is risk that the president may rollback the act.",
        )
        cpu = local_classifier.compute_cpu_classification(result)

        # Has uncertainty term, so should count
        assert cpu["is_cpu"] is True
        assert cpu["is_cpu_reversal"] is True

    def test_direction_without_uncertainty_regime_salience_only(self):
        """
        Certain policy change articles still count for regime salience
        (tracking policy attention) but not for uncertainty indices.
        """
        result = local_classifier.classify_article(
            title="IRA implementation proceeds",
            snippet="Treasury finalizes guidance for Inflation Reduction Act credits.",
        )
        cpu = local_classifier.compute_cpu_classification(result)

        # No uncertainty, so no CPU
        assert cpu["is_cpu"] is False
        # But IRA mentioned, so regime salience counts
        assert cpu["is_regime_ira"] is True
