"""
Local Keyword Classifier for CPU Index Builder

Classifies articles using keyword matching on title + snippet.
This replaces the need for multiple API queries with local text analysis.

Key principle (addressing Steve's critique):
- Direction terms alone (rollback, expand) do NOT indicate uncertainty
- Uncertainty REQUIRES an uncertainty term (uncertain, risk, etc.)
- Classification: uncertainty AND (reversal OR implementation OR upside)

Ablation Support:
- exclude_keywords: Remove specific terms from matching
- require_uncertainty: Toggle Steve's fix for ablation testing
- custom term lists for placebo indices
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from cpu_index import config


@dataclass
class ClassifierConfig:
    """
    Configuration for ablation studies.

    Allows toggling classifier behavior for robustness testing.
    """
    # Keywords to exclude (for keyword-dropping sensitivity tests)
    exclude_keywords: list[str] = field(default_factory=list)

    # Whether to require uncertainty terms (Steve's fix)
    # Set to False for ablation test: "what if we count direction alone?"
    require_uncertainty: bool = True

    # Custom term overrides (for placebo indices)
    custom_uncertainty_terms: Optional[list[str]] = None
    custom_reversal_terms: Optional[list[str]] = None
    custom_implementation_terms: Optional[list[str]] = None
    custom_upside_terms: Optional[list[str]] = None
    custom_ira_terms: Optional[list[str]] = None
    custom_obbba_terms: Optional[list[str]] = None

    def get_terms(self, term_type: str) -> list[str]:
        """Get terms for a category, applying exclusions."""
        # Get base terms (custom or from config)
        if term_type == "uncertainty":
            base = self.custom_uncertainty_terms or config.UNCERTAINTY_TERMS
        elif term_type == "reversal":
            base = self.custom_reversal_terms or config.REVERSAL_TERMS
        elif term_type == "implementation":
            base = self.custom_implementation_terms or config.IMPLEMENTATION_TERMS
        elif term_type == "upside":
            base = self.custom_upside_terms or config.UPSIDE_TERMS
        elif term_type == "regime_ira":
            base = self.custom_ira_terms or config.REGIME_IRA_TERMS
        elif term_type == "regime_obbba":
            base = self.custom_obbba_terms or config.REGIME_OBBBA_TERMS
        else:
            raise ValueError(f"Unknown term type: {term_type}")

        # Apply exclusions (case-insensitive)
        excluded_lower = [e.lower() for e in self.exclude_keywords]
        return [t for t in base if t.lower() not in excluded_lower]


# Default configuration (no ablation)
DEFAULT_CONFIG = ClassifierConfig()


def _normalize_text(text: str) -> str:
    """
    Normalize text for keyword matching.
    Converts to lowercase and normalizes whitespace.
    """
    if not text:
        return ""
    return " ".join(text.lower().split())


def _find_matches(text: str, terms: list[str]) -> list[str]:
    """
    Find all matching terms in text.

    Args:
        text: Text to search (will be normalized internally)
        terms: List of terms to search for

    Returns:
        List of matched terms
    """
    # Normalize input text for case-insensitive matching
    normalized_text = _normalize_text(text)
    matches = []
    for term in terms:
        # Normalize the term
        normalized_term = _normalize_text(term)
        # Use word boundary matching for single words
        # and substring matching for phrases
        if " " in normalized_term:
            # Multi-word phrase: direct substring match
            if normalized_term in normalized_text:
                matches.append(term)
        else:
            # Single word: word boundary match to avoid partial matches
            # e.g., "risk" should not match "brisk"
            pattern = r'\b' + re.escape(normalized_term) + r'\b'
            if re.search(pattern, normalized_text):
                matches.append(term)
    return matches


def classify_article(
    title: str,
    snippet: str,
    include_matched_terms: bool = True,
    classifier_config: Optional[ClassifierConfig] = None,
) -> dict:
    """
    Classify a single article using local keyword matching.

    The classification follows BBD methodology with Steve's critique addressed:
    - Uncertainty is REQUIRED for CPU indices (not just direction)
    - Direction terms (reversal/implementation/upside) are secondary qualifiers

    Args:
        title: Article title
        snippet: Article snippet/overview text
        include_matched_terms: Whether to include list of matched terms
        classifier_config: Optional ablation configuration (default: no ablation)

    Returns:
        Classification dict with boolean flags and optional matched terms:
        {
            "has_uncertainty": bool,
            "has_reversal_terms": bool,
            "has_implementation_terms": bool,
            "has_upside_terms": bool,
            "has_ira_mention": bool,
            "has_obbba_mention": bool,
            "matched_terms": {
                "uncertainty": [...],
                "reversal": [...],
                "implementation": [...],
                "upside": [...],
                "regime_ira": [...],
                "regime_obbba": [...],
            }
        }
    """
    cfg = classifier_config or DEFAULT_CONFIG

    # Combine title and snippet for searching
    combined_text = _normalize_text(f"{title or ''} {snippet or ''}")

    # Find matches for each category (with ablation-aware term lists)
    uncertainty_matches = _find_matches(combined_text, cfg.get_terms("uncertainty"))
    reversal_matches = _find_matches(combined_text, cfg.get_terms("reversal"))
    implementation_matches = _find_matches(combined_text, cfg.get_terms("implementation"))
    upside_matches = _find_matches(combined_text, cfg.get_terms("upside"))
    ira_matches = _find_matches(combined_text, cfg.get_terms("regime_ira"))
    obbba_matches = _find_matches(combined_text, cfg.get_terms("regime_obbba"))

    result = {
        "has_uncertainty": len(uncertainty_matches) > 0,
        "has_reversal_terms": len(reversal_matches) > 0,
        "has_implementation_terms": len(implementation_matches) > 0,
        "has_upside_terms": len(upside_matches) > 0,
        "has_ira_mention": len(ira_matches) > 0,
        "has_obbba_mention": len(obbba_matches) > 0,
    }

    if include_matched_terms:
        result["matched_terms"] = {
            "uncertainty": uncertainty_matches,
            "reversal": reversal_matches,
            "implementation": implementation_matches,
            "upside": upside_matches,
            "regime_ira": ira_matches,
            "regime_obbba": obbba_matches,
        }

    return result


def classify_articles_batch(
    articles: list[dict],
    include_matched_terms: bool = False,
    classifier_config: Optional[ClassifierConfig] = None,
) -> list[dict]:
    """
    Classify multiple articles efficiently.

    Args:
        articles: List of article dicts with 'id', 'title', 'snippet' fields
        include_matched_terms: Whether to include matched terms (adds overhead)
        classifier_config: Optional ablation configuration

    Returns:
        List of classification dicts with article_id added
    """
    results = []
    for article in articles:
        classification = classify_article(
            title=article.get("title", ""),
            snippet=article.get("snippet", ""),
            include_matched_terms=include_matched_terms,
            classifier_config=classifier_config,
        )
        classification["article_id"] = article["id"]
        results.append(classification)
    return results


def compute_cpu_classification(
    classification: dict,
    classifier_config: Optional[ClassifierConfig] = None,
) -> dict:
    """
    Compute CPU index classifications from article classification.

    This is where Steve's critique is addressed:
    - CPU_t: Article has uncertainty term
    - CPU_reversal_t: Article has uncertainty AND reversal terms
    - CPU_impl_t: Article has uncertainty AND implementation terms

    Without uncertainty term, direction terms do NOT count!
    (Unless require_uncertainty=False for ablation testing)

    Args:
        classification: Result from classify_article()
        classifier_config: Optional ablation configuration

    Returns:
        CPU classification flags:
        {
            "is_cpu": bool,           # Counts toward standard CPU
            "is_cpu_reversal": bool,  # Counts toward CPU_reversal
            "is_cpu_impl": bool,      # Counts toward CPU_impl
            "is_cpu_upside": bool,    # Counts toward CPU_upside
            "is_regime_ira": bool,    # Counts toward IRA salience
            "is_regime_obbba": bool,  # Counts toward OBBBA salience
        }
    """
    cfg = classifier_config or DEFAULT_CONFIG
    has_uncertainty = classification["has_uncertainty"]

    # For ablation: if require_uncertainty=False, direction terms alone count
    uncertainty_gate = has_uncertainty if cfg.require_uncertainty else True

    return {
        # Standard CPU: must have uncertainty term (unless ablation override)
        "is_cpu": has_uncertainty if cfg.require_uncertainty else (
            has_uncertainty or
            classification["has_reversal_terms"] or
            classification["has_implementation_terms"]
        ),

        # CPU_reversal: must have BOTH uncertainty AND reversal (unless ablation)
        "is_cpu_reversal": uncertainty_gate and classification["has_reversal_terms"],

        # CPU_impl: must have BOTH uncertainty AND implementation (unless ablation)
        "is_cpu_impl": uncertainty_gate and classification["has_implementation_terms"],

        # CPU_upside: must have BOTH uncertainty AND upside (unless ablation)
        "is_cpu_upside": uncertainty_gate and classification["has_upside_terms"],

        # Regime salience: does NOT require uncertainty
        # (Just tracking mentions of specific policies)
        "is_regime_ira": classification["has_ira_mention"],
        "is_regime_obbba": classification["has_obbba_mention"],
    }


def aggregate_classifications(
    classifications: list[dict],
    month: Optional[str] = None,
    classifier_config: Optional[ClassifierConfig] = None,
) -> dict:
    """
    Aggregate classifications to compute monthly counts.

    Args:
        classifications: List of classification dicts
        month: Optional month label for the result
        classifier_config: Optional ablation configuration

    Returns:
        Aggregated counts:
        {
            "month": str,
            "total_articles": int,
            "cpu_count": int,
            "cpu_reversal_count": int,
            "cpu_impl_count": int,
            "cpu_upside_count": int,
            "regime_ira_count": int,
            "regime_obbba_count": int,
        }
    """
    cpu_count = 0
    cpu_reversal_count = 0
    cpu_impl_count = 0
    cpu_upside_count = 0
    regime_ira_count = 0
    regime_obbba_count = 0

    for classification in classifications:
        cpu_flags = compute_cpu_classification(classification, classifier_config)

        if cpu_flags["is_cpu"]:
            cpu_count += 1
        if cpu_flags["is_cpu_reversal"]:
            cpu_reversal_count += 1
        if cpu_flags["is_cpu_impl"]:
            cpu_impl_count += 1
        if cpu_flags["is_cpu_upside"]:
            cpu_upside_count += 1
        if cpu_flags["is_regime_ira"]:
            regime_ira_count += 1
        if cpu_flags["is_regime_obbba"]:
            regime_obbba_count += 1

    return {
        "month": month,
        "total_articles": len(classifications),
        "cpu_count": cpu_count,
        "cpu_reversal_count": cpu_reversal_count,
        "cpu_impl_count": cpu_impl_count,
        "cpu_upside_count": cpu_upside_count,
        "regime_ira_count": regime_ira_count,
        "regime_obbba_count": regime_obbba_count,
    }


def classify_for_placebo(
    title: str,
    snippet: str,
    placebo_type: str = "trade",
) -> dict:
    """
    Classify article for placebo index (trade or monetary policy).

    Args:
        title: Article title
        snippet: Article snippet
        placebo_type: "trade" or "monetary"

    Returns:
        Classification for placebo index
    """
    combined_text = _normalize_text(f"{title or ''} {snippet or ''}")

    # Get placebo-specific terms
    if placebo_type == "trade":
        domain_terms = config.TRADE_TERMS
    elif placebo_type == "monetary":
        domain_terms = config.MONETARY_TERMS
    else:
        raise ValueError(f"Unknown placebo type: {placebo_type}")

    domain_matches = _find_matches(combined_text, domain_terms)
    uncertainty_matches = _find_matches(combined_text, config.UNCERTAINTY_TERMS)
    policy_matches = _find_matches(combined_text, config.POLICY_TERMS)

    return {
        "has_domain_terms": len(domain_matches) > 0,
        "has_policy_terms": len(policy_matches) > 0,
        "has_uncertainty": len(uncertainty_matches) > 0,
        "is_placebo_cpu": (
            len(domain_matches) > 0 and
            len(policy_matches) > 0 and
            len(uncertainty_matches) > 0
        ),
    }


def get_validation_status() -> dict:
    """Check if LLM validation is available."""
    ai_sdk_available = False
    try:
        import openai
        ai_sdk_available = True
    except ImportError:
        pass

    openai_key_set = bool(os.environ.get("OPENAI_API_KEY"))

    return {
        "ai_sdk_available": ai_sdk_available,
        "openai_key_set": openai_key_set,
    }


def estimate_classification_cost(sample_size: int) -> dict:
    """Estimate cost of LLM classification."""
    avg_tokens_per_article = 150
    total_input_tokens = sample_size * avg_tokens_per_article
    total_output_tokens = sample_size * 50

    cost_per_1m_input = 0.15
    cost_per_1m_output = 0.60

    estimated_cost = (
        (total_input_tokens / 1_000_000) * cost_per_1m_input +
        (total_output_tokens / 1_000_000) * cost_per_1m_output
    )

    return {
        "sample_size": sample_size,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_cost_usd": estimated_cost,
    }
