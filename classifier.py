"""
LLM Classification for CPU Index Builder.

Uses GPT-5 Nano via Vercel AI SDK to validate keyword accuracy.
This is a SAMPLING tool - we don't classify every article.
Cost: ~$0.01 per 100 articles.
"""

import os
from typing import Optional

from pydantic import BaseModel

import config

try:
    from ai_sdk import generate_object, openai
    AI_SDK_AVAILABLE = True
except ImportError:
    AI_SDK_AVAILABLE = False
    print("Warning: ai-sdk-python not installed. LLM classification disabled.")

INPUT_PRICE_PER_MILLION = 0.05
OUTPUT_PRICE_PER_MILLION = 0.40
INPUT_TOKENS_PER_ARTICLE = 500
OUTPUT_TOKENS_PER_ARTICLE = 100


class ArticleClassification(BaseModel):
    """Pydantic model for LLM response."""

    is_climate_policy: bool
    has_uncertainty: bool
    reasoning: str
    confidence: str


def classify_article(article: dict) -> Optional[ArticleClassification]:
    """Classify a single article using GPT-5 Nano."""
    if not AI_SDK_AVAILABLE:
        return None

    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Cannot classify.")
        return None

    model = openai(config.LLM_MODEL)
    result = generate_object(
        model=model,
        schema=ArticleClassification,
        prompt=_build_prompt(article),
        temperature=config.LLM_TEMPERATURE,
    )
    return result.object


def _build_prompt(article: dict) -> str:
    """Build classification prompt for an article."""
    text = article.get("full_text") or article.get("snippet") or ""
    truncated_text = text[:2000]

    return f"""You are classifying news articles for a Climate Policy Uncertainty Index.

ARTICLE:
- Title: {article.get('title', 'Unknown')}
- Date: {article.get('date', 'Unknown')}
- Source: {article.get('source', 'Unknown')}
- Text: {truncated_text}

TASK: Answer these questions:

1. Is this article primarily about U.S. climate or clean energy POLICY?
   - YES if about: legislation, regulation, tax credits, subsidies, government programs
   - NO if about: weather, climate science only, international policy, corporate actions without policy context

2. If YES to #1: Does it express UNCERTAINTY about policy?
   - YES if mentions: delays, legal challenges, rollbacks, unclear guidance, political disputes, pending decisions
   - NO if policy seems stable/certain

Be concise in your reasoning (1-2 sentences max).
Set confidence to "high" if clear-cut, "medium" if borderline, "low" if unsure."""


def classify_sample(
    month: Optional[str] = None,
    sample_size: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """
    Classify a random sample of articles.

    Args:
        month: Specific month to sample from (None = all months)
        sample_size: Number of articles (default from config)
        dry_run: If True, return fake results

    Returns:
        Summary dict with classification results
    """
    effective_size = sample_size if sample_size is not None else config.LLM_SAMPLE_SIZE

    if dry_run:
        return {
            "sample_size": effective_size,
            "classified": effective_size,
            "is_climate_policy_yes": int(effective_size * 0.85),
            "has_uncertainty_yes": int(effective_size * 0.20),
            "dry_run": True,
        }

    return {
        "sample_size": effective_size,
        "status": "not_implemented",
        "message": "Full implementation requires articles in database. Run collection first.",
    }


def estimate_classification_cost(num_articles: int) -> dict:
    """Estimate cost before running classification."""
    total_input = num_articles * INPUT_TOKENS_PER_ARTICLE
    total_output = num_articles * OUTPUT_TOKENS_PER_ARTICLE

    input_cost = (total_input / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (total_output / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost

    return {
        "articles": num_articles,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_cost_usd": round(total_cost, 4),
        "cost_per_100_articles": round(total_cost / num_articles * 100, 4),
    }


def get_validation_status() -> dict:
    """Get current validation status."""
    return {
        "total_classified": 0,
        "ai_sdk_available": AI_SDK_AVAILABLE,
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
    }
