"""
LLM-based Article Validation Pipeline for CPU Index

Validates keyword classification accuracy using GPT-5-nano with adaptive sampling.
This addresses the core methodological concern: keyword matching alone may miss
nuances in whether an article truly discusses policy uncertainty.

Adaptive Sampling Strategy:
1. Start with 1000-article sample
2. Run LLM classification
3. Compare to keyword classification
4. If accuracy < 85%, expand sample and repeat
5. Generate validation report

Classification Schema:
- is_climate_policy: bool - Does article genuinely discuss climate policy?
- uncertainty_type: "implementation" | "reversal" | "none"
  - implementation: Uncertainty about how policy will be implemented
  - reversal: Uncertainty about whether policy will be rolled back
  - none: No genuine policy uncertainty (false positive from keywords)
- certainty_level: 1-5 - LLM's confidence in its classification
- reasoning: str - Brief explanation of classification decision
"""

import json
import random
from collections import defaultdict
from typing import Optional

import openai

import config


class ClassificationError(Exception):
    """Raised when LLM classification fails."""

    pass


# Valid uncertainty types for classification
VALID_UNCERTAINTY_TYPES = {"implementation", "reversal", "none"}


def select_validation_sample(
    articles: list[dict],
    initial_sample_size: int = 1000,
    stratify_by: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Select a sample of articles for LLM validation.

    Args:
        articles: Full list of articles to sample from
        initial_sample_size: Target sample size
        stratify_by: Optional field to stratify sample by (e.g., "keyword_category")
        seed: Random seed for reproducibility

    Returns:
        List of sampled articles
    """
    if seed is not None:
        random.seed(seed)

    if len(articles) <= initial_sample_size:
        return list(articles)

    if stratify_by and any(stratify_by in a for a in articles):
        # Stratified sampling
        groups = defaultdict(list)
        for article in articles:
            key = article.get(stratify_by, "unknown")
            groups[key].append(article)

        # Sample proportionally from each group
        sample = []
        for group_name, group_articles in groups.items():
            proportion = len(group_articles) / len(articles)
            group_sample_size = max(1, int(initial_sample_size * proportion))
            group_sample_size = min(group_sample_size, len(group_articles))
            sample.extend(random.sample(group_articles, group_sample_size))

        # If we need more to hit target, sample from remainder
        if len(sample) < initial_sample_size:
            remaining = [a for a in articles if a not in sample]
            additional = min(initial_sample_size - len(sample), len(remaining))
            sample.extend(random.sample(remaining, additional))

        return sample[:initial_sample_size]
    else:
        # Simple random sampling
        return random.sample(articles, initial_sample_size)


def parse_classification(response: dict) -> dict:
    """
    Parse and validate LLM classification response.

    Args:
        response: Raw classification response dict

    Returns:
        Validated classification dict

    Raises:
        ValueError: If response contains invalid values
    """
    result = {}

    # Extract and validate is_climate_policy
    result["is_climate_policy"] = bool(response.get("is_climate_policy", False))

    # Extract and validate uncertainty_type
    uncertainty_type = response.get("uncertainty_type", "none")
    if uncertainty_type not in VALID_UNCERTAINTY_TYPES:
        raise ValueError(
            f"Invalid uncertainty_type: {uncertainty_type}. "
            f"Must be one of: {VALID_UNCERTAINTY_TYPES}"
        )
    result["uncertainty_type"] = uncertainty_type

    # Extract and validate certainty_level
    certainty_level = response.get("certainty_level", 3)
    if not isinstance(certainty_level, int) or not 1 <= certainty_level <= 5:
        raise ValueError(f"certainty_level must be 1-5, got: {certainty_level}")
    result["certainty_level"] = certainty_level

    # Optional reasoning
    if "reasoning" in response:
        result["reasoning"] = response["reasoning"]

    return result


def build_classification_prompt(article: dict) -> str:
    """
    Build the classification prompt for an article.

    Args:
        article: Article dict with title, content, etc.

    Returns:
        Formatted prompt string
    """
    title = article.get("title", "No title")
    content = article.get("content", article.get("snippet", "No content"))
    source = article.get("source", "Unknown")
    date = article.get("date", "Unknown")

    # Truncate content if too long
    max_content_length = 3000
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."

    prompt = f"""Analyze this news article and classify it according to the schema below.

ARTICLE:
Title: {title}
Source: {source}
Date: {date}
Content: {content}

CLASSIFICATION TASK:
Determine if this article discusses climate policy uncertainty.

DEFINITIONS:
- is_climate_policy: Does this article genuinely discuss government climate policy?
  (Not just climate science, weather, or business decisions unrelated to policy)

- uncertainty_type: What type of policy uncertainty does the article discuss?
  - "implementation": Uncertainty about HOW a policy will be implemented
    (timing, scope, details, enforcement, regulatory specifics)
  - "reversal": Uncertainty about WHETHER a policy will be rolled back, cancelled, or reversed
    (political opposition, court challenges, administration changes, repeal efforts)
  - "none": Article does not discuss genuine policy uncertainty
    (false positive from keyword matching, or just reports facts without uncertainty)

- certainty_level: How confident are you in this classification? (1-5)
  1 = Very uncertain, article is ambiguous
  5 = Very certain, classification is clear

RESPOND WITH JSON:
{{
    "is_climate_policy": true/false,
    "uncertainty_type": "implementation" | "reversal" | "none",
    "certainty_level": 1-5,
    "reasoning": "Brief explanation of your classification"
}}"""

    return prompt


def classify_article(article: dict) -> dict:
    """
    Classify a single article using LLM.

    Args:
        article: Article dict with title and content

    Returns:
        Classification result dict

    Raises:
        ClassificationError: If classification fails
    """
    prompt = build_classification_prompt(article)

    try:
        response = openai.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing news articles about climate policy. "
                    "Respond only with valid JSON matching the requested schema.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # Low temperature for consistency
        )

        raw_response = response.choices[0].message.content
        parsed = json.loads(raw_response)
        result = parse_classification(parsed)
        result["id"] = article.get("id")
        return result

    except json.JSONDecodeError as e:
        raise ClassificationError(f"Failed to parse LLM response as JSON: {e}")
    except ValueError as e:
        raise ClassificationError(f"Invalid classification values: {e}")
    except Exception as e:
        raise ClassificationError(f"LLM classification failed: {e}")


def classify_batch(
    articles: list[dict],
    progress_callback: Optional[callable] = None,
) -> list[dict]:
    """
    Classify a batch of articles.

    Args:
        articles: List of articles to classify
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of classification results
    """
    results = []
    total = len(articles)

    for i, article in enumerate(articles):
        try:
            result = classify_article(article)
            results.append(result)
        except ClassificationError as e:
            # Log error but continue with other articles
            results.append(
                {
                    "id": article.get("id"),
                    "error": str(e),
                    "is_climate_policy": None,
                    "uncertainty_type": None,
                    "certainty_level": None,
                }
            )

        if progress_callback:
            progress_callback(i + 1, total)

    return results


def calculate_accuracy(
    keyword_classifications: list[dict],
    llm_classifications: list[dict],
) -> dict:
    """
    Calculate accuracy metrics comparing keyword to LLM classifications.

    Args:
        keyword_classifications: Classifications from keyword matching
        llm_classifications: Classifications from LLM

    Returns:
        Dict with accuracy metrics
    """
    # Build lookup by ID
    llm_by_id = {c["id"]: c for c in llm_classifications if c.get("id")}

    # Track matches
    climate_policy_matches = 0
    climate_policy_total = 0
    uncertainty_type_matches = 0
    uncertainty_type_total = 0

    # Track by uncertainty type
    type_matches = defaultdict(int)
    type_totals = defaultdict(int)

    for kw_class in keyword_classifications:
        article_id = kw_class.get("id")
        if article_id not in llm_by_id:
            continue

        llm_class = llm_by_id[article_id]

        # Skip if LLM had an error
        if llm_class.get("is_climate_policy") is None:
            continue

        # Climate policy accuracy
        climate_policy_total += 1
        if kw_class.get("is_climate_policy") == llm_class.get("is_climate_policy"):
            climate_policy_matches += 1

        # Uncertainty type accuracy
        uncertainty_type_total += 1
        kw_type = kw_class.get("uncertainty_type")
        llm_type = llm_class.get("uncertainty_type")

        if kw_type == llm_type:
            uncertainty_type_matches += 1

        # Track by type
        if kw_type:
            type_totals[kw_type] += 1
            if kw_type == llm_type:
                type_matches[kw_type] += 1

    # Calculate metrics
    climate_policy_accuracy = (
        climate_policy_matches / climate_policy_total if climate_policy_total > 0 else 0
    )
    uncertainty_type_accuracy = (
        uncertainty_type_matches / uncertainty_type_total
        if uncertainty_type_total > 0
        else 0
    )

    # Overall accuracy (average of both)
    overall = (climate_policy_accuracy + uncertainty_type_accuracy) / 2

    result = {
        "overall": overall,
        "climate_policy_accuracy": climate_policy_accuracy,
        "uncertainty_type_accuracy": uncertainty_type_accuracy,
        "total_compared": climate_policy_total,
    }

    # Add per-type accuracy
    for utype in ["implementation", "reversal", "none"]:
        if type_totals[utype] > 0:
            result[f"{utype}_accuracy"] = type_matches[utype] / type_totals[utype]
        else:
            result[f"{utype}_accuracy"] = None

    return result


def should_expand_sample(
    accuracy_metrics: dict,
    threshold: float = 0.85,
) -> bool:
    """
    Determine if sample should be expanded based on accuracy.

    Args:
        accuracy_metrics: Accuracy metrics dict
        threshold: Minimum acceptable accuracy (default 85%)

    Returns:
        True if sample should be expanded
    """
    overall = accuracy_metrics.get("overall", 0)
    return overall < threshold


def expand_sample(
    all_articles: list[dict],
    current_sample: list[dict],
    additional_count: int = 500,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Expand sample with additional articles.

    Args:
        all_articles: Full article list
        current_sample: Current sample
        additional_count: Number of articles to add
        seed: Random seed for reproducibility

    Returns:
        Expanded sample
    """
    if seed is not None:
        random.seed(seed)

    # Find articles not in current sample
    current_ids = {a.get("id") for a in current_sample}
    available = [a for a in all_articles if a.get("id") not in current_ids]

    # Sample additional articles
    additional = min(additional_count, len(available))
    if additional > 0:
        new_articles = random.sample(available, additional)
        return list(current_sample) + new_articles
    else:
        return list(current_sample)


def run_validation(
    articles: list[dict],
    initial_sample_size: int = 1000,
    accuracy_threshold: float = 0.85,
    max_expansions: int = 3,
    expansion_size: int = 500,
    progress_callback: Optional[callable] = None,
) -> dict:
    """
    Run full validation pipeline with adaptive sampling.

    Args:
        articles: Full list of articles with keyword classifications
        initial_sample_size: Starting sample size
        accuracy_threshold: Target accuracy (default 85%)
        max_expansions: Maximum number of sample expansions
        expansion_size: Number of articles to add per expansion
        progress_callback: Optional progress callback

    Returns:
        Validation result dict with accuracy, sample_size, classifications
    """
    # Convert keyword classifications to standard format
    keyword_classifications = []
    for article in articles:
        kw_class = {
            "id": article.get("id"),
            "is_climate_policy": True,  # Keyword match implies climate policy
            "uncertainty_type": article.get("keyword_category", "none"),
        }
        keyword_classifications.append(kw_class)

    # Select initial sample
    sample = select_validation_sample(
        articles,
        initial_sample_size=initial_sample_size,
        stratify_by="keyword_category",
    )

    all_classifications = []
    expansions = 0

    while True:
        # Classify sample
        sample_classifications = classify_batch(sample, progress_callback)
        all_classifications = sample_classifications

        # Get keyword classifications for sample
        sample_ids = {a.get("id") for a in sample}
        sample_keyword_class = [
            kc for kc in keyword_classifications if kc.get("id") in sample_ids
        ]

        # Calculate accuracy
        accuracy = calculate_accuracy(sample_keyword_class, sample_classifications)

        # Check if we should expand
        if not should_expand_sample(accuracy, accuracy_threshold):
            break

        if expansions >= max_expansions:
            break

        # Expand sample
        sample = expand_sample(
            all_articles=articles,
            current_sample=sample,
            additional_count=expansion_size,
        )
        expansions += 1

    return {
        "accuracy": accuracy,
        "sample_size": len(sample),
        "expansions": expansions,
        "classifications": all_classifications,
    }


def generate_report(validation_result: dict) -> dict:
    """
    Generate a validation report from results.

    Args:
        validation_result: Result from run_validation

    Returns:
        Structured report dict
    """
    accuracy = validation_result.get("accuracy", {})
    sample_size = validation_result.get("sample_size", 0)
    expansions = validation_result.get("expansions", 0)

    overall = accuracy.get("overall", 0)
    passed = overall >= 0.85

    report = {
        "summary": {
            "passed": passed,
            "overall_accuracy": overall,
            "sample_size": sample_size,
            "expansions": expansions,
        },
        "accuracy_metrics": accuracy,
        "recommendations": [],
    }

    # Add recommendations
    if not passed:
        report["recommendations"].append(
            f"Accuracy ({overall:.1%}) is below 85% threshold. "
            "Review keyword definitions or expand sample for more data."
        )

        # Specific recommendations based on which metrics are low
        if accuracy.get("climate_policy_accuracy", 1) < 0.85:
            report["recommendations"].append(
                "Climate policy detection is low. Consider refining "
                "CLIMATE_TERMS or POLICY_TERMS in config."
            )

        if accuracy.get("uncertainty_type_accuracy", 1) < 0.85:
            report["recommendations"].append(
                "Uncertainty type classification is low. Review "
                "IMPLEMENTATION_KEYWORDS and REVERSAL_KEYWORDS definitions."
            )
    else:
        report["recommendations"].append(
            f"Validation passed with {overall:.1%} accuracy. "
            "Keyword classification is reliable for this dataset."
        )

    return report
