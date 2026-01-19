"""
Ablation Study Configuration for Climate Policy Uncertainty Index

This module defines all ablation studies required for SOTA publication,
following BBD (2016) methodology and reviewer expectations.

ABLATION PHASES:
- Phase 1 (REQUIRED): Non-negotiable for publication
- Phase 2 (RECOMMENDED): Expected by serious reviewers
- Phase 3 (OPTIONAL): Nice to have, time permitting

Run with: python ablation_runner.py --phase 1
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import config


class AblationPhase(Enum):
    """Publication readiness phases."""
    REQUIRED = 1      # Must have for any publication
    RECOMMENDED = 2   # Expected for tier-1 venues
    OPTIONAL = 3      # Nice to have


@dataclass
class AblationVariant:
    """Single ablation study configuration."""
    name: str
    description: str
    phase: AblationPhase

    # What to modify
    exclude_keywords: list[str] = field(default_factory=list)
    exclude_outlets: list[str] = field(default_factory=list)
    exclude_keyword_sets: list[str] = field(default_factory=list)

    # Classification overrides
    require_uncertainty: bool = True
    use_llm_validation: bool = False
    llm_confidence_threshold: float = 0.85

    # Normalization overrides
    base_start: Optional[str] = None
    base_end: Optional[str] = None
    skip_volume_scaling: bool = False
    skip_standardization: bool = False

    # Index type selection
    index_types: list[str] = field(default_factory=lambda: [
        "CPU", "CPU_impl", "CPU_reversal", "salience_ira", "salience_obbba"
    ])

    # Placebo mode
    is_placebo: bool = False
    placebo_domain_terms: list[str] = field(default_factory=list)


# =============================================================================
# PRE-REGISTERED EVENTS FOR VALIDATION
# =============================================================================
# These events are specified BEFORE running the index.
# CPU should spike/drop at these dates in expected directions.

VALIDATION_EVENTS = [
    {
        "date": "2021-01",
        "event": "Biden inauguration",
        "expected_cpu": "decrease",
        "expected_impl": "stable",
        "expected_reversal": "decrease",
        "description": "Pro-climate administration begins, reversal risk drops",
    },
    {
        "date": "2022-07",
        "event": "Manchin announces support for IRA",
        "expected_cpu": "spike_then_drop",
        "expected_impl": "increase",
        "expected_reversal": "decrease",
        "description": "Unexpected deal creates implementation questions, reduces reversal fear",
    },
    {
        "date": "2022-08",
        "event": "IRA signed into law",
        "expected_cpu": "decrease",
        "expected_impl": "increase",
        "expected_reversal": "decrease",
        "description": "Major policy enacted, implementation uncertainty high, reversal low",
    },
    {
        "date": "2023-01",
        "event": "Treasury guidance delays",
        "expected_cpu": "increase",
        "expected_impl": "increase",
        "expected_reversal": "stable",
        "description": "Implementation delays drive uncertainty",
    },
    {
        "date": "2024-01",
        "event": "Election year begins",
        "expected_cpu": "increase",
        "expected_impl": "stable",
        "expected_reversal": "increase",
        "description": "Electoral uncertainty increases reversal risk",
    },
    {
        "date": "2024-11",
        "event": "Trump wins election",
        "expected_cpu": "spike",
        "expected_impl": "stable",
        "expected_reversal": "spike",
        "description": "Major reversal risk spike, campaign promised rollbacks",
    },
    {
        "date": "2025-01",
        "event": "Trump inauguration + executive orders",
        "expected_cpu": "spike",
        "expected_impl": "increase",
        "expected_reversal": "spike",
        "description": "Executive actions freeze IRA implementation, increase reversal",
    },
    {
        "date": "2025-02",
        "event": "OBBBA introduced",
        "expected_cpu": "spike",
        "expected_impl": "stable",
        "expected_reversal": "spike",
        "description": "Legislative threat to IRA provisions",
    },
]


# =============================================================================
# PHASE 1: REQUIRED ABLATION STUDIES
# =============================================================================

def get_keyword_dropping_variants() -> list[AblationVariant]:
    """
    1.1 Keyword Dropping Sensitivity (BBD-style)

    Tests whether any single keyword dominates the index.
    Success: correlation >= 0.95 when dropping any keyword.
    """
    variants = []

    # Drop each uncertainty term
    for term in config.UNCERTAINTY_TERMS:
        variants.append(AblationVariant(
            name=f"drop_uncertainty_{term.replace(' ', '_')}",
            description=f"Drop '{term}' from UNCERTAINTY_TERMS",
            phase=AblationPhase.REQUIRED,
            exclude_keywords=[term],
        ))

    # Drop each climate term
    for term in config.CLIMATE_TERMS:
        variants.append(AblationVariant(
            name=f"drop_climate_{term.replace(' ', '_')}",
            description=f"Drop '{term}' from CLIMATE_TERMS",
            phase=AblationPhase.REQUIRED,
            exclude_keywords=[term],
        ))

    # Drop each policy term
    for term in config.POLICY_TERMS:
        variants.append(AblationVariant(
            name=f"drop_policy_{term.replace(' ', '_')}",
            description=f"Drop '{term}' from POLICY_TERMS",
            phase=AblationPhase.REQUIRED,
            exclude_keywords=[term],
        ))

    return variants


def get_outlet_robustness_variants() -> list[AblationVariant]:
    """
    1.3 Outlet Composition Robustness

    Tests whether results depend on particular newspaper choice.
    Success: correlation >= 0.80 when dropping any single outlet.
    """
    outlets = list(config.SOURCE_IDS.keys())

    variants = []
    for outlet in outlets:
        variants.append(AblationVariant(
            name=f"drop_outlet_{outlet.replace(' ', '_').lower()}",
            description=f"Exclude {outlet} from index calculation",
            phase=AblationPhase.REQUIRED,
            exclude_outlets=[outlet],
        ))

    return variants


def get_llm_comparison_variant() -> AblationVariant:
    """
    1.4 LLM vs. Keyword-Only Comparison

    Tests whether LLM validation adds value over pure keyword matching.
    Success: correlation >= 0.90 between keyword-only and LLM-validated.
    """
    return AblationVariant(
        name="llm_validated",
        description="Apply LLM validation to keyword classifications",
        phase=AblationPhase.REQUIRED,
        use_llm_validation=True,
        llm_confidence_threshold=0.85,
    )


def get_uncertainty_requirement_variant() -> AblationVariant:
    """
    1.5 Uncertainty Requirement Test

    Tests whether requiring uncertainty terms (Steve's fix) matters.
    This validates the "rollback alone != uncertainty" design decision.
    """
    return AblationVariant(
        name="no_uncertainty_requirement",
        description="Count direction terms without requiring uncertainty",
        phase=AblationPhase.REQUIRED,
        require_uncertainty=False,
    )


# =============================================================================
# PHASE 2: STRONGLY RECOMMENDED ABLATION STUDIES
# =============================================================================

def get_base_period_variants() -> list[AblationVariant]:
    """
    2.2 Normalization Base Period Sensitivity

    Tests whether CPU values are sensitive to base period choice.
    Success: correlation >= 0.99 across different base periods.
    """
    return [
        AblationVariant(
            name="base_2021",
            description="Normalize to 2021 (first year)",
            phase=AblationPhase.RECOMMENDED,
            base_start="2021-01",
            base_end="2021-12",
        ),
        AblationVariant(
            name="base_2022",
            description="Normalize to 2022 (pre-IRA signing year)",
            phase=AblationPhase.RECOMMENDED,
            base_start="2022-01",
            base_end="2022-12",
        ),
        AblationVariant(
            name="base_pre_ira",
            description="Normalize to pre-IRA period (2021-01 to 2022-07)",
            phase=AblationPhase.RECOMMENDED,
            base_start="2021-01",
            base_end="2022-07",
        ),
        AblationVariant(
            name="base_post_ira",
            description="Normalize to post-IRA period (2022-09 to 2024-10)",
            phase=AblationPhase.RECOMMENDED,
            base_start="2022-09",
            base_end="2024-10",
        ),
    ]


def get_classification_threshold_variants() -> list[AblationVariant]:
    """
    2.3 Classification Threshold Sensitivity

    Tests whether CPU is sensitive to LLM confidence threshold.
    Success: correlation >= 0.95 across thresholds.
    """
    return [
        AblationVariant(
            name="llm_threshold_50",
            description="LLM validation with 50% confidence threshold (inclusive)",
            phase=AblationPhase.RECOMMENDED,
            use_llm_validation=True,
            llm_confidence_threshold=0.50,
        ),
        AblationVariant(
            name="llm_threshold_70",
            description="LLM validation with 70% confidence threshold",
            phase=AblationPhase.RECOMMENDED,
            use_llm_validation=True,
            llm_confidence_threshold=0.70,
        ),
        AblationVariant(
            name="llm_threshold_95",
            description="LLM validation with 95% confidence threshold (strict)",
            phase=AblationPhase.RECOMMENDED,
            use_llm_validation=True,
            llm_confidence_threshold=0.95,
        ),
    ]


def get_placebo_variants() -> list[AblationVariant]:
    """
    2.5 Placebo Tests (Negative Control)

    Tests whether CPU responds to climate-SPECIFIC uncertainty.
    Success: correlation < 0.70 with trade/monetary policy indices.
    """
    return [
        AblationVariant(
            name="placebo_trade_policy",
            description="Trade Policy Uncertainty Index (placebo)",
            phase=AblationPhase.RECOMMENDED,
            is_placebo=True,
            placebo_domain_terms=config.TRADE_TERMS,
            index_types=["TPU"],  # Trade Policy Uncertainty
        ),
        AblationVariant(
            name="placebo_monetary_policy",
            description="Monetary Policy Uncertainty Index (placebo)",
            phase=AblationPhase.RECOMMENDED,
            is_placebo=True,
            placebo_domain_terms=config.MONETARY_TERMS,
            index_types=["MPU"],  # Monetary Policy Uncertainty
        ),
    ]


def get_decomposition_validation_variant() -> AblationVariant:
    """
    2.6 Decomposition Validation

    Tests whether CPU_impl and CPU_reversal capture distinct concepts.
    Success: correlation 0.40-0.80 (related but distinct).
    """
    return AblationVariant(
        name="decomposition_only",
        description="Build only CPU_impl and CPU_reversal for correlation analysis",
        phase=AblationPhase.RECOMMENDED,
        index_types=["CPU_impl", "CPU_reversal"],
    )


def get_normalization_step_variants() -> list[AblationVariant]:
    """
    2.7 Normalization Step Sensitivity

    Tests contribution of each BBD normalization step.
    """
    return [
        AblationVariant(
            name="skip_volume_scaling",
            description="Skip Step 1: No outlet volume scaling",
            phase=AblationPhase.RECOMMENDED,
            skip_volume_scaling=True,
        ),
        AblationVariant(
            name="skip_standardization",
            description="Skip Step 2: No unit std dev standardization",
            phase=AblationPhase.RECOMMENDED,
            skip_standardization=True,
        ),
    ]


# =============================================================================
# PHASE 3: OPTIONAL ABLATION STUDIES
# =============================================================================

def get_keyword_set_exclusion_variants() -> list[AblationVariant]:
    """
    3.1 Keyword Set Exclusion

    Tests contribution of entire keyword categories.
    """
    return [
        AblationVariant(
            name="exclude_implementation_terms",
            description="Exclude all IMPLEMENTATION_TERMS",
            phase=AblationPhase.OPTIONAL,
            exclude_keyword_sets=["IMPLEMENTATION_TERMS"],
            index_types=["CPU", "CPU_reversal", "salience_ira", "salience_obbba"],
        ),
        AblationVariant(
            name="exclude_reversal_terms",
            description="Exclude all REVERSAL_TERMS",
            phase=AblationPhase.OPTIONAL,
            exclude_keyword_sets=["REVERSAL_TERMS"],
            index_types=["CPU", "CPU_impl", "salience_ira", "salience_obbba"],
        ),
        AblationVariant(
            name="exclude_regime_terms",
            description="Exclude all REGIME_IRA_TERMS and REGIME_OBBBA_TERMS",
            phase=AblationPhase.OPTIONAL,
            exclude_keyword_sets=["REGIME_IRA_TERMS", "REGIME_OBBBA_TERMS"],
            index_types=["CPU", "CPU_impl", "CPU_reversal"],
        ),
    ]


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def get_all_ablation_variants(phase: Optional[AblationPhase] = None) -> list[AblationVariant]:
    """
    Get all ablation variants, optionally filtered by phase.

    Args:
        phase: If specified, only return variants for this phase and earlier.
               E.g., phase=RECOMMENDED returns REQUIRED + RECOMMENDED.

    Returns:
        List of AblationVariant configurations.
    """
    all_variants = []

    # Baseline (always included)
    all_variants.append(AblationVariant(
        name="baseline",
        description="Baseline CPU index with default configuration",
        phase=AblationPhase.REQUIRED,
    ))

    # Phase 1: Required
    all_variants.extend(get_keyword_dropping_variants())
    all_variants.extend(get_outlet_robustness_variants())
    all_variants.append(get_llm_comparison_variant())
    all_variants.append(get_uncertainty_requirement_variant())

    # Phase 2: Recommended
    all_variants.extend(get_base_period_variants())
    all_variants.extend(get_classification_threshold_variants())
    all_variants.extend(get_placebo_variants())
    all_variants.append(get_decomposition_validation_variant())
    all_variants.extend(get_normalization_step_variants())

    # Phase 3: Optional
    all_variants.extend(get_keyword_set_exclusion_variants())

    # Filter by phase if specified
    if phase is not None:
        all_variants = [v for v in all_variants if v.phase.value <= phase.value]

    return all_variants


def get_phase_summary() -> dict:
    """Get summary of ablation studies by phase."""
    all_variants = get_all_ablation_variants()

    summary = {
        "Phase 1 (Required)": [],
        "Phase 2 (Recommended)": [],
        "Phase 3 (Optional)": [],
    }

    for v in all_variants:
        if v.phase == AblationPhase.REQUIRED:
            summary["Phase 1 (Required)"].append(v.name)
        elif v.phase == AblationPhase.RECOMMENDED:
            summary["Phase 2 (Recommended)"].append(v.name)
        else:
            summary["Phase 3 (Optional)"].append(v.name)

    return summary


# =============================================================================
# SUCCESS CRITERIA
# =============================================================================

SUCCESS_CRITERIA = {
    "keyword_dropping": {
        "metric": "correlation_with_baseline",
        "threshold": 0.95,
        "description": "Index should correlate >= 0.95 when dropping any single keyword",
    },
    "outlet_robustness": {
        "metric": "correlation_with_baseline",
        "threshold": 0.80,
        "description": "Index should correlate >= 0.80 when dropping any single outlet",
    },
    "llm_validation": {
        "metric": "correlation_keyword_vs_llm",
        "threshold": 0.90,
        "description": "Keyword-only and LLM-validated should correlate >= 0.90",
    },
    "base_period": {
        "metric": "correlation_across_periods",
        "threshold": 0.99,
        "description": "Different base periods should correlate >= 0.99",
    },
    "placebo": {
        "metric": "correlation_with_placebo",
        "threshold": 0.70,
        "max_allowed": True,  # correlation should be BELOW threshold
        "description": "CPU should correlate < 0.70 with trade/monetary policy indices",
    },
    "decomposition": {
        "metric": "impl_reversal_correlation",
        "min_threshold": 0.40,
        "max_threshold": 0.80,
        "description": "CPU_impl and CPU_reversal should correlate 0.40-0.80 (related but distinct)",
    },
    "event_validation": {
        "metric": "event_direction_accuracy",
        "threshold": 0.75,
        "description": ">= 75% of pre-specified events should show expected direction",
    },
}


if __name__ == "__main__":
    # Print summary when run directly
    summary = get_phase_summary()

    print("=" * 60)
    print("ABLATION STUDY CONFIGURATION SUMMARY")
    print("=" * 60)

    for phase, variants in summary.items():
        print(f"\n{phase}: {len(variants)} variants")
        for v in variants[:5]:
            print(f"  - {v}")
        if len(variants) > 5:
            print(f"  ... and {len(variants) - 5} more")

    total = sum(len(v) for v in summary.values())
    print(f"\nTotal ablation variants: {total}")

    print("\n" + "=" * 60)
    print("VALIDATION EVENTS")
    print("=" * 60)
    for event in VALIDATION_EVENTS:
        print(f"\n{event['date']}: {event['event']}")
        print(f"  Expected CPU: {event['expected_cpu']}")
        print(f"  Expected impl: {event['expected_impl']}")
        print(f"  Expected reversal: {event['expected_reversal']}")
