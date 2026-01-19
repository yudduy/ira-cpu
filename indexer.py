"""
CPU Index calculation for Climate Policy Uncertainty Index Builder

Calculates indices following Baker, Bloom & Davis (2016) methodology:
1. CPU_t: Standard climate policy uncertainty (requires uncertainty term)
2. CPU_impl_t: Implementation uncertainty (uncertainty + implementation terms)
3. CPU_reversal_t: Reversal uncertainty (uncertainty + reversal terms)

Also calculates:
- Policy Regime Salience Index: IRA/OBBBA mentions normalized
- Directional balance metric: (impl - reversal) / (impl + reversal)
- Outlet-level indices for robustness analysis

All indices normalized so base period mean = 100.
"""

import statistics

import db_postgres
import normalizer


# Index type constants
INDEX_CPU = "CPU"  # Standard uncertainty
INDEX_CPU_IMPL = "CPU_impl"  # Implementation uncertainty
INDEX_CPU_REVERSAL = "CPU_reversal"  # Reversal uncertainty
INDEX_SALIENCE_IRA = "salience_ira"  # IRA mentions
INDEX_SALIENCE_OBBBA = "salience_obbba"  # OBBBA mentions


def _map_count_row_to_raw_counts(row: dict) -> dict:
    """Map a monthly_counts row to the raw counts format."""
    return {
        "month": row["month"],
        "total_articles": row["denominator"],
        "uncertainty_count": row["numerator_cpu"],
        "implementation_uncertainty_count": row["numerator_impl"],
        "reversal_uncertainty_count": row["numerator_reversal"],
        "ira_count": 0,
        "obbba_count": 0,
    }


def calculate_raw_counts() -> list[dict]:
    """
    Get raw classification counts from database.

    Prefers count-based data (from monthly_counts table) if available,
    otherwise falls back to article-based classification data.

    Returns list of dicts with:
    - month
    - total_articles (denominator)
    - uncertainty_count (CPU numerator)
    - implementation_uncertainty_count (CPU_impl numerator)
    - reversal_uncertainty_count (CPU_reversal numerator)
    - ira_count, obbba_count (salience indices, only for article-based)
    """
    count_data = db_postgres.get_monthly_counts()
    if count_data:
        return [_map_count_row_to_raw_counts(row) for row in count_data]

    return db_postgres.get_classification_counts_by_month()


def _calculate_direction_metric(impl_count: int, reversal_count: int) -> float:
    """Calculate directional balance: (impl - reversal) / (impl + reversal)."""
    total = impl_count + reversal_count
    if total == 0:
        return 0.0
    return (impl_count - reversal_count) / total


def calculate_raw_index_values(raw_counts: list[dict]) -> list[dict]:
    """
    Calculate raw ratios from counts.

    Args:
        raw_counts: From calculate_raw_counts()

    Returns:
        List of dicts with raw ratios for each index type
    """
    results = []

    for record in raw_counts:
        denominator = record.get("total_articles", 0)
        if denominator == 0:
            continue

        # Extract counts once
        cpu_count = record.get("uncertainty_count", 0)
        impl_count = record.get("implementation_uncertainty_count", 0)
        reversal_count = record.get("reversal_uncertainty_count", 0)
        ira_count = record.get("ira_count", 0)
        obbba_count = record.get("obbba_count", 0)

        results.append({
            "month": record["month"],
            "denominator": denominator,
            # Numerators
            "numerator_cpu": cpu_count,
            "numerator_impl": impl_count,
            "numerator_reversal": reversal_count,
            "numerator_ira": ira_count,
            "numerator_obbba": obbba_count,
            # Raw ratios
            "raw_ratio_cpu": cpu_count / denominator,
            "raw_ratio_impl": impl_count / denominator,
            "raw_ratio_reversal": reversal_count / denominator,
            "raw_ratio_ira": ira_count / denominator,
            "raw_ratio_obbba": obbba_count / denominator,
            # Direction metric
            "cpu_direction": _calculate_direction_metric(impl_count, reversal_count),
        })

    return results


def _normalize_series(
    values: list[float],
    base_indices: list[int] = None,
) -> list[float]:
    """
    Normalize a series so mean = 100 over base period.

    Args:
        values: Raw ratio values
        base_indices: Indices to use for base period (None = all)

    Returns:
        Normalized values
    """
    if not values:
        return []

    # Get base period values
    if base_indices:
        base_values = [values[i] for i in base_indices if i < len(values)]
    else:
        base_values = values

    # Calculate mean
    if not base_values or all(v == 0 for v in base_values):
        mean_ratio = statistics.mean(values) if values else 1.0
    else:
        mean_ratio = statistics.mean(base_values)

    # Normalize: (raw / mean) * 100
    if mean_ratio > 0:
        return [(v / mean_ratio) * 100 for v in values]
    return [0.0] * len(values)


def normalize_index(
    raw_values: list[dict],
    base_start: str = None,
    base_end: str = None,
) -> list[dict]:
    """
    Normalize all indices so mean = 100 over base period.

    Args:
        raw_values: List from calculate_raw_index_values()
        base_start: Start of base period (None = use all)
        base_end: End of base period (None = use all)

    Returns:
        Same list with added normalized fields
    """
    if not raw_values:
        return []

    # Find base period indices
    if base_start and base_end:
        base_indices = [
            i for i, v in enumerate(raw_values)
            if base_start <= v["month"] <= base_end
        ]
    else:
        base_indices = None  # Use all

    # Normalize each index type
    index_types = [
        ("raw_ratio_cpu", "normalized_cpu"),
        ("raw_ratio_impl", "normalized_impl"),
        ("raw_ratio_reversal", "normalized_reversal"),
        ("raw_ratio_ira", "normalized_ira"),
        ("raw_ratio_obbba", "normalized_obbba"),
    ]

    for raw_key, norm_key in index_types:
        if raw_key in raw_values[0]:
            raw_ratios = [v.get(raw_key, 0) for v in raw_values]
            normalized = _normalize_series(raw_ratios, base_indices)
            for i, v in enumerate(raw_values):
                v[norm_key] = normalized[i]

    return raw_values


def build_index(
    base_start: str = None,
    base_end: str = None,
    save_to_db: bool = True,
) -> dict:
    """
    Build complete CPU indices from database.

    Calculates:
    - CPU: Standard uncertainty index
    - CPU_impl: Implementation uncertainty index
    - CPU_reversal: Reversal uncertainty index
    - Salience indices for IRA/OBBBA

    Args:
        base_start: Start of base period (None = use all)
        base_end: End of base period (None = use all)
        save_to_db: Whether to save results to database

    Returns:
        Dict with metadata and series
    """
    raw_counts = calculate_raw_counts()

    if not raw_counts:
        return {
            "status": "error",
            "message": "No data found. Run data collection first.",
        }

    raw_values = calculate_raw_index_values(raw_counts)
    if not raw_values:
        return {
            "status": "error",
            "message": "No valid data for index calculation.",
        }

    normalized = normalize_index(raw_values, base_start, base_end)

    def calc_stats(values: list[float]) -> dict:
        """Calculate mean and std for a series."""
        valid = [v for v in values if v is not None]
        return {
            "mean": statistics.mean(valid) if valid else 0,
            "std": statistics.stdev(valid) if len(valid) > 1 else 0,
        }

    # Calculate statistics for each index type
    stats = {}
    index_stat_keys = [
        ("cpu", "raw_ratio_cpu", "normalized_cpu"),
        ("impl", "raw_ratio_impl", "normalized_impl"),
        ("reversal", "raw_ratio_reversal", "normalized_reversal"),
        ("ira", "raw_ratio_ira", "normalized_ira"),
        ("obbba", "raw_ratio_obbba", "normalized_obbba"),
    ]
    for name, raw_key, norm_key in index_stat_keys:
        if raw_key in normalized[0]:
            raw_stats = calc_stats([v.get(raw_key, 0) for v in normalized])
            norm_stats = calc_stats([v.get(norm_key, 0) for v in normalized])
            stats[name] = {
                "mean_raw": raw_stats["mean"],
                "std_raw": raw_stats["std"],
                "mean_norm": norm_stats["mean"],
                "std_norm": norm_stats["std"],
            }

    # Direction statistics
    directions = [v.get("cpu_direction", 0) for v in normalized]
    dir_stats = calc_stats(directions)
    stats["direction"] = {
        "mean": dir_stats["mean"],
        "std": dir_stats["std"],
        "min": min(directions) if directions else 0,
        "max": max(directions) if directions else 0,
    }

    # Save to database
    if save_to_db:
        for v in normalized:
            # Save each index type
            for index_type, num_key, norm_key in [
                (INDEX_CPU, "numerator_cpu", "normalized_cpu"),
                (INDEX_CPU_IMPL, "numerator_impl", "normalized_impl"),
                (INDEX_CPU_REVERSAL, "numerator_reversal", "normalized_reversal"),
                (INDEX_SALIENCE_IRA, "numerator_ira", "normalized_ira"),
                (INDEX_SALIENCE_OBBBA, "numerator_obbba", "normalized_obbba"),
            ]:
                if num_key in v:
                    db_postgres.save_index_value(
                        month=v["month"],
                        index_type=index_type,
                        denominator=v["denominator"],
                        numerator=v.get(num_key, 0),
                        normalized=v.get(norm_key),
                    )

    return {
        "status": "success",
        "metadata": {
            "period": f"{normalized[0]['month']} to {normalized[-1]['month']}",
            "base_period": f"{base_start or 'all'} to {base_end or 'all'}",
            "num_months": len(normalized),
            # CPU stats
            "mean_normalized_cpu": stats.get("cpu", {}).get("mean_norm", 0),
            "std_normalized_cpu": stats.get("cpu", {}).get("std_norm", 0),
            # Implementation stats
            "mean_normalized_impl": stats.get("impl", {}).get("mean_norm", 0),
            "std_normalized_impl": stats.get("impl", {}).get("std_norm", 0),
            # Reversal stats
            "mean_normalized_reversal": stats.get("reversal", {}).get("mean_norm", 0),
            "std_normalized_reversal": stats.get("reversal", {}).get("std_norm", 0),
            # Direction stats
            "mean_direction": stats["direction"]["mean"],
            "direction_range": f"{stats['direction']['min']:.2f} to {stats['direction']['max']:.2f}",
        },
        "series": normalized,
    }


def build_outlet_level_index(
    base_start: str = None,
    base_end: str = None,
    outlets: list[str] = None,
) -> dict:
    """
    Build outlet-level CPU indices for robustness analysis.

    Each outlet gets its own normalized index, allowing comparison
    of uncertainty patterns across sources.

    Args:
        base_start: Start of base period
        base_end: End of base period
        outlets: List of outlets to include (None = all)

    Returns:
        Dict mapping outlet names to their index series
    """
    outlet_data = db_postgres.get_classification_counts_by_outlet()

    if not outlet_data:
        return {"status": "error", "message": "No outlet data found."}

    # Group by outlet
    by_outlet = {}
    for record in outlet_data:
        outlet = record.get("outlet", "Unknown")
        if outlets and outlet not in outlets:
            continue

        if outlet not in by_outlet:
            by_outlet[outlet] = {}

        month = record["month"]
        by_outlet[outlet][month] = {
            "numerator": record.get("uncertainty_count", 0),
            "denominator": record.get("total_articles", 0),
        }

    # Use BBD-style normalization for outlet-level
    result = normalizer.compute_outlet_level_cpu(
        by_outlet,
        base_start=base_start or "1900-01",
        base_end=base_end or "2099-12",
    )

    # Save to database with outlet specified
    for outlet, series in result.items():
        for month, normalized_value in series.items():
            # Get raw data for this outlet/month
            outlet_months = by_outlet.get(outlet, {})
            month_data = outlet_months.get(month, {})

            db_postgres.save_index_value(
                month=month,
                index_type=INDEX_CPU,
                outlet=outlet,
                denominator=month_data.get("denominator", 0),
                numerator=month_data.get("numerator", 0),
                normalized=normalized_value,
            )

    return {
        "status": "success",
        "num_outlets": len(result),
        "outlets": list(result.keys()),
        "series": result,
    }


def validate_against_events() -> dict:
    """
    Sanity check: Compare index to known policy events.

    Returns validation results against expected spikes/drops.
    """
    # Key events to check
    events = [
        {"date": "2022-07", "event": "Manchin withdraws support", "expected": "spike"},
        {"date": "2022-08", "event": "IRA signed into law", "expected": "drop"},
        {"date": "2024-11", "event": "Trump wins election", "expected": "spike"},
        {"date": "2025-01", "event": "Trump executive orders", "expected": "spike"},
    ]

    index_values = db_postgres.get_index_values(INDEX_CPU)
    if not index_values:
        return {"status": "error", "message": "No index values. Build index first."}

    # Create lookup
    value_by_month = {v["month"]: v for v in index_values}

    results = []
    for event in events:
        month = event["date"]
        if month not in value_by_month:
            results.append({**event, "result": "NO DATA"})
            continue

        current = value_by_month[month].get("normalized", 0)

        # Get prior month
        prior_month = _get_prior_month(month)
        prior = value_by_month.get(prior_month, {}).get("normalized", current)

        change = ((current - prior) / prior * 100) if prior > 0 else 0

        # Check if matches expectation
        if event["expected"] == "spike":
            passed = change > 10  # At least 10% increase
        else:  # drop
            passed = change < -10  # At least 10% decrease

        results.append({
            **event,
            "actual_cpu": round(current, 1) if current else 0,
            "prior_cpu": round(prior, 1) if prior else 0,
            "change_percent": round(change, 1),
            "result": "PASS" if passed else "FAIL",
        })

    passed_count = sum(1 for r in results if r.get("result") == "PASS")
    total_with_data = sum(1 for r in results if r.get("result") != "NO DATA")

    return {
        "events": results,
        "summary": f"{passed_count}/{total_with_data} events validated",
    }


def _get_prior_month(month: str) -> str:
    """Get the month before the given month."""
    year, mon = int(month[:4]), int(month[5:7])
    if mon == 1:
        return f"{year - 1}-12"
    return f"{year}-{mon - 1:02d}"


def get_index_summary(index_type: str = INDEX_CPU) -> dict:
    """
    Get summary statistics for an index type.

    Args:
        index_type: Which index to summarize (default: CPU)

    Returns:
        Summary dict with statistics
    """
    values = db_postgres.get_index_values(index_type)

    if not values:
        return {"status": "empty", "message": f"No index calculated yet for {index_type}."}

    norm = [v["normalized"] for v in values if v.get("normalized")]

    if not norm:
        return {"status": "incomplete", "message": "Index not normalized yet."}

    # Find peaks and troughs
    sorted_by_value_desc = sorted(values, key=lambda x: x.get("normalized") or 0, reverse=True)
    sorted_by_value_asc = sorted(values, key=lambda x: x.get("normalized") or 0)

    return {
        "status": "ready",
        "index_type": index_type,
        "period": f"{values[0]['month']} to {values[-1]['month']}",
        "num_months": len(values),
        "mean": round(statistics.mean(norm), 1),
        "std": round(statistics.stdev(norm), 1) if len(norm) > 1 else 0,
        "min": round(min(norm), 1),
        "max": round(max(norm), 1),
        "top_3_peaks": [
            {"month": v["month"], "cpu": round(v["normalized"], 1)}
            for v in sorted_by_value_desc[:3]
        ],
        "top_3_troughs": [
            {"month": v["month"], "cpu": round(v["normalized"], 1)}
            for v in sorted_by_value_asc[:3]
        ],
    }


def compare_index_types(
    base_start: str = None,
    base_end: str = None,
) -> dict:
    """
    Compare different index types for analysis.

    Returns correlation and divergence metrics between
    CPU, CPU_impl, and CPU_reversal.
    """
    cpu = db_postgres.get_index_values(INDEX_CPU)
    cpu_impl = db_postgres.get_index_values(INDEX_CPU_IMPL)
    cpu_reversal = db_postgres.get_index_values(INDEX_CPU_REVERSAL)

    if not cpu or not cpu_impl or not cpu_reversal:
        return {"status": "error", "message": "Missing index data. Build indices first."}

    # Align series by month
    months = sorted(set(v["month"] for v in cpu))

    cpu_by_month = {v["month"]: v.get("normalized", 0) for v in cpu}
    impl_by_month = {v["month"]: v.get("normalized", 0) for v in cpu_impl}
    reversal_by_month = {v["month"]: v.get("normalized", 0) for v in cpu_reversal}

    # Calculate correlation
    aligned_cpu = [cpu_by_month.get(m, 0) for m in months]
    aligned_impl = [impl_by_month.get(m, 0) for m in months]
    aligned_reversal = [reversal_by_month.get(m, 0) for m in months]

    import numpy as np

    def corr(a, b):
        if len(a) < 2:
            return 0.0
        arr_a = np.array(a)
        arr_b = np.array(b)
        if np.std(arr_a) == 0 or np.std(arr_b) == 0:
            return 0.0
        return float(np.corrcoef(arr_a, arr_b)[0, 1])

    return {
        "status": "success",
        "num_months": len(months),
        "correlations": {
            "cpu_impl": corr(aligned_cpu, aligned_impl),
            "cpu_reversal": corr(aligned_cpu, aligned_reversal),
            "impl_reversal": corr(aligned_impl, aligned_reversal),
        },
        "summary": {
            "cpu_mean": statistics.mean(aligned_cpu) if aligned_cpu else 0,
            "impl_mean": statistics.mean(aligned_impl) if aligned_impl else 0,
            "reversal_mean": statistics.mean(aligned_reversal) if aligned_reversal else 0,
        },
    }


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================
# These functions provide backward compatibility with the old db module


def calculate_raw_index() -> list[dict]:
    """
    Legacy function for backward compatibility.

    Returns raw index data in the old format.
    """
    raw_counts = calculate_raw_counts()
    if not raw_counts:
        return []

    raw_values = calculate_raw_index_values(raw_counts)

    # Convert to legacy format
    results = []
    for v in raw_values:
        legacy_record = {
            "month": v["month"],
            "denominator": v["denominator"],
            "numerator": v.get("numerator_cpu", 0),
            "raw_ratio": v.get("raw_ratio_cpu", 0),
            # Map new names to old names for compatibility
            "numerator_down": v.get("numerator_reversal", 0),
            "raw_ratio_down": v.get("raw_ratio_reversal", 0),
            "numerator_up": v.get("numerator_impl", 0),
            "raw_ratio_up": v.get("raw_ratio_impl", 0),
        }
        results.append(legacy_record)

    return results


def get_all_index_values() -> list[dict]:
    """Legacy function to get all CPU index values."""
    return db_postgres.get_index_values(INDEX_CPU)


def get_full_index_data() -> dict:
    """
    Get full index data in format for exports/visualizations.

    Returns:
        Dict mapping months to index components:
        {month: {cpu, cpu_impl, cpu_reversal, salience_ira, salience_obbba, denominator}}
    """
    cpu_values = db_postgres.get_index_values(INDEX_CPU)
    impl_values = db_postgres.get_index_values(INDEX_CPU_IMPL)
    reversal_values = db_postgres.get_index_values(INDEX_CPU_REVERSAL)
    ira_values = db_postgres.get_index_values(INDEX_SALIENCE_IRA)
    obbba_values = db_postgres.get_index_values(INDEX_SALIENCE_OBBBA)

    cpu_by_month = {r["month"]: r for r in cpu_values}
    impl_by_month = {r["month"]: r for r in impl_values}
    reversal_by_month = {r["month"]: r for r in reversal_values}
    ira_by_month = {r["month"]: r for r in ira_values}
    obbba_by_month = {r["month"]: r for r in obbba_values}

    all_months = set(cpu_by_month.keys())

    result = {}
    for month in sorted(all_months):
        cpu = cpu_by_month.get(month, {})
        impl = impl_by_month.get(month, {})
        reversal = reversal_by_month.get(month, {})
        ira = ira_by_month.get(month, {})
        obbba = obbba_by_month.get(month, {})

        result[month] = {
            "cpu": cpu.get("normalized", 0),
            "cpu_impl": impl.get("normalized", 0),
            "cpu_reversal": reversal.get("normalized", 0),
            "salience_ira": ira.get("numerator", 0),
            "salience_obbba": obbba.get("numerator", 0),
            "denominator": cpu.get("denominator", 0),
        }

    return result


def get_outlet_level_indices() -> dict:
    """
    Get outlet-level CPU indices for robustness analysis.

    Returns:
        Dict mapping outlet names to their index series:
        {outlet: {month: cpu_value}}
    """
    outlet_data = db_postgres.get_classification_counts_by_outlet()

    if not outlet_data:
        return {}

    raw_by_outlet = {}
    for record in outlet_data:
        outlet = record["outlet"]
        month = record["month"]
        numerator = record.get("uncertainty_count", 0)
        denominator = record.get("total_articles", 0)

        if outlet not in raw_by_outlet:
            raw_by_outlet[outlet] = {}

        raw_by_outlet[outlet][month] = {
            "numerator": numerator,
            "denominator": denominator,
        }

    return normalizer.compute_outlet_level_cpu(
        raw_by_outlet,
        base_start="2021-01",
        base_end="2024-10",
    )
