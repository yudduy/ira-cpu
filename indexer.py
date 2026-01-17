"""
CPU Index calculation for CPU Index Builder

Calculates the Climate Policy Uncertainty index following
Baker, Bloom & Davis (2016) methodology, extended with
asymmetric uncertainty decomposition from:

- Segal, Shaliastovich & Yaron (2015) "Good and Bad Uncertainty"
- Forni, Gambetti & Sala (2025) "Downside and Upside Uncertainty Shocks"

Three indices are calculated:
- CPU (standard): neutral uncertainty
- CPU-Down: downside/rollback uncertainty
- CPU-Up: upside/expansion uncertainty

Plus a directional balance metric:
- CPU-Direction = (up - down) / (up + down), ranges -1 to +1

All normalized so mean = 100 for interpretability.
"""

import statistics

import db


def calculate_raw_index() -> list[dict]:
    """
    Calculate raw CPU ratios for each month from database.

    Returns list of dicts with:
    - month, denominator
    - numerator, raw_ratio (standard CPU)
    - numerator_down, raw_ratio_down (CPU-Down)
    - numerator_up, raw_ratio_up (CPU-Up)
    """
    progress = db.get_all_progress()

    # Group by month
    months_data = {}
    for record in progress:
        month = record["month"]
        if month not in months_data:
            months_data[month] = {}
        months_data[month][record["query_type"]] = record["count"]

    # Calculate ratios for all index types
    results = []
    for month in sorted(months_data.keys()):
        data = months_data[month]

        # Need at least denominator and standard numerator
        if "denominator" not in data or "numerator" not in data:
            continue

        denom = data["denominator"]
        numer = data["numerator"]
        numer_down = data.get("numerator_down", 0)
        numer_up = data.get("numerator_up", 0)

        # Calculate ratios (handle division by zero)
        ratio = numer / denom if denom > 0 else 0.0
        ratio_down = numer_down / denom if denom > 0 else 0.0
        ratio_up = numer_up / denom if denom > 0 else 0.0

        results.append({
            "month": month,
            "denominator": denom,
            # Standard CPU
            "numerator": numer,
            "raw_ratio": ratio,
            # CPU-Down (downside uncertainty)
            "numerator_down": numer_down,
            "raw_ratio_down": ratio_down,
            # CPU-Up (upside uncertainty)
            "numerator_up": numer_up,
            "raw_ratio_up": ratio_up,
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

    Uses COMMON base period for all three indices to allow
    direct comparison of levels (per research recommendation).

    Args:
        raw_values: List from calculate_raw_index()
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

    # Extract raw ratio series
    raw_ratios = [v["raw_ratio"] for v in raw_values]

    # Check if directional data is available (backward compatibility)
    has_directional = "raw_ratio_down" in raw_values[0] if raw_values else False

    if has_directional:
        raw_ratios_down = [v["raw_ratio_down"] for v in raw_values]
        raw_ratios_up = [v["raw_ratio_up"] for v in raw_values]

    # Normalize each series
    normalized = _normalize_series(raw_ratios, base_indices)

    if has_directional:
        normalized_down = _normalize_series(raw_ratios_down, base_indices)
        normalized_up = _normalize_series(raw_ratios_up, base_indices)

    # Add normalized values and calculate direction
    for i, v in enumerate(raw_values):
        v["normalized"] = normalized[i]

        if has_directional:
            v["normalized_down"] = normalized_down[i]
            v["normalized_up"] = normalized_up[i]

            # CPU-Direction: (up - down) / (up + down)
            # Ranges from -1 (all downside) to +1 (all upside)
            up_val = v.get("numerator_up", 0)
            down_val = v.get("numerator_down", 0)
            if up_val + down_val > 0:
                v["cpu_direction"] = (up_val - down_val) / (up_val + down_val)
            else:
                v["cpu_direction"] = 0.0

    return raw_values


def build_index(base_start: str = None, base_end: str = None) -> dict:
    """
    Build complete CPU indices from database.

    Calculates three indices:
    - CPU (standard): general uncertainty
    - CPU-Down: downside/rollback uncertainty
    - CPU-Up: upside/expansion uncertainty

    Plus CPU-Direction metric.

    Returns dict with metadata and series.
    """
    raw = calculate_raw_index()

    if not raw:
        return {
            "status": "error",
            "message": "No data found. Run data collection first.",
        }

    normalized = normalize_index(raw, base_start, base_end)

    def calc_stats(values: list[float]) -> dict:
        """Calculate mean and std for a series."""
        return {
            "mean": statistics.mean(values) if values else 0,
            "std": statistics.stdev(values) if len(values) > 1 else 0,
        }

    # Check if directional data is available
    has_directional = "normalized_down" in normalized[0] if normalized else False

    # Calculate statistics for standard CPU (always available)
    stats = {
        "standard": {
            "mean_raw": calc_stats([v["raw_ratio"] for v in normalized])["mean"],
            "std_raw": calc_stats([v["raw_ratio"] for v in normalized])["std"],
            "mean_norm": calc_stats([v["normalized"] for v in normalized])["mean"],
            "std_norm": calc_stats([v["normalized"] for v in normalized])["std"],
        }
    }

    # Calculate directional statistics only if available
    if has_directional:
        for key, ratio_key, norm_key in [
            ("down", "raw_ratio_down", "normalized_down"),
            ("up", "raw_ratio_up", "normalized_up"),
        ]:
            stats[key] = {
                "mean_raw": calc_stats([v[ratio_key] for v in normalized])["mean"],
                "std_raw": calc_stats([v[ratio_key] for v in normalized])["std"],
                "mean_norm": calc_stats([v[norm_key] for v in normalized])["mean"],
                "std_norm": calc_stats([v[norm_key] for v in normalized])["std"],
            }

        # CPU-Direction statistics
        directions = [v["cpu_direction"] for v in normalized]
        dir_stats = calc_stats(directions)
        stats["direction"] = {
            "mean": dir_stats["mean"],
            "std": dir_stats["std"],
            "min": min(directions) if directions else 0,
            "max": max(directions) if directions else 0,
        }
    else:
        # Default stats when directional data not available
        stats["down"] = {"mean_raw": 0, "std_raw": 0, "mean_norm": 0, "std_norm": 0}
        stats["up"] = {"mean_raw": 0, "std_raw": 0, "mean_norm": 0, "std_norm": 0}
        stats["direction"] = {"mean": 0, "std": 0, "min": 0, "max": 0}

    # Save to database
    for v in normalized:
        db.save_index_value(
            month=v["month"],
            denominator=v["denominator"],
            numerator=v["numerator"],
            raw_ratio=v["raw_ratio"],
            normalized=v["normalized"],
            numerator_down=v.get("numerator_down"),
            raw_ratio_down=v.get("raw_ratio_down"),
            normalized_down=v.get("normalized_down"),
            numerator_up=v.get("numerator_up"),
            raw_ratio_up=v.get("raw_ratio_up"),
            normalized_up=v.get("normalized_up"),
            cpu_direction=v.get("cpu_direction"),
        )

    return {
        "status": "success",
        "metadata": {
            "period": f"{normalized[0]['month']} to {normalized[-1]['month']}",
            "base_period": f"{base_start or 'all'} to {base_end or 'all'}",
            "num_months": len(normalized),
            # Standard CPU stats
            "mean_raw_ratio": stats["standard"]["mean_raw"],
            "std_raw_ratio": stats["standard"]["std_raw"],
            "mean_normalized": stats["standard"]["mean_norm"],
            # CPU-Down stats
            "mean_normalized_down": stats["down"]["mean_norm"],
            "std_normalized_down": stats["down"]["std_norm"],
            # CPU-Up stats
            "mean_normalized_up": stats["up"]["mean_norm"],
            "std_normalized_up": stats["up"]["std_norm"],
            # Direction stats
            "mean_direction": stats["direction"]["mean"],
            "direction_range": f"{stats['direction']['min']:.2f} to {stats['direction']['max']:.2f}",
        },
        "series": normalized,
    }


def validate_against_events() -> dict:
    """
    Sanity check: Compare index to known policy events.
    """
    # Key events to check
    events = [
        {"date": "2022-07", "event": "Manchin withdraws support", "expected": "spike"},
        {"date": "2022-08", "event": "IRA signed into law", "expected": "drop"},
        {"date": "2024-11", "event": "Trump wins election", "expected": "spike"},
        {"date": "2025-01", "event": "Trump executive orders", "expected": "spike"},
    ]

    index_values = db.get_all_index_values()
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

        current = value_by_month[month]["normalized"]

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
            "actual_cpu": round(current, 1),
            "prior_cpu": round(prior, 1),
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


def get_index_summary() -> dict:
    """Get summary statistics for display."""
    values = db.get_all_index_values()

    if not values:
        return {"status": "empty", "message": "No index calculated yet."}

    norm = [v["normalized"] for v in values if v["normalized"]]

    if not norm:
        return {"status": "incomplete", "message": "Index not normalized yet."}

    # Find peaks and troughs
    sorted_by_value = sorted(values, key=lambda x: x["normalized"] or 0, reverse=True)

    return {
        "status": "ready",
        "period": f"{values[0]['month']} to {values[-1]['month']}",
        "num_months": len(values),
        "mean": round(statistics.mean(norm), 1),
        "std": round(statistics.stdev(norm), 1) if len(norm) > 1 else 0,
        "min": round(min(norm), 1),
        "max": round(max(norm), 1),
        "top_3_peaks": [
            {"month": v["month"], "cpu": round(v["normalized"], 1)}
            for v in sorted_by_value[:3]
        ],
        "top_3_troughs": [
            {"month": v["month"], "cpu": round(v["normalized"], 1)}
            for v in sorted_by_value[-3:]
        ],
    }
