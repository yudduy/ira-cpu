"""
CPU Index calculation for CPU Index Builder

Calculates the Climate Policy Uncertainty index following
Baker, Bloom & Davis (2016) methodology:

CPU_t = (uncertainty articles) / (all climate-policy articles)

Normalized so mean = 100 for interpretability.
"""

import statistics

import db


def calculate_raw_index() -> list[dict]:
    """
    Calculate raw CPU ratio for each month from database.

    Returns list of dicts with month, denominator, numerator, raw_ratio
    """
    progress = db.get_all_progress()

    # Group by month
    months_data = {}
    for record in progress:
        month = record["month"]
        if month not in months_data:
            months_data[month] = {}
        months_data[month][record["query_type"]] = record["count"]

    # Calculate ratios
    results = []
    for month in sorted(months_data.keys()):
        data = months_data[month]
        if "denominator" in data and "numerator" in data:
            denom = data["denominator"]
            numer = data["numerator"]
            ratio = numer / denom if denom > 0 else 0.0

            results.append({
                "month": month,
                "denominator": denom,
                "numerator": numer,
                "raw_ratio": ratio,
            })

    return results


def normalize_index(
    raw_values: list[dict],
    base_start: str = None,
    base_end: str = None,
) -> list[dict]:
    """
    Normalize index so mean = 100 over base period.

    Args:
        raw_values: List from calculate_raw_index()
        base_start: Start of base period (None = use all)
        base_end: End of base period (None = use all)

    Returns:
        Same list with added 'normalized' field
    """
    if not raw_values:
        return []

    # Filter to base period if specified
    if base_start and base_end:
        base_values = [
            v for v in raw_values
            if base_start <= v["month"] <= base_end
        ]
    else:
        base_values = raw_values

    # Calculate mean of raw ratios in base period
    if not base_values:
        mean_ratio = statistics.mean([v["raw_ratio"] for v in raw_values])
    else:
        mean_ratio = statistics.mean([v["raw_ratio"] for v in base_values])

    # Normalize: (raw / mean) * 100
    for v in raw_values:
        if mean_ratio > 0:
            v["normalized"] = (v["raw_ratio"] / mean_ratio) * 100
        else:
            v["normalized"] = 0.0

    return raw_values


def build_index(base_start: str = None, base_end: str = None) -> dict:
    """
    Build complete CPU index from database.

    Returns dict with metadata and series.
    """
    raw = calculate_raw_index()

    if not raw:
        return {
            "status": "error",
            "message": "No data found. Run data collection first.",
        }

    normalized = normalize_index(raw, base_start, base_end)

    # Calculate statistics
    ratios = [v["raw_ratio"] for v in normalized]
    norm_values = [v["normalized"] for v in normalized]

    # Save to database
    for v in normalized:
        db.save_index_value(
            month=v["month"],
            denominator=v["denominator"],
            numerator=v["numerator"],
            raw_ratio=v["raw_ratio"],
            normalized=v["normalized"],
        )

    return {
        "status": "success",
        "metadata": {
            "period": f"{normalized[0]['month']} to {normalized[-1]['month']}",
            "base_period": f"{base_start or 'all'} to {base_end or 'all'}",
            "num_months": len(normalized),
            "mean_raw_ratio": statistics.mean(ratios),
            "std_raw_ratio": statistics.stdev(ratios) if len(ratios) > 1 else 0,
            "mean_normalized": statistics.mean(norm_values),  # Should be ~100
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
