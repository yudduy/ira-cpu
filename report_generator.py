"""
Report Generator for CPU Index Deliverables

Generates complete deliverable package with one function call:
- All CSV exports
- All PNG visualizations
- Populated methodology memo

Usage:
    from report_generator import generate_full_report
    generate_full_report(index_data, outlet_indices, output_dir="outputs")
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import config
from exports import export_all_csvs
from visualizations import (
    plot_cpu_timeseries,
    plot_cpu_decomposition,
    plot_direction_balance,
    plot_outlet_correlation_heatmap,
    plot_keyword_sensitivity,
    plot_placebo_comparison,
    plot_llm_validation_scatter,
    plot_article_volume,
    HEADLINE_EVENTS,
)


def generate_full_report(
    index_data: dict,
    outlet_indices: dict,
    output_dir: Union[str, Path] = "outputs",
    sensitivity_results: Optional[dict] = None,
    placebo_data: Optional[dict] = None,
    llm_data: Optional[dict] = None,
    base_period: str = "2021-01 to 2024-10",
    validation_results: Optional[dict] = None,
) -> dict:
    """
    Generate complete deliverable package.

    Args:
        index_data: Main index data with all components
                   {month: {cpu, cpu_impl, cpu_reversal, salience_ira, salience_obbba, denominator}}
        outlet_indices: Outlet-level CPU indices
                       {outlet: {month: cpu_value}}
        output_dir: Root output directory
        sensitivity_results: Optional keyword sensitivity results {variant: correlation}
        placebo_data: Optional dict with {"tpu": {...}, "mpu": {...}}
        llm_data: Optional dict with {"keyword_cpu": {...}, "llm_cpu": {...}}
        base_period: Base period string for memo
        validation_results: Optional event validation results

    Returns:
        Dict with paths to all generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = output_dir / "csv"
    fig_dir = output_dir / "figures"
    csv_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    result = {
        "csvs": [],
        "figures": [],
        "memo": None,
    }

    print("Generating CSV exports...")
    csv_paths = export_all_csvs(index_data, outlet_indices, csv_dir)
    result["csvs"] = [str(p) for p in csv_paths]
    print(f"  Created {len(csv_paths)} CSV files")

    print("Generating main text figures...")

    fig1 = plot_cpu_timeseries(
        index_data, HEADLINE_EVENTS, fig_dir / "fig1_cpu_timeseries.png"
    )
    result["figures"].append(str(fig1))

    fig2 = plot_cpu_decomposition(
        index_data, fig_dir / "fig2_cpu_decomposition.png"
    )
    result["figures"].append(str(fig2))

    fig3 = plot_direction_balance(
        index_data, fig_dir / "fig3_direction_balance.png"
    )
    result["figures"].append(str(fig3))

    print(f"  Created 3 main text figures")

    print("Generating appendix figures...")

    figA1 = plot_outlet_correlation_heatmap(
        outlet_indices, fig_dir / "figA1_outlet_heatmap.png"
    )
    result["figures"].append(str(figA1))

    if sensitivity_results:
        figA2 = plot_keyword_sensitivity(
            sensitivity_results, fig_dir / "figA2_keyword_sensitivity.png"
        )
        result["figures"].append(str(figA2))

    if placebo_data:
        cpu_series = {m: d.get("cpu", 0) for m, d in index_data.items()}
        figA3 = plot_placebo_comparison(
            cpu_series,
            placebo_data.get("tpu", {}),
            placebo_data.get("mpu", {}),
            fig_dir / "figA3_placebo_comparison.png"
        )
        result["figures"].append(str(figA3))

    if llm_data:
        figA4 = plot_llm_validation_scatter(
            llm_data.get("keyword_cpu", {}),
            llm_data.get("llm_cpu", {}),
            fig_dir / "figA4_llm_scatter.png"
        )
        result["figures"].append(str(figA4))

    volume_data = {m: d.get("denominator", 0) for m, d in index_data.items()}
    figA5 = plot_article_volume(
        volume_data, fig_dir / "figA5_article_volume.png"
    )
    result["figures"].append(str(figA5))

    print(f"  Created {len(result['figures']) - 3} appendix figures")

    print("Generating methodology memo...")
    memo_path = _generate_memo(
        index_data=index_data,
        output_path=output_dir / "cpu_methodology_memo.md",
        base_period=base_period,
        validation_results=validation_results,
        llm_data=llm_data,
        placebo_data=placebo_data,
    )
    result["memo"] = str(memo_path)
    print(f"  Created methodology memo")

    # Summary
    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"CSV files: {len(result['csvs'])}")
    print(f"Figures: {len(result['figures'])}")
    print(f"Memo: {result['memo']}")
    print("=" * 60)

    return result


def _generate_memo(
    index_data: dict,
    output_path: Path,
    base_period: str,
    validation_results: Optional[dict] = None,
    llm_data: Optional[dict] = None,
    placebo_data: Optional[dict] = None,
) -> Path:
    """Generate populated methodology memo from template."""

    # Read template
    template_path = Path(__file__).parent / "memo_template.md"
    with open(template_path, "r") as f:
        template = f.read()

    # Compute statistics from index_data
    sorted_months = sorted(index_data.keys())
    cpu_values = [index_data[m].get("cpu", 0) for m in sorted_months]

    current_month = sorted_months[-1] if sorted_months else "N/A"
    current_cpu = cpu_values[-1] if cpu_values else 0
    min_cpu = min(cpu_values) if cpu_values else 0
    max_cpu = max(cpu_values) if cpu_values else 0
    n_months = len(sorted_months)

    # Placeholder replacements
    replacements = {
        "{{GENERATION_DATE}}": datetime.now().strftime("%Y-%m-%d"),
        "{{BASE_PERIOD}}": base_period,
        "{{CURRENT_CPU}}": f"{current_cpu:.1f}",
        "{{CURRENT_MONTH}}": current_month,
        "{{MIN_CPU}}": f"{min_cpu:.1f}",
        "{{MAX_CPU}}": f"{max_cpu:.1f}",
        "{{N_MONTHS}}": str(n_months),

        # Event responses (placeholder - would come from validation_results)
        "{{IRA_RESPONSE}}": "decreased" if validation_results else "[TBD]",
        "{{TRUMP_RESPONSE}}": "spiked" if validation_results else "[TBD]",
        "{{OBBBA_RESPONSE}}": "spiked" if validation_results else "[TBD]",

        # Event validation placeholders
        "{{EVENT_2021_01}}": _get_event_response(validation_results, "2021-01"),
        "{{CHECK_2021_01}}": _get_event_check(validation_results, "2021-01"),
        "{{EVENT_2022_08}}": _get_event_response(validation_results, "2022-08"),
        "{{CHECK_2022_08}}": _get_event_check(validation_results, "2022-08"),
        "{{EVENT_2023_01}}": _get_event_response(validation_results, "2023-01"),
        "{{CHECK_2023_01}}": _get_event_check(validation_results, "2023-01"),
        "{{EVENT_2024_01}}": _get_event_response(validation_results, "2024-01"),
        "{{CHECK_2024_01}}": _get_event_check(validation_results, "2024-01"),
        "{{EVENT_2024_11}}": _get_event_response(validation_results, "2024-11"),
        "{{CHECK_2024_11}}": _get_event_check(validation_results, "2024-11"),
        "{{EVENT_2025_01}}": _get_event_response(validation_results, "2025-01"),
        "{{CHECK_2025_01}}": _get_event_check(validation_results, "2025-01"),
        "{{EVENT_2025_02}}": _get_event_response(validation_results, "2025-02"),
        "{{CHECK_2025_02}}": _get_event_check(validation_results, "2025-02"),
        "{{EVENT_ACCURACY}}": _get_event_accuracy(validation_results),

        # LLM validation placeholders
        "{{LLM_SAMPLE_SIZE}}": str(llm_data.get("sample_size", "N/A")) if llm_data else "N/A",
        "{{LLM_CORRELATION}}": f"{llm_data.get('correlation', 0):.2f}" if llm_data else "N/A",
        "{{FALSE_POSITIVE_RATE}}": f"{llm_data.get('false_positive_rate', 0):.1f}" if llm_data else "N/A",
        "{{FALSE_NEGATIVE_RATE}}": f"{llm_data.get('false_negative_rate', 0):.1f}" if llm_data else "N/A",

        # Placebo correlations
        "{{TPU_CORRELATION}}": f"{placebo_data.get('tpu_correlation', 0):.2f}" if placebo_data else "N/A",
        "{{MPU_CORRELATION}}": f"{placebo_data.get('mpu_correlation', 0):.2f}" if placebo_data else "N/A",

        # Keyword lists
        "{{CLIMATE_TERMS_LIST}}": ", ".join(config.CLIMATE_TERMS),
        "{{POLICY_TERMS_LIST}}": ", ".join(config.POLICY_TERMS),
        "{{UNCERTAINTY_TERMS_LIST}}": ", ".join(config.UNCERTAINTY_TERMS),
        "{{IMPLEMENTATION_TERMS_LIST}}": ", ".join(config.IMPLEMENTATION_TERMS),
        "{{REVERSAL_TERMS_LIST}}": ", ".join(config.REVERSAL_TERMS),
        "{{UPSIDE_TERMS_LIST}}": ", ".join(config.UPSIDE_TERMS),
        "{{IRA_TERMS_LIST}}": ", ".join(config.REGIME_IRA_TERMS),
        "{{OBBBA_TERMS_LIST}}": ", ".join(config.REGIME_OBBBA_TERMS),
    }

    # Apply replacements
    memo = template
    for placeholder, value in replacements.items():
        memo = memo.replace(placeholder, value)

    # Write populated memo
    with open(output_path, "w") as f:
        f.write(memo)

    return output_path


def _get_event_response(validation_results: Optional[dict], date: str) -> str:
    """Get actual event response from validation results."""
    if not validation_results:
        return "[TBD]"
    details = validation_results.get("details", [])
    for detail in details:
        if detail.get("date") == date:
            return detail.get("actual", "[TBD]")
    return "[TBD]"


def _get_event_check(validation_results: Optional[dict], date: str) -> str:
    """Get check mark for event validation."""
    if not validation_results:
        return "○"
    details = validation_results.get("details", [])
    for detail in details:
        if detail.get("date") == date:
            return "✓" if detail.get("status") == "passed" else "✗"
    return "○"


def _get_event_accuracy(validation_results: Optional[dict]) -> str:
    """Get overall event validation accuracy."""
    if not validation_results:
        return "N/A"
    accuracy = validation_results.get("accuracy")
    if accuracy is not None:
        return f"{accuracy * 100:.0f}"
    return "N/A"


if __name__ == "__main__":
    # Example usage with sample data
    sample_index = {
        "2021-01": {"cpu": 100.0, "cpu_impl": 45.0, "cpu_reversal": 55.0, "salience_ira": 0, "salience_obbba": 0, "denominator": 1250},
        "2021-06": {"cpu": 105.0, "cpu_impl": 50.0, "cpu_reversal": 55.0, "salience_ira": 5, "salience_obbba": 0, "denominator": 1300},
        "2022-01": {"cpu": 110.0, "cpu_impl": 55.0, "cpu_reversal": 55.0, "salience_ira": 10, "salience_obbba": 0, "denominator": 1400},
        "2022-08": {"cpu": 95.0, "cpu_impl": 60.0, "cpu_reversal": 35.0, "salience_ira": 50, "salience_obbba": 0, "denominator": 1500},
        "2023-01": {"cpu": 115.0, "cpu_impl": 65.0, "cpu_reversal": 50.0, "salience_ira": 30, "salience_obbba": 0, "denominator": 1450},
        "2024-01": {"cpu": 125.0, "cpu_impl": 55.0, "cpu_reversal": 70.0, "salience_ira": 25, "salience_obbba": 0, "denominator": 1600},
        "2024-11": {"cpu": 150.0, "cpu_impl": 50.0, "cpu_reversal": 100.0, "salience_ira": 40, "salience_obbba": 10, "denominator": 1800},
        "2025-01": {"cpu": 160.0, "cpu_impl": 55.0, "cpu_reversal": 105.0, "salience_ira": 35, "salience_obbba": 30, "denominator": 1900},
    }

    sample_outlets = {
        "NYT": {"2021-01": 98.0, "2022-08": 92.0, "2024-11": 155.0},
        "WSJ": {"2021-01": 102.0, "2022-08": 98.0, "2024-11": 145.0},
        "WaPo": {"2021-01": 100.0, "2022-08": 95.0, "2024-11": 150.0},
    }

    print("Generating sample report...")
    result = generate_full_report(
        index_data=sample_index,
        outlet_indices=sample_outlets,
        output_dir="outputs/sample_report",
    )
    print("\nGenerated files:")
    for key, paths in result.items():
        if isinstance(paths, list):
            for p in paths:
                print(f"  {p}")
        else:
            print(f"  {paths}")
