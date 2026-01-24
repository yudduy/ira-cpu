"""
CSV Export Functions for CPU Index Deliverables

Exports CPU index data to CSV files following BBD/policyuncertainty.com conventions.
All exports use UTF-8 encoding and handle NaN values gracefully.

Output files:
- cpu_monthly.csv: Main CPU index time series
- cpu_decomposition.csv: CPU with impl/reversal breakdown
- cpu_salience.csv: IRA/OBBBA mention counts
- cpu_robustness.csv: Outlet-level indices (wide format)
"""

import csv
import math
from pathlib import Path
from typing import Union


def export_monthly_index(
    index_data: dict,
    output_path: Union[str, Path],
) -> Path:
    """
    Export main CPU index to CSV.

    Args:
        index_data: Dict mapping months to {cpu, denominator}
                   e.g., {"2021-01": {"cpu": 100.0, "denominator": 1000}}
        output_path: Path for output CSV file

    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    sorted_months = sorted(index_data.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "cpu", "denominator"])

        for month in sorted_months:
            data = index_data[month]
            row = [
                month,
                _format_value(data.get("cpu", "")),
                _format_value(data.get("denominator", "")),
            ]
            writer.writerow(row)

    return output_path


def export_decomposition(
    index_data: dict,
    output_path: Union[str, Path],
) -> Path:
    """
    Export CPU decomposition (impl/reversal breakdown) to CSV.

    Args:
        index_data: Dict mapping months to {cpu, cpu_impl, cpu_reversal, denominator}
        output_path: Path for output CSV file

    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    sorted_months = sorted(index_data.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "cpu", "cpu_impl", "cpu_reversal", "denominator"])

        for month in sorted_months:
            data = index_data[month]
            row = [
                month,
                _format_value(data.get("cpu", "")),
                _format_value(data.get("cpu_impl", "")),
                _format_value(data.get("cpu_reversal", "")),
                _format_value(data.get("denominator", "")),
            ]
            writer.writerow(row)

    return output_path


def export_salience(
    index_data: dict,
    output_path: Union[str, Path],
) -> Path:
    """
    Export IRA/OBBBA salience counts to CSV.

    Args:
        index_data: Dict mapping months to {salience_ira, salience_obbba, denominator}
        output_path: Path for output CSV file

    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    sorted_months = sorted(index_data.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "salience_ira", "salience_obbba", "denominator"])

        for month in sorted_months:
            data = index_data[month]
            row = [
                month,
                _format_value(data.get("salience_ira", "")),
                _format_value(data.get("salience_obbba", "")),
                _format_value(data.get("denominator", "")),
            ]
            writer.writerow(row)

    return output_path


def export_outlet_robustness(
    outlet_data: dict,
    output_path: Union[str, Path],
) -> Path:
    """
    Export outlet-level CPU indices in wide format.

    Args:
        outlet_data: Dict mapping outlet names to {month: cpu_value}
                    e.g., {"NYT": {"2021-01": 98.5, "2021-02": 102.3}}
        output_path: Path for output CSV file

    Returns:
        Path to created file
    """
    output_path = Path(output_path)

    if not outlet_data:
        # Create empty file with just header
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date"])
        return output_path

    # Collect all months across all outlets
    all_months = set()
    for series in outlet_data.values():
        all_months.update(series.keys())
    sorted_months = sorted(all_months)

    # Get outlet names (preserve order)
    outlets = list(outlet_data.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date"] + outlets)

        for month in sorted_months:
            row = [month]
            for outlet in outlets:
                value = outlet_data[outlet].get(month, "")
                row.append(_format_value(value))
            writer.writerow(row)

    return output_path


def export_all_csvs(
    index_data: dict,
    outlet_data: dict,
    output_dir: Union[str, Path],
) -> list[Path]:
    """
    Export all CSV files in one call.

    Args:
        index_data: Dict mapping months to full index data
                   (cpu, cpu_impl, cpu_reversal, salience_ira, salience_obbba, denominator)
        outlet_data: Dict mapping outlet names to {month: cpu_value}
        output_dir: Directory to write CSV files

    Returns:
        List of paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    # 1. Monthly index
    paths.append(export_monthly_index(index_data, output_dir / "cpu_monthly.csv"))

    # 2. Decomposition
    paths.append(export_decomposition(index_data, output_dir / "cpu_decomposition.csv"))

    # 3. Salience
    paths.append(export_salience(index_data, output_dir / "cpu_salience.csv"))

    # 4. Outlet robustness
    paths.append(export_outlet_robustness(outlet_data, output_dir / "cpu_robustness.csv"))

    return paths


def _format_value(value) -> str:
    """Format a value for CSV output, handling NaN."""
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return str(round(value, 2)) if value != int(value) else str(int(value))
    return str(value)
