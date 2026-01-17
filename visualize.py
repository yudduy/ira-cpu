#!/usr/bin/env python3
"""
CPU Index Visualization

Generates charts for the Climate Policy Uncertainty index,
including the asymmetric decomposition (CPU-Down, CPU-Up).

Based on:
- Segal, Shaliastovich & Yaron (2015) "Good and Bad Uncertainty"
- Forni, Gambetti & Sala (2025) "Downside and Upside Uncertainty Shocks"
"""

import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

import config


# Common policy period dates used across charts
IRA_START = datetime(2022, 8, 1)
IRA_END = datetime(2024, 12, 31)
TRUMP_START = datetime(2025, 1, 1)


def format_date_axis(ax, interval: int = 6) -> None:
    """Apply consistent date formatting to an axis."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def shade_policy_periods(ax, end_date: datetime, ira_label: str = 'IRA Period',
                         post_ira_label: str = 'Post-IRA') -> None:
    """Add shaded regions for IRA and post-IRA policy periods."""
    ax.axvspan(IRA_START, IRA_END, alpha=0.1, color='green', label=ira_label)
    ax.axvspan(TRUMP_START, end_date, alpha=0.1, color='red', label=post_ira_label)


def load_index_data(csv_path: str = None) -> list[dict]:
    """Load CPU index data from CSV file (includes directional indices)."""
    if csv_path is None:
        csv_path = os.path.join(config.EXPORT_DIR, "cpu_index.csv")

    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                'month': row['month'],
                'date': datetime.strptime(row['month'], '%Y-%m'),
                'denominator': int(row['denominator']),
                'numerator': int(row['numerator']),
                'raw_ratio': float(row['raw_ratio']),
                'normalized': float(row['normalized']),
            }

            # Load directional indices if available
            if 'normalized_down' in row and row['normalized_down']:
                entry['numerator_down'] = int(row.get('numerator_down', 0) or 0)
                entry['raw_ratio_down'] = float(row.get('raw_ratio_down', 0) or 0)
                entry['normalized_down'] = float(row['normalized_down'])
                entry['numerator_up'] = int(row.get('numerator_up', 0) or 0)
                entry['raw_ratio_up'] = float(row.get('raw_ratio_up', 0) or 0)
                entry['normalized_up'] = float(row.get('normalized_up', 0) or 0)
                entry['cpu_direction'] = float(row.get('cpu_direction', 0) or 0)
                entry['has_directional'] = True
            else:
                entry['has_directional'] = False

            data.append(entry)
    return data


def create_cpu_chart(data: list[dict], output_path: str = None) -> str:
    """
    Create CPU index chart with annotated events and policy periods.

    Returns path to saved chart.
    """
    if output_path is None:
        output_path = os.path.join(config.EXPORT_DIR, "cpu_index_chart.png")

    # Extract data for plotting
    dates = [d['date'] for d in data]
    cpu_values = [d['normalized'] for d in data]
    denominators = [d['denominator'] for d in data]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    gridspec_kw={'hspace': 0.3})

    # Shade policy periods
    ax1.axvspan(IRA_START, IRA_END, alpha=0.15, color='green',
                label='IRA Implementation Period')
    ax1.axvspan(TRUMP_START, dates[-1], alpha=0.15, color='red',
                label='Post-IRA Uncertainty Period')

    # Plot the CPU index line
    ax1.plot(dates, cpu_values, 'b-', linewidth=2, label='CPU Index')

    # Add baseline at 100
    ax1.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label='Baseline (mean=100)')

    # Annotate key events
    events = [
        {'date': datetime(2022, 7, 1), 'label': 'Manchin\nwithdraws', 'y_offset': 8, 'color': 'darkred'},
        {'date': IRA_START, 'label': 'IRA\nsigned', 'y_offset': -10, 'color': 'darkgreen'},
        {'date': datetime(2024, 11, 1), 'label': 'Trump\nwins election', 'y_offset': 8, 'color': 'darkorange'},
        {'date': TRUMP_START, 'label': 'Trump\nexec orders', 'y_offset': -8, 'color': 'darkred'},
        {'date': datetime(2025, 11, 1), 'label': 'OBBBA\nsigned', 'y_offset': 5, 'color': 'darkred'},
    ]

    for event in events:
        if event['date'] > dates[-1]:
            continue

        # Find the CPU value at this date
        closest_data = min(data, key=lambda x: abs(x['date'] - event['date']))
        cpu_at_event = closest_data['normalized']

        # Add vertical line
        ax1.axvline(x=event['date'], color=event['color'], linestyle=':', alpha=0.6)

        # Add annotation
        ax1.annotate(event['label'],
                     xy=(event['date'], cpu_at_event),
                     xytext=(event['date'], cpu_at_event + event['y_offset']),
                     fontsize=8, ha='center', color=event['color'],
                     arrowprops=dict(arrowstyle='->', color=event['color'], alpha=0.6))

    # Formatting for top panel
    ax1.set_ylabel('CPU Index (normalized, mean=100)', fontsize=11)
    ax1.set_title('Climate Policy Uncertainty Index (CPU)\nJanuary 2021 - December 2025',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(85, 120)
    ax1.grid(True, alpha=0.3)
    format_date_axis(ax1)

    # Bottom panel: Article Volume
    ax2.bar(dates, denominators, width=25, color='steelblue', alpha=0.7,
            label='Total climate-policy articles')
    ax2.set_ylabel('Monthly Article Count', fontsize=10)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    format_date_axis(ax2)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_simple_chart(data: list[dict], output_path: str = None) -> str:
    """
    Create a simple CPU index chart without annotations.

    Returns path to saved chart.
    """
    if output_path is None:
        output_path = os.path.join(config.EXPORT_DIR, "cpu_index_simple.png")

    dates = [d['date'] for d in data]
    cpu_values = [d['normalized'] for d in data]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(dates, cpu_values, 'b-', linewidth=2)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_ylabel('CPU Index (normalized, mean=100)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_title('Climate Policy Uncertainty Index (CPU)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    format_date_axis(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_directional_chart(data: list[dict], output_path: str = None) -> str:
    """
    Create chart showing CPU-Down vs CPU-Up decomposition.

    Based on Segal, Shaliastovich & Yaron (2015) methodology for
    separating "good" and "bad" uncertainty.

    Returns path to saved chart.
    """
    if output_path is None:
        output_path = os.path.join(config.EXPORT_DIR, "cpu_directional_chart.png")

    # Check if directional data is available
    if not data[0].get('has_directional', False):
        print("Warning: Directional indices not available in data")
        return None

    dates = [d['date'] for d in data]
    cpu_down = [d['normalized_down'] for d in data]
    cpu_up = [d['normalized_up'] for d in data]
    cpu_direction = [d['cpu_direction'] for d in data]

    # Create figure with three panels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12),
                                         height_ratios=[2, 2, 1],
                                         gridspec_kw={'hspace': 0.3})

    # Top panel: CPU-Down vs CPU-Up
    shade_policy_periods(ax1, dates[-1])
    ax1.plot(dates, cpu_down, 'r-', linewidth=2, label='CPU-Down (Rollback Risk)',
             marker='o', markersize=3)
    ax1.plot(dates, cpu_up, 'g-', linewidth=2, label='CPU-Up (Expansion Potential)',
             marker='s', markersize=3)
    ax1.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Index Value (normalized, mean=100)', fontsize=11)
    ax1.set_title('Asymmetric Climate Policy Uncertainty\nCPU-Down (Rollback) vs CPU-Up (Expansion)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    format_date_axis(ax1)

    # Middle panel: Spread (CPU-Down minus CPU-Up)
    spread = [d - u for d, u in zip(cpu_down, cpu_up)]
    ax2.axvspan(IRA_START, IRA_END, alpha=0.1, color='green')
    ax2.axvspan(TRUMP_START, dates[-1], alpha=0.1, color='red')

    # Fill between zero and spread
    ax2.fill_between(dates, 0, spread,
                     where=[s >= 0 for s in spread],
                     color='red', alpha=0.4, label='Downside Dominates')
    ax2.fill_between(dates, 0, spread,
                     where=[s < 0 for s in spread],
                     color='green', alpha=0.4, label='Upside Dominates')

    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.plot(dates, spread, 'k-', linewidth=1.5)

    ax2.set_ylabel('CPU-Down minus CPU-Up', fontsize=11)
    ax2.set_title('Uncertainty Spread: Positive = Downside Dominates',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    format_date_axis(ax2)

    # Bottom panel: CPU-Direction
    ax3.axvspan(IRA_START, IRA_END, alpha=0.1, color='green')
    ax3.axvspan(TRUMP_START, dates[-1], alpha=0.1, color='red')
    colors = ['red' if d < 0 else 'green' for d in cpu_direction]
    ax3.bar(dates, cpu_direction, width=25, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_ylim(-1, 1)
    ax3.set_ylabel('CPU-Direction', fontsize=10)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('CPU-Direction: (Up-Down)/(Up+Down)\n-1 = All Downside, +1 = All Upside',
                  fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    format_date_axis(ax3)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_combined_chart(data: list[dict], output_path: str = None) -> str:
    """
    Create a combined chart showing all three indices.

    Returns path to saved chart.
    """
    if output_path is None:
        output_path = os.path.join(config.EXPORT_DIR, "cpu_all_indices.png")

    # Check if directional data is available
    has_directional = data[0].get('has_directional', False)

    dates = [d['date'] for d in data]
    cpu_standard = [d['normalized'] for d in data]

    fig, ax = plt.subplots(figsize=(14, 7))

    shade_policy_periods(ax, dates[-1], post_ira_label='Post-IRA Uncertainty')
    ax.plot(dates, cpu_standard, 'b-', linewidth=2.5, label='CPU (Standard)',
            marker='o', markersize=4)

    if has_directional:
        cpu_down = [d['normalized_down'] for d in data]
        cpu_up = [d['normalized_up'] for d in data]
        ax.plot(dates, cpu_down, 'r--', linewidth=1.5, label='CPU-Down (Rollback)',
                alpha=0.8)
        ax.plot(dates, cpu_up, 'g--', linewidth=1.5, label='CPU-Up (Expansion)',
                alpha=0.8)

    ax.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.5,
               label='Baseline (mean=100)')
    ax.set_ylabel('Index Value (normalized, mean=100)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Climate Policy Uncertainty Indices\nStandard, Downside, and Upside Decomposition',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    format_date_axis(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def main():
    """Generate all charts."""
    print("Loading CPU index data...")
    data = load_index_data()
    print(f"  Loaded {len(data)} months of data")

    has_directional = data[0].get('has_directional', False)
    print(f"  Directional indices available: {has_directional}")

    print("\nGenerating annotated chart...")
    path1 = create_cpu_chart(data)
    print(f"  Saved to: {path1}")

    print("\nGenerating simple chart...")
    path2 = create_simple_chart(data)
    print(f"  Saved to: {path2}")

    if has_directional:
        print("\nGenerating directional decomposition chart...")
        path3 = create_directional_chart(data)
        print(f"  Saved to: {path3}")

        print("\nGenerating combined indices chart...")
        path4 = create_combined_chart(data)
        print(f"  Saved to: {path4}")
    else:
        print("\nSkipping directional charts (no directional data in CSV)")

    print("\nDone!")


if __name__ == "__main__":
    main()
