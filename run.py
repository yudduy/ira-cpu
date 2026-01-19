#!/usr/bin/env python3
"""
CPU Index Builder - Interactive Runner

Just run: python run.py

A menu-driven interface for non-technical researchers.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import collector
import count_collector
import indexer
import db_postgres
import local_classifier
from exports import export_all_csvs
from report_generator import generate_full_report


def main():
    """Main entry point."""
    db_postgres.init_db()

    print_header()

    while True:
        print_menu()
        choice = input("\nEnter choice (1-9): ").strip()

        if choice == "1":
            show_status()
        elif choice == "2":
            estimate_usage()
        elif choice == "3":
            collect_data()
        elif choice == "4":
            validate_sample()
        elif choice == "5":
            build_index()
        elif choice == "6":
            export_deliverables()
        elif choice == "7":
            generate_report()
        elif choice == "8":
            generate_visualizations()
        elif choice == "9":
            print("\nGoodbye!\n")
            break
        else:
            print("\n[!] Invalid choice. Please enter 1-9.\n")

        input("\nPress Enter to continue...")
        print("\n" + "="*60 + "\n")


def print_header():
    """Print welcome header."""
    print("""
+===========================================================+
|     Climate Policy Uncertainty (CPU) Index Builder        |
|                                                           |
|  This tool collects news articles from LexisNexis and     |
|  builds a monthly uncertainty index for climate policy.   |
+===========================================================+
    """)


def print_menu():
    """Print main menu."""
    print("""
+-----------------------------------+
|           MAIN MENU               |
+-----------------------------------+
|  1. Show status & progress        |
|  2. Estimate API usage            |
|  3. Collect data from LexisNexis  |
|  4. Validate keywords (LLM)       |
|  5. Build CPU index               |
|  6. Export CSVs                   |
|  7. Generate full report          |
|  8. Generate visualizations only  |
|  9. Exit                          |
+-----------------------------------+
    """)


def confirm_action(prompt: str) -> bool:
    """Prompt user for yes/no confirmation."""
    response = input(f"\n{prompt} (yes/no): ").strip().lower()
    if response != "yes":
        print("\n[X] Cancelled.")
        return False
    return True


def print_progress_bar(current: int, total: int, label: str = "") -> None:
    """Print a progress bar to stdout."""
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    suffix = f" - {label}" if label else ""
    print(f"\r[{bar}] {current}/{total}{suffix}", end="", flush=True)


def print_collection_sample(results: list) -> None:
    """Print a sample of successful collection results."""
    successes = [r for r in results if r.get('status') == 'complete']
    if not successes:
        return

    sample = successes[-1]
    print(f"\n  Sample ({sample['month']}):")
    print(f"    Denominator: {sample['denominator']:,}")
    print(f"    CPU numerator: {sample['numerator_cpu']:,}")
    print(f"    CPU_impl: {sample['numerator_impl']:,}")
    print(f"    CPU_reversal: {sample['numerator_reversal']:,}")


def print_collection_errors(results: list) -> None:
    """Print any collection errors."""
    errors = [r for r in results if r.get('status') == 'error']
    if not errors:
        return

    print(f"\nErrors encountered: {len(errors)}")
    for err in errors[:5]:
        print(f"  - {err['month']}: {err.get('error', 'Unknown error')}")


def show_status():
    """Show current collection status."""
    print("\n" + "="*50)
    print("STATUS & PROGRESS")
    print("="*50)

    status = count_collector.get_collection_status()

    print(f"\nDate Range: {status['date_range']}")
    print(f"Total Months: {status['total_months']}")
    print(f"Completed: {status['completed_months']} ({status['percent_complete']:.0%})")
    print(f"Remaining: {status['pending_months']}")

    if status['pending_months'] > 0:
        print(f"\n{status['pending_months']} months remaining to collect.")
    else:
        print("\n[OK] All months collected!")

    idx_status = indexer.get_index_summary()
    print(f"\nIndex Status: {idx_status['status']}")
    if idx_status['status'] == 'ready':
        print(f"  Period: {idx_status['period']}")
        print(f"  Mean: {idx_status['mean']}, Std: {idx_status['std']}")

    val_status = local_classifier.get_validation_status()
    print(f"\nLLM Validation:")
    print(f"  AI SDK Available: {'Yes' if val_status['ai_sdk_available'] else 'No'}")
    print(f"  OpenAI Key Set: {'Yes' if val_status['openai_key_set'] else 'No'}")


def estimate_usage():
    """Estimate API usage before collection."""
    print("\n" + "="*50)
    print("API USAGE ESTIMATE")
    print("="*50)

    est = count_collector.estimate_api_usage()

    print(f"\nMonths to collect: {est['months']}")
    print(f"Estimated API searches: {est['searches']} (4 per month)")


def collect_data():
    """Run count-based data collection."""
    print("\n" + "="*50)
    print("DATA COLLECTION (Count-Based)")
    print("="*50)

    status = count_collector.get_collection_status()
    pending = status["pending_months"]

    if pending == 0:
        print("\n[OK] All months already collected!")
        return

    estimate = count_collector.estimate_api_usage()

    print(f"\nMonths to collect: {pending}")
    print(f"Estimated API searches: {estimate['searches']} (4 per month)")

    print("\n[!] This will use your LexisNexis API quota.")
    print("    Collecting COUNTS for: denominator, CPU, CPU_impl, CPU_reversal")

    if not confirm_action("Proceed with collection?"):
        return

    dry_run = confirm_action("Do a dry run first (no API calls)?")

    if dry_run:
        print("\n[DRY RUN] Simulating collection...")
    else:
        print("\n[LIVE] Starting collection...")

    result = count_collector.collect_all_counts(
        dry_run=dry_run,
        save=True,
        progress_callback=print_progress_bar,
    )

    print("\n")
    print(f"\n[OK] Collection complete!")
    print(f"  Months processed: {result['months_processed']}")

    print_collection_sample(result['results'])
    print_collection_errors(result['results'])


def validate_sample():
    """Run LLM validation on sample."""
    print("\n" + "="*50)
    print("LLM KEYWORD VALIDATION")
    print("="*50)

    val_status = local_classifier.get_validation_status()

    if not val_status['ai_sdk_available']:
        print("\n[!] OpenAI SDK not installed.")
        print("    Run: pip install openai")
        return

    if not val_status['openai_key_set']:
        print("\n[!] OpenAI API key not set.")
        print("    Add OPENAI_API_KEY to your .env file")
        return

    # Check if we have articles in database
    articles = db_postgres.get_all_articles()
    if not articles:
        print("\n[!] No articles in database.")
        print("    Run data collection first (option 3).")
        return

    sample_size = min(config.LLM_SAMPLE_SIZE, len(articles))
    cost_est = local_classifier.estimate_classification_cost(sample_size)

    print(f"\nArticles available: {len(articles)}")
    print(f"Sample size: {sample_size} articles")
    print(f"Estimated cost: ${cost_est['estimated_cost_usd']:.4f}")
    print(f"  (GPT-5-nano: ~$0.05/1M input, $0.40/1M output)")

    if not confirm_action("Run validation?"):
        return

    print("\nRunning LLM validation...")

    import llm_validator

    result = llm_validator.run_validation(
        articles=articles,
        initial_sample_size=sample_size,
        accuracy_threshold=config.LLM_ACCURACY_THRESHOLD,
        progress_callback=print_progress_bar,
    )

    print("\n")
    print(f"\n[OK] Validation complete!")
    print(f"  Accuracy: {result['accuracy']:.1%}")
    print(f"  Sample size: {result['sample_size']}")
    print(f"  Expansions: {result['expansions']}")

    if result['accuracy'] >= config.LLM_ACCURACY_THRESHOLD:
        print(f"\n  ✓ Accuracy meets threshold ({config.LLM_ACCURACY_THRESHOLD:.0%})")
    else:
        print(f"\n  ✗ Accuracy below threshold ({config.LLM_ACCURACY_THRESHOLD:.0%})")
        print("    Consider reviewing keyword definitions.")


def build_index():
    """Build CPU index from collected data."""
    print("\n" + "="*50)
    print("BUILD CPU INDEX")
    print("="*50)

    status = count_collector.get_collection_status()

    if status['completed_months'] == 0:
        print("\n[!] No data collected yet.")
        print("    Run data collection first (option 3).")
        return

    print(f"\nData available: {status['completed_months']} months")

    # Ask about base period
    print("\nNormalization base period:")
    print("  The index will be normalized so the MEAN = 100 over this period.")
    print("  Leave blank to use all available data.")

    base_start = input("\n  Base period start (YYYY-MM or blank): ").strip()
    base_end = input("  Base period end (YYYY-MM or blank): ").strip()

    base_start = base_start if base_start else None
    base_end = base_end if base_end else None

    print("\nBuilding index...")

    result = indexer.build_index(base_start=base_start, base_end=base_end)

    if result['status'] != 'success':
        print(f"\n[!] Error: {result.get('message', 'Unknown error')}")
        return

    meta = result['metadata']
    print(f"\n[OK] Index built successfully!")
    print(f"  Period: {meta['period']}")
    print(f"  Months: {meta['num_months']}")
    print(f"  Mean CPU (normalized): {meta['mean_normalized_cpu']:.1f}")
    print(f"  Std CPU (normalized): {meta['std_normalized_cpu']:.1f}")
    print(f"  Direction range: {meta['direction_range']}")

    # Run event validation
    print("\nValidating against known events...")
    validation = indexer.validate_against_events()

    print(f"\nEvent Validation: {validation.get('summary', 'N/A')}")
    for event in validation.get('events', []):
        event_result = event.get('result', 'N/A')
        status_icon = "[OK]" if event_result == 'PASS' else "[--]"
        print(f"  {status_icon} {event['event']} ({event['date']}): {event_result}")


def export_deliverables():
    """Export index to CSV files."""
    print("\n" + "="*50)
    print("EXPORT CSVs")
    print("="*50)

    idx_status = indexer.get_index_summary()

    if idx_status['status'] != 'ready':
        print("\n[!] No index to export.")
        print("    Build the index first (option 5).")
        return

    output_dir = os.path.join(config.EXPORT_DIR, "csv")
    print(f"\nExport destination: {output_dir}/")
    print("  - cpu_monthly.csv")
    print("  - cpu_decomposition.csv")
    print("  - cpu_salience.csv")
    print("  - cpu_robustness.csv")

    confirm = input("\nProceed with export? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("\n[X] Export cancelled.")
        return

    index_data = indexer.get_full_index_data()
    outlet_indices = indexer.get_outlet_level_indices()

    paths = export_all_csvs(index_data, outlet_indices, output_dir)
    print(f"\n[OK] Exported {len(paths)} CSV files:")
    for p in paths:
        print(f"    {p}")


def generate_report():
    """Generate full deliverable report (CSVs + figures + memo)."""
    print("\n" + "="*50)
    print("GENERATE FULL REPORT")
    print("="*50)

    idx_status = indexer.get_index_summary()

    if idx_status['status'] != 'ready':
        print("\n[!] No index available.")
        print("    Build the index first (option 5).")
        return

    output_dir = os.path.join(config.EXPORT_DIR, "report")
    print(f"\nOutput directory: {output_dir}/")
    print("  Will create:")
    print("    - csv/ (4 CSV files)")
    print("    - figures/ (8 PNG charts)")
    print("    - cpu_methodology_memo.md")

    confirm = input("\nGenerate full report? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("\n[X] Report generation cancelled.")
        return

    print("\nGenerating report...")

    index_data = indexer.get_full_index_data()
    outlet_indices = indexer.get_outlet_level_indices()

    result = generate_full_report(
        index_data=index_data,
        outlet_indices=outlet_indices,
        output_dir=output_dir,
    )

    print(f"\n[OK] Report generated!")
    print(f"  CSVs: {len(result['csvs'])}")
    print(f"  Figures: {len(result['figures'])}")
    print(f"  Memo: {result['memo']}")


def generate_visualizations():
    """Generate visualization charts only."""
    print("\n" + "="*50)
    print("GENERATE VISUALIZATIONS")
    print("="*50)

    idx_status = indexer.get_index_summary()

    if idx_status['status'] != 'ready':
        print("\n[!] No index available.")
        print("    Build the index first (option 5).")
        return

    output_dir = os.path.join(config.EXPORT_DIR, "figures")
    print(f"\nOutput directory: {output_dir}/")
    print("  Will create 8 PNG charts at 300 DPI")

    confirm = input("\nGenerate visualizations? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("\n[X] Visualization generation cancelled.")
        return

    print("\nGenerating visualizations...")

    from visualizations import (
        plot_cpu_timeseries,
        plot_cpu_decomposition,
        plot_direction_balance,
        plot_outlet_correlation_heatmap,
        plot_article_volume,
        HEADLINE_EVENTS,
    )

    os.makedirs(output_dir, exist_ok=True)

    index_data = indexer.get_full_index_data()
    outlet_indices = indexer.get_outlet_level_indices()

    paths = []
    paths.append(plot_cpu_timeseries(index_data, HEADLINE_EVENTS, os.path.join(output_dir, "fig1_cpu_timeseries.png")))
    paths.append(plot_cpu_decomposition(index_data, os.path.join(output_dir, "fig2_cpu_decomposition.png"), events=HEADLINE_EVENTS))
    paths.append(plot_direction_balance(index_data, os.path.join(output_dir, "fig3_direction_balance.png"), events=HEADLINE_EVENTS))
    paths.append(plot_outlet_correlation_heatmap(outlet_indices, os.path.join(output_dir, "figA1_outlet_heatmap.png")))

    volume_data = {m: d.get("denominator", 0) for m, d in index_data.items()}
    paths.append(plot_article_volume(volume_data, os.path.join(output_dir, "figA5_article_volume.png")))

    print(f"\n[OK] Generated {len(paths)} visualizations:")
    for p in paths:
        print(f"    {p}")


if __name__ == "__main__":
    main()
