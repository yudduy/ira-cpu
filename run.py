#!/usr/bin/env python3
"""
CPU Index Builder - Interactive Runner

Just run: python run.py

A menu-driven interface for non-technical researchers.
"""

import sys
import os

# TODO: Replace sys.path manipulation with proper package structure
# Current approach is fragile and can cause import conflicts.
# Better approach: Create pyproject.toml and install with `pip install -e .`
# Then imports become: from cpu_index import db, config, etc.
# See: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
import config
import collector
import classifier
import indexer


def main():
    """Main entry point."""
    # Initialize database
    db.init_db()

    print_header()

    while True:
        print_menu()
        choice = input("\nEnter choice (1-7): ").strip()

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
            export_csv()
        elif choice == "7":
            print("\nGoodbye!\n")
            break
        else:
            print("\n[!] Invalid choice. Please enter 1-7.\n")

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
|  6. Export index to CSV           |
|  7. Exit                          |
+-----------------------------------+
    """)


def show_status():
    """Show current collection status."""
    print("\n" + "="*50)
    print("STATUS & PROGRESS")
    print("="*50)

    status = collector.get_collection_status()

    print(f"\nDate Range: {status['date_range']}")
    print(f"Total Months: {status['total_months']}")
    print(f"Completed: {status['completed_months']} ({status['percent_complete']:.0%})")
    print(f"Remaining: {status['incomplete_months']}")

    if status['next_month']:
        print(f"\nNext month to collect: {status['next_month']}")
    else:
        print("\n[OK] All months collected!")

    # Show index status
    idx_status = indexer.get_index_summary()
    print(f"\nIndex Status: {idx_status['status']}")
    if idx_status['status'] == 'ready':
        print(f"  Period: {idx_status['period']}")
        print(f"  Mean: {idx_status['mean']}, Std: {idx_status['std']}")

    # Show validation status
    val_status = classifier.get_validation_status()
    print(f"\nLLM Validation:")
    print(f"  AI SDK Available: {'Yes' if val_status['ai_sdk_available'] else 'No'}")
    print(f"  OpenAI Key Set: {'Yes' if val_status['openai_key_set'] else 'No'}")


def estimate_usage():
    """Estimate API usage before collection."""
    print("\n" + "="*50)
    print("API USAGE ESTIMATE")
    print("="*50)

    est = collector.estimate_api_usage()

    print(f"\nMonths to collect: {est['months']}")
    print(f"Estimated API searches: {est['searches']}")
    print(f"Percent of annual quota: {est['percent_quota']:.2%}")
    print(f"  (Stanford limit: 24,000 searches/year shared)")

    if est['months'] > 0:
        print(f"\nFirst 5 months to collect:")
        for m in est['incomplete_months'][:5]:
            print(f"  - {m}")
        if est['months'] > 5:
            print(f"  ... and {est['months'] - 5} more")


def collect_data():
    """Run data collection with confirmation."""
    print("\n" + "="*50)
    print("DATA COLLECTION")
    print("="*50)

    est = collector.estimate_api_usage()

    if est['months'] == 0:
        print("\n[OK] All months already collected!")
        return

    print(f"\nMonths to collect: {est['months']}")
    print(f"Estimated API searches: {est['searches']}")
    print(f"Percent of annual quota: {est['percent_quota']:.2%}")

    print("\n[!] This will use your LexisNexis API quota.")
    print("    Make sure you have tested your queries in WSAPI first.")

    confirm = input("\nProceed with collection? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("\n[X] Collection cancelled.")
        return

    # Ask about dry run
    dry_run_input = input("Do a dry run first (no API calls)? (yes/no): ").strip().lower()
    dry_run = dry_run_input == "yes"

    if dry_run:
        print("\n[DRY RUN] Simulating collection without API calls...")
    else:
        print("\n[LIVE] Starting collection...")

    def progress_callback(current, total, month):
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {current}/{total} - {month}", end="", flush=True)

    result = collector.collect_all(dry_run=dry_run, progress_callback=progress_callback)

    print("\n")
    print(f"\n[OK] Collection complete!")
    print(f"  Processed: {result['months_processed']}")
    print(f"  Completed: {result['months_complete']}")
    print(f"  Skipped: {result['months_skipped']}")
    print(f"  Errors: {result['months_error']}")

    if result['errors']:
        print("\nErrors encountered:")
        for err in result['errors'][:5]:
            print(f"  - {err['month']}: {err.get('error', 'Unknown error')}")


def validate_sample():
    """Run LLM validation on sample."""
    print("\n" + "="*50)
    print("LLM KEYWORD VALIDATION")
    print("="*50)

    val_status = classifier.get_validation_status()

    if not val_status['ai_sdk_available']:
        print("\n[!] AI SDK not installed.")
        print("    Run: pip install ai-sdk-python")
        return

    if not val_status['openai_key_set']:
        print("\n[!] OpenAI API key not set.")
        print("    Add OPENAI_API_KEY to your .env file")
        return

    # Show cost estimate
    sample_size = config.LLM_SAMPLE_SIZE
    cost_est = classifier.estimate_classification_cost(sample_size)

    print(f"\nSample size: {sample_size} articles")
    print(f"Estimated cost: ${cost_est['estimated_cost_usd']:.4f}")
    print(f"  (GPT-5 Nano: $0.05/1M input, $0.40/1M output)")

    confirm = input("\nRun validation? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("\n[X] Validation cancelled.")
        return

    print("\n[!] LLM validation not fully implemented yet.")
    print("    This requires articles in the database.")
    print("    Run data collection first, then try again.")


def build_index():
    """Build CPU index from collected data."""
    print("\n" + "="*50)
    print("BUILD CPU INDEX")
    print("="*50)

    status = collector.get_collection_status()

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
    print(f"  Mean (normalized): {meta['mean_normalized']:.1f}")
    print(f"  Std Dev (raw): {meta['std_raw_ratio']:.4f}")

    # Run event validation
    print("\nValidating against known events...")
    validation = indexer.validate_against_events()

    print(f"\nEvent Validation: {validation.get('summary', 'N/A')}")
    for event in validation.get('events', []):
        event_result = event.get('result', 'N/A')
        status_icon = "[OK]" if event_result == 'PASS' else "[--]"
        print(f"  {status_icon} {event['event']} ({event['date']}): {event_result}")


def export_csv():
    """Export index to CSV file."""
    print("\n" + "="*50)
    print("EXPORT TO CSV")
    print("="*50)

    idx_status = indexer.get_index_summary()

    if idx_status['status'] != 'ready':
        print("\n[!] No index to export.")
        print("    Build the index first (option 5).")
        return

    output_path = os.path.join(config.EXPORT_DIR, "cpu_index.csv")

    print(f"\nExport destination: {output_path}")

    confirm = input("Proceed with export? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("\n[X] Export cancelled.")
        return

    try:
        num_rows = db.export_to_csv(output_path)
        print(f"\n[OK] Exported {num_rows} months to:")
        print(f"    {output_path}")
    except Exception as e:
        print(f"\n[!] Export failed: {e}")


if __name__ == "__main__":
    main()
