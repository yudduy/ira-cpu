# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research tool for measuring Climate Policy Uncertainty (CPU) using newspaper article analysis. Adapts the Baker, Bloom & Davis (2016) Economic Policy Uncertainty methodology to climate policy, collecting article counts from LexisNexis and calculating normalized uncertainty indices.

## Commands

```bash
# Setup
pip install -r requirements.txt
docker-compose up -d              # Start PostgreSQL

# Main entry point (interactive CLI)
PYTHONPATH=src python src/cpu_index/cli/run.py

# Standalone scripts (no database required)
PYTHONPATH=src python scripts/collect_cpu_data.py       # Collect CPU data to CSV
PYTHONPATH=src python scripts/run_cpu_vc_analysis.py    # CPU-VC correlation analysis
PYTHONPATH=src python scripts/generate_publication_figures.py  # Publication figures

# Tests
PYTHONPATH=src pytest tests/                                          # All tests
PYTHONPATH=src pytest tests/classification/test_local_classifier.py   # Specific file
PYTHONPATH=src pytest tests/analysis/test_indexer.py::TestBuildIndex -v  # Specific test
```

## Architecture

### Package Structure
```
src/cpu_index/
├── config.py              # Keywords, dates, LexisNexis source IDs
├── db_postgres.py         # Database operations (connection pooling)
├── collection/            # Data collection
│   ├── api.py             # LexisNexis API client (OAuth, query building)
│   ├── collector.py       # Full article collection with metadata
│   ├── count_collector.py # Count-based collection (fast, ~4 API calls/month)
│   └── deduplicator.py    # Article deduplication
├── classification/        # Article classification
│   ├── local_classifier.py  # Keyword-based (supports ablation studies)
│   └── llm_validator.py     # GPT validation with adaptive sampling
├── analysis/              # Index calculation
│   ├── indexer.py         # CPU index calculation (6 index types)
│   ├── normalizer.py      # BBD-style 4-step normalization
│   ├── ablation_config.py # Ablation study definitions
│   ├── ablation_runner.py # Ablation execution
│   ├── correlation.py     # Statistical correlation analysis
│   ├── vc_loader.py       # VC deal data loading
│   ├── vc_aggregator.py   # VC data aggregation
│   └── vc_visualizations.py
├── output/                # Reporting & visualization
│   ├── exports.py         # CSV exports (monthly, decomposition, robustness)
│   ├── visualizations.py  # 8 publication-quality PNG charts (300 DPI)
│   ├── report_generator.py
│   └── memo_template.md   # Methodology memo template
└── cli/
    └── run.py             # Interactive menu (9 options)

scripts/                   # Standalone scripts (no DB required)
├── collect_cpu_data.py    # Direct API collection to CSV
├── run_cpu_vc_analysis.py # CPU-VC correlation analysis
└── generate_publication_figures.py  # Publication figures with annotations
```

### Data Flow
1. **Collection** (`collection/`): LexisNexis API → article counts by month
2. **Classification** (`classification/`): Keyword matching + optional LLM validation
3. **Analysis** (`analysis/`): Raw ratios → BBD normalization → indices
4. **Output** (`output/`): CSVs, charts, reports

### Key Indices
- **CPU**: Climate policy uncertainty (requires uncertainty term)
- **CPU_impl**: Implementation uncertainty (uncertainty + implementation terms)
- **CPU_reversal**: Reversal uncertainty (uncertainty + reversal terms)
- **Salience indices**: IRA/OBBBA mentions (no uncertainty required)

### Classification Logic (Steve's Fix)
Direction terms alone do NOT indicate uncertainty. Requires BOTH:
```python
is_cpu_reversal = has_uncertainty_term AND has_reversal_terms
is_cpu_impl = has_uncertainty_term AND has_implementation_terms
```

## Configuration

All in `src/cpu_index/config.py`:
- `CLIMATE_TERMS`, `POLICY_TERMS`: Denominator keywords
- `UNCERTAINTY_TERMS`: Numerator keywords (direction-neutral)
- `IMPLEMENTATION_TERMS`, `REVERSAL_TERMS`: Decomposition keywords
- `START_DATE`, `END_DATE`: Analysis period (2021-01 to 2025-12)
- `SOURCE_IDS`: LexisNexis newspaper source IDs

## Environment Variables

Required in `.env`:
- `clientid`, `clientsecret`: LexisNexis API credentials
- `OPENAI_API_KEY`: For LLM validation (optional)
- `DATABASE_URL`: PostgreSQL connection (defaults to docker-compose)
