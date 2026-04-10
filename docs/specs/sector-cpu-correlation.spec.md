# Specification: Sector-Specific CPU-VC Correlation Analysis

> Use `/execute docs/specs/sector-cpu-correlation.spec.md` to implement.

## Goal

Identify which climate tech sectors/industries are most sensitive to Climate Policy Uncertainty (CPU), measuring the "dark spots" where VC investment activity is most affected by policy uncertainty—either suppressed (negative correlation) or stimulated (positive correlation).

## Background

**RA1 Context**: The CPU index measures market perceptions of climate policy uncertainty from news coverage. This analysis extends the existing aggregate CPU-VC correlation (Noailly et al. 2022 methodology) to sector-level disaggregation.

**Data Assets**:
- `cpu_index.csv`: Monthly CPU indices (2008-2025) with CPU, CPU_impl, CPU_reversal
- `judged_results.csv`: 16,562 climate tech companies classified into 7 categories and 110 subtopics
- `IRA_Index`: Pre-computed policy exposure scores (1-7 scale)

**Key Finding from EDA**:
- Energy sector has highest IRA exposure (mean 6.37) and funding ($111B)
- Alternative Proteins has lowest IRA exposure (mean 3.06)
- Energy Storage, Batteries, Electric Autos have highest IRA sensitivity (6.5-6.8)

## Requirements

### REQ-1: Sector-Level VC Aggregation
Add function to aggregate VC metrics by (month, sector) with full statistics.
- **Acceptance**: `aggregate_by_category_monthly()` returns DataFrame with (month, category) index and columns: deal_count, total_amount, median_amount, seed_count, early_count, late_count, size_coverage_pct
- **Location**: `src/cpu_index/analysis/vc_aggregator.py`

### REQ-2: Subtopic-Level VC Aggregation
Add function to aggregate at finer granularity for drill-down analysis.
- **Acceptance**: `aggregate_by_subtopic_monthly()` returns DataFrame with (month, subtopic) index and same columns as REQ-1
- **Location**: `src/cpu_index/analysis/vc_aggregator.py`

### REQ-3: Sector-CPU Correlation Analysis
Add function to compute cross-correlation between CPU and sector-specific VC activity.
- **Acceptance**: `analyze_sector_cpu_correlation()` returns dict with correlation at lags -12 to +12, optimal lag, and interpretation per sector
- **Location**: `src/cpu_index/analysis/sector_analysis.py` (new module)

### REQ-4: Batch Sector Analysis
Add function to run correlation analysis across all sectors and rank by sensitivity.
- **Acceptance**: `analyze_all_sectors()` returns DataFrame ranking sectors by |correlation| at optimal lag, with columns: sector, optimal_lag, correlation, abs_correlation, direction, cpu_type
- **Location**: `src/cpu_index/analysis/sector_analysis.py`

### REQ-5: IRA Exposure Stratification
Add function to compare CPU sensitivity between high-IRA and low-IRA sectors.
- **Acceptance**: `stratify_by_ira_exposure()` returns dict with high_ira (6-7), low_ira (1-3) group statistics and statistical test for difference
- **Location**: `src/cpu_index/analysis/sector_analysis.py`

### REQ-6: CPU Decomposition Analysis
Analyze sectors' differential response to CPU_impl vs CPU_reversal.
- **Acceptance**: `analyze_cpu_decomposition()` returns dict with impl_correlation, reversal_correlation, asymmetry_ratio per sector
- **Location**: `src/cpu_index/analysis/sector_analysis.py`

### REQ-7: Sector Heatmap Visualization
Create correlation heatmap showing sectors × CPU types × lags.
- **Acceptance**: `plot_sector_correlation_heatmap()` outputs PNG with rows=sectors, columns=lags, color=correlation strength
- **Location**: `src/cpu_index/analysis/sector_visualizations.py` (new module)

### REQ-8: Sector Time Series Overlay
Create dual-axis plots showing CPU vs sector VC over time.
- **Acceptance**: `plot_sector_timeseries()` outputs PNG with subplots for top N most sensitive sectors
- **Location**: `src/cpu_index/analysis/sector_visualizations.py`

### REQ-9: Sensitivity Ranking Bar Chart
Create ranked bar chart of sector CPU sensitivity.
- **Acceptance**: `plot_sensitivity_ranking()` outputs PNG with sectors sorted by |correlation|, colored by direction
- **Location**: `src/cpu_index/analysis/sector_visualizations.py`

### REQ-10: Standalone Analysis Script
Create script that runs full sector analysis pipeline.
- **Acceptance**: `scripts/run_sector_cpu_analysis.py` produces CSVs (sector_correlations.csv, sector_rankings.csv) and PNGs
- **Location**: `scripts/run_sector_cpu_analysis.py`

### REQ-11: Robustness Analysis (Classifier Comparison)
Add function to run analysis using alternative classifiers (ChatGPT, Gemini, DeepSeek).
- **Acceptance**: `run_classifier_robustness()` returns dict comparing results across classifiers
- **Location**: `src/cpu_index/analysis/sector_analysis.py`

### REQ-12: Full Sample Robustness
Support both IRA era (2021+) and full sample (2008+) analysis.
- **Acceptance**: Functions accept `start_date` parameter, default='2021-01-01', with option for '2008-01-01'
- **Location**: All analysis functions

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Granularity | 7 categories primary, 30+ subtopics for drill-down | Balance statistical power with granular insights |
| VC Metrics | Both deal count and amount | Count is less noisy, amount captures magnitude |
| CPU Types | Total CPU, CPU_impl, CPU_reversal | Decomposition reveals mechanism (impl vs reversal risk) |
| Time Period | IRA era primary (2021+), full sample robustness | Focus on current policy regime, robustness on historical |
| Dark Spots Definition | |correlation| regardless of sign | Identify most policy-sensitive sectors, either direction |
| Classification | judge_category primary, classifier comparison appendix | Consensus classification for main results |
| Statistical Method | Cross-correlation at lags | Standard approach, already implemented in correlation.py |
| IRA Stratification | High (6-7) vs Low (1-3), both continuous and categorical | Stratified for interpretability, continuous for regression |

## Completion Criteria

- [ ] All REQs implemented with passing tests
- [ ] Build + lint clean (`PYTHONPATH=src pytest tests/`)
- [ ] Script produces outputs: sector_correlations.csv, sector_rankings.csv, 3+ PNG figures
- [ ] README section documenting new functionality

## Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| Sector with <24 monthly observations | Skip sector, log warning, include in output with NaN |
| Subtopic with <12 observations | Aggregate to category level, note in metadata |
| No deals in a month for sector | Fill with 0 (deal_count), NaN for median |
| CPU data missing for month | Inner join drops that month, log warning |
| All classifiers disagree on category | Use judge_category (consensus), flag in robustness output |
| IRA_Index missing for company | Exclude from IRA stratification analysis |

## Technical Context

### Verified Data Availability

**judged_results.csv** (16,474 rows × 176 columns):
| Column | Completeness | Notes |
|--------|--------------|-------|
| judge_category | 100% | 8 sectors (exclude 'Others' n=85) |
| Semantic_Subtopic | 82% | 110 unique subtopics |
| IRA_Index | 100% | Integer 1-7 |
| Total Raised | 38% | Requires `pd.to_numeric(x, errors='coerce')` |
| Last Financing Date | 71% | Parseable with `pd.to_datetime()` |
| ChatGPT/Gemini/DeepSeek_Category | 100% | For robustness comparison |

**cpu_index.csv** (209 rows × 11 columns):
- Month range: 2008-01 to 2025-05 (209 months)
- All CPU columns 100% complete (cpu_index, cpu_impl, cpu_reversal)

### Sector Sample Sizes
| Sector | n | Statistical Power |
|--------|---|-------------------|
| Energy | 6,496 | Excellent |
| Industrial | 3,894 | Excellent |
| Built_Environment | 2,265 | Good |
| Food_Land_Use | 1,215 | Adequate |
| Transportation | 1,080 | Adequate |
| Climate_Mgmt | 779 | Adequate |
| Carbon | 658 | Adequate |
| Others | 85 | Exclude |

### Key Files to Modify
- `src/cpu_index/analysis/vc_aggregator.py`: Add REQ-1, REQ-2 functions
- `src/cpu_index/analysis/sector_analysis.py`: New module for REQ-3 through REQ-6, REQ-11, REQ-12
- `src/cpu_index/analysis/sector_visualizations.py`: New module for REQ-7 through REQ-9
- `scripts/run_sector_cpu_analysis.py`: New script for REQ-10

### Key Files to Read (dependencies)
- `src/cpu_index/analysis/correlation.py`: cross_correlation(), find_optimal_lag(), analyze_cpu_vc_correlation()
- `src/cpu_index/analysis/vc_loader.py`: load_vc_deals(), get_deal_summary()
- `data/judged_results.csv`: VC data with classifications
- `data/cpu_index.csv`: Monthly CPU indices

### Patterns to Follow
- **Aggregation**: Follow `aggregate_monthly()` pattern in vc_aggregator.py
- **Correlation**: Use existing `cross_correlation()` and `find_optimal_lag()` from correlation.py
- **Visualization**: Follow patterns in vc_visualizations.py (300 DPI, publication quality)
- **Script**: Follow run_cpu_vc_analysis.py pattern (load → aggregate → analyze → visualize)
- **Testing**: Add tests mirroring test_vc_aggregator.py and test_correlation.py structure

### Data Schema for judged_results.csv
Key columns:
- `judge_category`: Primary classification (Energy, Industrial, Built_Environment, Food_Land_Use, Transportation, Climate_Mgmt, Carbon, Others)
- `Semantic_Subtopic`: Fine-grained classification (Clean Power Generation, Energy Storage, etc.)
- `IRA_Index`: Policy exposure score (1-7)
- `Total Raised`: Funding amount in millions
- `Last Financing Date`: Most recent deal date
- `ChatGPT_Category`, `Gemini_Category`, `DeepSeek_Category`: Alternative classifier outputs

### CPU Index Schema
- `month`: YYYY-MM format
- `cpu_index`: Normalized CPU (mean=100)
- `cpu_impl`: Implementation uncertainty index
- `cpu_reversal`: Reversal uncertainty index

## Output Files

| File | Description |
|------|-------------|
| `outputs/sector_analysis/sector_correlations.csv` | Full correlation results by sector × CPU type × lag |
| `outputs/sector_analysis/sector_rankings.csv` | Sectors ranked by CPU sensitivity |
| `outputs/sector_analysis/ira_stratification.csv` | High-IRA vs Low-IRA comparison |
| `outputs/sector_analysis/fig_sector_heatmap.png` | Correlation heatmap |
| `outputs/sector_analysis/fig_sector_timeseries.png` | Time series overlays |
| `outputs/sector_analysis/fig_sensitivity_ranking.png` | Ranked bar chart |

## References

- Noailly, Nowzohour & van den Heuvel (2022): EnvPU methodology
- Baker, Bloom & Davis (2016): EPU index construction
- Existing: `docs/research/cpu-vc-correlation/KNOWLEDGE.md`
