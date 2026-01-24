# Sector Analysis Deliverables

This folder contains the deliverables for the sector-specific CPU-VC correlation analysis.

## Contents

### LaTeX Synopsis
- `sector_analysis_synopsis.tex` - Full synopsis document ready for compilation

### Figures (PNG, 300 DPI)
| File | Description |
|------|-------------|
| `fig_summary_dashboard.png` | Key findings at a glance |
| `fig_sensitivity_leadlag.png` | Magnitude and timing of correlations |
| `fig_decomposition_narrative.png` | Implementation vs Reversal story |
| `fig_timeline_events.png` | Time series with IRA/OBBBA markers |
| `fig_mechanism_diagram.png` | Conceptual framework |
| `fig_ira_vs_full_comparison.png` | Structural break evidence |
| `fig_sector_heatmap.png` | Correlation heatmap by lag |

### Data Files (CSV)
| File | Description |
|------|-------------|
| `sector_rankings.csv` | Sectors ranked by CPU sensitivity (IRA era) |
| `decomposition.csv` | CPU_impl vs CPU_reversal by sector |
| `ira_stratification.csv` | High vs Low IRA exposure comparison |
| `comparison_ira_vs_full.csv` | IRA era vs full sample comparison |
| `classifier_robustness.csv` | Robustness across classifiers |
| `sector_rankings_full_sample.csv` | Full sample (2008-2025) rankings |
| `decomposition_full_sample.csv` | Full sample decomposition |

## Compiling the LaTeX Document

```bash
cd exports/deliverables
pdflatex sector_analysis_synopsis.tex
pdflatex sector_analysis_synopsis.tex  # Run twice for TOC
```

## Key Findings Summary

1. **Industrial is Unique**: Only sector dominated by implementation uncertainty
2. **"Dark Spots"**: Energy, Built Env, Food, Climate Mgmt show CPU→VC suppression
3. **Structural Break**: IRA era shows 2-3x stronger correlations than full sample
4. **High-IRA More Sensitive**: Policy-dependent companies affected more in IRA era

## Analysis Period

- **IRA Era**: 2021-01 to 2025-05 (53 months)
- **Full Sample**: 2008-01 to 2025-05 (209 months)
- **VC Dataset**: 16,474 companies across 7 sectors
