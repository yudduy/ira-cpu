# Climate Policy Uncertainty Index (US)

A newspaper-based monthly index of US climate-policy uncertainty constructed via triple-keyword classification a la Baker, Bloom & Davis (2016, [QJE](https://doi.org/10.1093/qje/qjw024)). Adapts BBD's Economic Policy Uncertainty (EPU) construction to the climate domain and decomposes the resulting series into implementation and reversal components.
Running updates at [docs/LOG.md](docs/LOG.md).

## Background and motivation

Two threads in the policy-uncertainty literature frame this work:

- **Economic Policy Uncertainty** (Baker, Bloom & Davis 2016, [QJE 131(4):1593–1636](https://doi.org/10.1093/qje/qjw024)) — counts newspaper articles containing terms from three categories ({economy} ∩ {policy} ∩ {uncertain}), normalizes per outlet, and averages across 10 US dailies. EPU rises sharply around the 2008 financial crisis, the 2011 debt-ceiling fight, and the 2020 COVID shock. The construction is now the standard text-based instrument for policy uncertainty.
- **Climate Policy Uncertainty** (Gavriilidis 2021, [SSRN 3847388](https://ssrn.com/abstract=3847388)) — applies the same triple-keyword pattern with climate-domain seed terms. Documents that climate-policy uncertainty shocks reduce CO₂ emissions and dampen energy investment.

The substantive question this index targets:

> Can we measure month-by-month US climate-policy uncertainty cleanly enough to (a) identify the policy events that drive it and (b) decompose it into *implementation uncertainty* ("when will the rules take effect?") versus *reversal uncertainty* ("will the rules be repealed?")?

The decomposition follows Segal, Shaliastovich & Yaron (2015, [JFE 117(2):369–397](https://doi.org/10.1016/j.jfineco.2015.05.004)) on "good" vs "bad" uncertainty: not all uncertainty has the same sign for downstream investment, so the headline scalar should be unbundled.

## Construction

The index counts newspaper articles satisfying three keyword conditions over 8 BBD-approved US outlets, monthly from 2008-01 to 2025-05 (n=209 months). For each outlet × month, the count is normalized by total article volume; outlet series are standardized to unit variance, averaged, then scaled to mean 100 over the base period.

```
                ┌──────────────────────────────────────┐
articles ──▶ │ outlet × month counts (LexisNexis)   │
                │   denominator: climate AND policy    │
                │   numerator:   climate AND policy    │
                │                AND uncertainty       │
                └──────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────┐
                │ ratio = numerator / denominator      │
                │ standardize per outlet (σ = 1)       │
                │ average across 8 outlets             │
                │ rescale so base-period mean = 100    │
                └──────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────────────┐
                │ CPU         (headline)               │
                │ CPU_impl    (× implementation terms) │
                │ CPU_reversal (× reversal terms)      │
                └──────────────────────────────────────┘
```

### Triple-keyword classification

An article enters the **numerator** only if it contains ≥1 term from each of three categories. The **denominator** is climate ∩ policy (total climate-policy coverage), so the ratio reflects how *uncertainty-laden* climate-policy coverage is, not how much climate coverage there is.

| Category | Role | Examples (full lists in `src/cpu_index/config.py`) |
|---|---|---|
| `CLIMATE_TERMS` (16) | climate / energy topic | climate, renewable, carbon, EV, solar, wind, grid |
| `POLICY_TERMS` (23) | government action | policy, regulation, legislation, IRA, Congress |
| `UNCERTAINTY_TERMS` (9) | direction-neutral uncertainty | uncertain, unclear, risk, doubt, unpredictable |

### Three indices

| Index | Numerator condition | Captures |
|---|---|---|
| `CPU` | climate ∩ policy ∩ uncertainty | overall climate-policy uncertainty |
| `CPU_impl` | + implementation terms (delay, guidance, rulemaking, timeline, approval) | "when will the rules take effect?" |
| `CPU_reversal` | + reversal terms (rollback, repeal, rescind, terminate, overturn) | "will the rules be repealed?" |

`CPU_impl` and `CPU_reversal` are not a partition of `CPU` — they capture overlapping but distinct uncertainty modes. The asymmetry ratio `(|CPU_impl| − |CPU_reversal|) / (|CPU_impl| + |CPU_reversal|)` summarizes which mode dominates over a window.

### BBD-style normalization

Following BBD (2016, §II) verbatim:

1. **Scale by volume**: `ratio_outlet,t = numerator_outlet,t / denominator_outlet,t`
2. **Standardize per outlet**: divide each outlet's series by its own pre-base-period standard deviation
3. **Average across outlets**: equal-weighted mean over the 8 BBD-approved US dailies
4. **Rescale**: multiply by a constant so the base-period mean = 100

Implementation: `src/cpu_index/analysis/normalizer.py`. Each step is a separate function with a unit test (`tests/analysis/test_indexer.py`).

The eight outlets (`config.BBD_OUTLETS`): Financial Times, Wall Street Journal, New York Times, Washington Post, Reuters, Bloomberg, Politico, The Economist. BBD's original 10-paper basket is reduced to the 8 with consistent LexisNexis coverage over 2008–2025 — the two dropped (USA Today, Houston Chronicle) had structural breaks in coverage that broke the per-outlet standardization step.

### Validation framework

Two layers of robustness check:

- **LLM validation** (`src/cpu_index/classification/llm_validator.py`): a GPT-5-nano judge classifies a random sample of articles as climate-policy / uncertainty-type / certainty-level. Target ≥85% agreement with the keyword classifier; sample expands adaptively until either threshold is hit or 10k articles are exhausted.
- **Ablation suite** (`src/cpu_index/analysis/ablation_runner.py`): drop each keyword individually, drop each outlet individually, vary the base period, and re-run the full pipeline. Phase 1 (publication-required) ablations include the keyword-drop, outlet-drop, and uncertainty-requirement tests; Phase 2 adds base-period sensitivity, LLM-confidence thresholds, and placebo indices on Trade Policy and Monetary Policy domains.

### Pre-registered event validation

The index is validated against seven dated US climate-policy events with pre-specified expected directions (see [docs/LOG.md §3.2](docs/LOG.md)). Success criterion: ≥75% of events show the expected sign change in the month or the month following the event.

The empirical record — the index time series, the impl/reversal decomposition, the IRA-era structural break, the sector-VC correlations, and the methodological audit — is in [docs/LOG.md](docs/LOG.md).

## Setup

```bash
# 1. Clone and create a venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start PostgreSQL (only needed for full collection / LLM validation)
docker-compose up -d

# 3. Configure API keys
cp .env.example .env
# Edit .env: LEXISNEXIS_CLIENT_ID / SECRET, OPENAI_API_KEY for LLM validation
```

LexisNexis requires institutional access (this work used Stanford's WSAPI). Stored counts under `data/cpu_counts.csv` and the computed index under `data/cpu_index.csv` allow re-running the analysis pipeline without API access.

## Running

```bash
# Tests
PYTHONPATH=src pytest tests/

# Interactive CLI (collection, indexing, validation, ablations, exports)
PYTHONPATH=src python src/cpu_index/cli/run.py

# Standalone scripts (no database required)
PYTHONPATH=src python scripts/collect_cpu_data.py            # API → CSV
PYTHONPATH=src python scripts/run_cpu_vc_analysis.py         # CPU vs VC correlations
PYTHONPATH=src python scripts/run_sector_cpu_analysis.py     # sector-stratified analysis
PYTHONPATH=src python scripts/compare_ira_vs_full.py         # IRA era vs full sample
PYTHONPATH=src python scripts/generate_publication_figures.py
```

## Layout

```
src/cpu_index/
  config.py                     # keyword categories, outlets, dates
  db_postgres.py                # connection pooling
  collection/
    api.py                      # LexisNexis WSAPI client (OAuth)
    collector.py                # full article collection
    count_collector.py          # count-only (~2 API calls / month)
    deduplicator.py
  classification/
    local_classifier.py         # keyword-based, ablation-aware
    llm_validator.py            # GPT-5-nano adjudication, adaptive sampling
  analysis/
    indexer.py                  # 6 index variants (CPU, impl, reversal, …)
    normalizer.py               # 4-step BBD normalization
    ablation_config.py          # keyword-drop, outlet-drop, base-period defs
    ablation_runner.py          # ablation execution
    correlation.py              # cross-correlation, lead-lag, stationarity
    sector_analysis.py          # sector-stratified CPU–VC analysis
    vc_loader.py / vc_aggregator.py / vc_visualizations.py
  output/
    exports.py                  # CSVs (monthly, decomposition, robustness)
    visualizations.py           # 8 publication PNGs at 300 DPI
    report_generator.py
  cli/run.py                    # 9-option interactive menu

tests/
  analysis/                     # indexer, normalizer, correlation parity tests
  classification/               # keyword + LLM classifier tests
  collection/                   # API client + count-collector tests
  output/                       # export + visualization tests

scripts/                        # standalone, no DB required
data/                           # cpu_counts.csv, cpu_index.csv, judged_results.csv
outputs/sector_analysis*/       # sector ranking, decomposition, IRA stratification
exports/deliverables/           # paper-ready figures, tables, LaTeX synopsis
docs/
  LOG.md                        # genealogy: thesis, construction, empirical record, audit, open
  research/                     # background research notes
  specs/                        # spec docs for major features
```

## References

The work in this repo builds directly on:

- Baker, S. R., Bloom, N., & Davis, S. J. (2016). *Measuring Economic Policy Uncertainty.* The Quarterly Journal of Economics, 131(4), 1593–1636. https://doi.org/10.1093/qje/qjw024
- Gavriilidis, K. (2021). *Measuring Climate Policy Uncertainty.* Working Paper. https://ssrn.com/abstract=3847388
- Segal, G., Shaliastovich, I., & Yaron, A. (2015). *Good and Bad Uncertainty: Macroeconomic and Financial Market Implications.* Journal of Financial Economics, 117(2), 369–397. https://doi.org/10.1016/j.jfineco.2015.05.004

Adjacent / contextual:

- Noailly, J., Nowzohour, L., & van den Heuvel, M. (2022). *Does Environmental Policy Uncertainty Hinder Investments Towards a Low-Carbon Economy?* NBER Working Paper 30361. https://www.nber.org/papers/w30361
- Fuchs, S., Stroebel, J., & Terstegge, J. (2024). *Carbon VIX: Carbon Price Uncertainty and Decarbonization Investments.* NBER Working Paper 32937. https://www.nber.org/papers/w32937
- van den Heuvel, M., & Popp, D. (2022). *The Role of Venture Capital and Governments in Clean Energy.* CEPR VoxEU. https://cepr.org/voxeu/columns/role-venture-capital-and-governments-clean-energy
- Economic Policy Uncertainty data portal: https://www.policyuncertainty.com/

## License

Research use only. Contact maintainers for commercial licensing.
