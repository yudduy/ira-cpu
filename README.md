# Climate Policy Uncertainty (CPU) Index

A research tool for measuring uncertainty about U.S. climate policy using newspaper article analysis, following the methodology established by [Baker, Bloom & Davis (2016)](https://doi.org/10.1093/qje/qjw024).

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start PostgreSQL database
docker-compose up -d

# 3. Configure API keys
cp .env.example .env
# Edit .env with your LexisNexis and OpenAI credentials

# 4. Run the interactive tool
PYTHONPATH=src python src/cpu_index/cli/run.py
```

## Research Motivation

Climate policy uncertainty affects investment decisions in clean energy, influences corporate strategy, and impacts the pace of energy transition. This index provides a systematic measure of:

1. **Overall climate policy uncertainty** (CPU)
2. **Implementation uncertainty** (CPU_impl) - uncertainty about *how* policy will be executed
3. **Reversal uncertainty** (CPU_reversal) - uncertainty about *whether* policy will survive

---

## Methodology

### Foundation: Baker, Bloom & Davis (2016)

This index adapts the [Economic Policy Uncertainty (EPU) Index](https://www.policyuncertainty.com/) methodology to climate policy:

| Component | EPU (BBD 2016) | CPU (This Index) |
|-----------|----------------|------------------|
| Domain | Economic policy | Climate/energy policy |
| Sources | 10 major U.S. newspapers | 8 BBD-approved newspapers |
| Classification | Triple-keyword matching | Triple-keyword + directional |
| Normalization | BBD-style (mean=100) | BBD-style (mean=100) |

**Core Formula:**
```
CPU = (climate AND policy AND uncertainty articles) / (climate AND policy articles)
```

### Triple-Keyword Classification

An article counts toward the CPU **numerator** only if it contains terms from ALL THREE categories:

| Category | Purpose | Examples |
|----------|---------|----------|
| **Climate Terms** | Identifies climate/energy topic | climate, renewable, carbon, EV, solar, wind |
| **Policy Terms** | Identifies government action | policy, regulation, legislation, IRA, Congress |
| **Uncertainty Terms** | Identifies uncertain language | uncertain, unclear, risk, doubt, unpredictable |

The **denominator** requires only climate AND policy terms (total climate policy coverage).

### Addressing Steve's Critique: Direction ≠ Uncertainty

**The Problem:**
> "You can imagine that it might be very certain that Trump is going to roll back the legislation. So 'rollback' or 'terminate' or 'rescind'... could be associated with there's a very certain rollback that's going to happen."

**The Solution:**
Direction terms (rollback, expand, repeal, etc.) **alone do NOT indicate uncertainty**. They must be combined WITH uncertainty terms:

```python
# WRONG: Direction term alone
"Trump will rollback IRA"  # This is CERTAIN, not uncertain

# CORRECT: Direction + Uncertainty
"Uncertainty about whether Trump will rollback IRA"  # This IS uncertainty
```

**Implementation:**
```python
is_cpu_reversal = has_uncertainty_term AND has_reversal_terms
is_cpu_impl = has_uncertainty_term AND has_implementation_terms
```

### Directional Decomposition

Following [Segal, Shaliastovich & Yaron (2015)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3675787) on "Good and Bad Uncertainty," we decompose CPU into directional components:

| Index | Captures | Keywords |
|-------|----------|----------|
| **CPU_impl** | Implementation uncertainty | delay, guidance, rulemaking, timeline, approval |
| **CPU_reversal** | Reversal/downside risk | rollback, repeal, rescind, terminate, overturn |
| **CPU_upside** | Expansion uncertainty | expand, strengthen, accelerate, invest |

**Interpretation:**
- **2021-2022**: Implementation-dominated ("How will the IRA be executed?")
- **2024-2025**: Reversal-dominated ("Will the IRA survive?")

**Direction Balance Metric:**
```
Direction = (impl_count - reversal_count) / (impl_count + reversal_count)
```
- Direction = +1: All implementation uncertainty
- Direction = -1: All reversal uncertainty
- Direction = 0: Balanced

### BBD-Style Normalization

Following BBD (2016), the index is normalized in four steps:

1. **Scale by volume**: `ratio = numerator / denominator`
2. **Standardize**: Convert each outlet to unit standard deviation
3. **Average**: Combine across 8 newspapers
4. **Normalize**: Scale so base period mean = 100

**Base Period**: Configurable (default: full sample)

### Policy Regime Salience

In addition to uncertainty indices, we track explicit mentions of major legislation:

| Index | Tracks | Note |
|-------|--------|------|
| **IRA Salience** | Inflation Reduction Act mentions | No uncertainty required |
| **OBBBA Salience** | One Big Beautiful Bill Act mentions | No uncertainty required |

These capture policy attention regardless of uncertainty.

---

## Validation Framework

### LLM Validation (GPT-5-nano)

Keyword classification is validated against LLM judgment:

```
1. Sample 1,000 articles randomly
2. Classify with GPT-5-nano
3. Compare to keyword classification
4. Target: ≥85% agreement
5. If <85%, expand sample and repeat
```

**Classification Schema:**
```json
{
  "is_climate_policy": true/false,
  "uncertainty_type": "implementation" | "reversal" | "none",
  "certainty_level": 1-5,
  "reasoning": "Brief explanation"
}
```

### Ablation Testing

Structured robustness checks in three phases:

**Phase 1: Required (Publication-Ready)**
- Keyword dropping sensitivity (each keyword individually)
- Outlet robustness (drop each newspaper)
- LLM validation comparison
- **Uncertainty requirement test** (validates Steve's fix)

**Phase 2: Strongly Recommended**
- Base period sensitivity
- LLM confidence thresholds
- Placebo tests (Trade Policy, Monetary Policy)
- Decomposition validation (CPU_impl vs CPU_reversal correlation)

**Phase 3: Optional**
- Keyword set exclusion tests

### Pre-Registered Event Validation

The index should respond appropriately to known policy events:

| Event | Date | Expected CPU | Expected Reversal |
|-------|------|--------------|-------------------|
| Biden inauguration | 2021-01 | Decrease | Decrease |
| IRA signed | 2022-08 | Decrease | Decrease |
| Treasury guidance delays | 2023-01 | Increase | Stable |
| Election year begins | 2024-01 | Increase | Increase |
| Trump wins election | 2024-11 | Spike | Spike |
| Trump inauguration | 2025-01 | Spike | Spike |
| OBBBA introduced | 2025-02 | Spike | Spike |

**Success criterion**: ≥75% of events show expected direction

---

## Project Structure

```
src/cpu_index/
├── config.py              # Keywords, dates, sources, settings
├── db_postgres.py         # PostgreSQL database operations
├── collection/            # LexisNexis API client & data collection
├── classification/        # Keyword-based + LLM article classification
├── analysis/              # Index calculation, normalization, correlation
├── output/                # CSV exports, visualizations, reports
└── cli/run.py             # Interactive menu (main entry point)

scripts/                   # Standalone scripts (no DB required)
├── collect_cpu_data.py    # Direct API collection to CSV
├── run_cpu_vc_analysis.py # CPU-VC correlation analysis
├── run_sector_cpu_analysis.py  # Sector-specific analysis
├── generate_sector_figures.py  # Sector visualizations
├── compare_ira_vs_full.py # IRA era vs full sample comparison
└── generate_publication_figures.py  # Publication figures

exports/deliverables/      # Paper-ready outputs (figures, data, LaTeX)
```

---

## Current Results & Deliverables

See `exports/deliverables/` for paper-ready outputs:
- **LaTeX synopsis**: `sector_analysis_synopsis.tex`
- **Figures** (300 DPI PNGs): heatmap, timeseries, decomposition, mechanism diagrams
- **Data CSVs**: sector rankings, decomposition, IRA stratification, robustness checks

Key findings from sector-specific CPU-VC correlation analysis:
1. **Industrial** is the only sector dominated by implementation uncertainty
2. **Energy, Built Env, Food, Climate Mgmt** show CPU→VC suppression ("dark spots")
3. **IRA era (2021+)** correlations are 2-3x stronger than full sample (structural break)
4. **High-IRA companies** more sensitive to CPU in IRA era

---

## Keyword Categories

Edit in `config.py`:

| Category | Purpose | Count |
|----------|---------|-------|
| `CLIMATE_TERMS` | Climate/energy topics | 16 |
| `POLICY_TERMS` | Policy/government | 23 |
| `UNCERTAINTY_TERMS` | Direction-neutral uncertainty | 9 |
| `IMPLEMENTATION_TERMS` | Implementation delays/guidance | 35 |
| `REVERSAL_TERMS` | Rollback/repeal actions | 12 |
| `DOWNSIDE_TERMS` | Broader downside risk | 51 |
| `UPSIDE_TERMS` | Expansion/strengthening | 40 |
| `REGIME_IRA_TERMS` | IRA-specific mentions | 7 |
| `REGIME_OBBBA_TERMS` | OBBBA-specific mentions | 4 |

---

## API Quota

Stanford shares 24,000 searches / 1,200,000 documents per year via LexisNexis Academic.

**Count-based approach** (current):
- 2 API calls per month (denominator count + numerator count)
- 60 months × 2 = 120 API calls total
- ~0.5% of annual quota

**Article-based approach** (alternative):
- Fetches full article metadata
- Higher quota usage but enables local re-classification
- Use for LLM validation samples

---

## References

### Primary Methodology
- Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring Economic Policy Uncertainty. *The Quarterly Journal of Economics*, 131(4), 1593–1636. https://doi.org/10.1093/qje/qjw024

### Asymmetric Uncertainty
- Segal, G., Shaliastovich, I., & Yaron, A. (2015). Good and Bad Uncertainty: Macroeconomic and Financial Market Implications. *Journal of Financial Economics*, 117(2), 369–397. https://doi.org/10.1016/j.jfineco.2015.05.004

### Climate Policy Uncertainty
- Gavriilidis, K. (2021). Measuring Climate Policy Uncertainty. Working Paper. https://ssrn.com/abstract=3847388

### Economic Policy Uncertainty Data
- https://www.policyuncertainty.com/

---

## License

Research use only. Contact maintainers for commercial licensing.
