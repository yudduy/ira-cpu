# Climate Policy Uncertainty Index: Methodology Memo

> **Results-First Structure**: Key findings upfront, methodology details follow.
>
> Generated: {{GENERATION_DATE}}

---

## 1. Key Findings

### Headline Result

The Climate Policy Uncertainty (CPU) Index captures meaningful variation in uncertainty about U.S. climate policy from January 2021 to present. The index shows:

- **Baseline level**: Mean = 100 (normalized to {{BASE_PERIOD}})
- **Current level**: {{CURRENT_CPU}} (as of {{CURRENT_MONTH}})
- **Range**: {{MIN_CPU}} to {{MAX_CPU}}

### Decomposition Insight

The CPU decomposes into two components that tell a compelling story:

| Period | Dominant Component | Interpretation |
|--------|-------------------|----------------|
| 2021-2022 | **Implementation** | "How will climate policy be executed?" |
| 2024-2025 | **Reversal** | "Will climate policy survive?" |

This shift from implementation-dominated to reversal-dominated uncertainty represents a fundamental change in the nature of climate policy risk.

### Headline Events

The index responds appropriately to pre-registered policy events:

1. **IRA Signing (August 2022)**: CPU {{IRA_RESPONSE}} as implementation uncertainty rose while reversal risk fell
2. **Trump Election (November 2024)**: CPU {{TRUMP_RESPONSE}} as reversal risk dominated
3. **OBBBA Introduction (February 2025)**: CPU {{OBBBA_RESPONSE}} as legislative threat materialized

![CPU Time Series with Event Annotations](outputs/figures/fig1_cpu_timeseries.png)

---

## 2. Data Sources

### Newspaper Coverage

The index draws from **8 major U.S. newspapers** following Baker, Bloom & Davis (2016):

| Outlet | LexisNexis ID | Coverage |
|--------|--------------|----------|
| New York Times | nytf | Full text |
| Washington Post | washpost | Full text |
| Wall Street Journal | wsjbk | Full text |
| USA Today | usat | Full text |
| Los Angeles Times | lat | Full text |
| Chicago Tribune | chtrib | Full text |
| Boston Globe | bglobe | Full text |
| San Francisco Chronicle | sfchron | Full text |

### Time Period

- **Start**: January 2021 (Biden administration begins)
- **End**: Present (monthly updates)
- **Observations**: {{N_MONTHS}} monthly observations

### Data Collection

Articles retrieved via LexisNexis Academic API using headline and lead paragraph (snippet) metadata. Full article text not required for classification.

---

## 3. Methodology

### Classification Approach

Following BBD (2016), we use a **triple-keyword approach** requiring articles to contain:

1. **Climate terms**: climate, emissions, carbon, renewable, clean energy, ...
2. **Policy terms**: policy, regulation, legislation, law, government, ...
3. **Uncertainty terms**: uncertain, risk, unclear, unpredictable, ...

*Full keyword lists available in Appendix A.*

### Directional Decomposition

Beyond standard CPU, we classify articles by uncertainty direction:

- **CPU_impl**: Implementation uncertainty (delay, guidance, rulemaking, ...)
- **CPU_reversal**: Reversal risk (rollback, repeal, undo, overturn, ...)
- **CPU_upside**: Upside uncertainty (expand, strengthen, accelerate, ...)

**Key methodological decision**: Direction terms alone do NOT indicate uncertainty. An article about "policy rollback" only counts toward CPU_reversal if it ALSO contains an uncertainty term. This addresses the critique that directional terms could capture actual policy changes rather than uncertainty about changes.

### Normalization

BBD-style normalization ensures comparability across outlets and time:

1. **Scale by volume**: Divide numerator by total articles per outlet-month
2. **Standardize**: Convert to unit standard deviation per outlet
3. **Average**: Compute mean across outlets
4. **Normalize**: Scale to base period mean = 100

Base period: {{BASE_PERIOD}}

---

## 4. Validation

### Event Validation

Pre-registered events with expected CPU response:

| Event | Date | Expected | Actual | ✓ |
|-------|------|----------|--------|---|
| Biden inauguration | 2021-01 | Decrease | {{EVENT_2021_01}} | {{CHECK_2021_01}} |
| IRA signed | 2022-08 | Decrease | {{EVENT_2022_08}} | {{CHECK_2022_08}} |
| Treasury guidance delays | 2023-01 | Increase | {{EVENT_2023_01}} | {{CHECK_2023_01}} |
| Election year begins | 2024-01 | Increase | {{EVENT_2024_01}} | {{CHECK_2024_01}} |
| Trump wins | 2024-11 | Spike | {{EVENT_2024_11}} | {{CHECK_2024_11}} |
| Trump inauguration | 2025-01 | Spike | {{EVENT_2025_01}} | {{CHECK_2025_01}} |
| OBBBA introduced | 2025-02 | Spike | {{EVENT_2025_02}} | {{CHECK_2025_02}} |

**Accuracy**: {{EVENT_ACCURACY}}% of events show expected direction

### LLM Audit

A random sample of {{LLM_SAMPLE_SIZE}} articles was validated using GPT-5-nano:

- **Keyword vs LLM correlation**: {{LLM_CORRELATION}}
- **False positive rate**: {{FALSE_POSITIVE_RATE}}%
- **False negative rate**: {{FALSE_NEGATIVE_RATE}}%

![LLM Validation Scatter](outputs/figures/figA4_llm_scatter.png)

### Placebo Tests

The CPU index is climate-SPECIFIC, not general policy uncertainty:

| Comparison | Correlation | Interpretation |
|------------|-------------|----------------|
| CPU vs Trade Policy Uncertainty | {{TPU_CORRELATION}} | ✓ Distinct construct |
| CPU vs Monetary Policy Uncertainty | {{MPU_CORRELATION}} | ✓ Distinct construct |

Success criterion: correlation < 0.70 with placebo indices

![Placebo Comparison](outputs/figures/figA3_placebo_comparison.png)

---

## 5. Limitations

### Metadata-Only Approach

This index uses **headline and lead paragraph only**, not full article text. This may:
- Miss nuanced uncertainty signals in article body
- Over-weight sensationalized headlines
- Under-capture technical policy discussions

Validation against full-text analysis is recommended for future work.

### Outlet Selection

The 8-newspaper sample follows BBD convention but may not capture:
- Regional variation in climate policy coverage
- Newer digital-native outlets
- Specialized energy/environment publications

### Temporal Coverage

Starting from January 2021 means:
- No pre-Biden baseline for comparison
- Limited historical context for long-term trends
- Potential sensitivity to starting point choice

---

## Appendix References

- **Appendix A**: Full keyword lists (climate, policy, uncertainty, direction terms)
- **Appendix B**: Robustness visualizations (outlet heatmap, keyword sensitivity)
- **Appendix C**: Ablation study results (all 73 variants)

---

## Appendix A: Keyword Lists

### Climate Terms
{{CLIMATE_TERMS_LIST}}

### Policy Terms
{{POLICY_TERMS_LIST}}

### Uncertainty Terms
{{UNCERTAINTY_TERMS_LIST}}

### Implementation Terms
{{IMPLEMENTATION_TERMS_LIST}}

### Reversal Terms
{{REVERSAL_TERMS_LIST}}

### Upside Terms
{{UPSIDE_TERMS_LIST}}

### Regime-Specific Terms

**IRA (Inflation Reduction Act)**:
{{IRA_TERMS_LIST}}

**OBBBA (One Big Beautiful Bill Act)**:
{{OBBBA_TERMS_LIST}}

---

## References

Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic policy uncertainty. *The Quarterly Journal of Economics*, 131(4), 1593-1636.

---

*This memo was generated using the CPU Index Builder toolkit. For methodology questions, see the full documentation at [repository URL].*
