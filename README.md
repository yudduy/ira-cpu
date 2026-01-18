# Climate Policy Credibility & Uncertainty Measure (CPU)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   ```bash
   cp .env.example .env
   # Edit .env with your LexisNexis and OpenAI credentials
   ```

3. **Run the tool:**
   ```bash
   python run.py
   ```

4. **Follow the interactive menu:**
   - Option 1: Check status
   - Option 2: Estimate API usage
   - Option 3: Collect data
   - Option 5: Build index
   - Option 6: Export to CSV

## Files

| File | Purpose |
|------|---------|
| `config.py` | All settings (keywords, dates, sources) |
| `run.py` | Interactive menu (main entry point) |
| `db.py` | SQLite database operations |
| `api.py` | LexisNexis API client |
| `collector.py` | Data collection pipeline |
| `classifier.py` | LLM validation (GPT-5 Nano) |
| `indexer.py` | CPU index calculation |

## API Quota

Stanford shares 24,000 searches / 1,200,000 documents per year.
Always run "Estimate API usage" before collecting data.

## Modifying Keywords

Edit the keyword lists in `config.py`:
- `CLIMATE_TERMS`: Climate/energy topics
- `POLICY_TERMS`: Policy/government terms
- `UNCERTAINTY_TERMS`: Uncertainty language

## Methodology

### Text Source: Headlines and Leads

This index uses article **headlines and lead paragraphs** (snippets) rather than full article text, following the methodology established by Baker, Bloom & Davis (2016) and subsequent policy uncertainty research.

**Why metadata-only?**
- BBD's foundational Economic Policy Uncertainty Index uses headlines/leads exclusively
- Headlines represent editorial judgment about article salience—a stronger signal than keyword presence in body text
- Full text often contains tangential mentions that add noise rather than signal
- 10+ years of follow-up research validates this approach for policy uncertainty measurement

### Normalization

The index follows BBD-style normalization:
1. Scale article counts by outlet volume (numerator/denominator ratio)
2. Standardize each outlet to unit standard deviation
3. Average across outlets per month
4. Normalize to base period mean = 100

### References

- Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring Economic Policy Uncertainty. *The Quarterly Journal of Economics*, 131(4), 1593–1636. https://doi.org/10.1093/qje/qjw024
- Gavriilidis, K. (2021). Measuring Climate Policy Uncertainty. Working Paper. https://ssrn.com/abstract=3847388
- Economic Policy Uncertainty Index: https://www.policyuncertainty.com/
