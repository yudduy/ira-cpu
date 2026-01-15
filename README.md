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
