# Specification: Repository Reorganization

> Use `/execute docs/specs/repo-reorganization.spec.md` to implement.

## Goal
Reorganize the flat Python codebase into a logical folder structure grouped by pipeline stage, consolidate duplicate modules, and update .gitignore.

## Requirements

1. **[REQ-1]** Create source folder structure grouped by pipeline stage
   - Acceptance: `src/cpu_index/{collection, classification, analysis, output, cli}/` folders exist with appropriate modules

2. **[REQ-2]** Consolidate duplicate/overlapping modules
   - Merge `visualize.py` into `visualizations.py` (keep visualizations.py as primary)
   - Merge `classifier.py` into `llm_validator.py` (classifier.py has simpler LLM logic)
   - Remove `boilerplate.py` (unused LexisNexis reference file)
   - Acceptance: No duplicate functionality, all tests still pass

3. **[REQ-3]** Keep config.py and db_postgres.py at package root
   - Acceptance: `src/cpu_index/config.py` and `src/cpu_index/db_postgres.py` accessible without subpackage prefix

4. **[REQ-4]** Mirror test structure to match source structure
   - Acceptance: `tests/{collection, classification, analysis, output}/` folders with corresponding test files

5. **[REQ-5]** Add __init__.py files with minimal exports
   - Acceptance: Each package folder has __init__.py with key class/function exports

6. **[REQ-6]** Update .gitignore
   - Add `CLAUDE.md`
   - Add `ClimateTech_Deals.csv` (or `*.csv` pattern for large data files)
   - Acceptance: `git status` shows these files as untracked/ignored

7. **[REQ-7]** Update imports in all modules to use new package paths
   - Acceptance: `python -c "from src.cpu_index import config"` works; all tests pass

8. **[REQ-8]** Ensure run.py remains functional as main entry point
   - Acceptance: `python src/cpu_index/cli/run.py` launches interactive menu

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Folder grouping | By pipeline stage | Matches data flow: collect → classify → analyze → output |
| Config location | Package root | Imported by 11+ modules; avoid deep import paths |
| Ablation modules | Keep in analysis/ | Part of analysis pipeline, not separate concern |
| Test structure | Mirror source | Easier to find corresponding tests |
| __init__.py style | Minimal exports | Balance between usability and maintenance |

## Target Directory Structure

```
ira-cpu/
├── src/
│   └── cpu_index/
│       ├── __init__.py
│       ├── config.py              # Configuration (from root)
│       ├── db_postgres.py         # Database operations (from root)
│       │
│       ├── collection/            # Data collection pipeline
│       │   ├── __init__.py
│       │   ├── api.py             # LexisNexis API client
│       │   ├── collector.py       # Full article collection
│       │   ├── count_collector.py # Count-based collection
│       │   └── deduplicator.py    # Article deduplication
│       │
│       ├── classification/        # Article classification
│       │   ├── __init__.py
│       │   ├── local_classifier.py  # Keyword-based classification
│       │   └── llm_validator.py     # LLM validation (absorbs classifier.py)
│       │
│       ├── analysis/              # Index calculation & validation
│       │   ├── __init__.py
│       │   ├── indexer.py         # CPU index calculation
│       │   ├── normalizer.py      # BBD-style normalization
│       │   ├── ablation_config.py # Ablation study definitions
│       │   └── ablation_runner.py # Ablation execution
│       │
│       ├── output/                # Reporting & visualization
│       │   ├── __init__.py
│       │   ├── exports.py         # CSV export functions
│       │   ├── visualizations.py  # Charts (absorbs visualize.py)
│       │   └── report_generator.py # Complete report generation
│       │
│       └── cli/                   # User interface
│           ├── __init__.py
│           └── run.py             # Interactive CLI menu
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures
│   ├── collection/
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   ├── test_collector.py
│   │   └── test_deduplicator.py
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── test_classifier.py     # Renamed from test_local_classifier.py
│   │   └── test_llm_validator.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── test_indexer.py
│   │   └── test_normalizer.py
│   └── output/
│       ├── __init__.py
│       ├── test_exports.py
│       ├── test_visualizations.py
│       └── test_report_generator.py
│
├── docs/
│   └── specs/
├── data/                          # (gitignored - for local data files)
├── .env.example
├── .gitignore                     # Updated
├── .python-version
├── docker-compose.yml
├── init.sql
├── memo_template.md
├── README.md
└── requirements.txt
```

## Files to Remove (after merge)

| File | Reason |
|------|--------|
| `boilerplate.py` | Unused LexisNexis reference code |
| `visualize.py` | Merged into visualizations.py |
| `classifier.py` | Merged into llm_validator.py |

## Completion Criteria

- [ ] All REQs implemented
- [ ] `pytest` passes with no failures
- [ ] `python src/cpu_index/cli/run.py` launches successfully
- [ ] No Python files remain in repository root (except setup files)
- [ ] .gitignore updated and CLAUDE.md/CSV files ignored

## Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| Import from old path | Should fail with clear ImportError |
| Run pytest from repo root | Should discover tests in new structure |
| Relative imports within packages | Should work via __init__.py |
| Config access from any module | `from cpu_index.config import ...` works |

## Technical Context

### Key Files to Modify
- All 18 Python source files (update imports)
- All 15 test files (update imports + move)
- `.gitignore` (add entries)
- `requirements.txt` (may need `-e .` for editable install)

### Import Pattern Changes
```python
# Before (flat structure)
import config
import db_postgres
from collector import ArticleCollector

# After (package structure)
from cpu_index import config
from cpu_index import db_postgres
from cpu_index.collection import ArticleCollector
```

### Patterns to Follow
- Use relative imports within same package (`from . import api`)
- Use absolute imports for cross-package (`from cpu_index.classification import local_classifier`)
- Keep __init__.py exports minimal (only public API)

## Merge Details

### visualize.py → visualizations.py
- `visualize.py` has: `create_cpu_chart()`, `create_directional_chart()`, helper functions
- `visualizations.py` is more complete with 8 chart types
- Action: Check if any unique functions in visualize.py need to be preserved; if not, delete

### classifier.py → llm_validator.py
- `classifier.py` has: `classify_article()`, `classify_sample()`, `estimate_classification_cost()`
- `llm_validator.py` has: adaptive sampling, batch classification, report generation
- Action: Ensure llm_validator.py covers all classifier.py functionality; if yes, delete classifier.py
