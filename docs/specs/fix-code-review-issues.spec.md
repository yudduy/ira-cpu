# Specification: Fix Code Review Issues

> **For Claude:** Use `/execute docs/specs/fix-code-review-issues.spec.md` to implement this spec autonomously.

## Goal

Address all issues identified in the code review for the asymmetric uncertainty feature branch, including database migration, defensive coding, and bounds checking.

## Requirements

### Core Requirements

1. **[REQ-1] Database Auto-Migration**
   - Modify `init_db()` in `db.py` to detect and add missing columns to the `index_values` table
   - Use `PRAGMA table_info(index_values)` to get existing columns
   - Add missing columns via `ALTER TABLE index_values ADD COLUMN` for:
     - `numerator_down INTEGER`
     - `raw_ratio_down REAL`
     - `normalized_down REAL`
     - `numerator_up INTEGER`
     - `raw_ratio_up REAL`
     - `normalized_up REAL`
     - `cpu_direction REAL`
   - Migration should be idempotent (safe to run multiple times)
   - Acceptance: Existing database with old 5-column schema upgrades successfully to 12-column schema

2. **[REQ-2] Indexer Defensive Coding**
   - In `indexer.py`, move the statistics calculation loop (lines 218-230) inside a `has_directional` check
   - Only calculate `stats["down"]` and `stats["up"]` when directional data exists
   - In `build_index()`, use `.get()` with defaults when accessing directional fields for `save_index_value()`
   - Acceptance: No KeyError when processing data without directional fields

3. **[REQ-3] Visualization Bounds Check**
   - In `visualize.py` `create_cpu_chart()`, only add event annotations when the event date is within the data range
   - Add check: `if event['date'] <= dates[-1]:` before adding annotation
   - Acceptance: Charts render correctly even if data ends before Nov 2025

4. **[REQ-4] Add Tests for Migration**
   - Add test in `tests/test_db.py` that:
     - Creates a database with old schema (5 columns in index_values)
     - Calls `init_db()`
     - Verifies all 12 columns now exist
   - Acceptance: Test passes and covers the migration path

5. **[REQ-5] Add Tests for Edge Cases**
   - Add test in `tests/test_indexer.py` for processing data without directional fields
   - Verify `build_index()` handles legacy data gracefully
   - Acceptance: Test passes when directional fields are missing

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Migration approach | Auto-migrate in init_db() | User selected; simplest for research project |
| Column detection | PRAGMA table_info | SQLite standard, no external dependencies |
| Defensive access | .get() with defaults | Consistent with existing patterns in codebase |
| Test scope | Migration + edge cases | User confirmed tests are needed |

## Completion Criteria

All of these must be true for completion:

- [ ] `init_db()` automatically adds missing columns to existing databases
- [ ] `build_index()` works with both legacy (2 query types) and new (4 query types) data
- [ ] `create_cpu_chart()` handles event dates beyond data range
- [ ] All existing tests pass (162 tests)
- [ ] New migration test passes
- [ ] New edge case test passes
- [ ] No regressions in functionality

## Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| Existing DB with old schema | Columns added automatically on init_db() |
| Fresh DB (no tables) | Tables created with full schema |
| Data without directional fields | build_index() succeeds, directional stats omitted |
| Event date beyond data range | Annotation not rendered, no error |
| Mixed directional data (some months have it, some don't) | Process what's available, use 0 for missing |

## Integration Points

| System | Integration Method | Notes |
|--------|-------------------|-------|
| SQLite | sqlite3 module | Use PRAGMA for schema inspection |
| Existing tests | pytest fixtures | Use `initialized_db` fixture |

## Out of Scope

Explicitly NOT included in this implementation:
- Formal schema versioning table
- Rollback capability for migrations
- CHANGELOG.md updates
- Documentation of UNCERTAINTY_TERMS change

## Technical Context (from exploration)

### Key Files to Modify
- `db.py`: Add migration logic to `init_db()` (lines 40-112)
- `indexer.py`: Guard directional stats (lines 216-230), defensive access (lines 243-261)
- `visualize.py`: Bounds check for events (lines 111-137)
- `tests/test_db.py`: Add migration test
- `tests/test_indexer.py`: Add edge case test

### Existing Patterns to Follow
- Use `CREATE TABLE IF NOT EXISTS` for idempotency
- Use `INSERT OR REPLACE` for upserts
- Use `.get(key, default)` for optional dictionary access
- Test fixtures in `conftest.py` use `initialized_db` for fresh DB

### Migration Implementation Pattern
```python
def _migrate_index_values_table(cursor):
    """Add missing columns to index_values table for backward compatibility."""
    cursor.execute("PRAGMA table_info(index_values)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    new_columns = [
        ("numerator_down", "INTEGER"),
        ("raw_ratio_down", "REAL"),
        ("normalized_down", "REAL"),
        ("numerator_up", "INTEGER"),
        ("raw_ratio_up", "REAL"),
        ("normalized_up", "REAL"),
        ("cpu_direction", "REAL"),
    ]

    for col_name, col_type in new_columns:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE index_values ADD COLUMN {col_name} {col_type}")
```

### Gotchas Discovered
- SQLite `ALTER TABLE ADD COLUMN` only works for nullable columns (which these are)
- `CREATE TABLE IF NOT EXISTS` does NOT modify existing tables
- Tests use in-memory DB (`:memory:`) so migration test needs special setup
