# Scribe-Fork Integration Test Gaps - Plan & Status

## Summary

We're doing TDD to fix production failures in scribe's compaction scenarios. The MCP server loses session state when Claude Code compacts context.

## What's Been Done

### 1. Added `SessionInfo` Pydantic Model
**File**: `scribe/notebook/notebook_mcp_server.py`

Changed `_active_sessions` from `set[str]` (just session IDs) to `dict[str, SessionInfo]` where:
```python
class SessionInfo(BaseModel):
    session_id: str
    notebook_path: str
```

This fixes the design flaw where notebook paths weren't persisted.

### 2. Updated State Persistence
- `save_state()` now serializes `SessionInfo` objects with `model_dump()`
- `load_state()` / `ensure_server_running()` restores sessions with notebook paths
- State file version bumped to 2
- Backward compatible with old format (list of session IDs)

### 3. Added New Integration Tests
**File**: `tests/test_state_persistence.py`

New test class `TestCompactionScenarios` with 5 tests:
- `test_kernel_state_persists_across_compaction` - PASSES
- `test_list_sessions_then_execute_after_compaction` - FAILS
- `test_execute_code_with_stale_session_returns_clear_error` - PASSES
- `test_state_file_includes_notebook_paths` - PASSES
- `test_multiple_sessions_across_compaction` - FAILS

Also added `TestCompactionScenariosDirect` with direct (non-agent) test.

### 4. Added `TEST_MODEL` Constant
All tests now use `claude-haiku-4-5-20251001` for speed/cost.

### 5. Added pydantic dependency
Added to `pyproject.toml`.

## Current Status: 29/29 Tests Passing ✅

All tests pass including:
- Basic state persistence
- State file includes notebook paths
- Stale session error handling
- list_sessions -> execute_code workflow after compaction
- Multiple sessions across compaction

## Root Cause (Fixed)

`list_sessions()` was not calling `ensure_server_running()` before returning sessions. When MCP 2 started fresh after compaction:
1. `_active_sessions` was empty (fresh process)
2. `list_sessions()` called `get_server_status()` which doesn't load state
3. Empty sessions returned, agent couldn't find session IDs

**Fix**: Added `ensure_server_running()` call at the start of `list_sessions()` to load state from disk.

## Remaining Tasks

1. **Switch to structlog** - Replace `print(file=sys.stderr)` with proper structured logging
2. **Add file-based logging** - Persist logs to `~/.scribe/logs/`
3. **Run full project tests** - Verify no regressions in other test files

## Pyright Configuration Issue

Pyright can't resolve imports from `scribe.notebook.notebook_mcp_server` in test files. Runtime imports work fine.

**Root cause**: Package needs to be installed in editable mode in the venv that pyright uses.

**Fix**: Run `uv pip install -e . --python .venv/bin/python` or configure pyright properly.

Currently using `# pyright: ignore[reportAttributeAccessIssue]` as workaround - should be fixed properly.

## Files Modified

- `scribe/notebook/notebook_mcp_server.py` - SessionInfo model, state persistence, __all__ exports
- `tests/test_state_persistence.py` - New test classes, TEST_MODEL constant
- `pyproject.toml` - Added pydantic, pyright config

## How to Run Tests

```bash
cd /Users/bronson/apex/llm_sessions/scribe-fork
uv run pytest tests/test_state_persistence.py -v --tb=short
```

For just the compaction tests:
```bash
uv run pytest tests/test_state_persistence.py::TestCompactionScenarios -v --tb=short
```
