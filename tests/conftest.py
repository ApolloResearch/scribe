"""Shared fixtures and helpers for scribe tests."""

import importlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # type: ignore[import-not-found]

from scribe.notebook.notebook_mcp_server import SessionInfo, _get_state_file, save_state

# Path to the isolated venv with modified scribe installed
SCRIBE_FORK_DIR = Path(__file__).parent.parent
ISOLATED_PYTHON = SCRIBE_FORK_DIR / ".venv" / "bin" / "python"

# Test constants
TEST_MODEL = "claude-haiku-4-5-20251001"
UNUSED_PORT = 59999  # Port that should not have a server running
DEFAULT_MAX_TURNS = 10
LEGACY_SESSION_IDS = ["legacy-session-id-1", "legacy-session-id-2"]


# ============================================================================
# Helper Functions
# ============================================================================


def get_all_state_files() -> set[Path]:
    """Get all scribe state files in home directory."""
    return set(Path.home().glob(".scribe_state_*.json"))


def get_scribe_mcp_config(
    python_path: str,
    env: dict | None = None,
    session_id: str | None = None,
) -> dict:
    """Generate MCP config for scribe.

    Args:
        python_path: Path to Python interpreter
        env: Additional environment variables
        session_id: Session ID for state isolation (auto-generated if not provided)
    """
    effective_session_id = session_id or str(uuid.uuid4())

    config = {
        "scribe": {
            "type": "stdio",
            "command": python_path,
            "args": ["-m", "scribe.notebook.notebook_mcp_server"],
            "env": {
                "SCRIBE_SESSION_ID": effective_session_id,
            },
        }
    }
    if env:
        config["scribe"]["env"].update(env)
    return config


async def drain_response(client: ClaudeSDKClient) -> None:
    """Consume all messages from client response."""
    async for _ in client.receive_response():
        pass


def get_newest_state_file() -> Path | None:
    """Get the most recently modified state file."""
    state_files = list(Path.home().glob(".scribe_state_*.json"))
    return max(state_files, key=lambda p: p.stat().st_mtime) if state_files else None


def start_session_via_http(
    mcp_server,  # type: ignore[no-untyped-def]
    experiment_name: str = "test",
):
    """Start server and create a session via HTTP.

    Returns:
        tuple of (session_data, server_url, headers)
    """
    server_url = mcp_server.ensure_server_running()
    token = mcp_server.get_token()
    headers = {"Authorization": f"token {token}"} if token else {}

    response = requests.post(
        f"{server_url}/api/scribe/start",
        json={"experiment_name": experiment_name},
        headers=headers,
    )
    response.raise_for_status()
    session_data = response.json()

    # Register in active sessions
    mcp_server._active_sessions[session_data["session_id"]] = mcp_server.SessionInfo(  # pyright: ignore[reportAttributeAccessIssue]
        session_id=session_data["session_id"],
        notebook_path=session_data["notebook_path"],
    )

    return session_data, server_url, headers


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def require_anthropic_api_key():
    """Skip integration tests if ANTHROPIC_API_KEY is not set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set - skipping integration tests")


@pytest.fixture
def track_state_files():
    """Track state files created during test.

    Returns a callable that returns (new_files, removed_files) since fixture setup.
    """
    initial_files = get_all_state_files()

    def get_changes() -> tuple[set[Path], set[Path]]:
        current_files = get_all_state_files()
        new_files = current_files - initial_files
        removed_files = initial_files - current_files
        return new_files, removed_files

    yield get_changes

    # Cleanup: remove any new state files created during test
    new_files, _ = get_changes()
    for f in new_files:
        try:
            f.unlink()
        except FileNotFoundError:
            pass


@pytest.fixture
def python_path():
    """Get the Python interpreter path for the isolated venv with modified scribe."""
    if not ISOLATED_PYTHON.exists():
        pytest.skip(
            f"Isolated venv not found at {ISOLATED_PYTHON}. "
            "Run: uv venv .venv && uv pip install -e . --python .venv/bin/python"
        )
    return str(ISOLATED_PYTHON)


@pytest.fixture
def cleanup_jupyter_processes():
    """Fixture to clean up any Jupyter processes started during tests."""
    yield
    # Kill any orphaned scribe Jupyter processes from tests
    subprocess.run(
        ["pkill", "-f", "scribe.notebook.notebook_server"],
        capture_output=True,
        check=False,
    )


@pytest.fixture
def reset_mcp_module():
    """Factory to reset MCP server module state.

    Usage:
        mcp_server = reset_mcp_module(test_session_id)
        # mcp_server is now a fresh module with SCRIBE_SESSION_ID set
    """
    created_contexts = []

    def _reset(session_id: str):
        ctx = patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id})
        ctx.start()
        created_contexts.append(ctx)

        mcp_server = importlib.import_module("scribe.notebook.notebook_mcp_server")
        importlib.reload(mcp_server)

        mcp_server._server_process = None  # pyright: ignore[reportAttributeAccessIssue]
        mcp_server._server_port = None  # pyright: ignore[reportAttributeAccessIssue]
        mcp_server._server_url = None  # pyright: ignore[reportAttributeAccessIssue]
        mcp_server._server_token = None  # pyright: ignore[reportAttributeAccessIssue]
        mcp_server._is_external_server = False  # pyright: ignore[reportAttributeAccessIssue]
        mcp_server._active_sessions = {}  # pyright: ignore[reportAttributeAccessIssue]
        return mcp_server

    yield _reset

    # Cleanup: stop all patch contexts
    for ctx in created_contexts:
        ctx.stop()


@pytest.fixture
def make_claude_options(python_path):
    """Factory for creating ClaudeAgentOptions with consistent defaults.

    Usage:
        options = make_claude_options(
            allowed_tools=["mcp__scribe__start_new_session"],
            session_id=my_session_id,
        )
    """
    def _make(
        allowed_tools: list[str],
        session_id: str | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            model=TEST_MODEL,
            mcp_servers=get_scribe_mcp_config(python_path, session_id=session_id),
            allowed_tools=allowed_tools,
            max_turns=max_turns,
        )
    return _make
