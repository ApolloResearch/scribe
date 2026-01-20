"""Tests for scribe state persistence across MCP restarts.

These tests verify that:
1. State file is created when a session starts
2. Scribe can reconnect to an existing Jupyter server after MCP restart
3. Stale state is handled gracefully when Jupyter is dead
4. State files have restrictive permissions (0o600)
5. External server auth via SCRIBE_TOKEN works
6. Auth failures (401/403) are distinguished from connection failures
"""

import json
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # type: ignore[import-not-found]


# Path to the isolated venv with modified scribe installed
SCRIBE_FORK_DIR = Path(__file__).parent.parent
ISOLATED_PYTHON = SCRIBE_FORK_DIR / ".venv" / "bin" / "python"


# ============================================================================
# Test Fixtures
# ============================================================================


def get_all_state_files() -> set[Path]:
    """Get all scribe state files in home directory."""
    return set(Path.home().glob(".scribe_state_*.json"))


def get_scribe_mcp_config(python_path: str, env: dict | None = None, session_id: str | None = None) -> dict:
    """Generate MCP config for scribe.

    Args:
        python_path: Path to Python interpreter
        env: Additional environment variables
        session_id: Session ID for state isolation (auto-generated if not provided)
    """
    import uuid

    # Always include a session ID (required by MCP server)
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


# ============================================================================
# State Persistence Tests
# ============================================================================


class TestStatePersistence:
    """Test suite for state persistence functionality."""

    @pytest.mark.asyncio
    async def test_state_file_created_on_session_start(
        self,
        python_path: str,
        track_state_files,
    ):
        """Verify state file is created when a notebook session starts."""
        options = ClaudeAgentOptions(
            mcp_servers=get_scribe_mcp_config(python_path),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
                "mcp__scribe__shutdown_session",
            ],
            max_turns=5,
        )

        async with ClaudeSDKClient(options=options) as client:
            # Ask Claude to create a notebook session
            await client.query(
                "Use the start_new_session tool to create a new notebook session, "
                "then use execute_code to run: print('hello')"
            )
            async for _ in client.receive_response():
                pass  # Wait for completion

        # Verify state file was created
        new_files, _ = track_state_files()
        assert len(new_files) > 0, "State file should be created after session start"

        # Check contents of one of the new state files
        state_file = next(iter(new_files))
        state = json.loads(state_file.read_text())
        assert "server" in state
        assert state["server"]["port"] is not None
        assert state["server"]["token"] is not None
        assert "sessions" in state
        assert len(state["sessions"]) > 0

    @pytest.mark.asyncio
    async def test_reconnection_after_mcp_restart(
        self,
        python_path: str,
        track_state_files,
    ):
        """Verify scribe reconnects to existing Jupyter after MCP process restart."""
        options = ClaudeAgentOptions(
            mcp_servers=get_scribe_mcp_config(python_path),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
            ],
            max_turns=5,
        )

        # First session: create notebook and execute code
        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use start_new_session to create a notebook, "
                "then execute_code to run: x = 42"
            )
            async for _ in client.receive_response():
                pass

        # Capture state after first session
        new_files, _ = track_state_files()
        assert len(new_files) > 0, "State file should exist after first session"
        state_file = next(iter(new_files))
        state_before = json.loads(state_file.read_text())
        port_before = state_before["server"]["port"]

        # Second session: should reconnect to same Jupyter server
        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use execute_code to run: print(x)  # Should print 42 if reconnected"
            )
            async for _ in client.receive_response():
                pass

        # Verify same port was used (reconnection)
        state_after = json.loads(state_file.read_text())
        assert (
            state_after["server"]["port"] == port_before
        ), "Should reconnect to same Jupyter server"

    @pytest.mark.asyncio
    async def test_stale_state_handled_gracefully(
        self,
        python_path: str,
        track_state_files,
    ):
        """Verify stale state (dead Jupyter) is cleared and fresh server started."""
        # Get any existing state files first
        initial_new_files, _ = track_state_files()

        options = ClaudeAgentOptions(
            mcp_servers=get_scribe_mcp_config(python_path),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
            ],
            max_turns=5,
        )

        # First, create a real session to get a state file
        async with ClaudeSDKClient(options=options) as client:
            await client.query("Use start_new_session to create a notebook")
            async for _ in client.receive_response():
                pass

        # Get the state file that was created
        new_files, _ = track_state_files()
        new_files = new_files - initial_new_files
        assert len(new_files) > 0, "State file should exist"
        state_file = next(iter(new_files))

        # Now corrupt the state file with a fake dead server
        fake_state = {
            "version": 1,
            "server": {
                "port": 59999,  # Unlikely to be in use
                "token": "fake_token_that_wont_work",
                "pid": 99999,
                "url": "http://127.0.0.1:59999",
            },
            "sessions": ["fake_session"],
            "updated_at": "2026-01-01T00:00:00",
        }
        state_file.write_text(json.dumps(fake_state))

        # Create a new session - should detect dead server and start fresh
        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use start_new_session to create a notebook, "
                "then execute_code to run: print('recovered')"
            )
            async for _ in client.receive_response():
                pass

        # Verify state was updated with a new (different) port
        state_after = json.loads(state_file.read_text())
        assert (
            state_after["server"]["port"] != 59999
        ), "Should have started a new server, not used stale state"

    @pytest.mark.asyncio
    async def test_state_file_has_restrictive_permissions(
        self,
        python_path: str,
        track_state_files,
    ):
        """Verify state file is created with 0o600 permissions (owner read/write only)."""
        options = ClaudeAgentOptions(
            mcp_servers=get_scribe_mcp_config(python_path),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
            ],
            max_turns=5,
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query("Use start_new_session to create a notebook")
            async for _ in client.receive_response():
                pass

        # Get the state file that was created
        new_files, _ = track_state_files()
        assert len(new_files) > 0, "State file should be created"
        state_file = next(iter(new_files))

        # Check permissions - should be 0o600 (owner read/write only)
        file_stat = state_file.stat()
        mode = stat.S_IMODE(file_stat.st_mode)
        assert mode == 0o600, (
            f"State file should have 0o600 permissions, got {oct(mode)}. "
            "Token is stored in plaintext and should be protected."
        )


# ============================================================================
# Multiple Instance Tests
# ============================================================================


class TestMultipleInstances:
    """Test that multiple working directories get separate state files."""

    def test_different_dirs_get_different_state_files(self):
        """Verify different working directories use different state files."""
        from scribe.notebook.notebook_mcp_server import _get_state_file  # type: ignore[attr-defined]

        dir1 = "/tmp/scribe_test_dir1"
        dir2 = "/tmp/scribe_test_dir2"
        session_id = "test_session_12345678"

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("os.getcwd", return_value=dir1):
                state1 = _get_state_file()
            with patch("os.getcwd", return_value=dir2):
                state2 = _get_state_file()

        assert (
            state1 != state2
        ), "Different directories should have different state file paths"
        assert (
            state1.name != state2.name
        ), "State file names should differ based on directory hash"


# ============================================================================
# Server Status Check Tests (Unit Tests)
# ============================================================================


class TestServerStatusChecks:
    """Unit tests for server status checking logic.

    NOTE: These tests will fail until check_jupyter_status and ServerStatus
    are implemented in notebook_mcp_server.py. This is intentional (TDD).
    """

    def test_check_jupyter_status_healthy(self):
        """Verify healthy server returns HEALTHY status."""
        # Import the function we want to test
        # type: ignore comments needed until implementation exists
        from scribe.notebook.notebook_mcp_server import check_jupyter_status, ServerStatus  # type: ignore[attr-defined]

        with patch("scribe.notebook.notebook_mcp_server.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            status = check_jupyter_status(8888, "test_token")
            assert status == ServerStatus.HEALTHY

    def test_check_jupyter_status_unauthorized(self):
        """Verify 401/403 returns UNAUTHORIZED status (not UNREACHABLE)."""
        from scribe.notebook.notebook_mcp_server import check_jupyter_status, ServerStatus  # type: ignore[attr-defined]

        with patch("scribe.notebook.notebook_mcp_server.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_get.return_value = mock_response

            status = check_jupyter_status(8888, "wrong_token")
            assert status == ServerStatus.UNAUTHORIZED, (
                "401 should be UNAUTHORIZED, not treated as dead server"
            )

        with patch("scribe.notebook.notebook_mcp_server.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_get.return_value = mock_response

            status = check_jupyter_status(8888, "wrong_token")
            assert status == ServerStatus.UNAUTHORIZED, (
                "403 should be UNAUTHORIZED, not treated as dead server"
            )

    def test_check_jupyter_status_unreachable(self):
        """Verify connection errors return UNREACHABLE status."""
        from scribe.notebook.notebook_mcp_server import check_jupyter_status, ServerStatus  # type: ignore[attr-defined]

        with patch("scribe.notebook.notebook_mcp_server.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")

            status = check_jupyter_status(8888, "test_token")
            assert status == ServerStatus.UNREACHABLE

    def test_is_jupyter_alive_backwards_compatible(self):
        """Verify is_jupyter_alive returns bool (backwards compatible)."""
        from scribe.notebook.notebook_mcp_server import is_jupyter_alive  # type: ignore[attr-defined]
        from scribe.notebook.notebook_mcp_server import ServerStatus  # type: ignore[attr-defined]

        with patch("scribe.notebook.notebook_mcp_server.check_jupyter_status") as mock_check:
            mock_check.return_value = ServerStatus.HEALTHY
            assert is_jupyter_alive(8888, "token") is True

            mock_check.return_value = ServerStatus.UNAUTHORIZED
            assert is_jupyter_alive(8888, "token") is False

            mock_check.return_value = ServerStatus.UNREACHABLE
            assert is_jupyter_alive(8888, "token") is False


# ============================================================================
# External Server Tests
# ============================================================================


class TestExternalServer:
    """Tests for external server (SCRIBE_PORT/SCRIBE_TOKEN) functionality."""

    def test_scribe_token_env_var_is_used(self):
        """Verify SCRIBE_TOKEN environment variable is read for external servers."""
        # This is a unit test that verifies the env var is read
        # We can't easily integration test this without a real external server

        # Patch environment and test ensure_server_running
        with patch.dict(
            os.environ,
            {"SCRIBE_PORT": "9999", "SCRIBE_TOKEN": "external_test_token"},
        ):
            # Need to reimport to pick up env changes
            import importlib
            import scribe.notebook.notebook_mcp_server as mcp_server
            importlib.reload(mcp_server)

            # Reset module state
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._is_external_server = False

            # Call ensure_server_running with external server env vars
            url = mcp_server.ensure_server_running()

            assert mcp_server._server_port == 9999
            assert mcp_server._server_token == "external_test_token"
            assert mcp_server._is_external_server is True
            assert url == "http://127.0.0.1:9999"

            # Clean up
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._is_external_server = False


# ============================================================================
# Session Isolation Tests (Unit Tests)
# ============================================================================


class TestSessionIsolation:
    """Tests for session isolation via SCRIBE_SESSION_ID.

    These verify that concurrent scribe sessions in the same directory
    get separate state files, while the same session (after compaction)
    reconnects to its own Jupyter server.
    """

    def test_different_session_ids_use_different_state_files(self):
        """Verify different SCRIBE_SESSION_IDs result in different state file paths."""
        from scribe.notebook.notebook_mcp_server import _get_state_file  # type: ignore[attr-defined]

        cwd = "/tmp/test_cwd"
        # Use UUIDs that differ in first 8 chars (the truncation length)
        session_id_1 = "aaaaaaaa-1111-1111-1111-111111111111"
        session_id_2 = "bbbbbbbb-2222-2222-2222-222222222222"

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id_1}):
            with patch("os.getcwd", return_value=cwd):
                file1 = _get_state_file()

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id_2}):
            with patch("os.getcwd", return_value=cwd):
                file2 = _get_state_file()

        assert file1 != file2, "Different session IDs should use different state files"
        assert "aaaaaaaa" in file1.name
        assert "bbbbbbbb" in file2.name

    def test_same_session_id_uses_same_state_file(self):
        """Verify same SCRIBE_SESSION_ID (after compaction) uses same state file."""
        from scribe.notebook.notebook_mcp_server import _get_state_file  # type: ignore[attr-defined]

        cwd = "/tmp/test_cwd"
        session_id = "persistent_session_123"

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("os.getcwd", return_value=cwd):
                file1 = _get_state_file()
                file2 = _get_state_file()

        assert file1 == file2, "Same session ID should use same state file"

    def test_no_session_id_raises_error(self):
        """Verify missing SCRIBE_SESSION_ID raises RuntimeError."""
        from scribe.notebook.notebook_mcp_server import _get_state_file  # type: ignore[attr-defined]

        # Create a clean environment without SCRIBE_SESSION_ID
        clean_env = {k: v for k, v in os.environ.items() if k != "SCRIBE_SESSION_ID"}
        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(RuntimeError, match="SCRIBE_SESSION_ID environment variable is required"):
                _get_state_file()
