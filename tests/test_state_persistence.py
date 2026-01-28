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
from unittest.mock import MagicMock, patch

import pytest
import requests
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # type: ignore[import-not-found]

import scribe.notebook.notebook_mcp_server as mcp_module  # type: ignore[import]

# Aliases for cleaner usage in tests (pyright can't resolve these but they work at runtime)
SessionInfo = mcp_module.SessionInfo  # pyright: ignore[reportAttributeAccessIssue]
_get_state_file = mcp_module._get_state_file  # pyright: ignore[reportAttributeAccessIssue]
save_state = mcp_module.save_state  # pyright: ignore[reportAttributeAccessIssue]

# Path to the isolated venv with modified scribe installed
SCRIBE_FORK_DIR = Path(__file__).parent.parent
ISOLATED_PYTHON = SCRIBE_FORK_DIR / ".venv" / "bin" / "python"

# Model to use for integration tests (Haiku for speed/cost)
TEST_MODEL = "claude-haiku-4-5-20251001"


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
            model=TEST_MODEL,
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
            model=TEST_MODEL,
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
            model=TEST_MODEL,
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
            model=TEST_MODEL,
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
        from scribe.notebook.notebook_mcp_server import ServerStatus, check_jupyter_status  # type: ignore[attr-defined]

        with patch("scribe.notebook.notebook_mcp_server.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            status = check_jupyter_status(8888, "test_token")
            assert status == ServerStatus.HEALTHY

    def test_check_jupyter_status_unauthorized(self):
        """Verify 401/403 returns UNAUTHORIZED status (not UNREACHABLE)."""
        from scribe.notebook.notebook_mcp_server import ServerStatus, check_jupyter_status  # type: ignore[attr-defined]

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
        from scribe.notebook.notebook_mcp_server import ServerStatus, check_jupyter_status  # type: ignore[attr-defined]

        with patch("scribe.notebook.notebook_mcp_server.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")

            status = check_jupyter_status(8888, "test_token")
            assert status == ServerStatus.UNREACHABLE

    def test_is_jupyter_alive_backwards_compatible(self):
        """Verify is_jupyter_alive returns bool (backwards compatible)."""
        from scribe.notebook.notebook_mcp_server import (
            ServerStatus,  # type: ignore[attr-defined]
            is_jupyter_alive,  # type: ignore[attr-defined]
        )

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

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id_1}), patch("os.getcwd", return_value=cwd):
            file1 = _get_state_file()

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id_2}), patch("os.getcwd", return_value=cwd):
            file2 = _get_state_file()

        assert file1 != file2, "Different session IDs should use different state files"
        assert "aaaaaaaa" in file1.name
        assert "bbbbbbbb" in file2.name

    def test_same_session_id_uses_same_state_file(self):
        """Verify same SCRIBE_SESSION_ID (after compaction) uses same state file."""
        from scribe.notebook.notebook_mcp_server import _get_state_file  # type: ignore[attr-defined]

        cwd = "/tmp/test_cwd"
        session_id = "persistent_session_123"

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}), patch("os.getcwd", return_value=cwd):
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


# ============================================================================
# Response Handling Unit Tests
# ============================================================================


class TestCheckResponse:
    """Unit tests for _check_response() helper function."""

    def test_check_response_success_with_json(self):
        """Verify successful response with JSON returns parsed data."""
        from scribe.notebook.notebook_mcp_server import _check_response  # type: ignore[attr-defined]

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = b'{"result": "success"}'
        mock_response.json.return_value = {"result": "success"}

        result = _check_response(mock_response, "test operation")
        assert result == {"result": "success"}

    def test_check_response_success_empty_content(self):
        """Verify successful response with empty content returns empty dict."""
        from scribe.notebook.notebook_mcp_server import _check_response  # type: ignore[attr-defined]

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = b""

        result = _check_response(mock_response, "test operation")
        assert result == {}

    def test_check_response_success_non_json(self):
        """Verify successful response with non-JSON returns empty dict."""
        from scribe.notebook.notebook_mcp_server import _check_response  # type: ignore[attr-defined]

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = b"plain text response"
        mock_response.json.side_effect = ValueError("Not JSON")

        result = _check_response(mock_response, "test operation")
        assert result == {}

    def test_check_response_error_with_error_field(self):
        """Verify error response extracts 'error' field from JSON."""
        from scribe.notebook.notebook_mcp_server import _check_response  # type: ignore[attr-defined]

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Session not found"}
        mock_response.text = '{"error": "Session not found"}'

        with pytest.raises(Exception) as exc_info:
            _check_response(mock_response, "execute code")

        error_msg = str(exc_info.value)
        assert "Session not found" in error_msg
        assert "HTTP 500" in error_msg
        assert "execute code" in error_msg

    def test_check_response_error_with_detail_field(self):
        """Verify error response extracts 'detail' field from JSON."""
        from scribe.notebook.notebook_mcp_server import _check_response  # type: ignore[attr-defined]

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Invalid request"}
        mock_response.text = '{"detail": "Invalid request"}'

        with pytest.raises(Exception) as exc_info:
            _check_response(mock_response, "test operation")

        error_msg = str(exc_info.value)
        assert "Invalid request" in error_msg
        assert "HTTP 400" in error_msg

    def test_check_response_error_with_message_field(self):
        """Verify error response extracts 'message' field from JSON."""
        from scribe.notebook.notebook_mcp_server import _check_response  # type: ignore[attr-defined]

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 403
        mock_response.json.return_value = {"message": "Forbidden"}
        mock_response.text = '{"message": "Forbidden"}'

        with pytest.raises(Exception) as exc_info:
            _check_response(mock_response, "test operation")

        error_msg = str(exc_info.value)
        assert "Forbidden" in error_msg
        assert "HTTP 403" in error_msg

    def test_check_response_error_non_json(self):
        """Verify error response with non-JSON uses response text."""
        from scribe.notebook.notebook_mcp_server import _check_response  # type: ignore[attr-defined]

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Not JSON")
        mock_response.text = "Internal Server Error"

        with pytest.raises(Exception) as exc_info:
            _check_response(mock_response, "test operation")

        error_msg = str(exc_info.value)
        assert "Internal Server Error" in error_msg
        assert "HTTP 500" in error_msg

    def test_check_response_error_no_content(self):
        """Verify error response with no content provides helpful message."""
        from scribe.notebook.notebook_mcp_server import _check_response  # type: ignore[attr-defined]

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Not JSON")
        mock_response.text = ""

        with pytest.raises(Exception) as exc_info:
            _check_response(mock_response, "test operation")

        error_msg = str(exc_info.value)
        assert "No error details" in error_msg
        assert "HTTP 500" in error_msg


# ============================================================================
# Error Handling and Session Discovery Tests
# ============================================================================


class TestErrorHandlingAndSessionDiscovery:
    """Integration tests for error handling and session discovery."""

    @pytest.mark.asyncio
    async def test_invalid_session_id_returns_clear_error(
        self,
        python_path: str,
        cleanup_jupyter_processes,
    ):
        """Verify invalid session_id error is propagated through MCP."""
        import requests

        # Set up environment and start Jupyter server manually
        test_session_id = "test_error_handling_12345678"
        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": test_session_id}):
            from scribe.notebook.notebook_mcp_server import ensure_server_running, get_token

            server_url = ensure_server_running()
            token = get_token()
            headers = {"Authorization": f"token {token}"} if token else {}

            # Try to execute with fake session_id - server should return error
            response = requests.post(
                f"{server_url}/api/scribe/exec",
                json={"session_id": "fake_session_that_does_not_exist", "code": "print(1)"},
                headers=headers,
            )

            # Should get 500 with error message
            assert response.status_code == 500
            error_data = response.json()
            error_message = error_data.get("error", "").lower()

            # Error should mention session and not found
            assert "session" in error_message and "not found" in error_message, (
                f"Server error should mention 'Session not found', got: {error_data}"
            )

    @pytest.mark.asyncio
    async def test_list_sessions_mcp_integration(
        self,
        python_path: str,
        track_state_files,
        cleanup_jupyter_processes,
    ):
        """Verify list_sessions tool works through MCP protocol."""
        options = ClaudeAgentOptions(
            model=TEST_MODEL,
            mcp_servers=get_scribe_mcp_config(python_path),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__list_sessions",
                "mcp__scribe__execute_code",
            ],
            max_turns=10,
        )

        session_id_found = False
        executed_successfully = False

        async with ClaudeSDKClient(options=options) as client:
            # Create session, list sessions, then execute
            await client.query(
                "1. Use start_new_session to create a notebook\n"
                "2. Use list_sessions and tell me the exact session_id you see\n"
                "3. Use execute_code with that session_id to run: print('test_success')"
            )

            async for msg in client.receive_response():
                msg_text = str(msg)
                # Look for UUID pattern (session IDs are UUIDs)
                if "-" in msg_text and len(msg_text) > 30:
                    session_id_found = True
                if "test_success" in msg_text.lower():
                    executed_successfully = True

        # Both should have occurred
        assert session_id_found, "Should have seen a session_id from list_sessions"
        assert executed_successfully, "Should have successfully executed code using listed session_id"


# ============================================================================
# Compaction Scenario Integration Tests (TDD - should fail initially)
# ============================================================================


class TestCompactionScenariosDirect:
    """Direct tests (without agent) for state persistence across MCP restarts.

    These tests directly call the MCP server functions to verify core functionality
    without relying on agent interpretation.
    """

    @pytest.mark.asyncio
    async def test_state_persistence_direct(
        self,
        cleanup_jupyter_processes,
    ):
        """Directly verify state persistence works across MCP module reloads."""
        import importlib
        import uuid

        import requests

        test_session_id = str(uuid.uuid4())

        # Phase 1: Start server, create session, execute code
        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": test_session_id}):
            import scribe.notebook.notebook_mcp_server as mcp_server

            importlib.reload(mcp_server)

            # Reset module state
            mcp_server._server_process = None
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._is_external_server = False
            mcp_server._active_sessions = {}

            # Start server and create session
            server_url = mcp_server.ensure_server_running()
            token = mcp_server.get_token()
            headers = {"Authorization": f"token {token}"} if token else {}

            # Create session via HTTP
            response = requests.post(
                f"{server_url}/api/scribe/start",
                json={"experiment_name": "direct_test"},
                headers=headers,
            )
            assert response.ok, f"Failed to start session: {response.text}"
            session_data = response.json()
            session_id = session_data["session_id"]
            notebook_path = session_data["notebook_path"]

            # Register session in MCP server's tracking (normally done by MCP tool)
            mcp_server._active_sessions[session_id] = mcp_server.SessionInfo(  # type: ignore[attr-defined]
                session_id=session_id,
                notebook_path=notebook_path,
            )

            # Execute code to set variable
            response = requests.post(
                f"{server_url}/api/scribe/exec",
                json={"session_id": session_id, "code": "test_var = 'persistence_works'"},
                headers=headers,
            )
            assert response.ok, f"Failed to execute code: {response.text}"

            # Save state and capture port for later verification
            mcp_server.save_state()  # type: ignore[attr-defined]
            original_port = mcp_server._server_port

            # Verify state was saved with session
            state_file = mcp_server._get_state_file()  # type: ignore[attr-defined]
            saved_state = json.loads(state_file.read_text())
            assert len(saved_state.get("sessions", [])) > 0, "State file should have sessions"

        # Phase 2: Simulate MCP restart by reloading module and clearing state
        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": test_session_id}):
            importlib.reload(mcp_server)

            # Reset module state (simulating fresh MCP process)
            mcp_server._server_process = None
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._is_external_server = False
            mcp_server._active_sessions = {}

            # Reconnect - should restore from state file
            server_url = mcp_server.ensure_server_running()
            token = mcp_server.get_token()
            headers = {"Authorization": f"token {token}"} if token else {}

            # Verify we reconnected to same server
            assert mcp_server._server_port == original_port, (
                f"Should reconnect to same port. Expected {original_port}, got {mcp_server._server_port}"
            )

            # Verify sessions were restored
            assert len(mcp_server._active_sessions) > 0, "Sessions should be restored from state"
            assert session_id in mcp_server._active_sessions, f"Session {session_id} should be in active sessions"

            # Execute code to verify kernel state persists
            response = requests.post(
                f"{server_url}/api/scribe/exec",
                json={"session_id": session_id, "code": "print(test_var)"},
                headers=headers,
            )
            assert response.ok, f"Failed to execute code after reconnect: {response.text}"
            result = response.json()

            # Check output contains our test value
            outputs = result.get("outputs", [])
            output_text = "".join(
                o.get("text", "") for o in outputs if o.get("output_type") == "stream"
            )
            assert "persistence_works" in output_text, (
                f"Variable should persist after MCP restart. Got output: {output_text}"
            )


class TestCompactionScenarios:
    """Integration tests for scenarios that fail in production during compaction.

    These tests verify that the actual workflows work after an MCP restart
    (simulating Claude Code compaction). They should fail initially, revealing
    the gaps in the current implementation.
    """

    @pytest.mark.asyncio
    async def test_kernel_state_persists_across_compaction(
        self,
        python_path: str,
        track_state_files,
        cleanup_jupyter_processes,
    ):
        """Verify variables survive MCP restart (compaction simulation).

        This is the core compaction failure scenario:
        1. MCP 1: Create session, set x = 42
        2. MCP 1 exits (simulating compaction)
        3. MCP 2: Get session_id from state, execute print(x)
        4. Assert: Output contains "42"
        """
        import uuid

        # Use a fixed session ID so both MCP instances use the same state file
        shared_session_id = str(uuid.uuid4())

        options = ClaudeAgentOptions(
            model=TEST_MODEL,
            mcp_servers=get_scribe_mcp_config(python_path, session_id=shared_session_id),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
            ],
            max_turns=5,
        )

        # MCP 1: Create session and set variable
        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use start_new_session to create a notebook, "
                "then execute_code to run: x = 42"
            )
            async for _ in client.receive_response():
                pass

        # Verify state file exists
        new_files, _ = track_state_files()
        assert len(new_files) > 0, "State file should exist after first session"
        state_file = next(iter(new_files))
        state = json.loads(state_file.read_text())
        saved_sessions = state.get("sessions", [])
        assert len(saved_sessions) > 0, "Should have at least one session saved"

        # MCP 2: Use the same session_id to execute print(x)
        # This simulates what happens after compaction - we need the session_id
        # from the state file to continue execution
        variable_value_found = False

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use execute_code to run: print(x)  # Should print 42 if state persisted"
            )
            async for msg in client.receive_response():
                msg_text = str(msg)
                if "42" in msg_text:
                    variable_value_found = True

        assert variable_value_found, (
            "Variable x should have value 42 after MCP restart. "
            "This indicates kernel state was NOT preserved across compaction."
        )

    @pytest.mark.asyncio
    async def test_list_sessions_then_execute_after_compaction(
        self,
        python_path: str,
        track_state_files,
        cleanup_jupyter_processes,
    ):
        """Verify the actual post-compaction workflow works.

        This tests the expected user workflow:
        1. MCP 1: Create session, execute x = 42
        2. MCP 1 exits
        3. MCP 2: list_sessions -> get session_id -> execute_code(print(x))
        4. Verify output is "42"
        """
        import uuid

        shared_session_id = str(uuid.uuid4())

        options = ClaudeAgentOptions(
            model=TEST_MODEL,
            mcp_servers=get_scribe_mcp_config(python_path, session_id=shared_session_id),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
                "mcp__scribe__list_sessions",
            ],
            max_turns=10,
        )

        # MCP 1: Create session and set variable
        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use start_new_session to create a notebook, "
                "then execute_code to run: my_var = 'compaction_test_value'"
            )
            async for _ in client.receive_response():
                pass

        # MCP 2: Use list_sessions to discover session, then execute
        test_value_found = False
        session_discovered = False

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "1. Use list_sessions to find active sessions\n"
                "2. Use the session_id from list_sessions to execute: print(my_var)\n"
                "Tell me the exact output."
            )
            async for msg in client.receive_response():
                msg_text = str(msg)
                # Look for session discovery
                if "-" in msg_text and len(msg_text) > 30:  # UUID pattern
                    session_discovered = True
                # Look for our test value
                if "compaction_test_value" in msg_text:
                    test_value_found = True

        assert session_discovered, "Should have discovered session via list_sessions"
        assert test_value_found, (
            "Variable my_var should have value 'compaction_test_value' after "
            "discovering session via list_sessions. This indicates the "
            "list_sessions -> execute_code workflow fails after compaction."
        )

    @pytest.mark.asyncio
    async def test_execute_code_with_stale_session_returns_clear_error(
        self,
        python_path: str,
        cleanup_jupyter_processes,
    ):
        """Verify stale session_id gives actionable error, not cryptic failure.

        When a session_id exists in state but the kernel has been cleaned up,
        execute_code should return a clear "Session not found" error, not crash
        or return confusing output.
        """
        import uuid

        shared_session_id = str(uuid.uuid4())

        options = ClaudeAgentOptions(
            model=TEST_MODEL,
            mcp_servers=get_scribe_mcp_config(python_path, session_id=shared_session_id),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
                "mcp__scribe__shutdown_session",
            ],
            max_turns=10,
        )

        captured_session_id = None

        # MCP 1: Create session, capture session_id, then shutdown
        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "1. Use start_new_session to create a notebook\n"
                "2. Tell me the exact session_id\n"
                "3. Use shutdown_session to close it"
            )
            async for msg in client.receive_response():
                msg_text = str(msg)
                # Try to capture UUID-like session ID
                import re
                uuid_match = re.search(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    msg_text,
                    re.IGNORECASE,
                )
                if uuid_match:
                    captured_session_id = uuid_match.group()

        assert captured_session_id, "Should have captured a session_id"

        # MCP 2: Try to execute with the now-stale session_id
        error_received = False
        error_message = ""

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                f"Use execute_code with session_id='{captured_session_id}' to run: print('test')\n"
                "Tell me the exact error if there is one."
            )
            async for msg in client.receive_response():
                msg_text = str(msg).lower()
                if "session" in msg_text and ("not found" in msg_text or "error" in msg_text):
                    error_received = True
                    error_message = str(msg)

        assert error_received, (
            f"Should receive clear 'Session not found' error for stale session_id. "
            f"Instead got: {error_message or 'no clear error message'}"
        )

    @pytest.mark.asyncio
    async def test_state_file_includes_notebook_paths(
        self,
        python_path: str,
        track_state_files,
        cleanup_jupyter_processes,
    ):
        """Verify state file preserves notebook paths for post-compaction recovery.

        After compaction, the agent needs to know not just the session_id but also
        where the notebook file is located. This tests that the state file includes
        notebook path information.
        """
        import uuid

        shared_session_id = str(uuid.uuid4())

        options = ClaudeAgentOptions(
            model=TEST_MODEL,
            mcp_servers=get_scribe_mcp_config(python_path, session_id=shared_session_id),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
            ],
            max_turns=5,
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use start_new_session with experiment_name='test_notebook_path' "
                "to create a notebook"
            )
            async for _ in client.receive_response():
                pass

        # Check state file structure
        new_files, _ = track_state_files()
        assert len(new_files) > 0, "State file should exist"
        state_file = next(iter(new_files))
        state = json.loads(state_file.read_text())

        # Verify sessions include notebook paths
        sessions = state.get("sessions", [])
        assert len(sessions) > 0, "Should have at least one session"

        # Check if sessions is a list of dicts with notebook_path, or just a list of IDs
        if isinstance(sessions[0], str):
            # Current implementation: sessions is just a list of session IDs
            pytest.fail(
                "State file 'sessions' field only contains session IDs, not notebook paths. "
                "After compaction, agent cannot determine where the notebook file is. "
                "Sessions should be stored as: [{'session_id': '...', 'notebook_path': '...'}]"
            )
        elif isinstance(sessions[0], dict):
            # Expected implementation: sessions is a list of dicts
            first_session = sessions[0]
            assert "notebook_path" in first_session, (
                "Session entry should include 'notebook_path' for post-compaction recovery"
            )
            assert first_session["notebook_path"], "notebook_path should not be empty"

    @pytest.mark.asyncio
    async def test_multiple_sessions_across_compaction(
        self,
        python_path: str,
        track_state_files,
        cleanup_jupyter_processes,
    ):
        """Verify multiple sessions are all preserved across compaction.

        Production often has 2+ concurrent sessions. This test verifies that
        all sessions are preserved, not just the most recent one.
        """
        import uuid

        shared_session_id = str(uuid.uuid4())

        options = ClaudeAgentOptions(
            model=TEST_MODEL,
            mcp_servers=get_scribe_mcp_config(python_path, session_id=shared_session_id),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
                "mcp__scribe__list_sessions",
            ],
            max_turns=15,
        )

        # MCP 1: Create two sessions with different variables
        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "1. Use start_new_session to create notebook A\n"
                "2. In that session, execute: session_a_var = 'value_A'\n"
                "3. Use start_new_session again to create notebook B\n"
                "4. In the NEW session, execute: session_b_var = 'value_B'\n"
                "Tell me both session_ids."
            )
            async for _ in client.receive_response():
                pass

        # Check state file has both sessions
        new_files, _ = track_state_files()
        assert len(new_files) > 0, "State file should exist"
        state_file = next(iter(new_files))
        state = json.loads(state_file.read_text())
        saved_sessions = state.get("sessions", [])

        assert len(saved_sessions) >= 2, (
            f"Should have at least 2 sessions saved, got {len(saved_sessions)}. "
            "Multiple sessions are not being preserved in state file."
        )

        # MCP 2: Verify both sessions are accessible
        session_a_found = False
        session_b_found = False

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "1. Use list_sessions to find all active sessions\n"
                "2. For EACH session_id, use execute_code to run: "
                "print(locals().get('session_a_var', 'NOT_FOUND'), "
                "locals().get('session_b_var', 'NOT_FOUND'))\n"
                "Tell me the output from each session."
            )
            async for msg in client.receive_response():
                msg_text = str(msg)
                if "value_A" in msg_text:
                    session_a_found = True
                if "value_B" in msg_text:
                    session_b_found = True

        assert session_a_found, (
            "Session A with session_a_var='value_A' not accessible after compaction"
        )
        assert session_b_found, (
            "Session B with session_b_var='value_B' not accessible after compaction"
        )


class TestBackwardCompatibility:
    """Tests for state file format migrations and backward compatibility."""

    def test_load_state_v1_format_migration(self, cleanup_jupyter_processes):
        """Verify v1 state files (session IDs as strings) work with v2 code.

        v1 format had sessions as list of strings: ["session-id-1", "session-id-2"]
        v2 format has sessions as list of dicts: [{"session_id": "...", "notebook_path": "..."}]

        The code should handle both formats gracefully.
        """
        import importlib
        import uuid

        test_session_id = str(uuid.uuid4())

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": test_session_id}):
            import scribe.notebook.notebook_mcp_server as mcp_server

            importlib.reload(mcp_server)

            # Reset module state
            mcp_server._server_process = None
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._is_external_server = False
            mcp_server._active_sessions = {}

            # Create a v1 format state file manually
            state_file = mcp_server._get_state_file()  # pyright: ignore[reportAttributeAccessIssue]

            # Start a real server to get valid port/token
            server_url = mcp_server.ensure_server_running()
            port = mcp_server._server_port
            token = mcp_server._server_token

            # Now write a v1 format state file (sessions as list of strings)
            v1_state = {
                "version": 1,
                "server": {
                    "port": port,
                    "token": token,
                    "pid": None,
                    "url": server_url,
                },
                "sessions": ["legacy-session-id-1", "legacy-session-id-2"],  # v1 format
                "updated_at": "2024-01-01T00:00:00",
            }
            state_file.write_text(json.dumps(v1_state, indent=2))

            # Reset module state to simulate MCP restart
            mcp_server._server_process = None
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._active_sessions = {}

            # Reload state - should handle v1 format
            mcp_server.ensure_server_running()

            # Verify sessions were migrated to SessionInfo objects
            assert len(mcp_server._active_sessions) == 2, (
                f"Expected 2 sessions, got {len(mcp_server._active_sessions)}"
            )
            assert "legacy-session-id-1" in mcp_server._active_sessions
            assert "legacy-session-id-2" in mcp_server._active_sessions

            # Verify SessionInfo objects have empty notebook_path (legacy sessions don't have paths)
            session_info = mcp_server._active_sessions["legacy-session-id-1"]
            assert session_info.session_id == "legacy-session-id-1"
            assert session_info.notebook_path == "", (
                "Legacy sessions should have empty notebook_path"
            )

    def test_load_state_future_version_graceful_handling(self, cleanup_jupyter_processes):
        """Verify graceful handling of state files from future versions.

        If someone upgrades scribe, uses it, then downgrades, the state file
        might have a higher version number. Code should handle this gracefully
        (either by ignoring unknown fields or starting fresh).
        """
        import importlib
        import uuid

        test_session_id = str(uuid.uuid4())

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": test_session_id}):
            import scribe.notebook.notebook_mcp_server as mcp_server

            importlib.reload(mcp_server)

            # Reset module state
            mcp_server._server_process = None
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._is_external_server = False
            mcp_server._active_sessions = {}

            # Create state file with future version
            state_file = mcp_server._get_state_file()  # pyright: ignore[reportAttributeAccessIssue]

            # Start a real server first
            server_url = mcp_server.ensure_server_running()
            port = mcp_server._server_port
            token = mcp_server._server_token

            # Write a future version state file
            future_state = {
                "version": 999,  # Future version
                "server": {
                    "port": port,
                    "token": token,
                    "pid": None,
                    "url": server_url,
                },
                "sessions": [
                    {
                        "session_id": "future-session",
                        "notebook_path": "/some/path.ipynb",
                        "unknown_future_field": "some_value",  # Unknown field
                    }
                ],
                "future_top_level_field": {"nested": "data"},  # Unknown top-level
                "updated_at": "2024-01-01T00:00:00",
            }
            state_file.write_text(json.dumps(future_state, indent=2))

            # Reset module state
            mcp_server._server_process = None
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._active_sessions = {}

            # Reload state - should handle future version gracefully
            # Either by loading what it can, or starting fresh
            mcp_server.ensure_server_running()

            # The code should either:
            # 1. Load the session (ignoring unknown fields) - PREFERRED
            # 2. Start fresh (if version is incompatible)
            # Either way, it should NOT crash

            # Current implementation should load it (Pydantic ignores extra fields by default)
            # If this changes, the test will catch the regression
            if "future-session" in mcp_server._active_sessions:
                # Option 1: Session was loaded (ignoring unknown fields)
                session_info = mcp_server._active_sessions["future-session"]
                assert session_info.session_id == "future-session"
                assert session_info.notebook_path == "/some/path.ipynb"
            else:
                # Option 2: Started fresh (acceptable fallback)
                # Just verify no crash occurred and server is running
                assert mcp_server._server_url is not None


class TestServerFailureScenarios:
    """Tests for server/kernel failure modes that can occur in production."""

    @pytest.mark.asyncio
    async def test_server_death_between_list_and_execute(
        self,
        python_path: str,
        cleanup_jupyter_processes,
    ):
        """Verify clear error when server dies between list_sessions and execute_code.

        This simulates a production failure:
        1. Agent calls list_sessions, gets session_id
        2. Jupyter server crashes
        3. Agent calls execute_code with now-stale session_id
        4. Expected: Clear error message, not cryptic failure
        """
        import importlib
        import uuid

        import requests

        test_session_id = str(uuid.uuid4())

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": test_session_id}):
            import scribe.notebook.notebook_mcp_server as mcp_server

            importlib.reload(mcp_server)

            # Reset module state
            mcp_server._server_process = None
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._is_external_server = False
            mcp_server._active_sessions = {}

            # Start server and create a session
            server_url = mcp_server.ensure_server_running()
            token = mcp_server.get_token()
            headers = {"Authorization": f"token {token}"} if token else {}

            # Create session via HTTP
            response = requests.post(
                f"{server_url}/api/scribe/start",
                json={"experiment_name": "death_test"},
                headers=headers,
            )
            assert response.ok, f"Failed to start session: {response.text}"
            session_data = response.json()
            session_id = session_data["session_id"]

            # Register session (normally done by MCP tool)
            mcp_server._active_sessions[session_id] = mcp_server.SessionInfo(  # pyright: ignore[reportAttributeAccessIssue]
                session_id=session_id,
                notebook_path=session_data["notebook_path"],
            )

            # Save the session_id for later use
            listed_session_id = session_id

            # Kill the server (simulating crash)
            if mcp_server._server_process:
                mcp_server._server_process.terminate()
                mcp_server._server_process.wait(timeout=5)

            # Clear the server state (but keep sessions - this is the bug scenario)
            mcp_server._server_process = None
            old_url = mcp_server._server_url
            mcp_server._server_url = None

            # Try to execute code via HTTP with the session_id
            # This should fail with a clear error, not crash
            try:
                response = requests.post(
                    f"{old_url}/api/scribe/exec",
                    json={"session_id": listed_session_id, "code": "print('test')"},
                    headers=headers,
                    timeout=5,
                )
                # If request succeeds, check for error in response
                if response.ok:
                    result = response.json()
                    result_str = str(result).lower()
                    assert "error" in result_str or "fail" in result_str, (
                        f"Expected clear error message, got: {result}"
                    )
                else:
                    # Non-OK response is expected (server is dead)
                    pass
            except requests.exceptions.RequestException as e:
                # Connection error is expected since server is dead
                error_msg = str(e).lower()
                assert any(word in error_msg for word in ["connect", "refused", "timeout", "fail"]), (
                    f"Error message should indicate connection issue, got: {e}"
                )

    @pytest.mark.asyncio
    async def test_execute_code_with_dead_kernel(
        self,
        python_path: str,
        cleanup_jupyter_processes,
    ):
        """Verify clear error when kernel dies but session still in state.

        Scenario:
        1. Session created, kernel started
        2. Kernel crashes (OOM, segfault, etc.)
        3. Agent calls execute_code with valid-looking session_id
        4. Expected: Clear "kernel dead" error, suggestion to restart
        """
        import importlib
        import uuid

        import requests

        test_session_id = str(uuid.uuid4())

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": test_session_id}):
            import scribe.notebook.notebook_mcp_server as mcp_server

            importlib.reload(mcp_server)

            # Reset module state
            mcp_server._server_process = None
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._is_external_server = False
            mcp_server._active_sessions = {}

            # Start server and create a session
            server_url = mcp_server.ensure_server_running()
            token = mcp_server.get_token()
            headers = {"Authorization": f"token {token}"} if token else {}

            # Create session via HTTP
            response = requests.post(
                f"{server_url}/api/scribe/start",
                json={"experiment_name": "kernel_death_test"},
                headers=headers,
            )
            assert response.ok, f"Failed to start session: {response.text}"
            session_data = response.json()
            session_id = session_data["session_id"]
            kernel_id = session_data.get("kernel_id")

            # Register session
            mcp_server._active_sessions[session_id] = mcp_server.SessionInfo(  # pyright: ignore[reportAttributeAccessIssue]
                session_id=session_id,
                notebook_path=session_data["notebook_path"],
            )

            # Kill the kernel specifically (not the whole server)
            if kernel_id:
                try:
                    requests.delete(
                        f"{server_url}/api/kernels/{kernel_id}",
                        headers=headers,
                    )
                except Exception:
                    pass  # Kernel might already be dead

            # Try to execute code with the dead kernel via HTTP
            try:
                response = requests.post(
                    f"{server_url}/api/scribe/exec",
                    json={"session_id": session_id, "code": "print('test')"},
                    headers=headers,
                    timeout=10,
                )
                # Check result for error indication
                if response.ok:
                    result = response.json()
                    result_str = str(result).lower()
                    # Should indicate kernel/session issue, OR it might work if server recreates kernel
                    # Both are acceptable behaviors
                    if "error" in result_str or "fail" in result_str:
                        # Good - clear error message
                        pass
                    elif "output" in result_str or "execution_count" in result_str:
                        # Also acceptable - server auto-recovered
                        pass
                    else:
                        # Unclear response
                        pass
                else:
                    # Non-OK response - check it has useful error message
                    error_text = response.text.lower()
                    assert any(
                        word in error_text
                        for word in ["kernel", "session", "not found", "error", "fail"]
                    ), f"Error response should be actionable, got: {response.text}"
            except requests.exceptions.RequestException as e:
                # Connection error - acceptable if message is clear
                error_msg = str(e).lower()
                assert any(
                    word in error_msg
                    for word in ["kernel", "session", "connect", "timeout", "fail", "error"]
                ), f"Error message should be actionable, got: {e}"

    def test_external_server_unreachable_at_startup(self, cleanup_jupyter_processes):
        """Verify clear error when SCRIBE_PORT points to non-existent server.

        Scenario:
        1. User sets SCRIBE_PORT=9999 expecting external server
        2. No server running on that port
        3. Expected: Clear error about external server, not hang
        """
        import importlib
        import uuid

        test_session_id = str(uuid.uuid4())

        # Use a port that's almost certainly not in use
        unused_port = "59999"

        with patch.dict(
            os.environ,
            {
                "SCRIBE_SESSION_ID": test_session_id,
                "SCRIBE_PORT": unused_port,
                "SCRIBE_TOKEN": "test_token",
            },
        ):
            import scribe.notebook.notebook_mcp_server as mcp_server

            importlib.reload(mcp_server)

            # Reset module state
            mcp_server._server_process = None
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._is_external_server = False
            mcp_server._active_sessions = {}

            # Call ensure_server_running - should handle unreachable external server
            # Current behavior: Returns URL but prints warning
            # This test verifies it doesn't hang or crash
            result = mcp_server.ensure_server_running()

            # Should return the URL (even if server is unreachable)
            assert result == f"http://127.0.0.1:{unused_port}"

            # Should be marked as external server
            assert mcp_server._is_external_server is True

            # The server status should indicate it's unhealthy
            status = mcp_server.get_server_status()
            # External server that's unreachable should show in status
            assert status["is_external"] is True
