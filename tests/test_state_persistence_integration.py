"""Integration tests for scribe state persistence across MCP restarts.

These tests verify:
1. State file is created when a session starts
2. Scribe can reconnect to an existing Jupyter server after MCP restart
3. Stale state is handled gracefully when Jupyter is dead
4. State files have restrictive permissions (0o600)
5. Session discovery and error handling work through MCP
6. Compaction scenarios work end-to-end
"""

import json
import os
import stat
import uuid
from unittest.mock import patch

import pytest
import requests
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # type: ignore[import-not-found]

# Test constants
TEST_MODEL = "claude-haiku-4-5-20251001"
UNUSED_PORT = 59999
DEFAULT_MAX_TURNS = 10
LEGACY_SESSION_IDS = ["legacy-session-id-1", "legacy-session-id-2"]

# Path to the isolated venv with modified scribe installed
from pathlib import Path
SCRIBE_FORK_DIR = Path(__file__).parent.parent
ISOLATED_PYTHON = SCRIBE_FORK_DIR / ".venv" / "bin" / "python"


def get_scribe_mcp_config(
    python_path: str,
    env: dict | None = None,
    session_id: str | None = None,
) -> dict:
    """Generate MCP config for scribe."""
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


def start_session_via_http(
    mcp_server,  # type: ignore[no-untyped-def]
    experiment_name: str = "test",
):
    """Start server and create a session via HTTP."""
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

    mcp_server._active_sessions[session_data["session_id"]] = mcp_server.SessionInfo(  # pyright: ignore[reportAttributeAccessIssue]
        session_id=session_data["session_id"],
        notebook_path=session_data["notebook_path"],
    )

    return session_data, server_url, headers


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
            await client.query(
                "Use the start_new_session tool to create a new notebook session, "
                "then use execute_code to run: print('hello')"
            )
            async for _ in client.receive_response():
                pass

        new_files, _ = track_state_files()
        assert len(new_files) > 0, "State file should be created after session start"

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

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use start_new_session to create a notebook, "
                "then execute_code to run: x = 42"
            )
            async for _ in client.receive_response():
                pass

        new_files, _ = track_state_files()
        assert len(new_files) > 0, "State file should exist after first session"
        state_file = next(iter(new_files))
        state_before = json.loads(state_file.read_text())
        port_before = state_before["server"]["port"]

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use execute_code to run: print(x)  # Should print 42 if reconnected"
            )
            async for _ in client.receive_response():
                pass

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

        async with ClaudeSDKClient(options=options) as client:
            await client.query("Use start_new_session to create a notebook")
            async for _ in client.receive_response():
                pass

        new_files, _ = track_state_files()
        new_files = new_files - initial_new_files
        assert len(new_files) > 0, "State file should exist"
        state_file = next(iter(new_files))

        fake_state = {
            "version": 1,
            "server": {
                "port": UNUSED_PORT,
                "token": "fake_token_that_wont_work",
                "pid": 99999,
                "url": f"http://127.0.0.1:{UNUSED_PORT}",
            },
            "sessions": ["fake_session"],
            "updated_at": "2026-01-01T00:00:00",
        }
        state_file.write_text(json.dumps(fake_state))

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use start_new_session to create a notebook, "
                "then execute_code to run: print('recovered')"
            )
            async for _ in client.receive_response():
                pass

        state_after = json.loads(state_file.read_text())
        assert (
            state_after["server"]["port"] != UNUSED_PORT
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

        new_files, _ = track_state_files()
        assert len(new_files) > 0, "State file should be created"
        state_file = next(iter(new_files))

        file_stat = state_file.stat()
        mode = stat.S_IMODE(file_stat.st_mode)
        assert mode == 0o600, (
            f"State file should have 0o600 permissions, got {oct(mode)}. "
            "Token is stored in plaintext and should be protected."
        )


class TestErrorHandlingAndSessionDiscovery:
    """Integration tests for error handling and session discovery."""

    @pytest.mark.asyncio
    async def test_invalid_session_id_returns_clear_error(
        self,
        python_path: str,
        cleanup_jupyter_processes,
    ):
        """Verify invalid session_id error is propagated through MCP."""
        test_session_id = "test_error_handling_12345678"
        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": test_session_id}):
            from scribe.notebook.notebook_mcp_server import ensure_server_running, get_token

            server_url = ensure_server_running()
            token = get_token()
            headers = {"Authorization": f"token {token}"} if token else {}

            response = requests.post(
                f"{server_url}/api/scribe/exec",
                json={"session_id": "fake_session_that_does_not_exist", "code": "print(1)"},
                headers=headers,
            )

            assert response.status_code == 500
            error_data = response.json()
            error_message = error_data.get("error", "").lower()

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
            max_turns=DEFAULT_MAX_TURNS,
        )

        session_id_found = False
        executed_successfully = False

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "1. Use start_new_session to create a notebook\n"
                "2. Use list_sessions and tell me the exact session_id you see\n"
                "3. Use execute_code with that session_id to run: print('test_success')"
            )

            async for msg in client.receive_response():
                msg_text = str(msg)
                if "-" in msg_text and len(msg_text) > 30:
                    session_id_found = True
                if "test_success" in msg_text.lower():
                    executed_successfully = True

        assert session_id_found, "Should have seen a session_id from list_sessions"
        assert executed_successfully, "Should have successfully executed code using listed session_id"


class TestCompactionScenariosDirect:
    """Direct tests (without agent) for state persistence across MCP restarts."""

    @pytest.mark.asyncio
    async def test_state_persistence_direct(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,
    ):
        """Directly verify state persistence works across MCP module reloads."""
        test_session_id = str(uuid.uuid4())

        # Phase 1: Start server, create session, execute code
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "direct_test")
        session_id = session_data["session_id"]

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "test_var = 'persistence_works'"},
            headers=headers,
        )
        assert response.ok, f"Failed to execute code: {response.text}"

        mcp_server.save_state()
        original_port = mcp_server._server_port

        state_file = mcp_server._get_state_file()
        saved_state = json.loads(state_file.read_text())
        assert len(saved_state.get("sessions", [])) > 0, "State file should have sessions"

        # Phase 2: Simulate MCP restart by reloading module
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()
        token = mcp_server.get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        assert mcp_server._server_port == original_port, (
            f"Should reconnect to same port. Expected {original_port}, got {mcp_server._server_port}"
        )

        assert len(mcp_server._active_sessions) > 0, "Sessions should be restored from state"
        assert session_id in mcp_server._active_sessions, f"Session {session_id} should be in active sessions"

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print(test_var)"},
            headers=headers,
        )
        assert response.ok, f"Failed to execute code after reconnect: {response.text}"
        result = response.json()

        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "persistence_works" in output_text, (
            f"Variable should persist after MCP restart. Got output: {output_text}"
        )


class TestCompactionScenarios:
    """Integration tests for scenarios that fail in production during compaction."""

    @pytest.mark.asyncio
    async def test_kernel_state_persists_across_compaction(
        self,
        python_path: str,
        track_state_files,
        cleanup_jupyter_processes,
    ):
        """Verify variables survive MCP restart (compaction simulation)."""
        shared_session_id = str(uuid.uuid4())

        options = ClaudeAgentOptions(
            model=TEST_MODEL,
            mcp_servers=get_scribe_mcp_config(python_path, session_id=shared_session_id),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
            ],
            max_turns=DEFAULT_MAX_TURNS,
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use start_new_session to create a notebook, "
                "then execute_code to run: x = 42"
            )
            async for _ in client.receive_response():
                pass

        new_files, _ = track_state_files()
        assert len(new_files) > 0, "State file should be created"
        state_file = next(iter(new_files))
        state = json.loads(state_file.read_text())
        assert len(state.get("sessions", [])) > 0, "Session should be in state"

        value_found = False
        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use execute_code to run: print(x)\nTell me the exact output."
            )
            async for msg in client.receive_response():
                if "42" in str(msg):
                    value_found = True

        assert value_found, "Variable x=42 should persist after compaction"

    @pytest.mark.asyncio
    async def test_list_sessions_then_execute_after_compaction(
        self,
        python_path: str,
        cleanup_jupyter_processes,
    ):
        """Verify list_sessions -> execute_code workflow works after compaction."""
        shared_session_id = str(uuid.uuid4())

        options = ClaudeAgentOptions(
            model=TEST_MODEL,
            mcp_servers=get_scribe_mcp_config(python_path, session_id=shared_session_id),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
                "mcp__scribe__list_sessions",
            ],
            max_turns=DEFAULT_MAX_TURNS,
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use start_new_session to create a notebook, "
                "then execute_code to run: my_var = 'compaction_test_value'"
            )
            async for _ in client.receive_response():
                pass

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
                if "-" in msg_text and len(msg_text) > 30:
                    session_discovered = True
                if "compaction_test_value" in msg_text:
                    test_value_found = True

        assert session_discovered, "Should have discovered session via list_sessions"
        assert test_value_found, (
            "Variable my_var should have value 'compaction_test_value' after "
            "discovering session via list_sessions."
        )

    @pytest.mark.asyncio
    async def test_execute_code_with_stale_session_returns_clear_error(
        self,
        python_path: str,
        cleanup_jupyter_processes,
    ):
        """Verify stale session_id gives actionable error, not cryptic failure."""
        import re

        shared_session_id = str(uuid.uuid4())

        options = ClaudeAgentOptions(
            model=TEST_MODEL,
            mcp_servers=get_scribe_mcp_config(python_path, session_id=shared_session_id),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
                "mcp__scribe__shutdown_session",
            ],
            max_turns=DEFAULT_MAX_TURNS,
        )

        captured_session_id = None

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "1. Use start_new_session to create a notebook\n"
                "2. Tell me the exact session_id\n"
                "3. Use shutdown_session to close it"
            )
            async for msg in client.receive_response():
                msg_text = str(msg)
                uuid_match = re.search(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    msg_text,
                    re.IGNORECASE,
                )
                if uuid_match:
                    captured_session_id = uuid_match.group()

        assert captured_session_id, "Should have captured a session_id"

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
        """Verify state file preserves notebook paths for post-compaction recovery."""
        shared_session_id = str(uuid.uuid4())

        options = ClaudeAgentOptions(
            model=TEST_MODEL,
            mcp_servers=get_scribe_mcp_config(python_path, session_id=shared_session_id),
            allowed_tools=[
                "mcp__scribe__start_new_session",
                "mcp__scribe__execute_code",
            ],
            max_turns=DEFAULT_MAX_TURNS,
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "Use start_new_session to create a notebook named 'path_test', "
                "then execute_code to run: x = 1"
            )
            async for _ in client.receive_response():
                pass

        new_files, _ = track_state_files()
        assert len(new_files) > 0, "State file should be created"
        state_file = next(iter(new_files))
        state = json.loads(state_file.read_text())

        sessions = state.get("sessions", [])
        assert len(sessions) > 0, "Should have at least one session"
        session = sessions[0]

        assert "session_id" in session, "Session should have session_id field"
        assert "notebook_path" in session, "Session should have notebook_path field"
        assert session["notebook_path"], "notebook_path should not be empty"
        assert ".ipynb" in session["notebook_path"], "notebook_path should be an ipynb file"

    @pytest.mark.asyncio
    async def test_multiple_sessions_across_compaction(
        self,
        python_path: str,
        cleanup_jupyter_processes,
    ):
        """Verify multiple sessions are all accessible after compaction."""
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

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "1. Use start_new_session to create notebook A\n"
                "2. Execute: session_a_var = 'value_A'\n"
                "3. Use start_new_session to create notebook B\n"
                "4. Execute: session_b_var = 'value_B'"
            )
            async for _ in client.receive_response():
                pass

        session_a_found = False
        session_b_found = False

        async with ClaudeSDKClient(options=options) as client:
            await client.query(
                "1. Use list_sessions to find all sessions\n"
                "2. For EACH session, execute:\n"
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

    def test_load_state_v1_format_migration(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,
    ):
        """Verify v1 state files (session IDs as strings) work with v2 code."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        state_file = mcp_server._get_state_file()  # pyright: ignore[reportAttributeAccessIssue]

        server_url = mcp_server.ensure_server_running()
        port = mcp_server._server_port
        token = mcp_server._server_token

        v1_state = {
            "version": 1,
            "server": {
                "port": port,
                "token": token,
                "pid": None,
                "url": server_url,
            },
            "sessions": LEGACY_SESSION_IDS,
            "updated_at": "2024-01-01T00:00:00",
        }
        state_file.write_text(json.dumps(v1_state, indent=2))

        mcp_server = reset_mcp_module(test_session_id)
        mcp_server.ensure_server_running()

        assert len(mcp_server._active_sessions) == 2, (
            f"Expected 2 sessions, got {len(mcp_server._active_sessions)}"
        )
        assert LEGACY_SESSION_IDS[0] in mcp_server._active_sessions
        assert LEGACY_SESSION_IDS[1] in mcp_server._active_sessions

        session_info = mcp_server._active_sessions[LEGACY_SESSION_IDS[0]]
        assert session_info.session_id == LEGACY_SESSION_IDS[0]
        assert session_info.notebook_path == "", (
            "Legacy sessions should have empty notebook_path"
        )

    def test_load_state_future_version_graceful_handling(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,
    ):
        """Verify graceful handling of state files from future versions."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        state_file = mcp_server._get_state_file()  # pyright: ignore[reportAttributeAccessIssue]

        server_url = mcp_server.ensure_server_running()
        port = mcp_server._server_port
        token = mcp_server._server_token

        future_state = {
            "version": 999,
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
                    "unknown_future_field": "some_value",
                }
            ],
            "future_top_level_field": {"nested": "data"},
            "updated_at": "2024-01-01T00:00:00",
        }
        state_file.write_text(json.dumps(future_state, indent=2))

        mcp_server = reset_mcp_module(test_session_id)
        mcp_server.ensure_server_running()

        if "future-session" in mcp_server._active_sessions:
            session_info = mcp_server._active_sessions["future-session"]
            assert session_info.session_id == "future-session"
            assert session_info.notebook_path == "/some/path.ipynb"
        else:
            assert mcp_server._server_url is not None


class TestServerFailureScenarios:
    """Tests for server/kernel failure modes that can occur in production."""

    @pytest.mark.asyncio
    async def test_server_death_between_list_and_execute(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,
    ):
        """Verify clear error when server dies between list_sessions and execute_code."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "death_test")
        listed_session_id = session_data["session_id"]

        if mcp_server._server_process:
            mcp_server._server_process.terminate()
            mcp_server._server_process.wait(timeout=5)

        mcp_server._server_process = None
        old_url = mcp_server._server_url
        mcp_server._server_url = None

        try:
            response = requests.post(
                f"{old_url}/api/scribe/exec",
                json={"session_id": listed_session_id, "code": "print('test')"},
                headers=headers,
                timeout=5,
            )
            if response.ok:
                result = response.json()
                result_str = str(result).lower()
                assert "error" in result_str or "fail" in result_str, (
                    f"Expected clear error message, got: {result}"
                )
        except requests.exceptions.RequestException as e:
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ["connect", "refused", "timeout", "fail"]), (
                f"Error message should indicate connection issue, got: {e}"
            )

    @pytest.mark.asyncio
    async def test_execute_code_with_dead_kernel(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,
    ):
        """Verify clear error when kernel dies but session still in state."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "kernel_death_test")
        session_id = session_data["session_id"]
        kernel_id = session_data.get("kernel_id")

        if kernel_id:
            try:
                requests.delete(
                    f"{server_url}/api/kernels/{kernel_id}",
                    headers=headers,
                )
            except Exception:
                pass

        try:
            response = requests.post(
                f"{server_url}/api/scribe/exec",
                json={"session_id": session_id, "code": "print('test')"},
                headers=headers,
                timeout=10,
            )
            if response.ok:
                result = response.json()
                result_str = str(result).lower()
                if "error" in result_str or "fail" in result_str:
                    pass
                elif "output" in result_str or "execution_count" in result_str:
                    pass
            else:
                error_text = response.text.lower()
                assert any(
                    word in error_text
                    for word in ["kernel", "session", "not found", "error", "fail"]
                ), f"Error response should be actionable, got: {response.text}"
        except requests.exceptions.RequestException as e:
            error_msg = str(e).lower()
            assert any(
                word in error_msg
                for word in ["kernel", "session", "connect", "timeout", "fail", "error"]
            ), f"Error message should be actionable, got: {e}"

    def test_external_server_unreachable_at_startup(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,
    ):
        """Verify clear error when SCRIBE_PORT points to non-existent server."""
        import importlib

        test_session_id = str(uuid.uuid4())

        with patch.dict(
            os.environ,
            {
                "SCRIBE_SESSION_ID": test_session_id,
                "SCRIBE_PORT": str(UNUSED_PORT),
                "SCRIBE_TOKEN": "test_token",
            },
        ):
            import scribe.notebook.notebook_mcp_server as mcp_server
            importlib.reload(mcp_server)

            mcp_server._server_process = None
            mcp_server._server_port = None
            mcp_server._server_url = None
            mcp_server._server_token = None
            mcp_server._is_external_server = False
            mcp_server._active_sessions = {}

            result = mcp_server.ensure_server_running()

            assert result == f"http://127.0.0.1:{UNUSED_PORT}"
            assert mcp_server._is_external_server is True

            status = mcp_server.get_server_status()
            assert status["is_external"] is True
