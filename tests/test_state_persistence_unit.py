"""Unit tests for scribe state persistence (no real servers or agents).

These tests verify:
1. State file path generation and isolation
2. Server status checking logic
3. Response parsing helper functions
4. External server configuration
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests


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


class TestServerStatusChecks:
    """Unit tests for server status checking logic."""

    def test_check_jupyter_status_healthy(self):
        """Verify healthy server returns HEALTHY status."""
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


class TestExternalServer:
    """Tests for external server (SCRIBE_PORT/SCRIBE_TOKEN) functionality."""

    def test_scribe_token_env_var_is_used(self):
        """Verify SCRIBE_TOKEN environment variable is read for external servers."""
        import importlib

        with patch.dict(
            os.environ,
            {"SCRIBE_PORT": "9999", "SCRIBE_TOKEN": "external_test_token"},
        ):
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


class TestSessionIsolation:
    """Tests for session isolation via SCRIBE_SESSION_ID."""

    def test_different_session_ids_use_different_state_files(self):
        """Verify different SCRIBE_SESSION_IDs result in different state file paths."""
        from scribe.notebook.notebook_mcp_server import _get_state_file  # type: ignore[attr-defined]

        cwd = "/tmp/test_cwd"
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

        clean_env = {k: v for k, v in os.environ.items() if k != "SCRIBE_SESSION_ID"}
        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(RuntimeError, match="SCRIBE_SESSION_ID environment variable is required"):
                _get_state_file()


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


class TestStateFileMigration:
    """Tests for state file version migration (v1 → v2)."""

    def test_load_state_corrupted_json_returns_none(self):
        """Verify corrupted JSON state file returns None, doesn't crash."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_session_corrupted"

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = "{ invalid json }"
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is None, "Corrupted JSON should return None"

    def test_load_state_valid_json_returns_dict(self):
        """Verify valid JSON state file returns parsed dict."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_session_valid"
        state_data = {"version": 2, "server": {"port": 8888}}

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                import json
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result == state_data

    def test_load_state_missing_file_returns_none(self):
        """Verify missing state file returns None."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_session_missing"

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = False
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is None


class TestPortFinding:
    """Tests for find_safe_port() function."""

    def test_find_safe_port_returns_bindable_port(self):
        """Verify find_safe_port returns a port we can actually bind to."""
        from scribe.notebook._notebook_server_utils import find_safe_port
        import socket

        port = find_safe_port()
        assert port is not None, "find_safe_port should return a port"
        assert 35000 <= port <= 45000, "Port should be in expected range"

        # Verify we can actually bind to it
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))  # Should not raise

    def test_find_safe_port_custom_range(self):
        """Verify find_safe_port respects custom port range."""
        from scribe.notebook._notebook_server_utils import find_safe_port

        port = find_safe_port(start_port=40000, max_port=40100)
        assert port is not None
        assert 40000 <= port <= 40100


class TestExternalServerFailures:
    """Tests for external server (SCRIBE_PORT/SCRIBE_TOKEN) failure modes."""

    def test_external_server_unreachable_logs_warning(self):
        """Verify warning when SCRIBE_PORT is set but nothing is listening."""
        import importlib
        from io import StringIO

        with patch.dict(
            os.environ,
            {"SCRIBE_PORT": "59999", "SCRIBE_TOKEN": "test_token"},
            clear=False,
        ):
            # Mock check_jupyter_status to return UNREACHABLE
            with patch("scribe.notebook.notebook_mcp_server.check_jupyter_status") as mock_check:
                from scribe.notebook.notebook_mcp_server import ServerStatus
                mock_check.return_value = ServerStatus.UNREACHABLE

                import scribe.notebook.notebook_mcp_server as mcp_server
                importlib.reload(mcp_server)

                # Reset module state
                mcp_server._server_port = None
                mcp_server._server_url = None
                mcp_server._server_token = None
                mcp_server._is_external_server = False

                # Capture stderr to check for warning
                with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                    url = mcp_server.ensure_server_running()

                    # Should still return URL (external server mode)
                    assert url == "http://127.0.0.1:59999"
                    assert mcp_server._is_external_server is True

                    # Should have logged a warning
                    stderr_output = mock_stderr.getvalue()
                    assert "Warning" in stderr_output or "unreachable" in stderr_output.lower()

                # Clean up
                mcp_server._server_port = None
                mcp_server._server_url = None
                mcp_server._server_token = None
                mcp_server._is_external_server = False

    def test_external_server_unauthorized_logs_warning(self):
        """Verify warning when SCRIBE_TOKEN is invalid (401/403)."""
        import importlib
        from io import StringIO

        with patch.dict(
            os.environ,
            {"SCRIBE_PORT": "59999", "SCRIBE_TOKEN": "wrong_token"},
            clear=False,
        ):
            with patch("scribe.notebook.notebook_mcp_server.check_jupyter_status") as mock_check:
                from scribe.notebook.notebook_mcp_server import ServerStatus
                mock_check.return_value = ServerStatus.UNAUTHORIZED

                import scribe.notebook.notebook_mcp_server as mcp_server
                importlib.reload(mcp_server)

                mcp_server._server_port = None
                mcp_server._server_url = None
                mcp_server._server_token = None
                mcp_server._is_external_server = False

                with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                    url = mcp_server.ensure_server_running()

                    assert url == "http://127.0.0.1:59999"
                    stderr_output = mock_stderr.getvalue()
                    assert "Warning" in stderr_output or "unauthorized" in stderr_output.lower()

                mcp_server._server_port = None
                mcp_server._server_url = None
                mcp_server._server_token = None
                mcp_server._is_external_server = False


class TestCleanupHandlers:
    """Tests for cleanup_server() and related cleanup logic."""

    def test_cleanup_scribe_server_handles_none_process(self):
        """Verify cleanup handles None process gracefully."""
        from scribe.notebook._notebook_server_utils import cleanup_scribe_server

        # Should not raise (function has `if process:` check at runtime)
        cleanup_scribe_server(None)  # type: ignore[arg-type]

    def test_cleanup_scribe_server_terminates_running_process(self):
        """Verify cleanup terminates a running process."""
        from scribe.notebook._notebook_server_utils import cleanup_scribe_server

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running
        mock_process.wait.return_value = 0

        cleanup_scribe_server(mock_process)

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called()

    def test_cleanup_scribe_server_kills_stubborn_process(self):
        """Verify cleanup kills process that doesn't terminate gracefully."""
        import subprocess
        from scribe.notebook._notebook_server_utils import cleanup_scribe_server

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running
        mock_process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), 0]

        cleanup_scribe_server(mock_process)

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
