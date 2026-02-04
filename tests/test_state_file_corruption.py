"""Unit tests for state file corruption scenarios.

These tests verify scribe handles corrupted/malformed state files gracefully:
1. Truncated JSON from interrupted write
2. Empty files (0 bytes)
3. Valid JSON with wrong schema
4. Null values where objects expected
5. Extra unexpected fields (future-proofing)
6. Mixed session formats (v1 strings + v2 dicts)
"""

import json
import os
from unittest.mock import MagicMock, patch

class TestStateFileTruncation:
    """Tests for truncated/partial JSON state files."""

    def test_state_file_truncated_mid_json(self):
        """Verify truncated JSON (power failure) returns None, doesn't crash."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_truncated"
        # Simulate truncated write: valid JSON start, cut off mid-value
        truncated_json = '{"version": 2, "server": {"port": 35'

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = truncated_json
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is None, "Truncated JSON should return None"

    def test_state_file_truncated_no_closing_brace(self):
        """Verify JSON without closing brace returns None."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_truncated_brace"
        # Missing final closing brace
        truncated_json = '{"version": 2, "server": {"port": 35000}'

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = truncated_json
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is None, "JSON without closing brace should return None"


class TestStateFileEmpty:
    """Tests for empty state files."""

    def test_state_file_empty_zero_bytes(self):
        """Verify empty file (0 bytes) returns None, doesn't crash."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_empty"

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = ""
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is None, "Empty file should return None"

    def test_state_file_only_whitespace(self):
        """Verify file with only whitespace returns None."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_whitespace"

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = "   \n\t  \n  "
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is None, "Whitespace-only file should return None"


class TestStateFileWrongSchema:
    """Tests for valid JSON with wrong/missing schema."""

    def test_state_file_missing_server_key(self):
        """Verify JSON missing 'server' key is handled."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_no_server"
        # Valid JSON but missing required 'server' key
        state_data = {"version": 2, "sessions": []}

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                # Should return the dict (caller handles missing keys)
                # or return None - either is acceptable
                # What we DON'T want is an exception
                assert result is None or isinstance(result, dict)

    def test_state_file_missing_version_key(self):
        """Verify JSON missing 'version' key is handled."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_no_version"
        state_data = {"server": {"port": 35000, "token": "abc"}, "sessions": []}

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                # Should return dict or None, not crash
                assert result is None or isinstance(result, dict)

    def test_state_file_wrong_type_for_server(self):
        """Verify 'server' as string instead of object is handled."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_server_string"
        state_data = {"version": 2, "server": "not_an_object", "sessions": []}

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                # Should handle gracefully
                assert result is None or isinstance(result, dict)


class TestStateFileNullValues:
    """Tests for JSON with null values."""

    def test_state_file_server_is_null(self):
        """Verify null server value is handled."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_null_server"
        state_data = {"version": 2, "server": None, "sessions": []}

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                # Should handle gracefully - either return dict or None
                assert result is None or isinstance(result, dict)

    def test_state_file_sessions_is_null(self):
        """Verify null sessions value is handled."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_null_sessions"
        state_data = {"version": 2, "server": {"port": 35000}, "sessions": None}

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is None or isinstance(result, dict)

    def test_state_file_port_is_null(self):
        """Verify null port value is handled."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_null_port"
        state_data = {"version": 2, "server": {"port": None, "token": "abc"}, "sessions": []}

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is None or isinstance(result, dict)


class TestStateFileExtraFields:
    """Tests for future-proofing with unknown fields."""

    def test_state_file_extra_top_level_fields(self):
        """Verify extra top-level fields are ignored (future versions)."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_extra_fields"
        state_data = {
            "version": 2,
            "server": {"port": 35000, "token": "abc"},
            "sessions": [],
            "future_field": "some_value",
            "another_future_field": {"nested": "data"},
        }

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is not None, "Extra fields should not cause load_state to fail"
                assert result["version"] == 2
                assert result["server"]["port"] == 35000

    def test_state_file_extra_server_fields(self):
        """Verify extra fields in server object are ignored."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_extra_server"
        state_data = {
            "version": 2,
            "server": {
                "port": 35000,
                "token": "abc",
                "future_server_field": "value",
            },
            "sessions": [],
        }

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is not None
                assert result["server"]["port"] == 35000

    def test_state_file_extra_session_fields(self):
        """Verify extra fields in session objects are ignored."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_extra_session"
        state_data = {
            "version": 2,
            "server": {"port": 35000, "token": "abc"},
            "sessions": [
                {
                    "session_id": "sess-1",
                    "notebook_path": "/path/to/nb.ipynb",
                    "future_session_field": "value",
                }
            ],
        }

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is not None
                assert len(result["sessions"]) == 1


class TestStateFileMixedFormats:
    """Tests for mixed v1/v2 session formats."""

    def test_state_file_sessions_mixed_strings_and_dicts(self):
        """Verify mixed v1 (strings) and v2 (dicts) in sessions array."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_mixed_sessions"
        # Mix of v1 format (string) and v2 format (dict)
        state_data = {
            "version": 2,
            "server": {"port": 35000, "token": "abc"},
            "sessions": [
                "legacy-session-id",  # v1 format
                {"session_id": "new-session", "notebook_path": "/path.ipynb"},  # v2 format
            ],
        }

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                # Should handle gracefully
                assert result is None or isinstance(result, dict)

    def test_state_file_sessions_all_strings_v1(self):
        """Verify pure v1 format (all strings) is handled."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_v1_sessions"
        state_data = {
            "version": 1,
            "server": {"port": 35000, "token": "abc"},
            "sessions": ["session-1", "session-2", "session-3"],
        }

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.return_value = json.dumps(state_data)
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is not None
                assert result["version"] == 1


class TestStateFileReadErrors:
    """Tests for file system errors during state file read."""

    def test_state_file_permission_denied(self):
        """Verify permission denied on read returns None."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_permission"

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.side_effect = PermissionError("Permission denied")
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is None, "Permission error should return None"

    def test_state_file_io_error(self):
        """Verify generic IO error on read returns None."""
        from scribe.notebook.notebook_mcp_server import load_state

        session_id = "test_io_error"

        with patch.dict(os.environ, {"SCRIBE_SESSION_ID": session_id}):
            with patch("scribe.notebook.notebook_mcp_server._get_state_file") as mock_get_file:
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_path.read_text.side_effect = IOError("Disk read error")
                mock_get_file.return_value = mock_path

                result = load_state()
                assert result is None, "IO error should return None"
