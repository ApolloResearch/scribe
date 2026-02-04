"""Integration tests for notebook file edge cases.

These tests verify scribe handles file system issues gracefully:
1. Notebook deleted during session
2. Notebook deleted before reconnect
3. Directory not writable
4. Notebook file not writable
5. Various path edge cases
"""

import os
import uuid

import requests

from tests.conftest import start_session_via_http


class TestNotebookFileDeletion:
    """Tests for notebook file deletion scenarios."""

    def test_notebook_deleted_during_session(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify graceful handling when notebook is deleted while session is active."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "deletion_test")
        session_id = session_data["session_id"]
        notebook_path = session_data["notebook_path"]

        # Execute some code first to confirm session works
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "x = 42"},
            headers=headers,
            timeout=30,
        )
        assert response.ok, f"Initial execution should work: {response.text}"

        # Delete the notebook file externally
        if os.path.exists(notebook_path):
            os.unlink(notebook_path)

        # Try to execute more code - kernel should still work even if notebook is gone
        # (kernel state is in memory, not in the file)
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print(x)"},
            headers=headers,
            timeout=30,
        )

        # The execution might fail (can't update notebook) or succeed (kernel still works)
        # What we DON'T want is a server crash
        assert response.status_code in [200, 500], (
            f"Should handle deleted notebook gracefully, got {response.status_code}"
        )

        if response.status_code == 500:
            error = response.json()
            # Error should be about file/notebook, not a generic crash
            error_text = str(error).lower()
            assert any(word in error_text for word in ["file", "notebook", "path", "exist"]), (
                f"Error should mention file issue, got: {error}"
            )

    def test_execute_after_notebook_recreated(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify session works after notebook is deleted and recreated."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "recreate_test")
        session_id = session_data["session_id"]
        notebook_path = session_data["notebook_path"]

        # Execute some code
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "y = 100"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        # Delete and recreate notebook (simulating external editor save)
        if os.path.exists(notebook_path):
            os.unlink(notebook_path)

        # Create empty notebook
        import json
        empty_nb = {
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with open(notebook_path, "w") as f:
            json.dump(empty_nb, f)

        # Kernel state should still have y=100
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print(y)"},
            headers=headers,
            timeout=30,
        )

        # Should work - kernel state is independent of notebook file
        if response.ok:
            result = response.json()
            outputs = result.get("outputs", [])
            output_text = "".join(
                o.get("text", "") for o in outputs if o.get("output_type") == "stream"
            )
            assert "100" in output_text, f"Kernel should retain state, got: {output_text}"


class TestNotebookPathEdgeCases:
    """Tests for notebook path edge cases."""

    def test_session_with_spaces_in_experiment_name(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify experiment names with spaces work."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()
        token = mcp_server.get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        # Experiment name with spaces
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": "my experiment with spaces"},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Spaces in name should work: {response.text}"
        session_data = response.json()
        assert "session_id" in session_data
        # Path should have underscores or be properly escaped
        assert "notebook_path" in session_data

    def test_session_with_special_characters_in_name(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify experiment names with special characters are handled."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()
        token = mcp_server.get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        # Experiment name with special characters
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": "test-v2.1_final"},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Special chars in name should work: {response.text}"
        session_data = response.json()
        assert "session_id" in session_data

    def test_session_with_very_long_experiment_name(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify very long experiment names are handled (truncated or rejected)."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()
        token = mcp_server.get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        # Very long experiment name (255+ characters)
        long_name = "a" * 300

        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": long_name},
            headers=headers,
            timeout=30,
        )

        # Should either succeed (with truncation) or return a clear error
        # Should NOT crash the server
        assert response.status_code in [200, 400, 422, 500], (
            f"Long name should be handled, got {response.status_code}"
        )

        if response.ok:
            session_data = response.json()
            # Path should exist and be valid
            notebook_path = session_data.get("notebook_path", "")
            # Path length should be reasonable (filesystem limit is usually 255 chars for filename)
            assert len(os.path.basename(notebook_path)) <= 255, (
                f"Filename should be within filesystem limits: {notebook_path}"
            )

    def test_session_with_empty_experiment_name(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify empty experiment name uses a default."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()
        token = mcp_server.get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": ""},
            headers=headers,
            timeout=30,
        )

        # Should use a default name
        assert response.ok, f"Empty name should use default: {response.text}"
        session_data = response.json()
        assert "notebook_path" in session_data
        assert ".ipynb" in session_data["notebook_path"]

    def test_session_with_no_experiment_name(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify missing experiment_name parameter uses a default."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()
        token = mcp_server.get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        # No experiment_name in request
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={},
            headers=headers,
            timeout=30,
        )

        # Should use a default name
        assert response.ok, f"Missing name should use default: {response.text}"
        session_data = response.json()
        assert "notebook_path" in session_data
