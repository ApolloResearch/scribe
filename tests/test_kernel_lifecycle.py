"""Integration tests for kernel lifecycle edge cases.

These tests verify scribe handles kernel state transitions gracefully:
1. Execute immediately after session start
2. Rapid fire requests
3. Kernel busy handling
4. Kernel death scenarios
"""

import uuid

import requests

from tests.conftest import start_session_via_http


class TestKernelStartup:
    """Tests for kernel startup edge cases."""

    def test_execute_immediately_after_session_start(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify immediate execution after session start works (kernel readiness)."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "immediate_test")
        session_id = session_data["session_id"]

        # Execute immediately - no delay
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print('immediate')"},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Immediate execution should work: {response.text}"
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "immediate" in output_text, f"Should see output, got: {output_text}"

    def test_multiple_rapid_executions(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify multiple rapid execute requests in sequence work."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "rapid_test")
        session_id = session_data["session_id"]

        # Execute 5 times rapidly
        for i in range(5):
            response = requests.post(
                f"{server_url}/api/scribe/exec",
                json={"session_id": session_id, "code": f"x_{i} = {i}"},
                headers=headers,
                timeout=30,
            )
            assert response.ok, f"Execution {i} should work: {response.text}"

        # Verify all variables exist
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print(x_0, x_1, x_2, x_3, x_4)"},
            headers=headers,
            timeout=30,
        )
        assert response.ok
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "0 1 2 3 4" in output_text, f"All vars should exist, got: {output_text}"


class TestKernelBusy:
    """Tests for kernel busy scenarios."""

    def test_execute_after_slow_execution(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify execution works after a slow operation completes."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "slow_test")
        session_id = session_data["session_id"]

        # Execute slow operation (2 seconds)
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "import time; time.sleep(2); result = 'done'"},
            headers=headers,
            timeout=30,
        )
        assert response.ok, f"Slow execution should complete: {response.text}"

        # Execute immediately after - should work
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print(result)"},
            headers=headers,
            timeout=30,
        )
        assert response.ok
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "done" in output_text


class TestKernelErrors:
    """Tests for kernel error handling."""

    def test_kernel_recovers_after_exception(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify kernel works after code raises an exception."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "exception_test")
        session_id = session_data["session_id"]

        # Execute code that raises an exception
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "raise ValueError('test error')"},
            headers=headers,
            timeout=30,
        )
        assert response.ok
        result = response.json()
        outputs = result.get("outputs", [])
        assert any(o.get("output_type") == "error" for o in outputs), "Should have error output"

        # Kernel should still work after exception
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print('recovered')"},
            headers=headers,
            timeout=30,
        )
        assert response.ok
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "recovered" in output_text

    def test_kernel_recovers_after_syntax_error(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify kernel works after code has syntax error."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "syntax_test")
        session_id = session_data["session_id"]

        # Execute code with syntax error
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "def broken("},
            headers=headers,
            timeout=30,
        )
        assert response.ok
        result = response.json()
        outputs = result.get("outputs", [])
        assert any(o.get("output_type") == "error" for o in outputs)

        # Kernel should still work
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print('still works')"},
            headers=headers,
            timeout=30,
        )
        assert response.ok
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "still works" in output_text

    def test_kernel_state_preserved_after_error(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify variables defined before error are preserved."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "state_test")
        session_id = session_data["session_id"]

        # Define a variable
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "preserved_var = 'keep me'"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        # Cause an error
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "1/0"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        # Variable should still exist
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print(preserved_var)"},
            headers=headers,
            timeout=30,
        )
        assert response.ok
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "keep me" in output_text


class TestKernelImports:
    """Tests for kernel import handling."""

    def test_import_standard_library(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify standard library imports work."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "import_test")
        session_id = session_data["session_id"]

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "import json; print(json.dumps({'a': 1}))"},
            headers=headers,
            timeout=30,
        )
        assert response.ok
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert '{"a": 1}' in output_text

    def test_import_nonexistent_module(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify importing nonexistent module gives clear error."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "bad_import_test")
        session_id = session_data["session_id"]

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "import nonexistent_module_12345"},
            headers=headers,
            timeout=30,
        )
        assert response.ok
        result = response.json()
        outputs = result.get("outputs", [])
        error_outputs = [o for o in outputs if o.get("output_type") == "error"]
        assert len(error_outputs) > 0, "Should have import error"
        assert "ModuleNotFoundError" in error_outputs[0].get("ename", "")
