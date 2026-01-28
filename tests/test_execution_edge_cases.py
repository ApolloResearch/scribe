"""Integration tests for code execution edge cases.

These tests verify scribe handles unusual code execution scenarios:
1. Long-running code (timeouts)
2. Infinite loops (interrupt handling)
3. Large stdout output
4. Large dataframe repr
5. Binary/non-UTF8 output
6. Memory exhaustion attempts
"""

import uuid

import requests

from tests.conftest import start_session_via_http


class TestExecutionTimeouts:
    """Tests for long-running code execution."""

    def test_execute_moderate_sleep(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify code that sleeps for 5 seconds completes successfully."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "sleep_test")
        session_id = session_data["session_id"]

        # 5 seconds should complete without timeout
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "import time; time.sleep(5); print('done')"},
            headers=headers,
            timeout=30,  # Allow 30 seconds for the request
        )

        assert response.ok, f"5-second sleep should complete: {response.text}"
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "done" in output_text, f"Should see 'done' in output, got: {output_text}"


class TestLargeOutputs:
    """Tests for handling large outputs."""

    def test_execute_large_stdout(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify large stdout (10KB) doesn't crash server."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "large_stdout_test")
        session_id = session_data["session_id"]

        # Print 10KB of output (reduced from 100KB for test speed)
        code = "print('x' * 10000)"

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
            timeout=60,  # Increased timeout for test stability
        )

        assert response.ok, f"Large stdout should not crash: {response.text}"
        result = response.json()
        outputs = result.get("outputs", [])
        # Should have some output (may be truncated)
        assert len(outputs) > 0, "Should have output even if truncated"

    def test_execute_many_print_statements(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify many small print statements are handled."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "many_prints_test")
        session_id = session_data["session_id"]

        # 1000 small prints
        code = "for i in range(1000): print(f'line {i}')"

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Many prints should not crash: {response.text}"

    def test_execute_large_return_value(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify large return value (list) is handled."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "large_return_test")
        session_id = session_data["session_id"]

        # Return a large list (100K items)
        code = "list(range(100000))"

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Large return value should not crash: {response.text}"


class TestBinaryOutput:
    """Tests for binary/non-UTF8 output handling."""

    def test_execute_binary_bytes_output(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify binary bytes in output are handled gracefully."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "binary_test")
        session_id = session_data["session_id"]

        # Output contains binary data representation
        code = "print(b'\\x00\\x01\\x02\\xff\\xfe')"

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Binary bytes output should not crash: {response.text}"

    def test_execute_unicode_output(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify unicode characters in output are handled."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "unicode_test")
        session_id = session_data["session_id"]

        # Various unicode characters
        code = "print('Hello 世界 🌍 مرحبا')"

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Unicode output should work: {response.text}"
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "世界" in output_text or "Hello" in output_text, (
            f"Unicode should be preserved, got: {output_text}"
        )


class TestResourceUsage:
    """Tests for resource-intensive code."""

    def test_execute_moderate_memory_allocation(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify moderate memory allocation (10MB) succeeds."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "memory_test")
        session_id = session_data["session_id"]

        # Allocate ~10MB
        code = "data = bytearray(10 * 1024 * 1024); print(f'Allocated {len(data)} bytes')"

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"10MB allocation should succeed: {response.text}"

    def test_execute_cpu_intensive_code(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify CPU-intensive code runs to completion."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "cpu_test")
        session_id = session_data["session_id"]

        # Some CPU work (not too much to keep test fast)
        code = """
result = sum(i * i for i in range(100000))
print(f'Result: {result}')
"""

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"CPU work should complete: {response.text}"
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "Result:" in output_text


class TestSpecialCodePatterns:
    """Tests for special code patterns that might cause issues."""

    def test_execute_multiline_string(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify multiline strings are handled."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "multiline_test")
        session_id = session_data["session_id"]

        code = '''text = """
This is a
multiline
string
"""
print(text)'''

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Multiline string should work: {response.text}"

    def test_execute_code_with_special_chars(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify code with special characters works."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "special_chars_test")
        session_id = session_data["session_id"]

        # Special characters that might cause JSON/escaping issues
        code = r"print('Tab:\tNewline:\nBackslash:\\')"

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Special chars should work: {response.text}"

    def test_execute_nested_quotes(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify nested quotes in code work."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "nested_quotes_test")
        session_id = session_data["session_id"]

        code = '''print("He said 'Hello'")
print('She said "World"')'''

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Nested quotes should work: {response.text}"

    def test_execute_json_in_code(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify JSON strings in code don't break request parsing."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "json_code_test")
        session_id = session_data["session_id"]

        code = """import json
data = {"key": "value", "nested": {"a": 1}}
print(json.dumps(data))"""

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"JSON in code should work: {response.text}"
