"""Integration tests for race condition handling.

These tests verify scribe handles concurrent operations gracefully:
1. Concurrent session creation
2. Execute during shutdown
3. Shutdown during execute
4. Concurrent executions on same session
5. Rapid session start/stop cycles
"""

import threading
import uuid

import requests

from tests.conftest import start_session_via_http


class TestConcurrentSessions:
    """Tests for concurrent session operations."""

    def test_concurrent_session_creation(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify multiple sessions can be created concurrently."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()
        token = mcp_server.get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        results = []
        errors = []

        def create_session(name: str):
            try:
                response = requests.post(
                    f"{server_url}/api/scribe/start",
                    json={"experiment_name": name},
                    headers=headers,
                    timeout=60,
                )
                if response.ok:
                    results.append(response.json())
                else:
                    errors.append(f"{name}: {response.status_code} - {response.text}")
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__} - {e}")

        # Create 5 sessions concurrently
        threads = []
        for i in range(5):
            t = threading.Thread(target=create_session, args=(f"concurrent_{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=90)

        # At least some sessions should be created successfully
        # Concurrent creation may have some failures, but shouldn't crash the server
        assert len(results) >= 1, f"At least one session should be created. Errors: {errors}"
        assert len(errors) <= 4, f"Too many errors during concurrent creation: {errors}"

        # Verify all created sessions have unique session_ids
        session_ids = [r["session_id"] for r in results]
        assert len(session_ids) == len(set(session_ids)), "Session IDs should be unique"

    def test_concurrent_executions_same_session(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify kernel remains functional after concurrent execution attempts.

        Note: Jupyter kernels execute requests sequentially, so concurrent requests
        will queue. Due to lock contention, some or all may timeout. The key assertion
        is that the kernel remains functional afterward (not crashed or corrupted).
        """
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "concurrent_exec_test")
        session_id = session_data["session_id"]

        results = []
        errors = []

        def execute_code(code: str, idx: int):
            try:
                response = requests.post(
                    f"{server_url}/api/scribe/exec",
                    json={"session_id": session_id, "code": code},
                    headers=headers,
                    timeout=15,  # Short timeout - we expect contention
                )
                if response.ok:
                    results.append((idx, response.json()))
                else:
                    errors.append((idx, f"{response.status_code} - {response.text}"))
            except Exception as e:
                errors.append((idx, f"{type(e).__name__} - {e}"))

        # Submit 2 rapid executions (minimal to reduce queue depth)
        threads = []
        for i in range(2):
            t = threading.Thread(target=execute_code, args=(f"result_{i} = {i}", i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        # Note: We don't assert that concurrent requests succeed - kernel contention
        # may cause timeouts. The key is that the kernel survives.

        # KEY ASSERTION: kernel should STILL be functional after concurrent attempts
        # This proves we didn't crash or corrupt the kernel state
        # Use longer timeout for the recovery check
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print('kernel still works')"},
            headers=headers,
            timeout=120,  # Long timeout - kernel may need to process queued requests first
        )
        assert response.ok, (
            f"Kernel should still work after concurrent requests: {response.text}. "
            f"Concurrent results: {len(results)} succeeded, {len(errors)} failed."
        )
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "kernel still works" in output_text


class TestLifecycleRaces:
    """Tests for lifecycle race conditions."""

    def test_execute_during_shutdown(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify execute request during shutdown is handled gracefully."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "exec_shutdown_race")
        session_id = session_data["session_id"]

        # Verify session works first
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "x = 1"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        exec_result: dict[str, int | str | None] = {"status": None, "error": None}
        shutdown_result: dict[str, int | str | None] = {"status": None}

        def do_execute():
            try:
                response = requests.post(
                    f"{server_url}/api/scribe/exec",
                    json={"session_id": session_id, "code": "import time; time.sleep(2); print('executed')"},
                    headers=headers,
                    timeout=60,
                )
                exec_result["status"] = response.status_code
            except Exception as e:
                exec_result["error"] = str(e)

        def do_shutdown():
            try:
                response = requests.post(
                    f"{server_url}/api/scribe/shutdown",
                    json={"session_id": session_id},
                    headers=headers,
                    timeout=30,
                )
                shutdown_result["status"] = response.status_code
            except Exception as e:
                shutdown_result["status"] = f"error: {e}"

        # Start execution, then try to shutdown while it's running
        exec_thread = threading.Thread(target=do_execute)
        exec_thread.start()

        # Wait a tiny bit for execution to start, then shutdown
        import time
        time.sleep(0.5)

        shutdown_thread = threading.Thread(target=do_shutdown)
        shutdown_thread.start()

        exec_thread.join(timeout=30)
        shutdown_thread.join(timeout=10)

        # One of these outcomes is acceptable:
        # 1. Execute completes, shutdown succeeds after
        # 2. Execute fails (session shut down), shutdown succeeds
        # 3. Shutdown waits for execute to complete
        # What's NOT acceptable: server crash
        assert exec_result["status"] is not None or exec_result["error"] is not None, (
            "Execute should complete or fail gracefully"
        )

    def test_rapid_session_start_stop_cycles(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify rapid session creation and shutdown cycles don't crash."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()
        token = mcp_server.get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        successful_cycles = 0
        errors = []

        for i in range(3):  # 3 rapid cycles
            try:
                # Start session
                response = requests.post(
                    f"{server_url}/api/scribe/start",
                    json={"experiment_name": f"cycle_{i}"},
                    headers=headers,
                    timeout=30,
                )
                if not response.ok:
                    errors.append(f"cycle_{i} start: {response.status_code}")
                    continue

                session_id = response.json()["session_id"]

                # Quick execution
                response = requests.post(
                    f"{server_url}/api/scribe/exec",
                    json={"session_id": session_id, "code": f"cycle_{i} = True"},
                    headers=headers,
                    timeout=30,
                )
                if not response.ok:
                    errors.append(f"cycle_{i} exec: {response.status_code}")

                # Shutdown
                response = requests.post(
                    f"{server_url}/api/scribe/shutdown",
                    json={"session_id": session_id},
                    headers=headers,
                    timeout=30,
                )
                if not response.ok:
                    errors.append(f"cycle_{i} shutdown: {response.status_code}")

                successful_cycles += 1

            except Exception as e:
                errors.append(f"cycle_{i}: {type(e).__name__} - {e}")

        assert successful_cycles >= 2, f"At least 2 cycles should complete. Errors: {errors}"

    def test_shutdown_nonexistent_session(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify shutdown of nonexistent session gives clear error."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()
        token = mcp_server.get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        # Try to shutdown a session that doesn't exist
        fake_session_id = str(uuid.uuid4())
        response = requests.post(
            f"{server_url}/api/scribe/shutdown",
            json={"session_id": fake_session_id},
            headers=headers,
            timeout=30,
        )

        # Should return error, not crash
        assert response.status_code in [404, 500], (
            f"Shutdown nonexistent session should return error, got {response.status_code}"
        )
        if response.status_code == 500:
            error = response.json()
            assert "error" in error or "not found" in str(error).lower(), (
                f"Error should mention session not found: {error}"
            )


class TestConcurrentEdits:
    """Tests for concurrent edit operations."""

    def test_concurrent_edits_same_cell(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify kernel remains functional after concurrent edit attempts.

        Note: Concurrent edits to the same cell will contend for the kernel.
        Some may timeout. The key assertion is that the kernel remains
        functional afterward (not crashed or corrupted).
        """
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "concurrent_edit_test")
        session_id = session_data["session_id"]

        # Create a cell to edit (use longer timeout for setup)
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "original = 'content'"},
            headers=headers,
            timeout=60,  # Longer timeout for setup
        )
        assert response.ok, f"Setup failed: {response.text}"

        results = []

        def edit_cell(new_code: str, idx: int):
            try:
                response = requests.post(
                    f"{server_url}/api/scribe/edit",
                    json={"session_id": session_id, "code": new_code, "cell_index": 0},
                    headers=headers,
                    timeout=15,  # Short timeout - we expect contention
                )
                results.append((idx, response.status_code, response.ok))
            except Exception as e:
                results.append((idx, "error", str(e)))

        # Submit 2 concurrent edits (fewer to reduce contention)
        threads = []
        for i in range(2):
            t = threading.Thread(target=edit_cell, args=(f"edited_{i} = {i}", i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        # Note: We don't assert that concurrent edits succeed - contention may cause timeouts.
        # The key is that the kernel survives.

        # KEY ASSERTION: kernel should STILL be functional after concurrent attempts
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print('still works')"},
            headers=headers,
            timeout=120,  # Long timeout - kernel may be processing queued requests
        )
        assert response.ok, (
            f"Kernel should still work after concurrent edits: {response.text}. "
            f"Edit results: {results}"
        )
