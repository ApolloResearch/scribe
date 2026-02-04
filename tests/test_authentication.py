"""Integration tests for authentication edge cases.

These tests verify scribe handles authentication issues gracefully:
1. No Authorization header
2. Empty token
3. Malformed token format
4. Wrong auth scheme (Bearer vs token)
5. Invalid/wrong token value
6. Token with special characters
"""

import uuid

import requests

from tests.conftest import start_session_via_http


class TestMissingAuth:
    """Tests for missing or empty authentication."""

    def test_request_no_auth_header(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify request without Authorization header is rejected."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()

        # Make request with no Authorization header
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": "no_auth_test"},
            headers={},  # No Authorization header
            timeout=30,
        )

        # Jupyter Server should reject unauthenticated requests
        # Typically returns 403 Forbidden or 401 Unauthorized
        assert response.status_code in [401, 403], (
            f"Expected 401/403 for missing auth, got {response.status_code}: {response.text}"
        )

    def test_request_empty_token(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify request with empty token is rejected."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()

        # Make request with empty token
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": "empty_token_test"},
            headers={"Authorization": "token "},  # Empty token value
            timeout=30,
        )

        # Should be rejected
        assert response.status_code in [401, 403], (
            f"Expected 401/403 for empty token, got {response.status_code}: {response.text}"
        )

    def test_request_auth_header_no_value(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify request with malformed Authorization header is rejected."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()

        # Make request with malformed header (just "token" without value)
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": "malformed_auth_test"},
            headers={"Authorization": "token"},  # No value after scheme
            timeout=30,
        )

        # Should be rejected
        assert response.status_code in [401, 403], (
            f"Expected 401/403 for malformed auth, got {response.status_code}: {response.text}"
        )


class TestWrongAuthScheme:
    """Tests for wrong authentication scheme."""

    def test_request_bearer_scheme_with_valid_token(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify Jupyter accepts Bearer scheme with valid token.

        Note: Jupyter Server is flexible and accepts Bearer scheme as well as token scheme.
        This is reasonable behavior for HTTP compatibility.
        """
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()
        real_token = mcp_server.get_token()

        # Use Bearer scheme with valid token
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": "bearer_scheme_test"},
            headers={"Authorization": f"Bearer {real_token}"},
            timeout=30,
        )

        # Jupyter Server accepts Bearer scheme with valid token
        assert response.ok, (
            f"Bearer scheme with valid token should work, got {response.status_code}: {response.text}"
        )

    def test_request_basic_auth_scheme(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify request with Basic auth scheme is rejected."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()

        # Use Basic auth scheme
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": "basic_auth_test"},
            headers={"Authorization": "Basic dXNlcjpwYXNz"},  # base64 of user:pass
            timeout=30,
        )

        # Should be rejected
        assert response.status_code in [401, 403], (
            f"Expected 401/403 for Basic scheme, got {response.status_code}: {response.text}"
        )


class TestInvalidToken:
    """Tests for invalid token values."""

    def test_request_invalid_token_value(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify request with wrong token value is rejected."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()

        # Use a valid format but wrong token value
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": "wrong_token_test"},
            headers={"Authorization": "token totally-wrong-token-value"},
            timeout=30,
        )

        # Should be rejected
        assert response.status_code in [401, 403], (
            f"Expected 401/403 for wrong token, got {response.status_code}: {response.text}"
        )

    def test_request_token_with_special_characters(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify token with special characters is handled gracefully."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()

        # Token with special characters that might cause parsing issues
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": "special_char_token_test"},
            headers={"Authorization": "token abc=def&foo<bar>"},  # Special chars
            timeout=30,
        )

        # Should be rejected (invalid token) but not crash
        assert response.status_code in [401, 403, 500], (
            f"Expected rejection for special chars in token, got {response.status_code}"
        )

    def test_request_token_with_whitespace(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify token with whitespace is handled gracefully."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()

        # Token with embedded whitespace
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": "whitespace_token_test"},
            headers={"Authorization": "token some token with spaces"},
            timeout=30,
        )

        # Should be rejected but not crash
        assert response.status_code in [401, 403], (
            f"Expected 401/403 for whitespace token, got {response.status_code}"
        )


class TestValidAuth:
    """Tests to verify valid authentication still works."""

    def test_valid_token_accepts_request(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify request with valid token is accepted."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        server_url = mcp_server.ensure_server_running()
        token = mcp_server.get_token()

        # Use correct token
        response = requests.post(
            f"{server_url}/api/scribe/start",
            json={"experiment_name": "valid_token_test"},
            headers={"Authorization": f"token {token}"},
            timeout=30,
        )

        assert response.ok, f"Valid token should be accepted: {response.text}"
        session_data = response.json()
        assert "session_id" in session_data

    def test_authenticated_session_can_execute_code(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify full authenticated workflow works."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "auth_flow_test")
        session_id = session_data["session_id"]

        # Execute code with valid auth
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print('authenticated!')"},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Authenticated request should succeed: {response.text}"
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "authenticated" in output_text


class TestAuthPersistence:
    """Tests for authentication with session persistence."""

    def test_auth_required_for_execute_on_existing_session(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify execute on existing session still requires auth."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        # Start session with valid auth
        session_data, server_url, headers = start_session_via_http(mcp_server, "persist_test")
        session_id = session_data["session_id"]

        # Try to execute without auth - should fail
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print('should fail')"},
            headers={},  # No auth
            timeout=30,
        )

        assert response.status_code in [401, 403], (
            f"Execute without auth should fail, got {response.status_code}: {response.text}"
        )

        # Same session with auth should work
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print('should work')"},
            headers=headers,  # With auth
            timeout=30,
        )

        assert response.ok, f"Execute with auth should succeed: {response.text}"
