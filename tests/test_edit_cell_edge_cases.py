"""Integration tests for edit cell edge cases.

These tests verify scribe handles edit cell operations gracefully:
1. Negative indices (Python convention)
2. Large negative indices (out of bounds)
3. Indices beyond notebook length
4. Empty notebook (no code cells)
5. Empty string content
6. Very long content
7. Special characters in code
"""

import uuid

import requests

from tests.conftest import start_session_via_http


class TestEditCellIndices:
    """Tests for edit cell index handling."""

    def test_edit_cell_negative_one_edits_last_cell(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify cell_index=-1 edits the last code cell (Python convention)."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "neg_index_test")
        session_id = session_data["session_id"]

        # Create two code cells
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "first_cell = 'first'"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "second_cell = 'second'"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        # Edit with cell_index=-1 (should edit the second/last cell)
        response = requests.post(
            f"{server_url}/api/scribe/edit",
            json={"session_id": session_id, "code": "edited_last = 'edited'", "cell_index": -1},
            headers=headers,
            timeout=30,
        )
        assert response.ok, f"Edit with cell_index=-1 should work: {response.text}"

        # Verify: first cell variable should still exist, and edited_last should exist
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print(first_cell, edited_last)"},
            headers=headers,
            timeout=30,
        )
        assert response.ok
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "first" in output_text and "edited" in output_text

    def test_edit_cell_large_negative_index(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify cell_index=-999 on a 2-cell notebook gives clear error."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "large_neg_test")
        session_id = session_data["session_id"]

        # Create one code cell
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "x = 1"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        # Try to edit with cell_index=-999 (way out of bounds)
        response = requests.post(
            f"{server_url}/api/scribe/edit",
            json={"session_id": session_id, "code": "never_run = True", "cell_index": -999},
            headers=headers,
            timeout=30,
        )

        # Should return 500 with clear error message
        assert response.status_code == 500, "Large negative index should fail"
        error = response.json()
        error_text = str(error).lower()
        assert any(word in error_text for word in ["index", "range", "out"]), (
            f"Error should mention index out of range, got: {error}"
        )

    def test_edit_cell_beyond_length(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify cell_index=999 on a 2-cell notebook gives clear error."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "beyond_len_test")
        session_id = session_data["session_id"]

        # Create two code cells
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "a = 1"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "b = 2"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        # Try to edit cell_index=999 (way beyond the 2 cells)
        response = requests.post(
            f"{server_url}/api/scribe/edit",
            json={"session_id": session_id, "code": "never_run = True", "cell_index": 999},
            headers=headers,
            timeout=30,
        )

        # Should return 500 with clear error
        assert response.status_code == 500, "Index beyond length should fail"
        error = response.json()
        error_text = str(error).lower()
        assert any(word in error_text for word in ["index", "range", "out"]), (
            f"Error should mention index out of range, got: {error}"
        )

    def test_edit_cell_zero_on_empty_notebook(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify editing cell 0 when notebook has no code cells gives clear error."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        # Start session but don't execute any code (so no code cells exist)
        session_data, server_url, headers = start_session_via_http(mcp_server, "empty_nb_test")
        session_id = session_data["session_id"]

        # Try to edit cell 0 without any code cells
        response = requests.post(
            f"{server_url}/api/scribe/edit",
            json={"session_id": session_id, "code": "x = 1", "cell_index": 0},
            headers=headers,
            timeout=30,
        )

        # Should return 500 with clear error about no code cells
        assert response.status_code == 500, "Edit on empty notebook should fail"
        error = response.json()
        error_text = str(error).lower()
        assert any(word in error_text for word in ["no code cells", "not found", "empty"]), (
            f"Error should mention no code cells or similar, got: {error}"
        )


class TestEditCellContent:
    """Tests for edit cell content handling."""

    def test_edit_cell_to_empty_string(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify editing cell to empty string works (clears the cell)."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "empty_edit_test")
        session_id = session_data["session_id"]

        # Create a code cell
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "original = 'content'"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        # Edit to empty string
        response = requests.post(
            f"{server_url}/api/scribe/edit",
            json={"session_id": session_id, "code": "", "cell_index": 0},
            headers=headers,
            timeout=30,
        )

        # Should succeed - empty cell is valid
        assert response.ok, f"Edit to empty string should work: {response.text}"
        result = response.json()
        # Should have no outputs (empty code produces nothing)
        outputs = result.get("outputs", [])
        # Filter out any empty stream outputs
        meaningful_outputs = [o for o in outputs if o.get("text", "").strip()]
        assert len(meaningful_outputs) == 0, f"Empty code should produce no output, got: {outputs}"

    def test_edit_cell_with_very_long_content(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify editing cell with very long code works."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "long_content_test")
        session_id = session_data["session_id"]

        # Create a code cell
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "x = 1"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        # Generate very long code (but still valid Python)
        # Creating a string assignment with 100KB of content
        long_string = "a" * 100_000
        long_code = f"long_var = '{long_string}'\nprint(len(long_var))"

        response = requests.post(
            f"{server_url}/api/scribe/edit",
            json={"session_id": session_id, "code": long_code, "cell_index": 0},
            headers=headers,
            timeout=60,  # Allow more time for large code
        )

        assert response.ok, f"Edit with long content should work: {response.text}"
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        # Should print the length: 100000
        assert "100000" in output_text, f"Should show correct length, got: {output_text}"

    def test_edit_cell_with_special_characters(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify editing cell with special characters (unicode, emojis) works."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "special_chars_test")
        session_id = session_data["session_id"]

        # Create a code cell
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "x = 1"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        # Edit with unicode, emojis, and special chars
        special_code = '''# Comment with unicode: 你好世界 🎉
special_string = "Hello 世界! 🌍 π ≈ 3.14159"
print(special_string)
'''

        response = requests.post(
            f"{server_url}/api/scribe/edit",
            json={"session_id": session_id, "code": special_code, "cell_index": 0},
            headers=headers,
            timeout=30,
        )

        assert response.ok, f"Edit with special characters should work: {response.text}"
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        # Should contain the unicode/emoji output
        assert "世界" in output_text or "Hello" in output_text, (
            f"Should handle unicode properly, got: {output_text}"
        )


class TestEditCellTypes:
    """Tests for edit cell behavior with different cell types."""

    def test_edit_only_affects_code_cells(
        self,
        reset_mcp_module,
        cleanup_jupyter_processes,  # pyright: ignore[reportUnusedParameter]
    ):
        """Verify edit_cell only operates on code cells, skipping markdown."""
        test_session_id = str(uuid.uuid4())
        mcp_server = reset_mcp_module(test_session_id)

        session_data, server_url, headers = start_session_via_http(mcp_server, "cell_type_test")
        session_id = session_data["session_id"]

        # First, add a markdown cell
        response = requests.post(
            f"{server_url}/api/scribe/markdown",
            json={"session_id": session_id, "content": "# Markdown Header"},
            headers=headers,
            timeout=30,
        )
        assert response.ok, f"Adding markdown should work: {response.text}"

        # Then add a code cell
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "code_var = 'original'"},
            headers=headers,
            timeout=30,
        )
        assert response.ok

        # Edit cell_index=0 - this should edit the FIRST CODE CELL, not the markdown
        response = requests.post(
            f"{server_url}/api/scribe/edit",
            json={"session_id": session_id, "code": "code_var = 'edited'", "cell_index": 0},
            headers=headers,
            timeout=30,
        )
        assert response.ok, f"Edit should target code cells only: {response.text}"

        # Verify: the code cell was edited (code_var should be 'edited')
        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": "print(code_var)"},
            headers=headers,
            timeout=30,
        )
        assert response.ok
        result = response.json()
        outputs = result.get("outputs", [])
        output_text = "".join(
            o.get("text", "") for o in outputs if o.get("output_type") == "stream"
        )
        assert "edited" in output_text, f"Code cell should have been edited, got: {output_text}"
