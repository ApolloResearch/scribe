"""Scribe Notebook MCP Server - Model Context Protocol interface for agents to work with
the Scribe notebook server. MCP endpoints are easier for agents to interact with than
raw HTTP requests or Jupyter Server API calls.
"""

__all__ = [
    # Public API
    "SessionInfo",
    "ServerStatus",
    "ensure_server_running",
    "get_token",
    "save_state",
    "load_state",
    "clear_state",
    "check_jupyter_status",
    "is_jupyter_alive",
    # For testing
    "_get_state_file",
]

import atexit
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from pydantic import BaseModel

from scribe.notebook._notebook_server_utils import (
    check_server_health,
    cleanup_scribe_server,
    find_safe_port,
    process_jupyter_outputs,
    start_scribe_server,
)  # noqa: E402

# Initialize MCP server
mcp = FastMCP("scribe")


class SessionInfo(BaseModel):
    """Metadata for an active notebook session, persisted across compaction."""

    session_id: str
    notebook_path: str


def _check_response(response: requests.Response, operation: str) -> dict:
    """Check HTTP response and return JSON data, raising with server error message on failure.

    Args:
        response: The requests Response object
        operation: Description of the operation for error messages (e.g., "start session")

    Returns:
        The JSON response data (empty dict if response has no content)

    Raises:
        Exception: With the actual server error message if request failed
    """
    if not response.ok:
        # Try to extract error message from JSON with common error field names
        try:
            error_data = response.json()
            # Check multiple common error fields
            error_msg = (
                error_data.get("error")
                or error_data.get("detail")
                or error_data.get("message")
                or response.text
            )
        except Exception:
            error_msg = response.text or "No error details"
        raise Exception(f"Failed to {operation} (HTTP {response.status_code}): {error_msg}")

    # Handle empty/non-JSON success responses (e.g., 204 No Content)
    if not response.content:
        return {}

    try:
        return response.json()
    except Exception:
        # If response isn't JSON, return empty dict rather than crashing
        return {}


# Global server management
_server_process: subprocess.Popen | None = None
_server_port: int | None = None
_server_url: str | None = None
_server_token: str | None = None
# Down the line, we may wish to keep the Jupyter server around even after MCP server exits
_is_external_server: bool = False

SCRIBE_PROVIDER: str | None = os.environ.get("SCRIBE_PROVIDER")

# Session tracking for cleanup - maps session_id to SessionInfo
_active_sessions: dict[str, SessionInfo] = {}


# ============================================================================
# State Persistence (for surviving Claude Code compaction)
# ============================================================================


def _get_state_file() -> Path:
    """Get state file path unique to current working directory AND session.

    This allows:
    - Multiple Claude Code instances in different directories to have separate sessions
    - Concurrent scribe sessions in the SAME directory to have separate state files
    - Same session after compaction to reconnect to its own Jupyter server

    Raises:
        RuntimeError: If SCRIBE_SESSION_ID is not set (MCP server must be invoked via scribe CLI)
    """
    session_id = os.environ.get("SCRIBE_SESSION_ID")
    if not session_id:
        raise RuntimeError(
            "SCRIBE_SESSION_ID environment variable is required. "
            "The MCP server must be invoked via the scribe CLI (e.g., 'scribe claude'), "
            "which sets this variable automatically."
        )

    cwd_hash = hashlib.md5(os.getcwd().encode()).hexdigest()[:8]
    # Include first 8 chars of session_id for uniqueness
    state_file = Path.home() / f".scribe_state_{cwd_hash}_{session_id[:8]}.json"
    print(f"[scribe] Using state file: {state_file}", file=sys.stderr)
    return state_file


def save_state() -> None:
    """Persist current MCP server state to disk for recovery after compaction.

    Uses atomic write (write to temp file then rename) and sets restrictive
    permissions (0o600) since the state file contains the Jupyter auth token.
    """
    global _server_port, _server_token, _server_url, _server_process, _active_sessions

    state = {
        "version": 2,  # Bumped version for new session format with notebook paths
        "server": {
            "port": _server_port,
            "token": _server_token,
            "pid": _server_process.pid if _server_process else None,
            "url": _server_url,
        },
        "sessions": [s.model_dump() for s in _active_sessions.values()],
        "updated_at": datetime.now().isoformat(),
    }
    state_file = _get_state_file()
    temp_file = state_file.with_suffix(".tmp")
    try:
        # Write to temp file first
        temp_file.write_text(json.dumps(state, indent=2))
        # Set restrictive permissions (owner read/write only) - token is sensitive
        os.chmod(temp_file, 0o600)
        # Atomic rename
        os.replace(temp_file, state_file)
    except OSError as e:
        print(f"[scribe] Warning: Failed to save state: {e}", file=sys.stderr)
        # Clean up temp file if it exists
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass


def load_state() -> dict | None:
    """Load persisted state from disk if it exists."""
    state_file = _get_state_file()
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return None


def clear_state() -> None:
    """Remove state file (used when server is confirmed dead)."""
    state_file = _get_state_file()
    try:
        if state_file.exists():
            state_file.unlink()
    except OSError:
        pass


class ServerStatus(Enum):
    """Status of a Jupyter server health check."""

    HEALTHY = "healthy"  # Server responded successfully
    UNAUTHORIZED = "unauthorized"  # Server alive but rejected auth (401/403)
    UNREACHABLE = "unreachable"  # Connection refused/timeout


def check_jupyter_status(port: int, token: str) -> ServerStatus:
    """Check Jupyter server status with auth differentiation.

    Distinguishes between:
    - HEALTHY: Server is responding and accepting our token
    - UNAUTHORIZED: Server is alive but rejecting our token (401/403)
    - UNREACHABLE: Server is not responding (connection refused, timeout, etc.)

    This distinction is important because an UNAUTHORIZED response means the server
    is still running (just with a different token), while UNREACHABLE means it's dead.
    """
    try:
        headers = {"Authorization": f"token {token}"} if token else {}
        response = requests.get(
            f"http://127.0.0.1:{port}/api/scribe/health",
            headers=headers,
            timeout=2,
        )
        if response.status_code == 200:
            return ServerStatus.HEALTHY
        elif response.status_code in (401, 403):
            return ServerStatus.UNAUTHORIZED
        else:
            return ServerStatus.UNREACHABLE
    except requests.RequestException:
        return ServerStatus.UNREACHABLE


def is_jupyter_alive(port: int, token: str) -> bool:
    """Check if a Jupyter server is responding at the given port with the given token.

    This is a backwards-compatible wrapper around check_jupyter_status that returns
    a simple boolean (True only if HEALTHY).
    """
    return check_jupyter_status(port, token) == ServerStatus.HEALTHY


# ============================================================================
# Server Management
# ============================================================================


def start_jupyter_server() -> tuple[subprocess.Popen, int, str]:
    """Start a Jupyter server subprocess and return process, port, and URL."""
    port = find_safe_port()
    if port is None:
        raise Exception("Could not find an available port for Jupyter server")

    # Generate token for this server instance
    token = get_token()

    # Get notebook output directory from environment variable
    notebook_output_dir = os.environ.get("NOTEBOOK_OUTPUT_DIR")

    # Use utils function to start server
    process = start_scribe_server(port, token, notebook_output_dir)
    url = f"http://127.0.0.1:{port}"

    return process, port, url


def cleanup_server():
    """Clean up the managed Jupyter server."""
    global _server_process, _server_token, _active_sessions

    if _server_process and not _is_external_server:
        cleanup_scribe_server(_server_process)
        _server_process = None
        _server_token = None
        clear_state()  # Remove state file pointing to now-dead server


def ensure_server_running() -> str:
    """Ensure a Jupyter server is running and return its URL."""
    global _server_process, _server_port, _server_url, _server_token, _is_external_server, _active_sessions

    # Check if SCRIBE_PORT is set (external server)
    if "SCRIBE_PORT" in os.environ:
        port = os.environ["SCRIBE_PORT"]
        _server_port = int(port)
        _server_url = f"http://127.0.0.1:{port}"
        # Support SCRIBE_TOKEN for external server authentication
        _server_token = os.environ.get("SCRIBE_TOKEN", "")
        _is_external_server = True

        # Optionally verify external server is reachable
        if _server_token:
            status = check_jupyter_status(_server_port, _server_token)
            if status != ServerStatus.HEALTHY:
                print(
                    f"[scribe] Warning: External server at port {port} returned {status.value}",
                    file=sys.stderr,
                )

        return _server_url

    # Check if our managed server is still running
    if _server_process and _server_process.poll() is None:
        assert _server_url is not None  # Set when server started
        return _server_url

    # Try to restore from persisted state (survives Claude Code compaction)
    state = load_state()
    if state and state.get("server", {}).get("port"):
        saved_port = state["server"]["port"]
        saved_token = state["server"]["token"]

        if saved_token:
            status = check_jupyter_status(saved_port, saved_token)
            if status == ServerStatus.HEALTHY:
                print(
                    f"[scribe] Reconnected to existing Jupyter server at port {saved_port}",
                    file=sys.stderr,
                )
                _server_port = saved_port
                _server_token = saved_token
                _server_url = f"http://127.0.0.1:{saved_port}"
                # Restore sessions - handle both old format (list of IDs) and new format (list of dicts)
                saved_sessions = state.get("sessions", [])
                _active_sessions = {}
                for s in saved_sessions:
                    if isinstance(s, dict):
                        info = SessionInfo(**s)
                        _active_sessions[info.session_id] = info
                    elif isinstance(s, str):
                        # Legacy format: just session ID, no notebook path
                        _active_sessions[s] = SessionInfo(session_id=s, notebook_path="")
                _is_external_server = False  # We started it, but don't have process handle
                # Note: _server_process stays None since we don't own the process handle anymore
                return _server_url
            elif status == ServerStatus.UNAUTHORIZED:
                print(
                    f"[scribe] Saved Jupyter server (port {saved_port}) rejected auth token, starting new one",
                    file=sys.stderr,
                )
                clear_state()
            else:  # UNREACHABLE
                print(
                    f"[scribe] Saved Jupyter server (port {saved_port}) is dead, starting new one",
                    file=sys.stderr,
                )
                clear_state()
        else:
            # No token in saved state - clear and start fresh
            clear_state()

    # Start a new managed server
    _is_external_server = False
    _server_process, _server_port, _server_url = start_jupyter_server()

    # Register cleanup handlers
    atexit.register(cleanup_server)
    signal.signal(signal.SIGTERM, lambda _sig, _frame: cleanup_server())
    signal.signal(signal.SIGINT, lambda _sig, _frame: cleanup_server())

    print(f"[scribe] Started managed Jupyter server at {_server_url}", file=sys.stderr)

    # Persist state for recovery after compaction
    save_state()

    return _server_url


def get_token() -> str:
    """Generate or return cached auth token."""
    global _server_token
    if not _server_token and not _is_external_server:
        _server_token = secrets.token_urlsafe(32)
    return _server_token or ""


def get_server_status() -> dict[str, Any]:
    """Get current server status information."""
    global _server_port, _server_url, _is_external_server, _server_process

    if not _server_url:
        return {
            "status": "not_started",
            "url": None,
            "port": None,
            "vscode_url": None,
            "will_shutdown_on_exit": True,
            "is_external": False,
            "health": "unknown",
        }

    # Check health using utils function
    health_data = check_server_health(_server_port) if _server_port else None
    health = "healthy" if health_data else "unreachable"

    # Check if process is still running (for managed servers)
    process_running = True
    if not _is_external_server and _server_process:
        process_running = _server_process.poll() is None

    return {
        "status": "running" if process_running else "stopped",
        "url": _server_url,
        "port": _server_port,
        "vscode_url": f"{_server_url}/?token={get_token()}" if _server_url else None,
        "will_shutdown_on_exit": not _is_external_server,
        "is_external": _is_external_server,
        "health": health,
    }


async def _start_session_internal(
    experiment_name: str | None = None,
    notebook_path: str | None = None,
    fork_prev_notebook: bool = True,
    tool_name: str = "start_session",
) -> dict[str, Any]:
    """Internal helper function for starting sessions from scratch versus resuming versus forking existing notebook.

    Args:
        experiment_name: Custom name for the notebook
        notebook_path: Path to existing notebook (if any)
        fork_prev_notebook: If True, create new notebook; if False, use existing in-place
        tool_name: Name of the calling tool for logging/debugging
    """
    try:
        # Ensure server is running
        server_url = ensure_server_running()

        # Build request body
        request_body = {}
        if experiment_name:
            request_body["experiment_name"] = experiment_name
        if notebook_path:
            request_body["notebook_path"] = notebook_path
            request_body["fork_prev_notebook"] = fork_prev_notebook

        # Start session
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}
        print(f"[DEBUG MCP] {tool_name}: Connecting to {server_url}", file=sys.stderr)

        response = requests.post(
            f"{server_url}/api/scribe/start", json=request_body, headers=headers
        )
        data = _check_response(response, "start session")

        result = {
            "session_id": data["session_id"],
            "kernel_id": data.get("kernel_id"),
            "status": "started",
            "notebook_path": data["notebook_path"],
            "vscode_url": f"{data.get('server_url', server_url)}/?token={data.get('token', token)}",
            "kernel_name": data.get(
                "kernel_name", data.get("kernel_display_name", "Scribe Kernel")
            ),
        }

        # Track session for cleanup
        global _active_sessions
        _active_sessions[data["session_id"]] = SessionInfo(
            session_id=data["session_id"],
            notebook_path=data["notebook_path"],
        )

        # Persist state for recovery after compaction
        save_state()

        # Handle restoration results if present (only for notebook-based sessions)
        if notebook_path:
            # Pass through restoration summary if present
            if "restoration_summary" in data:
                result["restoration_summary"] = data["restoration_summary"]

            # Only pass error details for debugging, not full restoration results
            if "restoration_results" in data:
                errors = [
                    r for r in data["restoration_results"] if r.get("status") == "error"
                ]
                if errors:
                    # Summarize errors with cell numbers and error messages only
                    error_summary = []
                    for error in errors:
                        error_info = {
                            "cell": error.get("cell"),
                            "error": error.get("error", "").split(":")[0]
                            if ":" in error.get("error", "")
                            else error.get("error", ""),
                        }
                        error_summary.append(error_info)
                    result["restoration_errors"] = error_summary

            # Add guidance for agent when working with existing notebooks
            if "restoration_summary" in data:
                has_errors = (
                    "restoration_errors" in result and result["restoration_errors"]
                )
                if fork_prev_notebook:
                    # Continue/fork scenario
                    if has_errors:
                        result["note"] = (
                            f"A new notebook has been created at {data['notebook_path']} "
                            f"with the restored state from {notebook_path}. "
                            f"Some cells had errors during restoration - see restoration_errors for details. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )
                    else:
                        result["note"] = (
                            f"A new notebook has been created at {data['notebook_path']} "
                            f"with the restored state from {notebook_path}. "
                            "All cells executed successfully during restoration. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )
                else:
                    # Resume scenario
                    if has_errors:
                        result["note"] = (
                            f"Resumed notebook at {data['notebook_path']} in-place. "
                            f"Some cells had errors during restoration - see restoration_errors for details. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )
                    else:
                        result["note"] = (
                            f"Successfully resumed notebook at {data['notebook_path']} in-place. "
                            "All cells executed successfully during restoration. "
                            "Please use the NotebookRead tool to review the notebook contents."
                        )

        return result

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to start session ({tool_name}): {str(e)}")


@mcp.tool
async def start_new_session(experiment_name: str | None = None) -> dict[str, Any]:
    """Start a completely new Jupyter kernel session with an empty notebook.

    Args:
        experiment_name: Custom name for the notebook (e.g., "ImageGeneration")

    Returns:
        Dictionary with:
        - session_id: Unique session identifier
        - notebook_path: Path to the new notebook
        - status: "started"
        - kernel_id: The kernel ID
        - vscode_url: URL to connect with VSCode
        - kernel_name: Display name of the kernel
    """
    return await _start_session_internal(
        experiment_name=experiment_name,
        notebook_path=None,
        fork_prev_notebook=True,
        tool_name="start_new_session",
    )


@mcp.tool(
    name="start_session_resume_notebook",
    # Description pulled from docstring
    tags=None,
    annotations={
        "title": "Start Session - Resume Notebook"  # A human-readable title for the tool.
    },
)
async def start_session_resume_notebook(notebook_path: str) -> dict[str, Any]:
    """Start a new session by resuming an existing notebook in-place, modifying the original notebook file.

    This executes all cells in the existing notebook to restore the kernel state and updates
    the notebook file with new outputs. Use this to continue working in an existing notebook file.

    Args:
        notebook_path: Path to the existing notebook to resume from

    Returns:
        Dictionary with:
        - session_id: Unique session identifier
        - notebook_path: Path to the resumed notebook (same as input)
        - status: "started"
        - kernel_id: The kernel ID
        - vscode_url: URL to connect with VSCode
        - kernel_name: Display name of the kernel
        - restoration_summary: Summary of the resume operation
        - restoration_errors: List of any errors that occurred during cell execution
        - note: Guidance message about the resumed notebook
    """
    return await _start_session_internal(
        experiment_name=None,
        notebook_path=notebook_path,
        fork_prev_notebook=False,
        tool_name="start_session_resume_notebook",
    )


@mcp.tool
async def start_session_continue_notebook(
    notebook_path: str, experiment_name: str | None = None
) -> dict[str, Any]:
    """Start a session by continuing from an existing notebook (creates a new notebook file).

    This creates a new notebook with "_continued" suffix, copies all cells from the existing
    notebook, and executes them to restore the kernel state. The original notebook is unchanged.

    Args:
        notebook_path: Path to the existing notebook to continue from
        experiment_name: Optional custom name for the new notebook

    Returns:
        Dictionary with:
        - session_id: Unique session identifier
        - notebook_path: Path to the new notebook (with "_continued" suffix)
        - status: "started"
        - kernel_id: The kernel ID
        - vscode_url: URL to connect with VSCode
        - kernel_name: Display name of the kernel
        - restoration_summary: Summary of the continuation operation
        - restoration_errors: List of any errors that occurred during cell execution
        - note: Guidance to read the new notebook
    """
    return await _start_session_internal(
        experiment_name=experiment_name,
        notebook_path=notebook_path,
        fork_prev_notebook=True,
        tool_name="start_session_continue_notebook",
    )


@mcp.tool
async def execute_code(
    session_id: str, code: str
) -> list[dict[str, Any] | Image]:
    """Execute Python code in the specified kernel session.

    Images generated during execution (e.g., via .show()) are returned as
    fastmcp.Image objects that can be directly viewed.

    IMPORTANT: Sessions persist across context compaction. If you lose your session_id
    (e.g., after compaction), use list_sessions to find active sessions you can continue using.
    You do NOT need to resume the notebook - just use the session_id from list_sessions.

    Args:
        session_id: The session ID returned by start_session (or from list_sessions)
        code: Python code to execute

    Returns:
        Dictionary containing:
        - session_id: The session ID
        - execution_count: The cell execution number
        - outputs: List of output objects with type and content/data
    """
    # start_time = time.time()
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/exec",
            json={"session_id": session_id, "code": code},
            headers=headers,
        )
        data = _check_response(response, f"execute code in session {session_id}")

        # Process outputs using utils function
        outputs, images = process_jupyter_outputs(
            data["outputs"],
            session_id=session_id,
            save_images_locally=False,
        )

        # Create result list with execution metadata first, then images
        result: list[dict[str, Any] | Image] = [
            {
                "session_id": session_id,
                "execution_count": data["execution_count"],
                "outputs": outputs,
            }
        ]
        result.extend(images)
        return result

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to execute code: {str(e)}")


@mcp.tool
async def add_markdown(session_id: str, content: str) -> dict[str, int]:
    """Add a markdown cell to the notebook for documentation.

    Args:
        session_id: The session ID
        content: Markdown content to add

    Returns:
        Dictionary with the cell number
    """
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/markdown",
            json={"session_id": session_id, "content": content},
            headers=headers,
        )
        data = _check_response(response, f"add markdown in session {session_id}")

        return {"cell_number": data["cell_number"]}

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to add markdown: {str(e)}")


@mcp.tool
async def edit_cell(
    session_id: str, code: str, cell_index: int = -1
) -> list[dict[str, Any] | Image]:
    """Edit an existing code cell in the notebook and execute the new code.

    This is especially useful for fixing errors or modifying the most recent cell.

    Args:
        session_id: The session ID
        code: New Python code to replace the cell content
        cell_index: Index of the code cell to edit (default -1 for last cell)
                   Use -1 for the most recent cell, -2 for second to last, etc.
                   Or use 0, 1, 2... for specific cells from the beginning

    Returns:
        Dictionary containing:
        - session_id: The session ID
        - cell_index: The code cell index that was edited
        - actual_notebook_index: The actual index in the notebook (including markdown cells)
        - execution_count: The cell execution number
        - outputs: List of output objects with type and content/data
    """
    # start_time = time.time()
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/edit",
            json={"session_id": session_id, "code": code, "cell_index": cell_index},
            headers=headers,
        )
        data = _check_response(response, f"edit cell in session {session_id}")

        # Process outputs using utils function
        outputs, images = process_jupyter_outputs(
            data["outputs"],
            session_id=session_id,
            save_images_locally=False,
        )

        # Create result list with execution metadata first, then images
        result: list[dict[str, Any] | Image] = [
            {
                "session_id": session_id,
                "cell_index": data["cell_index"],
                "actual_notebook_index": data["actual_notebook_index"],
                "execution_count": data["execution_count"],
                "outputs": outputs,
            }
        ]
        result.extend(images)
        return result

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to edit cell: {str(e)}")


@mcp.tool
async def shutdown_session(session_id: str) -> str:
    """Shutdown a kernel session gracefully.

    Note: using this tool terminates kernel state; it should typically only be used if the user
    has instructured you to do so.

    Args:  session_id: The session ID to shutdown
    """
    try:
        server_url = ensure_server_running()
        token = get_token()
        headers = {"Authorization": f"token {token}"} if token else {}

        response = requests.post(
            f"{server_url}/api/scribe/shutdown",
            json={"session_id": session_id},
            headers=headers,
        )
        _check_response(response, f"shutdown session {session_id}")

        # Clean up session tracking
        global _active_sessions
        _active_sessions.pop(session_id, None)

        # Persist state for recovery after compaction
        save_state()

        return f"Session {session_id} shut down successfully"

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to shutdown session: {str(e)}")


@mcp.tool
async def list_sessions() -> dict[str, Any]:
    """List all active notebook sessions with their metadata.

    Use this to find valid session_ids for execute_code, add_markdown, edit_cell, etc.

    IMPORTANT: Call this after context compaction if you've lost your session_id.
    Sessions persist across compaction - you can continue using them without resuming
    the notebook. Just get the session_id from this function and pass it to execute_code.

    Note: Session IDs returned may include stale sessions if kernels have died. These
    are best-effort and will fail gracefully if used.

    Returns:
        Dictionary with:
        - sessions: List of session objects, each containing:
            - session_id: The UUID to pass to execute_code, edit_cell, etc.
            - notebook_path: Path to the notebook file for this session
        - server_status: Current server status (URL redacted for security)

    Example response:
        {
            "sessions": [
                {"session_id": "abc-123-...", "notebook_path": "/path/to/notebook.ipynb"}
            ],
            "server_status": {...}
        }
    """
    # Ensure server is running and state is loaded from disk (critical after compaction)
    ensure_server_running()

    status = get_server_status()

    # Redact auth token from vscode_url to prevent token leakage in logs/transcripts
    if status.get("vscode_url") and "?token=" in status["vscode_url"]:
        base_url = status["vscode_url"].split("?token=")[0]
        status["vscode_url"] = f"{base_url}?token=<redacted>"

    return {
        "sessions": [s.model_dump() for s in _active_sessions.values()],
        "server_status": status,
    }


@mcp.resource(
    uri="scribe://server/status",
    name="ScribeNotebookServerStatus",  # A human-readable name. If not provided, defaults to function name
    description="Get the current Scribe server status and connection information.",
)
async def server_status() -> str:
    status = get_server_status()

    # Format as a readable status report
    lines = [
        "# Scribe Server Status",
        "",
        f"**Status:** {status['status']}",
        f"**URL:** {status['url'] or 'Not available'}",
        f"**Port:** {status['port'] or 'Not available'}",
        f"**VSCode URL:** {status['vscode_url'] or 'Not available'}",
        f"**Health:** {status['health']}",
        f"**Auth Token:** {'Yes (auto-generated)' if get_token() else 'No'}",
        f"**External Server:** {'Yes' if status['is_external'] else 'No'}",
        f"**Will Shutdown on Exit:** {'Yes' if status['will_shutdown_on_exit'] else 'No'}",
    ]

    if not status["is_external"]:
        lines.extend(["", "*This server is automatically managed by the MCP server.*"])
    else:
        lines.extend(
            ["", "*This server was found via SCRIBE_PORT environment variable.*"]
        )

    return "\n".join(lines)


# Main entry point for STDIO transport
if __name__ == "__main__":
    mcp.run(transport="stdio")
