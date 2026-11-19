"""ASGI Remote MCP Server supporting Streamable HTTP, Legacy SSE, Bearer/Secret-Path Auth, and OAuth 2.1."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import html
import json
import secrets
import time
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional, Tuple

from starlette.applications import Starlette
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import __version__
from .runtime import ColabRuntime

SUPPORTED_PROTOCOL_VERSIONS = [
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
]
DEFAULT_PROTOCOL_VERSION = "2025-06-18"
MAX_OAUTH_CLIENTS = 256
MAX_OAUTH_AUTH_CODES = 256
MAX_SESSIONS = 512

CORS_HEADERS: Dict[str, str] = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": (
        "Authorization, Content-Type, Accept, Mcp-Session-Id, MCP-Protocol-Version, X-Api-Key"
    ),
    "Access-Control-Expose-Headers": "Mcp-Session-Id",
}

SERVER_INSTRUCTIONS = (
    "Connected directly to a remote Google Colab VM via colab-cloud-mcp. "
    "Python state in `execute_python` persists across calls. "
    "For tasks expected to run longer than 2 minutes (model training, evaluation sweeps, "
    "large weight downloads to Google Drive), always use `start_background_job` and poll "
    "with `get_job_status` so cloud chat HTTP timeouts are never exceeded."
)


def _consteq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _get_external_base_url(request: Request, configured_public_url: Optional[str] = None) -> str:
    if configured_public_url:
        return configured_public_url.rstrip("/")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme or "https"
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost"
    return f"{proto}://{host}".rstrip("/")


def _is_valid_redirect_uri(uri: Any) -> bool:
    if not isinstance(uri, str) or not uri or any(ch.isspace() for ch in uri):
        return False
    if "#" in uri:
        return False
    parsed = urllib.parse.urlsplit(uri)
    if parsed.fragment:
        return False
    try:
        _ = parsed.port
    except ValueError:
        return False
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if scheme == "https":
        return True
    if scheme == "http" and host in ("localhost", "127.0.0.1", "::1"):
        return True
    return False


async def _parse_urlencoded_form(request: Request) -> Dict[str, str]:
    raw_body = await request.body()
    parsed = urllib.parse.parse_qs(
        raw_body.decode("utf-8", errors="replace"),
        keep_blank_values=True,
    )
    return {k: v[0] if v else "" for k, v in parsed.items()}


def _contains_tool_call(payload: Any) -> bool:
    if isinstance(payload, dict):
        return payload.get("method") == "tools/call" and "id" in payload
    if isinstance(payload, list):
        return any(
            isinstance(item, dict) and item.get("method") == "tools/call" and "id" in item
            for item in payload
        )
    return False


def _is_initialize_payload(payload: Any) -> bool:
    if isinstance(payload, dict):
        return payload.get("method") == "initialize"
    if isinstance(payload, list):
        return any(isinstance(item, dict) and item.get("method") == "initialize" for item in payload)
    return False


class _CORSMiddleware:
    """Adds CORS headers to all HTTP responses and handles OPTIONS preflight requests."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope["method"] == "OPTIONS":
            response = Response(
                status_code=200,
                headers={**CORS_HEADERS, "Access-Control-Max-Age": "600"},
            )
            await response(scope, receive, send)
            return

        async def _send_with_cors(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for key, value in CORS_HEADERS.items():
                    headers.setdefault(key, value)
            await send(message)

        await self.app(scope, receive, _send_with_cors)


def _render_authorize_error(message: str) -> HTMLResponse:
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Authorization Error - Colab Cloud MCP</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 16px; }}
    .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; max-width: 420px; width: 100%; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.4); }}
    h1 {{ font-size: 1.25rem; margin: 0 0 8px; color: #fca5a5; }}
    p {{ color: #cbd5e1; font-size: 0.9rem; line-height: 1.5; margin: 0; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Invalid Authorization Request</h1>
    <p>{html.escape(message)}</p>
  </div>
</body>
</html>"""
    return HTMLResponse(page, status_code=400)


def _render_authorize_form(
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    code_challenge_method: str,
    error_message: Optional[str] = None,
) -> str:
    error_banner = (
        f'<div class="error">{html.escape(error_message)}</div>'
        if error_message
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Authorize Colab Cloud MCP</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 16px; }}
    .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; max-width: 420px; width: 100%; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.4); }}
    h1 {{ font-size: 1.25rem; margin: 0 0 8px; }}
    p {{ color: #94a3b8; font-size: 0.9rem; line-height: 1.5; margin: 0 0 16px; }}
    .error {{ background: #7f1d1d; border: 1px solid #ef4444; color: #fecaca; border-radius: 8px; padding: 10px 12px; font-size: 0.875rem; margin-bottom: 16px; }}
    input[type="password"] {{ width: 100%; box-sizing: border-box; padding: 10px 12px; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: #f8fafc; margin-bottom: 16px; font-family: monospace; }}
    button {{ width: 100%; padding: 10px 16px; border: none; border-radius: 8px; background: #3b82f6; color: white; font-weight: 600; cursor: pointer; }}
    button:hover {{ background: #2563eb; }}
  </style>
</head>
<body>
  <form class="card" method="post" action="/oauth/authorize">
    <h1>Connect to Google Colab</h1>
    <p>Enter the <code>MCP_TOKEN</code> printed in your Colab notebook to grant MCP access.</p>
    {error_banner}
    <input type="hidden" name="client_id" value="{html.escape(client_id, quote=True)}" />
    <input type="hidden" name="redirect_uri" value="{html.escape(redirect_uri, quote=True)}" />
    <input type="hidden" name="state" value="{html.escape(state, quote=True)}" />
    <input type="hidden" name="code_challenge" value="{html.escape(code_challenge, quote=True)}" />
    <input type="hidden" name="code_challenge_method" value="{html.escape(code_challenge_method, quote=True)}" />
    <input type="password" name="mcp_token" placeholder="Paste MCP_TOKEN..." required autofocus />
    <button type="submit">Authorize Connection</button>
  </form>
</body>
</html>"""


def build_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "get_runtime_status",
            "description": (
                "Inspect the Google Colab VM status: GPU model & VRAM (nvidia-smi), CPU/RAM/Disk, "
                "PyTorch CUDA availability, Google Drive mount status (/content/drive/MyDrive), "
                "active namespace variables, and running background jobs."
            ),
            "annotations": {"readOnlyHint": True},
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "execute_python",
            "description": (
                "Execute Python code statefully inside the persistent Colab namespace. "
                "Variables, imports, and working directory persist across calls. "
                "Matplotlib figures are automatically captured and returned as PNG images. "
                "Keep foreground runs under ~180s; use `start_background_job` for long training or downloads."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code cell to execute."},
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Foreground timeout in seconds (1-220, default 120).",
                        "default": 120,
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
        {
            "name": "reset_python_namespace",
            "description": "Clear persistent Python variables and trigger garbage collection + CUDA VRAM cache cleanup.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "clear_history": {
                        "type": "boolean",
                        "description": "Also clear recorded cell execution history used for notebook export (default False).",
                        "default": False,
                    }
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "run_shell",
            "description": "Run a bash command in the Colab VM (e.g. git, nvidia-smi, ls, df -h) with timeout control.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command string."},
                    "cwd": {
                        "type": "string",
                        "description": "Optional working directory (relative to /content or absolute).",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds (1-220, default 120).",
                        "default": 120,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
        {
            "name": "install_packages",
            "description": "Install or upgrade Python packages in the Colab runtime via pip.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of pip package specs (e.g. ['transformers', 'accelerate>=0.30']).",
                    },
                    "upgrade": {
                        "type": "boolean",
                        "description": "Pass -U to upgrade packages.",
                        "default": False,
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds for pip install (default 210).",
                        "default": 210,
                    },
                },
                "required": ["packages"],
                "additionalProperties": False,
            },
        },
        {
            "name": "start_background_job",
            "description": (
                "Launch a non-blocking background shell/Python job (ideal for multi-hour training, "
                "benchmarks, or downloading Hugging Face weights into /content/drive/MyDrive) "
                "that keeps running even when chat sessions disconnect."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute in the background (e.g. 'python train.py --epochs 50').",
                    },
                    "name": {
                        "type": "string",
                        "description": "Short human-readable job label.",
                        "default": "colab_job",
                    },
                    "cwd": {"type": "string", "description": "Optional working directory."},
                    "log_path": {
                        "type": "string",
                        "description": "Optional custom log file path (e.g. '/content/drive/MyDrive/logs/train.log').",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_job_status",
            "description": "Check status, exit code, elapsed time, and the latest log lines of a background job.",
            "annotations": {"readOnlyHint": True},
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID returned by start_background_job."},
                    "tail_lines": {
                        "type": "integer",
                        "description": "Number of trailing log lines to read (default 80, max 500).",
                        "default": 80,
                    },
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "list_jobs",
            "description": "List all background jobs started in this Colab session and their current status.",
            "annotations": {"readOnlyHint": True},
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "cancel_job",
            "description": "Stop a running background job and its child processes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID to terminate."}
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "list_files",
            "description": "List files and directories in the Colab workspace or mounted Google Drive.",
            "annotations": {"readOnlyHint": True},
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory or file path (default '.').",
                        "default": ".",
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum directory entries to return (default 200).",
                        "default": 200,
                    },
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "read_file",
            "description": "Read UTF-8 text from a file in the Colab VM or Google Drive by line range.",
            "annotations": {"readOnlyHint": True},
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file."},
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed starting line number (default 1).",
                        "default": 1,
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum lines to read (default 300, max 2000).",
                        "default": 300,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "write_file",
            "description": "Create, overwrite, or append UTF-8 text to a file in the Colab VM or Google Drive.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path."},
                    "content": {"type": "string", "description": "UTF-8 text content."},
                    "append": {
                        "type": "boolean",
                        "description": "Append to existing file instead of overwriting.",
                        "default": False,
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        {
            "name": "edit_file",
            "description": "Replace exact text in a UTF-8 file in the Colab VM or Google Drive.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to file to edit."},
                    "old_text": {
                        "type": "string",
                        "description": "Exact text substring to find and replace.",
                    },
                    "new_text": {"type": "string", "description": "Replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences instead of requiring a single unique match (default False).",
                        "default": False,
                    },
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
        {
            "name": "read_image",
            "description": (
                "Read an image file (PNG, JPEG, WebP, GIF) from the Colab VM or Google Drive "
                "and return it as an MCP image content block."
            ),
            "annotations": {"readOnlyHint": True},
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to image file."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "export_session_notebook",
            "description": "Save all executed `execute_python` cells and captured outputs/plots into a Jupyter .ipynb file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Target .ipynb path (e.g. '/content/drive/MyDrive/session.ipynb').",
                        "default": "mcp_session_export.ipynb",
                    }
                },
                "additionalProperties": False,
            },
        },
    ]


class OAuthState:
    """Lightweight single-owner OAuth 2.1 + PKCE + DCR state for Claude & Antigravity connectors."""

    def __init__(self, master_token: str) -> None:
        self.master_token = master_token
        self._signing_key = hashlib.sha256(f"colab-mcp-oauth:{master_token}".encode("utf-8")).digest()
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.auth_codes: Dict[str, Dict[str, Any]] = {}

    def issue_signed_token(self, kind: str = "access", ttl_seconds: int = 86400 * 30) -> str:
        exp = int(time.time()) + ttl_seconds
        nonce = secrets.token_urlsafe(12)
        payload = f"{kind}.{exp}.{nonce}"
        sig = hmac.new(self._signing_key, payload.encode("ascii"), hashlib.sha256).hexdigest()
        return f"cmcp_{payload}.{sig}"

    def verify_token(self, candidate: str, expected_kind: str = "access") -> bool:
        if not candidate:
            return False
        if expected_kind == "access" and _consteq(candidate, self.master_token):
            return True
        if not candidate.startswith("cmcp_"):
            return False
        body = candidate[5:]
        parts = body.rsplit(".", 1)
        if len(parts) != 2:
            return False
        payload, sig = parts
        try:
            expected = hmac.new(
                self._signing_key, payload.encode("ascii"), hashlib.sha256
            ).hexdigest()
        except UnicodeEncodeError:
            return False
        if not _consteq(sig, expected):
            return False
        fields = payload.split(".")
        if len(fields) != 3 or fields[0] != expected_kind:
            return False
        try:
            return int(fields[1]) >= int(time.time())
        except ValueError:
            return False


class MCPProtocolEngine:
    """Dispatches JSON-RPC 2.0 MCP requests against a ColabRuntime instance."""

    def __init__(self, runtime: ColabRuntime) -> None:
        self.runtime = runtime
        self.tools = build_tool_definitions()
        self.sessions: Dict[str, float] = {}

    def ensure_session(self, existing_id: Optional[str] = None) -> str:
        sid = existing_id or uuid.uuid4().hex
        if sid in self.sessions:
            del self.sessions[sid]
        self.sessions[sid] = time.time()
        while len(self.sessions) > MAX_SESSIONS:
            oldest = next(iter(self.sessions))
            del self.sessions[oldest]
        return sid

    async def dispatch_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        loop = asyncio.get_running_loop()
        try:
            if name == "get_runtime_status":
                res = await loop.run_in_executor(None, self.runtime.get_runtime_status)
                return [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}], False

            if name == "execute_python":
                code = str(arguments.get("code", ""))
                timeout_s = int(arguments.get("timeout_seconds", 120))
                payload, images = await loop.run_in_executor(
                    None, self.runtime.execute_python, code, timeout_s
                )
                blocks: List[Dict[str, Any]] = [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}
                ]
                for img_b64 in images:
                    blocks.append({"type": "image", "data": img_b64, "mimeType": "image/png"})
                return blocks, payload.get("status") != "ok"

            if name == "reset_python_namespace":
                clear_history = bool(arguments.get("clear_history", False))
                res = await loop.run_in_executor(
                    None, self.runtime.reset_python_namespace, clear_history
                )
                return (
                    [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}],
                    res.get("status") != "ok",
                )

            if name == "run_shell":
                cmd = str(arguments.get("command", ""))
                cwd = arguments.get("cwd")
                timeout_s = int(arguments.get("timeout_seconds", 120))
                res = await loop.run_in_executor(
                    None, self.runtime.run_shell, cmd, cwd, timeout_s
                )
                return (
                    [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}],
                    res.get("status") != "ok",
                )

            if name == "install_packages":
                pkgs = list(arguments.get("packages") or [])
                upgrade = bool(arguments.get("upgrade", False))
                timeout_s = int(arguments.get("timeout_seconds", 210))
                res = await loop.run_in_executor(
                    None, self.runtime.install_packages, pkgs, upgrade, timeout_s
                )
                return (
                    [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}],
                    res.get("status") != "ok",
                )

            if name == "start_background_job":
                cmd = str(arguments.get("command", ""))
                job_name = str(arguments.get("name", "colab_job"))
                cwd = arguments.get("cwd")
                log_path = arguments.get("log_path")
                res = await loop.run_in_executor(
                    None, self.runtime.start_background_job, cmd, job_name, cwd, log_path
                )
                return [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}], False

            if name == "get_job_status":
                job_id = str(arguments.get("job_id", ""))
                tail_lines = int(arguments.get("tail_lines", 80))
                res = await loop.run_in_executor(
                    None, self.runtime.get_job_status, job_id, tail_lines
                )
                return (
                    [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}],
                    res.get("status") == "not_found",
                )

            if name == "list_jobs":
                res = await loop.run_in_executor(None, self.runtime.list_jobs)
                return [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}], False

            if name == "cancel_job":
                job_id = str(arguments.get("job_id", ""))
                res = await loop.run_in_executor(None, self.runtime.cancel_job, job_id)
                return (
                    [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}],
                    res.get("status") == "not_found",
                )

            if name == "list_files":
                path = str(arguments.get("path", "."))
                max_entries = int(arguments.get("max_entries", 200))
                res = await loop.run_in_executor(
                    None, self.runtime.list_files, path, max_entries
                )
                return (
                    [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}],
                    res.get("status") != "ok",
                )

            if name == "read_file":
                path = str(arguments.get("path", ""))
                start_line = int(arguments.get("start_line", 1))
                max_lines = int(arguments.get("max_lines", 300))
                res = await loop.run_in_executor(
                    None, self.runtime.read_file, path, start_line, max_lines
                )
                return (
                    [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}],
                    res.get("status") != "ok",
                )

            if name == "write_file":
                path = str(arguments.get("path", ""))
                content = str(arguments.get("content", ""))
                append = bool(arguments.get("append", False))
                res = await loop.run_in_executor(
                    None, self.runtime.write_file, path, content, append
                )
                return (
                    [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}],
                    res.get("status") != "ok",
                )

            if name == "edit_file":
                path = str(arguments.get("path", ""))
                old_text = str(arguments.get("old_text", ""))
                new_text = str(arguments.get("new_text", ""))
                replace_all = bool(arguments.get("replace_all", False))
                res = await loop.run_in_executor(
                    None, self.runtime.edit_file, path, old_text, new_text, replace_all
                )
                return (
                    [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}],
                    res.get("status") != "ok",
                )

            if name == "read_image":
                path = str(arguments.get("path", ""))
                res = await loop.run_in_executor(None, self.runtime.read_image, path)
                if res.get("status") == "ok":
                    meta = dict(res)
                    b64 = meta.pop("data_base64")
                    mime_type = meta["mime_type"]
                    return (
                        [
                            {"type": "text", "text": json.dumps(meta, ensure_ascii=False, indent=2)},
                            {"type": "image", "data": b64, "mimeType": mime_type},
                        ],
                        False,
                    )
                return (
                    [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}],
                    True,
                )

            if name == "export_session_notebook":
                out_path = str(arguments.get("output_path", "mcp_session_export.ipynb"))
                res = await loop.run_in_executor(
                    None, self.runtime.export_session_notebook, out_path
                )
                return (
                    [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}],
                    res.get("status") != "ok",
                )

            return [{"type": "text", "text": f"Unknown tool: {name}"}], True
        except Exception as exc:
            return [{"type": "text", "text": f"Tool execution error: {exc}"}], True

    async def handle_rpc_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes a single JSON-RPC 2.0 message; returns None for notifications and responses."""
        if "method" not in msg:
            if "result" in msg or "error" in msg:
                return None
            return {
                "jsonrpc": "2.0",
                "id": msg.get("id"),
                "error": {"code": -32600, "message": "Invalid Request: missing method"},
            }

        if "id" not in msg:
            return None

        msg_id = msg.get("id")
        method = msg.get("method")
        if not isinstance(method, str) or not method:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32600, "message": "Invalid Request: method must be a string"},
            }

        raw_params = msg.get("params")
        if raw_params is None:
            params: Dict[str, Any] = {}
        elif isinstance(raw_params, dict):
            params = raw_params
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": "Invalid params: expected object"},
            }

        if method == "initialize":
            requested_ver = str(params.get("protocolVersion", DEFAULT_PROTOCOL_VERSION))
            negotiated = (
                requested_ver
                if requested_ver in SUPPORTED_PROTOCOL_VERSIONS
                else DEFAULT_PROTOCOL_VERSION
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": negotiated,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                    },
                    "serverInfo": {
                        "name": "colab-cloud-mcp",
                        "version": __version__,
                    },
                    "instructions": SERVER_INSTRUCTIONS,
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": self.tools}}

        if method == "tools/call":
            tool_name = params.get("name")
            if not tool_name or not isinstance(tool_name, str):
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32602, "message": "Missing tool name"},
                }
            tool_args = params.get("arguments")
            if tool_args is None:
                tool_args = {}
            elif not isinstance(tool_args, dict):
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32602, "message": "Invalid tool arguments: expected object"},
                }
            content_blocks, is_error = await self.dispatch_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": content_blocks, "isError": is_error},
            }

        if method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "resources": [
                        {
                            "uri": "colab://runtime/status",
                            "name": "Colab Runtime Status",
                            "description": "Live hardware, GPU, Google Drive, and job status of the Colab VM.",
                            "mimeType": "application/json",
                        }
                    ]
                },
            }

        if method == "resources/read":
            uri = params.get("uri")
            if uri == "colab://runtime/status":
                loop = asyncio.get_running_loop()
                status_data = await loop.run_in_executor(None, self.runtime.get_runtime_status)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(status_data, ensure_ascii=False, indent=2),
                            }
                        ]
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": f"Unknown resource URI: {uri}"},
            }

        if method == "prompts/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"prompts": []}}

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def create_app(
    token: str,
    runtime: Optional[ColabRuntime] = None,
    public_url: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> Starlette:
    """Creates the Starlette ASGI app exposing Streamable HTTP, SSE, Secret-Path, and OAuth 2.1."""
    if not token:
        raise ValueError("A non-empty secret token is required to secure colab-cloud-mcp.")

    rt = runtime or ColabRuntime()
    engine = MCPProtocolEngine(rt)
    oauth = OAuthState(token)
    sse_queues: Dict[str, "asyncio.Queue[Optional[str]]"] = {}

    def _is_authorized(request: Request) -> bool:
        path_tok = request.path_params.get("path_token")
        if path_tok and _consteq(path_tok, token):
            return True

        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            candidate = auth_header[7:].strip()
            if oauth.verify_token(candidate, expected_kind="access"):
                return True

        api_key = request.headers.get("x-api-key", "").strip()
        if api_key and oauth.verify_token(api_key, expected_kind="access"):
            return True

        return False

    def _unauthorized_response(request: Request) -> JSONResponse:
        base = _get_external_base_url(request, app.state.public_url)
        www_auth = f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource"'
        return JSONResponse(
            {
                "error": "unauthorized",
                "message": "Valid Bearer token, OAuth token, or secret URL path required.",
            },
            status_code=401,
            headers={"WWW-Authenticate": www_auth},
        )

    async def _execute_rpc_payload(payload: Any) -> Any:
        if isinstance(payload, dict):
            return await engine.handle_rpc_message(payload)
        responses: List[Dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                resp = await engine.handle_rpc_message(item)
                if resp is not None:
                    responses.append(resp)
            else:
                responses.append(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32600, "message": "Invalid Request"},
                    }
                )
        return responses

    async def health_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "colab-cloud-mcp",
                "version": __version__,
                "instance_id": app.state.instance_id,
                "transports": ["streamable-http", "sse"],
            }
        )

    async def mcp_streamable_endpoint(request: Request) -> Response:
        if not _is_authorized(request):
            return _unauthorized_response(request)

        incoming_sid = request.headers.get("mcp-session-id")
        base_headers: Dict[str, str] = {"Cache-Control": "no-cache"}

        if request.method == "DELETE":
            if incoming_sid:
                engine.sessions.pop(incoming_sid, None)
                base_headers["Mcp-Session-Id"] = incoming_sid
            return Response(status_code=200, headers=base_headers)

        if request.method == "GET":
            if incoming_sid:
                base_headers["Mcp-Session-Id"] = engine.ensure_session(incoming_sid)
            accept = request.headers.get("accept", "").lower()
            if "text/event-stream" not in accept and "*/*" not in accept and accept != "":
                return JSONResponse(
                    {"error": "GET /mcp requires Accept: text/event-stream"},
                    status_code=406,
                )

            async def _keepalive_stream():
                yield ": connected\n\n"
                try:
                    while True:
                        interval = float(getattr(app.state, "keepalive_interval", 15.0))
                        await asyncio.sleep(interval)
                        yield ": keepalive\n\n"
                except asyncio.CancelledError:
                    return

            return StreamingResponse(
                _keepalive_stream(),
                media_type="text/event-stream",
                headers={**base_headers, "X-Accel-Buffering": "no"},
            )

        try:
            payload = await request.json()
        except Exception:
            if incoming_sid:
                base_headers["Mcp-Session-Id"] = engine.ensure_session(incoming_sid)
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error: invalid JSON"},
                },
                status_code=400,
                headers=base_headers,
            )

        if incoming_sid:
            base_headers["Mcp-Session-Id"] = engine.ensure_session(incoming_sid)
        elif _is_initialize_payload(payload):
            base_headers["Mcp-Session-Id"] = engine.ensure_session(None)

        if isinstance(payload, list) and len(payload) == 0:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request: empty batch"},
                },
                status_code=400,
                headers=base_headers,
            )

        if not isinstance(payload, (dict, list)):
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid JSON-RPC request"},
                },
                status_code=400,
                headers=base_headers,
            )

        accept = request.headers.get("accept", "application/json").lower()

        if "text/event-stream" in accept and _contains_tool_call(payload):
            progress_token = None
            if isinstance(payload, dict) and payload.get("method") == "tools/call":
                params = payload.get("params")
                if isinstance(params, dict):
                    meta = params.get("_meta")
                    if isinstance(meta, dict) and meta.get("progressToken") is not None:
                        progress_token = meta.get("progressToken")

            async def _stream_rpc():
                yield ": connected\n\n"
                task = asyncio.create_task(_execute_rpc_payload(payload))
                progress_step = 0
                try:
                    while True:
                        interval = float(getattr(app.state, "keepalive_interval", 15.0))
                        done, _ = await asyncio.wait({task}, timeout=interval)
                        if task in done:
                            break
                        yield ": keepalive\n\n"
                        if progress_token is not None:
                            progress_step += 1
                            notif = {
                                "jsonrpc": "2.0",
                                "method": "notifications/progress",
                                "params": {
                                    "progressToken": progress_token,
                                    "progress": progress_step,
                                    "message": "Tool still running on Colab...",
                                },
                            }
                            yield f"event: message\ndata: {json.dumps(notif, ensure_ascii=False)}\n\n"
                    result_payload = task.result()
                    if result_payload is not None and result_payload != []:
                        yield (
                            "event: message\n"
                            f"data: {json.dumps(result_payload, ensure_ascii=False)}\n\n"
                        )
                except asyncio.CancelledError:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise
                finally:
                    if not task.done():
                        task.cancel()

            return StreamingResponse(
                _stream_rpc(),
                media_type="text/event-stream",
                headers={**base_headers, "X-Accel-Buffering": "no"},
            )

        result_payload = await _execute_rpc_payload(payload)
        if result_payload is None or result_payload == []:
            return Response(status_code=202, headers=base_headers)

        if "text/event-stream" in accept and "application/json" not in accept:
            body = f"event: message\ndata: {json.dumps(result_payload, ensure_ascii=False)}\n\n"
            return Response(
                content=body,
                media_type="text/event-stream",
                headers={**base_headers, "X-Accel-Buffering": "no"},
            )
        return JSONResponse(result_payload, headers=base_headers)

    async def legacy_sse_endpoint(request: Request) -> Response:
        if not _is_authorized(request):
            return _unauthorized_response(request)

        sse_sid = uuid.uuid4().hex
        queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        sse_queues[sse_sid] = queue
        path_tok = request.path_params.get("path_token")
        msg_endpoint = (
            f"/s/{path_tok}/messages?session_id={sse_sid}"
            if path_tok
            else f"/messages?session_id={sse_sid}"
        )

        async def _event_gen():
            yield f"event: endpoint\ndata: {msg_endpoint}\n\n"
            try:
                while True:
                    interval = float(getattr(app.state, "keepalive_interval", 15.0))
                    try:
                        data = await asyncio.wait_for(queue.get(), timeout=interval)
                        if data is None:
                            break
                        yield f"event: message\ndata: {data}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                sse_queues.pop(sse_sid, None)

        return StreamingResponse(
            _event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def legacy_messages_endpoint(request: Request) -> Response:
        if not _is_authorized(request):
            return _unauthorized_response(request)

        sse_sid = request.query_params.get("session_id", "")
        queue = sse_queues.get(sse_sid)
        if not queue:
            return JSONResponse({"error": "Unknown SSE session_id"}, status_code=404)

        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error: invalid JSON"},
                },
                status_code=400,
            )

        if isinstance(payload, list) and len(payload) == 0:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request: empty batch"},
                },
                status_code=400,
            )

        if not isinstance(payload, (dict, list)):
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid JSON-RPC request"},
                },
                status_code=400,
            )

        result_payload = await _execute_rpc_payload(payload)
        if result_payload is not None and result_payload != []:
            await queue.put(json.dumps(result_payload, ensure_ascii=False))
        return Response(status_code=202)

    async def oauth_protected_resource(request: Request) -> JSONResponse:
        base = _get_external_base_url(request, app.state.public_url)
        return JSONResponse(
            {
                "resource": f"{base}/mcp",
                "authorization_servers": [base],
                "bearer_methods_supported": ["header"],
            }
        )

    async def oauth_authorization_server(request: Request) -> JSONResponse:
        base = _get_external_base_url(request, app.state.public_url)
        return JSONResponse(
            {
                "issuer": base,
                "authorization_endpoint": f"{base}/oauth/authorize",
                "token_endpoint": f"{base}/oauth/token",
                "registration_endpoint": f"{base}/oauth/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": [
                    "none",
                    "client_secret_post",
                    "client_secret_basic",
                ],
                "code_challenge_methods_supported": ["S256"],
                "client_id_metadata_document_supported": False,
            }
        )

    async def oauth_register(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid_client_metadata"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "invalid_client_metadata"}, status_code=400)

        redirect_uris = body.get("redirect_uris")
        if (
            not isinstance(redirect_uris, list)
            or not redirect_uris
            or not all(_is_valid_redirect_uri(uri) for uri in redirect_uris)
        ):
            return JSONResponse({"error": "invalid_redirect_uri"}, status_code=400)

        client_id = f"client_{secrets.token_hex(8)}"
        record = {
            "client_id": client_id,
            "client_name": str(body.get("client_name", "MCP Client")),
            "redirect_uris": list(redirect_uris),
            "grant_types": body.get("grant_types", ["authorization_code", "refresh_token"]),
            "response_types": body.get("response_types", ["code"]),
            "token_endpoint_auth_method": body.get("token_endpoint_auth_method", "none"),
        }
        while len(oauth.clients) >= MAX_OAUTH_CLIENTS:
            oldest_cid = next(iter(oauth.clients))
            oauth.clients.pop(oldest_cid, None)
        oauth.clients[client_id] = record
        return JSONResponse(record, status_code=201)

    def _validate_authorize_params(
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        code_challenge_method: str,
        response_type: str = "",
    ) -> Optional[str]:
        if response_type and response_type != "code":
            return "Unsupported response_type; only 'code' is supported."
        client = oauth.clients.get(client_id)
        if not client:
            return "Unknown or unregistered client_id."
        if redirect_uri not in client["redirect_uris"]:
            return "Invalid or unregistered redirect_uri."
        if not code_challenge:
            return "Missing required PKCE code_challenge."
        if code_challenge_method != "S256":
            return "Unsupported code_challenge_method; S256 is required."
        return None

    async def oauth_authorize(request: Request) -> Response:
        if request.method == "GET":
            q = request.query_params
            client_id = q.get("client_id", "")
            redirect_uri = q.get("redirect_uri", "")
            state = q.get("state", "")
            code_challenge = q.get("code_challenge", "")
            code_challenge_method = q.get("code_challenge_method", "")
            response_type = q.get("response_type", "")
            err = _validate_authorize_params(
                client_id,
                redirect_uri,
                code_challenge,
                code_challenge_method,
                response_type=response_type,
            )
            if err:
                return _render_authorize_error(err)
            return HTMLResponse(
                _render_authorize_form(
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    state=state,
                    code_challenge=code_challenge,
                    code_challenge_method=code_challenge_method,
                )
            )

        form = await _parse_urlencoded_form(request)
        client_id = form.get("client_id", "")
        redirect_uri = form.get("redirect_uri", "")
        state = form.get("state", "")
        code_challenge = form.get("code_challenge", "")
        code_challenge_method = form.get("code_challenge_method", "")

        err = _validate_authorize_params(
            client_id, redirect_uri, code_challenge, code_challenge_method
        )
        if err:
            return _render_authorize_error(err)

        submitted = form.get("mcp_token", "").strip()
        if not _consteq(submitted, token):
            return HTMLResponse(
                _render_authorize_form(
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    state=state,
                    code_challenge=code_challenge,
                    code_challenge_method=code_challenge_method,
                    error_message="Invalid MCP_TOKEN, please try again.",
                ),
                status_code=403,
            )

        now = time.time()
        expired = [c for c, rec in oauth.auth_codes.items() if rec["expires_at"] < now]
        for c in expired:
            oauth.auth_codes.pop(c, None)
        while len(oauth.auth_codes) >= MAX_OAUTH_AUTH_CODES:
            oldest_code = next(iter(oauth.auth_codes))
            oauth.auth_codes.pop(oldest_code, None)

        code = secrets.token_urlsafe(24)
        oauth.auth_codes[code] = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "expires_at": now + 300,
        }
        sep = "&" if "?" in redirect_uri else "?"
        location = f"{redirect_uri}{sep}code={urllib.parse.quote(code, safe='')}"
        if state:
            location += f"&state={urllib.parse.quote(state, safe='')}"
        return Response(status_code=302, headers={"Location": location})

    async def oauth_token(request: Request) -> JSONResponse:
        ctype = request.headers.get("content-type", "").lower()
        if "application/json" in ctype:
            try:
                raw_data = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid_request"}, status_code=400)
            if not isinstance(raw_data, dict):
                return JSONResponse({"error": "invalid_request"}, status_code=400)
            data: Dict[str, Any] = dict(raw_data)
        else:
            data = await _parse_urlencoded_form(request)

        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("basic "):
            try:
                decoded = base64.b64decode(auth_header[6:].strip(), validate=True).decode("utf-8")
            except Exception:
                return JSONResponse({"error": "invalid_client"}, status_code=401)
            if ":" not in decoded:
                return JSONResponse({"error": "invalid_client"}, status_code=401)
            raw_cid, _ = decoded.split(":", 1)
            basic_cid = urllib.parse.unquote_plus(raw_cid)
            body_cid = str(data.get("client_id", "") or "")
            if body_cid and basic_cid and body_cid != basic_cid:
                return JSONResponse({"error": "invalid_request"}, status_code=400)
            if basic_cid:
                data["client_id"] = basic_cid

        grant_type = data.get("grant_type")
        if grant_type == "authorization_code":
            code = str(data.get("code", "") or "")
            record = oauth.auth_codes.pop(code, None)
            if not record or record["expires_at"] < time.time():
                return JSONResponse({"error": "invalid_grant"}, status_code=400)

            req_client_id = str(data.get("client_id", "") or "")
            if not req_client_id or not _consteq(req_client_id, record["client_id"]):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)

            if "redirect_uri" in data:
                if str(data.get("redirect_uri") or "") != record["redirect_uri"]:
                    return JSONResponse({"error": "invalid_grant"}, status_code=400)

            challenge = record.get("code_challenge", "")
            verifier = str(data.get("code_verifier", "") or "")
            if (
                not challenge
                or record.get("code_challenge_method") != "S256"
                or not verifier
            ):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)

            try:
                verifier_bytes = verifier.encode("ascii")
            except UnicodeEncodeError:
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            digest = hashlib.sha256(verifier_bytes).digest()
            computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
            if not _consteq(computed, challenge):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)

            access_token = oauth.issue_signed_token("access", ttl_seconds=86400 * 30)
            refresh_token = oauth.issue_signed_token("refresh", ttl_seconds=86400 * 90)
            return JSONResponse(
                {
                    "access_token": access_token,
                    "token_type": "Bearer",
                    "expires_in": 86400 * 30,
                    "refresh_token": refresh_token,
                },
                headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
            )

        if grant_type == "refresh_token":
            rt_tok = str(data.get("refresh_token", "") or "")
            if not oauth.verify_token(rt_tok, expected_kind="refresh"):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            access_token = oauth.issue_signed_token("access", ttl_seconds=86400 * 30)
            return JSONResponse(
                {
                    "access_token": access_token,
                    "token_type": "Bearer",
                    "expires_in": 86400 * 30,
                    "refresh_token": rt_tok,
                },
                headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
            )

        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    routes = [
        Route("/health", health_endpoint, methods=["GET"]),
        Route("/mcp", mcp_streamable_endpoint, methods=["GET", "POST", "DELETE"]),
        Route("/mcp/", mcp_streamable_endpoint, methods=["GET", "POST", "DELETE"]),
        Route("/s/{path_token}/mcp", mcp_streamable_endpoint, methods=["GET", "POST", "DELETE"]),
        Route("/s/{path_token}/mcp/", mcp_streamable_endpoint, methods=["GET", "POST", "DELETE"]),
        Route("/sse", legacy_sse_endpoint, methods=["GET"]),
        Route("/messages", legacy_messages_endpoint, methods=["POST"]),
        Route("/s/{path_token}/sse", legacy_sse_endpoint, methods=["GET"]),
        Route("/s/{path_token}/messages", legacy_messages_endpoint, methods=["POST"]),
        Route("/.well-known/oauth-protected-resource", oauth_protected_resource, methods=["GET"]),
        Route("/.well-known/oauth-protected-resource/mcp", oauth_protected_resource, methods=["GET"]),
        Route("/.well-known/oauth-authorization-server", oauth_authorization_server, methods=["GET"]),
        Route("/.well-known/openid-configuration", oauth_authorization_server, methods=["GET"]),
        Route("/oauth/register", oauth_register, methods=["POST"]),
        Route("/oauth/authorize", oauth_authorize, methods=["GET", "POST"]),
        Route("/oauth/token", oauth_token, methods=["POST"]),
    ]

    app = Starlette(
        debug=False,
        routes=routes,
        middleware=[Middleware(_CORSMiddleware)],
    )
    app.state.runtime = rt
    app.state.engine = engine
    app.state.token = token
    app.state.public_url = public_url
    app.state.instance_id = instance_id or uuid.uuid4().hex
    app.state.keepalive_interval = 15.0
    return app
