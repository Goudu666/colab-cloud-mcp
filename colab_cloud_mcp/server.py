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

SERVER_INSTRUCTIONS = (
    "Connected directly to a remote Google Colab VM via colab-cloud-mcp. "
    "Python state in `execute_python` persists across calls. "
    "For tasks expected to run longer than 2 minutes (model training, evaluation sweeps, "
    "large weight downloads to Google Drive), always use `start_background_job` and poll "
    "with `get_job_status` so cloud chat HTTP timeouts are never exceeded."
)


def _get_external_base_url(request: Request, configured_public_url: Optional[str] = None) -> str:
    if configured_public_url:
        return configured_public_url.rstrip("/")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme or "https"
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost"
    return f"{proto}://{host}".rstrip("/")


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
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
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

    def verify_token(self, candidate: str) -> bool:
        if not candidate:
            return False
        if hmac.compare_digest(candidate, self.master_token):
            return True
        if not candidate.startswith("cmcp_"):
            return False
        body = candidate[5:]
        parts = body.rsplit(".", 1)
        if len(parts) != 2:
            return False
        payload, sig = parts
        expected = hmac.new(self._signing_key, payload.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False
        fields = payload.split(".")
        if len(fields) != 3 or fields[0] != "access":
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
        self.sessions[sid] = time.time()
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
                res = await loop.run_in_executor(None, self.runtime.reset_python_namespace)
                return [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}], False

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
                res = await loop.run_in_executor(
                    None, self.runtime.install_packages, pkgs, upgrade
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
                return [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}], False

            if name == "export_session_notebook":
                out_path = str(arguments.get("output_path", "mcp_session_export.ipynb"))
                res = await loop.run_in_executor(
                    None, self.runtime.export_session_notebook, out_path
                )
                return [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}], False

            return [{"type": "text", "text": f"Unknown tool: {name}"}], True
        except Exception as exc:
            return [{"type": "text", "text": f"Tool execution error: {exc}"}], True

    async def handle_rpc_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes a single JSON-RPC 2.0 message; returns None for notifications."""
        msg_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}

        if "id" not in msg:
            return None

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
            if not tool_name:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32602, "message": "Missing tool name"},
                }
            tool_args = params.get("arguments") or {}
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
        if path_tok and hmac.compare_digest(path_tok, token):
            return True

        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            candidate = auth_header[7:].strip()
            if oauth.verify_token(candidate):
                return True

        api_key = request.headers.get("x-api-key", "").strip()
        if api_key and oauth.verify_token(api_key):
            return True

        return False

    def _unauthorized_response(request: Request) -> JSONResponse:
        base = _get_external_base_url(request, public_url)
        www_auth = f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource"'
        return JSONResponse(
            {"error": "unauthorized", "message": "Valid Bearer token, OAuth token, or secret URL path required."},
            status_code=401,
            headers={"WWW-Authenticate": www_auth},
        )

    async def health_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "colab-cloud-mcp",
                "version": __version__,
                "transports": ["streamable-http", "sse"],
            }
        )

    async def mcp_streamable_endpoint(request: Request) -> Response:
        if not _is_authorized(request):
            return _unauthorized_response(request)

        incoming_sid = request.headers.get("mcp-session-id")
        session_id = engine.ensure_session(incoming_sid)
        base_headers = {
            "Mcp-Session-Id": session_id,
            "Cache-Control": "no-cache",
        }

        if request.method == "DELETE":
            engine.sessions.pop(session_id, None)
            return Response(status_code=200, headers=base_headers)

        if request.method == "GET":
            accept = request.headers.get("accept", "")
            if "text/event-stream" not in accept and "*/*" not in accept and accept != "":
                return JSONResponse(
                    {"error": "GET /mcp requires Accept: text/event-stream"},
                    status_code=406,
                )

            async def _keepalive_stream():
                yield ": connected\n\n"
                try:
                    while True:
                        await asyncio.sleep(20)
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
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error: invalid JSON"},
                },
                status_code=400,
                headers=base_headers,
            )

        if isinstance(payload, list):
            responses = []
            for item in payload:
                if isinstance(item, dict):
                    resp = await engine.handle_rpc_message(item)
                    if resp is not None:
                        responses.append(resp)
            if not responses:
                return Response(status_code=202, headers=base_headers)
            return JSONResponse(responses, headers=base_headers)

        if isinstance(payload, dict):
            resp = await engine.handle_rpc_message(payload)
            if resp is None:
                return Response(status_code=202, headers=base_headers)

            accept = request.headers.get("accept", "application/json")
            if "text/event-stream" in accept and "application/json" not in accept:
                body = f"event: message\ndata: {json.dumps(resp, ensure_ascii=False)}\n\n"
                return Response(
                    content=body,
                    media_type="text/event-stream",
                    headers={**base_headers, "X-Accel-Buffering": "no"},
                )
            return JSONResponse(resp, headers=base_headers)

        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid JSON-RPC request"},
            },
            status_code=400,
            headers=base_headers,
        )

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
                    try:
                        data = await asyncio.wait_for(queue.get(), timeout=20)
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

        payload = await request.json()
        if isinstance(payload, dict):
            resp = await engine.handle_rpc_message(payload)
            if resp is not None:
                await queue.put(json.dumps(resp, ensure_ascii=False))
        return Response(status_code=202)

    async def oauth_protected_resource(request: Request) -> JSONResponse:
        base = _get_external_base_url(request, public_url)
        return JSONResponse(
            {
                "resource": f"{base}/mcp",
                "authorization_servers": [base],
                "bearer_methods_supported": ["header"],
            }
        )

    async def oauth_authorization_server(request: Request) -> JSONResponse:
        base = _get_external_base_url(request, public_url)
        return JSONResponse(
            {
                "issuer": base,
                "authorization_endpoint": f"{base}/oauth/authorize",
                "token_endpoint": f"{base}/oauth/token",
                "registration_endpoint": f"{base}/oauth/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
                "code_challenge_methods_supported": ["S256"],
                "client_id_metadata_document_supported": True,
            }
        )

    async def oauth_register(request: Request) -> JSONResponse:
        body = await request.json()
        client_id = f"client_{secrets.token_hex(8)}"
        redirect_uris = body.get("redirect_uris") or []
        record = {
            "client_id": client_id,
            "client_name": body.get("client_name", "MCP Client"),
            "redirect_uris": redirect_uris,
            "grant_types": body.get("grant_types", ["authorization_code", "refresh_token"]),
            "response_types": body.get("response_types", ["code"]),
            "token_endpoint_auth_method": body.get("token_endpoint_auth_method", "none"),
        }
        oauth.clients[client_id] = record
        return JSONResponse(record, status_code=201)

    async def oauth_authorize(request: Request) -> Response:
        if request.method == "GET":
            q = request.query_params
            client_id = q.get("client_id", "")
            redirect_uri = q.get("redirect_uri", "")
            state = q.get("state", "")
            code_challenge = q.get("code_challenge", "")
            code_challenge_method = q.get("code_challenge_method", "S256")
            page = f"""<!doctype html>
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
    input[type="password"] {{ width: 100%; box-sizing: border-box; padding: 10px 12px; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: #f8fafc; margin-bottom: 16px; font-family: monospace; }}
    button {{ width: 100%; padding: 10px 16px; border: none; border-radius: 8px; background: #3b82f6; color: white; font-weight: 600; cursor: pointer; }}
    button:hover {{ background: #2563eb; }}
  </style>
</head>
<body>
  <form class="card" method="post" action="/oauth/authorize">
    <h1>Connect to Google Colab</h1>
    <p>Enter the <code>MCP_TOKEN</code> printed in your Colab notebook to grant MCP access.</p>
    <input type="hidden" name="client_id" value="{html.escape(client_id)}" />
    <input type="hidden" name="redirect_uri" value="{html.escape(redirect_uri)}" />
    <input type="hidden" name="state" value="{html.escape(state)}" />
    <input type="hidden" name="code_challenge" value="{html.escape(code_challenge)}" />
    <input type="hidden" name="code_challenge_method" value="{html.escape(code_challenge_method)}" />
    <input type="password" name="mcp_token" placeholder="Paste MCP_TOKEN..." required autofocus />
    <button type="submit">Authorize Connection</button>
  </form>
</body>
</html>"""
            return HTMLResponse(page)

        form = await request.form()
        submitted = str(form.get("mcp_token", "")).strip()
        if not hmac.compare_digest(submitted, token):
            return HTMLResponse("Invalid MCP_TOKEN.", status_code=403)

        redirect_uri = str(form.get("redirect_uri", ""))
        state = str(form.get("state", ""))
        code = secrets.token_urlsafe(24)
        oauth.auth_codes[code] = {
            "client_id": str(form.get("client_id", "")),
            "redirect_uri": redirect_uri,
            "code_challenge": str(form.get("code_challenge", "")),
            "code_challenge_method": str(form.get("code_challenge_method", "S256")),
            "expires_at": time.time() + 300,
        }
        sep = "&" if "?" in redirect_uri else "?"
        location = f"{redirect_uri}{sep}code={urllib.parse.quote(code)}"
        if state:
            location += f"&state={urllib.parse.quote(state)}"
        return Response(status_code=302, headers={"Location": location})

    async def oauth_token(request: Request) -> JSONResponse:
        ctype = request.headers.get("content-type", "")
        if "application/json" in ctype:
            data = await request.json()
        else:
            form = await request.form()
            data = dict(form)

        grant_type = data.get("grant_type")
        if grant_type == "authorization_code":
            code = str(data.get("code", ""))
            record = oauth.auth_codes.pop(code, None)
            if not record or record["expires_at"] < time.time():
                return JSONResponse({"error": "invalid_grant"}, status_code=400)

            challenge = record.get("code_challenge")
            if challenge:
                verifier = str(data.get("code_verifier", ""))
                digest = hashlib.sha256(verifier.encode("ascii")).digest()
                computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
                if not hmac.compare_digest(computed, challenge):
                    return JSONResponse({"error": "invalid_grant"}, status_code=400)

            access_token = oauth.issue_signed_token("access", ttl_seconds=86400 * 30)
            refresh_token = oauth.issue_signed_token("access", ttl_seconds=86400 * 90)
            return JSONResponse(
                {
                    "access_token": access_token,
                    "token_type": "Bearer",
                    "expires_in": 86400 * 30,
                    "refresh_token": refresh_token,
                }
            )

        if grant_type == "refresh_token":
            rt_tok = str(data.get("refresh_token", ""))
            if not oauth.verify_token(rt_tok):
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            access_token = oauth.issue_signed_token("access", ttl_seconds=86400 * 30)
            return JSONResponse(
                {
                    "access_token": access_token,
                    "token_type": "Bearer",
                    "expires_in": 86400 * 30,
                    "refresh_token": rt_tok,
                }
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
        Route("/oauth/register", oauth_register, methods=["POST"]),
        Route("/oauth/authorize", oauth_authorize, methods=["GET", "POST"]),
        Route("/oauth/token", oauth_token, methods=["POST"]),
    ]

    app = Starlette(debug=False, routes=routes)
    app.state.runtime = rt
    app.state.engine = engine
    app.state.token = token
    return app
