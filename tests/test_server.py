"""Comprehensive integration and edge-case test suite for colab-cloud-mcp."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.parse
from typing import Any, Dict, List, Tuple

import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from colab_cloud_mcp import ActiveColabMCP, ColabRuntime, create_app, start_colab_mcp

EXPECTED_TOOLS = {
    "get_runtime_status",
    "execute_python",
    "reset_python_namespace",
    "run_shell",
    "install_packages",
    "start_background_job",
    "get_job_status",
    "list_jobs",
    "cancel_job",
    "list_files",
    "read_file",
    "write_file",
    "edit_file",
    "read_image",
    "export_session_notebook",
}


def _pkce_challenge(verifier: str) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _parse_sse_events(raw_text: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Returns (comment_lines, [(event_name, data_str), ...]) from an SSE stream body."""
    comments: List[str] = []
    events: List[Tuple[str, str]] = []
    current_event = "message"
    current_data: List[str] = []

    for line in raw_text.splitlines():
        if line.startswith(":"):
            comments.append(line.strip())
        elif line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_data.append(line[len("data:") :].strip())
        elif line == "":
            if current_data:
                events.append((current_event, "\n".join(current_data)))
                current_event = "message"
                current_data = []
    if current_data:
        events.append((current_event, "\n".join(current_data)))
    return comments, events


def test_core_streamable_http_cors_and_all_tools() -> None:
    secret_token = "test_super_secret_colab_token_2026"
    base_url = "https://localhost:8765"

    with tempfile.TemporaryDirectory(prefix="colab_mcp_core_") as tmpdir:
        rt = ColabRuntime(workdir=tmpdir)
        app = create_app(
            token=secret_token,
            runtime=rt,
            public_url=base_url,
            instance_id="inst_core_001",
        )
        transport = httpx.ASGITransport(app=app)

        async def _run() -> None:
            async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
                # /health includes instance_id and CORS headers
                r = await client.get("/health")
                assert r.status_code == 200, r.text
                health = r.json()
                assert health["status"] == "ok"
                assert health["instance_id"] == "inst_core_001"
                assert r.headers.get("access-control-allow-origin") == "*"
                assert r.headers.get("access-control-expose-headers") == "Mcp-Session-Id"

                # CORS OPTIONS preflight
                r_opt = await client.options("/mcp")
                assert r_opt.status_code == 200
                assert r_opt.headers.get("access-control-allow-origin") == "*"
                assert "POST" in r_opt.headers.get("access-control-allow-methods", "")
                assert "Mcp-Session-Id" in r_opt.headers.get("access-control-allow-headers", "")
                assert r_opt.headers.get("access-control-expose-headers") == "Mcp-Session-Id"

                # Unauthorized -> 401 with WWW-Authenticate resource_metadata
                r_unauth = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
                assert r_unauth.status_code == 401
                www_auth = r_unauth.headers.get("www-authenticate", "")
                assert f'resource_metadata="{base_url}/.well-known/oauth-protected-resource"' in www_auth

                # Dynamic app.state.public_url sync
                app.state.public_url = "https://dynamic-sync.trycloudflare.com"
                r_unauth2 = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
                assert r_unauth2.status_code == 401
                assert (
                    'resource_metadata="https://dynamic-sync.trycloudflare.com/.well-known/oauth-protected-resource"'
                    in r_unauth2.headers.get("www-authenticate", "")
                )
                r_pr_sync = await client.get("/.well-known/oauth-protected-resource")
                assert r_pr_sync.json()["resource"] == "https://dynamic-sync.trycloudflare.com/mcp"
                app.state.public_url = base_url

                headers = {"Authorization": f"Bearer {secret_token}"}

                # Stateless ping without Mcp-Session-Id should NOT leak sessions
                assert len(app.state.engine.sessions) == 0
                r_ping = await client.post(
                    "/mcp",
                    headers=headers,
                    json={"jsonrpc": "2.0", "id": 10, "method": "ping"},
                )
                assert r_ping.status_code == 200
                assert r_ping.json()["result"] == {}
                assert "mcp-session-id" not in r_ping.headers
                assert len(app.state.engine.sessions) == 0

                # initialize allocates Mcp-Session-Id
                r_init = await client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "antigravity-test", "version": "2.6"},
                        },
                    },
                )
                assert r_init.status_code == 200, r_init.text
                sid = r_init.headers.get("mcp-session-id")
                assert sid and sid in app.state.engine.sessions
                assert r_init.headers.get("access-control-expose-headers") == "Mcp-Session-Id"

                # Notification -> 202
                r_notif = await client.post(
                    "/mcp",
                    headers={**headers, "Mcp-Session-Id": sid},
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                )
                assert r_notif.status_code == 202

                # Client JSON-RPC response (result / error) -> 202
                r_cresp = await client.post(
                    "/mcp",
                    headers={**headers, "Mcp-Session-Id": sid},
                    json={"jsonrpc": "2.0", "id": 55, "result": {"roots": []}},
                )
                assert r_cresp.status_code == 202
                r_cerr = await client.post(
                    "/mcp",
                    headers={**headers, "Mcp-Session-Id": sid},
                    json={"jsonrpc": "2.0", "id": 56, "error": {"code": -1, "message": "client err"}},
                )
                assert r_cerr.status_code == 202

                # Empty batch [] -> 400 with -32600
                r_empty = await client.post(
                    "/mcp",
                    headers={**headers, "Mcp-Session-Id": sid},
                    json=[],
                )
                assert r_empty.status_code == 400
                assert r_empty.json()["error"]["code"] == -32600

                # Batch with non-dict item -> ping result + -32600 error item
                r_mixed = await client.post(
                    "/mcp",
                    headers={**headers, "Mcp-Session-Id": sid},
                    json=[{"jsonrpc": "2.0", "id": 7, "method": "ping"}, 123],
                )
                assert r_mixed.status_code == 200
                mixed = r_mixed.json()
                assert len(mixed) == 2
                assert mixed[0] == {"jsonrpc": "2.0", "id": 7, "result": {}}
                assert mixed[1]["error"]["code"] == -32600

                # Malformed JSON -> 400 -32700
                r_bad = await client.post(
                    "/mcp",
                    headers={**headers, "Content-Type": "application/json"},
                    content=b"{not-valid-json",
                )
                assert r_bad.status_code == 400
                assert r_bad.json()["error"]["code"] == -32700

                # tools/list returns all 15 tools
                r_tools = await client.post(
                    "/mcp",
                    headers={**headers, "Mcp-Session-Id": sid},
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                )
                assert r_tools.status_code == 200
                tools = r_tools.json()["result"]["tools"]
                tool_names = {t["name"] for t in tools}
                assert len(tools) == 15, f"Expected 15 tools, got {len(tools)}: {sorted(tool_names)}"
                assert tool_names == EXPECTED_TOOLS

                # Secret-path /s/<token>/mcp works without Authorization header; wrong token rejected
                r_secret = await client.post(
                    f"/s/{secret_token}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": "execute_python",
                            "arguments": {"code": "x_state = 41\nprint('hello colab')\nx_state + 1"},
                        },
                    },
                )
                assert r_secret.status_code == 200
                meta = json.loads(r_secret.json()["result"]["content"][0]["text"])
                assert meta["status"] == "ok" and meta["result"] == "42"
                assert "hello colab" in meta["stdout"]

                r_bad_secret = await client.post(
                    "/s/wrong_token/mcp",
                    json={"jsonrpc": "2.0", "id": 4, "method": "ping"},
                )
                assert r_bad_secret.status_code == 401

                # DELETE /mcp removes session
                r_del = await client.delete("/mcp", headers={**headers, "Mcp-Session-Id": sid})
                assert r_del.status_code == 200
                assert sid not in app.state.engine.sessions

        asyncio.run(_run())


def test_sse_streaming_heartbeats() -> None:
    secret_token = "test_sse_heartbeat_token"
    base_url = "https://localhost:8765"

    with tempfile.TemporaryDirectory(prefix="colab_mcp_sse_") as tmpdir:
        rt = ColabRuntime(workdir=tmpdir)
        app = create_app(token=secret_token, runtime=rt, public_url=base_url)
        app.state.keepalive_interval = 0.05
        transport = httpx.ASGITransport(app=app)

        async def _run() -> None:
            headers = {
                "Authorization": f"Bearer {secret_token}",
                "Accept": "application/json, text/event-stream",
            }
            async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
                r = await client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 101,
                        "method": "tools/call",
                        "params": {
                            "name": "execute_python",
                            "arguments": {"code": "import time\ntime.sleep(0.16)\n'heartbeat_ok'"},
                            "_meta": {"progressToken": "prog-42"},
                        },
                    },
                )
                assert r.status_code == 200, r.text
                assert r.headers.get("content-type", "").startswith("text/event-stream")
                assert r.headers.get("x-accel-buffering") == "no"

                comments, events = _parse_sse_events(r.text)
                assert ": connected" in comments, comments
                keepalives = [c for c in comments if c == ": keepalive"]
                assert len(keepalives) >= 2, f"Expected >=2 keepalives, got {comments!r}"

                parsed_events = [json.loads(data) for evt, data in events if evt == "message"]
                progress_msgs = [m for m in parsed_events if m.get("method") == "notifications/progress"]
                assert len(progress_msgs) >= 2, parsed_events
                assert progress_msgs[0]["params"]["progressToken"] == "prog-42"

                rpc_results = [m for m in parsed_events if m.get("id") == 101]
                assert len(rpc_results) == 1, parsed_events
                assert rpc_results[0]["result"]["isError"] is False
                payload = json.loads(rpc_results[0]["result"]["content"][0]["text"])
                assert payload["status"] == "ok"
                assert payload["result"] == "'heartbeat_ok'"

                # Plain Accept: application/json stays JSON (not SSE)
                r_json = await client.post(
                    "/mcp",
                    headers={
                        "Authorization": f"Bearer {secret_token}",
                        "Accept": "application/json",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": 102,
                        "method": "tools/call",
                        "params": {
                            "name": "execute_python",
                            "arguments": {"code": "2 + 2"},
                        },
                    },
                )
                assert r_json.status_code == 200
                assert "application/json" in r_json.headers.get("content-type", "")
                assert json.loads(r_json.json()["result"]["content"][0]["text"])["result"] == "4"

        asyncio.run(_run())


def test_oauth_2_1_security_and_spec() -> None:
    secret_token = "test_oauth_master_secret_2026"
    base_url = "https://colab-demo.example.com"
    callback_url = "https://claude.ai/api/mcp/auth_callback"

    with tempfile.TemporaryDirectory(prefix="colab_mcp_oauth_") as tmpdir:
        rt = ColabRuntime(workdir=tmpdir)
        app = create_app(token=secret_token, runtime=rt, public_url=base_url)
        transport = httpx.ASGITransport(app=app)

        async def _run() -> None:
            async with httpx.AsyncClient(
                transport=transport, base_url=base_url, follow_redirects=False
            ) as client:
                # 1. Discovery metadata endpoints
                r_pr = await client.get("/.well-known/oauth-protected-resource")
                assert r_pr.status_code == 200
                assert r_pr.json() == {
                    "resource": f"{base_url}/mcp",
                    "authorization_servers": [base_url],
                    "bearer_methods_supported": ["header"],
                }

                r_as = await client.get("/.well-known/oauth-authorization-server")
                assert r_as.status_code == 200
                as_meta = r_as.json()
                assert as_meta["issuer"] == base_url
                assert as_meta["authorization_endpoint"] == f"{base_url}/oauth/authorize"
                assert as_meta["token_endpoint"] == f"{base_url}/oauth/token"
                assert as_meta["registration_endpoint"] == f"{base_url}/oauth/register"
                assert as_meta["code_challenge_methods_supported"] == ["S256"]
                assert as_meta["client_id_metadata_document_supported"] is False
                assert "client_secret_basic" in as_meta["token_endpoint_auth_methods_supported"]

                r_oidc = await client.get("/.well-known/openid-configuration")
                assert r_oidc.status_code == 200
                assert r_oidc.json() == as_meta

                # 2. DCR negative security tests
                for bad_uri in (
                    "http://evil.example.com/cb",
                    "javascript:alert(1)",
                    "https://claude.ai/api/mcp/auth_callback#frag",
                    "ftp://example.com/cb",
                    "https://example.com/cb with space",
                ):
                    r_bad = await client.post(
                        "/oauth/register",
                        json={"client_name": "Bad Client", "redirect_uris": [bad_uri]},
                    )
                    assert r_bad.status_code == 400, f"Expected 400 for {bad_uri!r}, got {r_bad.status_code}"
                    assert r_bad.json()["error"] == "invalid_redirect_uri"

                # 3. Valid DCR (HTTPS + loopback HTTP)
                r_reg = await client.post(
                    "/oauth/register",
                    json={
                        "client_name": "MCP Client",
                        "redirect_uris": [callback_url, "http://127.0.0.1:43210/cb"],
                    },
                )
                assert r_reg.status_code == 201, r_reg.text
                client_id = r_reg.json()["client_id"]

                r_reg2 = await client.post(
                    "/oauth/register",
                    json={"client_name": "Other Client", "redirect_uris": ["http://localhost:8080/cb"]},
                )
                assert r_reg2.status_code == 201
                other_client_id = r_reg2.json()["client_id"]

                verifier = "test_pkce_code_verifier_random_string_1234567890"
                challenge = _pkce_challenge(verifier)

                # 4. /oauth/authorize negative security tests (GET & POST must return 400, NEVER 302)
                invalid_auth_cases = [
                    {
                        "client_id": "client_unregistered",
                        "redirect_uri": callback_url,
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                    },
                    {
                        "client_id": client_id,
                        "redirect_uri": "https://evil.example.com/steal",
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                    },
                    {
                        "client_id": client_id,
                        "redirect_uri": callback_url,
                        "code_challenge": "",
                        "code_challenge_method": "S256",
                    },
                    {
                        "client_id": client_id,
                        "redirect_uri": callback_url,
                        "code_challenge": challenge,
                        "code_challenge_method": "plain",
                    },
                ]
                for params in invalid_auth_cases:
                    r_g = await client.get("/oauth/authorize", params=params)
                    assert r_g.status_code == 400, (params, r_g.status_code)
                    assert "location" not in r_g.headers

                    r_p = await client.post(
                        "/oauth/authorize",
                        data={**params, "mcp_token": secret_token},
                    )
                    assert r_p.status_code == 400, (params, r_p.status_code)
                    assert "location" not in r_p.headers

                # 5. GET /oauth/authorize valid renders form
                r_form = await client.get(
                    "/oauth/authorize",
                    params={
                        "response_type": "code",
                        "client_id": client_id,
                        "redirect_uri": callback_url,
                        "state": "state_abc",
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                    },
                )
                assert r_form.status_code == 200
                assert client_id in r_form.text and 'name="mcp_token"' in r_form.text

                # 6. POST /oauth/authorize with wrong mcp_token -> 403 retryable HTML form
                r_wrong_tok = await client.post(
                    "/oauth/authorize",
                    data={
                        "client_id": client_id,
                        "redirect_uri": callback_url,
                        "state": "state_retry",
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                        "mcp_token": "wrong_secret",
                    },
                )
                assert r_wrong_tok.status_code == 403
                assert "Invalid MCP_TOKEN" in r_wrong_tok.text
                assert f'value="{client_id}"' in r_wrong_tok.text
                assert 'value="state_retry"' in r_wrong_tok.text

                async def _issue_auth_code(state_val: str = "xyz123") -> str:
                    r_ok = await client.post(
                        "/oauth/authorize",
                        data={
                            "client_id": client_id,
                            "redirect_uri": callback_url,
                            "state": state_val,
                            "code_challenge": challenge,
                            "code_challenge_method": "S256",
                            "mcp_token": secret_token,
                        },
                    )
                    assert r_ok.status_code == 302, r_ok.text
                    loc = urllib.parse.urlparse(r_ok.headers["location"])
                    qs = urllib.parse.parse_qs(loc.query)
                    assert qs["state"][0] == state_val
                    return qs["code"][0]

                # 7. /oauth/token negative tests (wrong verifier, non-ASCII verifier, mismatched client_id)
                code_bad_ver = await _issue_auth_code("s1")
                r_bad_ver = await client.post(
                    "/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": code_bad_ver,
                        "code_verifier": "completely_wrong_verifier",
                        "client_id": client_id,
                    },
                )
                assert r_bad_ver.status_code == 400
                assert r_bad_ver.json()["error"] == "invalid_grant"

                code_non_ascii = await _issue_auth_code("s2")
                r_non_ascii = await client.post(
                    "/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": code_non_ascii,
                        "code_verifier": "非ASCII验证码_verifier",
                        "client_id": client_id,
                    },
                )
                assert r_non_ascii.status_code == 400
                assert r_non_ascii.json()["error"] == "invalid_grant"

                code_bad_cid = await _issue_auth_code("s3")
                r_bad_cid = await client.post(
                    "/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": code_bad_cid,
                        "code_verifier": verifier,
                        "client_id": other_client_id,
                    },
                )
                assert r_bad_cid.status_code == 400
                assert r_bad_cid.json()["error"] == "invalid_grant"

                # 8. Valid token exchange (urlencoded form — zero python-multipart dependency)
                auth_code = await _issue_auth_code("valid_state")
                r_tok = await client.post(
                    "/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": auth_code,
                        "code_verifier": verifier,
                        "client_id": client_id,
                        "redirect_uri": callback_url,
                    },
                )
                assert r_tok.status_code == 200, r_tok.text
                assert r_tok.headers.get("cache-control") == "no-store"
                tok_data = r_tok.json()
                access_token = tok_data["access_token"]
                refresh_token = tok_data["refresh_token"]
                assert access_token.startswith("cmcp_access.")
                assert refresh_token.startswith("cmcp_refresh.")

                # One-time auth code replay rejected
                r_replay = await client.post(
                    "/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": auth_code,
                        "code_verifier": verifier,
                        "client_id": client_id,
                    },
                )
                assert r_replay.status_code == 400

                # Valid access_token works on /mcp
                r_ok_mcp = await client.post(
                    "/mcp",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={"jsonrpc": "2.0", "id": 99, "method": "ping"},
                )
                assert r_ok_mcp.status_code == 200

                # 9. Token kind separation security tests
                # Using refresh_token as Bearer on /mcp MUST be rejected (401)
                r_rt_as_bearer = await client.post(
                    "/mcp",
                    headers={"Authorization": f"Bearer {refresh_token}"},
                    json={"jsonrpc": "2.0", "id": 100, "method": "ping"},
                )
                assert r_rt_as_bearer.status_code == 401

                # Using refresh_token as X-Api-Key on /mcp MUST also be rejected (401)
                r_rt_as_apikey = await client.post(
                    "/mcp",
                    headers={"X-Api-Key": refresh_token},
                    json={"jsonrpc": "2.0", "id": 101, "method": "ping"},
                )
                assert r_rt_as_apikey.status_code == 401

                # Using access_token as refresh_token MUST be rejected (400)
                r_at_as_rt = await client.post(
                    "/oauth/token",
                    data={"grant_type": "refresh_token", "refresh_token": access_token},
                )
                assert r_at_as_rt.status_code == 400
                assert r_at_as_rt.json()["error"] == "invalid_grant"

                # Using master_token as refresh_token MUST be rejected (400)
                r_mt_as_rt = await client.post(
                    "/oauth/token",
                    data={"grant_type": "refresh_token", "refresh_token": secret_token},
                )
                assert r_mt_as_rt.status_code == 400
                assert r_mt_as_rt.json()["error"] == "invalid_grant"

                # 10. Valid refresh_token exchange yields a fresh access_token that works on /mcp
                r_ref = await client.post(
                    "/oauth/token",
                    data={"grant_type": "refresh_token", "refresh_token": refresh_token},
                )
                assert r_ref.status_code == 200, r_ref.text
                new_access_token = r_ref.json()["access_token"]
                assert new_access_token != access_token
                assert new_access_token.startswith("cmcp_access.")

                r_new_mcp = await client.post(
                    "/mcp",
                    headers={"Authorization": f"Bearer {new_access_token}"},
                    json={"jsonrpc": "2.0", "id": 102, "method": "ping"},
                )
                assert r_new_mcp.status_code == 200

                # 11. HTTP Basic Auth (client_secret_basic) supported on /oauth/token
                code_basic = await _issue_auth_code("basic_state")
                basic_creds = base64.b64encode(f"{client_id}:unused_secret".encode("utf-8")).decode("ascii")
                r_basic = await client.post(
                    "/oauth/token",
                    headers={"Authorization": f"Basic {basic_creds}"},
                    data={
                        "grant_type": "authorization_code",
                        "code": code_basic,
                        "code_verifier": verifier,
                    },
                )
                assert r_basic.status_code == 200, r_basic.text

        asyncio.run(_run())


def test_runtime_edge_cases_matplotlib_files_shell_and_jobs() -> None:
    secret_token = "test_runtime_edge_token"
    base_url = "https://localhost:8765"

    with tempfile.TemporaryDirectory(prefix="colab_mcp_rt_") as tmpdir:
        rt = ColabRuntime(workdir=tmpdir)
        app = create_app(token=secret_token, runtime=rt, public_url=base_url)
        transport = httpx.ASGITransport(app=app)

        # 1. Concurrent main-thread print(...) must NOT leak into execute_python stdout
        exec_result: Dict[str, Any] = {}
        started_evt = threading.Event()

        def _bg_cell() -> None:
            started_evt.set()
            payload, _ = rt.execute_python(
                "import time\nprint('CELL_START')\ntime.sleep(0.15)\nprint('CELL_END')\n'isolated_ok'",
                timeout_seconds=5,
            )
            exec_result["payload"] = payload

        t = threading.Thread(target=_bg_cell)
        t.start()
        started_evt.wait(timeout=2.0)
        for i in range(6):
            print(f"MAIN_THREAD_NOISE_{i}", flush=True)
            time.sleep(0.02)
        t.join(timeout=5.0)
        assert not t.is_alive()
        p_iso = exec_result["payload"]
        assert p_iso["status"] == "ok", p_iso
        assert "CELL_START" in p_iso["stdout"] and "CELL_END" in p_iso["stdout"]
        assert "MAIN_THREAD_NOISE_" not in p_iso["stdout"], p_iso["stdout"]

        # 2. Pure-Python infinite loop interrupted cleanly, and very next cell succeeds
        p_to, imgs_to = rt.execute_python("while True:\n    pass", timeout_seconds=1)
        assert p_to["status"] == "timeout", p_to
        assert p_to["interrupted"] is True, p_to
        assert imgs_to == []

        p_next, _ = rt.execute_python("100 + 23", timeout_seconds=5)
        assert p_next["status"] == "ok", p_next
        assert p_next["result"] == "123"

        # 2b. Uninterruptible native sleep triggers stuck-worker busy guard, then auto-recovers
        p_stuck, _ = rt.execute_python("import time; time.sleep(2.4)", timeout_seconds=1)
        assert p_stuck["status"] == "timeout", p_stuck
        assert p_stuck["interrupted"] is False, p_stuck
        p_busy, _ = rt.execute_python("1 + 1")
        assert p_busy["status"] == "busy", p_busy
        r_busy = rt.reset_python_namespace()
        assert r_busy["status"] == "busy", r_busy
        time.sleep(0.7)
        p_recovered, _ = rt.execute_python("1 + 1")
        assert p_recovered["status"] == "ok" and p_recovered["result"] == "2", p_recovered

        # 3. MCP tool tests: real Matplotlib multi-plot capture, notebook export, reset,
        #    edit_file, read_image, run_shell timeout, install_packages flag guard
        async def _mcp_tool_checks() -> None:
            headers = {"Authorization": f"Bearer {secret_token}"}

            async def _call_tool(
                client: httpx.AsyncClient, call_id: int, name: str, arguments: Dict[str, Any]
            ) -> Dict[str, Any]:
                r = await client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": call_id,
                        "method": "tools/call",
                        "params": {"name": name, "arguments": arguments},
                    },
                )
                assert r.status_code == 200, r.text
                return r.json()["result"]

            async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
                # Real Matplotlib multi-plot capture (fresh import + pre-loaded non-Agg backend)
                mpl_code = (
                    "import matplotlib.pyplot as plt\n"
                    "plt.figure(figsize=(3, 2))\n"
                    "plt.plot([1, 2, 3], [1, 4, 9])\n"
                    "plt.title('First Plot')\n"
                    "plt.show()\n"
                    "plt.figure(figsize=(3, 2))\n"
                    "plt.plot([1, 2, 3], [9, 4, 1])\n"
                    "plt.title('Second Plot')\n"
                    "plt.show()\n"
                    "'two_plots_done'\n"
                )
                res_mpl = await _call_tool(client, 1, "execute_python", {"code": mpl_code})
                assert res_mpl["isError"] is False, res_mpl
                blocks = res_mpl["content"]
                assert len(blocks) == 3, f"Expected 1 text + 2 image blocks, got {len(blocks)}"
                mpl_meta = json.loads(blocks[0]["text"])
                assert mpl_meta["status"] == "ok"
                assert mpl_meta["figures_captured"] == 2
                assert mpl_meta["result"] == "'two_plots_done'"

                # Simulate pre-loaded non-Agg backend (like Colab inline) + 1 shown + 1 un-shown figure
                import matplotlib

                matplotlib.use("svg", force=True)
                res_mpl2 = await _call_tool(
                    client,
                    19,
                    "execute_python",
                    {
                        "code": (
                            "import matplotlib.pyplot as plt\n"
                            "plt.figure(figsize=(2, 2)); plt.plot([0, 1], [1, 0]); plt.show()\n"
                            "plt.figure(figsize=(2, 2)); plt.plot([0, 1], [0, 1])\n"
                        )
                    },
                )
                assert res_mpl2["isError"] is False, res_mpl2
                assert len(res_mpl2["content"]) == 3
                assert json.loads(res_mpl2["content"][0]["text"])["figures_captured"] == 2
                assert str(matplotlib.get_backend()).lower() == "agg"

                png_bytes_list: List[bytes] = []
                for idx in (1, 2):
                    assert blocks[idx]["type"] == "image"
                    assert blocks[idx]["mimeType"] == "image/png"
                    raw_png = base64.b64decode(blocks[idx]["data"])
                    assert raw_png.startswith(b"\x89PNG\r\n\x1a\n")
                    png_bytes_list.append(raw_png)

                # export_session_notebook includes valid .ipynb with both embedded PNGs
                res_nb = await _call_tool(
                    client, 2, "export_session_notebook", {"output_path": "plots_session.ipynb"}
                )
                assert res_nb["isError"] is False
                nb_meta = json.loads(res_nb["content"][0]["text"])
                assert nb_meta["status"] == "ok"
                assert nb_meta["cells_exported"] >= 3
                with open(nb_meta["notebook_path"], "r", encoding="utf-8") as f:
                    nb_doc = json.load(f)
                assert nb_doc["nbformat"] == 4
                mpl_cells = [c for c in nb_doc["cells"] if "First Plot" in "".join(c["source"])]
                assert len(mpl_cells) == 1
                img_outputs = [
                    o for o in mpl_cells[0]["outputs"] if o.get("output_type") == "display_data"
                ]
                assert len(img_outputs) == 2
                assert img_outputs[0]["data"]["image/png"] == blocks[1]["data"]
                assert img_outputs[1]["data"]["image/png"] == blocks[2]["data"]

                # reset_python_namespace(clear_history=False) vs clear_history=True
                res_reset1 = await _call_tool(
                    client, 3, "reset_python_namespace", {"clear_history": False}
                )
                assert res_reset1["isError"] is False
                r1_meta = json.loads(res_reset1["content"][0]["text"])
                assert r1_meta["history_cleared"] is False
                assert len(rt.cell_history) >= 3

                res_reset2 = await _call_tool(
                    client, 4, "reset_python_namespace", {"clear_history": True}
                )
                assert res_reset2["isError"] is False
                r2_meta = json.loads(res_reset2["content"][0]["text"])
                assert r2_meta["history_cleared"] is True
                assert len(rt.cell_history) == 0

                # write_file + edit_file (unique, ambiguous error, replace_all) + read_file + list_files
                res_wf = await _call_tool(
                    client,
                    5,
                    "write_file",
                    {"path": "notes.txt", "content": "alpha\nbeta\nalpha\n"},
                )
                assert res_wf["isError"] is False

                res_e1 = await _call_tool(
                    client,
                    6,
                    "edit_file",
                    {"path": "notes.txt", "old_text": "beta", "new_text": "BETA"},
                )
                assert res_e1["isError"] is False
                assert json.loads(res_e1["content"][0]["text"])["replacements"] == 1

                res_e_amb = await _call_tool(
                    client,
                    7,
                    "edit_file",
                    {
                        "path": "notes.txt",
                        "old_text": "alpha",
                        "new_text": "ALPHA",
                        "replace_all": False,
                    },
                )
                assert res_e_amb["isError"] is True
                assert "matched 2 locations" in json.loads(res_e_amb["content"][0]["text"])["error"]

                res_e_all = await _call_tool(
                    client,
                    8,
                    "edit_file",
                    {
                        "path": "notes.txt",
                        "old_text": "alpha",
                        "new_text": "ALPHA",
                        "replace_all": True,
                    },
                )
                assert res_e_all["isError"] is False
                assert json.loads(res_e_all["content"][0]["text"])["replacements"] == 2

                res_rf = await _call_tool(
                    client,
                    9,
                    "read_file",
                    {"path": "notes.txt", "start_line": 2, "max_lines": 1},
                )
                rf_meta = json.loads(res_rf["content"][0]["text"])
                assert rf_meta["content"] == "BETA\n"
                assert rf_meta["total_lines"] == 3

                res_lf = await _call_tool(client, 10, "list_files", {"path": ".", "max_entries": 1})
                lf_meta = json.loads(res_lf["content"][0]["text"])
                assert lf_meta["status"] == "ok"
                assert lf_meta["total_entries"] >= 2
                assert lf_meta["truncated"] is True
                assert "mtime_epoch" in lf_meta["entries"][0]

                # read_image (valid PNG -> [text_without_b64, image_block], missing & non-image -> isError)
                img_path = os.path.join(tmpdir, "saved_plot.png")
                with open(img_path, "wb") as f:
                    f.write(png_bytes_list[0])

                res_img = await _call_tool(client, 11, "read_image", {"path": "saved_plot.png"})
                assert res_img["isError"] is False, res_img
                assert len(res_img["content"]) == 2
                img_meta = json.loads(res_img["content"][0]["text"])
                assert img_meta["status"] == "ok"
                assert img_meta["mime_type"] == "image/png"
                assert "data_base64" not in img_meta
                assert res_img["content"][1]["type"] == "image"
                assert res_img["content"][1]["mimeType"] == "image/png"
                assert base64.b64decode(res_img["content"][1]["data"]) == png_bytes_list[0]

                res_img_missing = await _call_tool(client, 12, "read_image", {"path": "nope.png"})
                assert res_img_missing["isError"] is True

                res_img_txt = await _call_tool(client, 13, "read_image", {"path": "notes.txt"})
                assert res_img_txt["isError"] is True

                # run_shell timeout kills child process group and preserves partial stdout
                child_pid_file = os.path.join(tmpdir, "child.pid")
                shell_cmd = (
                    f"echo PARTIAL_STDOUT_LINE; "
                    f"python3 -c 'import time; time.sleep(30)' & "
                    f"echo $! > '{child_pid_file}'; "
                    f"wait"
                )
                res_sh = await _call_tool(
                    client, 14, "run_shell", {"command": shell_cmd, "timeout_seconds": 1}
                )
                assert res_sh["isError"] is True
                sh_meta = json.loads(res_sh["content"][0]["text"])
                assert sh_meta["status"] == "timeout", sh_meta
                assert "PARTIAL_STDOUT_LINE" in sh_meta["stdout"], sh_meta
                assert os.path.isfile(child_pid_file)
                with open(child_pid_file, "r", encoding="utf-8") as f:
                    child_pid = int(f.read().strip())
                # Wait briefly for OS cleanup and verify child process group was terminated
                alive = True
                for _ in range(20):
                    try:
                        os.kill(child_pid, 0)
                        time.sleep(0.05)
                    except ProcessLookupError:
                        alive = False
                        break
                assert not alive, f"Child process {child_pid} survived run_shell timeout!"

                # install_packages rejects leading '-' flag injection
                res_pip = await _call_tool(
                    client,
                    15,
                    "install_packages",
                    {
                        "packages": ["--index-url=https://evil.example/simple", "requests"],
                        "timeout_seconds": 30,
                    },
                )
                assert res_pip["isError"] is True
                pip_meta = json.loads(res_pip["content"][0]["text"])
                assert pip_meta["status"] == "error"
                assert "must not start with '-'" in pip_meta["error"]

        asyncio.run(_mcp_tool_checks())

        # 4. Background job: large >150KB log with \r progress updates + frozen elapsed_seconds
        big_script = (
            "import sys\n"
            "payload = 'X' * 120\n"
            "for i in range(1500):\n"
            "    print(f'line_{i:04d} {payload}')\n"
            "sys.stdout.write('step 10%\\rstep 50%\\rstep 100% done\\n')\n"
            "print('FINAL_TAIL_MARKER')\n"
        )
        big_script_path = os.path.join(tmpdir, "gen_big_log.py")
        with open(big_script_path, "w", encoding="utf-8") as f:
            f.write(big_script)

        job_big = rt.start_background_job(
            command=f"'{sys.executable}' '{big_script_path}'",
            name="big_log_job",
        )
        job_big_id = job_big["job_id"]
        st_big: Dict[str, Any] = {}
        for _ in range(100):
            st_big = rt.get_job_status(job_big_id, tail_lines=5)
            if st_big["status"] != "running":
                break
            time.sleep(0.05)
        assert st_big["status"] == "succeeded", st_big
        assert st_big["log_size_bytes"] > 150_000, st_big["log_size_bytes"]
        tail_lines = st_big["log_tail"].splitlines()
        assert len(tail_lines) == 5, tail_lines
        assert tail_lines[-1] == "FINAL_TAIL_MARKER", tail_lines
        assert tail_lines[-2] == "step 100% done", tail_lines
        assert "step 10%" not in st_big["log_tail"]

        frozen_elapsed = st_big["elapsed_seconds"]
        time.sleep(0.25)
        st_big_later = rt.get_job_status(job_big_id, tail_lines=5)
        assert st_big_later["elapsed_seconds"] == frozen_elapsed, (
            frozen_elapsed,
            st_big_later["elapsed_seconds"],
        )

        # 5. Background job persistence across ColabRuntime restart
        persist_job = rt.start_background_job(
            command=f"'{sys.executable}' -u -c 'import time; print(\"PERSIST_READY\"); time.sleep(30)'",
            name="persist_sleeper",
        )
        persist_id = persist_job["job_id"]
        try:
            for _ in range(50):
                st = rt.get_job_status(persist_id, tail_lines=10)
                if "PERSIST_READY" in st["log_tail"]:
                    break
                time.sleep(0.05)

            # Instantiate a brand-new ColabRuntime pointing to the same workdir
            rt_reloaded = ColabRuntime(workdir=tmpdir)
            reloaded_jobs = {j["job_id"]: j for j in rt_reloaded.list_jobs()}
            assert persist_id in reloaded_jobs, reloaded_jobs
            assert reloaded_jobs[persist_id]["status"] == "running"

            st_reloaded = rt_reloaded.get_job_status(persist_id, tail_lines=10)
            assert st_reloaded["status"] == "running", st_reloaded
            assert "PERSIST_READY" in st_reloaded["log_tail"], st_reloaded

            cancel_res = rt_reloaded.cancel_job(persist_id)
            assert cancel_res["status"] == "cancelled", cancel_res
            assert cancel_res["cancelled"] is True, cancel_res
        finally:
            # Ensure original process is reaped cleanly even if an assertion failed
            rt.cancel_job(persist_id)
            rt.list_jobs()


def test_local_start_colab_mcp_lifecycle_and_legacy_sse() -> None:
    port = _find_free_port()
    busy_port = _find_free_port()
    while busy_port == port:
        busy_port = _find_free_port()

    active1: ActiveColabMCP | None = None
    active2: ActiveColabMCP | None = None

    with tempfile.TemporaryDirectory(prefix="colab_mcp_live_") as tmpdir:
        try:
            # 1. Start local server with tok_v1 (even if HTTP_PROXY is set without NO_PROXY)
            prev_http_proxy = os.environ.get("HTTP_PROXY")
            prev_no_proxy = os.environ.get("NO_PROXY")
            os.environ["HTTP_PROXY"] = "http://127.0.0.1:1"
            os.environ["NO_PROXY"] = ""
            try:
                active1 = start_colab_mcp(
                    token="tok_v1",
                    port=port,
                    provider="none",
                    workdir=tmpdir,
                    block=False,
                )
            finally:
                if prev_http_proxy is None:
                    os.environ.pop("HTTP_PROXY", None)
                else:
                    os.environ["HTTP_PROXY"] = prev_http_proxy
                if prev_no_proxy is None:
                    os.environ.pop("NO_PROXY", None)
                else:
                    os.environ["NO_PROXY"] = prev_no_proxy
            assert isinstance(active1, ActiveColabMCP)
            assert active1.public_url == f"http://127.0.0.1:{port}"

            with httpx.Client(base_url=active1.public_url, trust_env=False, timeout=5.0) as client:
                r_health = client.get("/health")
                assert r_health.status_code == 200
                inst1 = r_health.json()["instance_id"]
                assert inst1

                # app.state.public_url dynamically synced to active1.public_url
                r_pr = client.get("/.well-known/oauth-protected-resource")
                assert r_pr.json()["resource"] == f"http://127.0.0.1:{port}/mcp"

                r_v1 = client.post(
                    "/mcp",
                    headers={"Authorization": "Bearer tok_v1"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                )
                assert r_v1.status_code == 200, r_v1.text
                assert r_v1.json()["result"] == {}

                r_v2_fail = client.post(
                    "/mcp",
                    headers={"Authorization": "Bearer tok_v2"},
                    json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
                )
                assert r_v2_fail.status_code == 401

            # 2. Live Legacy /sse + /messages (including malformed JSON -> 400)
            async def _legacy_sse_check() -> None:
                auth_h = {"Authorization": "Bearer tok_v1"}
                async with httpx.AsyncClient(
                    base_url=active1.public_url, trust_env=False, timeout=5.0
                ) as aclient:
                    async with aclient.stream("GET", "/sse", headers=auth_h) as sse_resp:
                        assert sse_resp.status_code == 200
                        lines_iter = sse_resp.aiter_lines()
                        evt_line = await asyncio.wait_for(lines_iter.__anext__(), timeout=3.0)
                        data_line = await asyncio.wait_for(lines_iter.__anext__(), timeout=3.0)
                        assert evt_line == "event: endpoint"
                        assert data_line.startswith("data: /messages?session_id=")
                        msg_path = data_line[len("data: ") :].strip()

                        # Malformed JSON on /messages -> 400 -32700
                        r_bad_msg = await aclient.post(
                            msg_path,
                            headers={**auth_h, "Content-Type": "application/json"},
                            content=b"{malformed",
                        )
                        assert r_bad_msg.status_code == 400
                        assert r_bad_msg.json()["error"]["code"] == -32700

                        # Valid ping on /messages -> 202 and response arrives over SSE stream
                        r_msg = await aclient.post(
                            msg_path,
                            headers=auth_h,
                            json={"jsonrpc": "2.0", "id": 77, "method": "ping"},
                        )
                        assert r_msg.status_code == 202

                        received_rpc = None
                        for _ in range(10):
                            line = await asyncio.wait_for(lines_iter.__anext__(), timeout=3.0)
                            if line.startswith("data:"):
                                received_rpc = json.loads(line[len("data:") :].strip())
                                break
                        assert received_rpc == {"jsonrpc": "2.0", "id": 77, "result": {}}

            asyncio.run(_legacy_sse_check())

            # 3. Re-run start_colab_mcp on SAME port with tok_v2 (simulates Colab Step 3 re-run)
            old_thread = active1.server_thread
            active2 = start_colab_mcp(
                token="tok_v2",
                port=port,
                provider="none",
                workdir=tmpdir,
                block=False,
            )
            assert old_thread is not None and not old_thread.is_alive()

            with httpx.Client(base_url=active2.public_url, trust_env=False, timeout=5.0) as client:
                r_health2 = client.get("/health")
                assert r_health2.status_code == 200
                assert r_health2.json()["instance_id"] != inst1

                r_v2_ok = client.post(
                    "/mcp",
                    headers={"Authorization": "Bearer tok_v2"},
                    json={"jsonrpc": "2.0", "id": 3, "method": "ping"},
                )
                assert r_v2_ok.status_code == 200, r_v2_ok.text

                r_v1_old = client.post(
                    "/mcp",
                    headers={"Authorization": "Bearer tok_v1"},
                    json={"jsonrpc": "2.0", "id": 4, "method": "ping"},
                )
                assert r_v1_old.status_code == 401

            # 4. Occupied port raises RuntimeError cleanly
            blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            blocker.bind(("127.0.0.1", busy_port))
            blocker.listen(5)
            try:
                raised = False
                try:
                    start_colab_mcp(
                        token="tok_busy",
                        port=busy_port,
                        provider="none",
                        workdir=tmpdir,
                        block=False,
                    )
                except RuntimeError:
                    raised = True
                assert raised, "Expected RuntimeError when port is occupied by another socket!"
            finally:
                blocker.close()

            # 5. active2.stop() shuts down cleanly and is idempotent
            active2.stop()
            assert active2.server_thread is not None and not active2.server_thread.is_alive()
            active2.stop()  # idempotent second call

            # Confirm local port is closed after stop()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.5)
                assert probe.connect_ex(("127.0.0.1", port)) != 0
        finally:
            if active2 is not None:
                active2.stop()
            if active1 is not None:
                active1.stop()


def run_tests() -> None:
    test_core_streamable_http_cors_and_all_tools()
    test_sse_streaming_heartbeats()
    test_oauth_2_1_security_and_spec()
    test_runtime_edge_cases_matplotlib_files_shell_and_jobs()
    test_local_start_colab_mcp_lifecycle_and_legacy_sse()
    print("ALL E2E TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
