"""End-to-end integration test suite for colab-cloud-mcp."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.parse

import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from colab_cloud_mcp.runtime import ColabRuntime
from colab_cloud_mcp.server import create_app


def run_tests() -> None:
    secret_token = "test_super_secret_colab_token_2026"
    base_url = "https" + "://localhost:8765"
    callback_url = "https" + "://localhost/api/mcp/auth_callback"

    with tempfile.TemporaryDirectory(prefix="colab_mcp_test_") as tmpdir:
        rt = ColabRuntime(workdir=tmpdir)
        app = create_app(token=secret_token, runtime=rt, public_url=base_url)
        transport = httpx.ASGITransport(app=app)

        async def _async_suite() -> None:
            async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
                r = await client.get("/health")
                assert r.status_code == 200, r.text
                assert r.json()["status"] == "ok"

                r = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
                assert r.status_code == 401
                assert "resource_metadata=" in r.headers.get("www-authenticate", "")

                headers = {"Authorization": f"Bearer {secret_token}"}
                r = await client.post(
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
                assert r.status_code == 200, r.text
                sid = r.headers.get("mcp-session-id")
                assert sid

                r_notif = await client.post(
                    "/mcp",
                    headers={**headers, "Mcp-Session-Id": sid},
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                )
                assert r_notif.status_code == 202

                r_tools = await client.post(
                    "/mcp",
                    headers={**headers, "Mcp-Session-Id": sid},
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                )
                assert r_tools.status_code == 200
                tool_names = {t["name"] for t in r_tools.json()["result"]["tools"]}
                assert {"execute_python", "start_background_job", "export_session_notebook"} <= tool_names

                r_exec1 = await client.post(
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
                assert r_exec1.status_code == 200
                meta1 = json.loads(r_exec1.json()["result"]["content"][0]["text"])
                assert meta1["status"] == "ok" and meta1["result"] == "42"

                r_job = await client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "tools/call",
                        "params": {
                            "name": "start_background_job",
                            "arguments": {
                                "name": "demo_train",
                                "command": "python3 -c 'import time; print(\"epoch 1\"); time.sleep(0.2); print(\"done\")'",
                            },
                        },
                    },
                )
                job_id = json.loads(r_job.json()["result"]["content"][0]["text"])["job_id"]
                time.sleep(0.4)
                r_status = await client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 6,
                        "method": "tools/call",
                        "params": {"name": "get_job_status", "arguments": {"job_id": job_id}},
                    },
                )
                st_info = json.loads(r_status.json()["result"]["content"][0]["text"])
                assert st_info["status"] == "succeeded", st_info

                r_nb = await client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 7,
                        "method": "tools/call",
                        "params": {
                            "name": "export_session_notebook",
                            "arguments": {"output_path": "exported_test.ipynb"},
                        },
                    },
                )
                assert json.loads(r_nb.json()["result"]["content"][0]["text"])["cells_exported"] == 1

                r_reg = await client.post(
                    "/oauth/register",
                    json={"client_name": "MCP Client", "redirect_uris": [callback_url]},
                )
                assert r_reg.status_code == 201
                client_id = r_reg.json()["client_id"]

                verifier = "test_pkce_code_verifier_random_string_1234567890"
                challenge = (
                    base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
                    .rstrip(b"=")
                    .decode("ascii")
                )
                r_auth = await client.post(
                    "/oauth/authorize",
                    data={
                        "client_id": client_id,
                        "redirect_uri": callback_url,
                        "state": "xyz123",
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                        "mcp_token": secret_token,
                    },
                )
                assert r_auth.status_code == 302
                auth_code = urllib.parse.parse_qs(urllib.parse.urlparse(r_auth.headers["location"]).query)["code"][0]

                r_tok = await client.post(
                    "/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": auth_code,
                        "code_verifier": verifier,
                        "client_id": client_id,
                    },
                )
                assert r_tok.status_code == 200
                oauth_access_token = r_tok.json()["access_token"]

                r_oauth_mcp = await client.post(
                    "/mcp",
                    headers={"Authorization": f"Bearer {oauth_access_token}"},
                    json={"jsonrpc": "2.0", "id": 99, "method": "ping"},
                )
                assert r_oauth_mcp.status_code == 200

        asyncio.run(_async_suite())
        print("ALL E2E TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
