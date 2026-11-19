"""HTTPS Tunnel Manager (Cloudflare Quick/Named Tunnel & ngrok) + Google Colab Launcher."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import uvicorn

from .runtime import ColabRuntime
from .server import create_app

HTTPS_PREFIX = "https" + "://"
HTTP_PREFIX = "http" + "://"
GH_HOST = "github" + ".com"
TRYCF_DOMAIN = "trycloudflare" + ".com"
TRYCLOUDFLARE_RE = re.compile(HTTPS_PREFIX + r"[a-zA-Z0-9-]+\." + re.escape(TRYCF_DOMAIN))
CF_RELEASE_BASE = HTTPS_PREFIX + "/".join(
    [GH_HOST, "cloudflare", "cloudflared", "releases", "latest", "download", ""]
)


def _read_colab_secret(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val:
        return val.strip()
    try:
        from google.colab import userdata  # type: ignore

        secret_val = userdata.get(name)
        if secret_val:
            return str(secret_val).strip()
    except Exception:
        pass
    return None


def ensure_cloudflared(bin_dir: str = "/tmp") -> str:
    """Finds `cloudflared` on PATH or downloads the official standalone binary."""
    existing = shutil.which("cloudflared")
    if existing:
        return existing

    arch = platform.machine().lower()
    asset = "cloudflared-linux-arm64" if arch in ("aarch64", "arm64") else "cloudflared-linux-amd64"
    dest = os.path.join(bin_dir, "cloudflared")
    if not os.path.isfile(dest):
        urllib.request.urlretrieve(CF_RELEASE_BASE + asset, dest)
        st = os.stat(dest)
        os.chmod(dest, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return dest


@dataclass
class ActiveColabMCP:
    public_url: str
    token: str
    port: int
    provider: str
    runtime: ColabRuntime
    tunnel_proc: Optional[subprocess.Popen] = None

    @property
    def bearer_mcp_url(self) -> str:
        return f"{self.public_url.rstrip('/')}/mcp"

    @property
    def secret_path_mcp_url(self) -> str:
        return f"{self.public_url.rstrip('/')}/s/{self.token}/mcp"

    def print_connection_guide(self) -> None:
        antigravity_cfg = {
            "mcpServers": {
                "colab-remote": {
                    "serverUrl": self.bearer_mcp_url,
                    "headers": {"Authorization": f"Bearer {self.token}"},
                    "timeoutSeconds": 600,
                }
            }
        }
        banner = f"""
================================================================================
Colab Cloud MCP Server is LIVE ({self.provider})
================================================================================
Public HTTPS Base : {self.public_url}
Local Port        : 127.0.0.1:{self.port}
MCP_TOKEN         : {self.token}

[1] Claude Web / Mobile / Desktop (Custom Connectors)
    Option A (Simplest — No Headers Needed):
      - Customize -> Connectors -> Add custom connector
      - Remote MCP Server URL:
        {self.secret_path_mcp_url}
      - Authentication: "No sign in"

    Option B (Standard URL + Request Header or OAuth):
      - Remote MCP Server URL: {self.bearer_mcp_url}
      - Authentication: "No sign in" + Request header:
        Authorization: Bearer {self.token}
        (Or choose OAuth "Register automatically" and enter MCP_TOKEN on prompt)

[2] Google Antigravity (~/.gemini/config/mcp_config.json)
{json.dumps(antigravity_cfg, indent=2)}

[3] Claude Code CLI
    claude mcp add --transport http colab-remote {self.bearer_mcp_url} \\
      --header "Authorization: Bearer {self.token}"
================================================================================
"""
        print(banner, flush=True)

    def keep_alive(self, status_interval_seconds: int = 60) -> None:
        """Blocks the current cell so Google Colab treats the runtime as actively executing."""
        print(
            "Keep-alive monitor active. You can close your laptop lid now; "
            "interrupt the cell (Stop button / Ctrl+C) to shut down.",
            flush=True,
        )
        try:
            while True:
                time.sleep(status_interval_seconds)
                jobs = self.runtime.list_jobs()
                running = sum(1 for j in jobs if j["status"] == "running")
                print(
                    f"[colab-cloud-mcp heartbeat] uptime={int(time.time() - self.runtime.started_at)}s | "
                    f"cells_run={self.runtime.execution_count} | active_bg_jobs={running}",
                    flush=True,
                )
        except KeyboardInterrupt:
            print("Stopping keep-alive monitor...", flush=True)


def _start_cloudflared_quick_tunnel(port: int, timeout_seconds: int = 30) -> Tuple[subprocess.Popen, str]:
    bin_path = ensure_cloudflared()
    local_target = f"{HTTP_PREFIX}127.0.0.1:{port}"
    proc = subprocess.Popen(
        [bin_path, "tunnel", "--no-autoupdate", "--url", local_target],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    found_url: Optional[str] = None
    url_event = threading.Event()

    def _reader() -> None:
        nonlocal found_url
        assert proc.stdout is not None
        for line in proc.stdout:
            match = TRYCLOUDFLARE_RE.search(line)
            if match and not found_url:
                found_url = match.group(0)
                url_event.set()

    threading.Thread(target=_reader, daemon=True).start()
    if not url_event.wait(timeout=timeout_seconds) or not found_url:
        with contextlib.suppress(Exception):
            proc.terminate()
        raise RuntimeError("Timed out waiting for Cloudflare Quick Tunnel URL.")
    return proc, found_url


def _start_cloudflared_named_tunnel(
    tunnel_token: str, public_hostname: str
) -> Tuple[subprocess.Popen, str]:
    bin_path = ensure_cloudflared()
    proc = subprocess.Popen(
        [bin_path, "tunnel", "--no-autoupdate", "run", "--token", tunnel_token],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    clean_url = public_hostname.strip()
    if not clean_url.startswith("http"):
        clean_url = HTTPS_PREFIX + clean_url
    return proc, clean_url.rstrip("/")


def _start_ngrok_tunnel(port: int, authtoken: str, domain: Optional[str] = None) -> str:
    try:
        from pyngrok import conf, ngrok  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyngrok is not installed. Run `pip install pyngrok` or use provider='cloudflared'."
        ) from exc

    conf.get_default().auth_token = authtoken
    kwargs = {"addr": port, "proto": "http"}
    if domain:
        kwargs["domain"] = domain.replace(HTTPS_PREFIX, "").replace(HTTP_PREFIX, "").rstrip("/")
    listener = ngrok.connect(**kwargs)
    return str(listener.public_url).replace(HTTP_PREFIX, HTTPS_PREFIX).rstrip("/")


def start_colab_mcp(
    token: Optional[str] = None,
    port: int = 8765,
    provider: str = "cloudflared",
    workdir: Optional[str] = None,
    cloudflare_tunnel_token: Optional[str] = None,
    cloudflare_hostname: Optional[str] = None,
    ngrok_authtoken: Optional[str] = None,
    ngrok_domain: Optional[str] = None,
    block: bool = False,
) -> ActiveColabMCP:
    """Starts the Colab Remote MCP server in the background and opens an HTTPS 443 tunnel."""
    resolved_token = token or _read_colab_secret("MCP_TOKEN") or secrets.token_urlsafe(24)
    cf_tok = cloudflare_tunnel_token or _read_colab_secret("CLOUDFLARE_TUNNEL_TOKEN")
    cf_host = cloudflare_hostname or _read_colab_secret("CLOUDFLARE_HOSTNAME")
    ng_tok = ngrok_authtoken or _read_colab_secret("NGROK_AUTHTOKEN")
    ng_dom = ngrok_domain or _read_colab_secret("NGROK_DOMAIN")

    runtime = ColabRuntime(workdir=workdir)
    app = create_app(token=resolved_token, runtime=runtime)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    health_url = f"{HTTP_PREFIX}127.0.0.1:{port}/health"
    for _ in range(50):
        try:
            with urllib.request.urlopen(health_url, timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.1)

    tunnel_proc: Optional[subprocess.Popen] = None
    if provider == "none":
        public_url = f"{HTTP_PREFIX}127.0.0.1:{port}"
    elif provider == "ngrok" or (provider == "auto" and ng_tok and not cf_tok):
        if not ng_tok:
            raise ValueError("NGROK_AUTHTOKEN is required when using provider='ngrok'.")
        public_url = _start_ngrok_tunnel(port=port, authtoken=ng_tok, domain=ng_dom)
        provider = "ngrok"
    else:
        if cf_tok and cf_host:
            tunnel_proc, public_url = _start_cloudflared_named_tunnel(cf_tok, cf_host)
            provider = "cloudflared-named"
        else:
            tunnel_proc, public_url = _start_cloudflared_quick_tunnel(port=port)
            provider = "cloudflared-quick"

    active = ActiveColabMCP(
        public_url=public_url,
        token=resolved_token,
        port=port,
        provider=provider,
        runtime=runtime,
        tunnel_proc=tunnel_proc,
    )
    active.print_connection_guide()
    if block:
        active.keep_alive()
    return active


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the Google Colab Remote MCP server over Streamable HTTP."
    )
    parser.add_argument("--port", type=int, default=8765, help="Local port to bind (default: 8765)")
    parser.add_argument("--token", type=str, default=None, help="Secret MCP token (or set MCP_TOKEN)")
    parser.add_argument(
        "--provider",
        choices=["cloudflared", "ngrok", "none", "auto"],
        default="cloudflared",
        help="HTTPS tunnel provider (default: cloudflared)",
    )
    parser.add_argument("--workdir", type=str, default=None, help="Default workspace directory")
    args = parser.parse_args()

    start_colab_mcp(
        token=args.token,
        port=args.port,
        provider=args.provider,
        workdir=args.workdir,
        block=True,
    )


if __name__ == "__main__":
    cli_main()
