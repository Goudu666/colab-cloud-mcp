"""HTTPS Tunnel Manager (Cloudflare Quick/Named Tunnel & ngrok) + Google Colab Launcher."""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import uvicorn

from .runtime import ColabRuntime
from .server import create_app

TRYCLOUDFLARE_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
CF_RELEASE_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"

_ACTIVE_SERVERS: Dict[int, ActiveColabMCP] = {}


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
    """Finds `cloudflared` on PATH or downloads the official standalone Linux binary."""
    existing = shutil.which("cloudflared")
    if existing:
        return existing

    if platform.system() != "Linux":
        raise RuntimeError(
            "Automatic cloudflared binary download is only supported on Linux/Colab. "
            "On macOS, run `brew install cloudflared` first."
        )

    arch = platform.machine().lower()
    asset = "cloudflared-linux-arm64" if arch in ("aarch64", "arm64") else "cloudflared-linux-amd64"
    dest = os.path.join(bin_dir, "cloudflared")
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000 and os.access(dest, os.X_OK):
        return dest

    os.makedirs(bin_dir, exist_ok=True)
    tmp_dest = f"{dest}.tmp.{os.getpid()}"
    try:
        urllib.request.urlretrieve(CF_RELEASE_BASE + asset, tmp_dest)
        os.chmod(tmp_dest, 0o755)
        os.replace(tmp_dest, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            if os.path.exists(tmp_dest):
                os.remove(tmp_dest)
        raise
    return dest


@dataclass
class ActiveColabMCP:
    public_url: str
    token: str
    port: int
    provider: str
    runtime: ColabRuntime
    tunnel_proc: Optional[subprocess.Popen] = None
    uvicorn_server: Optional[uvicorn.Server] = None
    server_thread: Optional[threading.Thread] = None
    ngrok_url: Optional[str] = None
    _named_tunnel_token: Optional[str] = None

    @property
    def bearer_mcp_url(self) -> str:
        return f"{self.public_url.rstrip('/')}/mcp"

    @property
    def secret_path_mcp_url(self) -> str:
        return f"{self.public_url.rstrip('/')}/s/{self.token}/mcp"

    def _sync_app_public_url(self, public_url: str) -> None:
        if self.uvicorn_server is not None:
            app = getattr(self.uvicorn_server.config, "app", None)
            if app is not None and hasattr(app, "state"):
                app.state.public_url = public_url

    def stop(self) -> None:
        """Idempotently shuts down uvicorn, terminates the tunnel, and unregisters the port."""
        if self.uvicorn_server is not None:
            self.uvicorn_server.should_exit = True
        if (
            self.server_thread is not None
            and self.server_thread.is_alive()
            and self.server_thread is not threading.current_thread()
        ):
            self.server_thread.join(timeout=3.0)

        proc = self.tunnel_proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=1.0)

        if self.ngrok_url:
            url = self.ngrok_url
            self.ngrok_url = None
            with contextlib.suppress(Exception):
                from pyngrok import ngrok  # type: ignore

                ngrok.disconnect(url)

        if _ACTIVE_SERVERS.get(self.port) is self:
            _ACTIVE_SERVERS.pop(self.port, None)

    def print_connection_guide(self) -> None:
        antigravity_cfg = {
            "mcpServers": {
                "colab-remote": {
                    "url": self.bearer_mcp_url,
                    "headers": {"Authorization": f"Bearer {self.token}"},
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
    claude mcp add -s user --transport http colab-remote {self.bearer_mcp_url} \\
      --header "Authorization: Bearer {self.token}"
================================================================================
"""
        print(banner, flush=True)

    def keep_alive(self, status_interval_seconds: int = 60) -> None:
        """Blocks the current cell so Google Colab treats the runtime as actively executing."""
        print(
            "Keep-alive monitor active. You can close your laptop lid now; "
            "interrupt the cell (Stop button / Ctrl+C) to stop monitoring.",
            flush=True,
        )
        try:
            while True:
                time.sleep(status_interval_seconds)

                if self.server_thread is not None and not self.server_thread.is_alive():
                    print(
                        "[colab-cloud-mcp] WARNING: Local uvicorn server thread has stopped. "
                        "Re-run start_colab_mcp(...) to restart.",
                        flush=True,
                    )

                if self.tunnel_proc is not None and self.tunnel_proc.poll() is not None:
                    exit_code = self.tunnel_proc.returncode
                    if self.provider == "cloudflared-named" and self._named_tunnel_token:
                        print(
                            f"[colab-cloud-mcp] WARNING: Named Tunnel exited (code={exit_code}); "
                            "respawning...",
                            flush=True,
                        )
                        try:
                            new_proc, new_url = _start_cloudflared_named_tunnel(
                                self._named_tunnel_token, self.public_url
                            )
                            self.tunnel_proc = new_proc
                            self.public_url = new_url
                            self._sync_app_public_url(new_url)
                            print(
                                f"[colab-cloud-mcp] Recovered Named Tunnel -> {self.public_url}",
                                flush=True,
                            )
                        except Exception as exc:
                            print(
                                f"[colab-cloud-mcp] ERROR: Failed to respawn Named Tunnel: {exc}",
                                flush=True,
                            )
                    elif self.provider == "cloudflared-quick":
                        print(
                            f"[colab-cloud-mcp] WARNING: Quick Tunnel exited (code={exit_code}); "
                            "respawning a new Quick Tunnel (URL will change)...",
                            flush=True,
                        )
                        try:
                            new_proc, new_url = _start_cloudflared_quick_tunnel(port=self.port)
                            self.tunnel_proc = new_proc
                            self.public_url = new_url
                            self._sync_app_public_url(new_url)
                            print(
                                f"[colab-cloud-mcp] Recovered Quick Tunnel -> {self.public_url}",
                                flush=True,
                            )
                            self.print_connection_guide()
                        except Exception as exc:
                            print(
                                f"[colab-cloud-mcp] ERROR: Failed to respawn Quick Tunnel: {exc}",
                                flush=True,
                            )

                jobs = self.runtime.list_jobs()
                running = sum(1 for j in jobs if j["status"] == "running")
                tunnel_state = (
                    "ok"
                    if self.tunnel_proc is None or self.tunnel_proc.poll() is None
                    else f"exited({self.tunnel_proc.returncode})"
                )
                print(
                    f"[colab-cloud-mcp heartbeat] uptime={int(time.time() - self.runtime.started_at)}s | "
                    f"cells_run={self.runtime.execution_count} | active_bg_jobs={running} | "
                    f"tunnel={tunnel_state}",
                    flush=True,
                )
        except KeyboardInterrupt:
            print("Stopping keep-alive monitor...", flush=True)


def _start_cloudflared_quick_tunnel(
    port: int, timeout_seconds: int = 30
) -> Tuple[subprocess.Popen, str]:
    bin_path = ensure_cloudflared()
    local_target = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [bin_path, "tunnel", "--no-autoupdate", "--url", local_target],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    found_url: Optional[str] = None
    url_event = threading.Event()
    log_lines: collections.deque[str] = collections.deque(maxlen=30)

    def _reader() -> None:
        nonlocal found_url
        assert proc.stdout is not None
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            if not found_url:
                match = TRYCLOUDFLARE_RE.search(line)
                if match:
                    found_url = match.group(0)
                    url_event.set()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if url_event.wait(timeout=0.25):
            break
        if proc.poll() is not None:
            reader_thread.join(timeout=1.0)
            tail = "\n".join(list(log_lines))
            raise RuntimeError(
                f"cloudflared Quick Tunnel exited prematurely (code {proc.returncode})."
                + (f"\nOutput:\n{tail}" if tail else "")
            )

    if not found_url:
        with contextlib.suppress(OSError):
            proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=2.0)
        reader_thread.join(timeout=1.0)
        tail = "\n".join(list(log_lines))
        raise RuntimeError(
            f"Timed out after {timeout_seconds}s waiting for Cloudflare Quick Tunnel URL."
            + (f"\nOutput:\n{tail}" if tail else "")
        )

    probe_deadline = time.monotonic() + 6.0
    probe_url = f"{found_url.rstrip('/')}/health"
    while time.monotonic() < probe_deadline:
        if proc.poll() is not None:
            reader_thread.join(timeout=1.0)
            tail = "\n".join(list(log_lines))
            raise RuntimeError(
                f"cloudflared Quick Tunnel exited during readiness probe (code {proc.returncode})."
                + (f"\nOutput:\n{tail}" if tail else "")
            )
        try:
            with urllib.request.urlopen(probe_url, timeout=2) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass
        remaining = probe_deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.5, remaining))

    return proc, found_url


def _start_cloudflared_named_tunnel(
    tunnel_token: str, public_hostname: str
) -> Tuple[subprocess.Popen, str]:
    bin_path = ensure_cloudflared()
    proc = subprocess.Popen(
        [bin_path, "tunnel", "--no-autoupdate", "run", "--token", tunnel_token],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_lines: collections.deque[str] = collections.deque(maxlen=30)

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            log_lines.append(line.rstrip())

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            reader_thread.join(timeout=1.0)
            tail = "\n".join(list(log_lines))
            raise RuntimeError(
                f"cloudflared Named Tunnel exited prematurely (code {proc.returncode})."
                + (f"\nOutput:\n{tail}" if tail else "")
            )
        time.sleep(0.15)

    clean_url = public_hostname.strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url
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
        kwargs["domain"] = domain.replace("https://", "").replace("http://", "").rstrip("/")
    listener = ngrok.connect(**kwargs)
    return str(listener.public_url).replace("http://", "https://").rstrip("/")


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
    if provider not in ("cloudflared", "ngrok", "none", "auto"):
        raise ValueError(
            f"Unsupported provider {provider!r}. Expected one of: cloudflared, ngrok, none, auto."
        )

    resolved_token = token or _read_colab_secret("MCP_TOKEN") or secrets.token_urlsafe(24)
    cf_tok = cloudflare_tunnel_token or _read_colab_secret("CLOUDFLARE_TUNNEL_TOKEN")
    cf_host = cloudflare_hostname or _read_colab_secret("CLOUDFLARE_HOSTNAME")
    ng_tok = ngrok_authtoken or _read_colab_secret("NGROK_AUTHTOKEN")
    ng_dom = ngrok_domain or _read_colab_secret("NGROK_DOMAIN")

    if provider == "ngrok" and not ng_tok:
        raise ValueError("NGROK_AUTHTOKEN is required when using provider='ngrok'.")

    if port in _ACTIVE_SERVERS:
        _ACTIVE_SERVERS[port].stop()

    runtime = ColabRuntime(workdir=workdir)
    instance_id = secrets.token_hex(8)
    app = create_app(token=resolved_token, runtime=runtime, instance_id=instance_id)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    def _stop_local_server() -> None:
        server.should_exit = True
        if server_thread.is_alive():
            server_thread.join(timeout=3.0)

    health_url = f"http://127.0.0.1:{port}/health"
    local_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = False
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with local_opener.open(health_url, timeout=1) as resp:
                if resp.status == 200:
                    try:
                        data = json.loads(resp.read().decode("utf-8"))
                    except Exception:
                        data = None
                    if isinstance(data, dict) and data.get("instance_id") == instance_id:
                        started = True
                        break
                    _stop_local_server()
                    raise RuntimeError(
                        f"Port {port} is already in use by another process. "
                        "Choose a different port or stop that process."
                    )
        except RuntimeError:
            raise
        except Exception:
            pass
        if not server_thread.is_alive():
            break
        time.sleep(0.1)

    if not started:
        _stop_local_server()
        raise RuntimeError(f"Uvicorn failed to start on 127.0.0.1:{port}.")

    tunnel_proc: Optional[subprocess.Popen] = None
    ngrok_url: Optional[str] = None
    named_tunnel_token: Optional[str] = None
    try:
        if provider == "none":
            public_url = f"http://127.0.0.1:{port}"
            resolved_provider = "none"
        elif provider == "ngrok" or (provider == "auto" and ng_tok and not cf_tok):
            assert ng_tok is not None
            public_url = _start_ngrok_tunnel(port=port, authtoken=ng_tok, domain=ng_dom)
            ngrok_url = public_url
            resolved_provider = "ngrok"
        else:
            if cf_tok and cf_host:
                tunnel_proc, public_url = _start_cloudflared_named_tunnel(cf_tok, cf_host)
                named_tunnel_token = cf_tok
                resolved_provider = "cloudflared-named"
            else:
                if bool(cf_tok) != bool(cf_host):
                    print(
                        "[colab-cloud-mcp] Warning: Both CLOUDFLARE_TUNNEL_TOKEN and "
                        "CLOUDFLARE_HOSTNAME must be set for a Cloudflare Named Tunnel; "
                        "falling back to Quick Tunnel.",
                        flush=True,
                    )
                tunnel_proc, public_url = _start_cloudflared_quick_tunnel(port=port)
                resolved_provider = "cloudflared-quick"
    except BaseException:
        _stop_local_server()
        raise

    app.state.public_url = public_url
    active = ActiveColabMCP(
        public_url=public_url,
        token=resolved_token,
        port=port,
        provider=resolved_provider,
        runtime=runtime,
        tunnel_proc=tunnel_proc,
        uvicorn_server=server,
        server_thread=server_thread,
        ngrok_url=ngrok_url,
        _named_tunnel_token=named_tunnel_token,
    )
    _ACTIVE_SERVERS[port] = active
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

    active = start_colab_mcp(
        token=args.token,
        port=args.port,
        provider=args.provider,
        workdir=args.workdir,
        block=False,
    )
    try:
        active.keep_alive()
    finally:
        active.stop()


if __name__ == "__main__":
    cli_main()
