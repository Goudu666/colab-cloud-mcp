"""Stateful Colab Python execution, shell runner, background job manager, and file tools."""

from __future__ import annotations

import ast
import base64
import builtins
import contextlib
import ctypes
import gc
import importlib.util
import io
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAX_OUTPUT_CHARS = 100_000
MAX_CELL_HISTORY = 200
TAIL_MAX_BYTES = 2 * 1024 * 1024
TAIL_CHUNK_SIZE = 65_536

_IMAGE_EXT_MIME: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_STREAM_INSTALL_LOCK = threading.Lock()


class _ThreadLocalStream:
    """Routes writes to a per-thread buffer when bound, else the underlying stream."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self._local = threading.local()

    def bind(self, target: Any) -> None:
        self._local.target = target

    def unbind(self) -> None:
        self._local.target = None

    def _current(self) -> Any:
        target = getattr(self._local, "target", None)
        return target if target is not None else self._wrapped

    @property
    def encoding(self) -> str:
        return getattr(self._wrapped, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self._wrapped, "errors", None) or "replace"

    def write(self, data: str) -> int:
        return self._current().write(data)

    def writelines(self, lines: Any) -> None:
        self._current().writelines(lines)

    def flush(self) -> None:
        self._current().flush()

    def isatty(self) -> bool:
        if getattr(self._local, "target", None) is not None:
            return False
        return bool(self._wrapped.isatty())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current(), name)


def _ensure_stream_proxies() -> Tuple[_ThreadLocalStream, _ThreadLocalStream]:
    with _STREAM_INSTALL_LOCK:
        if not isinstance(sys.stdout, _ThreadLocalStream):
            sys.stdout = _ThreadLocalStream(sys.stdout)
        if not isinstance(sys.stderr, _ThreadLocalStream):
            sys.stderr = _ThreadLocalStream(sys.stderr)
        return sys.stdout, sys.stderr


def _truncate_text(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = (limit - 200) // 2
    omitted = len(text) - (half * 2)
    return (
        f"{text[:half]}\n\n"
        f"... [truncated {omitted:,} characters to fit MCP output limits] ...\n\n"
        f"{text[-half:]}"
    )


def _coerce_output(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _parse_optional_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_pid_alive(pid: int) -> bool:
    """Checks whether a PID is still alive and not a zombie process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass

    stat_path = f"/proc/{pid}/stat"
    if os.path.exists(stat_path):
        try:
            with open(stat_path, "r", encoding="utf-8", errors="replace") as f:
                stat_text = f.read()
            rparen = stat_text.rfind(")")
            if rparen != -1:
                fields = stat_text[rparen + 1 :].split()
                if fields and fields[0] == "Z":
                    return False
        except OSError:
            return False
    elif not os.path.isdir("/proc") and shutil.which("ps"):
        try:
            proc = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=2,
            )
            state = proc.stdout.strip()
            if proc.returncode != 0 or not state or state.startswith("Z"):
                return False
        except (OSError, subprocess.SubprocessError):
            pass
    return True


def _read_log_tail(
    path: str,
    tail_lines: int,
    max_bytes: int = TAIL_MAX_BYTES,
    chunk_size: int = TAIL_CHUNK_SIZE,
) -> Tuple[str, int]:
    """Reads the last `tail_lines` lines from `path` in O(1) memory via reverse seek."""
    file_size = os.path.getsize(path)
    if file_size == 0:
        return "", 0

    chunks: List[bytes] = []
    bytes_read = 0
    newline_count = 0
    pos = file_size

    with open(path, "rb") as f:
        while pos > 0 and bytes_read < max_bytes and newline_count <= tail_lines:
            step = min(chunk_size, pos, max_bytes - bytes_read)
            pos -= step
            f.seek(pos, os.SEEK_SET)
            chunk = f.read(step)
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            newline_count += chunk.count(b"\n")

    raw = b"".join(reversed(chunks))
    if pos > 0 and b"\n" in raw:
        # Drop leading partial line/UTF-8 fragment when we did not read from BOF.
        raw = raw.split(b"\n", 1)[1]

    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    raw_lines = text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()

    cleaned_lines: List[str] = []
    for line in raw_lines:
        if "\r" in line:
            parts = [p for p in line.split("\r") if p]
            line = parts[-1] if parts else ""
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines[-tail_lines:]), file_size


def _detect_image_mime(path: str, header: bytes) -> Optional[str]:
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return _IMAGE_EXT_MIME.get(os.path.splitext(path)[1].lower())


def _run_subprocess_with_timeout(
    cmd: List[str],
    cwd: Optional[str],
    timeout_seconds: int,
) -> Tuple[bool, Optional[int], str, str, float]:
    """Executes `cmd` in a new process group; sends SIGTERM then SIGKILL on timeout."""
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return False, proc.returncode, stdout, stderr, round(time.time() - t0, 3)
    except subprocess.TimeoutExpired as exc:
        partial_out: Any = exc.stdout
        partial_err: Any = exc.stderr
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=2)
            partial_out, partial_err = stdout, stderr
        except subprocess.TimeoutExpired as exc2:
            if exc2.stdout is not None:
                partial_out = exc2.stdout
            if exc2.stderr is not None:
                partial_err = exc2.stderr
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            try:
                stdout, stderr = proc.communicate(timeout=2)
                partial_out, partial_err = stdout, stderr
            except subprocess.TimeoutExpired as exc3:
                if exc3.stdout is not None:
                    partial_out = exc3.stdout
                if exc3.stderr is not None:
                    partial_err = exc3.stderr
        return (
            True,
            None,
            _coerce_output(partial_out),
            _coerce_output(partial_err),
            round(time.time() - t0, 3),
        )


@dataclass
class BackgroundJob:
    job_id: str
    name: str
    pid: int
    command: str
    cwd: str
    log_path: str
    started_at: float
    process: Optional[subprocess.Popen] = field(repr=False, default=None)
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    status: str = "running"

    def refresh(self) -> bool:
        """Polls process/PID state; returns True if persisted metadata changed."""
        changed = False
        if self.process is not None:
            rc = self.process.poll()
            if rc is not None:
                if self.exit_code != rc:
                    self.exit_code = rc
                    changed = True
                if self.finished_at is None:
                    self.finished_at = time.time()
                    changed = True
                if self.status == "running":
                    self.status = "succeeded" if rc == 0 else "failed"
                    changed = True
        elif self.status == "running":
            if not _is_pid_alive(self.pid):
                self.status = "finished"
                if self.finished_at is None:
                    finish_ts = time.time()
                    with contextlib.suppress(OSError):
                        mtime = os.path.getmtime(self.log_path)
                        if mtime >= self.started_at:
                            finish_ts = mtime
                    self.finished_at = finish_ts
                changed = True
        return changed

    def is_running(self) -> bool:
        self.refresh()
        return self.status == "running"

    def to_status_dict(self) -> Dict[str, Any]:
        end_t = self.finished_at if self.finished_at is not None else time.time()
        elapsed = round(max(0.0, end_t - self.started_at), 1)
        return {
            "job_id": self.job_id,
            "name": self.name,
            "pid": self.pid,
            "status": self.status,
            "exit_code": self.exit_code,
            "elapsed_seconds": elapsed,
            "cwd": self.cwd,
            "log_path": self.log_path,
            "command": self.command,
        }

    def poll_status(self) -> Dict[str, Any]:
        self.refresh()
        return self.to_status_dict()

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "pid": self.pid,
            "command": self.command,
            "cwd": self.cwd,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "status": self.status,
        }


class ColabRuntime:
    """Manages persistent Python execution, background jobs, shell commands, and files."""

    def __init__(self, workdir: Optional[str] = None) -> None:
        default_dir = "/content" if os.path.isdir("/content") else os.getcwd()
        self.workdir = os.path.abspath(workdir or default_dir)
        os.makedirs(self.workdir, exist_ok=True)

        self.started_at = time.time()
        self._exec_lock = threading.Lock()
        self._jobs_lock = threading.Lock()
        self.jobs_dir = os.path.join(self.workdir, ".colab_mcp_jobs")
        os.makedirs(self.jobs_dir, exist_ok=True)
        self._jobs_meta_path = os.path.join(self.jobs_dir, "jobs.json")

        self.namespace: Dict[str, Any] = {"__name__": "__main__", "__builtins__": __builtins__}
        self.execution_count = 0
        self.cell_history: List[Dict[str, Any]] = []
        self.jobs: Dict[str, BackgroundJob] = {}
        self._stuck_worker: Optional[threading.Thread] = None
        self._pending_restore_cwd: Optional[str] = None
        self._active_cell_images: Optional[List[str]] = None

        self._load_jobs()

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return os.path.abspath(path)
        return os.path.abspath(os.path.join(self.workdir, path))

    def _save_jobs_locked(self) -> None:
        os.makedirs(self.jobs_dir, exist_ok=True)
        payload = [job.to_metadata() for job in self.jobs.values()]
        tmp_path = f"{self._jobs_meta_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._jobs_meta_path)

    def _load_jobs(self) -> None:
        if not os.path.isfile(self._jobs_meta_path):
            return
        try:
            with open(self._jobs_meta_path, "r", encoding="utf-8") as f:
                items = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            return
        if not isinstance(items, list):
            return

        dirty = False
        with self._jobs_lock:
            for item in items:
                if not isinstance(item, dict) or "job_id" not in item or "pid" not in item:
                    continue
                job = BackgroundJob(
                    job_id=str(item["job_id"]),
                    name=str(item.get("name", "job")),
                    pid=int(item["pid"]),
                    command=str(item.get("command", "")),
                    cwd=str(item.get("cwd", self.workdir)),
                    log_path=str(item.get("log_path", "")),
                    started_at=float(item.get("started_at", self.started_at)),
                    process=None,
                    finished_at=(
                        float(item["finished_at"])
                        if item.get("finished_at") is not None
                        else None
                    ),
                    exit_code=(
                        int(item["exit_code"])
                        if item.get("exit_code") is not None
                        else None
                    ),
                    status=str(item.get("status", "running")),
                )
                if job.refresh():
                    dirty = True
                self.jobs[job.job_id] = job
            if dirty:
                self._save_jobs_locked()

    def _check_stuck_worker_locked(self) -> Optional[Dict[str, Any]]:
        if self._stuck_worker is not None:
            if self._stuck_worker.is_alive():
                return {
                    "status": "busy",
                    "error": (
                        "A previously timed-out Python cell is still running in a background "
                        "thread (likely blocked in native/CUDA code). Wait for it to exit "
                        "before running new Python cells or resetting the namespace."
                    ),
                }
            self._stuck_worker = None
            if self._pending_restore_cwd and os.path.isdir(self._pending_restore_cwd):
                with contextlib.suppress(OSError):
                    os.chdir(self._pending_restore_cwd)
            self._pending_restore_cwd = None
        return None

    def get_runtime_status(self) -> Dict[str, Any]:
        """Collects CPU, RAM, Disk, GPU (nvidia-smi), PyTorch, and Google Drive mount status."""
        disk = shutil.disk_usage(self.workdir)
        ram_info: Dict[str, Any] = {}
        if os.path.exists("/proc/meminfo"):
            try:
                mem_map: Dict[str, int] = {}
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val_kb = int(parts[1].strip().split()[0])
                            mem_map[key] = val_kb * 1024
                if "MemTotal" in mem_map:
                    total = mem_map["MemTotal"]
                    avail = mem_map.get("MemAvailable", mem_map.get("MemFree", 0))
                    ram_info = {
                        "total_gb": round(total / (1024**3), 2),
                        "available_gb": round(avail / (1024**3), 2),
                        "used_gb": round((total - avail) / (1024**3), 2),
                    }
            except OSError:
                pass

        gpus: List[Dict[str, Any]] = []
        if shutil.which("nvidia-smi"):
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    for line in proc.stdout.strip().splitlines():
                        cols = [c.strip() for c in line.split(",")]
                        if len(cols) >= 6:
                            gpus.append(
                                {
                                    "index": _parse_optional_int(cols[0]),
                                    "name": cols[1],
                                    "vram_total_mb": _parse_optional_int(cols[2]),
                                    "vram_used_mb": _parse_optional_int(cols[3]),
                                    "vram_free_mb": _parse_optional_int(cols[4]),
                                    "utilization_pct": _parse_optional_int(cols[5]),
                                }
                            )
            except (OSError, subprocess.SubprocessError):
                pass

        torch_info: Dict[str, Any] = {"installed": False}
        if "torch" in sys.modules:
            torch_mod = sys.modules["torch"]
            with contextlib.suppress(Exception):
                torch_info = {
                    "installed": True,
                    "version": getattr(torch_mod, "__version__", "unknown"),
                    "cuda_available": bool(torch_mod.cuda.is_available()),
                }
        else:
            torch_info["installed"] = importlib.util.find_spec("torch") is not None

        drive_path = "/content/drive/MyDrive"
        drive_dir_exists = os.path.isdir(drive_path)
        drive_mounted = drive_dir_exists and (
            os.path.ismount("/content/drive") or os.path.ismount(drive_path)
        )

        with self._jobs_lock:
            dirty = False
            active_jobs = 0
            for job in self.jobs.values():
                if job.refresh():
                    dirty = True
                if job.status == "running":
                    active_jobs += 1
            if dirty:
                self._save_jobs_locked()
            total_jobs = len(self.jobs)

        user_vars = sorted(
            k for k in self.namespace.keys() if not k.startswith("_") and k != "In" and k != "Out"
        )

        return {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "is_colab_vm": os.path.isdir("/content") or "COLAB_GPU" in os.environ,
            "workdir": self.workdir,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "cpu_count": os.cpu_count(),
            "ram": ram_info,
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
            },
            "gpus": gpus,
            "pytorch": torch_info,
            "google_drive_mounted": drive_mounted,
            "google_drive_dir_exists": drive_dir_exists,
            "google_drive_path": drive_path if drive_mounted else None,
            "execution_count": self.execution_count,
            "namespace_variables": user_vars[:50],
            "background_jobs": {"running": active_jobs, "total": total_jobs},
        }

    def _capture_matplotlib_figures(self) -> List[str]:
        """Captures open Matplotlib figures as base64 PNG strings and closes them."""
        plt = sys.modules.get("matplotlib.pyplot")
        if plt is None or not hasattr(plt, "get_fignums"):
            return []
        images_b64: List[str] = []
        try:
            for num in list(plt.get_fignums()):
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                plt.close(fig)
                images_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        except Exception:
            pass
        return images_b64

    def _prepare_matplotlib_backend(self) -> None:
        if "matplotlib" not in sys.modules:
            os.environ["MPLBACKEND"] = "Agg"
            return
        mpl = sys.modules["matplotlib"]
        with contextlib.suppress(Exception):
            if str(mpl.get_backend()).lower() != "agg":
                mpl.use("Agg", force=True)

    def execute_python(
        self, code: str, timeout_seconds: int = 120
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Executes Python code statefully in the persistent namespace."""
        timeout_seconds = max(1, min(int(timeout_seconds), 220))
        if not self._exec_lock.acquire(timeout=5):
            return (
                {
                    "status": "busy",
                    "error": (
                        "Another foreground Python cell is currently executing. "
                        "Wait or use start_background_job for long tasks."
                    ),
                },
                [],
            )

        try:
            busy = self._check_stuck_worker_locked()
            if busy is not None:
                return busy, []

            self._prepare_matplotlib_backend()
            stdout_proxy, stderr_proxy = _ensure_stream_proxies()

            self.execution_count += 1
            cell_num = self.execution_count

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            captured_images: List[str] = []
            exec_state: Dict[str, Any] = {
                "result_repr": None,
                "error": None,
            }

            try:
                prev_cwd: Optional[str] = os.getcwd()
            except OSError:
                prev_cwd = None
            os.chdir(self.workdir)

            self._active_cell_images = captured_images
            hook_state: Dict[str, Any] = {
                "plt": None,
                "orig_show": None,
                "hooked_show": None,
                "orig_import": None,
                "import_hook": None,
            }

            def _patch_pyplot() -> None:
                plt = sys.modules.get("matplotlib.pyplot")
                if plt is None:
                    return
                show_fn = getattr(plt, "show", None)
                if not callable(show_fn) or show_fn is hook_state["hooked_show"]:
                    return
                orig_show = show_fn

                def _hooked_show(*args: Any, **kwargs: Any) -> Any:
                    active = self._active_cell_images
                    if active is not None:
                        active.extend(self._capture_matplotlib_figures())
                        return None
                    return orig_show(*args, **kwargs)

                hook_state["plt"] = plt
                hook_state["orig_show"] = orig_show
                hook_state["hooked_show"] = _hooked_show
                plt.show = _hooked_show

            def _restore_hooks() -> None:
                import_hook = hook_state["import_hook"]
                if import_hook is not None and builtins.__import__ is import_hook:
                    builtins.__import__ = hook_state["orig_import"]
                    hook_state["import_hook"] = None
                plt = hook_state["plt"]
                if plt is not None and getattr(plt, "show", None) is hook_state["hooked_show"]:
                    plt.show = hook_state["orig_show"]
                    hook_state["plt"] = None

            def _runner() -> None:
                worker_tid = threading.get_ident()
                stdout_proxy.bind(stdout_buf)
                stderr_proxy.bind(stderr_buf)
                try:
                    if "matplotlib.pyplot" in sys.modules:
                        _patch_pyplot()
                    else:
                        orig_import = builtins.__import__
                        hook_state["orig_import"] = orig_import

                        def _import_hook(
                            name: str,
                            globals: Any = None,
                            locals: Any = None,
                            fromlist: Any = (),
                            level: int = 0,
                        ) -> Any:
                            mod = orig_import(name, globals, locals, fromlist, level)
                            if (
                                threading.get_ident() == worker_tid
                                and hook_state["plt"] is None
                                and "matplotlib.pyplot" in sys.modules
                            ):
                                _patch_pyplot()
                            return mod

                        hook_state["import_hook"] = _import_hook
                        builtins.__import__ = _import_hook

                    parsed = ast.parse(code, filename=f"<colab-mcp-cell-{cell_num}>", mode="exec")
                    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                        last_expr = ast.Expression(parsed.body.pop().value)
                        ast.fix_missing_locations(parsed)
                        ast.fix_missing_locations(last_expr)
                        if parsed.body:
                            exec(
                                compile(parsed, f"<colab-mcp-cell-{cell_num}>", "exec"),
                                self.namespace,
                            )
                        val = eval(
                            compile(last_expr, f"<colab-mcp-cell-{cell_num}>", "eval"),
                            self.namespace,
                        )
                        self.namespace["_"] = val
                        if val is not None:
                            exec_state["result_repr"] = repr(val)
                    else:
                        exec(
                            compile(parsed, f"<colab-mcp-cell-{cell_num}>", "exec"),
                            self.namespace,
                        )
                    captured_images.extend(self._capture_matplotlib_figures())
                except BaseException:
                    exec_state["error"] = traceback.format_exc()
                finally:
                    _restore_hooks()
                    stdout_proxy.unbind()
                    stderr_proxy.unbind()

            worker = threading.Thread(target=_runner, daemon=True)
            t0 = time.time()
            worker.start()
            worker.join(timeout=timeout_seconds)

            if worker.is_alive():
                tid = worker.ident
                if tid is not None:
                    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(tid), ctypes.py_object(KeyboardInterrupt)
                    )
                    if res > 1:
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
                worker.join(timeout=1.0)
                elapsed = round(time.time() - t0, 3)
                still_alive = worker.is_alive()

                self._active_cell_images = None
                _restore_hooks()

                if still_alive:
                    self._stuck_worker = worker
                    self._pending_restore_cwd = prev_cwd
                elif prev_cwd and os.path.isdir(prev_cwd):
                    with contextlib.suppress(OSError):
                        os.chdir(prev_cwd)

                return (
                    {
                        "status": "timeout",
                        "execution_count": cell_num,
                        "elapsed_seconds": elapsed,
                        "interrupted": not still_alive,
                        "stdout": _truncate_text(stdout_buf.getvalue()),
                        "stderr": _truncate_text(stderr_buf.getvalue()),
                        "error": (
                            f"Execution exceeded {timeout_seconds}s foreground limit. "
                            "For long training or downloads, use `start_background_job` instead."
                        ),
                    },
                    [],
                )

            elapsed = round(time.time() - t0, 3)
            self._active_cell_images = None
            _restore_hooks()

            self.workdir = os.getcwd()
            if prev_cwd and prev_cwd != self.workdir and os.path.isdir(prev_cwd):
                with contextlib.suppress(OSError):
                    os.chdir(prev_cwd)

            stdout_str = _truncate_text(stdout_buf.getvalue())
            stderr_str = _truncate_text(stderr_buf.getvalue())
            result_repr = (
                _truncate_text(exec_state["result_repr"]) if exec_state["result_repr"] else None
            )
            error_str = _truncate_text(exec_state["error"]) if exec_state["error"] else None

            payload = {
                "status": "error" if error_str else "ok",
                "execution_count": cell_num,
                "elapsed_seconds": elapsed,
                "workdir": self.workdir,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "result": result_repr,
                "error": error_str,
                "figures_captured": len(captured_images),
            }

            self.cell_history.append(
                {
                    "execution_count": cell_num,
                    "code": code,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "result": result_repr,
                    "error": error_str,
                    "images": captured_images,
                }
            )
            if len(self.cell_history) > MAX_CELL_HISTORY:
                del self.cell_history[:-MAX_CELL_HISTORY]

            return payload, captured_images
        finally:
            self._active_cell_images = None
            self._exec_lock.release()

    def reset_python_namespace(self, clear_history: bool = False) -> Dict[str, Any]:
        """Clears user variables and releases GPU VRAM cache if PyTorch is loaded."""
        if not self._exec_lock.acquire(timeout=5):
            return {
                "status": "busy",
                "error": "A foreground Python cell is currently executing; wait before resetting.",
            }
        try:
            busy = self._check_stuck_worker_locked()
            if busy is not None:
                return busy

            removed = [k for k in self.namespace.keys() if k not in ("__name__", "__builtins__")]
            self.namespace.clear()
            self.namespace.update({"__name__": "__main__", "__builtins__": __builtins__})
            if clear_history:
                self.cell_history.clear()

            gc.collect()
            cuda_cleared = False
            if "torch" in sys.modules:
                with contextlib.suppress(Exception):
                    torch_mod = sys.modules["torch"]
                    if torch_mod.cuda.is_available():
                        torch_mod.cuda.empty_cache()
                        cuda_cleared = True
            return {
                "status": "ok",
                "cleared_variables_count": len(removed),
                "cuda_cache_cleared": cuda_cleared,
                "history_cleared": bool(clear_history),
            }
        finally:
            self._exec_lock.release()

    def run_shell(
        self, command: str, cwd: Optional[str] = None, timeout_seconds: int = 120
    ) -> Dict[str, Any]:
        """Executes a shell command with process-group timeout cleanup."""
        timeout_seconds = max(1, min(int(timeout_seconds), 220))
        target_cwd = self._resolve_path(cwd) if cwd else self.workdir
        if not os.path.isdir(target_cwd):
            return {"status": "error", "error": f"Directory does not exist: {target_cwd}"}

        timed_out, returncode, stdout, stderr, elapsed = _run_subprocess_with_timeout(
            ["/bin/bash", "-lc", command], target_cwd, timeout_seconds
        )
        if timed_out:
            return {
                "status": "timeout",
                "exit_code": None,
                "elapsed_seconds": elapsed,
                "cwd": target_cwd,
                "stdout": _truncate_text(stdout),
                "stderr": _truncate_text(stderr),
                "error": (
                    f"Command timed out after {timeout_seconds}s and was terminated. "
                    "Use `start_background_job` for long-running scripts or downloads."
                ),
            }

        return {
            "status": "ok" if returncode == 0 else "error",
            "exit_code": returncode,
            "elapsed_seconds": elapsed,
            "cwd": target_cwd,
            "stdout": _truncate_text(stdout),
            "stderr": _truncate_text(stderr),
        }

    def install_packages(
        self,
        packages: List[str],
        upgrade: bool = False,
        timeout_seconds: int = 210,
    ) -> Dict[str, Any]:
        """Installs Python packages into the current runtime environment."""
        cleaned = [str(p).strip() for p in packages if p and str(p).strip()]
        if not cleaned:
            return {"status": "error", "error": "No package names provided."}

        invalid = [p for p in cleaned if p.startswith("-")]
        if invalid:
            return {
                "status": "error",
                "error": f"Package specifications must not start with '-': {', '.join(invalid)}",
            }

        timeout_seconds = max(1, min(int(timeout_seconds), 600))
        cmd = [sys.executable, "-m", "pip", "install", "-q"]
        if upgrade:
            cmd.append("-U")
        cmd.extend(cleaned)

        timed_out, returncode, stdout, stderr, elapsed = _run_subprocess_with_timeout(
            cmd, self.workdir, timeout_seconds
        )
        if timed_out:
            return {
                "status": "timeout",
                "exit_code": None,
                "packages": cleaned,
                "elapsed_seconds": elapsed,
                "stdout": _truncate_text(stdout),
                "stderr": _truncate_text(stderr),
                "error": (
                    f"Package installation timed out after {timeout_seconds}s and was terminated. "
                    "Use `start_background_job` for very large package builds."
                ),
            }

        return {
            "status": "ok" if returncode == 0 else "error",
            "exit_code": returncode,
            "packages": cleaned,
            "elapsed_seconds": elapsed,
            "stdout": _truncate_text(stdout),
            "stderr": _truncate_text(stderr),
        }

    def start_background_job(
        self,
        command: str,
        name: str = "job",
        cwd: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Starts a detached background command streaming logs to disk (or Google Drive)."""
        target_cwd = self._resolve_path(cwd) if cwd else self.workdir
        os.makedirs(target_cwd, exist_ok=True)

        job_id = uuid.uuid4().hex[:8]
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:32] or "job"
        resolved_log = (
            self._resolve_path(log_path)
            if log_path
            else os.path.join(self.jobs_dir, f"{safe_name}_{job_id}.log")
        )
        os.makedirs(os.path.dirname(resolved_log), exist_ok=True)

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        log_f = open(resolved_log, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                ["/bin/bash", "-lc", command],
                cwd=target_cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        finally:
            log_f.close()

        job = BackgroundJob(
            job_id=job_id,
            name=name,
            pid=proc.pid,
            command=command,
            cwd=target_cwd,
            log_path=resolved_log,
            started_at=time.time(),
            process=proc,
        )
        with self._jobs_lock:
            self.jobs[job_id] = job
            job.refresh()
            self._save_jobs_locked()
            return job.to_status_dict()

    def get_job_status(self, job_id: str, tail_lines: int = 80) -> Dict[str, Any]:
        """Returns status and the latest log lines for a background job."""
        with self._jobs_lock:
            job = self.jobs.get(job_id)
            if not job:
                return {"status": "not_found", "error": f"Unknown job_id: {job_id}"}
            changed = job.refresh()
            info = job.to_status_dict()
            if changed:
                self._save_jobs_locked()

        tail_lines = max(1, min(int(tail_lines), 500))
        log_tail = ""
        log_size = 0
        if job.log_path and os.path.exists(job.log_path):
            try:
                log_tail, log_size = _read_log_tail(job.log_path, tail_lines=tail_lines)
                log_tail = _truncate_text(log_tail)
            except OSError as exc:
                log_tail = f"[Error reading log: {exc}]"
        info["log_size_bytes"] = log_size
        info["log_tail"] = log_tail
        return info

    def list_jobs(self) -> List[Dict[str, Any]]:
        """Lists all background jobs launched or recovered in this workdir."""
        with self._jobs_lock:
            dirty = False
            results: List[Dict[str, Any]] = []
            for job in self.jobs.values():
                if job.refresh():
                    dirty = True
                results.append(job.to_status_dict())
            if dirty:
                self._save_jobs_locked()
            return results

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Terminates a running background job process group."""
        with self._jobs_lock:
            job = self.jobs.get(job_id)
        if not job:
            return {"status": "not_found", "error": f"Unknown job_id: {job_id}"}

        was_running = job.is_running()
        if was_running:
            if job.process is not None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(job.pid, signal.SIGTERM)
                try:
                    job.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(job.pid, signal.SIGKILL)
                    with contextlib.suppress(Exception):
                        job.process.wait(timeout=2)
                job.exit_code = job.process.poll()
            else:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(job.pid, signal.SIGTERM)
                deadline = time.time() + 3.0
                while time.time() < deadline and _is_pid_alive(job.pid):
                    time.sleep(0.1)
                if _is_pid_alive(job.pid):
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(job.pid, signal.SIGKILL)
                    deadline = time.time() + 2.0
                    while time.time() < deadline and _is_pid_alive(job.pid):
                        time.sleep(0.1)
            job.status = "cancelled"
            job.finished_at = time.time()

        with self._jobs_lock:
            job.refresh()
            self._save_jobs_locked()
            info = job.to_status_dict()
        info["cancelled"] = was_running or job.status == "cancelled"
        return info

    def list_files(self, path: str = ".", max_entries: int = 200) -> Dict[str, Any]:
        """Lists directory contents with sizes and modification times."""
        target = self._resolve_path(path)
        if not os.path.exists(target):
            return {"status": "error", "error": f"Path not found: {target}"}
        if not os.path.isdir(target):
            st = os.stat(target)
            return {
                "status": "ok",
                "path": target,
                "type": "file",
                "size_bytes": st.st_size,
                "mtime_epoch": round(st.st_mtime, 1),
            }

        max_entries = max(1, int(max_entries))
        with os.scandir(target) as it:
            all_entries = sorted(it, key=lambda e: (not e.is_dir(), e.name.lower()))
        total_count = len(all_entries)

        entries: List[Dict[str, Any]] = []
        for entry in all_entries:
            if len(entries) >= max_entries:
                break
            with contextlib.suppress(OSError):
                st = entry.stat(follow_symlinks=False)
                is_dir = entry.is_dir()
                entries.append(
                    {
                        "name": entry.name + ("/" if is_dir else ""),
                        "type": "dir" if is_dir else "file",
                        "size_bytes": st.st_size if entry.is_file() else None,
                        "mtime_epoch": round(st.st_mtime, 1),
                    }
                )
        return {
            "status": "ok",
            "path": target,
            "entries": entries,
            "count": len(entries),
            "total_entries": total_count,
            "truncated": total_count > len(entries),
        }

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 300) -> Dict[str, Any]:
        """Reads text file contents by line range using streaming iteration."""
        target = self._resolve_path(path)
        if not os.path.isfile(target):
            return {"status": "error", "error": f"File not found: {target}"}

        start_line = max(1, int(start_line))
        max_lines = max(1, min(int(max_lines), 2000))
        end_bound = start_line + max_lines - 1

        sliced: List[str] = []
        total = 0
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                total = lineno
                if start_line <= lineno <= end_bound:
                    sliced.append(line)

        return {
            "status": "ok",
            "path": target,
            "start_line": start_line,
            "end_line": start_line - 1 + len(sliced),
            "total_lines": total,
            "content": _truncate_text("".join(sliced)),
        }

    def write_file(self, path: str, content: str, append: bool = False) -> Dict[str, Any]:
        """Writes or appends UTF-8 text to a file."""
        target = self._resolve_path(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        mode = "a" if append else "w"
        with open(target, mode, encoding="utf-8") as f:
            f.write(content)
        return {
            "status": "ok",
            "path": target,
            "bytes_written": len(content.encode("utf-8")),
            "mode": "append" if append else "overwrite",
        }

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> Dict[str, Any]:
        """Performs exact string replacement in a UTF-8 text file."""
        target = self._resolve_path(path)
        if not os.path.isfile(target):
            return {"status": "error", "error": f"File not found: {target}"}
        if old_text == "":
            return {"status": "error", "error": "old_text must be non-empty."}

        try:
            with open(target, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return {"status": "error", "error": f"File is not valid UTF-8 text: {target}"}

        count = content.count(old_text)
        if count == 0:
            return {"status": "error", "error": "old_text not found in file."}
        if count > 1 and not replace_all:
            return {
                "status": "error",
                "error": (
                    f"old_text matched {count} locations; "
                    "provide more context or set replace_all=True."
                ),
            }

        updated = (
            content.replace(old_text, new_text)
            if replace_all
            else content.replace(old_text, new_text, 1)
        )
        replaced_count = count if replace_all else 1
        with open(target, "w", encoding="utf-8") as f:
            f.write(updated)

        return {
            "status": "ok",
            "path": target,
            "replacements": replaced_count,
        }

    def read_image(self, path: str, max_bytes: int = 10 * 1024 * 1024) -> Dict[str, Any]:
        """Reads a PNG/JPEG/WebP/GIF image file and returns base64-encoded payload."""
        target = self._resolve_path(path)
        if not os.path.isfile(target):
            return {"status": "error", "error": f"File not found: {target}"}

        size = os.path.getsize(target)
        if size == 0:
            return {"status": "error", "error": f"Image file is empty: {target}"}
        if size > max_bytes:
            return {
                "status": "error",
                "error": f"Image file exceeds max_bytes limit ({size} > {max_bytes}).",
            }

        with open(target, "rb") as f:
            data = f.read()

        mime = _detect_image_mime(target, data[:16])
        if mime is None:
            return {
                "status": "error",
                "error": (
                    f"Unsupported image format for {target}; "
                    "supported formats are PNG, JPEG, WebP, and GIF."
                ),
            }

        return {
            "status": "ok",
            "path": target,
            "mime_type": mime,
            "size_bytes": size,
            "data_base64": base64.b64encode(data).decode("ascii"),
        }

    def export_session_notebook(self, output_path: str = "mcp_session_export.ipynb") -> Dict[str, Any]:
        """Exports all executed Python cells and outputs (including plots) to a Jupyter .ipynb file."""
        target = self._resolve_path(output_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        cells: List[Dict[str, Any]] = []
        for item in self.cell_history:
            outputs: List[Dict[str, Any]] = []
            if item["stdout"]:
                outputs.append(
                    {"output_type": "stream", "name": "stdout", "text": item["stdout"].splitlines(True)}
                )
            if item["stderr"]:
                outputs.append(
                    {"output_type": "stream", "name": "stderr", "text": item["stderr"].splitlines(True)}
                )
            if item["result"] is not None:
                outputs.append(
                    {
                        "output_type": "execute_result",
                        "execution_count": item["execution_count"],
                        "data": {"text/plain": item["result"].splitlines(True)},
                        "metadata": {},
                    }
                )
            for img_b64 in item.get("images", []):
                outputs.append(
                    {
                        "output_type": "display_data",
                        "data": {"image/png": img_b64},
                        "metadata": {},
                    }
                )
            if item["error"]:
                outputs.append(
                    {
                        "output_type": "error",
                        "ename": "ExecutionError",
                        "evalue": "",
                        "traceback": item["error"].splitlines(),
                    }
                )
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": item["execution_count"],
                    "metadata": {},
                    "source": item["code"].splitlines(True),
                    "outputs": outputs,
                }
            )

        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "colab": {"name": Path(target).name},
            },
            "cells": cells,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=2)

        return {
            "status": "ok",
            "notebook_path": target,
            "cells_exported": len(cells),
        }
