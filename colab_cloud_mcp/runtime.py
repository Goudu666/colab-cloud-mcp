"""Stateful Colab Python execution, shell runner, background job manager, and file tools."""

from __future__ import annotations

import ast
import base64
import contextlib
import gc
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


@dataclass
class BackgroundJob:
    job_id: str
    name: str
    command: str
    cwd: str
    log_path: str
    started_at: float
    process: subprocess.Popen
    _log_handle: Any = field(repr=False, default=None)

    def poll_status(self) -> Dict[str, Any]:
        rc = self.process.poll()
        running = rc is None
        if not running and self._log_handle and not self._log_handle.closed:
            with contextlib.suppress(Exception):
                self._log_handle.flush()
                self._log_handle.close()
        elapsed = round(time.time() - self.started_at, 1)
        return {
            "job_id": self.job_id,
            "name": self.name,
            "pid": self.process.pid,
            "status": "running" if running else ("succeeded" if rc == 0 else "failed"),
            "exit_code": rc,
            "elapsed_seconds": elapsed,
            "cwd": self.cwd,
            "log_path": self.log_path,
            "command": self.command,
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

        self.namespace: Dict[str, Any] = {"__name__": "__main__", "__builtins__": __builtins__}
        self.execution_count = 0
        self.cell_history: List[Dict[str, Any]] = []
        self.jobs: Dict[str, BackgroundJob] = {}

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return os.path.abspath(path)
        return os.path.abspath(os.path.join(self.workdir, path))

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
            except Exception:
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
                                    "index": cols[0],
                                    "name": cols[1],
                                    "vram_total_mb": cols[2],
                                    "vram_used_mb": cols[3],
                                    "vram_free_mb": cols[4],
                                    "utilization_pct": cols[5],
                                }
                            )
            except Exception:
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
            try:
                import importlib.util

                torch_info["installed"] = importlib.util.find_spec("torch") is not None
            except Exception:
                pass

        drive_path = "/content/drive/MyDrive"
        with self._jobs_lock:
            active_jobs = sum(1 for j in self.jobs.values() if j.process.poll() is None)
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
            "google_drive_mounted": os.path.isdir(drive_path),
            "google_drive_path": drive_path if os.path.isdir(drive_path) else None,
            "execution_count": self.execution_count,
            "namespace_variables": user_vars[:50],
            "background_jobs": {"running": active_jobs, "total": total_jobs},
        }

    def _capture_matplotlib_figures(self) -> List[str]:
        """Captures open Matplotlib figures as base64 PNG strings and closes them."""
        if "matplotlib.pyplot" not in sys.modules:
            return []
        images_b64: List[str] = []
        try:
            plt = sys.modules["matplotlib.pyplot"]
            fignums = plt.get_fignums()
            for num in fignums:
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                plt.close(fig)
                images_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        except Exception:
            pass
        return images_b64

    def execute_python(
        self, code: str, timeout_seconds: int = 120
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Executes Python code statefully in the persistent namespace."""
        timeout_seconds = max(1, min(int(timeout_seconds), 220))
        if not self._exec_lock.acquire(timeout=5):
            return (
                {
                    "status": "busy",
                    "error": "Another foreground Python cell is currently executing. Wait or use start_background_job for long tasks.",
                },
                [],
            )

        try:
            os.environ.setdefault("MPLBACKEND", "Agg")
            self.execution_count += 1
            cell_num = self.execution_count

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            exec_state: Dict[str, Any] = {
                "result_repr": None,
                "error": None,
                "images": [],
            }

            def _runner() -> None:
                prev_cwd = os.getcwd()
                try:
                    os.chdir(self.workdir)
                    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
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
                        exec_state["images"] = self._capture_matplotlib_figures()
                except BaseException:
                    exec_state["error"] = traceback.format_exc()
                finally:
                    with contextlib.suppress(Exception):
                        self.workdir = os.getcwd()
                        os.chdir(prev_cwd)

            worker = threading.Thread(target=_runner, daemon=True)
            t0 = time.time()
            worker.start()
            worker.join(timeout=timeout_seconds)
            elapsed = round(time.time() - t0, 3)

            stdout_str = _truncate_text(stdout_buf.getvalue())
            stderr_str = _truncate_text(stderr_buf.getvalue())

            if worker.is_alive():
                return (
                    {
                        "status": "timeout",
                        "execution_count": cell_num,
                        "elapsed_seconds": elapsed,
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                        "error": (
                            f"Execution exceeded {timeout_seconds}s foreground limit. "
                            "For long training or downloads, use `start_background_job` instead."
                        ),
                    },
                    [],
                )

            result_repr = (
                _truncate_text(exec_state["result_repr"]) if exec_state["result_repr"] else None
            )
            error_str = _truncate_text(exec_state["error"]) if exec_state["error"] else None
            images_b64: List[str] = exec_state["images"]

            payload = {
                "status": "error" if error_str else "ok",
                "execution_count": cell_num,
                "elapsed_seconds": elapsed,
                "workdir": self.workdir,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "result": result_repr,
                "error": error_str,
                "figures_captured": len(images_b64),
            }

            self.cell_history.append(
                {
                    "execution_count": cell_num,
                    "code": code,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "result": result_repr,
                    "error": error_str,
                    "images": images_b64,
                }
            )
            return payload, images_b64
        finally:
            self._exec_lock.release()

    def reset_python_namespace(self) -> Dict[str, Any]:
        """Clears user variables and releases GPU VRAM cache if PyTorch is loaded."""
        with self._exec_lock:
            removed = [k for k in self.namespace.keys() if k not in ("__name__", "__builtins__")]
            self.namespace.clear()
            self.namespace.update({"__name__": "__main__", "__builtins__": __builtins__})
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
            }

    def run_shell(
        self, command: str, cwd: Optional[str] = None, timeout_seconds: int = 120
    ) -> Dict[str, Any]:
        """Executes a shell command with process-group timeout cleanup."""
        timeout_seconds = max(1, min(int(timeout_seconds), 220))
        target_cwd = self._resolve_path(cwd) if cwd else self.workdir
        if not os.path.isdir(target_cwd):
            return {"status": "error", "error": f"Directory does not exist: {target_cwd}"}

        t0 = time.time()
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=target_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            elapsed = round(time.time() - t0, 3)
            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "exit_code": proc.returncode,
                "elapsed_seconds": elapsed,
                "cwd": target_cwd,
                "stdout": _truncate_text(stdout),
                "stderr": _truncate_text(stderr),
            }
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            with contextlib.suppress(Exception):
                stdout, stderr = proc.communicate(timeout=2)
            return {
                "status": "timeout",
                "exit_code": None,
                "elapsed_seconds": round(time.time() - t0, 3),
                "cwd": target_cwd,
                "error": (
                    f"Command timed out after {timeout_seconds}s and was terminated. "
                    "Use `start_background_job` for long-running scripts or downloads."
                ),
            }

    def install_packages(self, packages: List[str], upgrade: bool = False) -> Dict[str, Any]:
        """Installs Python packages into the current runtime environment."""
        cleaned = [p.strip() for p in packages if p and p.strip()]
        if not cleaned:
            return {"status": "error", "error": "No package names provided."}

        cmd = [sys.executable, "-m", "pip", "install", "-q"]
        if upgrade:
            cmd.append("-U")
        cmd.extend(cleaned)

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=210)
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "packages": cleaned,
            "elapsed_seconds": round(time.time() - t0, 2),
            "stdout": _truncate_text(proc.stdout),
            "stderr": _truncate_text(proc.stderr),
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

        log_f = open(resolved_log, "w", encoding="utf-8", buffering=1)
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=target_cwd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        job = BackgroundJob(
            job_id=job_id,
            name=name,
            command=command,
            cwd=target_cwd,
            log_path=resolved_log,
            started_at=time.time(),
            process=proc,
            _log_handle=log_f,
        )
        with self._jobs_lock:
            self.jobs[job_id] = job
        return job.poll_status()

    def get_job_status(self, job_id: str, tail_lines: int = 80) -> Dict[str, Any]:
        """Returns status and the latest log lines for a background job."""
        with self._jobs_lock:
            job = self.jobs.get(job_id)
        if not job:
            return {"status": "not_found", "error": f"Unknown job_id: {job_id}"}

        info = job.poll_status()
        tail_lines = max(1, min(int(tail_lines), 500))
        log_tail = ""
        if os.path.exists(job.log_path):
            try:
                with open(job.log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    log_tail = _truncate_text("".join(lines[-tail_lines:]))
                    info["total_log_lines"] = len(lines)
            except Exception as exc:
                log_tail = f"[Error reading log: {exc}]"
        info["log_tail"] = log_tail
        return info

    def list_jobs(self) -> List[Dict[str, Any]]:
        """Lists all background jobs launched in this session."""
        with self._jobs_lock:
            jobs_list = list(self.jobs.values())
        return [j.poll_status() for j in jobs_list]

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Terminates a running background job process group."""
        with self._jobs_lock:
            job = self.jobs.get(job_id)
        if not job:
            return {"status": "not_found", "error": f"Unknown job_id: {job_id}"}

        if job.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(job.process.pid, signal.SIGTERM)
            try:
                job.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(job.process.pid, signal.SIGKILL)
                with contextlib.suppress(Exception):
                    job.process.wait(timeout=2)

        info = job.poll_status()
        info["cancelled"] = True
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
            }

        entries: List[Dict[str, Any]] = []
        for entry in sorted(os.scandir(target), key=lambda e: (not e.is_dir(), e.name.lower())):
            if len(entries) >= max_entries:
                break
            with contextlib.suppress(OSError):
                st = entry.stat(follow_symlinks=False)
                entries.append(
                    {
                        "name": entry.name + ("/" if entry.is_dir() else ""),
                        "type": "dir" if entry.is_dir() else "file",
                        "size_bytes": st.st_size if entry.is_file() else None,
                    }
                )
        return {"status": "ok", "path": target, "entries": entries, "count": len(entries)}

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 300) -> Dict[str, Any]:
        """Reads text file contents by line range."""
        target = self._resolve_path(path)
        if not os.path.isfile(target):
            return {"status": "error", "error": f"File not found: {target}"}

        start_line = max(1, int(start_line))
        max_lines = max(1, min(int(max_lines), 2000))
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        total = len(all_lines)
        sliced = all_lines[start_line - 1 : start_line - 1 + max_lines]
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
