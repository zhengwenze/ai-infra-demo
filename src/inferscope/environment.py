"""Capture reproducibility metadata without collecting secrets."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil

TRACKED_PACKAGES = (
    "inferscope",
    "httpx",
    "numpy",
    "pydantic",
    "pyyaml",
    "prometheus-client",
    "psutil",
    "torch",
    "transformers",
    "vllm",
)


def _run_command(args: list[str], cwd: Path | None = None) -> str | None:
    """Run a read-only command and return stripped stdout on success."""
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def capture_git_state(project_dir: Path) -> dict[str, Any]:
    """Return commit and dirty-state metadata, tolerating non-Git directories."""
    inside = _run_command(["git", "rev-parse", "--is-inside-work-tree"], project_dir)
    if inside != "true":
        return {"available": False, "commit": None, "branch": None, "dirty": None}
    commit = _run_command(["git", "rev-parse", "HEAD"], project_dir)
    branch = _run_command(["git", "branch", "--show-current"], project_dir)
    porcelain = _run_command(["git", "status", "--porcelain"], project_dir)
    return {
        "available": True,
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(porcelain),
    }


def capture_package_versions() -> dict[str, str | None]:
    """Capture only the package versions relevant to an InferScope run."""
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def capture_nvidia_state() -> dict[str, Any]:
    """Capture NVIDIA device metadata via nvidia-smi when available."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "devices": [], "reason": "nvidia-smi not found"}
    query = "index,name,uuid,memory.total,driver_version,temperature.gpu,power.limit,compute_cap"
    output = _run_command(
        [
            executable,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return {"available": False, "devices": [], "reason": "nvidia-smi query failed"}
    devices: list[dict[str, str]] = []
    fields = (
        "index",
        "name",
        "uuid",
        "memory_total_mib",
        "driver_version",
        "temperature_c",
        "power_limit_w",
        "compute_capability",
    )
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        devices.append(dict(zip(fields, values, strict=False)))
    return {"available": True, "devices": devices, "reason": None}


def capture_environment(project_dir: Path | None = None) -> dict[str, Any]:
    """Capture a whitelisted environment fingerprint suitable for run manifests."""
    resolved_project = (project_dir or Path.cwd()).resolve()
    virtual_memory = psutil.virtual_memory()
    return {
        "schema_version": "1.0",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "executable": sys.executable,
            "cpu_logical_count": os.cpu_count(),
            "memory_total_bytes": virtual_memory.total,
        },
        "packages": capture_package_versions(),
        "git": capture_git_state(resolved_project),
        "nvidia": capture_nvidia_state(),
    }


def environment_as_json(project_dir: Path | None = None) -> str:
    """Serialize the captured environment using deterministic JSON formatting."""
    return json.dumps(capture_environment(project_dir), indent=2, sort_keys=True)
