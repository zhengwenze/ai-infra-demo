"""Safe, append-oriented experiment artifact persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from inferscope.errors import ConfigurationError

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _json_default(value: Any) -> Any:
    """Serialize common model/path objects without silently stringifying everything."""
    if isinstance(value, Path):
        return str(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=False)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class ArtifactStore:
    """Create immutable run directories and atomically write experiment artifacts."""

    def __init__(self, results_root: Path, run_id: str) -> None:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ConfigurationError(
                "run_id must contain only letters, numbers, dots, underscores, and dashes"
            )
        self.results_root = results_root.expanduser().resolve()
        self.run_id = run_id
        self.run_dir = (self.results_root / "raw" / run_id).resolve()
        expected_parent = (self.results_root / "raw").resolve()
        if self.run_dir.parent != expected_parent:
            raise ConfigurationError("run directory escaped the configured results root")

    def create(self) -> Path:
        """Create a new run directory, refusing to reuse an existing one."""
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(mode=0o750, exist_ok=False)
        (self.run_dir / "logs").mkdir(mode=0o750)
        return self.run_dir

    def path(self, relative_name: str) -> Path:
        """Resolve a single-file artifact name within the run directory."""
        if not relative_name or Path(relative_name).name != relative_name:
            raise ConfigurationError("artifact name must be a plain file name")
        resolved = (self.run_dir / relative_name).resolve()
        if resolved.parent != self.run_dir:
            raise ConfigurationError("artifact path escaped the run directory")
        return resolved

    def write_json(self, relative_name: str, value: Any) -> Path:
        """Atomically write deterministic UTF-8 JSON."""
        content = json.dumps(
            value,
            default=_json_default,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return self._atomic_write(self.path(relative_name), f"{content}\n")

    def write_yaml(self, relative_name: str, value: Mapping[str, Any]) -> Path:
        """Atomically write a resolved YAML configuration."""
        content = yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=True)
        return self._atomic_write(self.path(relative_name), content)

    def append_jsonl(self, relative_name: str, values: Iterable[Any]) -> Path:
        """Append JSONL records and fsync before returning."""
        target = self.path(relative_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for value in values:
                line = json.dumps(
                    value,
                    default=_json_default,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                handle.write(f"{line}\n")
            handle.flush()
            os.fsync(handle.fileno())
        return target

    @staticmethod
    def _atomic_write(target: Path, content: str) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target
