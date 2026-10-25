"""Small, dependency-light experiment ledger used by every server run."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _git(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *command], cwd=cwd, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


class RunLedger:
    """Create a self-describing run directory and maintain its status."""

    def __init__(
        self,
        root: str | Path,
        name: str,
        config: Any,
        *,
        project_dir: str | Path | None = None,
        command: str | None = None,
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(root).expanduser().resolve() / f"{timestamp}_{name}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.project_dir = Path(project_dir or Path.cwd()).resolve()
        config_data = asdict(config) if is_dataclass(config) else config
        (self.path / "config.yaml").write_text(
            yaml.safe_dump(config_data, sort_keys=True, allow_unicode=True)
        )
        (self.path / "command.txt").write_text((command or " ".join(os.sys.argv)) + "\n")
        metadata = {
            "created_at": datetime.now(UTC).isoformat(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": _git(["rev-parse", "HEAD"], self.project_dir),
            "git_status": _git(["status", "--short"], self.project_dir),
        }
        _atomic_json(self.path / "metadata.json", metadata)
        (self.path / "checkpoints").mkdir()
        self._metric_fields: list[str] | None = None
        self.set_status("running")

    def set_status(self, state: str, **details: Any) -> None:
        payload = {
            "state": state,
            "updated_at": datetime.now(UTC).isoformat(),
            **details,
        }
        _atomic_json(self.path / "status.json", payload)

    def log_metrics(self, metrics: dict[str, int | float | str]) -> None:
        """Append one stable-schema row to ``metrics.csv`` and flush it."""
        fields = list(metrics)
        if self._metric_fields is None:
            self._metric_fields = fields
        elif fields != self._metric_fields:
            raise ValueError(
                f"metric schema changed: expected {self._metric_fields}, got {fields}"
            )
        path = self.path / "metrics.csv"
        write_header = not path.exists()
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._metric_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
            stream.flush()
            os.fsync(stream.fileno())
