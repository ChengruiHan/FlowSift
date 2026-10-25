"""Run-ledger tests: every experiment must remain auditable after a crash."""

import csv
import json
from pathlib import Path

import yaml

from flowvla_rl.run_logging import RunLedger


def test_run_ledger_writes_required_artifacts(tmp_path: Path) -> None:
    ledger = RunLedger(
        tmp_path,
        "unit",
        {"seed": 7, "eta": 0.4},
        project_dir=Path(__file__).parents[1],
        command="python train.py --seed 7",
    )
    ledger.log_metrics({"step": 0, "loss": 1.5})
    ledger.log_metrics({"step": 1, "loss": 1.0})
    ledger.set_status("complete", checkpoint="checkpoints/final")

    assert yaml.safe_load((ledger.path / "config.yaml").read_text())["seed"] == 7
    assert (ledger.path / "command.txt").read_text().strip().endswith("--seed 7")
    assert json.loads((ledger.path / "metadata.json").read_text())["git_commit"]
    assert json.loads((ledger.path / "status.json").read_text())["state"] == "complete"
    with (ledger.path / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["step"] for row in rows] == ["0", "1"]
    assert (ledger.path / "checkpoints").is_dir()
