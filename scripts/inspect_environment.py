#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from flowvla_rl.checkpointing import atomic_json
from flowvla_rl.provenance import environment_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/environment.json")
    args = parser.parse_args()
    report = environment_report(Path(__file__).resolve().parents[1])
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
