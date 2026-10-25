#!/usr/bin/env python
"""Check the public evidence hashes, historical source snapshot and result tables."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

from flowvla_rl.reporting import summarize_runs


def main():
    root = Path(__file__).resolve().parents[1] / 'docs/experiments/seed101'
    provenance = json.loads((root / 'provenance.json').read_text())
    for entry in provenance['files']:
        actual = hashlib.sha256((root / entry['path']).read_bytes()).hexdigest()
        if actual != entry['published_sha256']:
            raise ValueError(f"Evidence hash mismatch: {entry['path']}")
    archive = root / provenance['training_source_archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != provenance['training_source_archive_sha256']:
        raise ValueError('Historical source archive hash mismatch')
    with zipfile.ZipFile(archive) as z:
        hashes = {name: hashlib.sha256(z.read(name)).hexdigest() for name in sorted(z.namelist())
                  if (name.startswith('src/') and name.endswith('.py'))
                  or (name.startswith('scripts/') and name.count('/') == 1 and name.endswith('.py'))
                  or name == 'pyproject.toml'}
    source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if source_hash != provenance['original_source_hash']:
        raise ValueError('Historical source does not match the experiment source hash')
    online, costs, _ = summarize_runs(root / 'runs', root / 'evaluations')
    with (root / 'report/online_per_seed.csv').open() as f:
        original = {row['run_id']: row for row in csv.DictReader(f)}
    for row in online:
        actual, expected = row.get('ode_success_rate'), original[row['run_id']]['ode_success_rate']
        if (actual is None) != (expected == '') or (actual is not None and abs(actual-float(expected)) > 1e-12):
            raise ValueError(f"Success-rate mismatch: {row['run_id']}")
    with (root / 'report/system_cost.csv').open() as f:
        original_costs = {row['run_id']: row for row in csv.DictReader(f)}
    for row in costs:
        for key in ('rollout_seconds', 'learner_seconds', 'diagnostic_seconds', 'checkpoint_seconds', 'train_elapsed_seconds'):
            if abs(row[key] - float(original_costs[row['run_id']][key])) > 1e-6:
                raise ValueError(f"Cost mismatch: {row['run_id']} {key}")
    print(f"Verified {len(provenance['files'])} evidence files, historical source, success rates and training costs.")


if __name__ == '__main__':
    main()
