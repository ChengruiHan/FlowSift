#!/usr/bin/env python
"""Durable single-seed execution of the v3 protocol; stops on any invalid stage."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from flowvla_rl.checkpointing import atomic_json
from flowvla_rl.step_selection import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/single_seed'))
    parser.add_argument('--sft', type=Path, required=True)
    parser.add_argument('--stage-a-budget', type=int, default=20)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sft = str(args.sft.resolve())
    runs = root / 'runs'
    evals = root / 'evaluations'
    caches = root / 'profile_caches'
    diagnostic = root / 'fixed_diagnostic_chunks.pt'
    gate = root / 'gpu_correctness.json'
    lock = root / 'protocol.lock.json'
    logs = root / 'stage_logs'
    logs.mkdir(exist_ok=True)
    started = time.time()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')

    def run(name, script, arguments):
        command = [sys.executable, str(Path(__file__).with_name(script)), *map(str, arguments)]
        atomic_json(root / 'progress.json', {'status': 'running', 'stage': name,
                    'started_unix': started, 'stage_started_unix': time.time(), 'command': command})
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {name}', flush=True)
        with (logs / f'{name}-{int(time.time())}.log').open('w') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)

    def complete(directory):
        path = directory / 'summary.json'
        return path.exists() and json.loads(path.read_text()).get('status') == 'complete'

    def evaluate(name, checkpoint, run_id, split):
        destination = evals / name
        if complete(destination):
            return
        arguments = ['--checkpoint', checkpoint, '--run-id', run_id, '--split', split, '--output', destination]
        if split == 'test':
            arguments += ['--protocol-lock', lock, '--matrix-runs', runs]
        run(name, 'evaluate_libero.py', arguments)

    def final_model(directory):
        summary = json.loads((directory / 'summary.json').read_text())
        return str(Path(summary['final_checkpoint']) / 'pretrained_model')

    try:
        for label in ('replay_a', 'replay_b'):
            evaluate(label, sft, 'sft', 'replay')
        replay_rows = []
        for label in ('replay_a', 'replay_b'):
            with (evals / label / 'eval_episodes.csv').open() as handle:
                replay_rows.append(list(csv.DictReader(handle)))
        keys = ('task_id', 'evaluation_seed', 'initial_observation_digest_if_available', 'success', 'env_steps')
        differences = [{key: [a[key], b[key]] for key in keys if a[key] != b[key]}
                       for a, b in zip(*replay_rows, strict=True)]
        atomic_json(root / 'replay_comparison.json', {'conditions': len(differences),
                    'exact_matches': sum(not d for d in differences), 'differences': differences,
                    'paired_bootstrap_enabled': False})
        stage_a = root / 'stage_a'
        if not complete(stage_a):
            arguments = ['--config', 'configs/full10.yaml', '--stage', 'smoke', '--seed', 17,
                         '--budget', args.stage_a_budget, '--sft', sft, '--run-dir', stage_a,
                         '--export-profile-groups', caches]
            latest = stage_a / 'checkpoints/latest.json'
            if latest.exists():
                arguments += ['--resume', latest]
            run('stage_a', 'train_grpo_libero.py', arguments)
        bundles = sorted(caches.glob('cache_*.pt'))
        if len(bundles) != 3:
            raise RuntimeError(f'Stage A produced {len(bundles)}/3 mixed task caches; investigate before continuing')
        if not diagnostic.exists():
            run('diagnostic_cache', 'make_diagnostic_cache.py', ['--caches', caches, '--output', diagnostic])
        if not gate.exists() or json.loads(gate.read_text())['status'] != 'passed':
            run('gpu_correctness', 'verify_cache.py', ['--sft', sft, '--cache', bundles[0], '--output', gate])
        profile = root / 'gpu_profile'
        if not (profile / 'learner_profile.csv').exists():
            run('profile', 'profile_learner.py', ['--caches', caches, '--sft', sft,
                                               '--output', profile, '--gradient-probe'])
        common = ['--sft', sft, '--diagnostic-cache', diagnostic, '--correctness-report', gate, '--run-root', runs]
        run('pilot_matrix', 'run_matrix.py', ['--stage', 'pilot', *common])
        evaluate('dev_sft', sft, 'sft', 'dev')
        for method in METHODS:
            run_id = f'pilot_{method}_seed17'
            evaluate(f'dev_{run_id}', final_model(runs / run_id), run_id, 'dev')
        if not lock.exists():
            run('lock_single_seed', 'lock_protocol.py', ['--full-pilot', runs / 'pilot_full10_seed17',
                                                       '--seeds', '101', '--output', lock])
        run('formal_matrix', 'run_matrix.py', ['--stage', 'formal', '--protocol-lock', lock, *common])
        evaluate('sft', sft, 'sft', 'test')
        for method in METHODS:
            run_id = f'formal_{method}_seed101'
            evaluate(run_id, final_model(runs / run_id), run_id, 'test')
        run('report', 'summarize_results.py', ['--runs', runs, '--evaluations', evals,
                    '--profile', profile / 'learner_profile.csv', '--output', root / 'report'])
        atomic_json(root / 'progress.json', {'status': 'complete', 'started_unix': started,
                    'finished_unix': time.time(), 'seeds': [101], 'report': str(root / 'report')})
    except BaseException as exc:
        path = root / 'progress.json'
        progress = json.loads(path.read_text()) if path.exists() else {}
        atomic_json(path, {**progress, 'status': 'failed', 'error': repr(exc), 'failed_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
