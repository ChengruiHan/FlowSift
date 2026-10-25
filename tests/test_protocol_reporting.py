import copy
from pathlib import Path

import pytest
from flowvla_rl.protocol import load_config, validate_config, planned_budget
from flowvla_rl.reporting import profile_summary, task_success

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('method', ['full10', 'late5', 'uniform5'])
def test_method_configs(method):
    assert load_config(ROOT / 'configs' / f'{method}.yaml')['selection']['method'] == method

@pytest.mark.parametrize('section,key,value', [
    ('train','epochs',2), ('train','group_size',8), ('train','eta',0.2),
    ('train','budget_unit','updates'), ('model','denoising_steps',5),
    ('model','subdirectory','grpo-seed11-update300'), ('selection','normalize_by_selected_count',False),
])
def test_protocol_rejects_confounded_comparisons(section, key, value):
    config = load_config(ROOT / 'configs/full10.yaml')
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_no_signal_does_not_launch_formal_budget():
    with pytest.raises(ValueError, match='no mixed'):
        planned_budget(0)


def test_profile_uses_cache_pair_medians_and_separates_backends():
    rows = []
    for backend, scale in [('toy_cpu', 1), ('smolvla_cuda', 100)]:
        for method, cost in [('full10', 10), ('late5', 6)]:
            rows.extend(dict(backend=backend, method=method, cache_id='0', learner_ms=scale*x)
                        for x in [cost, cost+1, cost+100])
    result = profile_summary(rows)
    late = [r for r in result if r['method']=='late5']
    assert all(r['speedup'] == pytest.approx(11/7) for r in late)
    assert {r['median_learner_ms'] for r in late} == {7, 700}


def test_task_success_requires_complete_and_unique_evaluations():
    rows = [dict(task_id=i, episode_index=j, success=j%2) for i in range(10) for j in range(20)]
    assert set(task_success(rows, 20).values()) == {0.5}
    with pytest.raises(ValueError):
        task_success(rows[:-1], 20)
    with pytest.raises(ValueError):
        task_success(rows + [rows[0]], 20)

@pytest.mark.parametrize('section,key,value', [
    ('evaluation', 'test_seed', 1), ('evaluation', 'dev_episodes_per_task', 20),
    ('evaluation', 'checkpoint', 'best'), ('diagnostics', 'score_all_10_steps', False),
])
def test_protocol_rejects_unimplemented_evaluation_settings(section, key, value):
    config = load_config(ROOT / 'configs/full10.yaml')
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def evaluation_fixture(tmp_path):
    import csv
    import json
    runs, evaluations = tmp_path / 'runs', tmp_path / 'evaluations'
    run, ev = runs / 'full10_seed101', evaluations / 'test'
    run.mkdir(parents=True)
    ev.mkdir(parents=True)
    manifest = dict(run_id=run.name, method='full10', train_seed=101,
                    budget=300, config={'run': {'stage': 'formal'}})
    (run / 'run_manifest.json').write_text(json.dumps(manifest))
    (run / 'summary.json').write_text(json.dumps(dict(status='complete', final_policy_hash='hash-A')))
    meta = dict(run_id=run.name, split='test', checkpoint_hash='hash-A',
                sampler='ode', seed=1000, episodes_per_task=20)
    (ev / 'eval_manifest.json').write_text(json.dumps(meta))
    status = dict(status='complete', run_id=run.name, episodes=200,
                  success_rate=0.5, evaluation_seconds=60)
    (ev / 'summary.json').write_text(json.dumps(status))
    rows = [dict(run_id=run.name, split='test', checkpoint_hash='hash-A', task_id=t,
                 episode_index=i, evaluation_seed=1000+i, success=i%2)
            for t in range(10) for i in range(20)]
    with (ev / 'eval_episodes.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return runs, evaluations, meta, status, rows


def test_report_binds_evaluation_to_final_checkpoint(tmp_path):
    import json
    from flowvla_rl.reporting import summarize_runs
    runs, evaluations, *_ = evaluation_fixture(tmp_path)
    assert summarize_runs(runs, evaluations)[0][0]['ode_success_rate'] == 0.5
    path = runs / 'full10_seed101' / 'summary.json'
    summary = json.loads(path.read_text())
    summary['final_policy_hash'] = 'hash-B'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='final policy'):
        summarize_runs(runs, evaluations)


@pytest.mark.parametrize('field,value', [('run_id', 'other'), ('checkpoint_hash', 'hash-B'),
                                        ('split', 'dev'), ('evaluation_seed', 999),
                                        ('episode_index', 20)])
def test_evaluation_rejects_mixed_or_wrong_episode_records(tmp_path, field, value):
    from flowvla_rl.reporting import validate_evaluation
    _, _, meta, status, rows = evaluation_fixture(tmp_path)
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_evaluation(rows, meta, status)


def test_test_matrix_rejects_pilot_even_when_formal_matrix_is_complete(tmp_path):
    import json
    from flowvla_rl.protocol import canonical_hash, validate_test_matrix
    lock = dict(seeds=[101], methods=['full10', 'late5', 'uniform5'], attempted_budget=300)
    for method in lock['methods']:
        path = tmp_path / f'{method}_seed101'
        path.mkdir()
        (path / 'run_manifest.json').write_text(json.dumps(dict(
            train_seed=101, method=method, config={'run': {'stage': 'formal'}},
            protocol_lock_hash=canonical_hash(lock))))
        (path / 'summary.json').write_text(json.dumps(dict(status='complete', attempted_groups=300)))
    validate_test_matrix(tmp_path, lock, 'full10_seed101')
    validate_test_matrix(tmp_path, lock, 'sft')
    with pytest.raises(ValueError, match='member'):
        validate_test_matrix(tmp_path, lock, 'full10_pilot')
