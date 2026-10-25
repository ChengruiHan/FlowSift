# Contributing

Please keep correctness changes separate from changes to the experiment protocol. Describe the trigger, expected behavior, evidence and effect on existing results in an issue or pull request.

Before submitting code:

```bash
python -m pytest -q
python scripts/smoke_cpu.py --output results/contribution-smoke
python scripts/verify_published_results.py
```

Add focused regression tests for behavior changes. CPU tests do not establish CUDA numerical equivalence or simulator reproducibility. Changes to source invalidate existing GPU correctness gates; regenerate the gate and protocol lock before starting a new formal experiment.

Treat `docs/experiments/seed101` as historical evidence. Do not overwrite it with a new run or edit measurements to match current code. Put new experiments under a distinct identifier, preserve per-seed results and disclose failures. Never commit weights, optimizer snapshots, credentials, private server addresses or unfiltered deployment logs.

Retain upstream license notices and record borrowed implementation changes in [UPSTREAM.md](UPSTREAM.md).
