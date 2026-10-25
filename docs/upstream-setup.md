# Environment setup, with the pitfalls

> Historical FlowVLA-RL setup notes, retained as an upstream reference. Hardware and findings here are not FlowSift measurements. Use the [FlowSift training guide](server_training.md) for this project.

> First set up on: WSL2 Ubuntu / RTX 5060 Ti 16GB (sm_120 Blackwell) / 20 vCPU / 32GB host RAM.
> Purpose of this file: replay the setup on a rented machine without re-debugging.

## 1. Pinned versions

| component | version / hash |
|---|---|
| LeRobot | tag v0.6.0 (2026-07-06 release, not HEAD) |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128, `arch_list` includes `sm_120` |
| torchvision | 0.26.0+cu128 |
| hf-libero | 0.1.4 |
| mujoco | 3.8.1 |
| robosuite | 1.4.0 |
| gymnasium | 1.3.0 |
| transformers | 5.5.4 |
| smolvla_base | snapshot `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |

## 2. Reproduction commands

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv                     # 3.12, not 3.11 — see pitfall A
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/huggingface/lerobot.git && (cd lerobot && git checkout v0.6.0)
uv pip install -e "./lerobot[libero,smolvla]"
```

Before running any LIBERO code: `export MUJOCO_GL=egl`

## 3. Pitfalls, in the order they were hit

### Pitfall A — Python 3.11 cannot install LeRobot

The first attempt used `uv venv --python 3.11`, but LeRobot v0.6.0 declares
`requires-python = ">=3.12"`. 3.12+ is required (only v0.4.4 and earlier accept >=3.10).

### Pitfall B — install torch first, then lerobot

The other way round, lerobot pulls the default torch wheel from PyPI, which may not include
`sm_120`; the first `.cuda()` on Blackwell then fails with `no kernel image`. Verify after
installing:

```python
torch.cuda.get_arch_list()   # must include 'sm_120'
```

### Pitfall C — LIBERO's first import blocks on an interactive prompt

When `~/.libero/config.yaml` does not exist, `libero/libero/__init__.py` calls
`input("Do you want to specify a custom path...")`, which raises `EOFError` in non-interactive
environments. Write the config up front (pointing datasets at `~/data`, so rebuilding the venv
does not lose data):

```bash
PKG=$(python -c "import libero.libero as m,os;print(os.path.dirname(m.__file__))")
mkdir -p ~/.libero ~/data/datasets/libero
cat > ~/.libero/config.yaml <<EOF
benchmark_root: $PKG
bddl_files: $PKG/bddl_files
init_states: $PKG/init_files
datasets: $HOME/data/datasets/libero
assets: $PKG/assets
EOF
```

`assets` auto-downloads from HF to `~/.cache/libero/assets` on first run; nothing to prepare.

### Pitfall D — LeRobot's `truncated` is always False (the important one)

`lerobot/envs/libero.py:377` hardcodes `truncated = False`, so the env never truncates on its
own, even though it computes `self._max_episode_steps` (libero_object=280). The official
evaluator reads that attribute itself, so evaluation is capped; what breaks is any custom rollout
loop that trusts the env to stop.

Consequences: episodes only terminate at robosuite's internal horizon (~1000 steps), about 3.5×
longer than the suite specifies. First, 3.5× wasted wall-clock. Second, inflated success rates —
an extra ~720 steps means extra chances to succeed, incomparable with the paper protocol. Third,
stepping all the way to the end raises `ValueError: executing action in terminated episode`.

Bottom line: the caller must enforce the cap. The evaluation protocol and the rollout harness
both use `TASK_SUITE_MAX_STEPS` explicitly (spatial/object=280, goal=300, libero_10=520).

### Pitfall E — EGL destructor noise (ignorable)

On process exit, `MjRenderContext.__del__` / `EGLGLContext.__del__` throw
`EGLError: <exception str() failed>`. Teardown noise, no effect on results. Write results to
disk before exiting; note that `os._exit()` does not flush the stdout buffer — either
`sys.stdout.flush()` first or write to a file directly.

### Pitfall F — three WSL configurations

1. `sshd` does not autostart (`systemctl is-enabled ssh` = disabled); after `wsl --shutdown` the
   machine is unreachable remotely. `sudo systemctl enable ssh`.
2. `.wslconfig`'s `memory=` only applies when the utility VM starts; restarting the distro is
   not enough, a full `wsl --shutdown` is required.
3. The PATH export for `/usr/lib/wsl/lib` (which contains `nvidia-smi`) lives in `~/.bashrc`,
   and Ubuntu's `.bashrc` returns early for non-interactive shells — `ssh host cmd` does not see
   it. Scripts that need `nvidia-smi` must `export PATH=/usr/lib/wsl/lib:$PATH` themselves.
   Training is unaffected: `libcuda.so.1` resolves via ldconfig.

## 4. Measured baseline numbers (the basis for RL budgeting)

| metric | measured |
|---|---|
| LIBERO single-env step rate | 17.3 steps/s |
| one 280-step episode, wall-clock | 16.2 s |
| observations | 2 cameras, `(256,256,3) uint8` (agentview + eye-in-hand) |
| action space | `Box(-1, 1, (7,))` |
| MuJoCo EGL headless rendering | works |
| SmolVLA bf16 weights VRAM | 877 MiB (peak 884 MiB) |
| SmolVLA load time | 98.5 s |
