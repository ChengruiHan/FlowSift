# FlowSift: selective transition rescoring for SmolVLA reinforcement learning

**Research protocol v3.0 — originally planned on 2026-10-22.**

This English edition organizes the original plan around the research question, objective, controls and acceptance criteria. It is a protocol, not a claim that every proposed check was completed. The [README](README.md), [results](docs/results_and_limitations.md) and [correctness report](docs/correctness_report.md) describe actual execution and evidence. The recorded formal study uses the protocol's single-seed option.

## 1. Research question and scope

**With the complete action-generation procedure retained, how much learner computation can be removed by rescoring fewer stored transitions, and what task-performance trade-off follows?**

The study separates generation work from update work. It does not shorten inference, redesign the model, retrain SFT, or introduce a new reward. Its deliverables are a selective-rescoring implementation, correctness tests, fixed-cache benchmarks, a three-method online comparison and an evidence-bounded conclusion.

| Decision | Protocol |
|---|---|
| Base | Pinned FlowVLA-RL and released SmolVLA SFT |
| Tasks | All ten LIBERO-Object tasks; one multitask policy per run |
| Required methods | Full-10, Late-5 and Uniform-5 |
| Training generation | Original ten-step SDE, η=0.4 |
| Evaluation generation | Original ten-step ODE, at most 280 control steps |
| Reward / advantage | Binary success; group size four; center within group, no standard-deviation normalization |
| Experimental intervention | Learner transition indices for both PG and reference regularization |
| Main budget | Attempted groups; also report episodes, environment steps and effective updates |
| Default formal replication | Two common training seeds; three may be selected before test evaluation |
| Minimum formal study | All three methods with one common seed, explicitly labeled exploratory |
| Formal budget | Same B for every run, B in {300, 400, 500, 600} |

## 2. Hypotheses and contribution boundaries

Full-10 uses all stored transitions. Uniform-5 uniformly samples five without replacement per chunk, preserving the full-grid loss and gradient **in expectation for fixed cached data and parameters**. Late-5 selects a fixed low-noise window and deliberately changes timestep weighting. Both PG and k3 are subsampled, so a Late-5 effect cannot be attributed solely to PG.

All timesteps share parameters. Updating selected losses can change the entire denoising field; unselected timesteps are not frozen. All-ten-step drift diagnostics therefore remain relevant.

Fewer calls do not imply a proportional total speedup. A useful approximation is:

$$T_{learner}(m)\approx T_{fixed}+mT_{selected}.$$

Fixed work includes transfer, prefix handling and optimizer overhead. Online training also includes simulation, full SDE rollout, diagnostics, evaluation and storage. For a fixed workload with learner fraction f and learner speedup s, the ideal total speedup is:

$$\frac{1}{(1-f)+f/s}.$$

Online policies also change episode length and mixed-group frequency. Measure loop time separately rather than inferring it from call counts.

The project's work is the SmolVLA selective-rescoring implementation, budget/recovery semantics, probability/gradient regression checks and controlled cost–performance experiment. SDE retrofit, GRPO, uniform timestep subsampling, SmolVLA, LIBERO, and inherited precision/scene-restoration fixes are not original contributions. See [UPSTREAM.md](UPSTREAM.md).

The original plan identified prior work including [DDPO timestep subsampling](https://github.com/kvablack/ddpo-pytorch/blob/1958463f020112c9a7bc85768d296daacc2e1b4b/scripts/train.py#L460), [Flow-GRPO](https://arxiv.org/abs/2505.05470), [MixGRPO](https://arxiv.org/abs/2507.21802) and [DPPO](https://github.com/irom-princeton/dppo). FlowSift's comparison retains the full ten-step SDE path and a single shared current policy; it does not evaluate shortened sampling grids or timestep-dependent routing between separate base and fine-tuned actors.

## 3. Source, model and runtime identity

| Asset | Pin |
|---|---|
| FlowVLA-RL | `50e73afc07212f48bed370ea4c88a3131d6f914a` |
| SFT repository | `MorpheusTzz/smolvla-grpo-libero-object` |
| SFT revision | `a120aa9b51f8e55db5ad8284e051963c61a2f5ed` |
| SFT subdirectory | `sft-100pct-baseline` |
| LeRobot | v0.6.0; record the resolved commit |
| Model generation settings | 10 denoising steps; chunk size 50; 50 executed actions per full chunk |
| Action dimensions | 7 real dimensions; padded model dimension 32 |

Load the checkpoint together with its preprocessors/postprocessors. Never initialize from the already GRPO-trained subdirectory. Online RL does not require rerunning SFT or downloading the full demonstration dataset.

Before formal training, record model/preprocessor SHA256 hashes, FlowSift commit/source hash, LeRobot commit, Python/PyTorch/CUDA/driver/MuJoCo/robosuite/hf-libero versions, GPU and memory, actual CPU/RAM quota and storage capacity. Reuse a working CUDA stack consistently across methods and record it. The original hardware target was a 32 GB GPU; the completed study used a 24 GB RTX 4090 D. A complete learner update must establish actual feasibility.

Retain initial SFT provenance, final checkpoints, the latest recoverable checkpoint, representative caches and logs. Rotate temporary checkpoints only after a new complete checkpoint commits. Video collection, if performed, is limited to final success/failure examples rather than continuous training capture.

## 4. Optimization objective

### 4.1 Original SDE transitions

Generation runs from noise time τ=1 to action time τ=0. With K=10 and h=0.1:

$$\tau_k=1-kh,\qquad k=0,\ldots,9.$$

For stored state z_k and conditioning c, the expert predicts velocity v_θ. The inherited transition is:

$$s_\theta=-\frac{z_k+(1-\tau_k)v_\theta}{\tau_k},\qquad g_k=\eta\sqrt{\tau_k},$$

$$q_\theta(z_{k+1}\mid z_k,c)=\mathcal N(\mu_{\theta,k},\sigma_k^2I),$$

$$\mu_{\theta,k}=z_k-h\left(v_\theta-\tfrac12g_k^2s_\theta\right),\qquad \sigma_k^2=g_k^2h.$$

Both endpoints come from the detached old-policy cache. Recompute only the current velocity contribution. Keep the original time grid and η; do not regenerate paths or renumber the last five transitions as a five-step integration grid. Finite discretization, model error and RL updates do not guarantee equal SDE/ODE task performance; measure final ODE performance directly.

### 4.2 Masked transition log-probability

$$\ell_{i,c,k}(\theta)=\sum_{j,d}M_{i,c,j,d}\log q_\theta(z_{k+1,j,d}\mid z_k,c).$$

The mask removes padded dimensions and action positions not executed because of success, termination or the horizon cap. Old/current/reference use the same mask, state pair and index. This inherited masked surrogate is not the exact marginal likelihood of the final action chunk; no claim of an unbiased environment-return gradient is made.

### 4.3 Group advantage and transition loss

For four episodes sharing a task and initial scene, with R_i in {0,1}:

$$A_i=R_i-\frac1G\sum_r R_r.$$

Each chunk in an episode receives the same advantage (`chunk_discount=1`). The implemented ratio and combined loss are:

$$\rho=\exp(\operatorname{clamp}(\ell_\theta-\ell_{old},-20,20)),$$

$$P=-\min\left(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A\right),$$

$$d=\ell_{ref}-\ell_\theta,\qquad D=e^d-d-1,\qquad F=P+\beta D,$$

with ε=0.2 and β=0.01. Record raw log-ratio differences as well as clamp hits. A nonfinite k3 exponential is a numerical failure; do not introduce method-specific clamps to keep only one method running.

### 4.4 Episode/chunk/step weighting

For N episodes, C_i chunks in episode i, and selected set S_ic:

$$L=\frac1N\sum_{i=1}^{N}\frac1{C_i}\sum_{c=1}^{C_i}\frac1{|S_{ic}|}\sum_{k\in S_{ic}}F_{i,c,k}.$$

| Method | S | Original times | Current / reference calls per chunk |
|---|---|---|---:|
| Full-10 | 0–9 | 1.0, 0.9, …, 0.1 | 10 / 10 |
| Late-5 | 5–9 | 0.5, 0.4, …, 0.1 | 5 / 5 |
| Uniform-5 | Five independent per-chunk draws without replacement | Corresponding original times; indices sorted for execution | 5 / 5 |

Use one epoch. Accumulate all chunk gradients with episode-balanced weights, then clip and perform one optimizer step. Before updating, current should match the sampling policy, so initial ratios are near one and clipping is normally inactive. The centered group's initial PG scalar can be near zero while its gradient is nonzero. When current initially equals reference, k3 is also zero. Assess learning through gradients, parameter changes and task behavior, not a nonzero initial scalar loss.

## 5. Implementation contract

`recompute_log_probs(..., step_indices=None)` treats `None` as the full grid and otherwise accepts nonempty, unique integer indices in `[0,K)`. Each selected k reads `states[:,k]` and `states[:,k+1]` at time `1-k/K`. It returns scores in the supplied index order. The first implementation processes one chunk at a time; a future shared index vector across larger batches must not be described as independent per-sample selection.

Uniform selection derives a local CPU generator from a stable SHA256 key containing selection seed, attempted-group ID, episode ID and chunk ID. The root seed is derived from `SHA256("FlowSift-v3-selection:" + train_seed)`. Record the derivation version. Do not use Python's process-dependent `hash()`, wall time, run order or completed-update count. Selection, diagnostics and logging must not advance rollout/reset RNG.

Freeze VLM and state projection; train the action expert and applicable action/time projections. Record declared trainable names and actual gradient names. Reference parameters are frozen at initial SFT. Current and reference can share the frozen prefix. Collection and rescoring use the same autocast/mode convention; CUDA forward uses the inherited BF16 path and density calculations use FP32.

One update follows: collect a complete reset-matched group → compute rewards/advantages → identify mixed/collapsed → select indices per chunk → rescore reference/current and slice old consistently → validate probabilities → accumulate weighted losses → clip gradients → optimizer step → record counters/timing. A collapsed group skips the learner.

## 6. Budgets and recovery

Count **attempted groups**, not successful updates. Every complete group contributes four episodes, including all-success/all-failure groups. Record actual environment steps and completed updates. A partial group must not consume a completed-group budget unit.

Stop at the locked attempted budget and save final even with zero updates or a collapsed last group. Save recoverable checkpoints at complete-group boundaries every 50 attempts and at final. Payloads include current parameters, optimizer, counters, RNG and immutable provenance. Write atomically before advancing `latest` and rotating old ordinary checkpoints.

On resume, construct reference from the original SFT **before** restoring current, optimizer and RNG. Reject incompatible method, source, model, reference, protocol or RNG metadata. Keep work beyond the restored boundary in a separate replay/resource ledger. Environment reconstruction does not by itself provide bitwise MuJoCo trajectory recovery.

## 7. Correctness gates

| ID | Required check |
|---|---|
| C1 | Upstream Full, full-index and omitted-index paths agree on cached log-probabilities, loss, gradients and one update. |
| C2 | Subset scores equal full slices with original times and masks. |
| C3 | Instrumented current/reference denoiser calls are 10/10 or 5/5 per chunk. |
| C4 | Legal sampling without replacement; normalization does not shrink a constant objective. |
| C5 | On a small model, the average loss and per-parameter gradient over all 252 subsets equal Full. |
| C6 | Fixed-density masks exclude invalid elements; cached states/reference receive no gradients. |
| C7 | Old/current probabilities are finite and agree before updating; gate fixtures cover all ten steps. |
| C8 | Actual gradient scope matches declaration; initial clipped/unclipped PG gradients agree. |
| C9 | Selection and diagnostics preserve rollout RNG and reproduce indices across restart. |
| C10 | Mixed/collapsed/partial/already-completed budget boundaries terminate and save correctly. |
| C11 | Two updates agree with a save/restore boundary between them; reference hash is unchanged. |

CPU starting tolerances are `atol=1e-6, rtol=1e-5`; report maximum absolute and relative L2 errors. Exact CPU checkpoint-continuation comparisons use zero tolerance. Establish real-model BF16 repeatability before uniformly locking numerical tolerances. Do not relax tolerance independently for a method.

Every online selected transition must satisfy `abs(current_logp-old_logp) <= 0.01` and be finite before the update. Failure requires investigating precision, mode, cache or batch differences. Mask tests concern density aggregation, not invariance of a coupled neural network to changes in masked input coordinates.

Distinguish shared engineering fixes from selection effects using upstream → common Full → indexed Full regression checks. Failed correctness gates support a failure audit, not a valid algorithm comparison. Actual coverage is documented in the [correctness report](docs/correctness_report.md).

## 8. Experimental sequence and budget lock

| Stage | Workload | Purpose |
|---|---|---|
| A | Environment/SFT replay, cache collection and correctness checks | Establish the numerical and data path |
| B | Three fixed mixed-group caches, three methods, repeated updates | Measure learner work independently of online outcomes |
| C | Three pilots; seed 17; 100 attempted groups each | Check stability, learning signal and cost |
| D | All three methods at common seeds and budget | Formal online comparison |
| E | Initial SFT and every formal final; 200 ODE episodes each | Task/cost tables and conclusions |

Stage-A rollout costs are separate from all formal method budgets. Formal runs restart from the same SFT; they do not inherit pilot parameters or optimizer state.

Before formal test evaluation, estimate the Full pilot's mixed-group fraction p̂ = mixed/100 and select:

$$B=\min\left(600,\max\left(300,100\left\lceil\frac{200}{100\max(\hat p,0.1)}\right\rceil\right)\right).$$

Examples: p̂=0.8 gives 300; 0.6 gives 400; 0.4 gives 500; 0.2 reaches the 600 cap. The target of roughly 200 updates is a planning target, not a guarantee. Every method still stops at the same B. A Full pilot with zero mixed groups requires an audit before formal expansion.

Create an immutable lock before Stage D. The default is seeds `[101,202]`; the minimum formal study is `[101]` with all three methods. Optional seed 303 must be selected for every method before formal test results are opened. If resources require B=300 instead of the formula, record the reason uniformly. If even three 300-group runs plus final evaluation are infeasible, report a pilot study rather than a complete formal comparison.

Complete all three methods for seed 101 before proceeding to another seed. Open final tests only after the locked matrix is complete. Document any forced reduction, unfinished runs and protocol deviations; do not selectively add seeds after inspecting outcomes.

## 9. Stop and failure rules

- Numerical failures, old/current mismatch or a failed correctness check stop the affected run. Preserve diagnostics; repair the common cause before a consistent rerun.
- At most one identical-condition replay is used to diagnose the same failure. Do not repeatedly restart until a favorable trajectory appears.
- Reproducible method instability under the fixed shared settings is a result; retain it rather than tuning that method alone or assigning failure a zero success rate.
- If Full has valid gradients/updates but no observed SFT gain, complete the locked budget and report that baseline improvement was not established. Do not extend training until the result becomes positive.
- Lower success or no speedup is a valid negative result. Finish the planned comparison when feasible; do not add branches after seeing the answer.
- Changes to learning rate, η, epochs, KL weight or the data distribution define a new protocol; do not merge those results into the original main table.

## 10. Training and evaluation schedule

Training rotates across all ten tasks. Within each group, task and restored initial scene must match. Stable group/episode/chunk identifiers must survive a restart. Record actual groups, episodes and environment steps; an equal episode budget is not strict environment-step sample efficiency.

| Setting | Development | Final test |
|---|---|---|
| Purpose | Pilot health and budget interpretation | Evaluate the locked formal matrix |
| Tasks | All 10 Object tasks | All 10 Object tasks |
| Episodes per task | 10 | 20 |
| Base evaluation seed | 2000 | 1000 |
| Sampler | 10-step ODE | 10-step ODE |
| Horizon / batch | 280 / 1 | 280 / 1 |
| Checkpoints | Initial SFT and pilot finals | Initial SFT and formal finals |

Seeds are reset consistently per episode. Record task, episode index, seed, available initial-state ID/digest, success, environment steps and checkpoint hash. Use final checkpoints rather than selecting the best checkpoint from final test outcomes. Verify repeated SFT conditions before claiming strict scenario pairing. The default is same-distribution evaluation, not a disjoint initial-state generalization study.

## 11. Fixed-cache profiling and diagnostics

Use at least three mixed-group caches from different tasks with full trajectories, current weights and warmed optimizer states. Restore the same snapshot for every method/repetition. Run at least one untimed warmup, then restore again.

Measure 3 caches × 3 methods × 3 repetitions = 27 complete updates, rotating method order. Report each cache's repetition median and paired timing ratios. Independent method processes avoid CUDA-reserved-memory contamination. If noise produces a within-cache timing-ratio spread above 10%, the original plan allows increasing to five repetitions per cache, then requires reporting persistent variability rather than benchmarking indefinitely.

Total learner time includes transfer, prefix, reference/current scoring, loss/backward, clipping and optimizer work; exclude diagnostics/checkpoints. Synchronize CUDA only at total timing boundaries. A separate pass on 16 fixed chunks records component timings without presenting synchronization-distorted measurements as natural throughput.

Report current/reference calls, allocated/reserved peaks, initialization peak and retained cache size. All methods retain the same full trajectory. Three timing repetitions do not establish strong statistical evidence.

For a limited real-model gradient probe, use three representative chunks with one Full gradient, one Late gradient and eight Uniform gradients each. Record norms and cosine similarity to Full; near-zero Full gradients make cosine interpretation unreliable. This probe observes direction/variance, not an exhaustive proof of real-model unbiasedness.

Use eight common fixed diagnostic chunks, separate from final test evaluation, to rescore all ten transitions at attempt 0, every 50 attempts and final. Record per-step log-probability drift, ratio quantiles, k3 and nonfinite values. These stored old scores no longer correspond to the updated current model, so do not apply the pre-update near-one-ratio health gate to drift diagnostics. Conditional Gaussian KL is optional and does not replace the training k3 term or represent final-action marginal KL.

## 12. Reporting and interpretation

Report three distinct outcomes:

1. **Task performance:** ten-task mean ODE success, per-task values, differences from same-seed Full and initial SFT in percentage points.
2. **Learner cost:** paired fixed-cache update time, calls and process memory.
3. **System cost:** loop wall time, episodes, environment steps, effective updates, collapse rate, evaluation and failed/replayed work; distinguish cloud allocation time.

Stability logs include group rewards/mixed status, maximum pre-update log-probability mismatch, raw clamp hits, maximum k3 d, NaN/Inf, pre-clip gradient norm and clipping status. Selection indices must be reconstructible from stable IDs/version. Save failure tensors/caches when needed without overwriting prior evidence.

Descriptive engineering targets, not statistical tests:

| Quantity | Original target / interpretation |
|---|---|
| Five-step denoiser calls | 50% fewer current and reference rescoring calls |
| Learner time | ≥20% median reduction meets the target; 0–20% is limited benefit; ≤0% is no speedup |
| Success-rate estimate | A drop no larger than 3 pp lies within the descriptive tolerance band; it is not statistical non-inferiority |
| Across-seed behavior | Report every completed seed, not only an average hiding a large regression |
| Complete loop | Report measured percentage change without presupposing acceleration |

One seed supports single-seed observations. Two seeds permit individual values/ranges and a descriptive mean, not a strong seed-level interval. Three seeds still provide limited replication. Evaluation episodes cannot replace independent training repeats. Only verified paired scenarios support paired bootstrap. A nonsignificant difference, an interval crossing zero or similar point estimates does not prove equivalence.

A five-step method may be faster but worse at the task, faster only in the learner, or unstable. Report each outcome directly. If Uniform exceeds Late, the evidence concerns this window and setting; if Late exceeds Uniform, it does not establish universal late-step importance. If no method establishes improvement over SFT, do not describe the experiment as preserving an established RL gain.

## 13. Configuration and deliverables

The executable contract is in [configs/full10.yaml](configs/full10.yaml), [late5.yaml](configs/late5.yaml) and [uniform5.yaml](configs/uniform5.yaml). Shared optimizer settings are AdamW, learning rate `1e-6`, betas `[0.9,0.999]`, epsilon `1e-8`, weight decay `1e-10`, one epoch and maximum gradient norm 1. Runtime saves resolved configurations and rejects incompatible recovery.

For the originally planned two-seed matrix, training comprises 1,200 pilot episodes plus 24B formal episodes (7,200 at B=300); development evaluation is 400 episodes and final evaluation is 1,400 episodes, plus actual replay/cache-collection work. The executed one-seed study has a smaller formal matrix, explicitly recorded in its lock.

Required tables are per-seed task results, paired learner costs and complete system-time decomposition. Useful figures show task success versus learner cost, per-task differences and training-time breakdown. Preserve raw per-episode CSVs, logs, source/model hashes, environment metadata and failures.

A completed engineering study requires valid correctness evidence, measured call reduction, a fixed-cache profile and reproducible artifacts. A completed formal online comparison additionally requires the locked three-method budget and final evaluations. Statistical claims require evidence beyond merely completing those stages. Publish conclusions at the level the measurements support.
