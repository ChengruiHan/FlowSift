# Selective transition rescoring

FlowSift changes how many stored transitions the GRPO learner rescores. Each rollout still produces ten SDE transitions and caches eleven states; evaluation still uses ten-step ODE generation. Index `k=0` is the high-noise end, and Late-5 selects `k=5,…,9`.

## The experiment's intervention

All methods share the sampler, model initialization, rewards, optimizer, attempted-group budget and evaluation procedure. Only the transition set used in the learner changes:

- Full-10 scores every transition.
- Late-5 scores the fixed final five transitions.
- Uniform-5 independently draws five of the ten transitions without replacement for each chunk.

Selection happens **before** the current/reference denoiser forward passes. Computing all ten probabilities and slicing the loss afterward would not provide the intended forward-compute savings. All methods keep the complete cached trajectory and the original timestep grid.

## Loss computation

A group contains G=4 episodes from the same initial scene. Episode i receives binary success reward R_i, with centered advantage:

$$A_i=R_i-\frac{1}{G}\sum_{j=1}^{G}R_j.$$

There is no standard-deviation normalization. For each chunk c, select a transition set S. For each k in S, sum Gaussian transition log-probabilities over **executed action positions and real action dimensions**. Padding dimensions and unexecuted positions are excluded. Current, old and reference probabilities use identical states, indices and masks.

$$r_{ick}=\exp(\operatorname{clamp}(\ell^\theta_{ick}-\ell^{old}_{ick},-20,20)),\qquad d_{ick}=\ell^{ref}_{ick}-\ell^\theta_{ick}.$$

$$L_{ick}=-\min\left(r_{ick}A_i,\operatorname{clip}(r_{ick},0.8,1.2)A_i\right)+0.01\left(e^{d_{ick}}-d_{ick}-1\right).$$

If episode i contains C_i chunks, the complete group objective is:

$$L=\frac{1}{G}\sum_i\frac{1}{C_i}\sum_c\frac{1}{|S_{ic}|}\sum_{k\in S_{ic}}L_{ick}.$$

Both the policy-gradient and k3 terms are averaged over the selected count: five for Late/Uniform and ten for Full. The learner accumulates these weighted chunk gradients, clips their total norm to 1, and performs **one AdamW step per mixed group**. Each group uses one epoch.

Old/reference scores and cached states are detached. Gradients flow through the current denoiser. The k3 term is a regularizer evaluated on sampled transitions, not an exact calculation of the complete environment-policy KL. Likewise, the masked transition-density surrogate is not the exact marginal likelihood of the final action chunk.

Implementation: [grpo.py](../src/flowvla_rl/grpo.py), [training.py](../src/flowvla_rl/training.py), [smolvla_sde.py](../src/flowvla_rl/smolvla_sde.py).

## Does selecting five steps update only those steps?

Only five transition losses are differentiated, but the denoiser uses **shared parameters at all timesteps**. Updating those parameters can change outputs at unselected timesteps too. The rollout itself runs without gradients; the learner does not backpropagate through the ten-step generation chain.

With one epoch and current approximately equal to old before the update, the initial ratio is approximately one and PPO clipping is usually inactive. Its derivative with respect to current parameters is still nonzero. A near-zero group-averaged initial policy-gradient loss does not imply a zero gradient.

The vision-language model and state projection are frozen. The action expert and applicable action/time projections are trainable. The frozen reference is always initialized from the original SFT, including when current weights resume from a checkpoint.

## What Uniform-5 preserves in expectation

For a fixed cache and fixed parameters, uniform sampling without replacement followed by division by five gives the same expected loss and gradient as the Full-10 mean. The CPU correctness suite enumerates all 252 five-element subsets to check this identity. A local keyed RNG selects indices without advancing rollout RNG.

This identity is for the **sampled objective and pre-optimizer gradient**. Gradient clipping, Adam and future online sampling are nonlinear; it does not imply identical parameter trajectories or equal final success. Late-5 deliberately changes timestep weighting and the coverage of both policy-gradient and regularization terms.

## Interpreting the observed ranking

Uniform-5 has the highest observed final success in the seed-101 experiment, exceeding Late-5 by five successful evaluation episodes. This does not establish a causal explanation or a reliable ranking. Multiple independent training seeds are needed to distinguish repeatable benefits from run variation.

The useful experimental distinction is between **random coverage of the full time grid** and **a fixed low-noise window**. A result for this one window does not rank all fixed windows or prove that late transitions are intrinsically more useful.

## Why learner savings are larger than loop savings

A learner update contains fixed work (transfer, prefix and optimizer) as well as work proportional to selected transitions. The overall training loop also includes simulation and full SDE rollout. Online policies change episode length and the frequency of mixed groups, so actual loop time must be measured separately from a fixed-cache benchmark.

See the [protocol](../FlowSift-final-plan.md) for the complete definitions and the [results](results_and_limitations.md) for measurements.
