# AGENTS.md

## Repository Purpose

This repository studies RSU-UAV-Vehicle video delivery using
two-timescale Lyapunov optimization and reinforcement learning.

The repository contains multiple historical and experimental branches.
Do not assume that every remaining PPO, DQN, HRL, or legacy file is part
of the currently active algorithm.

Always determine actual execution flow from:
- branch,
- config,
- imports,
- entry point,
- controller construction,
- and runtime call graph.

Do not infer the active formulation merely from file or directory names.

---

## Current Research Baseline

Unless the PR explicitly changes the research formulation, use the following
current baseline when reviewing the proposed method.

System:
- RSU-UAV-Vehicle user video-delivery system.
- Each RSU coverage region can employ at most one UAV.
- The RSU performs primary user scheduling.
- The UAV supports residual users.
- A hired UAV may support multiple users requesting compatible content.
- UAV trajectory optimization is not part of the current main problem.
- A hired UAV is treated as hovering during service.
- Travel time and travel energy between the service location and charging
  station are ignored under the current assumption unless the branch
  explicitly introduces them.

Two timescales:
- Slow-timescale decisions are made at the beginning of a round/frame.
- Slow decisions remain fixed throughout that round/frame.
- Slow-timescale includes RSU-user scheduling, UAV employment, and
  UAV-user candidate scheduling.
- The current main slow-timescale approach is DPP-based rather than
  assuming that legacy slow-PPO/HRL code is active.
- Fast-timescale decisions are updated every slot.
- Fast-timescale uses PPO for transmission decisions such as video chunks,
  layer/quality, and UAV transmit power.

If branch-local code intentionally uses a different experimental formulation,
review against that branch while clearly flagging divergence from the current
main formulation.

---

## Review Philosophy

Prioritize:
1. research-formulation correctness,
2. physical feasibility,
3. algorithm correctness,
4. experiment validity,
5. reproducibility,
6. runtime correctness.

Do not spend review comments on cosmetic style unless it can produce a real
bug or experimental ambiguity.

Do not silently recommend changing:
- research objectives,
- reward definitions,
- DPP equations,
- hyperparameters,
- baseline algorithms,
- or scenario assumptions

unless there is a concrete correctness problem.

A review finding should state:
- code location,
- observed behavior,
- expected behavior,
- consequence,
- and a minimal verification or correction when possible.

---

## Source of Truth

Use this priority when judging conflicting implementation details:

1. requirements explicitly stated in the PR,
2. latest branch-local formulation/configuration,
3. actual executable code and call graph,
4. current tests,
5. older code or comments.

If documentation and executable code disagree, flag the mismatch.
Do not arbitrarily assume one is correct.

Distinguish:
- intended formulation,
- actual code behavior,
- mismatch,
- suggested fix.

---

## Two-Timescale Consistency

The most important invariant is separation of slow and fast decisions.

Verify that slow-timescale variables:
- are selected at the intended frame/round boundary,
- remain unchanged during every fast slot of the same round,
- are refreshed only at the next slow decision point.

Flag any code path where a supposedly slow action changes inside a round.

Verify that fast-timescale decisions:
- use the current slot state,
- respect the fixed slow association/scheduling decisions,
- and do not silently overwrite slow decisions.

Pay attention to off-by-one errors at round boundaries.

---

## Scheduling and Association

Check the coupling between:
- RSU-user scheduling,
- residual-user construction,
- UAV hiring,
- UAV-user candidate scheduling,
- and actual fast-timescale service.

Flag:
- the same user being served by incompatible simultaneous associations,
- a UAV serving a user outside its slow candidate set,
- hiring a UAV when the corresponding action prohibits service,
- UAV service when the UAV is unavailable/charging,
- invalid region/UAV/user indexing,
- or residual users being computed before the required RSU scheduling result.

If a hired UAV has no valid candidate user, ensure that this behavior is
intentional and its hiring cost is treated consistently.

---

## Cache / Content Compatibility

If content compatibility is represented in the active branch, verify that
a UAV serves only users compatible with the UAV's available content.

Do not introduce or remove caching assumptions unless explicitly requested.

If a branch uses a simplified no-cache formulation, do not flag the absence
of cache selection as a bug.

---

## Queue Dynamics

Queue updates must respect the intended event ordering.

For each queue or virtual queue, verify:
- units,
- sign,
- clipping,
- min/max operation,
- initialization,
- upper/lower bounds,
- and update timing.

Distinguish physical playback queue from virtual queue.

Flag:
- negative physical queues,
- delivery counted twice,
- playback consumption applied twice,
- virtual-queue sign reversal,
- use of next-slot state before it is observable,
- and incorrect reset at frame/episode boundaries.

Do not assume historical definitions of `Q`, `Z`, `B`, `theta`, or battery
queues are still active.
Read their actual definitions in the branch under review.

---

## DPP / Lyapunov Logic

When DPP is implemented, verify that code terms correspond to the intended
mathematical objective.

Check:
- sign of drift terms,
- sign of penalty terms,
- scaling coefficients,
- hiring-cost contribution,
- quality/degradation terms,
- energy terms,
- and constants that should not affect decisions.

If the code minimizes DPP but PPO maximizes reward, verify that the reward
uses the intended negative sign.

Flag accidental double rewards or double penalties.

In particular, inspect whether both queue terms and quality terms reward
the same delivery quantity in an unintended way.

Do not claim classical Lyapunov guarantees automatically hold when a PPO
policy approximates a subproblem.
Such guarantees require explicit assumptions and proof.

---

## Slow DPP and Fast Policy Coupling

If slow DPP evaluates future fast behavior, verify how that quantity is
actually obtained.

Acceptable implementations may include:
- a myopic surrogate,
- analytical expectation,
- historical statistics,
- sampled rollout,
- or policy rollout,

but the code must make the approximation explicit.

Flag any implementation that appears to use unavailable future information.

When fast-policy rollouts are used inside slow decisions, check:
- checkpoint identity,
- observation normalization,
- deterministic/stochastic action mode,
- random seed handling,
- environment copying/reset,
- and whether rollout state leaks into the real environment.

---

## PPO Mixed Action

Fast PPO may contain mixed action types.

Review categorical and continuous components separately.

Check:
- chunk categorical distribution,
- layer/quality categorical distribution,
- transmit-power distribution,
- action masks,
- log probabilities,
- entropy,
- sampling,
- deterministic evaluation,
- and action decoding.

For inactive dimensions, ensure masks are applied consistently to:
- sampling,
- log_prob,
- entropy,
- PPO ratio,
- and loss.

The `chunk == 0` case requires special attention:
- layer choice should not create artificial reward/cost,
- transmit power should not be consumed unless intended,
- invalid inactive actions should not contaminate PPO statistics.

Check power scaling between:
- policy output,
- normalized action,
- physical transmit power,
- environment,
- and logged metrics.

---

## PPO Training Correctness

Verify:
- rollout storage,
- GAE,
- return computation,
- done handling,
- bootstrap value,
- minibatch indexing,
- PPO clipping,
- entropy,
- value loss,
- gradient clipping,
- and checkpoint state.

Gymnasium termination handling must distinguish:
- `terminated`
- `truncated`

Bootstrap behavior must match the intended semantics.

Flag training/evaluation inconsistencies such as:
- different observation normalization,
- different action masking,
- stochastic evaluation unintentionally enabled,
- checkpoint architecture mismatch,
- or missing normalizer state.

---

## Gymnasium API

For environment code, check:
- `reset()` return structure,
- `step()` return structure,
- observation space,
- action space,
- `terminated`,
- `truncated`,
- and `info`.

Observation values and shapes must agree with declared spaces.

Action decoding must agree with the declared action representation.

---

## UAV Battery and Energy

Battery causality is a hard correctness requirement.

Verify:
- current SoC,
- hovering energy,
- communication energy,
- charging energy,
- threshold logic,
- upper/lower bounds,
- charging state,
- and service availability.

Flag any slot that spends energy unavailable at the beginning of the
corresponding action.

Check units carefully:
- joules,
- watts,
- seconds,
- normalized SoC,
- percentages,
- or scaled queue units.

Do not mix them without explicit conversion.

If the active formulation assumes automatic charging, do not introduce
charging as an RL action.

If a UAV is charging and therefore unavailable, verify that:
- no communication delivery is credited,
- no transmit power is consumed,
- and associations/metrics treat the UAV consistently.

---

## Hovering and Communication Energy

When hovering power is modeled, confirm that it is applied during the
intended slots only.

Communication energy should use the actual transmit power and slot duration
with the intended conversion to battery/SoC units.

Flag:
- energy consumed by inactive UAVs,
- double-counted hovering energy,
- transmit-power energy when no transmission occurred,
- or missing energy when an active UAV transmits.

---

## Channel and Capacity

RSU-user and UAV-user channels are not assumed identical.

The application-level video quality definition may be shared, but:
- channel gain,
- SNR,
- bandwidth,
- capacity,
- and power behavior

may differ.

Flag code that accidentally reuses the same channel/capacity value for both
link types without an explicit modeling reason.

Check:
- distance,
- altitude,
- path loss,
- gain,
- noise,
- bandwidth,
- SNR,
- and capacity units.

Verify that selected chunks/layers satisfy the intended capacity constraint.

---

## Video Delivery and QoE

Review delivery accounting together with:
- chunk count,
- quality/layer,
- quality degradation,
- switching/degradation metrics,
- buffer,
- stall,
- and service rate.

Do not compare total values across experiments with different horizons unless
they are properly normalized.

Flag metrics whose denominator can be zero.

Quality-per-chunk and degradation-per-chunk should only divide by actual
delivered chunks.

---

## Baselines

Baselines and the proposed method must remain logically separate.

Do not silently copy proposed-method functionality into a baseline.

For each baseline, preserve the intended core idea while sharing only common
environment/scenario parameters needed for fair comparison.

When comparing methods, verify consistency of:
- mobility traces,
- channel parameters,
- user population,
- horizon,
- seeds,
- quality ladder,
- workload,
- and evaluation metric definitions.

If a baseline intentionally removes a paper component to match the common
scenario, ensure the adaptation is documented rather than silently changed.

---

## Mobility

If SUMO or another mobility trace is used, treat mobility generation separately
from communication/control logic.

For common evaluation, proposed and baseline methods should consume the same
mobility realization whenever the experiment claims a controlled comparison.

Flag:
- different trace seeds,
- inconsistent user indexing,
- stale positions,
- and mismatched simulation clocks.

---

## Logging and Metrics

Do not trust reward alone.

Important experiment outputs may include:
- delivered chunks,
- quality/chunk,
- degradation/chunk,
- stall ratio,
- service rate,
- hiring count/rate/cost,
- UAV energy / SoC,
- battery outage,
- action distributions,
- and association sets.

Ensure logging does not alter environment state.

Check ratio denominators and aggregation:
- per slot,
- per frame,
- per episode,
- and global aggregates

must not be mixed accidentally.

Avoid averaging ratios too early when the intended metric is a global ratio.

---

## Numerical and Runtime Safety

Check:
- NaN / Inf,
- divide-by-zero,
- empty user sets,
- zero active actions,
- no hired UAV,
- fully charged battery,
- depleted battery,
- end-of-route users,
- and episode/frame boundary cases.

Validate:
- numpy / torch dtype,
- device,
- shape,
- batch dimension,
- and scalar-vs-vector assumptions.

Flag silent CPU fallback for experiments intended to run on GPU.

---

## Reproducibility

Training/evaluation runs should preserve enough information to identify:
- branch,
- commit,
- dirty state when possible,
- config,
- seed,
- checkpoint,
- Python version,
- PyTorch version,
- CUDA version,
- GPU,
- environment parameters,
- and normalization state.

Do not overwrite existing experiment outputs silently.

Checkpoint save/load must include every state required for valid resume or
evaluation.

---

## Tests and Validation

Prefer the smallest relevant validation first.

Where applicable:
1. syntax / import check,
2. unit tests,
3. deterministic smoke test,
4. short rollout,
5. full experiment.

Pay special attention to tests for:
- slow-action invariance within a round,
- queue bounds,
- battery causality,
- no service while charging,
- action masks,
- `chunk == 0`,
- power bounds,
- finite reward/DPP,
- checkpoint save/load,
- deterministic seed,
- and Gymnasium termination semantics.

Do not state that a test passed unless it was actually executed.

---

## Seraph Execution

Long research workloads should not be run on the login/master node.

Use:
- `srun` for short debugging,
- `sbatch` for long training/evaluation.

Repository paths and private Conda environments should generally live under
`/data/$USER`.

Do not hard-code an old partition, node, GPU count, or resource limit as a
universal truth.

Resource availability should be checked from the current Slurm configuration.

For GPU experiments, silent CPU fallback should be treated as a failure unless
CPU execution is explicitly the experiment.

---

## Review Output

Prioritize findings that can affect paper conclusions.

Suggested severity:
- Critical: invalid formulation, data leakage, physically impossible behavior,
  or results that cannot be trusted.
- High: training/evaluation bug, incorrect DPP/reward, battery causality error,
  wrong action masking, or unfair baseline comparison.
- Medium: reproducibility issue, metric error, missing boundary handling,
  checkpoint incompatibility.
- Low: realistic maintainability issue that can later cause incorrect experiments.

Avoid cosmetic-only comments.
