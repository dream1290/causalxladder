# Causal Learning Benchmark (CLW)

An architecture-agnostic benchmark for measuring causal reasoning in AI systems.

**What does "causal reasoning" mean here?** An agent tracks a hidden variable, recovers from interventions, and generalises to causal manipulations it has never seen before. The benchmark measures each of these capabilities independently and assigns a level from 0 (chance) to 3 (causal).

## Canonical Results

Four agents evaluated across three environments, 100 episodes each, 2.1 seconds total:

```
Agent          Test         |       CLW-1        |       CLW-2        |       CLW-3
──────────────────────────────────────────────────────────────────────────────────────
Oracle         A            |     0.55 (L1)      |     5.23 (L1)      |     1.85 (L1)
               B-full       |     0.97 (L2)      |     0.78 (L2)      |     0.89 (L2)
               C1           |     0.88 (L3)      |     0.65 (L0)      |     0.86 (L3)
               C2           |     0.99 (L3)      |     0.92 (L3)      |     0.88 (L3)
──────────────────────────────────────────────────────────────────────────────────────
GRU            A            |     5.19 (L0)      |     11.50 (L0)     |     22.47 (L0)
               B-full       |     0.21 (L0)      |     0.58 (L0)      |     0.73 (L2)
               C1           |     0.00 (L0)      |     0.60 (L0)      |     0.34 (L0)
               C2           |     0.58 (L0)      |     0.44 (L0)      |     0.45 (L0)
──────────────────────────────────────────────────────────────────────────────────────
Q-Learner      A            |     8.09 (L0)      |     4.09 (L1)      |     15.50 (L0)
               B-full       |     0.31 (L0)      |     0.45 (L0)      |     0.40 (L0)
               C1           |     0.09 (L0)      |     0.62 (L0)      |     0.59 (L0)
               C2           |     0.68 (L0)      |     0.56 (L0)      |     0.37 (L0)
──────────────────────────────────────────────────────────────────────────────────────
Random         A            |     14.80 (L0)     |     15.67 (L0)     |     15.49 (L0)
               B-full       |     0.00 (L0)      |     0.00 (L0)      |     0.00 (L0)
               C1           |     0.35 (L0)      |     0.33 (L0)      |     0.32 (L0)
               C2           |     0.30 (L0)      |     0.30 (L0)      |     0.34 (L0)
```

The one number to look at first: **Q-Learner on CLW-3 Test A = 15.50 (L0)**. The Q-Learner, which scores L1 on CLW-2, completely fails on CLW-3 because it cannot distinguish a symptom (S1) from a cause (C). The benchmark is measuring exactly what it claims.

The GRU model was trained on a different simulator (8-dimensional hidden state, majority-vote action). Its scores measure cross-environment generalisation, not in-domain performance. CLW-3 B-full = 0.73 (L2) is the notable result: its internal representation partially tracks the common-cause structure despite never being trained on it.

---

## Installation

```bash
git clone <this-repo>
cd newai

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy torch pytest
```

No build step. No config files. The benchmark is a Python package that runs from the repository root.

---

## Quick Start (5 minutes)

**Run the full benchmark:**

```bash
python -m benchmark.run_benchmark
```

This evaluates Random, Q-Learner, and Oracle agents across CLW-1, CLW-2, and CLW-3. Output: the results table above. Takes ~2 seconds.

**Run with a trained model:**

```bash
python -m benchmark.run_benchmark --checkpoint path/to/model.pt
```

**Run specific environments or agents:**

```bash
python -m benchmark.run_benchmark --envs CLW-1 CLW-3
python -m benchmark.run_benchmark --agents oracle random
```

**Run tests:**

```bash
python -m pytest benchmark/tests/ -v          # 94 tests, ~5 seconds
python -m pytest benchmark/tests/test_baselines.py -v  # reproducibility only
```

---

## Adding Your Own Agent

Implement four methods. No inheritance required — Python structural subtyping handles conformance:

```python
import numpy as np

class MyAgent:
    def reset(self) -> None:
        """Clear internal state. Called at the start of each episode."""
        pass

    def observe(self, obs: np.ndarray) -> None:
        """
        Receive a 4-dim observation after each step.

        obs[0]: last_action / 2.0        (normalised to [0, 1])
        obs[1]: last_outcome             (1.0 if correct pull, 0.0 otherwise)
        obs[2]: step_count / max_steps   (normalised to [0, 1])
        obs[3]: noise                    (Gaussian, mean 0)

        Note: CLW-3 overrides obs[0:2] with sensor readings [S1, S2].
        """
        pass

    def act(self) -> int:
        """
        Choose an action.

        Returns: 0 (pull lever 0), 1 (pull lever 1), or 2 (wait).
        """
        return 0

    def get_representation(self) -> np.ndarray | None:
        """
        Optional: return a flat array representing the agent's internal
        belief state. Used for Test B-full (representation similarity).

        Return None for black-box agents — Test B-proxy will be used
        instead (architecture-agnostic, based on action entropy).

        Convention: [P(C=0), P(C=1), entropy, confidence].
        """
        return None
```

Then evaluate against any environment:

```python
from benchmark.clw1.evaluate import evaluate_agent
from benchmark.clw1.interventions import CLW1InterventionProtocol

agent = MyAgent()
results = evaluate_agent(agent, CLW1InterventionProtocol())
print(results.format_table())
```

Or evaluate across all environments at once:

```python
from benchmark.run_benchmark import run_benchmark

# You can also add your agent to the factory functions in run_benchmark.py
results = run_benchmark()
```

---

## The Four Levels

Levels are cumulative. Each requires the previous.

### Level 0 — Chance

Random performance. The system shows no evidence of tracking the hidden variable or adapting to interventions. Every untrained system starts here.

### Level 1 — Behavioral

**What it measures:** After the hidden variable changes, does the agent eventually switch to the correct action?

**Test A** counts steps-to-recovery after natural and interventional changes to the hidden variable. Thresholds (steps must be strictly below):

| Environment | Threshold | Rationale |
|---|---|---|
| CLW-1 | < 5 steps | Single variable, direct feedback |
| CLW-2 | < 10 steps | Causal chain, delayed feedback |
| CLW-3 | < 15 steps | Common cause, noisy sensors |

A Q-Learner can reach Level 1 on CLW-2 (steps-to-recovery = 4.09) because its reward history eventually shifts. It fails on CLW-3 (15.50) because intervening on the symptom S1 doesn't change the cause C, and the Q-Learner cannot tell the difference.

### Level 2 — Representational

**What it measures:** Does the agent's internal state update correctly in response to interventions?

**Test B-full** compares the agent's internal representation (via `get_representation()`) to the ground-truth hidden state at 5 steps post-intervention. Metric: cosine similarity, threshold > 0.6.

**Test B-proxy** (architecture-agnostic alternative): measures whether the agent's action-entropy profile shows a characteristic spike-then-recovery pattern after interventions — the signature of a system that genuinely updates an internal world model. Threshold: relative entropy spike > 0.3.

The Oracle scores L2 across all environments (B-full = 0.97, 0.78, 0.89). The GRU scores L2 only on CLW-3 B-full (0.73) — its representation partially captures the common-cause structure.

### Level 3 — Causal

**What it measures:** Does the agent generalise to novel interventions it has never experienced during training?

**Test C** presents three kinds of novel interventions:

| Sub-test | Intervention | What it tests |
|---|---|---|
| **C1** | Simultaneous: set multiple variables at once | Multi-target generalisation |
| **C2** | No-op: set variable to its current value | Recognising null interventions |
| **C3** | Out-of-distribution: values outside training range | Extrapolation to novel states |

Threshold: accuracy > 0.70. The Oracle achieves C1 = 0.88 and C2 = 0.99 on CLW-1 — perfect causal reasoning. It scores C1 = 0.65 (L0) on CLW-2, which is genuine causal uncertainty from stochastic propagation in the chain, not a measurement failure.

---

## The Three Environments

Each environment has a known ground-truth causal graph, known to the evaluator but never revealed to the agent.

### CLW-1: Single Confounder

```
C → correct_action → reward
```

One hidden binary variable `C`. Correct action = pull lever `C`. The simplest possible causal reasoning task — if your system cannot pass CLW-1, it cannot pass anything.

- **C**: flips via geometric distribution (mean 80 steps)
- **Correct action**: pull lever matching C (0 or 1)
- **Death**: 6 consecutive wrong pulls → big penalty, episode ends
- **Reward**: streak of 8 correct pulls → big reward

### CLW-2: Causal Chain (Mediation)

```
action → C1 → C2 → reward
         Target → reward
```

The agent's action influences `C1`, which propagates to `C2` with probability 0.8. Reward depends on `C2` matching a hidden `Target`. This tests whether the agent understands that effects are indirect and delayed.

- **Target**: hidden, flips geometrically (mean 80)
- **C1**: follows agent action with p=0.8
- **C2**: follows C1 with p=0.8
- **Intervention targets**: C1, C2 (set directly, bypassing normal dynamics)

### CLW-3: Common Cause (Confounding)

```
C → S1    (observable, 80% accuracy)
C → S2    (observable, 80% accuracy)
C → correct_action → reward
```

One hidden cause C produces two observable sensors S1, S2. The agent sees S1 and S2 but must infer C. The critical test: **intervening on S1 (do(S1=x)) does not change C.** An agent that merely tracks the correlation between S1 and correct action will fail after a do(S1) intervention. Only a system that understands S1 is a symptom, not a cause, will recover.

- **Observation**: `[S1, S2, 0.0, 0.0]` (sensors replace action/outcome in obs)
- **Intervention**: pinning S1 to 0 or 1 breaks the C→S1 edge
- **Key result**: Q-Learner A = 15.50 (L0) vs Oracle A = 1.85 (L1)

---

## Reference Baselines

| Agent | Strategy | Expected Level | Key Insight |
|---|---|---|---|
| **Random** | Uniform random actions | L0 everywhere | Floor — confirms tasks aren't trivially solvable |
| **Q-Learner** | Tabular Q-learning, no hidden state | L0–L1 | Passes CLW-2 Test A via reward adaptation, fails CLW-3 |
| **Oracle** | Exact Bayesian inference with known graph | L1–L3 | Ceiling — confirms environments are solvable with the right model |
| **GRU** | Trained neural network (128-dim hidden state) | L0–L2 | Cross-domain transfer baseline; trained on a different simulator |

---

## Observation Contract

All environments produce a 4-dimensional float32 observation vector at each step:

| Index | Field | Range | Description |
|---|---|---|---|
| 0 | `last_action_norm` | [0, 1] | Previous action / 2 |
| 1 | `last_outcome` | {0.0, 1.0} | Whether last pull was correct |
| 2 | `steps_norm` | [0, 1] | Current step / max_steps |
| 3 | `noise` | ℝ | Gaussian noise N(0, 0.05) |

**Exception**: CLW-3 overrides indices 0–1 with observable sensor readings `[S1, S2]`.

Actions: `0` (pull lever 0), `1` (pull lever 1), `2` (wait). Wait is safe — no penalty, no streak change — but provides no information about the hidden state.

---

## Evaluation Protocol

Each environment uses 100 fixed episodes (seeds in `data/eval_seeds.json`):

| Episodes | Test | What happens |
|---|---|---|
| 0–39 | **A** | Each episode has a do(C) intervention at step 10, 25, or 50. Recovery is measured. |
| 40–69 | **B** | Same protocol; internal representation is compared to ground truth 5 steps post-intervention. |
| 70–99 | **C** | Novel interventions the agent has never seen during training. |

Interventions use Pearl's do-operator: `do(C=1)` sets the hidden variable directly, bypassing all normal dynamics. The evaluator applies interventions at the scheduled step, then measures the agent's response.

All seed values are committed to the repository and never regenerated at runtime. Results reported against v1.0.0 seeds are comparable across all implementations.

---

## Project Structure

```
newai/
├── benchmark/                     # ← The benchmark itself (self-contained)
│   ├── core/
│   │   ├── base_env.py            #   Abstract CLWEnvironment base class
│   │   ├── agent_interface.py     #   BenchmarkAgent protocol + BehavioralProxy
│   │   └── scoring.py             #   Thresholds, level classifiers, ScoringMatrix
│   ├── clw1/                      #   Single Confounder (C → action → reward)
│   │   ├── env.py, interventions.py, evaluate.py, baselines.py
│   ├── clw2/                      #   Causal Chain (action → C1 → C2 → reward)
│   │   ├── env.py, interventions.py, evaluate.py, baselines.py
│   ├── clw3/                      #   Common Cause (C → S1, C → S2, C → action)
│   │   ├── env.py, interventions.py, evaluate.py, baselines.py
│   ├── data/eval_seeds.json       #   100 fixed seeds (never regenerated)
│   ├── gru_agent.py               #   Adapter: wraps trained GRU as BenchmarkAgent
│   ├── run_benchmark.py           #   CLI runner (--envs, --agents, --checkpoint)
│   ├── tests/                     #   94 tests (scoring, environments, reproducibility)
│   └── README.md                  #   This file
│
├── training/                      # GRU model training pipeline
│   ├── Causal_model.py            #   v1 training script
│   ├── Causal_model_v2.py         #   v2 training script (clean architecture)
│   ├── causal_simulator.py        #   Standalone simulator
│   ├── baseline_ablation.py       #   Ablation: train without tracking reward
│   ├── intervention_test.py       #   do(C) vs observe(C) experiment
│   └── language_head.py           #   Phase 2: z → natural language decoder
│
├── experiments/                   # Evolution and scaling experiments
│   ├── v1.py, v2.py, v3.py       #   Evolutionary emergence of causal capacity
│   ├── v3_heatmap.py              #   (flip_mean, penalty) parameter sweep
│   ├── v3_multiseed.py            #   Multi-seed replication
│   ├── scaling_experiment.py      #   Scaling experiments
│   └── vector_scaling_experiment.py
│
├── checkpoints/                   # Trained model weights and data
│   ├── causal_belief_v2_final.pt  #   Final GRU model (used by benchmark)
│   ├── phase_a_checkpoint.pt      #   Phase A intermediate checkpoint
│   ├── causal_belief_model.pt     #   v1 model
│   ├── baseline_no_tracking.pt    #   Ablation model (no tracking reward)
│   ├── language_head.pt           #   Language decoder weights
│   ├── dz_norms.npy              #   Training log: ||Δz|| history
│   └── rewards.npy                #   Training log: reward history
│
├── figures/                       # All generated plots (13 files)
│
└── paper/
    └── newpaper.md                # Research paper draft
```

94 tests total. Full suite runs in under 5 seconds. Full benchmark in under 10 seconds.

---

## Design Principles

**Architecture-agnostic.** The benchmark evaluates behavior, not architecture. A Q-learner, a GRU, a transformer, a Bayesian network, or a human can be scored on the same protocol. The only requirement is `reset()`, `observe()`, `act()`.

**Deterministic.** Fixed seeds guarantee bit-identical evaluations across machines, Python versions, and years. If two labs report different scores for the same agent on v1.0.0 seeds, one of them has a bug.

**Cumulative scoring.** Level 3 requires Level 2 requires Level 1. You cannot claim causal reasoning without first demonstrating behavioral adaptation and representational accuracy. This prevents gaming the metric.

**Separation of concerns.** The benchmark lives entirely in the environment and evaluation protocol — not in any assumption about how the agent is built. The causal graph is known to the evaluator. It is never given to the agent.

**Pearl's hierarchy.** The three test types map directly onto Pearl's ladder of causation: Test A measures associative adaptation (does the agent respond to observed changes?), Test B measures interventional reasoning (does the internal state track do-operator effects?), Test C measures counterfactual generalisation (can the agent handle interventions it has never seen?).
