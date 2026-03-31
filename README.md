# The Causal Ladder

**Can selection pressure drive the emergence of causal reasoning?**

This repository contains the benchmark, evolutionary experiments, and neural models from our investigation of causal capacity as an adaptive trait. The core claim: when environmental volatility is high enough and the metabolic cost of tracking hidden variables is low enough, populations evolve causal reasoning as a stable equilibrium — not as an all-or-nothing switch, but as a continuous parameter that settles proportional to selection pressure.

**Research Papers**

- Paper 1 — Environment: [Zenodo 19118615](https://zenodo.org/records/19118615)
- Paper 2 — Architecture: [Zenodo 19342006](https://zenodo.org/records/19342006)

---

## Headline Results

### CLW Benchmark — v3 Separated Architecture (2026-03-30)

| Agent | CLW-1 | CLW-2 | CLW-3 | Best Level |
|---|---|---|---|---|
| Oracle (Bayesian) | L1 | L1 | L1 | L3 |
| **GRU-v3 (separated)** | **L2** | **L2** | **L2** | **L3** (CLW-1 C1, C2) |
| GRU-v2 (baseline) | L0 | L0 | L0 | L2 (CLW-3 B-full) |
| Q-Learner | L0 | L1 | L0 | L1 |
| Random | L0 | L0 | L0 | L0 |

The v3 separated architecture achieves L2 on all three B-full tests and L3 on CLW-1 C1/C2 — the first neural agent to do so on this benchmark.

### v2 to v3 Comparison

| Test | v2 | v3 | Change |
|---|---|---|---|
| CLW-1 B-full | 0.21 (L0) | 0.61 (L2) | +0.40 |
| CLW-2 B-full | 0.58 (L0) | 0.71 (L2) | +0.13 |
| CLW-3 B-full | 0.73 (L2) | 0.78 (L2) | +0.05 |
| CLW-1 C1 | 0.00 (L0) | 1.00 (L3) | +1.00 |
| CLW-1 C2 | 0.58 (L0) | 0.78 (L3) | +0.20 |
| CLW-3 A | 22.47 (L0) | 3.17 (L1) | -19.3 steps |

### Intervention Test

After `do(C)` — Pearl's do-operator, setting the hidden state directly:

- 98.4% recovery within 5 steps
- Approximately 80% accuracy by step 15 (v2 was approximately 40%)
- Steps to first correct action: 0.4 (v2 was 8-12)

---

## What's Here

### benchmark/ — Causal Learning Benchmark (CLW)

An architecture-agnostic benchmark for measuring causal reasoning in AI systems. Three environments of increasing causal complexity, three test types, four scoring levels. Any system that implements `reset()`, `observe()`, `act()` can be evaluated.

Full details in [benchmark/README.md](benchmark/README.md).

### training/ — Neural Models

#### v3 — Separated World Model / Policy (current)

The key architectural insight: separate the world model from the policy with a gradient wall.

```
obs --> [world_gru (128-dim)] --> z_w --+--> [action_head] --> action
                                        |        ^
                                   detach()      |
                                        +--> [policy_gru (64-dim)] --> z_p
```

`z_w.detach()` ensures policy gradients cannot contaminate the world model. The world model is trained only on observation tracking signals. The policy reads `z_w` (detached) to make decisions. This forces all causal state information into `z_w`, where it can be measured independently.

Training uses a 3-phase curriculum (52k episodes, approximately 4-6 hours on CPU):

1. **Phase A** (25k episodes) — Episodic. Both `z_w` and `z_p` reset each episode. Streak curriculum ramps from 3 to 8.
2. **Phase B** (15k episodes) — `z_w` persists across episodes, `z_p` resets. Forces causal state into the world model.
3. **Phase C** (12k episodes) — Both persist. Full deployment configuration.

Key finding: `z_w` encodes the hidden state C at 70.3% accuracy (non-linear probe), despite PCA showing only 12.8% correlation on PC1. The world model uses a non-linear representation that the policy GRU can decode but linear methods cannot. The gradient wall prevents polcy gradients from corrupting this partial signal — which is what happened in v2.

Files:

- `causal_model_v3.py` — Architecture, training, 3-phase curriculum
- `intervention_test_v3.py` — `do(C)` vs `observe(C)` experiment for v3
- `train_probe.py` — Non-linear probe: trains MLP on (z_w, C_label) pairs

#### v2 — Single GRU Baseline

- `Causal_model_v2.py` — GRUCell(4 to 128), single belief state z, tracking reward
- `causal_simulator.py` — Standalone lever-world simulator
- `intervention_test.py` — Original `do(C)` vs `observe(C)` experiment
- `baseline_ablation.py` — Ablation: same architecture without tracking reward
- `language_head.py` — Decodes belief state z into natural language descriptions

### experiments/ — Evolutionary Emergence

Computational experiments demonstrating that causal capacity emerges as a continuous, evolvable trait under selection pressure:

- `v3.py` — Main evolutionary simulation. Populations evolve `causal_capacity` in [0, 1] under selection pressure from environmental volatility.
- `v3_multiseed.py` — Multi-seed replication (8 seeds, mean +/- std envelope).
- `v3_heatmap.py` — Parameter sweep across (flip_mean, penalty) space, mapping the equilibrium surface.
- `scaling_experiment.py` / `vector_scaling_experiment.py` — Scaling laws for causal capacity vs hidden dimensionality.

### results/ — Evaluation Outputs

Raw output logs from all evaluations, organized by version:

- `results/v3/training_log.txt` — Full 52k episode training log
- `results/v3/intervention_test.txt` — 200-trial do(C) experiment
- `results/v3/benchmark_pca.txt` — CLW benchmark with PCA representation
- `results/v3/benchmark_probe.txt` — CLW benchmark with non-linear probe
- `results/v3/probe_training.txt` — Non-linear probe training log

### figures/ — Generated Plots

All diagnostic plots. Key v3 figures:

- `figures/v3/causal_model_v3_diagnostics.png` — Training curves, PCA scatter
- `figures/v3/intervention_test_v3.png` — do(C) vs natural flip accuracy, world model vs policy responsiveness

---

## Quick Start

```bash
git clone https://github.com/dream1290/causalxladder.git
cd causalxladder
python -m venv venv
source venv/bin/activate
pip install numpy torch scikit-learn pytest

# Run the benchmark with baselines (2 seconds)
python -m benchmark.run_benchmark

# Run with v3 model (checkpoints included)
python -m benchmark.run_benchmark_v3 --checkpoint checkpoints/v3/causal_belief_v3_final.pt

# Run tests (94 tests, 5 seconds)
python -m pytest benchmark/tests/ -v
```

## Training from Scratch

```bash
# Full v3 training (approximately 4-6 hours CPU)
cd training
python causal_model_v3.py

# Or phase by phase:
python causal_model_v3.py --phase-a-only
python causal_model_v3.py --phase-b-only
python causal_model_v3.py --phase-c-only

# Intervention test (after training)
python intervention_test_v3.py

# Non-linear probe (after training)
python train_probe.py
```

## Checkpoints

v3 model checkpoints are included in `checkpoints/v3/`:

| File | Size | Contents |
|---|---|---|
| `causal_belief_v3_final.pt` | 457 KB | Final model weights and PCA data |
| `v3_probe.pt` | 19 KB | Trained non-linear probe (128 to 32 to 1) |
| `v3_phase_a_checkpoint.pt` | 454 KB | Phase A intermediate |
| `v3_phase_b_checkpoint.pt` | 455 KB | Phase B intermediate |

v2 checkpoints are available on the [Releases](https://github.com/dream1290/causalxladder/releases) page.

---

## Project Structure

```
causalxladder/
    benchmark/                  Causal Learning Benchmark (self-contained)
        core/                   Base env, agent protocol, scoring
        clw1/                   Single Confounder (C -> action -> reward)
        clw2/                   Causal Chain (action -> C1 -> C2 -> reward)
        clw3/                   Common Cause (C -> S1, C -> S2, C -> action)
        data/                   Fixed evaluation seeds
        tests/                  94 tests
        gru_agent.py            v2 GRU adapter
        gru_v3_agent.py         v3 GRU adapter (probe + PCA)
        run_benchmark.py        CLI runner (baselines + v2)
        run_benchmark_v3.py     CLI runner (v3 agent)
        README.md               Detailed benchmark documentation
    training/                   Neural model training
        causal_model_v3.py      v3: separated world model / policy
        Causal_model_v2.py      v2: single GRU baseline
        causal_simulator.py     Lever-world simulator
        intervention_test_v3.py v3 do(C) vs observe(C)
        intervention_test.py    v2 do(C) vs observe(C)
        train_probe.py          Non-linear probe for z_w
        baseline_ablation.py    Ablation (no tracking reward)
        language_head.py        z to natural language decoder
    experiments/                Evolution and scaling experiments
    results/v3/                 Raw evaluation outputs
    checkpoints/v3/             Trained model weights
    figures/v3/                 Diagnostic plots
    README.md
    LICENSE                     Apache 2.0
    .gitignore
```

---

## Citation

```bibtex
@article{bahloul2026causalladder,
  title={The Causal Ladder: Selection Pressure and the Emergence of Causal Reasoning},
  author={Oualid Bahloul},
  year={2026}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
