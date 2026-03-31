# The Causal Ladder

**Can selection pressure drive the emergence of causal reasoning?**

This repository contains the benchmark, evolutionary experiments, and neural model from our investigation of causal capacity as an adaptive trait. The core claim: when environmental volatility is high enough and the metabolic cost of tracking hidden variables is low enough, populations evolve causal reasoning as a stable equilibrium — not as an all-or-nothing switch, but as a continuous parameter that settles proportional to selection pressure.

Research paper 1 (environment) : https://zenodo.org/records/19118615?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijc4MzEwMjYwLWE4MDMtNDRmYS1hNjUyLTUyMWU4YmQwZTI3MyIsImRhdGEiOnt9LCJyYW5kb20iOiJlZjE5MDNiZjQxNTk4YjcwMGI5NjFjZmNjYzYwZWU1YiJ9.L58LTdhRj1tqy0dORFaSMaDMW3XXVVSRmAmOxjk8syOAHJOToiItb0oFkpQ2Yx8YwqP5qo3X5gN1X-p3S4u8yQ
Research paper 2 (architecture) : https://zenodo.org/records/19342006?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhkOWY3Y2UzLTRhMDItNGVkZi04M2FkLTNhODY2NTRkOGI0OCIsImRhdGEiOnt9LCJyYW5kb20iOiJlNmY2ZTE3OGYzNjExOTY2MDU3NDczNDBmMWNkNzEwYiJ9.9nyOMC0GOaxapQTzFFUziHEZJY1OEwBCtNEIfF_ucvu9XUREdcQfuYjbgyz8Bz_0liI8eaItYvnb0G-YTj68Jw
## What's Here

### [`benchmark/`](benchmark/) — Causal Learning Benchmark (CLW)

An architecture-agnostic benchmark for measuring causal reasoning in AI systems. Three environments of increasing causal complexity, three test types, four scoring levels. Any system that implements `reset()`, `observe()`, `act()` can be evaluated.

**Headline result:**

| Agent | CLW-1 | CLW-2 | CLW-3 | Best Level |
|---|---|---|---|---|
| Oracle (Bayesian) | L1 | L1 | L1 | **L3** |
| GRU (neural) | L0 | L0 | L0 | **L2** (CLW-3 B-full) |
| Q-Learner | L0 | L1 | L0 | L1 |
| Random | L0 | L0 | L0 | L0 |

The Q-Learner passes CLW-2 (causal chain — reward history shifts) but fails CLW-3 (common cause — cannot distinguish symptom from cause). The GRU, trained on a different simulator, partially tracks the common-cause structure in CLW-3 despite never seeing it during training.

→ Full details: [`benchmark/README.md`](benchmark/README.md)

### [`experiments/`](experiments/) — Evolutionary Emergence

Computational experiments demonstrating that causal capacity emerges as a continuous, evolvable trait under selection pressure:

- **`v3.py`** — Main evolutionary simulation. Populations evolve `causal_capacity ∈ [0, 1]` under selection pressure from environmental volatility.
- **`v3_multiseed.py`** — Multi-seed replication (8 seeds, mean ± std envelope).
- **`v3_heatmap.py`** — Parameter sweep across (flip_mean, penalty) space, mapping the equilibrium surface.
- **`scaling_experiment.py`** / **`vector_scaling_experiment.py`** — Scaling laws for causal capacity vs. hidden dimensionality.
- **`v2.py`** — Earlier version with binary causal capacity (superseded by v3's continuous model, included for reproducibility).

### [`training/`](training/) — Neural Model

The GRU-based Causal Belief Model and supporting experiments:

- **`Causal_model_v2.py`** — Clean v2 architecture: GRUCell(4→128), persistent belief state z, trained with tracking reward.
- **`causal_simulator.py`** — Standalone lever-world simulator used for training.
- **`intervention_test.py`** — Core scientific result: `do(C)` vs `observe(C)` discrimination.
- **`baseline_ablation.py`** — Ablation: same architecture trained without the tracking reward.
- **`language_head.py`** — Phase 2: decodes belief state z into natural language descriptions.

## Quick Start

```bash
git clone https://github.com/dream1290/causalxladder.git
cd causalxladder
python -m venv venv
source venv/bin/activate
pip install numpy torch pytest

# Run the benchmark (2 seconds)
python -m benchmark.run_benchmark

# Run tests (94 tests, 5 seconds)
python -m pytest benchmark/tests/ -v
```

## Pretrained Weights

Model checkpoints (`.pt` files) are not stored in git. They are available in the [GitHub Releases](https://github.com/dream1290/causalxladder/releases) page.

To run the benchmark with the GRU agent:

```bash
# Download checkpoints/causal_belief_v2_final.pt from Releases, then:
python -m benchmark.run_benchmark --checkpoint checkpoints/causal_belief_v2_final.pt
```

## Project Structure

```
causalxladder/
├── benchmark/              # Causal Learning Benchmark (self-contained)
│   ├── core/               #   Base env, agent protocol, scoring
│   ├── clw1/               #   Single Confounder (C → action → reward)
│   ├── clw2/               #   Causal Chain (action → C1 → C2 → reward)
│   ├── clw3/               #   Common Cause (C → S1, C → S2, C → action)
│   ├── data/               #   Fixed evaluation seeds
│   ├── tests/              #   94 tests
│   ├── gru_agent.py        #   GRU → BenchmarkAgent adapter
│   ├── run_benchmark.py    #   CLI runner
│   └── README.md           #   Detailed benchmark documentation
├── training/               # GRU model training pipeline
├── experiments/            # Evolutionary emergence experiments
├── README.md               # This file
├── LICENSE                 # Apache 2.0
└── .gitignore
```

## Citation

```bibtex
@article{bahloul2025causalladder,
  title={The Causal Ladder: Selection Pressure and the Emergence of Causal Reasoning},
  author={Oualid Bahloul},
  year={2025}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
