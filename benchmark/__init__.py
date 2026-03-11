"""
Causal Learning Benchmark (CLW) Suite
======================================
Architecture-agnostic benchmark for evaluating causal reasoning in AI systems.

Three environments of increasing causal complexity:
  CLW-1: Single Confounder
  CLW-2: Causal Chain (mediation)
  CLW-3: Common Cause (confounding)

Three test types:
  A: Behavioral Recovery
  B: Representation Update (full + proxy)
  C: Novel Intervention Generalisation

Four scoring levels:
  0: Chance
  1: Behavioral
  2: Representational
  3: Causal
  4: Counterfactual (defined, not yet measurable)

Usage:
  python -m benchmark.run_benchmark
"""

__version__ = "1.0.0"
