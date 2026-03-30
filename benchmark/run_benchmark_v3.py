#!/usr/bin/env python3
"""
CLW Benchmark Runner for v3 Separated Architecture
====================================================
Evaluates the v3 GRU agent (separated world model / policy) against
all three CLW environments using the existing benchmark infrastructure.

Does NOT modify run_benchmark.py — uses the same evaluate_agent()
functions with the v3 adapter.

Usage:
    python -m benchmark.run_benchmark_v3 --checkpoint training/causal_belief_v3_final.pt
    python -m benchmark.run_benchmark_v3 --checkpoint training/causal_belief_v3_final.pt --envs CLW-1
"""

import argparse
import os
import sys
import time

from .core.scoring import ScoringMatrix, TEST_TYPES
from .gru_v3_agent import GRUV3BenchmarkAgent


def _evaluate_clw1(agent):
    from .clw1.evaluate import evaluate_agent
    from .clw1.interventions import CLW1InterventionProtocol
    return evaluate_agent(agent, CLW1InterventionProtocol())

def _evaluate_clw2(agent):
    from .clw2.evaluate import evaluate_agent
    from .clw2.interventions import CLW2InterventionProtocol
    return evaluate_agent(agent, CLW2InterventionProtocol())

def _evaluate_clw3(agent):
    from .clw3.evaluate import evaluate_agent
    from .clw3.interventions import CLW3InterventionProtocol
    return evaluate_agent(agent, CLW3InterventionProtocol())


ENV_EVALUATORS = {
    'CLW-1': _evaluate_clw1,
    'CLW-2': _evaluate_clw2,
    'CLW-3': _evaluate_clw3,
}


def run_v3_benchmark(checkpoint_path, envs=None, seed=42):
    """Run the CLW benchmark with the v3 separated architecture agent."""
    if envs is None:
        envs = list(ENV_EVALUATORS.keys())

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("═" * 72)
    print("  CLW Benchmark — v3 Separated World Model / Policy")
    print(f"  Checkpoint: {checkpoint_path}")
    print("═" * 72)

    agent = GRUV3BenchmarkAgent(checkpoint_path, seed=seed)
    results = {}
    total_start = time.time()

    for env_name in envs:
        if env_name not in ENV_EVALUATORS:
            print(f"  ⚠ Unknown environment: {env_name}")
            continue

        t0 = time.time()
        print(f"  Evaluating GRU-v3 on {env_name}...", end='', flush=True)

        # Reset agent for each environment
        agent.reset()
        matrix = ENV_EVALUATORS[env_name](agent)
        elapsed = time.time() - t0

        results[env_name] = matrix
        print(f" done ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start

    # Format results
    print(f"\n{'='*72}")
    print(f"  GRU-v3 Results  (total: {total_elapsed:.1f}s)")
    print(f"{'='*72}\n")

    tests = TEST_TYPES
    header = f"{'Test':<15s}"
    for env in envs:
        header += f" | {env:^18s}"
    print(header)
    print("─" * len(header))

    for test in tests:
        row = f"{test:<15s}"
        for env in envs:
            matrix = results.get(env)
            if matrix is None:
                cell = "—"
            else:
                score = matrix.get_score(env, test)
                if score is None:
                    cell = "—"
                elif score.level == -1:
                    cell = "N/A"
                else:
                    cell = f"{score.value:.2f} (L{score.level})"
            row += f" | {cell:^18s}"
        print(row)

    print("─" * len(header))
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='CLW Benchmark for v3 separated architecture',
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='training/causal_belief_v3_final.pt',
        help='Path to v3 model checkpoint',
    )
    parser.add_argument(
        '--envs', nargs='+', default=None,
        choices=list(ENV_EVALUATORS.keys()),
        help='Environments to evaluate (default: all)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
    )
    args = parser.parse_args()

    run_v3_benchmark(args.checkpoint, envs=args.envs, seed=args.seed)


if __name__ == '__main__':
    main()
