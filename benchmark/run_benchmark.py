#!/usr/bin/env python3
"""
Causal Learning Benchmark (CLW) — Unified Runner
=================================================
Runs all baseline agents across all three CLW environments.
Produces a single combined results table.

Usage:
    python -m benchmark.run_benchmark
    python -m benchmark.run_benchmark --agents random qlearner oracle
    python -m benchmark.run_benchmark --envs CLW-1 CLW-3
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

from .core.scoring import ScoringMatrix, TestScore, ENVIRONMENTS, TEST_TYPES

# ── Agent factories ──────────────────────────────────────────────────────

def _make_gru_agent(checkpoint_path: str, seed: int = 42):
    """Create GRU agent from checkpoint, or None if checkpoint missing."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    from .gru_agent import GRUBenchmarkAgent
    return GRUBenchmarkAgent(checkpoint_path, seed=seed)

def _make_clw1_agents(seed: int = 42, gru_checkpoint: str = '') -> dict:
    from .clw1.baselines import RandomAgent, QLearnerAgent, OracleBayesianAgent
    agents = {
        'Random': RandomAgent(seed=seed),
        'Q-Learner': QLearnerAgent(seed=seed),
        'Oracle': OracleBayesianAgent(seed=seed),
    }
    gru = _make_gru_agent(gru_checkpoint, seed)
    if gru is not None:
        agents['GRU'] = gru
    return agents

def _make_clw2_agents(seed: int = 42, gru_checkpoint: str = '') -> dict:
    from .clw2.baselines import RandomAgent, QLearnerAgent, OracleBayesianAgent
    agents = {
        'Random': RandomAgent(seed=seed),
        'Q-Learner': QLearnerAgent(seed=seed),
        'Oracle': OracleBayesianAgent(seed=seed),
    }
    gru = _make_gru_agent(gru_checkpoint, seed)
    if gru is not None:
        agents['GRU'] = gru
    return agents

def _make_clw3_agents(seed: int = 42, gru_checkpoint: str = '') -> dict:
    from .clw3.baselines import RandomAgent, QLearnerAgent, OracleBayesianAgent
    agents = {
        'Random': RandomAgent(seed=seed),
        'Q-Learner': QLearnerAgent(seed=seed),
        'Oracle': OracleBayesianAgent(seed=seed),
    }
    gru = _make_gru_agent(gru_checkpoint, seed)
    if gru is not None:
        agents['GRU'] = gru
    return agents


# ── Per-environment evaluators ───────────────────────────────────────────

def _evaluate_clw1(agent, protocol=None) -> ScoringMatrix:
    from .clw1.evaluate import evaluate_agent
    from .clw1.interventions import CLW1InterventionProtocol
    if protocol is None:
        protocol = CLW1InterventionProtocol()
    return evaluate_agent(agent, protocol)

def _evaluate_clw2(agent, protocol=None) -> ScoringMatrix:
    from .clw2.evaluate import evaluate_agent
    from .clw2.interventions import CLW2InterventionProtocol
    if protocol is None:
        protocol = CLW2InterventionProtocol()
    return evaluate_agent(agent, protocol)

def _evaluate_clw3(agent, protocol=None) -> ScoringMatrix:
    from .clw3.evaluate import evaluate_agent
    from .clw3.interventions import CLW3InterventionProtocol
    if protocol is None:
        protocol = CLW3InterventionProtocol()
    return evaluate_agent(agent, protocol)


# ── Registry ─────────────────────────────────────────────────────────────

ENV_REGISTRY = {
    'CLW-1': {'agents': _make_clw1_agents, 'evaluate': _evaluate_clw1},
    'CLW-2': {'agents': _make_clw2_agents, 'evaluate': _evaluate_clw2},
    'CLW-3': {'agents': _make_clw3_agents, 'evaluate': _evaluate_clw3},
}

AGENT_NAMES = ['Random', 'Q-Learner', 'Oracle']


# ── Combined table formatter ────────────────────────────────────────────

def format_combined_table(
    results: Dict[str, Dict[str, ScoringMatrix]],
    envs: List[str],
) -> str:
    """
    Format a combined results table: agents × (environments × test types).

    results[agent_name][env_name] = ScoringMatrix
    """
    lines = []
    tests = TEST_TYPES

    # Header
    env_col_w = 18
    test_col_w = 12
    agent_col_w = 14

    header1 = f"{'Agent':<{agent_col_w}} {'Test':<{test_col_w}}"
    for env in envs:
        header1 += f" | {env:^{env_col_w}}"
    sep = '─' * len(header1)

    lines.append(sep)
    lines.append(header1)
    lines.append(sep)

    for agent_name in sorted(results.keys()):
        first_row = True
        for test in tests:
            row = f"{agent_name if first_row else '':<{agent_col_w}} {test:<{test_col_w}}"
            for env in envs:
                matrix = results[agent_name].get(env)
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
                row += f" | {cell:^{env_col_w}}"
            lines.append(row)
            first_row = False
        lines.append(sep)

    return '\n'.join(lines)


# ── Main ─────────────────────────────────────────────────────────────────

def run_benchmark(
    envs: List[str] = None,
    agent_filter: List[str] = None,
    seed: int = 42,
    verbose: bool = True,
    gru_checkpoint: str = '',
) -> Dict[str, Dict[str, ScoringMatrix]]:
    """
    Run the full CLW benchmark.

    Args:
        envs: list of environment names to evaluate (default: all)
        agent_filter: list of agent names to include (default: all)
        seed: random seed for reproducibility
        verbose: print progress

    Returns:
        results[agent_name][env_name] = ScoringMatrix
    """
    if envs is None:
        envs = list(ENV_REGISTRY.keys())

    results: Dict[str, Dict[str, ScoringMatrix]] = {}
    total_start = time.time()

    for env_name in envs:
        if env_name not in ENV_REGISTRY:
            print(f"  ⚠ Unknown environment: {env_name}, skipping.")
            continue

        reg = ENV_REGISTRY[env_name]
        agents = reg['agents'](seed=seed, gru_checkpoint=gru_checkpoint)
        evaluator = reg['evaluate']

        for agent_name, agent in agents.items():
            if agent_filter and agent_name.lower() not in [a.lower() for a in agent_filter]:
                continue

            if agent_name not in results:
                results[agent_name] = {}

            t0 = time.time()
            if verbose:
                print(f"  Evaluating {agent_name:14s} on {env_name}...", end='', flush=True)

            matrix = evaluator(agent)
            elapsed = time.time() - t0

            results[agent_name][env_name] = matrix

            if verbose:
                print(f" done ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start

    if verbose:
        print(f"\n{'='*72}")
        print(f"  CLW Benchmark Results  (total: {total_elapsed:.1f}s)")
        print(f"{'='*72}\n")
        print(format_combined_table(results, envs))
        print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Causal Learning Benchmark — evaluate agents across CLW environments',
    )
    parser.add_argument(
        '--envs', nargs='+', default=None,
        choices=list(ENV_REGISTRY.keys()),
        help='Environments to evaluate (default: all)',
    )
    parser.add_argument(
        '--agents', nargs='+', default=None,
        help='Agent names to include (default: all)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output',
    )
    parser.add_argument(
        '--checkpoint', type=str, default='checkpoints/causal_belief_v2_final.pt',
        help='Path to GRU model checkpoint (default: checkpoints/causal_belief_v2_final.pt)',
    )
    args = parser.parse_args()

    run_benchmark(
        envs=args.envs,
        agent_filter=args.agents,
        seed=args.seed,
        verbose=not args.quiet,
        gru_checkpoint=args.checkpoint,
    )


if __name__ == '__main__':
    main()
