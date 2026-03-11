"""
Baseline Score Reproducibility Tests
=====================================
Verifies that baseline agents produce deterministic, reproducible scores
across runs with the same seed. This is the gold standard for benchmark
integrity: if scores drift between runs, comparisons are meaningless.
"""

import pytest
import numpy as np

from benchmark.clw1.baselines import (
    RandomAgent as CLW1Random,
    QLearnerAgent as CLW1QLearner,
    OracleBayesianAgent as CLW1Oracle,
)
from benchmark.clw1.evaluate import evaluate_agent as evaluate_clw1
from benchmark.clw1.interventions import CLW1InterventionProtocol

from benchmark.clw2.baselines import (
    RandomAgent as CLW2Random,
    QLearnerAgent as CLW2QLearner,
    OracleBayesianAgent as CLW2Oracle,
)
from benchmark.clw2.evaluate import evaluate_agent as evaluate_clw2
from benchmark.clw2.interventions import CLW2InterventionProtocol

from benchmark.clw3.baselines import (
    RandomAgent as CLW3Random,
    QLearnerAgent as CLW3QLearner,
    OracleBayesianAgent as CLW3Oracle,
)
from benchmark.clw3.evaluate import evaluate_agent as evaluate_clw3
from benchmark.clw3.interventions import CLW3InterventionProtocol


SEED = 42


def _score_value(matrix, env, test_type):
    """Extract raw value from a ScoringMatrix cell."""
    score = matrix.get_score(env, test_type)
    if score is None:
        return None
    return score.value


class TestCLW1Reproducibility:
    """CLW-1 baselines must produce identical scores across two runs."""

    def _run_agent(self, agent_cls, **kwargs):
        protocol = CLW1InterventionProtocol()
        agent = agent_cls(seed=SEED, **kwargs)
        return evaluate_clw1(agent, protocol)

    def test_random_deterministic(self):
        m1 = self._run_agent(CLW1Random)
        m2 = self._run_agent(CLW1Random)
        for test in ['A', 'B-full', 'B-proxy', 'C1', 'C2']:
            v1 = _score_value(m1, 'CLW-1', test)
            v2 = _score_value(m2, 'CLW-1', test)
            if v1 is not None and v2 is not None:
                assert abs(v1 - v2) < 0.01, f"CLW-1 Random {test}: {v1} != {v2}"

    def test_qlearner_deterministic(self):
        m1 = self._run_agent(CLW1QLearner)
        m2 = self._run_agent(CLW1QLearner)
        for test in ['A', 'B-full', 'B-proxy', 'C1', 'C2']:
            v1 = _score_value(m1, 'CLW-1', test)
            v2 = _score_value(m2, 'CLW-1', test)
            if v1 is not None and v2 is not None:
                assert abs(v1 - v2) < 0.01, f"CLW-1 QLearner {test}: {v1} != {v2}"

    def test_oracle_deterministic(self):
        m1 = self._run_agent(CLW1Oracle)
        m2 = self._run_agent(CLW1Oracle)
        for test in ['A', 'B-full', 'B-proxy', 'C1', 'C2']:
            v1 = _score_value(m1, 'CLW-1', test)
            v2 = _score_value(m2, 'CLW-1', test)
            if v1 is not None and v2 is not None:
                assert abs(v1 - v2) < 0.01, f"CLW-1 Oracle {test}: {v1} != {v2}"

    def test_oracle_has_stable_b_score(self):
        """Oracle B-full score should be deterministic (same each run)."""
        m1 = self._run_agent(CLW1Oracle)
        m2 = self._run_agent(CLW1Oracle)
        v1 = _score_value(m1, 'CLW-1', 'B-full')
        v2 = _score_value(m2, 'CLW-1', 'B-full')
        if v1 is not None and v2 is not None:
            assert abs(v1 - v2) < 0.01, f"CLW-1 Oracle B-full: {v1} != {v2}"


class TestCLW2Reproducibility:
    """CLW-2 baselines must produce identical scores across two runs."""

    def _run_agent(self, agent_cls, **kwargs):
        protocol = CLW2InterventionProtocol()
        agent = agent_cls(seed=SEED, **kwargs)
        return evaluate_clw2(agent, protocol)

    def test_random_deterministic(self):
        m1 = self._run_agent(CLW2Random)
        m2 = self._run_agent(CLW2Random)
        for test in ['A', 'B-full', 'B-proxy', 'C1', 'C2']:
            v1 = _score_value(m1, 'CLW-2', test)
            v2 = _score_value(m2, 'CLW-2', test)
            if v1 is not None and v2 is not None:
                assert abs(v1 - v2) < 0.01, f"CLW-2 Random {test}: {v1} != {v2}"

    def test_oracle_deterministic(self):
        m1 = self._run_agent(CLW2Oracle)
        m2 = self._run_agent(CLW2Oracle)
        for test in ['A', 'B-full', 'B-proxy', 'C1', 'C2']:
            v1 = _score_value(m1, 'CLW-2', test)
            v2 = _score_value(m2, 'CLW-2', test)
            if v1 is not None and v2 is not None:
                assert abs(v1 - v2) < 0.01, f"CLW-2 Oracle {test}: {v1} != {v2}"


class TestCLW3Reproducibility:
    """CLW-3 baselines must produce identical scores across two runs."""

    def _run_agent(self, agent_cls, **kwargs):
        protocol = CLW3InterventionProtocol()
        agent = agent_cls(seed=SEED, **kwargs)
        return evaluate_clw3(agent, protocol)

    def test_random_deterministic(self):
        m1 = self._run_agent(CLW3Random)
        m2 = self._run_agent(CLW3Random)
        for test in ['A', 'B-full', 'B-proxy', 'C1', 'C2']:
            v1 = _score_value(m1, 'CLW-3', test)
            v2 = _score_value(m2, 'CLW-3', test)
            if v1 is not None and v2 is not None:
                assert abs(v1 - v2) < 0.01, f"CLW-3 Random {test}: {v1} != {v2}"

    def test_oracle_deterministic(self):
        m1 = self._run_agent(CLW3Oracle)
        m2 = self._run_agent(CLW3Oracle)
        for test in ['A', 'B-full', 'B-proxy', 'C1', 'C2']:
            v1 = _score_value(m1, 'CLW-3', test)
            v2 = _score_value(m2, 'CLW-3', test)
            if v1 is not None and v2 is not None:
                assert abs(v1 - v2) < 0.01, f"CLW-3 Oracle {test}: {v1} != {v2}"

    def test_oracle_achieves_level1_on_a(self):
        m = self._run_agent(CLW3Oracle)
        score = m.get_score('CLW-3', 'A')
        if score is not None:
            assert score.level >= 1, \
                f"CLW-3 Oracle should achieve L1+ on A, got L{score.level}"

    def test_oracle_achieves_level3_on_c1(self):
        m = self._run_agent(CLW3Oracle)
        score = m.get_score('CLW-3', 'C1')
        assert score is not None
        assert score.level == 3, f"Oracle should achieve L3 on C1, got L{score.level}"
