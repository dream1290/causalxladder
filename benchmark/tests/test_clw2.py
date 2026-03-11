import pytest
import numpy as np
from benchmark.clw2.env import CLW2Environment
from benchmark.clw2.interventions import CLW2InterventionProtocol
from benchmark.clw2.baselines import RandomAgent, QLearnerAgent, OracleBayesianAgent
from benchmark.clw2.evaluate import evaluate_agent


class TestCLW2Environment:
    def test_initialization(self):
        env = CLW2Environment(seed=42)
        obs = env.reset()
        assert obs.shape == (4,)
        gt = env.get_ground_truth()
        assert 'Target' in gt
        assert 'C1' in gt
        assert 'C2' in gt

    def test_inertia(self):
        env = CLW2Environment(seed=42)
        env.reset()
        gt_before = env.get_ground_truth()
        
        # Take action matching C1 -> C1 should NOT change, C2 should NOT change
        for _ in range(5):
            _, _, _, info = env.step(gt_before['C1'])
            gt_after = env.get_ground_truth()
            assert gt_after['C1'] == gt_before['C1']
            assert gt_after['C2'] == gt_before['C2']

    def test_interventions(self):
        env = CLW2Environment(seed=42)
        env.reset()
        
        # Test do(C1)
        env.intervene('C1', 1)
        gt = env.get_ground_truth()
        assert gt['C1'] == 1
        
        # Test do(C2)
        env.intervene('C2', 1)
        gt = env.get_ground_truth()
        assert gt['C2'] == 1
        
        env.intervene('C2', 0)
        gt = env.get_ground_truth()
        assert gt['C1'] == 1
        assert gt['C2'] == 0


class TestCLW2Evaluate:
    def test_evaluate_runs_without_crashing(self):
        # We just test Oracle to make sure it runs
        agent = OracleBayesianAgent(seed=42)
        protocol = CLW2InterventionProtocol()
        # Mock protocol to just 2 episodes to be fast
        episodes = protocol.get_episodes()
        # take 1 of A, 1 of B, 1 of C
        fast_eps = [
            next(e for e in episodes if e.test_type == 'A'),
            next(e for e in episodes if e.test_type == 'B'),
            next(e for e in episodes if e.test_type == 'C')
        ]
        protocol.get_episodes = lambda: fast_eps
        
        matrix = evaluate_agent(agent, protocol)
        assert matrix is not None
        
        a_score = matrix.get_score('CLW-2', 'A')
        assert a_score is not None
