"""Tests for CLW-3 environment, interventions, and baselines."""
import pytest
import numpy as np
from benchmark.clw3.env import CLW3Environment
from benchmark.clw3.interventions import CLW3InterventionProtocol
from benchmark.clw3.baselines import RandomAgent, QLearnerAgent, OracleBayesianAgent
from benchmark.clw3.evaluate import evaluate_agent


class TestCLW3Environment:
    def test_initialization(self):
        env = CLW3Environment(seed=42)
        obs = env.reset()
        assert obs.shape == (4,)
        gt = env.get_ground_truth()
        assert 'C' in gt
        assert 'S1' in gt
        assert 'S2' in gt

    def test_determinism(self):
        """Same seed → same trajectory."""
        env1 = CLW3Environment(seed=42)
        env2 = CLW3Environment(seed=42)
        obs1 = env1.reset()
        obs2 = env2.reset()
        np.testing.assert_array_equal(obs1, obs2)
        
        for _ in range(20):
            obs1, r1, d1, i1 = env1.step(0)
            obs2, r2, d2, i2 = env2.step(0)
            np.testing.assert_array_equal(obs1, obs2)
            assert r1 == r2
            assert d1 == d2

    def test_intervention_s1(self):
        """do(S1=1) should pin S1 to 1 regardless of C."""
        env = CLW3Environment(seed=42)
        env.reset()
        env.intervene('S1', 1)
        
        # Step a few times — S1 should always be 1
        for _ in range(10):
            obs, r, d, info = env.step(2)  # wait
            assert info['S1'] == 1
            if d:
                break

    def test_intervention_s2(self):
        """do(S2=0) should pin S2 to 0."""
        env = CLW3Environment(seed=42)
        env.reset()
        env.intervene('S2', 0)
        
        for _ in range(10):
            obs, r, d, info = env.step(2)
            assert info['S2'] == 0
            if d:
                break

    def test_intervention_c(self):
        """do(C=1) should set C directly."""
        env = CLW3Environment(seed=42)
        env.reset()
        env.intervene('C', 1)
        gt = env.get_ground_truth()
        assert gt['C'] == 1

    def test_s1_intervention_does_not_change_c(self):
        """do(S1) must NOT affect C."""
        env = CLW3Environment(seed=42)
        env.reset()
        gt_before = env.get_ground_truth()
        c_before = gt_before['C']
        
        env.intervene('S1', 1 - c_before)  # pin S1 to opposite of C
        gt_after = env.get_ground_truth()
        assert gt_after['C'] == c_before  # C unchanged

    def test_obs_contains_sensors(self):
        """CLW-3 obs should expose S1 and S2."""
        env = CLW3Environment(seed=42)
        obs = env.reset()
        gt = env.get_ground_truth()
        assert obs[0] == float(gt['S1'])
        assert obs[1] == float(gt['S2'])


class TestCLW3InterventionProtocol:
    def test_episode_count(self):
        protocol = CLW3InterventionProtocol()
        episodes = protocol.get_episodes()
        assert len(episodes) == 100

    def test_episode_allocation(self):
        protocol = CLW3InterventionProtocol()
        episodes = protocol.get_episodes()
        a_count = sum(1 for e in episodes if e.test_type == 'A')
        b_count = sum(1 for e in episodes if e.test_type == 'B')
        c_count = sum(1 for e in episodes if e.test_type == 'C')
        assert a_count == 40
        assert b_count == 30
        assert c_count == 30


class TestCLW3Baselines:
    def test_random_agent_runs(self):
        env = CLW3Environment(seed=42)
        agent = RandomAgent(seed=42)
        obs = env.reset()
        agent.reset()
        agent.observe(obs)
        for _ in range(20):
            a = agent.act()
            obs, r, d, info = env.step(a)
            agent.observe(obs)
            if d:
                break

    def test_oracle_representation(self):
        agent = OracleBayesianAgent(seed=42)
        agent.reset()
        # Observe S1=1, S2=1 → should update P(C=1) upward
        obs = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        agent.observe(obs)
        rep = agent.get_representation()
        assert rep[0] > 0.5  # P(C=1) should be > 0.5

    def test_oracle_tracks_c(self):
        """Oracle should correctly track C over multiple steps."""
        env = CLW3Environment(seed=42)
        agent = OracleBayesianAgent(seed=42)
        obs = env.reset()
        agent.reset()
        agent.observe(obs)
        
        correct = 0
        total = 0
        for _ in range(50):
            action = agent.act()
            obs, r, d, info = env.step(action)
            agent.observe(obs)
            if action < 2:
                if action == info['C']:
                    correct += 1
                total += 1
            if d:
                break
        
        # Oracle should get most actions correct
        if total > 0:
            assert correct / total > 0.5


class TestCLW3Evaluate:
    def test_evaluate_runs(self):
        """Evaluation should complete without errors."""
        agent = OracleBayesianAgent(seed=42)
        protocol = CLW3InterventionProtocol()
        # Use only a few episodes for speed
        episodes = protocol.get_episodes()
        fast_eps = [
            next(e for e in episodes if e.test_type == 'A'),
            next(e for e in episodes if e.test_type == 'B'),
            next(e for e in episodes if e.test_type == 'C'),
        ]
        protocol.get_episodes = lambda: fast_eps
        
        matrix = evaluate_agent(agent, protocol)
        assert matrix is not None
        a_score = matrix.get_score('CLW-3', 'A')
        assert a_score is not None
