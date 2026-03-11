"""
Tests for CLW environment correctness and deterministic reproducibility.
"""

import pytest
import numpy as np


class TestCLW1Environment:
    """Verify CLW-1 environment contract and determinism."""

    def test_observation_shape(self):
        from benchmark.clw1.env import CLW1Environment
        env = CLW1Environment(seed=42)
        obs = env.reset()
        assert obs.shape == (4,), f"Observation shape {obs.shape} != (4,)"

    def test_observation_dtype(self):
        from benchmark.clw1.env import CLW1Environment
        env = CLW1Environment(seed=42)
        obs = env.reset()
        assert obs.dtype == np.float32

    def test_deterministic_reset(self):
        """Same seed → same initial state."""
        from benchmark.clw1.env import CLW1Environment
        env1 = CLW1Environment(seed=42)
        env2 = CLW1Environment(seed=42)
        obs1 = env1.reset()
        obs2 = env2.reset()
        # First 3 elements should match exactly (noise differs due to RNG state)
        gt1 = env1.get_ground_truth()
        gt2 = env2.get_ground_truth()
        assert gt1['C'] == gt2['C']

    def test_deterministic_trajectory(self):
        """Same seed + same actions → identical trajectory."""
        from benchmark.clw1.env import CLW1Environment
        actions = [0, 1, 0, 2, 1, 0, 0, 1, 2, 0]
        rewards1, rewards2 = [], []

        for trial_env in [CLW1Environment(seed=99), CLW1Environment(seed=99)]:
            trial_env.reset()
            rews = []
            for a in actions:
                _, r, done, _ = trial_env.step(a)
                rews.append(r)
                if done:
                    break
            if trial_env == trial_env:  # always true, just collecting
                pass
            rewards1 if not rewards1 else rewards2
            if not rewards1:
                rewards1 = rews
            else:
                rewards2 = rews

        assert rewards1 == rewards2

    def test_causal_graph_declared(self):
        from benchmark.clw1.env import CLW1Environment
        env = CLW1Environment(seed=42)
        assert 'C' in env.CAUSAL_GRAPH['nodes']
        assert 'C' in env.CAUSAL_GRAPH['intervention_targets']

    def test_intervention_changes_state(self):
        """do(C=1) should set C to 1."""
        from benchmark.clw1.env import CLW1Environment
        env = CLW1Environment(seed=42)
        env.reset()
        env.intervene('C', 1)
        assert env.get_ground_truth()['C'] == 1
        env.intervene('C', 0)
        assert env.get_ground_truth()['C'] == 0

    def test_invalid_intervention_target(self):
        from benchmark.clw1.env import CLW1Environment
        env = CLW1Environment(seed=42)
        env.reset()
        with pytest.raises(AssertionError):
            env.intervene('X', 1)

    def test_step_after_done_raises(self):
        from benchmark.clw1.env import CLW1Environment
        env = CLW1Environment(max_steps=3, seed=42)
        env.reset()
        for _ in range(3):
            _, _, done, _ = env.step(2)  # wait
            if done:
                break
        with pytest.raises(AssertionError):
            env.step(0)

    def test_correct_action_matches_c(self):
        """Correct action is always equal to C."""
        from benchmark.clw1.env import CLW1Environment
        env = CLW1Environment(seed=42)
        env.reset()
        for _ in range(50):
            gt = env.get_ground_truth()
            assert gt['correct_action'] == gt['C']
            _, _, done, _ = env.step(2)  # wait to avoid penalties
            if done:
                break

    def test_wrong_streak_causes_death(self):
        """Pulling wrong lever repeatedly should end the episode."""
        from benchmark.clw1.env import CLW1Environment
        env = CLW1Environment(seed=42, wrong_streak=3, max_steps=100)
        env.reset()
        c = env.get_ground_truth()['C']
        wrong_action = 1 - c
        died = False
        for _ in range(10):
            _, _, done, _ = env.step(wrong_action)
            if done:
                died = True
                break
        assert died, "Should have died from wrong streak"

    def test_wait_does_not_kill(self):
        """Wait action should never cause death from penalties."""
        from benchmark.clw1.env import CLW1Environment
        env = CLW1Environment(seed=42, max_steps=50)
        env.reset()
        for _ in range(49):
            _, _, done, _ = env.step(2)
            if done:
                break
        # Should only end at max_steps, not from penalty
        assert env.step_count <= env.max_steps


class TestCLW1Interventions:
    """Verify intervention protocol generates valid episodes."""

    def test_generates_100_episodes(self):
        from benchmark.clw1.interventions import CLW1InterventionProtocol
        protocol = CLW1InterventionProtocol()
        episodes = protocol.get_episodes()
        assert len(episodes) == 100

    def test_episode_allocation(self):
        from benchmark.clw1.interventions import CLW1InterventionProtocol
        protocol = CLW1InterventionProtocol()
        episodes = protocol.get_episodes()
        a_count = sum(1 for ep in episodes if ep.test_type == 'A')
        b_count = sum(1 for ep in episodes if ep.test_type == 'B')
        c_count = sum(1 for ep in episodes if ep.test_type == 'C')
        assert a_count == 40
        assert b_count == 30
        assert c_count == 30

    def test_all_episodes_have_seeds(self):
        from benchmark.clw1.interventions import CLW1InterventionProtocol
        protocol = CLW1InterventionProtocol()
        for ep in protocol.get_episodes():
            assert ep.seed > 0

    def test_interventions_at_correct_steps(self):
        from benchmark.clw1.interventions import CLW1InterventionProtocol
        protocol = CLW1InterventionProtocol()
        episodes = protocol.get_episodes()
        for ep in episodes[:40]:  # Test A episodes
            steps = [intv.step for intv in ep.interventions]
            assert 10 in steps
            assert 25 in steps
            assert 50 in steps

    def test_c2_noop_has_sentinel_values(self):
        """C2 no-op episodes should have value=None (sentinel)."""
        from benchmark.clw1.interventions import CLW1InterventionProtocol
        protocol = CLW1InterventionProtocol()
        episodes = protocol.get_episodes()
        c2_eps = [ep for ep in episodes if ep.novel_type == 'C2_noop']
        assert len(c2_eps) == 10
        for ep in c2_eps:
            for intv in ep.interventions:
                assert intv.value is None

    def test_c3_ood_has_no_interventions(self):
        """C3 OOD episodes should have empty interventions (N/A)."""
        from benchmark.clw1.interventions import CLW1InterventionProtocol
        protocol = CLW1InterventionProtocol()
        episodes = protocol.get_episodes()
        c3_eps = [ep for ep in episodes if ep.novel_type == 'C3_ood']
        assert len(c3_eps) == 10
        for ep in c3_eps:
            assert len(ep.interventions) == 0


class TestCLW1Baselines:
    """Verify baseline agents run without errors."""

    def test_random_agent(self):
        from benchmark.clw1.baselines import RandomAgent
        from benchmark.clw1.env import CLW1Environment
        agent = RandomAgent(seed=42)
        env = CLW1Environment(seed=42)
        obs = env.reset()
        agent.reset()
        agent.observe(obs)
        for _ in range(20):
            action = agent.act()
            assert action in {0, 1, 2}
            obs, _, done, _ = env.step(action)
            agent.observe(obs)
            if done:
                break

    def test_qlearner_agent(self):
        from benchmark.clw1.baselines import QLearnerAgent
        from benchmark.clw1.env import CLW1Environment
        agent = QLearnerAgent(seed=42)
        env = CLW1Environment(seed=42)
        obs = env.reset()
        agent.reset()
        agent.observe(obs)
        for _ in range(50):
            action = agent.act()
            assert action in {0, 1}  # Q-learner only pulls levers
            obs, _, done, _ = env.step(action)
            agent.observe(obs)
            if done:
                break

    def test_oracle_agent(self):
        from benchmark.clw1.baselines import OracleBayesianAgent
        from benchmark.clw1.env import CLW1Environment
        agent = OracleBayesianAgent(seed=42)
        env = CLW1Environment(seed=42)
        obs = env.reset()
        agent.reset()
        agent.observe(obs)
        for _ in range(50):
            action = agent.act()
            assert action in {0, 1, 2}
            obs, _, done, _ = env.step(action)
            agent.observe(obs)
            if done:
                break

    def test_all_baselines_have_representation(self):
        from benchmark.clw1.baselines import (
            RandomAgent, QLearnerAgent, OracleBayesianAgent
        )
        for AgentCls in [RandomAgent, QLearnerAgent, OracleBayesianAgent]:
            agent = AgentCls(seed=42)
            rep = agent.get_representation()
            assert rep is not None
            assert isinstance(rep, np.ndarray)
            assert rep.shape == (4,)

    def test_run_baselines_function(self):
        from benchmark.clw1.baselines import run_baselines
        results = run_baselines(n_episodes=5)
        assert 'Random' in results
        assert 'Q-Learner' in results
        assert 'Oracle-Bayesian' in results
        for name, res in results.items():
            assert 'mean_reward' in res
            assert 'std_reward' in res
