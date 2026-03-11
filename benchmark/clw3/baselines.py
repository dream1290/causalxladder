"""
CLW-3: Baseline Agents
=======================
RandomAgent, QLearnerAgent, and OracleBayesianAgent for CLW-3.

The OracleBayesianAgent uses Bayesian inference to track P(C=1) from
noisy sensor observations S1 and S2, demonstrating optimal causal reasoning.
It knows that do(S1) breaks C→S1, so it ignores S1 when it's pinned.
"""

from typing import Optional
import numpy as np


class RandomAgent:
    """Level 0 floor — random actions, no learning."""
    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)
        self._rep = np.zeros(4, dtype=np.float32)

    def reset(self) -> None:
        pass

    def observe(self, obs: np.ndarray) -> None:
        pass

    def act(self) -> int:
        return self._rng.randint(0, 3)

    def get_representation(self) -> Optional[np.ndarray]:
        return self._rep.copy()


class QLearnerAgent:
    """
    Associative learner for CLW-3.
    
    Uses S1 and S2 from observations to estimate C, but treats them as 
    equally reliable (doesn't understand that do(S1) makes S1 unreliable).
    This will fail Test A/B/C when S1 is intervened upon.
    """
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.15,
                 seed: Optional[int] = None):
        self._alpha = alpha
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        # Q-values indexed by (s1, s2) state → 3 actions
        self._q = np.zeros((2, 2, 3), dtype=np.float64)
        self._last_obs = np.zeros(4, dtype=np.float32)
        self._last_action = None

    def reset(self) -> None:
        self._q[:] = 0.0
        self._last_action = None
        self._last_obs = np.zeros(4, dtype=np.float32)

    def observe(self, obs: np.ndarray) -> None:
        self._last_obs = obs.copy()
        # Update Q-value from outcome
        if self._last_action is not None and self._last_action < 2:
            s1 = int(obs[0] > 0.5)
            s2 = int(obs[1] > 0.5)
            # Simple outcome-based reward signal
            outcome = obs[1]  # in base_env this is last_outcome
            reward = 1.0 if outcome > 0.5 else -0.5
            old_q = self._q[s1, s2, self._last_action]
            self._q[s1, s2, self._last_action] = old_q + self._alpha * (reward - old_q)

    def act(self) -> int:
        s1 = int(self._last_obs[0] > 0.5)
        s2 = int(self._last_obs[1] > 0.5)
        if self._rng.rand() < self._epsilon:
            self._last_action = self._rng.randint(0, 2)
        else:
            self._last_action = int(np.argmax(self._q[s1, s2, :2]))
        return self._last_action

    def get_representation(self) -> Optional[np.ndarray]:
        # Q-learner has no belief about C; it just uses S1, S2 directly
        s1 = int(self._last_obs[0] > 0.5)
        s2 = int(self._last_obs[1] > 0.5)
        return np.array([0.5, float(s1), float(s2), 0.0], dtype=np.float32)


class OracleBayesianAgent:
    """
    Exact Bayesian tracker for the hidden cause C in CLW-3.
    
    Knows the true causal structure: C → S1, C → S2.
    Uses both sensor readings to maintain P(C=1) via Bayes' rule.
    
    Key: knows that S1 and S2 are noisy sensors of C with accuracy 0.8.
    The observation vector is [S1, S2, steps_norm, noise].
    """
    def __init__(
        self,
        sensor_acc: float = 0.8,
        flip_prob: float = 1.0 / 80,
        seed: Optional[int] = None,
    ):
        self._sensor_acc = sensor_acc
        self._flip_prob = flip_prob
        self._rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> None:
        self._p_C1 = 0.5  # P(C=1)
        self._last_action = None
        self._last_s1 = 0
        self._last_s2 = 0

    def observe(self, obs: np.ndarray) -> None:
        s1 = int(obs[0] > 0.5)
        s2 = int(obs[1] > 0.5)
        self._last_s1 = s1
        self._last_s2 = s2

        # 1. Transition prior: C can flip
        p_C1_prior = self._p_C1 * (1 - self._flip_prob) + (1 - self._p_C1) * self._flip_prob

        # 2. Likelihood of (S1, S2) given C
        # P(S1=s1 | C=1) * P(S2=s2 | C=1)
        acc = self._sensor_acc
        p_obs_given_C1 = (acc if s1 == 1 else 1 - acc) * (acc if s2 == 1 else 1 - acc)
        p_obs_given_C0 = (acc if s1 == 0 else 1 - acc) * (acc if s2 == 0 else 1 - acc)

        # 3. Bayes update
        numerator = p_obs_given_C1 * p_C1_prior
        denominator = numerator + p_obs_given_C0 * (1 - p_C1_prior)

        if denominator > 1e-12:
            self._p_C1 = numerator / denominator
        else:
            self._p_C1 = 0.5

    def act(self) -> int:
        # Optimal: play action = most likely C
        action = 1 if self._p_C1 >= 0.5 else 0
        self._last_action = action
        return action

    def get_representation(self) -> Optional[np.ndarray]:
        p_C1 = self._p_C1
        # Expected S1 = P(C=1)*acc + P(C=0)*(1-acc)
        acc = self._sensor_acc
        expected_s1 = p_C1 * acc + (1 - p_C1) * (1 - acc)
        expected_s2 = p_C1 * acc + (1 - p_C1) * (1 - acc)
        confidence = abs(p_C1 - 0.5) * 2
        return np.array([p_C1, expected_s1, expected_s2, confidence], dtype=np.float32)


def run_baselines(n_episodes: int = 20) -> dict:
    """Quick self-test: run all three baselines."""
    from .env import CLW3Environment
    
    results = {}
    agents = {
        'Random': RandomAgent(seed=42),
        'Q-Learner': QLearnerAgent(seed=42),
        'Oracle-Bayesian': OracleBayesianAgent(seed=42),
    }

    for name, agent in agents.items():
        total_rewards = []
        for ep in range(n_episodes):
            env = CLW3Environment(seed=ep * 17 + 7)
            obs = env.reset()
            agent.reset()
            agent.observe(obs)

            while not env.done:
                action = agent.act()
                obs, reward, done, info = env.step(action)
                agent.observe(obs)

            total_rewards.append(env.total_reward)

        results[name] = {
            'mean_reward': float(np.mean(total_rewards)),
            'std_reward': float(np.std(total_rewards)),
        }

    return results
