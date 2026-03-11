"""
CLW-1: Baseline Agents
=======================
Four reference agents for the single-confounder environment.
Each implements BenchmarkAgent for plug-and-play evaluation.

  1. RandomAgent        — uniform random, defines Level 0 floor
  2. QLearnerAgent      — associative Q-learning, no hidden state tracking
  3. OracleBayesianAgent — full Bayesian tracker with known causal graph
  4. (User's own model)  — plugged in via the BenchmarkAgent protocol
"""

from typing import Optional
import numpy as np


class RandomAgent:
    """
    Uniform random policy. Defines the Level 0 (chance) floor.

    Representation: constant zero vector (no internal state).
    """

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
    Tabular Q-learner with no hidden state tracking.

    Learns action values from rewards alone. No belief about C.
    Defines what Level 1 looks like without any causal structure —
    it can eventually adapt its behavior but does so slowly through
    trial-and-error, not through causal inference.

    Representation: [Q(0), Q(1), last_outcome, epsilon].
    """

    def __init__(
        self,
        alpha: float = 0.1,
        epsilon: float = 0.15,
        seed: Optional[int] = None,
    ):
        self._alpha = alpha
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)

        self._q = np.zeros(2, dtype=np.float64)
        self._last_action: Optional[int] = None
        self._last_outcome: float = 0.0

    def reset(self) -> None:
        self._q[:] = 0.0
        self._last_action = None
        self._last_outcome = 0.0

    def observe(self, obs: np.ndarray) -> None:
        """Update Q-values from the observation."""
        outcome = obs[1]  # last_outcome

        if self._last_action is not None and self._last_action < 2:
            # Simple reward-based Q update
            reward_signal = 1.0 if outcome > 0.5 else -0.5
            self._q[self._last_action] += self._alpha * (
                reward_signal - self._q[self._last_action]
            )

        self._last_outcome = outcome

    def act(self) -> int:
        if self._rng.rand() < self._epsilon:
            return self._rng.randint(0, 2)  # explore (only levers, not wait)
        return int(np.argmax(self._q))

    def get_representation(self) -> Optional[np.ndarray]:
        return np.array([
            self._q[0], self._q[1],
            self._last_outcome, self._epsilon
        ], dtype=np.float32)


class OracleBayesianAgent:
    """
    Full Bayesian tracker with the ground-truth causal graph GIVEN.

    Maintains a posterior belief over C ∈ {0, 1}, updated exactly
    using Bayes' rule after each lever pull outcome. This is the
    theoretical ceiling — what perfect causal reasoning achieves.

    It knows:
      - C determines which lever is correct
      - Pull outcomes are deterministic: correct=1.0, wrong=0.0
      - C can flip (geometric distribution, p ≈ 1/80)

    Representation: [P(C=0), P(C=1), belief_entropy, confidence].
    """

    def __init__(
        self,
        flip_prob: float = 1.0 / 80,
        obs_reliability: float = 1.0,
        seed: Optional[int] = None,
    ):
        self._flip_prob = flip_prob
        self._obs_reliability = obs_reliability
        self._rng = np.random.RandomState(seed)

        self._belief_c0: float = 0.5  # P(C=0)
        self._last_action: Optional[int] = None

    def reset(self) -> None:
        self._belief_c0 = 0.5
        self._last_action = None

    def observe(self, obs: np.ndarray) -> None:
        """Bayesian update of belief over C."""
        outcome = obs[1]  # 1.0 if last pull was correct

        # Transition prior: C might have flipped
        p0 = self._belief_c0 * (1 - self._flip_prob) + \
             (1 - self._belief_c0) * self._flip_prob

        # Likelihood update if we pulled a lever
        if self._last_action is not None and self._last_action < 2:
            is_correct = outcome > 0.5

            if self._last_action == 0:
                # If action=0 was correct, C is likely 0
                p_correct_if_c0 = self._obs_reliability
                p_correct_if_c1 = 1.0 - self._obs_reliability
            else:
                # If action=1 was correct, C is likely 1
                p_correct_if_c0 = 1.0 - self._obs_reliability
                p_correct_if_c1 = self._obs_reliability

            if is_correct:
                l0 = p_correct_if_c0
                l1 = p_correct_if_c1
            else:
                l0 = 1.0 - p_correct_if_c0
                l1 = 1.0 - p_correct_if_c1

            # Clamp likelihoods to avoid zero (for numerical stability)
            l0 = max(l0, 1e-6)
            l1 = max(l1, 1e-6)

            # Bayes update
            numerator = p0 * l0
            denominator = p0 * l0 + (1 - p0) * l1
            if denominator > 1e-10:
                p0 = numerator / denominator

        self._belief_c0 = np.clip(p0, 0.001, 0.999)

    def act(self) -> int:
        """
        Act according to belief.
        
        When confident: exploit (pull the lever matching the MAP estimate of C).
        When uncertain: explore (pull a random lever to gather information).
        Never wait — waiting provides zero information about C.
        """
        confidence = abs(self._belief_c0 - 0.5) * 2  # 0=uncertain, 1=certain

        if confidence < 0.3:
            # Uncertain: explore by pulling a random lever.
            # This is critical — waiting gives no outcome feedback,
            # so belief would remain stuck at 0.5 forever.
            action = self._rng.randint(0, 2)
        else:
            # Confident: exploit the MAP estimate
            action = 0 if self._belief_c0 > 0.5 else 1

        self._last_action = action
        return action

    def get_representation(self) -> Optional[np.ndarray]:
        p0 = self._belief_c0
        p1 = 1.0 - p0
        entropy = -p0 * np.log2(max(p0, 1e-10)) - p1 * np.log2(max(p1, 1e-10))
        confidence = abs(p0 - 0.5) * 2  # 0=uncertain, 1=certain
        return np.array([p0, p1, entropy, confidence], dtype=np.float32)


def run_baselines(n_episodes: int = 40) -> dict:
    """
    Run all baselines and return a summary dict.

    This is a convenience function for quick verification.
    Full evaluation should use CLW1Evaluator for comparable scoring.
    """
    from .env import CLW1Environment

    results = {}
    agents = {
        'Random': RandomAgent(seed=42),
        'Q-Learner': QLearnerAgent(seed=42),
        'Oracle-Bayesian': OracleBayesianAgent(seed=42),
    }

    for name, agent in agents.items():
        total_rewards = []
        for ep in range(n_episodes):
            env = CLW1Environment(seed=ep * 17 + 7)
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
