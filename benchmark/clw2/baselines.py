"""
CLW-2: Baseline Agents
=======================
RandomAgent, QLearnerAgent (adapted from CLW-1), and OracleBayesianAgent.

The OracleBayesianAgent perfectly tracks the 8-state HMM (Target, C1, C2)
using exact Bayesian inference, demonstrating the theoretical ceiling for
causal tracking and mediation understanding.
"""

from typing import Optional
import numpy as np


class RandomAgent:
    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)
        self._rep = np.zeros(4, dtype=np.float32)

    def reset(self) -> None: pass
    def observe(self, obs: np.ndarray) -> None: pass
    def act(self) -> int: return self._rng.randint(0, 3)
    def get_representation(self) -> Optional[np.ndarray]: return self._rep.copy()


class QLearnerAgent:
    """Associative learner. Learns to pull Target eventually, but fails Test B/C."""
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.15, seed: Optional[int] = None):
        self._alpha = alpha
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        self._q = np.zeros(2, dtype=np.float64)
        self._last_action = None

    def reset(self) -> None:
        self._q[:] = 0.0
        self._last_action = None

    def observe(self, obs: np.ndarray) -> None:
        outcome = obs[1]
        if self._last_action is not None and self._last_action < 2:
            reward = 1.0 if outcome > 0.5 else -0.5
            self._q[self._last_action] += self._alpha * (reward - self._q[self._last_action])

    def act(self) -> int:
        if self._rng.rand() < self._epsilon:
            self._last_action = self._rng.randint(0, 2)
        else:
            self._last_action = int(np.argmax(self._q))
        return self._last_action

    def get_representation(self) -> Optional[np.ndarray]:
        # Q-learner has no idea about Target, C1, or C2
        return np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)


class OracleBayesianAgent:
    """
    Exact Bayesian tracker for the 8-state (Target, C1, C2) HMM.
    Uses a simple dictionary to track the 8 states, which is fast and bug-free.
    """
    def __init__(self, flip_prob=1/80, control_p=0.8, c1_c2_p=0.8, seed: Optional[int]=None):
        self._flip_prob = flip_prob
        self._control_p = control_p
        self._c1_c2_p = c1_c2_p
        self._rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> None:
        self._belief = {}
        for t in (0, 1):
            for c1 in (0, 1):
                for c2 in (0, 1):
                    self._belief[(t, c1, c2)] = 1.0 / 8.0
        self._last_action = None

    def observe(self, obs: np.ndarray) -> None:
        outcome = True if obs[1] > 0.5 else False
        
        # 1. Transition update
        new_belief = {(t, c1, c2): 0.0 for t in (0, 1) for c1 in (0, 1) for c2 in (0, 1)}
        
        for (prev_T, prev_c1, prev_c2), p_prev in self._belief.items():
            if p_prev < 1e-9: continue
            
            # Target transitions
            for next_T in (0, 1):
                p_T = (1 - self._flip_prob) if next_T == prev_T else self._flip_prob
                
                if self._last_action is None or self._last_action == 2:
                    # Wait: C1 stays, C2 stays
                    new_belief[(next_T, prev_c1, prev_c2)] += p_prev * p_T
                else:
                    action = self._last_action
                    if action == prev_c1:
                        # C1 stays, C2 stays
                        new_belief[(next_T, prev_c1, prev_c2)] += p_prev * p_T
                    else:
                        # Action != C1.
                        for next_c1 in (0, 1):
                            p_c1 = self._control_p if next_c1 == action else (1 - self._control_p)
                            c1_changed = (next_c1 != prev_c1)
                            
                            for next_c2 in (0, 1):
                                if c1_changed:
                                    p_c2 = self._c1_c2_p if next_c2 == next_c1 else (1 - self._c1_c2_p)
                                else:
                                    p_c2 = 1.0 if next_c2 == prev_c2 else 0.0
                                    
                                new_belief[(next_T, next_c1, next_c2)] += p_prev * p_T * p_c1 * p_c2

        # 2. Emission update
        if self._last_action is not None and self._last_action < 2:
            for (t, c1, c2) in new_belief.keys():
                matches = (c2 == t)
                prob_obs = 1.0 if matches == outcome else 0.0
                prob_obs = 0.99 if prob_obs > 0.5 else 0.01
                new_belief[(t, c1, c2)] *= prob_obs

        # Normalize
        s = sum(new_belief.values())
        if s > 1e-12:
            for k in new_belief:
                self._belief[k] = new_belief[k] / s
        else:
            self.reset()

    def act(self) -> int:
        p_T1 = sum(p for (t, c1, c2), p in self._belief.items() if t == 1)
        action = 1 if p_T1 >= 0.5 else 0
        self._last_action = action
        return action

    def get_representation(self) -> Optional[np.ndarray]:
        p_T1 = sum(p for (t, c1, c2), p in self._belief.items() if t == 1)
        p_C1_1 = sum(p for (t, c1, c2), p in self._belief.items() if c1 == 1)
        p_C2_1 = sum(p for (t, c1, c2), p in self._belief.items() if c2 == 1)
        
        confidence = abs(p_T1 - 0.5) * 2
        return np.array([p_T1, p_C1_1, p_C2_1, confidence], dtype=np.float32)


def run_baselines(n_episodes: int = 20) -> dict:
    from .env import CLW2Environment
    results = {}
    agents = {
        'Random': RandomAgent(seed=42),
        'Q-Learner': QLearnerAgent(seed=42),
        'Oracle-Bayesian': OracleBayesianAgent(seed=42),
    }

    for name, agent in agents.items():
        total_rewards = []
        for ep in range(n_episodes):
            env = CLW2Environment(seed=ep * 17 + 7)
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
