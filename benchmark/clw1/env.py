"""
CLW-1: Single Confounder Environment
=====================================
One hidden variable C ∈ {0, 1} determines the correct lever.
The agent never observes C directly.

Causal graph:
    C → correct_action → reward

C flips according to a geometric distribution (mean = flip_mean steps).
Pulling the correct lever builds a streak toward a big reward.
Pulling the wrong lever builds a streak toward a big penalty (and death).
Wait resets both streaks.

This is the baseline environment. If a system cannot pass CLW-1,
it cannot pass anything.
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np

from ..core.base_env import CLWEnvironment


class CLW1Environment(CLWEnvironment):
    """
    Single Confounder Lever World.

    Hidden state: C ∈ {0, 1}.
    Correct action: pull lever C.
    Actions: 0 (pull lever 0), 1 (pull lever 1), 2 (wait).
    """

    CAUSAL_GRAPH = {
        'nodes': ['C', 'correct_action', 'reward'],
        'edges': [('C', 'correct_action'), ('correct_action', 'reward')],
        'intervention_targets': ['C'],
        'observable': [],  # C is fully hidden
    }

    def __init__(
        self,
        max_steps: int = 200,
        flip_mean: int = 80,
        big_reward: float = 10.0,
        big_penalty: float = -30.0,
        step_cost: float = -0.01,
        pull_cost: float = -0.05,
        correct_streak: int = 8,
        wrong_streak: int = 6,
        seed: Optional[int] = None,
    ):
        self.flip_mean = flip_mean
        self.big_reward = big_reward
        self.big_penalty = big_penalty
        self.step_cost = step_cost
        self.pull_cost = pull_cost
        self.correct_streak = correct_streak
        self.wrong_streak = wrong_streak

        # State variables (initialised in _reset_state)
        self._C: int = 0
        self._flip_timer: int = 0
        self._correct_consec: int = 0
        self._wrong_consec: int = 0
        self._total_reward: float = 0.0

        super().__init__(max_steps=max_steps, seed=seed)

    def _reset_state(self) -> None:
        """Reset hidden state for a new episode."""
        self._C = self._rng.randint(0, 2)
        self._flip_timer = self._rng.geometric(1.0 / self.flip_mean)
        self._correct_consec = 0
        self._wrong_consec = 0
        self._total_reward = 0.0

    def _step_impl(self, action: int) -> Tuple[float, bool, Dict]:
        """
        Execute one step of CLW-1 dynamics.

        Returns (reward, done, info).
        """
        reward = self.step_cost
        done = False
        info = {'correct_pull': False, 'C': self._C}

        if action == 2:  # Wait
            self._correct_consec = 0
            self._wrong_consec = 0
            self._advance_timer()
            done = self._step_count + 1 >= self.max_steps
            self._total_reward += reward
            return reward, done, info

        # Pull a lever
        reward += self.pull_cost
        is_correct = (action == self._C)

        if is_correct:
            reward += 0.2
            self._correct_consec += 1
            self._wrong_consec = 0
            info['correct_pull'] = True
            if self._correct_consec >= self.correct_streak:
                reward += self.big_reward
                self._correct_consec = 0
        else:
            reward += -0.2
            self._wrong_consec += 1
            self._correct_consec = 0
            if self._wrong_consec >= self.wrong_streak:
                reward += self.big_penalty
                done = True

        self._advance_timer()
        self._total_reward += reward

        if self._step_count + 1 >= self.max_steps:
            done = True

        return reward, done, info

    def _advance_timer(self) -> None:
        """Advance the hidden state flip timer."""
        self._flip_timer -= 1
        if self._flip_timer <= 0:
            self._C = 1 - self._C
            self._flip_timer = self._rng.geometric(1.0 / self.flip_mean)

    def _apply_intervention(self, target: str, value: Any) -> None:
        """do(C = value): set hidden state directly."""
        assert target == 'C'
        assert value in (0, 1), f"C must be 0 or 1, got {value}"
        self._C = int(value)
        # Reset flip timer after intervention (new regime starts)
        self._flip_timer = self._rng.geometric(1.0 / self.flip_mean)

    def get_ground_truth(self) -> Dict[str, Any]:
        """Return the true hidden state."""
        return {
            'C': self._C,
            'correct_action': self._C,
            'flip_timer': self._flip_timer,
            'correct_consec': self._correct_consec,
            'wrong_consec': self._wrong_consec,
        }

    @property
    def total_reward(self) -> float:
        return self._total_reward
