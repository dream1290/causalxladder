"""
CLW-2: Causal Chain Environment
================================
Two hidden physical variables C1 and C2, and a hidden Target.
The agent must control C2 to match Target to get rewards.

Causal graph:
    action → C1 → C2 → reward
    Target → reward

Dynamics:
- Target flips periodically (mean 80 steps), just like CLW-1.
- C1 is influenced by action:
  - If action == C1: C1 stays.
  - If action != C1: C1 flips to match action with p=0.8.
- C2 follows C1: C2 = C1 with p=0.8, else 1-C1.
- Reward is based on whether C2 == Target.

To maximize reward, the agent must make C2 match Target.
Therefore, it must make C1 match Target.
Therefore, the optimal action is always to play `Target`.
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np

from ..core.base_env import CLWEnvironment


class CLW2Environment(CLWEnvironment):
    """
    Causal Chain Lever World.

    Hidden state: Target ∈ {0, 1}, C1 ∈ {0, 1}, C2 ∈ {0, 1}.
    Correct action: pull lever Target.
    Actions: 0 (pull lever 0), 1 (pull lever 1), 2 (wait).
    """

    CAUSAL_GRAPH = {
        'nodes': ['Target', 'C1', 'C2', 'reward', 'correct_action'],
        'edges': [
            ('Target', 'reward'), ('Target', 'correct_action'),
            ('C1', 'C2'), ('C2', 'reward')
        ],
        'intervention_targets': ['C1', 'C2'],
        'observable': [],  # all fully hidden
    }

    def __init__(
        self,
        max_steps: int = 200,
        flip_mean: int = 80,
        Control_p: float = 0.8,  # p(C1 flips to action if mismatched)
        C1_C2_p: float = 0.8,    # p(C2 == C1)
        big_reward: float = 10.0,
        big_penalty: float = -30.0,
        step_cost: float = -0.01,
        pull_cost: float = -0.05,
        correct_streak: int = 8,
        wrong_streak: int = 6,
        seed: Optional[int] = None,
    ):
        self.flip_mean = flip_mean
        self.Control_p = Control_p
        self.C1_C2_p = C1_C2_p
        self.big_reward = big_reward
        self.big_penalty = big_penalty
        self.step_cost = step_cost
        self.pull_cost = pull_cost
        self.correct_streak = correct_streak
        self.wrong_streak = wrong_streak

        self._Target: int = 0
        self._C1: int = 0
        self._C2: int = 0
        self._flip_timer: int = 0
        self._correct_consec: int = 0
        self._wrong_consec: int = 0
        self._total_reward: float = 0.0

        super().__init__(max_steps=max_steps, seed=seed)

    def _reset_state(self) -> None:
        self._Target = self._rng.randint(0, 2)
        self._C1 = self._rng.randint(0, 2)
        self._C2 = self._C1 if self._rng.rand() < self.C1_C2_p else 1 - self._C1
        self._flip_timer = self._rng.geometric(1.0 / self.flip_mean)
        self._correct_consec = 0
        self._wrong_consec = 0
        self._total_reward = 0.0

    def _step_impl(self, action: int) -> Tuple[float, bool, Dict]:
        reward = self.step_cost
        done = False
        
        # Advance Target Timer
        self._flip_timer -= 1
        if self._flip_timer <= 0:
            self._Target = 1 - self._Target
            self._flip_timer = self._rng.geometric(1.0 / self.flip_mean)

        info = {'correct_pull': False, 'Target': self._Target, 'C1': self._C1, 'C2': self._C2}

        if action == 2:  # Wait
            self._correct_consec = 0
            self._wrong_consec = 0
            self._total_reward += reward
            done = self._step_count + 1 >= self.max_steps
            return reward, done, info

        # Pull a lever
        reward += self.pull_cost
        
        # 1. Action -> C1
        c1_changed = False
        if action != self._C1:
            if self._rng.rand() < self.Control_p:
                self._C1 = action
                c1_changed = True
        
        # 2. C1 -> C2 (only updates if C1 changed!)
        if c1_changed:
            self._C2 = self._C1 if self._rng.rand() < self.C1_C2_p else 1 - self._C1
        
        # 3. C2 -> Reward
        c2_matches_target = (self._C2 == self._Target)

        if c2_matches_target:
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

        self._total_reward += reward

        if self._step_count + 1 >= self.max_steps:
            done = True

        return reward, done, info

    def _apply_intervention(self, target: str, value: Any) -> None:
        """do(C1=v) or do(C2=v)"""
        if target == 'C1':
            assert value in (0, 1)
            # If intervention changes C1, C2 updates to match (with p=C1_C2_p)
            # This perfectly models mediation: intervening on root cause flows downstream
            if self._C1 != int(value):
                self._C1 = int(value)
                self._C2 = self._C1 if self._rng.rand() < self.C1_C2_p else 1 - self._C1
        elif target == 'C2':
            assert value in (0, 1)
            self._C2 = int(value)
            # do(C2) does NOT affect C1 (bypasses C1)

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            'Target': self._Target,
            'C1': self._C1,
            'C2': self._C2,
            'correct_action': self._Target,
            'flip_timer': self._flip_timer,
            'correct_consec': self._correct_consec,
            'wrong_consec': self._wrong_consec,
        }

    @property
    def total_reward(self) -> float:
        return self._total_reward
