"""
CLW-3: Common Cause Environment
=================================
One hidden cause C drives two observable sensors S1, S2 and the correct action.

Causal graph:
    C → S1
    C → S2
    C → correct_action → reward

Variables:
- C: Hidden binary confounder, flips via geometric distribution (mean=80).
- S1: Observable sensor 1. Matches C with probability 0.8.
- S2: Observable sensor 2. Matches C with probability 0.8.
- Action: {0, 1, 2=Wait}. Reward: +1 if action==C, -0.5 otherwise.

Key Dynamic:
Under natural observation, S1 and S2 are strongly correlated because they
share the common cause C. Intervening on S1 (do(S1=x)) breaks C → S1,
pinning S1 to x while C and S2 continue normal dynamics. An associative
agent will fail because the spurious S1↔S2 correlation is broken.

Observation vector: [last_action_norm, last_outcome, steps_norm, noise]
  (standard base_env contract — S1 and S2 are NOT in the obs vector,
   they are accessible via get_ground_truth() for evaluation only)

Wait — the plan says obs = [S1, S2, 0.0, 0.0]. But the base class builds
obs as [last_action_norm, last_outcome, steps_norm, noise]. We need to
decide: does the agent observe S1 and S2?

YES. In CLW-3, the agents DO observe S1 and S2 — that's the whole point.
The sensors are the observable symptoms of the hidden cause C.
So we override _build_observation to put S1, S2 in the obs vector.
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np

from ..core.base_env import CLWEnvironment, OBS_DIM


class CLW3Environment(CLWEnvironment):
    """
    Common Cause Lever World.

    Hidden state: C ∈ {0, 1}.
    Observable signals: S1, S2 (noisy indicators of C).
    Correct action: pull lever C.
    Actions: 0 (pull lever 0), 1 (pull lever 1), 2 (wait).
    """

    CAUSAL_GRAPH = {
        'nodes': ['C', 'S1', 'S2', 'correct_action', 'reward'],
        'edges': [
            ('C', 'S1'), ('C', 'S2'),
            ('C', 'correct_action'), ('correct_action', 'reward'),
        ],
        'intervention_targets': ['S1', 'S2', 'C'],
        'observable': ['S1', 'S2'],
    }

    def __init__(
        self,
        max_steps: int = 200,
        flip_mean: int = 80,
        sensor_acc: float = 0.8,
        big_reward: float = 10.0,
        big_penalty: float = -30.0,
        step_cost: float = -0.01,
        pull_cost: float = -0.05,
        correct_streak: int = 8,
        wrong_streak: int = 6,
        seed: Optional[int] = None,
    ):
        self.flip_mean = flip_mean
        self.sensor_acc = sensor_acc
        self.big_reward = big_reward
        self.big_penalty = big_penalty
        self.step_cost = step_cost
        self.pull_cost = pull_cost
        self.correct_streak = correct_streak
        self.wrong_streak = wrong_streak

        self._C: int = 0
        self._S1: int = 0
        self._S2: int = 0
        self._flip_timer: int = 0
        self._correct_consec: int = 0
        self._wrong_consec: int = 0
        self._total_reward: float = 0.0

        # Intervention pins: None means natural dynamics, int means pinned
        self._S1_pinned: Optional[int] = None
        self._S2_pinned: Optional[int] = None

        super().__init__(max_steps=max_steps, seed=seed)

    def _sample_sensor(self, c_val: int) -> int:
        """Sample a sensor value from C with accuracy self.sensor_acc."""
        if self._rng.rand() < self.sensor_acc:
            return c_val
        return 1 - c_val

    def _reset_state(self) -> None:
        self._C = self._rng.randint(0, 2)
        self._S1 = self._sample_sensor(self._C)
        self._S2 = self._sample_sensor(self._C)
        self._flip_timer = self._rng.geometric(1.0 / self.flip_mean)
        self._correct_consec = 0
        self._wrong_consec = 0
        self._total_reward = 0.0
        self._S1_pinned = None
        self._S2_pinned = None

    def _step_impl(self, action: int) -> Tuple[float, bool, Dict]:
        reward = self.step_cost
        done = False

        # 1. Advance C flip timer
        self._flip_timer -= 1
        if self._flip_timer <= 0:
            self._C = 1 - self._C
            self._flip_timer = self._rng.geometric(1.0 / self.flip_mean)

        # 2. Resample sensors from (possibly new) C
        self._S1 = self._sample_sensor(self._C)
        self._S2 = self._sample_sensor(self._C)

        # 3. Apply intervention pins (override natural sensor values)
        if self._S1_pinned is not None:
            self._S1 = self._S1_pinned
        if self._S2_pinned is not None:
            self._S2 = self._S2_pinned

        info = {
            'correct_pull': False,
            'C': self._C, 'S1': self._S1, 'S2': self._S2,
        }

        if action == 2:  # Wait
            self._correct_consec = 0
            self._wrong_consec = 0
            self._total_reward += reward
            done = self._step_count + 1 >= self.max_steps
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

        self._total_reward += reward

        if self._step_count + 1 >= self.max_steps:
            done = True

        return reward, done, info

    def _build_observation(self) -> np.ndarray:
        """
        Override base: obs = [S1, S2, steps_norm, noise].
        S1 and S2 are the observable sensors in CLW-3.
        """
        obs = np.array([
            float(self._S1),
            float(self._S2),
            self._step_count / max(self.max_steps, 1),
            self._rng.normal(0, 0.05),
        ], dtype=np.float32)
        assert obs.shape == (OBS_DIM,)
        return obs

    def _apply_intervention(self, target: str, value: Any) -> None:
        """
        do(S1=v): pins S1 to v, breaking C→S1.
        do(S2=v): pins S2 to v, breaking C→S2.
        do(C=v):  sets C directly, resets flip timer.
        """
        if target == 'S1':
            assert value in (0, 1), f"S1 must be 0 or 1, got {value}"
            self._S1_pinned = int(value)
            self._S1 = int(value)
        elif target == 'S2':
            assert value in (0, 1), f"S2 must be 0 or 1, got {value}"
            self._S2_pinned = int(value)
            self._S2 = int(value)
        elif target == 'C':
            assert value in (0, 1), f"C must be 0 or 1, got {value}"
            self._C = int(value)
            self._flip_timer = self._rng.geometric(1.0 / self.flip_mean)
            # Resample un-pinned sensors from new C
            if self._S1_pinned is None:
                self._S1 = self._sample_sensor(self._C)
            if self._S2_pinned is None:
                self._S2 = self._sample_sensor(self._C)

    def get_ground_truth(self) -> Dict[str, Any]:
        return {
            'C': self._C,
            'S1': self._S1,
            'S2': self._S2,
            'correct_action': self._C,
            'flip_timer': self._flip_timer,
            'correct_consec': self._correct_consec,
            'wrong_consec': self._wrong_consec,
        }

    @property
    def total_reward(self) -> float:
        return self._total_reward
