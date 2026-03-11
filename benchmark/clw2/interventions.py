"""
CLW-2: Intervention Protocol
=============================
Deterministic intervention schedule for CLW-2 evaluation.

Uses fixed seeds from eval_seeds.json. 
CLW-2 introduces two intervention targets: C1 and C2.
The protocol splits Test A and Test B episodes evenly between do(C1) and do(C2).
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class InterventionSpec:
    """One planned intervention within an episode."""
    step: int
    target: str      # "C1" or "C2"
    value: Any


@dataclass
class EpisodeSpec:
    """Full specification for one evaluation episode."""
    episode_index: int
    seed: int
    test_type: str          # "A", "B", or "C"
    interventions: List[InterventionSpec]
    novel_type: Optional[str] = None


def _load_seeds() -> dict:
    seeds_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'eval_seeds.json'
    )
    with open(seeds_path, 'r') as f:
        return json.load(f)


class CLW2InterventionProtocol:
    """
    Deterministic evaluation protocol for CLW-2.

    Episode allocation:
        0–19:   Test A — do(C1)
        20–39:  Test A — do(C2)
        40–54:  Test B — do(C1)
        55–69:  Test B — do(C2)
        70–79:  Test C1 (simultaneous do(C1) and do(C2))
        80–89:  Test C2 (no-op on both)
        90–99:  Test C3 (OOD — N/A for binary)
    """

    def __init__(self):
        self._seeds_data = _load_seeds()
        self._episode_seeds = self._seeds_data['episode_seeds']
        self._intervention_steps = self._seeds_data['intervention_steps']

    def get_episodes(self) -> List[EpisodeSpec]:
        episodes = []

        for idx, seed in enumerate(self._episode_seeds):
            rng = np.random.RandomState(seed)

            if idx < 20:
                episodes.append(self._make_standard_episode(idx, seed, rng, 'A', 'C1'))
            elif idx < 40:
                episodes.append(self._make_standard_episode(idx, seed, rng, 'A', 'C2'))
            elif idx < 55:
                episodes.append(self._make_standard_episode(idx, seed, rng, 'B', 'C1'))
            elif idx < 70:
                episodes.append(self._make_standard_episode(idx, seed, rng, 'B', 'C2'))
            elif idx < 80:
                episodes.append(self._make_test_c1_episode(idx, seed, rng))
            elif idx < 90:
                episodes.append(self._make_test_c2_episode(idx, seed, rng))
            else:
                episodes.append(self._make_test_c3_episode(idx, seed))

        return episodes

    def _make_standard_episode(
        self, idx: int, seed: int, rng: np.random.RandomState, test_type: str, target: str
    ) -> EpisodeSpec:
        """Standard Test A or B episode intervening on a single target."""
        interventions = []
        for step in self._intervention_steps:
            val = rng.randint(0, 2)
            interventions.append(InterventionSpec(step=step, target=target, value=val))
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type=test_type,
            interventions=interventions,
        )

    def _make_test_c1_episode(
        self, idx: int, seed: int, rng: np.random.RandomState
    ) -> EpisodeSpec:
        """
        Test C1: simultaneous intervention on C1 and C2.
        Assigns contradicting values (v and 1-v) to test if agent knows
        which one dominates immediate reward (C2).
        """
        interventions = []
        for step in self._intervention_steps:
            v1 = rng.randint(0, 2)
            v2 = 1 - v1
            # Apply C1 then C2 (C2 will overwrite the propagated effect of C1)
            interventions.append(InterventionSpec(step=step, target='C1', value=v1))
            interventions.append(InterventionSpec(step=step, target='C2', value=v2))
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type='C',
            interventions=interventions, novel_type='C1_simultaneous',
        )

    def _make_test_c2_episode(
        self, idx: int, seed: int, rng: np.random.RandomState
    ) -> EpisodeSpec:
        """
        Test C2: no-op intervention on both variables.
        """
        interventions = []
        for step in self._intervention_steps:
            interventions.append(InterventionSpec(step=step, target='C1', value=None))
            interventions.append(InterventionSpec(step=step, target='C2', value=None))
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type='C',
            interventions=interventions, novel_type='C2_noop',
        )

    def _make_test_c3_episode(self, idx: int, seed: int) -> EpisodeSpec:
        """OOD — N/A for CLW-2."""
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type='C',
            interventions=[], novel_type='C3_ood',
        )
