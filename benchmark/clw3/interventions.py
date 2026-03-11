"""
CLW-3: Intervention Protocol
=============================
Deterministic intervention schedule for CLW-3 evaluation.

CLW-3 introduces three intervention targets: S1, S2, and C.
The protocol splits Test A and Test B episodes evenly between do(S1) and do(S2).
Test C includes do(C) to verify full causal structure understanding.
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
    target: str      # "S1", "S2", or "C"
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


class CLW3InterventionProtocol:
    """
    Deterministic evaluation protocol for CLW-3.

    Episode allocation:
        0–19:   Test A — do(S1)
        20–39:  Test A — do(S2)
        40–54:  Test B — do(S1)
        55–69:  Test B — do(S2)
        70–79:  Test C1 (do(C))
        80–89:  Test C2 (no-op on S1 and S2)
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
                episodes.append(self._make_standard_episode(idx, seed, rng, 'A', 'S1'))
            elif idx < 40:
                episodes.append(self._make_standard_episode(idx, seed, rng, 'A', 'S2'))
            elif idx < 55:
                episodes.append(self._make_standard_episode(idx, seed, rng, 'B', 'S1'))
            elif idx < 70:
                episodes.append(self._make_standard_episode(idx, seed, rng, 'B', 'S2'))
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
        """Standard Test A or B episode intervening on a single sensor."""
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
        Test C1: Intervene on the hidden common cause C directly.
        This tests if the agent properly propagates the intervention to expected S1, S2, and Action.
        """
        interventions = []
        for step in self._intervention_steps:
            val = rng.randint(0, 2)
            interventions.append(InterventionSpec(step=step, target='C', value=val))
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type='C',
            interventions=interventions, novel_type='C1_do_cause',
        )

    def _make_test_c2_episode(
        self, idx: int, seed: int, rng: np.random.RandomState
    ) -> EpisodeSpec:
        """
        Test C2: no-op intervention on both sensors.
        """
        interventions = []
        for step in self._intervention_steps:
            interventions.append(InterventionSpec(step=step, target='S1', value=None))
            interventions.append(InterventionSpec(step=step, target='S2', value=None))
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type='C',
            interventions=interventions, novel_type='C2_noop',
        )

    def _make_test_c3_episode(self, idx: int, seed: int) -> EpisodeSpec:
        """OOD — N/A for CLW-3."""
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type='C',
            interventions=[], novel_type='C3_ood',
        )
