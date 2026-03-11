"""
CLW-1: Intervention Protocol
=============================
Deterministic intervention schedule for CLW-1 evaluation.

Uses fixed seeds and fixed intervention steps (10, 25, 50) loaded from
benchmark/data/eval_seeds.json. Results are reproducible across machines
and implementations.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class InterventionSpec:
    """One planned intervention within an episode."""
    step: int        # when to intervene
    target: str      # variable to set ("C" for CLW-1)
    value: Any       # value to set


@dataclass
class EpisodeSpec:
    """Full specification for one evaluation episode."""
    episode_index: int
    seed: int
    test_type: str          # "A", "B", or "C"
    interventions: List[InterventionSpec]
    novel_type: Optional[str] = None  # "C1_simultaneous", "C2_noop", "C3_ood"


def _load_seeds() -> dict:
    """Load eval seeds from the committed static file."""
    seeds_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'eval_seeds.json'
    )
    with open(seeds_path, 'r') as f:
        return json.load(f)


class CLW1InterventionProtocol:
    """
    Deterministic evaluation protocol for CLW-1.

    Generates 100 evaluation episodes with pre-planned intervention schedules.
    The schedule is entirely determined by eval_seeds.json — no randomness
    at evaluation time.

    Episode allocation:
        0–39:   Test A (behavioral recovery)
        40–69:  Test B (representation update)
        70–79:  Test C1 (simultaneous — N/A for CLW-1, mapped to double-flip)
        80–89:  Test C2 (no-op intervention)
        90–99:  Test C3 (OOD — N/A for binary C, skipped)
    """

    def __init__(self):
        self._seeds_data = _load_seeds()
        self._episode_seeds = self._seeds_data['episode_seeds']
        self._intervention_steps = self._seeds_data['intervention_steps']

    def get_episodes(self) -> List[EpisodeSpec]:
        """Generate all 100 evaluation episode specifications."""
        episodes = []

        for idx, seed in enumerate(self._episode_seeds):
            rng = np.random.RandomState(seed)

            if idx < 40:
                # Test A: standard interventions (flip C to opposite)
                episodes.append(self._make_test_a_episode(idx, seed, rng))
            elif idx < 70:
                # Test B: standard interventions for representation testing
                episodes.append(self._make_test_b_episode(idx, seed, rng))
            elif idx < 80:
                # Test C1: simultaneous (double-flip for CLW-1)
                episodes.append(self._make_test_c1_episode(idx, seed, rng))
            elif idx < 90:
                # Test C2: no-op intervention
                episodes.append(self._make_test_c2_episode(idx, seed, rng))
            else:
                # Test C3: OOD — N/A for binary CLW-1
                episodes.append(self._make_test_c3_episode(idx, seed))

        return episodes

    def _make_test_a_episode(
        self, idx: int, seed: int, rng: np.random.RandomState
    ) -> EpisodeSpec:
        """Test A: flip C to opposite at each intervention step."""
        interventions = []
        for step in self._intervention_steps:
            # Value alternates based on seed parity (deterministic)
            val = rng.randint(0, 2)
            interventions.append(InterventionSpec(step=step, target='C', value=val))
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type='A',
            interventions=interventions,
        )

    def _make_test_b_episode(
        self, idx: int, seed: int, rng: np.random.RandomState
    ) -> EpisodeSpec:
        """Test B: same structure as A but representation is recorded."""
        interventions = []
        for step in self._intervention_steps:
            val = rng.randint(0, 2)
            interventions.append(InterventionSpec(step=step, target='C', value=val))
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type='B',
            interventions=interventions,
        )

    def _make_test_c1_episode(
        self, idx: int, seed: int, rng: np.random.RandomState
    ) -> EpisodeSpec:
        """
        Test C1: simultaneous intervention.

        For CLW-1 (single hidden variable), this is a double-flip at the
        same step — set C to a value, then immediately set it again.
        Tests whether the agent handles rapid successive changes.
        """
        interventions = []
        for step in self._intervention_steps:
            v1 = rng.randint(0, 2)
            v2 = 1 - v1  # opposite value immediately after
            interventions.append(InterventionSpec(step=step, target='C', value=v1))
            interventions.append(InterventionSpec(step=step, target='C', value=v2))
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type='C',
            interventions=interventions, novel_type='C1_simultaneous',
        )

    def _make_test_c2_episode(
        self, idx: int, seed: int, rng: np.random.RandomState
    ) -> EpisodeSpec:
        """
        Test C2: no-op intervention.

        Set C to its CURRENT value. A genuine causal model correctly predicts
        no change in the correct action. An associative system might still show
        a disturbance in its representation.

        We use value=None as a sentinel — the evaluator must read the current
        ground truth and set C to its current value at runtime.
        """
        interventions = []
        for step in self._intervention_steps:
            # value=None signals "use current value" — evaluator resolves this
            interventions.append(InterventionSpec(step=step, target='C', value=None))
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type='C',
            interventions=interventions, novel_type='C2_noop',
        )

    def _make_test_c3_episode(self, idx: int, seed: int) -> EpisodeSpec:
        """
        Test C3: OOD intervention.

        N/A for CLW-1 (binary C). Included for protocol completeness.
        Evaluator should report this as N/A in the scoring matrix.
        """
        return EpisodeSpec(
            episode_index=idx, seed=seed, test_type='C',
            interventions=[], novel_type='C3_ood',
        )
