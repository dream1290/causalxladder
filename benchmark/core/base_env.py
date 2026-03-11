"""
Base environment for the Causal Learning Benchmark.

Defines:
  1. The abstract interface every CLW environment must implement
  2. The observation contract (4-dim vector, enforced by assertion)
  3. The causal graph declaration (metadata for scoring and validation)

Every subclass must:
  - Set CAUSAL_GRAPH as a class variable
  - Implement _reset_state(), _step_impl(), _apply_intervention(),
    get_ground_truth()
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Set, Tuple, Optional
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVATION CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════

OBS_DIM = 4                        # fixed across all environments
ACTION_SPACE: Set[int] = {0, 1, 2}  # Pull-0, Pull-1, Wait
MAX_ACTIONS = 3


class CLWEnvironment(ABC):
    """
    Abstract base for all Causal Lever World environments.

    Enforces three contracts:
      1. Observations are always 4-dim: [last_action_norm, last_outcome,
         steps_norm, noise]
      2. A causal graph is declared at class level
      3. Interventions are restricted to declared targets

    Subclasses implement the physics; this class handles the protocol.
    """

    # ── Causal graph declaration ──────────────────────────────────────────
    # Subclasses MUST override this with a dict containing:
    #   nodes:                list of variable names
    #   edges:                list of (parent, child) tuples
    #   intervention_targets: list of variables that can be intervened on
    #   observable:           list of variables the agent can (partially) observe
    CAUSAL_GRAPH: Dict[str, Any] = {}

    def __init__(self, max_steps: int = 200, seed: Optional[int] = None):
        self.max_steps = max_steps
        self._rng = np.random.RandomState(seed)
        self._step_count = 0
        self._last_action = -1      # no action yet
        self._last_outcome = 0.0
        self._done = False
        self._validate_causal_graph()

    def _validate_causal_graph(self) -> None:
        """Fail loudly if subclass did not declare a valid causal graph."""
        required_keys = {'nodes', 'edges', 'intervention_targets', 'observable'}
        if not self.CAUSAL_GRAPH:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define CAUSAL_GRAPH class variable."
            )
        missing = required_keys - set(self.CAUSAL_GRAPH.keys())
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}.CAUSAL_GRAPH missing keys: {missing}"
            )
        # Intervention targets must be a subset of nodes
        nodes = set(self.CAUSAL_GRAPH['nodes'])
        for target in self.CAUSAL_GRAPH['intervention_targets']:
            if target not in nodes:
                raise ValueError(
                    f"Intervention target '{target}' is not in "
                    f"CAUSAL_GRAPH nodes: {nodes}"
                )

    # ── Observation construction ──────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        """
        Build the standard 4-dim observation vector.

        [0] last_action_normalised:  last action / (num_actions - 1), in [0, 1]
        [1] last_outcome:            1.0 if last pull was correct, 0.0 otherwise
        [2] steps_normalised:        current step / max_steps, in [0, 1]
        [3] noise:                   Gaussian noise N(0, 0.05)
        """
        obs = np.array([
            self._last_action / max(MAX_ACTIONS - 1, 1),
            self._last_outcome,
            self._step_count / max(self.max_steps, 1),
            self._rng.normal(0, 0.05),
        ], dtype=np.float32)

        assert obs.shape == (OBS_DIM,), (
            f"Observation shape {obs.shape} != ({OBS_DIM},). "
            f"Bug in {self.__class__.__name__}._build_observation()."
        )
        return obs

    # ── Public interface ──────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Returns:
            Initial 4-dim observation.
        """
        self._step_count = 0
        self._last_action = -1
        self._last_outcome = 0.0
        self._done = False
        self._reset_state()
        return self._build_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step.

        Args:
            action: 0 (pull lever 0), 1 (pull lever 1), or 2 (wait).

        Returns:
            Tuple of (observation, reward, done, info).
            info must contain 'correct_pull' (bool).
        """
        assert action in ACTION_SPACE, (
            f"Invalid action {action}. Must be in {ACTION_SPACE}."
        )
        assert not self._done, "Episode is done. Call reset() first."

        reward, done, info = self._step_impl(action)

        self._last_action = action
        self._last_outcome = float(info.get('correct_pull', False))
        self._step_count += 1
        self._done = done

        obs = self._build_observation()
        return obs, reward, done, info

    def intervene(self, target: str, value: Any) -> None:
        """
        Apply a causal intervention: do(target = value).

        This sets the target variable directly, bypassing normal dynamics.
        The target must be declared in CAUSAL_GRAPH['intervention_targets'].

        Args:
            target: name of the variable to intervene on.
            value:  the value to set.
        """
        valid = self.CAUSAL_GRAPH['intervention_targets']
        assert target in valid, (
            f"Invalid intervention target '{target}'. "
            f"Valid targets for {self.__class__.__name__}: {valid}"
        )
        self._apply_intervention(target, value)

    @abstractmethod
    def get_ground_truth(self) -> Dict[str, Any]:
        """
        Return the true hidden state of the environment.

        For evaluation only — never given to agents during training.
        The evaluator uses this to compute correct-action predictions
        and verify intervention effects.

        Returns:
            Dict mapping variable names to their current values.
        """
        ...

    @property
    def step_count(self) -> int:
        """Current step within the episode."""
        return self._step_count

    @property
    def done(self) -> bool:
        """Whether the current episode has ended."""
        return self._done

    # ── Subclass hooks (must implement) ───────────────────────────────────

    @abstractmethod
    def _reset_state(self) -> None:
        """Reset environment-specific internal state."""
        ...

    @abstractmethod
    def _step_impl(self, action: int) -> Tuple[float, bool, Dict]:
        """
        Execute one step of environment-specific dynamics.

        Args:
            action: validated action (0, 1, or 2).

        Returns:
            Tuple of (reward, done, info).
            info MUST contain 'correct_pull' (bool).
        """
        ...

    @abstractmethod
    def _apply_intervention(self, target: str, value: Any) -> None:
        """
        Apply intervention. Target has already been validated against
        CAUSAL_GRAPH['intervention_targets'].
        """
        ...
