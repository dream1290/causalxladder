"""
Agent interface for the Causal Learning Benchmark.

Defines:
  1. BenchmarkAgent — the minimal protocol any system must implement.
     Uses Python's Protocol (structural subtyping) so agents don't need
     to inherit from anything.

  2. BehavioralProxy — a transparent wrapper that records action entropy
     at every step, enabling Type B-proxy scoring for any agent without
     requiring internal representation access.
"""

from typing import Optional, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class BenchmarkAgent(Protocol):
    """
    Architecture-agnostic agent protocol.

    Any system implementing these four methods can be evaluated by the
    benchmark. No inheritance required — Python's structural subtyping
    handles conformance checking.

    Required:
        reset()                     — clear state for a new episode
        observe(obs: np.ndarray)    — receive a 4-dim observation
        act() -> int                — return an action (0, 1, or 2)

    Optional:
        get_representation()        — return internal state for Test B
                                      (return None for black-box agents)
    """

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        ...

    def observe(self, obs: np.ndarray) -> None:
        """
        Receive a 4-dim observation from the environment.

        Args:
            obs: np.ndarray of shape (4,) — [last_action_norm, last_outcome,
                 steps_norm, noise]
        """
        ...

    def act(self) -> int:
        """
        Choose an action.

        Returns:
            0 (pull lever 0), 1 (pull lever 1), or 2 (wait).
        """
        ...

    def get_representation(self) -> Optional[np.ndarray]:
        """
        Return the agent's internal representation as a flat numpy array.

        Returns None if the agent does not expose internals.
        This enables full Type B testing for systems that support it,
        while keeping the interface compatible with black-box agents.
        """
        ...


class BehavioralProxy:
    """
    Transparent wrapper that records action entropy for Type B-proxy scoring.

    Wraps any BenchmarkAgent. Intercepts act() calls to build a rolling
    entropy profile. The evaluator extracts the entropy profile around
    intervention points to detect the world-model signature:
      - Genuine updater: sharp entropy spike → rapid recovery
      - Associative agent: gradual, flat entropy shift

    This makes Type B-proxy scoring fully automatic and architecture-agnostic.

    Usage:
        agent = SomeAgent()
        proxy = BehavioralProxy(agent)
        # Use proxy in place of agent during evaluation
        # Then: proxy.get_entropy_around_step(intervention_step)
    """

    def __init__(self, agent: BenchmarkAgent, action_space_size: int = 3):
        self._agent = agent
        self._action_space_size = action_space_size
        self._window_size = 5          # rolling window for entropy estimation
        self._recent_actions: list = []
        self._entropy_history: list = []

    def reset(self) -> None:
        """Reset the wrapped agent and clear entropy history."""
        self._agent.reset()
        self._recent_actions = []
        self._entropy_history = []

    def observe(self, obs: np.ndarray) -> None:
        """Pass observation through to the wrapped agent."""
        self._agent.observe(obs)

    def act(self) -> int:
        """
        Get action from wrapped agent and record the entropy.

        Entropy is computed over a rolling window of recent actions.
        Uses base-2 logarithm: max entropy for 3 actions = log2(3) ≈ 1.585.
        """
        action = self._agent.act()
        self._recent_actions.append(action)

        # Compute rolling entropy over the last _window_size actions
        window = self._recent_actions[-self._window_size:]
        counts = np.zeros(self._action_space_size)
        for a in window:
            counts[a] += 1
        probs = counts / len(window)
        # Only include nonzero probabilities for log
        nonzero = probs[probs > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        self._entropy_history.append(entropy)

        return action

    def get_representation(self) -> Optional[np.ndarray]:
        """Delegate to the wrapped agent."""
        return self._agent.get_representation()

    def get_entropy_profile(self) -> np.ndarray:
        """Return the full entropy history across all steps."""
        return np.array(self._entropy_history, dtype=np.float64)

    def get_entropy_around_step(self, step: int, window: int = 10) -> np.ndarray:
        """
        Return entropy values from step to step+window.

        Used for Type B-proxy scoring around intervention points.

        Args:
            step:   the step index (0-based) of the intervention.
            window: number of post-intervention steps to include.

        Returns:
            np.ndarray of entropy values, length min(window, available).
        """
        profile = np.array(self._entropy_history, dtype=np.float64)
        start = max(0, step)
        end = min(len(profile), step + window)
        if start >= end:
            return np.array([], dtype=np.float64)
        return profile[start:end]
