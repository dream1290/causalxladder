"""
Scoring module for the Causal Learning Benchmark.

Defines thresholds, level classifiers, Type B-proxy scoring, and the
results matrix. All threshold values are named constants — change them
here to update the entire benchmark.

Levels:
  0 — Chance: random policy performance
  1 — Behavioral: Test A passes (system eventually adapts)
  2 — Representational: Tests A+B pass (internal state updates correctly)
  3 — Causal: Tests A+B+C pass (generalises to novel interventions)
  4 — Counterfactual: defined, not yet measurable
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD CONSTANTS
#
# These are the published defaults. When the field adjusts thresholds,
# the change happens HERE and only here — not scattered through eval code.
# ═══════════════════════════════════════════════════════════════════════════════

# Level 1 — Behavioral Recovery: max steps-to-recovery (strict <)
LEVEL1_RECOVERY_THRESHOLDS: Dict[str, int] = {
    'CLW-1': 5,    # single confounder: must recover in under 5 steps
    'CLW-2': 10,   # causal chain: under 10 steps
    'CLW-3': 15,   # common cause: under 15 steps
}

# Level 2 — Representational: min cosine similarity at step 5 post-intervention (strict >)
LEVEL2_SIMILARITY_THRESHOLD: float = 0.6

# Level 3 — Causal: min accuracy on novel interventions (strict >)
LEVEL3_ACCURACY_THRESHOLD: float = 0.70

# Type B-proxy: min relative entropy spike to detect world-model signature (strict >)
BPROXY_SPIKE_THRESHOLD: float = 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestScore:
    """Score for a single (environment × test type) cell in the results matrix."""
    value: float       # raw metric value (steps, similarity, accuracy, etc.)
    level: int         # level achieved (-1 for N/A)
    confidence: float  # confidence in this score (0.0–1.0)


@dataclass
class BProxyResult:
    """Result of Type B-proxy (behavioral) scoring."""
    spike_ratio: float        # (early_entropy - late_entropy) / late_entropy
    spike_detected: bool      # whether spike exceeds threshold
    entropy_profile: np.ndarray  # full entropy history


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_level_a(steps_to_recovery: float, env_name: str) -> int:
    """
    Level 1 — Behavioral Recovery.

    Returns 1 if the agent recovers in strictly fewer steps than
    the threshold for the given environment, 0 otherwise.

    Raises KeyError if env_name is not in LEVEL1_RECOVERY_THRESHOLDS.
    """
    threshold = LEVEL1_RECOVERY_THRESHOLDS[env_name]
    return 1 if steps_to_recovery < threshold else 0


def classify_level_b(similarity: float) -> int:
    """
    Level 2 — Representational Update.

    Returns 2 if cosine similarity between post-intervention representation
    and expected representation strictly exceeds the threshold, 0 otherwise.
    """
    return 2 if similarity > LEVEL2_SIMILARITY_THRESHOLD else 0


def classify_level_c(accuracy: float) -> int:
    """
    Level 3 — Novel Intervention Generalisation.

    Returns 3 if accuracy on novel interventions strictly exceeds
    the threshold, 0 otherwise.
    """
    return 3 if accuracy > LEVEL3_ACCURACY_THRESHOLD else 0


def score_bproxy(entropy_profile: np.ndarray) -> BProxyResult:
    """
    Type B-proxy: detect world-model signature from action entropy profile.

    Expected signature for a genuine internal updater:
      - Entropy spike in steps 1–3 post-intervention (uncertainty as z updates)
      - Recovery by step 7 (new belief stabilises)

    Metric: (mean_entropy[1:4] - mean_entropy[-5:]) / mean_entropy[-5:]
    A spike_ratio above BPROXY_SPIKE_THRESHOLD indicates a world-model signature.

    This is explicitly a WEAK signal — documented as such. It is
    architecture-agnostic and reproducible, but does not replace full
    Type B testing for systems that expose internals.

    Args:
        entropy_profile: array of action entropy values, one per step,
                         for the post-intervention window (typically 10 steps).

    Returns:
        BProxyResult with spike_ratio, spike_detected flag, and the profile.
    """
    if len(entropy_profile) < 10:
        return BProxyResult(
            spike_ratio=0.0,
            spike_detected=False,
            entropy_profile=entropy_profile,
        )

    early = float(np.mean(entropy_profile[1:4]))   # steps 1–3 (0-indexed)
    late = float(np.mean(entropy_profile[-5:]))     # last 5 steps

    if late < 1e-10:
        # Avoid division by zero. If late entropy is ~0 and early is nonzero,
        # that's a spike. If both are ~0, no spike.
        spike_ratio = float(early > 1e-10)
    else:
        spike_ratio = (early - late) / late

    spike_detected = spike_ratio > BPROXY_SPIKE_THRESHOLD

    return BProxyResult(
        spike_ratio=spike_ratio,
        spike_detected=spike_detected,
        entropy_profile=entropy_profile,
    )


def classify_overall_level(test_levels: Dict[str, int]) -> int:
    """
    Overall level for an agent on one environment.

    Levels are cumulative — each requires the previous:
      Level 0: nothing passes
      Level 1: Test A passes (level >= 1)
      Level 2: Level 1 + Test B passes (level >= 2)
      Level 3: Level 2 + Test C passes (level >= 3)
      Level 4: Level 3 + counterfactual (not yet measurable)

    Missing keys in test_levels are treated as 0 (not achieved).
    """
    a_level = test_levels.get('A', 0)
    b_level = test_levels.get('B', 0)
    c_level = test_levels.get('C', 0)

    if a_level < 1:
        return 0
    if b_level < 2:
        return 1
    if c_level < 3:
        return 2
    return 3


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical environment and test type names
ENVIRONMENTS = ['CLW-1', 'CLW-2', 'CLW-3']
TEST_TYPES = ['A', 'B-full', 'B-proxy', 'C1', 'C2', 'C3']


class ScoringMatrix:
    """
    Results matrix for one agent across all environments and test types.

    Rows: test types (A, B-full, B-proxy, C1, C2, C3).
    Columns: environments (CLW-1, CLW-2, CLW-3).
    Cells: TestScore(value, level, confidence).

    This matrix is the single output of run_benchmark.py. It is what
    gets reported in papers.
    """

    def __init__(self):
        self._scores: Dict[str, Dict[str, TestScore]] = {}

    def set_score(self, env: str, test_type: str, score: TestScore) -> None:
        """Set a score for a (environment, test_type) cell."""
        if env not in self._scores:
            self._scores[env] = {}
        self._scores[env][test_type] = score

    def get_score(self, env: str, test_type: str) -> Optional[TestScore]:
        """Get a score for a (environment, test_type) cell, or None if unset."""
        return self._scores.get(env, {}).get(test_type)

    def format_table(self) -> str:
        """
        Human-readable results table for terminal output or paper inclusion.

        Format:
          Test       |      CLW-1       |      CLW-2       |      CLW-3
          A          |  3.20 (L1)       |  7.10 (L1)       |  N/A
          B-full     |  0.72 (L2)       |  N/A             |  N/A
        """
        envs = sorted(self._scores.keys())
        if not envs:
            return "No scores recorded."

        # Collect all test types present
        all_tests: set = set()
        for env_scores in self._scores.values():
            all_tests.update(env_scores.keys())
        tests = sorted(all_tests)

        # Column widths
        test_col_w = max(10, max((len(t) for t in tests), default=4) + 2)
        env_col_w = 18

        # Header
        header = f"{'Test':<{test_col_w}}"
        for e in envs:
            header += f" | {e:^{env_col_w}}"

        sep = '-' * len(header)
        lines = [sep, header, sep]

        # Rows
        for t in tests:
            row = f"{t:<{test_col_w}}"
            for e in envs:
                score = self.get_score(e, t)
                if score is None:
                    cell = "N/A"
                else:
                    cell = f"{score.value:.2f} (L{score.level})"
                row += f" | {cell:^{env_col_w}}"
            lines.append(row)

        lines.append(sep)
        return '\n'.join(lines)
