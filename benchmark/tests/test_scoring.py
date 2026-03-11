"""
Tests for scoring.py — written BEFORE the implementation (TDD).

These tests define the scoring contract. scoring.py must pass these
before any environment code is written. If scoring has bugs, every
result reported against the benchmark is silently corrupted.

Test categories:
  1. Threshold constants exist and are well-formed
  2. Level classifiers return correct levels for known inputs
  3. Type B-proxy detects genuine updater vs associative signature
  4. ScoringMatrix construction and formatting
  5. Overall level assignment from component levels
"""

import pytest
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 1. THRESHOLD CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestThresholdConstants:
    """Verify that all threshold constants are defined and well-formed."""

    def test_level1_recovery_thresholds_exist(self):
        from benchmark.core.scoring import LEVEL1_RECOVERY_THRESHOLDS
        assert isinstance(LEVEL1_RECOVERY_THRESHOLDS, dict)
        for env in ['CLW-1', 'CLW-2', 'CLW-3']:
            assert env in LEVEL1_RECOVERY_THRESHOLDS
            assert LEVEL1_RECOVERY_THRESHOLDS[env] > 0

    def test_level1_thresholds_increase_with_complexity(self):
        """Harder environments should allow more recovery steps."""
        from benchmark.core.scoring import LEVEL1_RECOVERY_THRESHOLDS
        assert LEVEL1_RECOVERY_THRESHOLDS['CLW-1'] < LEVEL1_RECOVERY_THRESHOLDS['CLW-2']
        assert LEVEL1_RECOVERY_THRESHOLDS['CLW-2'] < LEVEL1_RECOVERY_THRESHOLDS['CLW-3']

    def test_level1_threshold_values(self):
        from benchmark.core.scoring import LEVEL1_RECOVERY_THRESHOLDS
        assert LEVEL1_RECOVERY_THRESHOLDS['CLW-1'] == 5
        assert LEVEL1_RECOVERY_THRESHOLDS['CLW-2'] == 10
        assert LEVEL1_RECOVERY_THRESHOLDS['CLW-3'] == 15

    def test_level2_similarity_threshold(self):
        from benchmark.core.scoring import LEVEL2_SIMILARITY_THRESHOLD
        assert 0 < LEVEL2_SIMILARITY_THRESHOLD <= 1.0
        assert LEVEL2_SIMILARITY_THRESHOLD == 0.6

    def test_level3_accuracy_threshold(self):
        from benchmark.core.scoring import LEVEL3_ACCURACY_THRESHOLD
        assert 0 < LEVEL3_ACCURACY_THRESHOLD <= 1.0
        assert LEVEL3_ACCURACY_THRESHOLD == 0.70

    def test_bproxy_spike_threshold(self):
        from benchmark.core.scoring import BPROXY_SPIKE_THRESHOLD
        assert BPROXY_SPIKE_THRESHOLD > 0
        assert BPROXY_SPIKE_THRESHOLD == 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TEST SCORE DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTestScore:
    def test_creation(self):
        from benchmark.core.scoring import TestScore
        score = TestScore(value=3.2, level=1, confidence=0.95)
        assert score.value == 3.2
        assert score.level == 1
        assert score.confidence == 0.95

    def test_na_score(self):
        """N/A scores for tests that don't apply."""
        from benchmark.core.scoring import TestScore
        score = TestScore(value=float('nan'), level=-1, confidence=0.0)
        assert score.level == -1
        assert np.isnan(score.value)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LEVEL A — BEHAVIORAL RECOVERY
# ═══════════════════════════════════════════════════════════════════════════════

class TestLevelA:
    def test_pass_clw1(self):
        from benchmark.core.scoring import classify_level_a
        assert classify_level_a(steps_to_recovery=3, env_name='CLW-1') == 1

    def test_fail_clw1(self):
        from benchmark.core.scoring import classify_level_a
        assert classify_level_a(steps_to_recovery=8, env_name='CLW-1') == 0

    def test_boundary_clw1(self):
        """Exactly at threshold should fail (strict <)."""
        from benchmark.core.scoring import classify_level_a
        assert classify_level_a(steps_to_recovery=5, env_name='CLW-1') == 0

    def test_pass_clw2(self):
        from benchmark.core.scoring import classify_level_a
        assert classify_level_a(steps_to_recovery=7, env_name='CLW-2') == 1

    def test_fail_clw2(self):
        from benchmark.core.scoring import classify_level_a
        assert classify_level_a(steps_to_recovery=15, env_name='CLW-2') == 0

    def test_pass_clw3(self):
        from benchmark.core.scoring import classify_level_a
        assert classify_level_a(steps_to_recovery=12, env_name='CLW-3') == 1

    def test_fail_clw3(self):
        from benchmark.core.scoring import classify_level_a
        assert classify_level_a(steps_to_recovery=20, env_name='CLW-3') == 0

    def test_perfect_score(self):
        from benchmark.core.scoring import classify_level_a
        assert classify_level_a(steps_to_recovery=0, env_name='CLW-1') == 1

    def test_invalid_env_raises(self):
        from benchmark.core.scoring import classify_level_a
        with pytest.raises(KeyError):
            classify_level_a(steps_to_recovery=3, env_name='CLW-99')


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LEVEL B — REPRESENTATION UPDATE
# ═══════════════════════════════════════════════════════════════════════════════

class TestLevelB:
    def test_pass(self):
        from benchmark.core.scoring import classify_level_b
        assert classify_level_b(similarity=0.75) == 2

    def test_fail(self):
        from benchmark.core.scoring import classify_level_b
        assert classify_level_b(similarity=0.4) == 0

    def test_boundary(self):
        """Exactly at threshold should fail (strict >)."""
        from benchmark.core.scoring import classify_level_b
        assert classify_level_b(similarity=0.6) == 0

    def test_perfect(self):
        from benchmark.core.scoring import classify_level_b
        assert classify_level_b(similarity=1.0) == 2

    def test_negative_similarity(self):
        from benchmark.core.scoring import classify_level_b
        assert classify_level_b(similarity=-0.5) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TYPE B-PROXY — ENTROPY SPIKE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestBProxy:
    def test_genuine_updater(self):
        """Genuine world-model: sharp entropy spike then recovery."""
        from benchmark.core.scoring import score_bproxy
        entropy = np.array([
            0.30, 0.92, 0.88, 0.70,   # spike at steps 1-3
            0.48, 0.35, 0.32, 0.31,   # recovery
            0.30, 0.30                 # stable
        ])
        result = score_bproxy(entropy)
        assert result.spike_detected is True
        assert result.spike_ratio > 0.3

    def test_associative_agent(self):
        """Associative: flat, gradual shift — no spike."""
        from benchmark.core.scoring import score_bproxy
        entropy = np.array([
            0.30, 0.33, 0.36, 0.39,
            0.42, 0.45, 0.48, 0.50,
            0.52, 0.53
        ])
        result = score_bproxy(entropy)
        assert result.spike_detected is False

    def test_constant_entropy(self):
        """No change at all — no spike."""
        from benchmark.core.scoring import score_bproxy
        entropy = np.array([0.5] * 10)
        result = score_bproxy(entropy)
        assert result.spike_detected is False

    def test_short_profile_handled(self):
        """Profile shorter than 10 steps should not crash."""
        from benchmark.core.scoring import score_bproxy
        entropy = np.array([0.3, 0.9, 0.4])
        result = score_bproxy(entropy)
        assert result.spike_detected is False  # too short to evaluate

    def test_zero_late_entropy(self):
        """Zero baseline entropy should not cause division by zero."""
        from benchmark.core.scoring import score_bproxy
        entropy = np.array([0.0, 0.5, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = score_bproxy(entropy)
        # Should not raise, spike_detected may be True (infinite ratio)
        assert isinstance(result.spike_detected, bool)

    def test_result_contains_profile(self):
        from benchmark.core.scoring import score_bproxy
        entropy = np.array([0.5] * 10)
        result = score_bproxy(entropy)
        np.testing.assert_array_equal(result.entropy_profile, entropy)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LEVEL C — NOVEL INTERVENTION GENERALISATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestLevelC:
    def test_pass(self):
        from benchmark.core.scoring import classify_level_c
        assert classify_level_c(accuracy=0.80) == 3

    def test_fail(self):
        from benchmark.core.scoring import classify_level_c
        assert classify_level_c(accuracy=0.50) == 0

    def test_boundary(self):
        """Exactly at threshold should fail (strict >)."""
        from benchmark.core.scoring import classify_level_c
        assert classify_level_c(accuracy=0.70) == 0

    def test_perfect(self):
        from benchmark.core.scoring import classify_level_c
        assert classify_level_c(accuracy=1.0) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SCORING MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoringMatrix:
    def test_set_and_get(self):
        from benchmark.core.scoring import ScoringMatrix, TestScore
        matrix = ScoringMatrix()
        score = TestScore(value=3.2, level=1, confidence=0.95)
        matrix.set_score('CLW-1', 'A', score)
        retrieved = matrix.get_score('CLW-1', 'A')
        assert retrieved.value == 3.2
        assert retrieved.level == 1

    def test_get_missing_returns_none(self):
        from benchmark.core.scoring import ScoringMatrix
        matrix = ScoringMatrix()
        assert matrix.get_score('CLW-1', 'A') is None

    def test_multiple_environments(self):
        from benchmark.core.scoring import ScoringMatrix, TestScore
        matrix = ScoringMatrix()
        matrix.set_score('CLW-1', 'A', TestScore(3.2, 1, 0.95))
        matrix.set_score('CLW-2', 'A', TestScore(7.1, 1, 0.88))
        matrix.set_score('CLW-1', 'B-full', TestScore(0.72, 2, 0.90))
        assert matrix.get_score('CLW-1', 'A').value == 3.2
        assert matrix.get_score('CLW-2', 'A').value == 7.1
        assert matrix.get_score('CLW-1', 'B-full').level == 2

    def test_format_table_contains_environments(self):
        from benchmark.core.scoring import ScoringMatrix, TestScore
        matrix = ScoringMatrix()
        matrix.set_score('CLW-1', 'A', TestScore(3.2, 1, 0.95))
        matrix.set_score('CLW-2', 'A', TestScore(7.1, 1, 0.88))
        table = matrix.format_table()
        assert 'CLW-1' in table
        assert 'CLW-2' in table

    def test_format_table_contains_scores(self):
        from benchmark.core.scoring import ScoringMatrix, TestScore
        matrix = ScoringMatrix()
        matrix.set_score('CLW-1', 'A', TestScore(3.20, 1, 0.95))
        table = matrix.format_table()
        assert '3.20' in table
        assert 'L1' in table

    def test_format_empty_matrix(self):
        from benchmark.core.scoring import ScoringMatrix
        matrix = ScoringMatrix()
        table = matrix.format_table()
        assert 'No scores' in table


# ═══════════════════════════════════════════════════════════════════════════════
# 8. OVERALL LEVEL CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverallLevel:
    def test_level_0(self):
        """Nothing passes → Level 0."""
        from benchmark.core.scoring import classify_overall_level
        assert classify_overall_level({'A': 0, 'B': 0, 'C': 0}) == 0

    def test_level_1(self):
        """Only A passes → Level 1."""
        from benchmark.core.scoring import classify_overall_level
        assert classify_overall_level({'A': 1, 'B': 0, 'C': 0}) == 1

    def test_level_2(self):
        """A and B pass → Level 2."""
        from benchmark.core.scoring import classify_overall_level
        assert classify_overall_level({'A': 1, 'B': 2, 'C': 0}) == 2

    def test_level_3(self):
        """A, B, and C all pass → Level 3."""
        from benchmark.core.scoring import classify_overall_level
        assert classify_overall_level({'A': 1, 'B': 2, 'C': 3}) == 3

    def test_c_without_b(self):
        """C passes but B does not → Level 1 (levels are cumulative)."""
        from benchmark.core.scoring import classify_overall_level
        assert classify_overall_level({'A': 1, 'B': 0, 'C': 3}) == 1

    def test_b_without_a(self):
        """B passes but A does not → Level 0 (need A first)."""
        from benchmark.core.scoring import classify_overall_level
        assert classify_overall_level({'A': 0, 'B': 2, 'C': 3}) == 0

    def test_missing_keys_default_to_zero(self):
        """Missing test types treated as not achieved."""
        from benchmark.core.scoring import classify_overall_level
        assert classify_overall_level({'A': 1}) == 1
        assert classify_overall_level({}) == 0
