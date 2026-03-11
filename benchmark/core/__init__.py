"""Core module: base environment, agent interface, and scoring."""

from .scoring import (
    TestScore,
    BProxyResult,
    ScoringMatrix,
    classify_level_a,
    classify_level_b,
    classify_level_c,
    score_bproxy,
    classify_overall_level,
)
from .agent_interface import BenchmarkAgent, BehavioralProxy
