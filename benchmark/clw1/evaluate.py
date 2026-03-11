"""
CLW-1: Evaluation Suite
========================
Runs any BenchmarkAgent through the full CLW-1 test protocol.

Test Type A — Behavioral Recovery:
    Steps until agent consistently takes the correct action after
    each intervention. Scored against LEVEL1_RECOVERY_THRESHOLDS.

Test Type B — Representation Update:
    Full:  cosine similarity of agent's representation vs expected.
    Proxy: entropy spike detection via BehavioralProxy.

Test Type C — Novel Intervention Generalisation:
    C1: double-flip (simultaneous)
    C2: no-op (set C to current value)
    C3: OOD (N/A for binary CLW-1)
"""

from typing import Optional
import numpy as np

from ..core.scoring import (
    ScoringMatrix,
    TestScore,
    BProxyResult,
    classify_level_a,
    classify_level_b,
    classify_level_c,
    score_bproxy,
)
from ..core.agent_interface import BenchmarkAgent, BehavioralProxy
from .env import CLW1Environment
from .interventions import CLW1InterventionProtocol, EpisodeSpec, InterventionSpec


ENV_NAME = 'CLW-1'

# Post-intervention measurement windows
RECOVERY_WINDOW = 20       # max steps to measure recovery
REPR_STEPS = [1, 3, 5, 10, 30]  # steps at which to sample representation
BPROXY_WINDOW = 10         # steps for entropy profile
NOVEL_EVAL_WINDOW = 5      # steps after novel intervention for accuracy


def evaluate_agent(
    agent: BenchmarkAgent,
    protocol: Optional[CLW1InterventionProtocol] = None,
) -> ScoringMatrix:
    """
    Run the full CLW-1 evaluation suite on an agent.

    Args:
        agent: any system implementing BenchmarkAgent.
        protocol: intervention protocol (uses default if None).

    Returns:
        ScoringMatrix with scores for CLW-1 across all test types.
    """
    if protocol is None:
        protocol = CLW1InterventionProtocol()

    episodes = protocol.get_episodes()
    matrix = ScoringMatrix()

    # ── Test A: Behavioral Recovery ───────────────────────────────────────
    test_a_eps = [ep for ep in episodes if ep.test_type == 'A']
    a_recoveries = []
    for ep in test_a_eps:
        recoveries = _run_episode_test_a(agent, ep)
        a_recoveries.extend(recoveries)

    if a_recoveries:
        mean_recovery = float(np.mean(a_recoveries))
        level_a = classify_level_a(mean_recovery, ENV_NAME)
        confidence_a = 1.0 - float(np.std(a_recoveries)) / max(mean_recovery, 1.0)
        matrix.set_score(ENV_NAME, 'A', TestScore(
            value=mean_recovery, level=level_a,
            confidence=max(0.0, min(1.0, confidence_a)),
        ))

    # ── Test B: Representation Update ─────────────────────────────────────
    test_b_eps = [ep for ep in episodes if ep.test_type == 'B']

    # Full Test B (requires get_representation)
    has_repr = _agent_has_representation(agent)
    if has_repr:
        b_similarities = []
        for ep in test_b_eps:
            sims = _run_episode_test_b_full(agent, ep)
            b_similarities.extend(sims)

        if b_similarities:
            mean_sim = float(np.mean(b_similarities))
            level_b = classify_level_b(mean_sim)
            matrix.set_score(ENV_NAME, 'B-full', TestScore(
                value=mean_sim, level=level_b,
                confidence=max(0.0, 1.0 - float(np.std(b_similarities))),
            ))

    # B-proxy (always available)
    bproxy_results = []
    for ep in test_b_eps:
        result = _run_episode_test_b_proxy(agent, ep)
        if result is not None:
            bproxy_results.append(result)

    if bproxy_results:
        spike_rate = float(np.mean([r.spike_detected for r in bproxy_results]))
        mean_ratio = float(np.mean([r.spike_ratio for r in bproxy_results]))
        matrix.set_score(ENV_NAME, 'B-proxy', TestScore(
            value=mean_ratio, level=2 if spike_rate > 0.5 else 0,
            confidence=spike_rate,
        ))

    # ── Test C: Novel Intervention Generalisation ─────────────────────────
    test_c_eps = [ep for ep in episodes if ep.test_type == 'C']

    # C1: simultaneous
    c1_eps = [ep for ep in test_c_eps if ep.novel_type == 'C1_simultaneous']
    if c1_eps:
        c1_accs = []
        for ep in c1_eps:
            acc = _run_episode_test_c(agent, ep)
            if acc is not None:
                c1_accs.append(acc)
        if c1_accs:
            mean_c1 = float(np.mean(c1_accs))
            matrix.set_score(ENV_NAME, 'C1', TestScore(
                value=mean_c1, level=classify_level_c(mean_c1), confidence=0.8,
            ))

    # C2: no-op
    c2_eps = [ep for ep in test_c_eps if ep.novel_type == 'C2_noop']
    if c2_eps:
        c2_accs = []
        for ep in c2_eps:
            acc = _run_episode_test_c(agent, ep)
            if acc is not None:
                c2_accs.append(acc)
        if c2_accs:
            mean_c2 = float(np.mean(c2_accs))
            matrix.set_score(ENV_NAME, 'C2', TestScore(
                value=mean_c2, level=classify_level_c(mean_c2), confidence=0.8,
            ))

    # C3: OOD — N/A for binary CLW-1
    matrix.set_score(ENV_NAME, 'C3', TestScore(
        value=float('nan'), level=-1, confidence=0.0,
    ))

    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNERS
#
# Design: each runner steps through the episode exactly once in a linear loop.
# Interventions are applied at scheduled steps. Measurement windows consume
# steps from the episode's budget — they do NOT use separate sub-loops that
# could overshoot max_steps.
# ═══════════════════════════════════════════════════════════════════════════════

def _run_episode_test_a(agent: BenchmarkAgent, ep: EpisodeSpec) -> list:
    """Run one Test A episode. Returns list of steps-to-recovery values."""
    env = CLW1Environment(seed=ep.seed, max_steps=200)
    obs = env.reset()
    agent.reset()
    agent.observe(obs)

    recoveries = []
    # Build set of intervention steps for O(1) lookup
    intv_map = {}  # step -> list of InterventionSpec
    for intv in ep.interventions:
        intv_map.setdefault(intv.step, []).append(intv)

    # State for measuring recovery after most recent intervention
    measuring_recovery = False
    recovery_streak = 0
    recovery_steps_since = 0
    required_streak = 3

    step = 0
    while step < env.max_steps and not env.done:
        # Apply any scheduled interventions at this step
        if step in intv_map:
            for intv in intv_map[step]:
                _apply_intervention(env, intv)
            # Start measuring recovery
            measuring_recovery = True
            recovery_streak = 0
            recovery_steps_since = 0

        # Agent acts
        action = agent.act()
        obs, reward, done, info = env.step(action)
        agent.observe(obs)

        # Track recovery if we're in a measurement window
        if measuring_recovery:
            correct_action = env.get_ground_truth()['correct_action']
            if action == correct_action:
                recovery_streak += 1
            else:
                recovery_streak = 0
            recovery_steps_since += 1

            if recovery_streak >= required_streak:
                recoveries.append(
                    recovery_steps_since - required_streak
                )
                measuring_recovery = False
            elif recovery_steps_since >= RECOVERY_WINDOW:
                recoveries.append(RECOVERY_WINDOW)
                measuring_recovery = False

        if done:
            if measuring_recovery:
                recoveries.append(RECOVERY_WINDOW)
            break

        step += 1

    return recoveries


def _run_episode_test_b_full(agent: BenchmarkAgent, ep: EpisodeSpec) -> list:
    """Run one Test B episode (full). Returns list of cosine similarities."""
    env = CLW1Environment(seed=ep.seed, max_steps=200)
    obs = env.reset()
    agent.reset()
    agent.observe(obs)

    similarities = []
    intv_map = {}
    for intv in ep.interventions:
        intv_map.setdefault(intv.step, []).append(intv)

    # Track post-intervention state for representation measurement
    post_intv_expected = None   # expected repr after last intervention
    steps_since_intv = None     # how many steps since last intervention
    repr_steps_set = set(REPR_STEPS)

    step = 0
    while step < env.max_steps and not env.done:
        if step in intv_map:
            for intv in intv_map[step]:
                _apply_intervention(env, intv)
            post_intv_expected = _expected_representation(
                env.get_ground_truth()
            )
            steps_since_intv = 0

        action = agent.act()
        obs, reward, done, info = env.step(action)
        agent.observe(obs)

        if steps_since_intv is not None:
            steps_since_intv += 1
            if steps_since_intv in repr_steps_set:
                rep = agent.get_representation()
                if rep is not None and post_intv_expected is not None:
                    sim = _cosine_similarity(rep, post_intv_expected)
                    similarities.append(sim)
            # Stop tracking after the last measurement step
            if steps_since_intv > max(REPR_STEPS):
                steps_since_intv = None
                post_intv_expected = None

        if done:
            break
        step += 1

    return similarities


def _run_episode_test_b_proxy(
    agent: BenchmarkAgent, ep: EpisodeSpec
) -> Optional[BProxyResult]:
    """Run one Test B-proxy episode. Returns BProxyResult or None."""
    env = CLW1Environment(seed=ep.seed, max_steps=200)
    obs = env.reset()
    proxy = BehavioralProxy(agent)
    proxy.reset()
    proxy.observe(obs)

    first_intervention_step = None
    intv_map = {}
    for intv in ep.interventions:
        intv_map.setdefault(intv.step, []).append(intv)

    step = 0
    while step < env.max_steps and not env.done:
        if step in intv_map:
            for intv in intv_map[step]:
                _apply_intervention(env, intv)
            if first_intervention_step is None:
                first_intervention_step = step

        action = proxy.act()
        obs, reward, done, info = env.step(action)
        proxy.observe(obs)

        if done:
            break
        step += 1

    if first_intervention_step is not None:
        profile = proxy.get_entropy_around_step(
            first_intervention_step, BPROXY_WINDOW
        )
        if len(profile) >= BPROXY_WINDOW:
            return score_bproxy(profile)

    return None


def _run_episode_test_c(
    agent: BenchmarkAgent, ep: EpisodeSpec
) -> Optional[float]:
    """
    Run one Test C episode. Returns accuracy of correct-action predictions
    in the NOVEL_EVAL_WINDOW steps after each novel intervention.
    """
    env = CLW1Environment(seed=ep.seed, max_steps=200)
    obs = env.reset()
    agent.reset()
    agent.observe(obs)

    correct_count = 0
    total_count = 0
    intv_map = {}
    for intv in ep.interventions:
        intv_map.setdefault(intv.step, []).append(intv)

    measuring_accuracy = False
    accuracy_steps = 0

    step = 0
    while step < env.max_steps and not env.done:
        if step in intv_map:
            for intv in intv_map[step]:
                _apply_intervention(env, intv)
            measuring_accuracy = True
            accuracy_steps = 0

        action = agent.act()
        correct_action = env.get_ground_truth()['correct_action']
        obs, reward, done, info = env.step(action)
        agent.observe(obs)

        if measuring_accuracy:
            if action == correct_action:
                correct_count += 1
            total_count += 1
            accuracy_steps += 1
            if accuracy_steps >= NOVEL_EVAL_WINDOW:
                measuring_accuracy = False

        if done:
            break
        step += 1

    if total_count == 0:
        return None
    return correct_count / total_count


# ═══════════════════════════════════════════════════════════════════════════════
# MEASUREMENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_intervention(env: CLW1Environment, intv: InterventionSpec) -> None:
    """Apply one intervention, resolving sentinel values."""
    if intv.value is None:
        # C2 no-op: set to current value
        current = env.get_ground_truth()['C']
        env.intervene(intv.target, current)
    else:
        env.intervene(intv.target, intv.value)


def _expected_representation(ground_truth: dict) -> np.ndarray:
    """
    Create the expected representation for comparison.

    For CLW-1, the ideal representation encodes P(C=0) and P(C=1)
    with high confidence in the true state.
    """
    c = ground_truth['C']
    if c == 0:
        return np.array([0.95, 0.05, 0.0, 1.0], dtype=np.float32)
    else:
        return np.array([0.05, 0.95, 0.0, 1.0], dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _agent_has_representation(agent: BenchmarkAgent) -> bool:
    """Check if agent provides non-None representations."""
    try:
        rep = agent.get_representation()
        return rep is not None
    except (NotImplementedError, AttributeError):
        return False
