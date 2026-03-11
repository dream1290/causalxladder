"""
CLW-2: Evaluation Suite
========================
Runs any BenchmarkAgent through the CLW-2 causal chain tests.

Averaging across both do(C1) and do(C2) ensures the agent can:
1. Handle expected downstream effects (do(C1) -> C2 updates)
2. Handle broken upstream links (do(C2) -> agent must perturb C1 to fix C2)
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
from .env import CLW2Environment
from .interventions import CLW2InterventionProtocol, EpisodeSpec, InterventionSpec


ENV_NAME = 'CLW-2'

RECOVERY_WINDOW = 30       # CLW-2 is harder, give more steps to recover
REPR_STEPS = [1, 3, 5, 10, 30]
BPROXY_WINDOW = 10
NOVEL_EVAL_WINDOW = 10


def evaluate_agent(
    agent: BenchmarkAgent,
    protocol: Optional[CLW2InterventionProtocol] = None,
) -> ScoringMatrix:
    if protocol is None:
        protocol = CLW2InterventionProtocol()

    episodes = protocol.get_episodes()
    matrix = ScoringMatrix()

    # ── Test A: Behavioral Recovery ───────────────────────────────────────
    test_a_eps = [ep for ep in episodes if ep.test_type == 'A']
    a_recoveries = []
    for ep in test_a_eps:
        a_recoveries.extend(_run_episode_test_a(agent, ep))

    if a_recoveries:
        mean_recovery = float(np.mean(a_recoveries))
        level_a = classify_level_a(mean_recovery, ENV_NAME)
        confidence_a = max(0.0, 1.0 - float(np.std(a_recoveries)) / max(mean_recovery, 1.0))
        matrix.set_score(ENV_NAME, 'A', TestScore(mean_recovery, level_a, confidence_a))

    # ── Test B: Representation Update ─────────────────────────────────────
    test_b_eps = [ep for ep in episodes if ep.test_type == 'B']

    if _agent_has_representation(agent):
        b_similarities = []
        for ep in test_b_eps:
            b_similarities.extend(_run_episode_test_b_full(agent, ep))
        if b_similarities:
            mean_sim = float(np.mean(b_similarities))
            matrix.set_score(ENV_NAME, 'B-full', TestScore(
                mean_sim, classify_level_b(mean_sim), max(0.0, 1.0 - float(np.std(b_similarities)))
            ))

    # B-proxy
    bproxy_results = []
    for ep in test_b_eps:
        result = _run_episode_test_b_proxy(agent, ep)
        if result is not None:
            bproxy_results.append(result)

    if bproxy_results:
        spike_rate = float(np.mean([r.spike_detected for r in bproxy_results]))
        mean_ratio = float(np.mean([r.spike_ratio for r in bproxy_results]))
        matrix.set_score(ENV_NAME, 'B-proxy', TestScore(
            mean_ratio, 2 if spike_rate > 0.5 else 0, spike_rate
        ))

    # ── Test C: Novel Interventions ───────────────────────────────────────
    test_c_eps = [ep for ep in episodes if ep.test_type == 'C']

    for novel_type, tag in [('C1_simultaneous', 'C1'), ('C2_noop', 'C2')]:
        eps = [ep for ep in test_c_eps if ep.novel_type == novel_type]
        accs = []
        for ep in eps:
            acc = _run_episode_test_c(agent, ep)
            if acc is not None:
                accs.append(acc)
        if accs:
            mean_acc = float(np.mean(accs))
            matrix.set_score(ENV_NAME, tag, TestScore(
                mean_acc, classify_level_c(mean_acc), 0.8
            ))

    matrix.set_score(ENV_NAME, 'C3', TestScore(float('nan'), -1, 0.0))
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNERS (Linear single-pass design)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_episode_test_a(agent: BenchmarkAgent, ep: EpisodeSpec) -> list:
    env = CLW2Environment(seed=ep.seed, max_steps=200)
    obs = env.reset()
    agent.reset()
    agent.observe(obs)

    recoveries = []
    intv_map = {}
    for intv in ep.interventions:
        intv_map.setdefault(intv.step, []).append(intv)

    measuring = False
    streak = 0
    steps_since = 0

    step = 0
    while step < env.max_steps and not env.done:
        if step in intv_map:
            for intv in intv_map[step]:
                _apply_intervention(env, intv)
            measuring = True
            streak = 0
            steps_since = 0

        action = agent.act()
        obs, _, done, _ = env.step(action)
        agent.observe(obs)

        if measuring:
            if action == env.get_ground_truth()['correct_action']:
                streak += 1
            else:
                streak = 0
            steps_since += 1

            if streak >= 3:
                recoveries.append(steps_since - 3)
                measuring = False
            elif steps_since >= RECOVERY_WINDOW:
                recoveries.append(RECOVERY_WINDOW)
                measuring = False

        if done:
            if measuring:
                recoveries.append(RECOVERY_WINDOW)
            break
        step += 1
    return recoveries


def _run_episode_test_b_full(agent: BenchmarkAgent, ep: EpisodeSpec) -> list:
    env = CLW2Environment(seed=ep.seed, max_steps=200)
    obs = env.reset()
    agent.reset()
    agent.observe(obs)

    sims = []
    intv_map = {}
    for intv in ep.interventions:
        intv_map.setdefault(intv.step, []).append(intv)

    expected_repr = None
    steps_since = None
    target_steps = set(REPR_STEPS)

    step = 0
    while step < env.max_steps and not env.done:
        if step in intv_map:
            for intv in intv_map[step]:
                _apply_intervention(env, intv)
            expected_repr = _expected_representation(env.get_ground_truth())
            steps_since = 0

        action = agent.act()
        obs, _, done, _ = env.step(action)
        agent.observe(obs)

        if steps_since is not None:
            steps_since += 1
            if steps_since in target_steps:
                rep = agent.get_representation()
                if rep is not None and expected_repr is not None:
                    sims.append(_cosine_similarity(rep, expected_repr))
            if steps_since > max(REPR_STEPS):
                steps_since = None

        if done: break
        step += 1
    return sims


def _run_episode_test_b_proxy(agent: BenchmarkAgent, ep: EpisodeSpec) -> Optional[BProxyResult]:
    env = CLW2Environment(seed=ep.seed, max_steps=200)
    obs = env.reset()
    proxy = BehavioralProxy(agent)
    proxy.reset()
    proxy.observe(obs)

    first_intv_step = None
    intv_map = {}
    for intv in ep.interventions:
        intv_map.setdefault(intv.step, []).append(intv)

    step = 0
    while step < env.max_steps and not env.done:
        if step in intv_map:
            for intv in intv_map[step]:
                _apply_intervention(env, intv)
            if first_intv_step is None:
                first_intv_step = step

        action = proxy.act()
        obs, _, done, _ = env.step(action)
        proxy.observe(obs)

        if done: break
        step += 1

    if first_intv_step is not None:
        profile = proxy.get_entropy_around_step(first_intv_step, BPROXY_WINDOW)
        if len(profile) >= BPROXY_WINDOW:
            return score_bproxy(profile)
    return None


def _run_episode_test_c(agent: BenchmarkAgent, ep: EpisodeSpec) -> Optional[float]:
    env = CLW2Environment(seed=ep.seed, max_steps=200)
    obs = env.reset()
    agent.reset()
    agent.observe(obs)

    correct, total = 0, 0
    intv_map = {}
    for intv in ep.interventions:
        intv_map.setdefault(intv.step, []).append(intv)

    measuring = False
    steps = 0

    step = 0
    while step < env.max_steps and not env.done:
        if step in intv_map:
            for intv in intv_map[step]:
                _apply_intervention(env, intv)
            measuring = True
            steps = 0

        action = agent.act()
        target = env.get_ground_truth()['correct_action']
        obs, _, done, _ = env.step(action)
        agent.observe(obs)

        if measuring:
            if action == target: correct += 1
            total += 1
            steps += 1
            if steps >= NOVEL_EVAL_WINDOW:
                measuring = False

        if done: break
        step += 1

    return correct / total if total > 0 else None


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_intervention(env: CLW2Environment, intv: InterventionSpec) -> None:
    if intv.value is None:
        current = env.get_ground_truth()[intv.target]
        env.intervene(intv.target, current)
    else:
        env.intervene(intv.target, intv.value)

def _expected_representation(gt: dict) -> np.ndarray:
    """Expected representation: [Target, C1, C2, 1.0]."""
    return np.array([gt['Target'], gt['C1'], gt['C2'], 1.0], dtype=np.float32)

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Pad or truncate generic agent representations safely
    if len(a) < 4:
        a = np.pad(a, (0, 4 - len(a)))
    elif len(a) > 4:
        a = a[:4]
    a_f, b_f = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    norm_a, norm_b = np.linalg.norm(a_f), np.linalg.norm(b_f)
    return 0.0 if norm_a < 1e-10 or norm_b < 1e-10 else float(np.dot(a_f, b_f) / (norm_a * norm_b))

def _agent_has_representation(agent: BenchmarkAgent) -> bool:
    try:
        return agent.get_representation() is not None
    except (NotImplementedError, AttributeError):
        return False
