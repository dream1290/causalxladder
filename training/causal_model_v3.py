"""
Causal Belief Model — v3 (Separated World Model / Policy)

Core hypothesis: architectural separation of world model and policy is a
necessary condition for Level 2 causal inference under identical pressure.

Architecture:
  world_gru:   GRUCell(4, 128)   — tracks hidden state C from observations
  policy_gru:  GRUCell(132, 64)  — reads z_w (detached) + obs for decisions
  action_head: Linear(192→64→3)  — reads [z_w_detached, z_p]
  value_head:  Linear(192→64→1)  — reads [z_w_detached, z_p]

  ═══ GRADIENT WALL ═══
  z_w.detach() feeds into policy_gru and heads.
  Policy gradients NEVER flow into world_gru.
  Tracking loss gradients NEVER flow into policy_gru.

Training phases:
  Phase A  (25k eps): episodic z_w + z_p. Streak 3→8 curriculum.
  Phase B  (15k eps): persistent z_w, episodic z_p. The key innovation.
  Phase C  (12k eps): persistent z_w + z_p. Full persistence.

Usage:
  python causal_model_v3.py                  # full run (~52k episodes)
  python causal_model_v3.py --phase-a-only   # test Phase A alone
  python causal_model_v3.py --phase-b-only   # load Phase A checkpoint
  python causal_model_v3.py --phase-c-only   # load Phase B checkpoint
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os, sys, time


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIMULATOR  (identical to v2 — same pressure, same environment)
# ══════════════════════════════════════════════════════════════════════════════
class CausalSimulator:
    def __init__(self, hidden_dim=8, flip_mean=80, max_steps=200,
                 big_reward=10.0, big_penalty=-30.0,
                 step_cost=-0.01, pull_cost=-0.05, wrong_streak=6):
        self.hidden_dim  = hidden_dim
        self.flip_mean   = flip_mean
        self.max_steps   = max_steps
        self.big_reward  = big_reward
        self.big_penalty = big_penalty
        self.step_cost   = step_cost
        self.pull_cost   = pull_cost
        self.wrong_streak = wrong_streak
        self.total_steps  = 0
        self.streak_cap   = 8
        self.shaped_correct_floor = None

    @property
    def streak_required(self):
        if self.total_steps < 500_000:   base = 3
        elif self.total_steps < 1_500_000: base = 5
        else: base = 8
        return min(base, self.streak_cap)

    @property
    def shaped_correct(self):
        if self.total_steps < 500_000:   base = 1.0
        elif self.total_steps < 1_500_000: base = 0.5
        else: base = 0.2
        if self.shaped_correct_floor is not None:
            return max(base, self.shaped_correct_floor)
        return base

    def reset(self):
        self.C               = np.random.randint(0, 2, self.hidden_dim).astype(float)
        self.steps           = 0
        self.correct_consec  = 0
        self.wrong_consec    = 0
        self.flip_timers     = np.random.geometric(1.0 / self.flip_mean, self.hidden_dim)
        self._last_correct   = 0.0
        return self._get_obs()

    def step(self, action):
        reward = self.step_cost
        done   = False
        self.total_steps += 1

        if action == 2:
            self.steps += 1
            self.correct_consec = self.wrong_consec = 0
            if self.steps >= self.max_steps:
                done = True
            return self._get_obs(), reward, done

        majority       = 1 if np.sum(self.C) > self.hidden_dim // 2 else 0
        corr_flip      = (self.C[0] == 1 and self.C[1] == 1)
        correct_action = 1 - majority if corr_flip else majority
        is_correct     = (action == correct_action)
        self._last_correct = 1.0 if is_correct else 0.0

        reward += self.pull_cost
        if is_correct:
            reward += self.shaped_correct
            self.correct_consec += 1
            self.wrong_consec    = 0
            if self.correct_consec >= self.streak_required:
                reward += self.big_reward
        else:
            reward += -0.2
            self.wrong_consec   += 1
            self.correct_consec  = 0
            if self.wrong_consec >= self.wrong_streak:
                reward += self.big_penalty
                done = True

        self.steps += 1
        self.flip_timers -= 1
        for i in range(self.hidden_dim):
            if self.flip_timers[i] <= 0:
                self.C[i]           = 1 - self.C[i]
                self.flip_timers[i] = np.random.geometric(1.0 / self.flip_mean)

        if self.steps >= self.max_steps:
            done = True
        return self._get_obs(), reward, done

    def _get_obs(self):
        return np.array([
            0.0,
            self._last_correct,
            min(self.steps / 50.0, 1.0),
            float(np.random.randn()),
        ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL — Separated World Model / Policy
# ══════════════════════════════════════════════════════════════════════════════
class SeparatedCausalModel(nn.Module):
    """
    Two-module architecture with one-way information flow.

    Information flow:
        obs ──→ world_gru ──→ z_w_new ──→ [DETACH] ──→ policy_gru ──→ z_p_new
                                  │                          │
                                  │              [z_w_detached, z_p] ──→ heads
                                  │
                          tracking loss (→ world_gru grads only)

    The detach() is the gradient wall. Policy loss gradients flow through
    z_p and heads but NEVER through z_w or world_gru.
    """

    def __init__(self):
        super().__init__()
        # World model: obs → z_w  (tracks hidden state C)
        self.world_gru = nn.GRUCell(4, 128)

        # Policy: [z_w_detached, obs] → z_p  (reads world model for decisions)
        self.policy_gru = nn.GRUCell(128 + 4, 64)

        # Heads read concatenated [z_w_detached, z_p] = 192-dim
        self.action_head = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.value_head = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, obs_t, z_w_prev, z_p_prev):
        # ── World model update ───────────────────────────────────────
        z_w_new = self.world_gru(obs_t, z_w_prev)
        delta_z_w = ((z_w_new - z_w_prev) ** 2).sum(dim=-1)

        # ═══════════════ GRADIENT WALL ═══════════════
        z_w_detached = z_w_new.detach()

        # ── Policy update ────────────────────────────────────────────
        policy_input = torch.cat([z_w_detached, obs_t], dim=-1)  # 132-dim
        z_p_new = self.policy_gru(policy_input, z_p_prev)

        # ── Heads ────────────────────────────────────────────────────
        combined = torch.cat([z_w_detached, z_p_new], dim=-1)    # 192-dim
        logits = self.action_head(combined)
        value  = self.value_head(combined).squeeze(-1)

        return logits, value, z_w_new, z_p_new, delta_z_w

    def init_world_belief(self):
        return torch.zeros(1, 128)

    def init_policy_belief(self):
        return torch.zeros(1, 64)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SHARED EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_episode(env, model, z_w, z_p, persistent_w=False, persistent_p=False):
    obs_np      = env.reset()
    done        = False
    last_action = 0

    if not persistent_w:
        z_w = model.init_world_belief()
    if not persistent_p:
        z_p = model.init_policy_belief()

    log_probs, values, rewards = [], [], []
    dz_w_vals, entropies, pull_flags, ep_dz_w = [], [], [], []
    snap_z_w, snap_label = None, None

    while not done:
        obs_np[0] = last_action / 2.0
        obs_t     = torch.FloatTensor(obs_np).unsqueeze(0)

        logits, val, z_w_new, z_p_new, dz_w = model(obs_t, z_w, z_p)
        dist   = Categorical(logits=logits)
        action = dist.sample()

        obs_np, reward, done = env.step(action.item())
        last_action = action.item()

        log_probs.append(dist.log_prob(action))
        values.append(val)
        rewards.append(reward)
        dz_w_vals.append(dz_w)
        entropies.append(dist.entropy())
        pull_flags.append(1.0 if action.item() < 2 else 0.0)
        ep_dz_w.append(dz_w.item())

        z_w = z_w_new.detach()
        z_p = z_p_new.detach()

    snap_z_w   = z_w.squeeze(0).numpy().copy()
    snap_label = 1 if np.sum(env.C) > env.hidden_dim // 2 else 0

    return (log_probs, values, rewards, dz_w_vals, entropies, pull_flags,
            ep_dz_w, z_w, z_p, snap_z_w, snap_label)


# ══════════════════════════════════════════════════════════════════════════════
# 4. INCENTIVE STRUCTURE  (same coefficients as v2)
# ══════════════════════════════════════════════════════════════════════════════
TRACKING_COEFF  = 0.1
TRACKING_CAP    = 0.5
PHASE_A_M_COEFF = 0.005
GUARD_M_COEFF   = 0.15
GUARD_THRESHOLD = 0.5
ENTROPY_COEFF   = -0.005

def explosion_guard(avg_dz):
    if avg_dz > GUARD_THRESHOLD:
        overshoot = (avg_dz - GUARD_THRESHOLD) / GUARD_THRESHOLD
        return GUARD_M_COEFF * min(overshoot, 1.0)
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. PHASE A — Episodic z_w + z_p, streak 3→8
# ══════════════════════════════════════════════════════════════════════════════
def phase_a(model, env, n_episodes=25_000, print_every=500):
    """Both z_w and z_p reset each episode. Same streak curriculum as v2."""

    STREAK_SCHEDULE = [
        (5_000,   3),
        (8_000,   4),
        (12_000,  5),
        (17_000,  6),
        (21_000,  7),
    ]

    print("\nPHASE A — Episodic training (z_w and z_p reset each episode)")
    print("  Curriculum: 3(0-5k) → 4(5k-8k) → 5(8k-12k) → 6(12k-17k) → 7(17k-21k) → 8(21k-25k)")
    print("  shaped_correct locked at 0.5")
    print(f"  Tracking: {TRACKING_COEFF}  |  m_ceiling: {PHASE_A_M_COEFF}  |  Entropy: {ENTROPY_COEFF}")

    env.shaped_correct_floor = 0.5

    world_params  = list(model.world_gru.parameters())
    policy_params = (list(model.policy_gru.parameters()) +
                     list(model.action_head.parameters()) +
                     list(model.value_head.parameters()))
    opt = optim.Adam([
        {'params': world_params,  'lr': 2e-4},
        {'params': policy_params, 'lr': 5e-4},
    ])

    ema_baseline = 0.0
    ema_alpha    = 0.05
    ema_dz       = 0.0
    ep_rewards   = []
    dz_norms     = []
    z_snapshots  = []
    z_labels     = []
    t_start      = time.time()

    for ep in range(n_episodes):
        env.streak_cap = 8
        for threshold, streak in STREAK_SCHEDULE:
            if ep < threshold:
                env.streak_cap = streak
                break

        lp, vals, rews, dzv, ents, pulls, ep_dz, z_w, z_p, sz, sl = run_episode(
            env, model, None, None, persistent_w=False, persistent_p=False
        )

        ema_baseline = (1 - ema_alpha) * ema_baseline + ema_alpha * np.mean(rews)
        avg_dz = float(np.mean(ep_dz))
        ema_dz = 0.95 * ema_dz + 0.05 * avg_dz

        G = 0.0
        returns = []
        for r in reversed(rews):
            G = r + 0.97 * G
            returns.insert(0, G)
        returns  = torch.FloatTensor(returns)
        values_t = torch.stack(vals)
        adv = returns - ema_baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dz_stack  = torch.stack(dzv)
        pull_mask = torch.FloatTensor(pulls)

        p_loss = -(torch.stack(lp) * adv.detach()).mean()
        v_loss = 0.5 * ((values_t - returns) ** 2).mean()
        m_loss = PHASE_A_M_COEFF * dz_stack.mean()
        t_loss = -TRACKING_COEFF * (torch.clamp(dz_stack, max=TRACKING_CAP) * pull_mask).mean()
        e_loss = ENTROPY_COEFF * torch.stack(ents).mean()
        loss   = p_loss + v_loss + m_loss + t_loss + e_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(world_params, 1.0)
        nn.utils.clip_grad_norm_(policy_params, 2.0)
        opt.step()

        ep_rewards.append(sum(rews))
        dz_norms.append(avg_dz)
        if ep % 200 == 0 and len(z_snapshots) < 1000:
            z_snapshots.append(sz)
            z_labels.append(sl)

        if (ep + 1) % print_every == 0:
            rr  = np.mean(ep_rewards[-print_every:])
            rdz = np.mean(dz_norms[-print_every:])
            elapsed = int(time.time() - t_start)
            print(f"  A ep {ep+1:6d} | reward {rr:8.1f} | "
                  f"||Δz_w|| {rdz:.4f} | "
                  f"streak={env.streak_cap} | time {elapsed}s")

    final_r = np.mean(ep_rewards[-1000:])
    print(f"\nPhase A complete. Final avg reward: {final_r:.1f}")
    return ep_rewards, dz_norms, z_snapshots, z_labels


# ══════════════════════════════════════════════════════════════════════════════
# 6. PHASE B — Persistent z_w, episodic z_p
# ══════════════════════════════════════════════════════════════════════════════
def phase_b(model, env, n_episodes=15_000, print_every=500):
    """
    The key innovation. z_w persists across episodes, z_p resets each time.
    Forces z_w to be the sole carrier of cross-episode causal information.
    Policy learns to READ z_w, not to encode world state itself.
    """
    B0_END =  3_000   # warmup: world_gru only
    B1_END =  8_000   # consolidate: world + policy GRUs, heads frozen
    # B2: 8k-15k, everything unfrozen

    print("\nPHASE B — Persistent z_w, episodic z_p (the separation test)")
    print(f"  B0 (0-{B0_END//1000}k):    warmup, only world_gru trains")
    print(f"  B1 ({B0_END//1000}k-{B1_END//1000}k):  world + policy GRUs, heads frozen")
    print(f"  B2 ({B1_END//1000}k+):  everything unfrozen")
    print(f"  sc_floor=0.5 throughout")
    print(f"  Tracking: {TRACKING_COEFF}  |  Entropy: {ENTROPY_COEFF}")

    # Start frozen: only world_gru
    for p in model.policy_gru.parameters(): p.requires_grad = False
    for p in model.action_head.parameters(): p.requires_grad = False
    for p in model.value_head.parameters():  p.requires_grad = False

    world_params = list(model.world_gru.parameters())
    opt = optim.Adam(world_params, lr=5e-5)

    env.streak_cap = 8
    env.shaped_correct_floor = 0.5

    ema_baseline    = 0.0
    ema_alpha       = 0.03
    ema_dz          = 0.0
    ep_rewards      = []
    dz_norms        = []
    z_snapshots     = []
    z_labels        = []
    t_start         = time.time()
    z_w             = model.init_world_belief()
    policy_unfrozen = False
    heads_unfrozen  = False
    policy_params   = None
    head_params     = None

    for ep in range(n_episodes):
        # ── Sub-phase transitions ───────────────────────────────────
        if ep == B0_END and not policy_unfrozen:
            for p in model.policy_gru.parameters(): p.requires_grad = True
            policy_unfrozen = True
            policy_params = list(model.policy_gru.parameters())
            opt = optim.Adam([
                {'params': world_params,  'lr': 5e-5},
                {'params': policy_params, 'lr': 3e-4},
            ])
            print(f"\n  ── B1 start: policy_gru unfrozen (lr=3e-4) ──")

        if ep == B1_END and not heads_unfrozen:
            for p in model.action_head.parameters(): p.requires_grad = True
            for p in model.value_head.parameters():  p.requires_grad = True
            heads_unfrozen = True
            head_params = (list(model.action_head.parameters()) +
                           list(model.value_head.parameters()))
            opt = optim.Adam([
                {'params': world_params,  'lr': 5e-5},
                {'params': policy_params, 'lr': 2e-4},
                {'params': head_params,   'lr': 3e-5},
            ])
            print(f"\n  ── B2 start: heads unfrozen (lr=3e-5) ──")

        env.shaped_correct_floor = 0.5

        # ── Episode: z_w persists, z_p resets ────────────────────────
        lp, vals, rews, dzv, ents, pulls, ep_dz, z_w, z_p, sz, sl = run_episode(
            env, model, z_w, None, persistent_w=True, persistent_p=False
        )

        ema_baseline = (1 - ema_alpha) * ema_baseline + ema_alpha * np.mean(rews)
        avg_dz = float(np.mean(ep_dz))
        ema_dz = 0.95 * ema_dz + 0.05 * avg_dz

        G = 0.0
        returns = []
        for r in reversed(rews):
            G = r + 0.97 * G
            returns.insert(0, G)
        returns  = torch.FloatTensor(returns)
        values_t = torch.stack(vals)
        adv = returns - ema_baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dz_stack  = torch.stack(dzv)
        pull_mask = torch.FloatTensor(pulls)
        guard     = explosion_guard(ema_dz)

        p_loss = -(torch.stack(lp) * adv.detach()).mean()
        v_loss = 0.5 * ((values_t - returns) ** 2).mean()
        t_loss = -TRACKING_COEFF * (torch.clamp(dz_stack, max=TRACKING_CAP) * pull_mask).mean()
        g_loss = guard * dz_stack.mean()
        e_loss = ENTROPY_COEFF * torch.stack(ents).mean()
        loss   = p_loss + v_loss + t_loss + g_loss + e_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(world_params, 0.5)
        if policy_unfrozen:
            nn.utils.clip_grad_norm_(policy_params, 1.0)
        if heads_unfrozen:
            nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()

        ep_rewards.append(sum(rews))
        dz_norms.append(avg_dz)
        if ep % 200 == 0 and len(z_snapshots) < 1000:
            z_snapshots.append(sz)
            z_labels.append(sl)

        if (ep + 1) % print_every == 0:
            rr  = np.mean(ep_rewards[-print_every:])
            rdz = np.mean(dz_norms[-print_every:])
            elapsed = int(time.time() - t_start)

            if   ep < B0_END: tag = "B0"
            elif ep < B1_END: tag = "B1"
            else:             tag = "B2"

            g_tag = "⚡" if guard > 0 else " "
            print(f"  {tag} ep {ep+1:6d} | reward {rr:8.1f} | "
                  f"||Δz_w|| {rdz:.4f} {g_tag}| time {elapsed}s")

    final_r  = np.mean(ep_rewards[-1000:])
    final_dz = np.mean(dz_norms[-1000:])
    print(f"\nPhase B complete. Final avg reward: {final_r:.1f}")
    print(f"Final avg ||Δz_w||: {final_dz:.4f}")
    return ep_rewards, dz_norms, z_snapshots, z_labels, z_w


# ══════════════════════════════════════════════════════════════════════════════
# 7. PHASE C — Full persistence (z_w + z_p both persist)
# ══════════════════════════════════════════════════════════════════════════════
def phase_c(model, env, z_w, n_episodes=12_000, print_every=500):
    """
    Both z_w and z_p persist across episodes.
    Evaluate ||Δz_w|| responsiveness — expect clean state tracking from
    the world model, with stable policy behavior from z_p.
    """
    C0_END = 4_000   # warmup: heads frozen
    # C1: 4k-12k, full training

    print("\nPHASE C — Full persistence (z_w and z_p persist)")
    print(f"  C0 (0-{C0_END//1000}k):    warmup, heads frozen")
    print(f"  C1 ({C0_END//1000}k+):  full training")
    print(f"  sc_floor=0.5 throughout")

    for p in model.action_head.parameters(): p.requires_grad = False
    for p in model.value_head.parameters():  p.requires_grad = False

    world_params  = list(model.world_gru.parameters())
    policy_params = list(model.policy_gru.parameters())
    opt = optim.Adam([
        {'params': world_params,  'lr': 3e-5},
        {'params': policy_params, 'lr': 1e-4},
    ])

    env.streak_cap = 8
    env.shaped_correct_floor = 0.5

    ema_baseline   = 0.0
    ema_alpha      = 0.03
    ema_dz         = 0.0
    ep_rewards     = []
    dz_norms       = []
    z_snapshots    = []
    z_labels       = []
    t_start        = time.time()
    z_p            = model.init_policy_belief()
    heads_unfrozen = False
    head_params    = None

    for ep in range(n_episodes):
        if ep == C0_END and not heads_unfrozen:
            for p in model.action_head.parameters(): p.requires_grad = True
            for p in model.value_head.parameters():  p.requires_grad = True
            heads_unfrozen = True
            head_params = (list(model.action_head.parameters()) +
                           list(model.value_head.parameters()))
            opt = optim.Adam([
                {'params': world_params,  'lr': 3e-5},
                {'params': policy_params, 'lr': 5e-5},
                {'params': head_params,   'lr': 2e-5},
            ])
            print(f"\n  ── C1 start: heads unfrozen ──")

        env.shaped_correct_floor = 0.5

        # ── Episode: both persist ────────────────────────────────────
        lp, vals, rews, dzv, ents, pulls, ep_dz, z_w, z_p, sz, sl = run_episode(
            env, model, z_w, z_p, persistent_w=True, persistent_p=True
        )

        ema_baseline = (1 - ema_alpha) * ema_baseline + ema_alpha * np.mean(rews)
        avg_dz = float(np.mean(ep_dz))
        ema_dz = 0.95 * ema_dz + 0.05 * avg_dz

        G = 0.0
        returns = []
        for r in reversed(rews):
            G = r + 0.97 * G
            returns.insert(0, G)
        returns  = torch.FloatTensor(returns)
        values_t = torch.stack(vals)
        adv = returns - ema_baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dz_stack  = torch.stack(dzv)
        pull_mask = torch.FloatTensor(pulls)
        guard     = explosion_guard(ema_dz)

        p_loss = -(torch.stack(lp) * adv.detach()).mean()
        v_loss = 0.5 * ((values_t - returns) ** 2).mean()
        t_loss = -TRACKING_COEFF * (torch.clamp(dz_stack, max=TRACKING_CAP) * pull_mask).mean()
        g_loss = guard * dz_stack.mean()
        e_loss = ENTROPY_COEFF * torch.stack(ents).mean()
        loss   = p_loss + v_loss + t_loss + g_loss + e_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(world_params, 0.5)
        nn.utils.clip_grad_norm_(policy_params, 0.5)
        if heads_unfrozen:
            nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()

        ep_rewards.append(sum(rews))
        dz_norms.append(avg_dz)
        if ep % 200 == 0 and len(z_snapshots) < 1000:
            z_snapshots.append(sz)
            z_labels.append(sl)

        if (ep + 1) % print_every == 0:
            rr  = np.mean(ep_rewards[-print_every:])
            rdz = np.mean(dz_norms[-print_every:])
            elapsed = int(time.time() - t_start)

            tag   = "C0" if ep < C0_END else "C1"
            g_tag = "⚡" if guard > 0 else " "
            print(f"  {tag} ep {ep+1:6d} | reward {rr:8.1f} | "
                  f"||Δz_w|| {rdz:.4f} {g_tag}| time {elapsed}s")

    final_r  = np.mean(ep_rewards[-1000:])
    final_dz = np.mean(dz_norms[-1000:])
    print(f"\nPhase C complete. Final avg reward: {final_r:.1f}")
    print(f"Final avg ||Δz_w||: {final_dz:.4f}")
    return ep_rewards, dz_norms, z_snapshots, z_labels


# ══════════════════════════════════════════════════════════════════════════════
# 8. PCA FITTING — for benchmark representation extraction
# ══════════════════════════════════════════════════════════════════════════════
def fit_representation_pca(z_snapshots, z_labels):
    """
    Fit PCA on z_w snapshots and determine orientation.
    Returns dict suitable for saving alongside model checkpoint.
    """
    Z   = np.array(z_snapshots)
    lbl = np.array(z_labels)

    pca = PCA(n_components=2)
    Z2  = pca.fit_transform(Z)

    # Determine sign: PC1 should correlate positively with majority=1
    corr = np.corrcoef(Z2[:, 0], lbl)[0, 1]
    sign = 1.0 if corr >= 0 else -1.0

    # Determine scale: fit a simple logistic scaling
    # Map PC1 to [0, 1] via sigmoid(scale * PC1)
    pc1_signed = Z2[:, 0] * sign
    # Use std-based scaling: sigmoid(x / std) maps ±1σ to ~[0.27, 0.73]
    scale = 1.0 / max(np.std(pc1_signed), 1e-6)

    print(f"\n  PCA fit: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")
    print(f"  Correlation(PC1, label): {corr:.3f} (sign={sign:+.0f})")
    print(f"  Scale: {scale:.3f}")

    return {
        'components': pca.components_,
        'mean':       pca.mean_,
        'sign':       sign,
        'scale':      scale,
        'explained_variance_ratio': pca.explained_variance_ratio_,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
def smooth(x, w=300):
    if len(x) < w: return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='valid')

def plot_diagnostics(a_rew, a_dz, b_rew, b_dz, c_rew, c_dz,
                     c_snaps, c_lbls):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes.ravel()

    # Phase A reward
    ax[0].plot(smooth(a_rew), color="#1F3864")
    ax[0].axhline(0, color="gray", linestyle="--", lw=0.8)
    ax[0].set_title("Phase A — Reward (episodic, streak 3→8)", fontweight="bold")
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Reward"); ax[0].grid(alpha=0.3)

    # Phase B + C reward
    if b_rew:
        ax[1].plot(smooth(b_rew), color="#1F6420", alpha=0.7, label="Phase B")
    if c_rew:
        offset = len(smooth(b_rew)) if b_rew else 0
        ax[1].plot(range(offset, offset + len(smooth(c_rew))),
                   smooth(c_rew), color="#2980B9", alpha=0.7, label="Phase C")
    ax[1].axhline(0, color="gray", linestyle="--", lw=0.8)
    ax[1].set_title("Phase B+C — Reward (persistent z_w)", fontweight="bold")
    ax[1].set_xlabel("Episode"); ax[1].set_ylabel("Reward")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    # ||Δz_w|| across all phases
    offset = 0
    if a_dz:
        ax[2].plot(smooth(a_dz), color="#C0392B", label="A (episodic)", alpha=0.6)
        offset = len(smooth(a_dz))
    if b_dz:
        ax[2].plot(range(offset, offset + len(smooth(b_dz))),
                   smooth(b_dz), color="#2980B9", label="B (z_w persists)", alpha=0.6)
        offset += len(smooth(b_dz))
    if c_dz:
        ax[2].plot(range(offset, offset + len(smooth(c_dz))),
                   smooth(c_dz), color="#27AE60", label="C (both persist)", alpha=0.6)
    ax[2].axhspan(0.15, 0.60, alpha=0.08, color="green", label="Target zone")
    ax[2].set_title("World Model Responsiveness ||Δz_w||", fontweight="bold")
    ax[2].set_xlabel("Episode"); ax[2].set_ylabel("||Δz_w||")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    # PCA of z_w
    ax[3].set_title("Phase C — PCA of z_w", fontweight="bold")
    if len(c_snaps) >= 50:
        Z   = np.array(c_snaps)
        lbl = np.array(c_lbls)
        pca = PCA(n_components=2)
        Z2  = pca.fit_transform(Z)
        sc  = ax[3].scatter(Z2[:,0], Z2[:,1], c=lbl, cmap="RdBu", alpha=0.4, s=8)
        plt.colorbar(sc, ax=ax[3], label="Majority vote of C")
        var = pca.explained_variance_ratio_
        ax[3].set_title(f"Phase C — PCA of z_w\n"
                        f"PC1={var[0]:.1%}, PC2={var[1]:.1%}", fontweight="bold")
    ax[3].set_xlabel("PC1"); ax[3].set_ylabel("PC2"); ax[3].grid(alpha=0.3)

    plt.suptitle(
        "Causal Belief Model v3 — Separated World Model / Policy\n"
        "A: episodic | B: persistent z_w, episodic z_p | C: fully persistent",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("causal_model_v3_diagnostics.png", dpi=150)
    print("Saved → causal_model_v3_diagnostics.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    phase_a_only = "--phase-a-only" in sys.argv
    phase_b_only = "--phase-b-only" in sys.argv
    phase_c_only = "--phase-c-only" in sys.argv

    print("═" * 60)
    print("Causal Belief Model v3 — Separated World Model / Policy")
    if phase_a_only:
        print("MODE: Phase A only (streak 3→8 test)")
    elif phase_b_only:
        print("MODE: Phase B only (loading Phase A checkpoint)")
    elif phase_c_only:
        print("MODE: Phase C only (loading Phase B checkpoint)")
    else:
        print("Phase A: 25k | Phase B: 15k (z_w persists) | Phase C: 12k (both)")
    print("═" * 60)

    env   = CausalSimulator(hidden_dim=8, flip_mean=80)
    model = SeparatedCausalModel()

    # Parameter count
    total_p = sum(p.numel() for p in model.parameters())
    world_p = sum(p.numel() for p in model.world_gru.parameters())
    policy_p = (sum(p.numel() for p in model.policy_gru.parameters()) +
                sum(p.numel() for p in model.action_head.parameters()) +
                sum(p.numel() for p in model.value_head.parameters()))
    print(f"Total parameters: {total_p:,}")
    print(f"  World model (world_gru):     {world_p:,}")
    print(f"  Policy (policy_gru + heads): {policy_p:,}")

    # ── Load checkpoints for partial runs ────────────────────────────
    a_rewards, a_dz = [], []

    if phase_b_only:
        ckpt = torch.load("v3_phase_a_checkpoint.pt", weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        env.total_steps = 2_000_000
        print("Loaded Phase A checkpoint")

    elif phase_c_only:
        ckpt = torch.load("v3_phase_b_checkpoint.pt", weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        env.total_steps = 2_000_000
        print("Loaded Phase B checkpoint")

    # ── PHASE A ──────────────────────────────────────────────────────
    if not phase_b_only and not phase_c_only:
        a_rewards, a_dz, _, _ = phase_a(model, env, n_episodes=25_000)
        torch.save({'model_state_dict': model.state_dict()},
                   "v3_phase_a_checkpoint.pt")
        print("Saved → v3_phase_a_checkpoint.pt")

    if phase_a_only:
        final_r = np.mean(a_rewards[-1000:])
        print(f"\n{'═'*60}")
        print(f"Phase A test complete. Final avg reward: {final_r:.1f}")
        print(f"Final avg ||Δz_w||: {np.mean(a_dz[-1000:]):.4f}")
        if final_r > 0:
            print("→ POSITIVE reward at streak=8. Ready for Phase B.")
        else:
            print("→ Negative reward. Needs adjustment.")
        print(f"{'═'*60}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(smooth(a_rewards), color="#1F3864")
        ax1.axhline(0, color="gray", linestyle="--", lw=0.8)
        ax1.set_title("Phase A — Reward"); ax1.set_xlabel("Episode"); ax1.grid(alpha=0.3)
        ax2.plot(smooth(a_dz), color="#C0392B")
        ax2.set_title("Phase A — ||Δz_w||"); ax2.set_xlabel("Episode"); ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("v3_phase_a_test.png", dpi=150)
        print("Saved → v3_phase_a_test.png")
        sys.exit(0)

    # ── PHASE B ──────────────────────────────────────────────────────
    b_rewards, b_dz, b_snaps, b_lbls = [], [], [], []
    z_w_carry = None

    if not phase_c_only:
        b_rewards, b_dz, b_snaps, b_lbls, z_w_carry = phase_b(
            model, env, n_episodes=15_000
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'z_w': z_w_carry.detach().numpy(),
        }, "v3_phase_b_checkpoint.pt")
        print("Saved → v3_phase_b_checkpoint.pt")

    # ── PHASE C ──────────────────────────────────────────────────────
    if phase_c_only:
        z_w_carry = torch.FloatTensor(ckpt.get('z_w', np.zeros((1, 128))))

    c_rewards, c_dz, c_snaps, c_lbls = phase_c(
        model, env, z_w_carry, n_episodes=12_000
    )

    # ── Fit PCA on final z_w snapshots ───────────────────────────────
    all_snaps  = b_snaps + c_snaps
    all_labels = b_lbls  + c_lbls
    pca_data = {}
    if len(all_snaps) >= 100:
        pca_data = fit_representation_pca(all_snaps, all_labels)

    # ── Save final checkpoint ────────────────────────────────────────
    torch.save({
        'model_state_dict': model.state_dict(),
        'pca': pca_data,
    }, "causal_belief_v3_final.pt")
    print("Saved → causal_belief_v3_final.pt")

    # ── Diagnostics ──────────────────────────────────────────────────
    plot_diagnostics(a_rewards, a_dz, b_rewards, b_dz,
                     c_rewards, c_dz, c_snaps, c_lbls)

    final_r  = np.mean(c_rewards[-2000:])
    final_dz = np.mean(c_dz[-2000:])
    print(f"\nFinal avg reward  : {final_r:.1f}")
    print(f"Final avg ||Δz_w||: {final_dz:.4f}")

    if final_r > 0 and 0.10 <= final_dz <= 0.60:
        print("\n→ READY FOR INTERVENTION TEST: python intervention_test_v3.py")
    else:
        print("\n→ Stabilize further before intervention test")
