"""
Causal Belief Model — v2 (Clean Architecture)

Design philosophy: ONE VARIABLE AT A TIME.

Phase A  (25k eps, z RESET each episode):
  Teach the full streak=3→8 policy. The model learns to infer C from
  binary feedback and execute N consecutive correct pulls. No cascading
  failures because z resets every episode.

Phase B0 (2k eps, z PERSISTS, heads frozen):
  Gentle warmup. GRU adapts to persistent z at streak=8 with strong
  reward floor. Only z dynamics change — nothing else.

Phase B1 (8k eps, z persists, heads frozen):
  Consolidate z-persistence. Policy reads from persistent z it wasn't
  trained on. GRU-only learning at streak=8.

Phase B2 (10k eps, z persists, heads UNFROZEN):
  Heads adapt to persistent-z regime. sc_floor ramps 0.5→0.2.
  Streak stays at 8 — policy doesn't need to learn harder tasks.

Phase B3 (7k eps, full training):
  sc_floor released. Natural curriculum takes over.

Key incentives:
  - Causal tracking bonus: -0.1 × clamp(||Δz||, 0.5) × pull_mask
  - Phase A: light m_loss (0.005) ceiling + tracking bonus
  - Phase B: tracking bonus ONLY + explosion guard (no base penalty)

Usage:
  python Causal_model_v2.py                  # full run
  python Causal_model_v2.py --phase-a-only   # test Phase A alone
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
# 1. SIMULATOR
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
        self.streak_cap   = 8            # external cap on streak difficulty
        self.shaped_correct_floor = None # external floor on shaped reward

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
# 2. MODEL
# ══════════════════════════════════════════════════════════════════════════════
class CausalBeliefModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru         = nn.GRUCell(4, 128)
        self.action_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.value_head  = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, obs_t, z_prev):
        z_new        = self.gru(obs_t, z_prev)
        delta_z_norm = ((z_new - z_prev) ** 2).sum(dim=-1)
        return (
            self.action_head(z_new),
            self.value_head(z_new).squeeze(-1),
            z_new,
            delta_z_norm,
        )

    def init_belief(self):
        return torch.zeros(1, 128)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SHARED EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_episode(env, model, z, persistent=False):
    obs_np      = env.reset()
    done        = False
    last_action = 0

    if not persistent:
        z = model.init_belief()

    log_probs, values, rewards, dz_vals, entropies, pull_flags, ep_dz = [], [], [], [], [], [], []
    snap_z, snap_label = None, None

    while not done:
        obs_np[0] = last_action / 2.0
        obs_t     = torch.FloatTensor(obs_np).unsqueeze(0)

        logits, val, z_new, dz_norm = model(obs_t, z)
        dist   = Categorical(logits=logits)
        action = dist.sample()

        obs_np, reward, done = env.step(action.item())
        last_action = action.item()

        log_probs.append(dist.log_prob(action))
        values.append(val)
        rewards.append(reward)
        dz_vals.append(dz_norm)
        entropies.append(dist.entropy())
        pull_flags.append(1.0 if action.item() < 2 else 0.0)
        ep_dz.append(dz_norm.item())

        z = z_new.detach()

    snap_z     = z.squeeze(0).numpy().copy()
    snap_label = 1 if np.sum(env.C) > env.hidden_dim // 2 else 0

    return log_probs, values, rewards, dz_vals, entropies, pull_flags, ep_dz, z, snap_z, snap_label


# ══════════════════════════════════════════════════════════════════════════════
# 4. INCENTIVE STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
TRACKING_COEFF  = 0.1       # reward z updates after informative observations
TRACKING_CAP    = 0.5       # clamp ||Δz|| at this level (prevent runaway)
PHASE_A_M_COEFF = 0.005     # light ceiling penalty in Phase A
GUARD_M_COEFF   = 0.15      # emergency brake if ||Δz|| explodes
GUARD_THRESHOLD = 0.5       # ||Δz|| above this triggers the guard
ENTROPY_COEFF   = -0.005    # entropy bonus (negative = maximize H(π))

def explosion_guard(avg_dz):
    """Only penalize ||Δz|| when it's exploding. Returns 0 normally."""
    if avg_dz > GUARD_THRESHOLD:
        overshoot = (avg_dz - GUARD_THRESHOLD) / GUARD_THRESHOLD
        return GUARD_M_COEFF * min(overshoot, 1.0)
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. PHASE A — Full streak curriculum with episodic z
# ══════════════════════════════════════════════════════════════════════════════
def phase_a(model, env, n_episodes=25_000, print_every=500):
    """Episodic training: z resets each episode. Teach policy from streak=3 to 8."""

    # ── Streak schedule: 3→4→5→6→7→8 with enough time at each level ──
    STREAK_SCHEDULE = [
        (5_000,   3),   # 0-5k:   master streak=3
        (8_000,   4),   # 5k-8k:  adapt to streak=4
        (12_000,  5),   # 8k-12k: adapt to streak=5
        (17_000,  6),   # 12k-17k: adapt to streak=6
        (21_000,  7),   # 17k-21k: adapt to streak=7
    ]                    # 21k-25k: consolidate at streak=8

    print("\nPHASE A — Episodic training (z reset each episode)")
    print("  Curriculum: 3(0-5k) → 4(5k-8k) → 5(8k-12k) → 6(12k-17k) → 7(17k-21k) → 8(21k-25k)")
    print("  shaped_correct locked at 0.5")
    print(f"  Tracking: {TRACKING_COEFF}  |  m_ceiling: {PHASE_A_M_COEFF}  |  Entropy: {ENTROPY_COEFF}")

    env.shaped_correct_floor = 0.5

    gru_params  = [p for n, p in model.named_parameters() if 'gru' in n]
    head_params = [p for n, p in model.named_parameters() if 'gru' not in n]
    opt = optim.Adam([
        {'params': gru_params,  'lr': 2e-4},
        {'params': head_params, 'lr': 5e-4},
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
        # ── Streak schedule ─────────────────────────────────────────
        env.streak_cap = 8  # default: final level
        for threshold, streak in STREAK_SCHEDULE:
            if ep < threshold:
                env.streak_cap = streak
                break

        # ── Episode ─────────────────────────────────────────────────
        lp, vals, rews, dzv, ents, pulls, ep_dz, z, sz, sl = run_episode(
            env, model, None, persistent=False
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

        # ── Loss: light ceiling + tracking bonus + entropy ──────────
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
        nn.utils.clip_grad_norm_(gru_params, 1.0)
        nn.utils.clip_grad_norm_(head_params, 2.0)
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
                  f"||Δz|| {rdz:.4f} | "
                  f"streak={env.streak_cap} | time {elapsed}s")

    final_r = np.mean(ep_rewards[-1000:])
    print(f"\nPhase A complete. Final avg reward: {final_r:.1f}")
    return ep_rewards, dz_norms, z_snapshots, z_labels


# ══════════════════════════════════════════════════════════════════════════════
# 6. PHASE B — Persistent z (4 sub-phases)
# ══════════════════════════════════════════════════════════════════════════════
def phase_b(model, env, n_episodes=27_000, print_every=500):
    """
    Persistent z. Policy already knows streak=8 from Phase A.
    We only teach z to carry information across episodes.
    """
    B0_END  =  2_000   # warmup: gentle z-persistence adaptation
    B1_END  = 10_000   # consolidate: GRU-only at streak=8
    B2_END  = 20_000   # adapt: heads unfrozen
    # B3: 20k-27k, full training

    print("\nPHASE B — Persistent z (4 sub-phases)")
    print(f"  B0 (0-{B0_END//1000}k):    warmup, heads frozen, streak=8, floor=0.5")
    print(f"  B1 ({B0_END//1000}k-{B1_END//1000}k):  consolidate, heads frozen, streak=8, floor=0.5")
    print(f"  B2 ({B1_END//1000}k-{B2_END//1000}k): heads unfrozen (lr=3e-5), floor=0.5")
    print(f"  B3 ({B2_END//1000}k+):  full training, floor=0.5")
    print(f"  sc_floor=0.5 throughout (no ramp — policy depends on dense reward)")
    print(f"  Tracking: {TRACKING_COEFF}  |  Entropy: {ENTROPY_COEFF}  |  Guard: {GUARD_M_COEFF} @ ||Δz||>{GUARD_THRESHOLD}")

    # ── Start frozen — GRU only ─────────────────────────────────────
    for p in model.action_head.parameters(): p.requires_grad = False
    for p in model.value_head.parameters():  p.requires_grad = False

    gru_params = [p for n, p in model.named_parameters() if 'gru' in n]
    opt = optim.Adam(gru_params, lr=5e-5)

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
    z              = model.init_belief()
    heads_unfrozen = False
    head_params    = None

    for ep in range(n_episodes):
        # ── Sub-phase transitions ───────────────────────────────────
        if ep == B1_END and not heads_unfrozen:
            # B2: unfreeze heads
            for p in model.action_head.parameters(): p.requires_grad = True
            for p in model.value_head.parameters():  p.requires_grad = True
            heads_unfrozen = True
            head_params = [p for n, p in model.named_parameters() if 'gru' not in n]
            opt = optim.Adam([
                {'params': gru_params, 'lr': 5e-5},
                {'params': head_params, 'lr': 3e-5},
            ])
            print(f"\n  ── B2 start: heads unfrozen (lr=3e-5), sc_floor ramp begins ──")

        # ── shaped_correct floor: FIXED at 0.5 throughout ──────────
        env.shaped_correct_floor = 0.5

        # ── Episode ─────────────────────────────────────────────────
        lp, vals, rews, dzv, ents, pulls, ep_dz, z, sz, sl = run_episode(
            env, model, z, persistent=True
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

        # ── Loss: tracking bonus + explosion guard (NO base penalty) ─
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
        nn.utils.clip_grad_norm_(gru_params, 0.5)
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

            if ep < B0_END:       tag = "B0"
            elif ep < B1_END:     tag = "B1"
            elif ep < B2_END:     tag = "B2"
            else:                 tag = "B3"

            g_tag = "⚡" if guard > 0 else " "
            sc = "0.50"
            print(f"  {tag} ep {ep+1:6d} | reward {rr:8.1f} | "
                  f"||Δz|| {rdz:.4f} {g_tag}| "
                  f"sc_floor={sc} | time {elapsed}s")

    final_r  = np.mean(ep_rewards[-1000:])
    final_dz = np.mean(dz_norms[-1000:])
    print(f"\nPhase B complete. Final avg reward: {final_r:.1f}")
    print(f"Final avg ||Δz||: {final_dz:.4f}")
    return ep_rewards, dz_norms, z_snapshots, z_labels


# ══════════════════════════════════════════════════════════════════════════════
# 7. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
def smooth(x, w=300):
    if len(x) < w: return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='valid')

def plot_diagnostics(a_rewards, a_dz, b_rewards, b_dz, b_snapshots, b_labels):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes.ravel()

    ax[0].plot(smooth(a_rewards), color="#1F3864")
    ax[0].axhline(0, color="gray", linestyle="--", lw=0.8)
    ax[0].set_title("Phase A — Reward (episodic z, streak 3→8)", fontweight="bold")
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Reward"); ax[0].grid(alpha=0.3)

    ax[1].plot(smooth(b_rewards), color="#1F6420")
    ax[1].axhline(0, color="gray", linestyle="--", lw=0.8)
    ax[1].set_title("Phase B — Reward (persistent z)", fontweight="bold")
    ax[1].set_xlabel("Episode"); ax[1].set_ylabel("Reward"); ax[1].grid(alpha=0.3)

    ax[2].plot(smooth(a_dz), color="#C0392B", label="Phase A", alpha=0.6)
    if b_dz:
        offset = len(smooth(a_dz))
        ax[2].plot(range(offset, offset + len(smooth(b_dz))), smooth(b_dz),
                   color="#2980B9", label="Phase B", alpha=0.6)
    ax[2].axhspan(0.15, 0.60, alpha=0.08, color="green", label="Target zone")
    ax[2].set_title("Causal Capacity ||Δz||", fontweight="bold")
    ax[2].set_xlabel("Episode"); ax[2].set_ylabel("||Δz||")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    ax[3].set_title("Phase B — PCA of Belief State z", fontweight="bold")
    if len(b_snapshots) >= 50:
        Z = np.array(b_snapshots)
        lbl = np.array(b_labels)
        pca = PCA(n_components=2)
        Z2 = pca.fit_transform(Z)
        sc = ax[3].scatter(Z2[:,0], Z2[:,1], c=lbl, cmap="RdBu", alpha=0.4, s=8)
        plt.colorbar(sc, ax=ax[3], label="Majority vote of C")
        var = pca.explained_variance_ratio_
        ax[3].set_title(f"Phase B — PCA of z\nPC1={var[0]:.1%}, PC2={var[1]:.1%}", fontweight="bold")
    ax[3].set_xlabel("PC1"); ax[3].set_ylabel("PC2"); ax[3].grid(alpha=0.3)

    plt.suptitle(
        "Causal Belief Model v2 — Clean Architecture\n"
        "A: streak 3→8 episodic | B0: warmup | B1: consolidate | B2: heads+ramp | B3: full",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig("causal_model_v2_diagnostics.png", dpi=150)
    print("Saved → causal_model_v2_diagnostics.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    phase_a_only = "--phase-a-only" in sys.argv
    phase_b_only = "--phase-b-only" in sys.argv

    print("═" * 55)
    print("Causal Belief Model v2 — Clean Architecture")
    if phase_a_only:
        print("MODE: Phase A only (streak 3→8 test)")
    elif phase_b_only:
        print("MODE: Phase B only (loading Phase A checkpoint)")
    else:
        print("Phase A: 25k eps (streak 3→8) | Phase B: 27k eps (z persists)")
    print("═" * 55)

    env   = CausalSimulator(hidden_dim=8, flip_mean=80)
    model = CausalBeliefModel()

    if phase_b_only:
        ckpt_path = "phase_a_checkpoint.pt"
        if not os.path.exists(ckpt_path):
            print(f"ERROR: {ckpt_path} not found. Run Phase A first.")
            sys.exit(1)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print(f"Loaded Phase A checkpoint from {ckpt_path}")
        # Set total_steps past all curriculum thresholds so streak_required=8
        # and shaped_correct follows the floor (0.5), not the base schedule
        env.total_steps = 2_000_000
        a_rewards, a_dz = [], []
    else:
        a_rewards, a_dz, _, _ = phase_a(model, env, n_episodes=25_000)
        torch.save(model.state_dict(), "phase_a_checkpoint.pt")

    if phase_a_only:
        final_r = np.mean(a_rewards[-1000:])
        print(f"\n{'═'*55}")
        print(f"Phase A test complete.")
        print(f"Final avg reward at streak=8: {final_r:.1f}")
        print(f"Final avg ||Δz||: {np.mean(a_dz[-1000:]):.4f}")
        if final_r > 0:
            print("→ POSITIVE reward at streak=8. Ready for full run.")
        else:
            print("→ Negative reward at streak=8. Needs adjustment.")
        print(f"{'═'*55}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(smooth(a_rewards), color="#1F3864")
        ax1.axhline(0, color="gray", linestyle="--", lw=0.8)
        ax1.set_title("Phase A — Reward"); ax1.set_xlabel("Episode"); ax1.grid(alpha=0.3)
        ax2.plot(smooth(a_dz), color="#C0392B")
        ax2.set_title("Phase A — ||Δz||"); ax2.set_xlabel("Episode"); ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("phase_a_test.png", dpi=150)
        print("Saved → phase_a_test.png")
        sys.exit(0)

    b_rewards, b_dz, b_snaps, b_lbls = phase_b(model, env, n_episodes=27_000)
    torch.save(model.state_dict(), "causal_belief_v2_final.pt")
    plot_diagnostics(a_rewards, a_dz, b_rewards, b_dz, b_snaps, b_lbls)

    final_r  = np.mean(b_rewards[-2000:])
    final_dz = np.mean(b_dz[-2000:])
    print(f"\nFinal avg reward  : {final_r:.1f}")
    print(f"Final avg ||Δz||  : {final_dz:.4f}")

    if final_r > 0 and 0.10 <= final_dz <= 0.60:
        print("\n→ READY FOR PHASE 2: add language head")
    else:
        print("\n→ Stabilize further before Phase 2")