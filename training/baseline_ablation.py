"""
Baseline Ablation — Is the tracking reward essential?
=====================================================
Train the EXACT same architecture without the tracking reward.
Compare: does ||Δz|| still stabilise? Does PCA still show separation?

If yes → the architecture alone produces the result, tracking reward unnecessary
If no  → the tracking reward is doing essential work (must be reported)

This runs a shortened version (10k Phase A + 10k Phase B) since we only
need to see whether the dynamics converge or not. Full training isn't needed.

Run:
  python baseline_ablation.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIMULATOR (identical)
# ══════════════════════════════════════════════════════════════════════════════
class CausalSimulator:
    def __init__(self, hidden_dim=8, flip_mean=80, max_steps=200,
                 big_reward=10.0, big_penalty=-30.0,
                 step_cost=-0.01, pull_cost=-0.05, wrong_streak=6):
        self.hidden_dim   = hidden_dim
        self.flip_mean    = flip_mean
        self.max_steps    = max_steps
        self.big_reward   = big_reward
        self.big_penalty  = big_penalty
        self.step_cost    = step_cost
        self.pull_cost    = pull_cost
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
            if self.steps >= self.max_steps: done = True
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

        if self.steps >= self.max_steps: done = True
        return self._get_obs(), reward, done

    def _get_obs(self):
        return np.array([
            0.0,
            self._last_correct,
            min(self.steps / 50.0, 1.0),
            float(np.random.randn()),
        ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL (identical)
# ══════════════════════════════════════════════════════════════════════════════
class CausalBeliefModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru         = nn.GRUCell(4, 128)
        self.action_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        self.value_head  = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
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
# 3. EPISODE RUNNER (identical)
# ══════════════════════════════════════════════════════════════════════════════
def run_episode(env, model, z, persistent=False):
    obs_np = env.reset()
    done = False
    last_action = 0
    if not persistent:
        z = model.init_belief()

    log_probs, values, rewards, dz_vals, entropies, ep_dz = [], [], [], [], [], []
    snap_z, snap_label = None, None

    while not done:
        obs_np[0] = last_action / 2.0
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        logits, val, z_new, dz_norm = model(obs_t, z)
        dist = Categorical(logits=logits)
        action = dist.sample()
        obs_np, reward, done = env.step(action.item())
        last_action = action.item()

        log_probs.append(dist.log_prob(action))
        values.append(val)
        rewards.append(reward)
        dz_vals.append(dz_norm)
        entropies.append(dist.entropy())
        ep_dz.append(dz_norm.item())
        z = z_new.detach()

    snap_z = z.squeeze(0).numpy().copy()
    snap_label = 1 if np.sum(env.C) > env.hidden_dim // 2 else 0
    return log_probs, values, rewards, dz_vals, entropies, ep_dz, z, snap_z, snap_label


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING — NO TRACKING REWARD
# ══════════════════════════════════════════════════════════════════════════════
ENTROPY_COEFF = -0.005

def train_baseline(n_phase_a=10_000, n_phase_b=10_000, print_every=500):
    """
    Same architecture, same curriculum, but NO tracking reward and NO m_loss.
    The loss is ONLY: policy gradient + value loss + entropy.
    """
    print("\n" + "═" * 60)
    print("BASELINE ABLATION — No tracking reward")
    print(f"Phase A: {n_phase_a} eps (episodic z, streak 3→5)")
    print(f"Phase B: {n_phase_b} eps (persistent z, streak=5)")
    print(f"Loss: policy + value + entropy ONLY (no tracking, no m_loss)")
    print("═" * 60)

    env   = CausalSimulator(hidden_dim=8, flip_mean=80)
    model = CausalBeliefModel()

    # ── Phase A: Episodic ────────────────────────────────────────────
    STREAK_SCHEDULE = [
        (3_000, 3),
        (6_000, 4),
    ]  # hold at 5 from 6k+

    env.shaped_correct_floor = 0.5

    gru_params  = [p for n, p in model.named_parameters() if 'gru' in n]
    head_params = [p for n, p in model.named_parameters() if 'gru' not in n]
    opt = optim.Adam([
        {'params': gru_params,  'lr': 2e-4},
        {'params': head_params, 'lr': 5e-4},
    ])

    ema_baseline = 0.0
    a_rewards, a_dz = [], []
    t_start = time.time()

    print("\nPhase A — Episodic (no tracking reward)")
    for ep in range(n_phase_a):
        env.streak_cap = 5
        for threshold, streak in STREAK_SCHEDULE:
            if ep < threshold:
                env.streak_cap = streak
                break

        lp, vals, rews, dzv, ents, ep_dz, z, sz, sl = run_episode(
            env, model, None, persistent=False
        )

        ema_baseline = 0.95 * ema_baseline + 0.05 * np.mean(rews)

        G = 0.0
        returns = []
        for r in reversed(rews):
            G = r + 0.97 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        values_t = torch.stack(vals)
        adv = returns - ema_baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── BASELINE LOSS: NO t_loss, NO m_loss ──────────────────────
        p_loss = -(torch.stack(lp) * adv.detach()).mean()
        v_loss = 0.5 * ((values_t - returns) ** 2).mean()
        e_loss = ENTROPY_COEFF * torch.stack(ents).mean()
        loss   = p_loss + v_loss + e_loss   # <-- NO tracking, NO metabolic

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(gru_params, 1.0)
        nn.utils.clip_grad_norm_(head_params, 2.0)
        opt.step()

        a_rewards.append(sum(rews))
        a_dz.append(float(np.mean(ep_dz)))

        if (ep + 1) % print_every == 0:
            rr  = np.mean(a_rewards[-print_every:])
            rdz = np.mean(a_dz[-print_every:])
            elapsed = int(time.time() - t_start)
            print(f"  A ep {ep+1:6d} | reward {rr:8.1f} | "
                  f"||Δz|| {rdz:.4f} | streak={env.streak_cap} | time {elapsed}s")

    print(f"Phase A complete. Reward: {np.mean(a_rewards[-1000:]):.1f}")

    # ── Phase B: Persistent z, heads frozen ──────────────────────────
    for p in model.action_head.parameters(): p.requires_grad = False
    for p in model.value_head.parameters():  p.requires_grad = False
    opt = optim.Adam(gru_params, lr=5e-5)
    env.streak_cap = 5
    env.shaped_correct_floor = 0.5

    b_rewards, b_dz = [], []
    b_snapshots, b_labels = [], []
    z = model.init_belief()
    t_start = time.time()

    print("\nPhase B — Persistent z (no tracking reward)")
    for ep in range(n_phase_b):
        lp, vals, rews, dzv, ents, ep_dz, z, sz, sl = run_episode(
            env, model, z, persistent=True
        )

        ema_baseline = 0.95 * ema_baseline + 0.05 * np.mean(rews)

        G = 0.0
        returns = []
        for r in reversed(rews):
            G = r + 0.97 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        values_t = torch.stack(vals)
        adv = returns - ema_baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        p_loss = -(torch.stack(lp) * adv.detach()).mean()
        v_loss = 0.5 * ((values_t - returns) ** 2).mean()
        e_loss = ENTROPY_COEFF * torch.stack(ents).mean()
        loss   = p_loss + v_loss + e_loss   # <-- NO tracking, NO metabolic

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(gru_params, 0.5)
        opt.step()

        b_rewards.append(sum(rews))
        b_dz.append(float(np.mean(ep_dz)))
        if ep % 200 == 0 and len(b_snapshots) < 500:
            b_snapshots.append(sz)
            b_labels.append(sl)

        if (ep + 1) % print_every == 0:
            rr  = np.mean(b_rewards[-print_every:])
            rdz = np.mean(b_dz[-print_every:])
            elapsed = int(time.time() - t_start)
            print(f"  B ep {ep+1:6d} | reward {rr:8.1f} | "
                  f"||Δz|| {rdz:.4f} | time {elapsed}s")

    print(f"Phase B complete. Reward: {np.mean(b_rewards[-1000:]):.1f}")

    return model, a_rewards, a_dz, b_rewards, b_dz, b_snapshots, b_labels


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def smooth(x, w=200):
    if len(x) < w: return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='valid')


def plot_comparison(a_rew, a_dz, b_rew, b_dz, b_snaps, b_labels):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes.ravel()

    # Phase A reward comparison
    ax[0].plot(smooth(a_rew), color="#C0392B", label="Baseline (no tracking)")
    ax[0].axhline(0, color="gray", linestyle="--", lw=0.8)
    ax[0].set_title("Phase A — Reward (Baseline)", fontweight="bold")
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Reward")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)

    # Phase B reward comparison
    ax[1].plot(smooth(b_rew), color="#C0392B", label="Baseline (no tracking)")
    ax[1].axhline(0, color="gray", linestyle="--", lw=0.8)
    ax[1].set_title("Phase B — Reward (Baseline)", fontweight="bold")
    ax[1].set_xlabel("Episode"); ax[1].set_ylabel("Reward")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9)

    # ||Δz|| comparison
    ax[2].plot(smooth(a_dz), color="#E74C3C", alpha=0.6, label="Phase A baseline")
    if b_dz:
        offset = len(smooth(a_dz))
        ax[2].plot(range(offset, offset + len(smooth(b_dz))),
                   smooth(b_dz), color="#C0392B", alpha=0.8, label="Phase B baseline")
    ax[2].axhspan(0.15, 0.60, alpha=0.08, color="green", label="Target zone (trained)")
    ax[2].set_title("||Δz|| — Baseline (no tracking reward)", fontweight="bold")
    ax[2].set_xlabel("Episode"); ax[2].set_ylabel("||Δz||")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    # PCA of z
    ax[3].set_title("PCA of z — Baseline", fontweight="bold")
    if len(b_snaps) >= 30:
        Z = np.array(b_snaps)
        lbl = np.array(b_labels)
        pca = PCA(n_components=2)
        Z2 = pca.fit_transform(Z)
        var = pca.explained_variance_ratio_
        sc = ax[3].scatter(Z2[:,0], Z2[:,1], c=lbl, cmap="RdBu", alpha=0.4, s=8)
        plt.colorbar(sc, ax=ax[3], label="Majority of C")
        ax[3].set_title(f"PCA of z — Baseline\n"
                        f"PC1={var[0]:.1%}, PC2={var[1]:.1%}", fontweight="bold")
    ax[3].set_xlabel("PC1"); ax[3].set_ylabel("PC2"); ax[3].grid(alpha=0.3)

    plt.suptitle("Baseline Ablation — Same Architecture, No Tracking Reward\n"
                 "Does the architecture alone produce causal tracking?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("baseline_ablation.png", dpi=150)
    print(f"Saved → baseline_ablation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model, a_rew, a_dz, b_rew, b_dz, b_snaps, b_lbls = train_baseline(
        n_phase_a=10_000, n_phase_b=10_000
    )

    torch.save(model.state_dict(), "baseline_no_tracking.pt")

    plot_comparison(a_rew, a_dz, b_rew, b_dz, b_snaps, b_lbls)

    # Summary
    print("\n" + "═" * 60)
    print("BASELINE ABLATION RESULTS")
    print("═" * 60)

    final_r  = np.mean(b_rew[-1000:])
    final_dz = np.mean(b_dz[-1000:])
    print(f"  Final reward (Phase B):  {final_r:.1f}")
    print(f"  Final ||Δz|| (Phase B):  {final_dz:.4f}")

    print(f"\n  Compare with trained model:")
    print(f"    Trained:  reward=+5.3, ||Δz||=0.55")
    print(f"    Baseline: reward={final_r:.1f}, ||Δz||={final_dz:.4f}")

    if final_dz < 0.05:
        print("\n  ✓ Without tracking reward, ||Δz|| → 0")
        print("  → The tracking reward is ESSENTIAL for maintaining z dynamics")
        print("  → The GRU naturally collapses without explicit incentive")
    elif 0.10 <= final_dz <= 0.60:
        print("\n  ✗ ||Δz|| stabilises even without tracking reward")
        print("  → The architecture alone produces the result")
        print("  → The tracking reward may be unnecessary")
    else:
        print(f"\n  ~ ||Δz|| at {final_dz:.4f} — ambiguous result")
        print("  → Tracking reward has some effect but isn't solely responsible")

    print("═" * 60)
