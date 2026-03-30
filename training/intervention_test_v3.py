"""
Intervention Test v3 — do(C) vs observe(C) for Separated Architecture
======================================================================

Same protocol as intervention_test.py but for SeparatedCausalModel.
Tests whether the architectural separation (z_w / z_p) improves
causal tracking under do(C) interventions.

New diagnostic: ||Δz_w|| vs ||Δz_p|| after do(C).
If the separation works:
  - z_w should spike (world model detects state change from feedback)
  - z_p should be relatively stable (policy state, not world model)

Run:
  python intervention_test_v3.py
  python intervention_test_v3.py --compare   # side-by-side with v2
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os, sys, time

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIMULATOR (intervention-capable, same as v2 test)
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
        self.total_steps  = 10_000_000

    @property
    def streak_required(self): return 8

    @property
    def shaped_correct(self): return 0.5

    def reset(self):
        self.C               = np.random.randint(0, 2, self.hidden_dim).astype(float)
        self.steps           = 0
        self.correct_consec  = 0
        self.wrong_consec    = 0
        self.flip_timers     = np.random.geometric(1.0 / self.flip_mean, self.hidden_dim)
        self._last_correct   = 0.0
        return self._get_obs()

    def intervene(self, new_C):
        """do(C) — set hidden state directly, bypassing normal dynamics."""
        self.C = np.array(new_C, dtype=float)

    def step(self, action):
        reward = self.step_cost
        done   = False

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

    def get_majority(self):
        return 1 if np.sum(self.C) > self.hidden_dim // 2 else 0

    def get_correct_action(self):
        majority  = self.get_majority()
        corr_flip = (self.C[0] == 1 and self.C[1] == 1)
        return 1 - majority if corr_flip else majority


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL (same architecture as training)
# ══════════════════════════════════════════════════════════════════════════════
class SeparatedCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.world_gru = nn.GRUCell(4, 128)
        self.policy_gru = nn.GRUCell(128 + 4, 64)
        self.action_head = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        self.value_head = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, obs_t, z_w_prev, z_p_prev):
        z_w_new = self.world_gru(obs_t, z_w_prev)
        delta_z_w = ((z_w_new - z_w_prev) ** 2).sum(dim=-1)

        z_w_detached = z_w_new.detach()
        policy_input = torch.cat([z_w_detached, obs_t], dim=-1)
        z_p_new = self.policy_gru(policy_input, z_p_prev)
        delta_z_p = ((z_p_new - z_p_prev) ** 2).sum(dim=-1)

        combined = torch.cat([z_w_detached, z_p_new], dim=-1)
        logits = self.action_head(combined)
        value  = self.value_head(combined).squeeze(-1)

        return logits, value, z_w_new, z_p_new, delta_z_w, delta_z_p

    def init_world_belief(self):
        return torch.zeros(1, 128)

    def init_policy_belief(self):
        return torch.zeros(1, 64)


# ══════════════════════════════════════════════════════════════════════════════
# 3. INTERVENTION EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
def run_intervention_trial(model, env, z_w, z_p,
                           warmup_steps=30, post_steps=30,
                           target_majority=None):
    """
    Same protocol as v2:
      1. Warmup: 30 steps normal play
      2. Intervene: do(C) to opposite majority
      3. Post: 30 steps, measure adaptation

    New: tracks ||Δz_w|| AND ||Δz_p|| separately.
    """
    obs_np = env.reset()
    last_action = 0
    env.flip_timers = np.full(env.hidden_dim, 10000)
    pre_majority = env.get_majority()

    # ── Warmup ───────────────────────────────────────────────────────
    pre_correct = 0
    pre_total   = 0
    pre_dz_w    = []
    pre_dz_p    = []

    for step in range(warmup_steps):
        obs_np[0] = last_action / 2.0
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)

        with torch.no_grad():
            logits, _, z_w_new, z_p_new, dz_w, dz_p = model(obs_t, z_w, z_p)

        action = logits.argmax(-1).item()

        if action < 2:
            correct_a = env.get_correct_action()
            if action == correct_a:
                pre_correct += 1
            pre_total += 1

        obs_np, _, done = env.step(action)
        last_action = action
        z_w = z_w_new.detach()
        z_p = z_p_new.detach()
        pre_dz_w.append(dz_w.item())
        pre_dz_p.append(dz_p.item())

        if done:
            break

    pre_accuracy = pre_correct / max(pre_total, 1)

    # ── Intervention: do(C) ──────────────────────────────────────────
    if target_majority is None:
        target_majority = 1 - pre_majority

    if target_majority == 1:
        new_C = np.array([1, 0, 1, 1, 1, 1, 1, 0])
    else:
        new_C = np.array([0, 0, 0, 1, 0, 0, 1, 0])

    env.intervene(new_C)
    env.flip_timers = np.full(env.hidden_dim, 10000)
    post_majority = env.get_majority()

    # ── Post-intervention ────────────────────────────────────────────
    post_correct  = []
    post_dz_w     = []
    post_dz_p     = []
    first_correct = None

    for step in range(post_steps):
        obs_np[0] = last_action / 2.0
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)

        with torch.no_grad():
            logits, _, z_w_new, z_p_new, dz_w, dz_p = model(obs_t, z_w, z_p)

        action = logits.argmax(-1).item()

        if action < 2:
            correct_a = env.get_correct_action()
            is_correct = (action == correct_a)
            post_correct.append(1 if is_correct else 0)
            if is_correct and first_correct is None:
                first_correct = step
        else:
            post_correct.append(0)

        obs_np, _, done = env.step(action)
        last_action = action
        z_w = z_w_new.detach()
        z_p = z_p_new.detach()
        post_dz_w.append(dz_w.item())
        post_dz_p.append(dz_p.item())

        if done:
            break

    return {
        'pre_majority':   pre_majority,
        'post_majority':  post_majority,
        'pre_accuracy':   pre_accuracy,
        'post_accuracy':  post_correct,
        'post_dz_w':      post_dz_w,
        'post_dz_p':      post_dz_p,
        'pre_dz_w':       pre_dz_w,
        'pre_dz_p':       pre_dz_p,
        'first_correct':  first_correct,
        'z_w_final':      z_w.squeeze(0).detach().numpy().copy(),
        'z_p_final':      z_p.squeeze(0).detach().numpy().copy(),
    }


def run_natural_flip_trial(model, env, z_w, z_p,
                           warmup_steps=30, post_steps=30):
    """Control condition: natural flip, same protocol."""
    obs_np = env.reset()
    last_action = 0
    env.flip_timers = np.full(env.hidden_dim, 10000)
    pre_majority = env.get_majority()

    for step in range(warmup_steps):
        obs_np[0] = last_action / 2.0
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        with torch.no_grad():
            logits, _, z_w_new, z_p_new, dz_w, dz_p = model(obs_t, z_w, z_p)
        action = logits.argmax(-1).item()
        obs_np, _, done = env.step(action)
        last_action = action
        z_w = z_w_new.detach()
        z_p = z_p_new.detach()
        if done: break

    if pre_majority == 1:
        new_C = np.array([0, 0, 0, 1, 0, 0, 1, 0])
    else:
        new_C = np.array([1, 0, 1, 1, 1, 1, 1, 0])

    env.C = new_C
    env.flip_timers = np.full(env.hidden_dim, 10000)

    post_correct  = []
    post_dz_w     = []
    post_dz_p     = []
    first_correct = None

    for step in range(post_steps):
        obs_np[0] = last_action / 2.0
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        with torch.no_grad():
            logits, _, z_w_new, z_p_new, dz_w, dz_p = model(obs_t, z_w, z_p)
        action = logits.argmax(-1).item()

        if action < 2:
            correct_a = env.get_correct_action()
            is_correct = (action == correct_a)
            post_correct.append(1 if is_correct else 0)
            if is_correct and first_correct is None:
                first_correct = step
        else:
            post_correct.append(0)

        obs_np, _, done = env.step(action)
        last_action = action
        z_w = z_w_new.detach()
        z_p = z_p_new.detach()
        post_dz_w.append(dz_w.item())
        post_dz_p.append(dz_p.item())
        if done: break

    return {
        'post_accuracy':  post_correct,
        'post_dz_w':      post_dz_w,
        'post_dz_p':      post_dz_p,
        'first_correct':  first_correct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    compare_v2 = "--compare" in sys.argv

    print("═" * 65)
    print("INTERVENTION TEST v3 — Separated World Model / Policy")
    print("Does z_w track the actual hidden state after do(C)?")
    print("═" * 65)

    # Load frozen model
    model = SeparatedCausalModel()
    # weights_only=False needed because checkpoint contains numpy PCA arrays
    ckpt = torch.load("causal_belief_v3_final.pt", map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded frozen model: causal_belief_v3_final.pt")

    env = CausalSimulator(hidden_dim=8, flip_mean=80)
    N_TRIALS = 200

    # ── Intervention trials ──────────────────────────────────────────
    print(f"\nRunning {N_TRIALS} intervention trials...")
    intervention_results = []
    z_w = model.init_world_belief()
    z_p = model.init_policy_belief()

    for trial in range(N_TRIALS):
        result = run_intervention_trial(model, env, z_w, z_p,
                                        warmup_steps=30, post_steps=30)
        intervention_results.append(result)
        z_w = torch.FloatTensor(result['z_w_final']).unsqueeze(0)
        z_p = torch.FloatTensor(result['z_p_final']).unsqueeze(0)

        if (trial + 1) % 50 == 0:
            recent = intervention_results[-50:]
            fc = [r['first_correct'] for r in recent if r['first_correct'] is not None]
            avg_fc = np.mean(fc) if fc else float('inf')
            acc_5  = np.mean([np.mean(r['post_accuracy'][:5]) for r in recent])
            acc_30 = np.mean([np.mean(r['post_accuracy']) for r in recent])
            print(f"  Trial {trial+1:4d} | steps-to-correct: {avg_fc:.1f} | "
                  f"acc@5: {acc_5:.1%} | acc@30: {acc_30:.1%}")

    # ── Natural flip control ─────────────────────────────────────────
    print(f"\nRunning {N_TRIALS} natural flip control trials...")
    natural_results = []
    z_w = model.init_world_belief()
    z_p = model.init_policy_belief()

    for trial in range(N_TRIALS):
        result = run_natural_flip_trial(model, env, z_w, z_p,
                                        warmup_steps=30, post_steps=30)
        natural_results.append(result)

        if (trial + 1) % 50 == 0:
            recent = natural_results[-50:]
            fc = [r['first_correct'] for r in recent if r['first_correct'] is not None]
            avg_fc = np.mean(fc) if fc else float('inf')
            acc_5  = np.mean([np.mean(r['post_accuracy'][:5]) for r in recent])
            acc_30 = np.mean([np.mean(r['post_accuracy']) for r in recent])
            print(f"  Trial {trial+1:4d} | steps-to-correct: {avg_fc:.1f} | "
                  f"acc@5: {acc_5:.1%} | acc@30: {acc_30:.1%}")

    # ══════════════════════════════════════════════════════════════════
    # 5. ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 65)
    print("RESULTS")
    print("═" * 65)

    int_fc = [r['first_correct'] for r in intervention_results
              if r['first_correct'] is not None]
    nat_fc = [r['first_correct'] for r in natural_results
              if r['first_correct'] is not None]

    print(f"\n{'Metric':<35s} {'Intervention':>15s} {'Natural flip':>15s}")
    print("─" * 65)

    int_avg_fc = np.mean(int_fc) if int_fc else float('inf')
    nat_avg_fc = np.mean(nat_fc) if nat_fc else float('inf')
    print(f"{'Steps to first correct':<35s} {int_avg_fc:>15.1f} {nat_avg_fc:>15.1f}")

    int_recovery = sum(1 for f in int_fc if f <= 5) / max(len(int_fc), 1)
    nat_recovery = sum(1 for f in nat_fc if f <= 5) / max(len(nat_fc), 1)
    print(f"{'Recovery within 5 steps':<35s} {int_recovery:>15.1%} {nat_recovery:>15.1%}")

    for window, label in [(5, "Acc @ 5 steps"), (10, "Acc @ 10 steps"),
                          (30, "Acc @ 30 steps")]:
        int_acc = np.mean([np.mean(r['post_accuracy'][:window])
                           for r in intervention_results])
        nat_acc = np.mean([np.mean(r['post_accuracy'][:window])
                           for r in natural_results])
        print(f"{label:<35s} {int_acc:>15.1%} {nat_acc:>15.1%}")

    # ── NEW: ||Δz_w|| vs ||Δz_p|| after intervention ────────────────
    print(f"\n{'── Separation diagnostic ──':^65s}")
    print(f"{'Metric':<35s} {'Intervention':>15s} {'Natural flip':>15s}")
    print("─" * 65)

    int_dz_w_spike = np.mean([np.mean(r['post_dz_w'][:5])
                              for r in intervention_results])
    int_dz_p_spike = np.mean([np.mean(r['post_dz_p'][:5])
                              for r in intervention_results])
    nat_dz_w_spike = np.mean([np.mean(r['post_dz_w'][:5])
                              for r in natural_results])
    nat_dz_p_spike = np.mean([np.mean(r['post_dz_p'][:5])
                              for r in natural_results])

    print(f"{'||Δz_w|| first 5 steps':<35s} {int_dz_w_spike:>15.4f} {nat_dz_w_spike:>15.4f}")
    print(f"{'||Δz_p|| first 5 steps':<35s} {int_dz_p_spike:>15.4f} {nat_dz_p_spike:>15.4f}")
    print(f"{'z_w/z_p ratio':<35s} "
          f"{int_dz_w_spike/max(int_dz_p_spike,1e-6):>15.2f} "
          f"{nat_dz_w_spike/max(nat_dz_p_spike,1e-6):>15.2f}")

    int_dz_w_late = np.mean([np.mean(r['post_dz_w'][15:])
                             for r in intervention_results
                             if len(r['post_dz_w']) > 15])
    nat_dz_w_late = np.mean([np.mean(r['post_dz_w'][15:])
                             for r in natural_results
                             if len(r['post_dz_w']) > 15])
    print(f"{'||Δz_w|| steps 15-30':<35s} {int_dz_w_late:>15.4f} {nat_dz_w_late:>15.4f}")

    # ── Interpretation ───────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("INTERPRETATION")
    print("═" * 65)

    if int_avg_fc <= 5 and int_recovery > 0.5:
        print("✓ Model adapts to do(C) within 5 steps")
        print("  → z_w uses feedback to infer new C")
        print("  → Separation enables causal tracking")
    elif int_avg_fc <= nat_avg_fc * 1.2:
        print("~ Model adapts to do(C) at similar speed to natural flips")
        print("  → z_w tracks observational evidence regardless of cause")
    else:
        print("✗ Model adapts SLOWER to do(C) than natural flips")
        print("  → Separation alone is insufficient")

    if int_dz_w_spike > int_dz_p_spike * 1.5:
        print("✓ ||Δz_w|| >> ||Δz_p|| after intervention")
        print("  → World model responds more than policy — separation is functional")
    else:
        print("~ ||Δz_w|| ≈ ||Δz_p|| — separation may not be clean")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Post-accuracy curves
    ax = axes[0, 0]
    max_len = max(len(r['post_accuracy']) for r in intervention_results)
    int_acc_curve = np.zeros(max_len)
    int_counts    = np.zeros(max_len)
    for r in intervention_results:
        for i, v in enumerate(r['post_accuracy']):
            int_acc_curve[i] += v
            int_counts[i] += 1
    int_acc_curve /= np.maximum(int_counts, 1)

    max_len_n = max(len(r['post_accuracy']) for r in natural_results)
    nat_acc_curve = np.zeros(max_len_n)
    nat_counts    = np.zeros(max_len_n)
    for r in natural_results:
        for i, v in enumerate(r['post_accuracy']):
            nat_acc_curve[i] += v
            nat_counts[i] += 1
    nat_acc_curve /= np.maximum(nat_counts, 1)

    ax.plot(int_acc_curve, color="#C0392B", label="do(C)", linewidth=2)
    ax.plot(nat_acc_curve, color="#2980B9", label="Natural flip", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_title("Accuracy After Majority Change", fontweight="bold")
    ax.set_xlabel("Steps after intervention")
    ax.set_ylabel("Fraction correct")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ||Δz_w|| curves
    ax = axes[0, 1]
    max_len = max(len(r['post_dz_w']) for r in intervention_results)
    int_dz_w_curve = np.zeros(max_len)
    int_dz_w_cnt   = np.zeros(max_len)
    for r in intervention_results:
        for i, v in enumerate(r['post_dz_w']):
            int_dz_w_curve[i] += v
            int_dz_w_cnt[i] += 1
    int_dz_w_curve /= np.maximum(int_dz_w_cnt, 1)

    max_len_n = max(len(r['post_dz_w']) for r in natural_results)
    nat_dz_w_curve = np.zeros(max_len_n)
    nat_dz_w_cnt   = np.zeros(max_len_n)
    for r in natural_results:
        for i, v in enumerate(r['post_dz_w']):
            nat_dz_w_curve[i] += v
            nat_dz_w_cnt[i] += 1
    nat_dz_w_curve /= np.maximum(nat_dz_w_cnt, 1)

    ax.plot(int_dz_w_curve, color="#C0392B", label="do(C)", linewidth=2)
    ax.plot(nat_dz_w_curve, color="#2980B9", label="Natural", linewidth=2)
    ax.set_title("||Δz_w|| After Majority Change", fontweight="bold")
    ax.set_xlabel("Steps"); ax.set_ylabel("||Δz_w||")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ||Δz_p|| curves (NEW)
    ax = axes[1, 0]
    max_len = max(len(r['post_dz_p']) for r in intervention_results)
    int_dz_p_curve = np.zeros(max_len)
    int_dz_p_cnt   = np.zeros(max_len)
    for r in intervention_results:
        for i, v in enumerate(r['post_dz_p']):
            int_dz_p_curve[i] += v
            int_dz_p_cnt[i] += 1
    int_dz_p_curve /= np.maximum(int_dz_p_cnt, 1)

    max_len_n = max(len(r['post_dz_p']) for r in natural_results)
    nat_dz_p_curve = np.zeros(max_len_n)
    nat_dz_p_cnt   = np.zeros(max_len_n)
    for r in natural_results:
        for i, v in enumerate(r['post_dz_p']):
            nat_dz_p_curve[i] += v
            nat_dz_p_cnt[i] += 1
    nat_dz_p_curve /= np.maximum(nat_dz_p_cnt, 1)

    ax.plot(int_dz_p_curve, color="#C0392B", label="do(C)", linewidth=2)
    ax.plot(nat_dz_p_curve, color="#2980B9", label="Natural", linewidth=2)
    ax.set_title("||Δz_p|| After Majority Change (policy stability)", fontweight="bold")
    ax.set_xlabel("Steps"); ax.set_ylabel("||Δz_p||")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Steps-to-correct histogram
    ax = axes[1, 1]
    bins = range(0, 31, 2)
    ax.hist(int_fc, bins=bins, alpha=0.6, color="#C0392B", label="do(C)", density=True)
    ax.hist(nat_fc, bins=bins, alpha=0.6, color="#2980B9", label="Natural", density=True)
    ax.set_title("Steps to First Correct Action", fontweight="bold")
    ax.set_xlabel("Steps"); ax.set_ylabel("Density")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("Intervention Test v3 — Separated World Model / Policy\n"
                 "do(C) vs observe(C): does z_w track C independently of z_p?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("intervention_test_v3.png", dpi=150)
    print(f"\nSaved → intervention_test_v3.png")


if __name__ == "__main__":
    main()
