"""
Intervention Test — do(C) vs observe(C)
========================================
Pearl's ladder of causation:  P(y|x) vs P(y|do(x))

This experiment answers: does z encode the ACTUAL hidden state C,
or just the observation history? A causal model responds to interventions.
An associative model responds to its history regardless of true state.

Protocol:
  1. Run the frozen model normally for 30 steps to build up z
  2. Secretly intervene: set C to a known value (do(C))
  3. Continue running for 30 more steps
  4. Measure: does the model's behavior (action accuracy) track the
     NEW C, or does it continue acting as if C didn't change?

Comparison:
  - "Natural flip": z updates because observations change (correlational)
  - "Silent intervention": C changes but observations haven't accumulated
     yet. If z STILL updates quickly → genuinely causal tracking

Key metric: steps-to-correct after intervention.
  - Causal model: corrects within 3-5 steps (uses z to infer C from feedback)
  - Associative model: corrects slowly or not at all (ignores feedback direction)

Run:
  python intervention_test.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIMULATOR (same as training, with intervention support)
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
        self.total_steps  = 10_000_000   # bypass curriculum

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
        # Don't change flip_timers — the intervention is surgical
        # Don't provide any observation signal — the model must infer from pulls

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
# 2. FROZEN MODEL
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
# 3. INTERVENTION EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
def run_intervention_trial(model, env, z,
                           warmup_steps=30, post_steps=30,
                           target_majority=None):
    """
    Run one trial:
      1. Warmup: 30 steps of normal play (establish z)
      2. Intervene: secretly set C to force a specific majority
      3. Post-intervention: 30 more steps, measure adaptation

    Returns: dict with pre/post accuracy, dz trajectory, steps-to-correct
    """
    obs_np = env.reset()
    last_action = 0

    # Freeze natural flips during test by setting very long timers
    env.flip_timers = np.full(env.hidden_dim, 10000)

    pre_majority = env.get_majority()

    # ── Phase 1: Warmup ──────────────────────────────────────────────
    pre_correct = 0
    pre_total   = 0
    pre_dz      = []

    for step in range(warmup_steps):
        obs_np[0] = last_action / 2.0
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)

        with torch.no_grad():
            logits, _, z_new, dz_norm = model(obs_t, z)

        action = logits.argmax(-1).item()  # greedy for clean measurement

        if action < 2:
            correct_a = env.get_correct_action()
            if action == correct_a:
                pre_correct += 1
            pre_total += 1

        obs_np, _, done = env.step(action)
        last_action = action
        z = z_new.detach()
        pre_dz.append(dz_norm.item())

        if done:
            break

    pre_accuracy = pre_correct / max(pre_total, 1)

    # ── Phase 2: Intervention — do(C) ────────────────────────────────
    # Set C to the OPPOSITE majority to maximise signal
    if target_majority is None:
        target_majority = 1 - pre_majority

    if target_majority == 1:
        new_C = np.array([1, 0, 1, 1, 1, 1, 1, 0])  # majority=1, no corr_flip
    else:
        new_C = np.array([0, 0, 0, 1, 0, 0, 1, 0])  # majority=0, no corr_flip

    env.intervene(new_C)
    env.flip_timers = np.full(env.hidden_dim, 10000)  # keep state frozen
    post_majority = env.get_majority()

    # ── Phase 3: Post-intervention ──────────────────────────────────
    post_correct   = []
    post_dz        = []
    first_correct  = None

    for step in range(post_steps):
        obs_np[0] = last_action / 2.0
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)

        with torch.no_grad():
            logits, _, z_new, dz_norm = model(obs_t, z)

        action = logits.argmax(-1).item()

        if action < 2:
            correct_a = env.get_correct_action()
            is_correct = (action == correct_a)
            post_correct.append(1 if is_correct else 0)
            if is_correct and first_correct is None:
                first_correct = step
        else:
            post_correct.append(0)  # wait = not demonstrating knowledge

        obs_np, _, done = env.step(action)
        last_action = action
        z = z_new.detach()
        post_dz.append(dz_norm.item())

        if done:
            break

    return {
        'pre_majority':   pre_majority,
        'post_majority':  post_majority,
        'pre_accuracy':   pre_accuracy,
        'post_accuracy':  post_correct,
        'post_dz':        post_dz,
        'pre_dz':         pre_dz,
        'first_correct':  first_correct,
        'z_final':        z.squeeze(0).detach().numpy().copy(),
    }


def run_natural_flip_trial(model, env, z, warmup_steps=30, post_steps=30):
    """
    Control condition: wait for a NATURAL flip, then measure adaptation.
    Same protocol but no artificial intervention — the model adapts
    through normal dynamics.
    """
    obs_np = env.reset()
    last_action = 0
    env.flip_timers = np.full(env.hidden_dim, 10000)  # no flips during warmup

    pre_majority = env.get_majority()

    # Warmup
    for step in range(warmup_steps):
        obs_np[0] = last_action / 2.0
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        with torch.no_grad():
            logits, _, z_new, dz_norm = model(obs_t, z)
        action = logits.argmax(-1).item()
        obs_np, _, done = env.step(action)
        last_action = action
        z = z_new.detach()
        if done: break

    # Natural flip: flip ALL majority bits (same as intervention but "natural")
    # The key difference: model gets normal observation-action feedback
    if pre_majority == 1:
        new_C = np.array([0, 0, 0, 1, 0, 0, 1, 0])
    else:
        new_C = np.array([1, 0, 1, 1, 1, 1, 1, 0])

    env.C = new_C
    env.flip_timers = np.full(env.hidden_dim, 10000)
    post_majority = env.get_majority()

    # Post-flip measurement
    post_correct = []
    post_dz      = []
    first_correct = None

    for step in range(post_steps):
        obs_np[0] = last_action / 2.0
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        with torch.no_grad():
            logits, _, z_new, dz_norm = model(obs_t, z)
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
        z = z_new.detach()
        post_dz.append(dz_norm.item())
        if done: break

    return {
        'post_majority': post_majority,
        'post_accuracy': post_correct,
        'post_dz':       post_dz,
        'first_correct': first_correct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("═" * 60)
    print("INTERVENTION TEST — do(C) vs observe(C)")
    print("Does z track the actual hidden state or just history?")
    print("═" * 60)

    # Load frozen model
    model = CausalBeliefModel()
    ckpt  = torch.load("causal_belief_v2_final.pt", map_location="cpu",
                        weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded frozen model: causal_belief_v2_final.pt")

    env = CausalSimulator(hidden_dim=8, flip_mean=80)
    N_TRIALS = 200

    # ── Run intervention trials ─────────────────────────────────────
    print(f"\nRunning {N_TRIALS} intervention trials...")
    intervention_results = []
    z = model.init_belief()

    for trial in range(N_TRIALS):
        result = run_intervention_trial(model, env, z, warmup_steps=30,
                                        post_steps=30)
        intervention_results.append(result)
        z = torch.FloatTensor(result['z_final']).unsqueeze(0)  # persist z

        if (trial + 1) % 50 == 0:
            # Running stats
            recent = intervention_results[-50:]
            fc = [r['first_correct'] for r in recent if r['first_correct'] is not None]
            avg_fc = np.mean(fc) if fc else float('inf')
            acc_5  = np.mean([np.mean(r['post_accuracy'][:5]) for r in recent])
            acc_30 = np.mean([np.mean(r['post_accuracy']) for r in recent])
            print(f"  Trial {trial+1:4d} | steps-to-correct: {avg_fc:.1f} | "
                  f"acc@5: {acc_5:.1%} | acc@30: {acc_30:.1%}")

    # ── Run natural flip control trials ─────────────────────────────
    print(f"\nRunning {N_TRIALS} natural flip control trials...")
    natural_results = []
    z = model.init_belief()

    for trial in range(N_TRIALS):
        result = run_natural_flip_trial(model, env, z, warmup_steps=30,
                                        post_steps=30)
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
    print("\n" + "═" * 60)
    print("RESULTS")
    print("═" * 60)

    # Steps to first correct action after majority flip
    int_fc = [r['first_correct'] for r in intervention_results
              if r['first_correct'] is not None]
    nat_fc = [r['first_correct'] for r in natural_results
              if r['first_correct'] is not None]

    print(f"\n{'Metric':<30s} {'Intervention':>15s} {'Natural flip':>15s}")
    print("─" * 60)

    int_avg_fc = np.mean(int_fc) if int_fc else float('inf')
    nat_avg_fc = np.mean(nat_fc) if nat_fc else float('inf')
    print(f"{'Steps to first correct':<30s} {int_avg_fc:>15.1f} {nat_avg_fc:>15.1f}")

    int_recovery = sum(1 for f in int_fc if f <= 5) / max(len(int_fc), 1)
    nat_recovery = sum(1 for f in nat_fc if f <= 5) / max(len(nat_fc), 1)
    print(f"{'Recovery within 5 steps':<30s} {int_recovery:>15.1%} {nat_recovery:>15.1%}")

    # Accuracy at different windows
    for window, label in [(5, "Acc @ 5 steps"), (10, "Acc @ 10 steps"),
                          (30, "Acc @ 30 steps")]:
        int_acc = np.mean([np.mean(r['post_accuracy'][:window])
                           for r in intervention_results])
        nat_acc = np.mean([np.mean(r['post_accuracy'][:window])
                           for r in natural_results])
        print(f"{label:<30s} {int_acc:>15.1%} {nat_acc:>15.1%}")

    # ||Δz|| spike after intervention
    int_dz_spike = np.mean([np.mean(r['post_dz'][:5])
                            for r in intervention_results])
    nat_dz_spike = np.mean([np.mean(r['post_dz'][:5])
                            for r in natural_results])
    int_dz_late  = np.mean([np.mean(r['post_dz'][15:])
                            for r in intervention_results
                            if len(r['post_dz']) > 15])
    nat_dz_late  = np.mean([np.mean(r['post_dz'][15:])
                            for r in natural_results
                            if len(r['post_dz']) > 15])
    print(f"{'||Δz|| first 5 steps':<30s} {int_dz_spike:>15.3f} {nat_dz_spike:>15.3f}")
    print(f"{'||Δz|| steps 15-30':<30s} {int_dz_late:>15.3f} {nat_dz_late:>15.3f}")

    # ── Interpretation ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("INTERPRETATION")
    print("═" * 60)

    if int_avg_fc <= 5 and int_recovery > 0.5:
        print("✓ Model adapts to intervention within 5 steps")
        print("  → z uses FEEDBACK to infer new C, not just history")
        print("  → This is consistent with causal tracking")
    elif int_avg_fc <= nat_avg_fc * 1.2:
        print("~ Model adapts to intervention at similar speed to natural flips")
        print("  → z tracks observational evidence regardless of cause")
        print("  → Consistent with strong Bayesian updating (but not necessarily causal)")
    else:
        print("✗ Model adapts SLOWER to interventions than natural flips")
        print("  → z may rely on dynamic patterns, not causal inference")

    if abs(int_dz_spike - nat_dz_spike) / max(nat_dz_spike, 0.01) < 0.3:
        print("✓ ||Δz|| spike similar for both conditions")
        print("  → GRU responds to prediction errors, not to state change per se")
    else:
        print(f"~ ||Δz|| spike differs: intervention={int_dz_spike:.3f} vs "
              f"natural={nat_dz_spike:.3f}")

    # ── Plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Post-accuracy over time
    ax = axes[0]
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

    ax.plot(int_acc_curve, color="#C0392B", label="do(C) intervention", linewidth=2)
    ax.plot(nat_acc_curve, color="#2980B9", label="Natural flip", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_title("Accuracy After Majority Change", fontweight="bold")
    ax.set_xlabel("Steps after intervention")
    ax.set_ylabel("Fraction correct")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ||Δz|| over time
    ax = axes[1]
    max_len = max(len(r['post_dz']) for r in intervention_results)
    int_dz_curve = np.zeros(max_len)
    int_dz_counts = np.zeros(max_len)
    for r in intervention_results:
        for i, v in enumerate(r['post_dz']):
            int_dz_curve[i] += v
            int_dz_counts[i] += 1
    int_dz_curve /= np.maximum(int_dz_counts, 1)

    max_len_n = max(len(r['post_dz']) for r in natural_results)
    nat_dz_curve = np.zeros(max_len_n)
    nat_dz_counts = np.zeros(max_len_n)
    for r in natural_results:
        for i, v in enumerate(r['post_dz']):
            nat_dz_curve[i] += v
            nat_dz_counts[i] += 1
    nat_dz_curve /= np.maximum(nat_dz_counts, 1)

    ax.plot(int_dz_curve, color="#C0392B", label="do(C)", linewidth=2)
    ax.plot(nat_dz_curve, color="#2980B9", label="Natural", linewidth=2)
    ax.set_title("||Δz|| After Majority Change", fontweight="bold")
    ax.set_xlabel("Steps after intervention")
    ax.set_ylabel("||Δz||")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Steps-to-correct histogram
    ax = axes[2]
    bins = range(0, 31, 2)
    ax.hist(int_fc, bins=bins, alpha=0.6, color="#C0392B", label="do(C)", density=True)
    ax.hist(nat_fc, bins=bins, alpha=0.6, color="#2980B9", label="Natural", density=True)
    ax.set_title("Steps to First Correct Action", fontweight="bold")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle("Intervention Test — do(C) vs observe(C)\n"
                 "Does z track the actual hidden state or just observation history?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("intervention_test.png", dpi=150)
    print(f"\nSaved → intervention_test.png")


if __name__ == "__main__":
    main()
