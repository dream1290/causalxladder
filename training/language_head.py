"""
Language Head — Phase 2
=======================
Decodes the persistent belief state z into human-readable descriptions
of the model's causal beliefs about the hidden world state C.

Architecture:
  - CausalBeliefModel weights: FROZEN (loaded from phase B checkpoint)
  - LanguageHead: z (128) → 3-layer MLP → token logits over small vocabulary
  - Training: supervised on (z_snapshot, label) pairs collected from live rollouts
  - Labels: generated from simulator ground truth — no human annotation needed

Output vocabulary (3 dimensions × levels):
  Confidence : "High" | "Medium" | "Low"
  Majority   : "Majority-1" | "Majority-0" | "Uncertain"
  Dynamics   : "Stable" | "Flux" | "Post-flip"

Example outputs:
  "High confidence. Majority-1. Stable."
  "Low confidence. Uncertain. Post-flip."
  "Medium confidence. Majority-0. Flux."

Run:
  python language_head.py --checkpoint causal_belief_v2_final.pt
  python language_head.py --checkpoint causal_belief_v2_final.pt --demo

Requires: torch, numpy, matplotlib, sklearn
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIMULATOR  (identical to Phase 1 — do not modify)
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
        self.total_steps  = 10_000_000   # bypass curriculum — streak=8 always

    @property
    def streak_required(self): return 8

    @property
    def shaped_correct(self): return 0.5

    def reset(self):
        self.C              = np.random.randint(0, 2, self.hidden_dim).astype(float)
        self.steps          = 0
        self.correct_consec = 0
        self.wrong_consec   = 0
        self.flip_timers    = np.random.geometric(1.0/self.flip_mean, self.hidden_dim)
        self._last_correct  = 0.0
        self._steps_since_flip = 0
        self._recent_flip   = False
        return self._get_obs()

    def step(self, action):
        reward = self.step_cost
        done   = False
        self._recent_flip = False

        if action == 2:
            self.steps += 1
            self.correct_consec = self.wrong_consec = 0
            self._steps_since_flip += 1
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
        self._steps_since_flip += 1
        self.flip_timers -= 1
        for i in range(self.hidden_dim):
            if self.flip_timers[i] <= 0:
                self.C[i]             = 1 - self.C[i]
                self.flip_timers[i]   = np.random.geometric(1.0/self.flip_mean)
                self._steps_since_flip = 0
                self._recent_flip      = True

        if self.steps >= self.max_steps: done = True
        return self._get_obs(), reward, done

    def _get_obs(self):
        return np.array([
            0.0,
            self._last_correct,
            min(self.steps / 50.0, 1.0),
            float(np.random.randn()),
        ], dtype=np.float32)

    # ── Ground truth label extraction ─────────────────────────────────────
    def get_belief_label(self):
        """
        Returns a structured label dict describing the true world state.
        Used to generate supervised training signal for the language head.
        """
        majority = 1 if np.sum(self.C) > self.hidden_dim // 2 else 0
        margin   = abs(np.sum(self.C) - self.hidden_dim / 2) / (self.hidden_dim / 2)

        # Confidence from margin + recency of flip
        if self._recent_flip or self._steps_since_flip < 5:
            confidence = "Low"
        elif margin > 0.5:
            confidence = "High"
        else:
            confidence = "Medium"

        # Majority
        if margin < 0.25:
            majority_str = "Uncertain"
        elif majority == 1:
            majority_str = "Majority-1"
        else:
            majority_str = "Majority-0"

        # Dynamics
        min_timer = min(self.flip_timers)
        if self._recent_flip or self._steps_since_flip < 8:
            dynamics = "Post-flip"
        elif min_timer < 10:
            dynamics = "Flux"
        else:
            dynamics = "Stable"

        return {
            "confidence": confidence,
            "majority":   majority_str,
            "dynamics":   dynamics,
            "text":       f"{confidence} confidence. {majority_str}. {dynamics}.",
            "majority_int": majority,
            "margin":     margin,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. FROZEN BELIEF MODEL  (exact copy from Phase 1)
# ══════════════════════════════════════════════════════════════════════════════
class CausalBeliefModel(nn.Module):
    def __init__(self, obs_dim=4, hidden_dim=128, action_dim=3):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.gru         = nn.GRUCell(obs_dim, hidden_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )
        self.value_head  = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
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
        return torch.zeros(1, self.hidden_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 3. VOCABULARY
# ══════════════════════════════════════════════════════════════════════════════
CONFIDENCE_VOCAB = ["High", "Medium", "Low"]
MAJORITY_VOCAB   = ["Majority-1", "Majority-0", "Uncertain"]
DYNAMICS_VOCAB   = ["Stable", "Flux", "Post-flip"]

C_IDX = {v: i for i, v in enumerate(CONFIDENCE_VOCAB)}
M_IDX = {v: i for i, v in enumerate(MAJORITY_VOCAB)}
D_IDX = {v: i for i, v in enumerate(DYNAMICS_VOCAB)}

def label_to_indices(label):
    return (
        C_IDX[label["confidence"]],
        M_IDX[label["majority"]],
        D_IDX[label["dynamics"]],
    )

def indices_to_text(c_idx, m_idx, d_idx):
    return (f"{CONFIDENCE_VOCAB[c_idx]} confidence. "
            f"{MAJORITY_VOCAB[m_idx]}. "
            f"{DYNAMICS_VOCAB[d_idx]}.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. LANGUAGE HEAD
# ══════════════════════════════════════════════════════════════════════════════
class LanguageHead(nn.Module):
    """
    Maps z (128) → 3 independent classifiers:
      - confidence (3 classes)
      - majority   (3 classes)
      - dynamics   (3 classes)

    Three separate heads allow each dimension to specialise independently.
    Shared trunk extracts common features from z.
    """
    def __init__(self, z_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.confidence_head = nn.Linear(64, len(CONFIDENCE_VOCAB))
        self.majority_head   = nn.Linear(64, len(MAJORITY_VOCAB))
        self.dynamics_head   = nn.Linear(64, len(DYNAMICS_VOCAB))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        h = self.trunk(z)
        return (
            self.confidence_head(h),
            self.majority_head(h),
            self.dynamics_head(h),
        )

    def decode(self, z):
        """Greedy decode: z tensor → text string"""
        with torch.no_grad():
            c_log, m_log, d_log = self(z)
            c = c_log.argmax(-1).item()
            m = m_log.argmax(-1).item()
            d = d_log.argmax(-1).item()
        return indices_to_text(c, m, d), c, m, d


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
def collect_dataset(belief_model, env, n_episodes=3000):
    """
    Run belief_model on env with persistent z.
    At each step, record (z, ground_truth_label).
    Returns tensors ready for training.
    """
    print(f"\nCollecting dataset: {n_episodes} episodes...")
    belief_model.eval()

    z_list, c_list, m_list, d_list = [], [], [], []
    z = belief_model.init_belief()
    last_action = 0

    for ep in range(n_episodes):
        obs_np = env.reset()
        done   = False

        while not done:
            obs_np[0] = last_action / 2.0
            obs_t     = torch.FloatTensor(obs_np).unsqueeze(0)

            with torch.no_grad():
                _, _, z_new, _ = belief_model(obs_t, z)

            z = z_new.detach()

            # Record z and ground truth label at this moment
            label   = env.get_belief_label()
            c, m, d = label_to_indices(label)

            z_list.append(z.squeeze(0).numpy().copy())
            c_list.append(c)
            m_list.append(m)
            d_list.append(d)

            obs_np, _, done = env.step(
                torch.argmax(
                    belief_model.action_head(z)
                ).item()
            )
            last_action = obs_np[0]

        if (ep + 1) % 500 == 0:
            print(f"  Collected ep {ep+1}/{n_episodes} — "
                  f"{len(z_list)} samples so far")

    Z = torch.FloatTensor(np.array(z_list))
    C = torch.LongTensor(c_list)
    M = torch.LongTensor(m_list)
    D = torch.LongTensor(d_list)

    print(f"Dataset: {len(Z)} samples collected.")
    print(f"  Confidence distribution: "
          f"{[(CONFIDENCE_VOCAB[i], (C==i).sum().item()) for i in range(3)]}")
    print(f"  Majority distribution:   "
          f"{[(MAJORITY_VOCAB[i],   (M==i).sum().item()) for i in range(3)]}")
    print(f"  Dynamics distribution:   "
          f"{[(DYNAMICS_VOCAB[i],   (D==i).sum().item()) for i in range(3)]}")

    return Z, C, M, D


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_language_head(Z, C, M, D,
                        n_epochs=50, batch_size=256, lr=1e-3):
    print(f"\nTraining language head: {n_epochs} epochs, batch={batch_size}")

    lang_head = LanguageHead(z_dim=128)
    opt       = optim.Adam(lang_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Inverse-frequency weights for majority head (fix class imbalance)
    m_counts = torch.FloatTensor([(M == i).sum().item() for i in range(3)])
    m_weights = m_counts.sum() / (3.0 * m_counts + 1e-8)
    m_weights = m_weights / m_weights.min()  # normalize so smallest weight = 1.0
    criterion_m = nn.CrossEntropyLoss(weight=m_weights)
    print(f"  Majority class weights: {[f'{w:.2f}' for w in m_weights.tolist()]}")

    n      = len(Z)
    idx    = torch.randperm(n)
    split  = int(n * 0.85)
    tr_idx = idx[:split]
    va_idx = idx[split:]

    Z_tr, C_tr, M_tr, D_tr = Z[tr_idx], C[tr_idx], M[tr_idx], D[tr_idx]
    Z_va, C_va, M_va, D_va = Z[va_idx], C[va_idx], M[va_idx], D[va_idx]

    train_losses, val_accs = [], []
    t_start = time.time()

    for epoch in range(n_epochs):
        lang_head.train()
        perm  = torch.randperm(len(Z_tr))
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(Z_tr), batch_size):
            b    = perm[i:i+batch_size]
            z_b  = Z_tr[b]
            c_b, m_b, d_b = C_tr[b], M_tr[b], D_tr[b]

            c_log, m_log, d_log = lang_head(z_b)
            loss = (criterion(c_log, c_b) +
                    criterion_m(m_log, m_b) +   # weighted for class balance
                    criterion(d_log, d_b))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(lang_head.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        # Validation
        lang_head.eval()
        with torch.no_grad():
            c_log, m_log, d_log = lang_head(Z_va)
            c_acc = (c_log.argmax(-1) == C_va).float().mean().item()
            m_acc = (m_log.argmax(-1) == M_va).float().mean().item()
            d_acc = (d_log.argmax(-1) == D_va).float().mean().item()
            avg_acc = (c_acc + m_acc + d_acc) / 3

        avg_loss = total_loss / n_batches
        train_losses.append(avg_loss)
        val_accs.append(avg_acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | loss {avg_loss:.4f} | "
                  f"val acc: conf={c_acc:.2%} maj={m_acc:.2%} dyn={d_acc:.2%} "
                  f"| avg={avg_acc:.2%} | time {int(time.time()-t_start)}s")

    return lang_head, train_losses, val_accs, (c_acc, m_acc, d_acc)


# ══════════════════════════════════════════════════════════════════════════════
# 7. DEMO — live belief narration
# ══════════════════════════════════════════════════════════════════════════════
def run_demo(belief_model, lang_head, env, n_episodes=5):
    """
    Run live episodes, printing z → language at each step.
    Shows the model narrating its own causal beliefs in real time.
    """
    print("\n" + "═"*60)
    print("LIVE BELIEF NARRATION DEMO")
    print("Model narrates its internal belief state at each step.")
    print("═"*60)

    belief_model.eval()
    lang_head.eval()
    z = belief_model.init_belief()

    for ep in range(n_episodes):
        obs_np      = env.reset()
        done        = False
        last_action = 0
        step        = 0

        true_majority = 1 if np.sum(env.C) > env.hidden_dim // 2 else 0
        print(f"\n─── Episode {ep+1} "
              f"(True hidden state: C={env.C.astype(int).tolist()}, "
              f"majority={true_majority}) ───")

        while not done and step < 30:   # cap at 30 steps for readability
            obs_np[0] = last_action / 2.0
            obs_t     = torch.FloatTensor(obs_np).unsqueeze(0)

            with torch.no_grad():
                logits, _, z_new, dz_norm = belief_model(obs_t, z)
                action = logits.argmax(-1).item()
                text, c_i, m_i, d_i = lang_head.decode(z_new)

            true_label = env.get_belief_label()
            action_str = ["Red", "Blue", "Wait"][action]

            # Check correctness
            majority   = 1 if np.sum(env.C) > env.hidden_dim // 2 else 0
            corr_flip  = (env.C[0] == 1 and env.C[1] == 1)
            correct_a  = 1 - majority if corr_flip else majority
            correct_str = "✓" if action == correct_a else "✗"

            print(f"  Step {step+1:3d} | Action: {action_str} {correct_str} | "
                  f"||Δz||={dz_norm.item():.3f} | "
                  f"Belief: \"{text}\" | "
                  f"True: {true_label['text']}")

            obs_np, reward, done = env.step(action)
            last_action = action
            z = z_new.detach()

            # Highlight flips
            if env._recent_flip:
                new_majority = 1 if np.sum(env.C) > env.hidden_dim // 2 else 0
                print(f"  *** HIDDEN STATE FLIPPED → new majority={new_majority} ***")

            step += 1

        if done:
            print(f"  Episode ended (reward signal triggered)")


# ══════════════════════════════════════════════════════════════════════════════
# 8. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
def plot_language_diagnostics(train_losses, val_accs, final_accs,
                               Z, C, M, D, lang_head, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes.ravel()

    # Training curve
    ax[0].plot(train_losses, color="#1F3864", label="Train loss")
    ax[0].set_title("Language Head — Training Loss", fontweight="bold")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Cross-entropy loss")
    ax[0].grid(alpha=0.3)

    # Validation accuracy
    ax[1].plot(val_accs, color="#1F6420", label="Avg val accuracy")
    ax[1].axhline(1/3, color="gray", linestyle="--",
                  linewidth=0.8, label="Random baseline (33%)")
    ax[1].set_title("Language Head — Validation Accuracy", fontweight="bold")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(0, 1.0)

    # PCA of z coloured by PREDICTED majority
    lang_head.eval()
    with torch.no_grad():
        _, m_log, _ = lang_head(Z)
        pred_majority = m_log.argmax(-1).numpy()

    pca = PCA(n_components=2)
    Z2  = pca.fit_transform(Z.numpy())
    var = pca.explained_variance_ratio_

    # True majority labels (0=Majority-0, 1=Majority-1, 2=Uncertain)
    true_binary = (M == 0).numpy().astype(float)   # 1 if Majority-1
    sc = ax[2].scatter(Z2[:,0], Z2[:,1],
                       c=true_binary, cmap="RdBu", alpha=0.4, s=6)
    plt.colorbar(sc, ax=ax[2], label="True: Majority-1 (blue) vs Majority-0 (red)")
    ax[2].set_title(f"PCA of z — True World State\n"
                    f"PC1={var[0]:.1%}, PC2={var[1]:.1%}",
                    fontweight="bold")
    ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2"); ax[2].grid(alpha=0.3)

    # PCA coloured by PREDICTED majority
    pred_binary = (pred_majority == 0).astype(float)
    sc2 = ax[3].scatter(Z2[:,0], Z2[:,1],
                        c=pred_binary, cmap="RdBu", alpha=0.4, s=6)
    plt.colorbar(sc2, ax=ax[3], label="Predicted: Majority-1 (blue) vs other (red)")
    ax[3].set_title(f"PCA of z — Language Head Predictions\n"
                    f"Majority acc={final_accs[1]:.1%}",
                    fontweight="bold")
    ax[3].set_xlabel("PC1"); ax[3].set_ylabel("PC2"); ax[3].grid(alpha=0.3)

    plt.suptitle(
        "Language Head — Decoding Causal Beliefs from z\n"
        f"Final accuracy: Confidence={final_accs[0]:.1%}, "
        f"Majority={final_accs[1]:.1%}, Dynamics={final_accs[2]:.1%}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "language_head_diagnostics.png")
    plt.savefig(path, dpi=150)
    print(f"\nSaved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="causal_belief_v2_final.pt",
                        help="Path to Phase B model checkpoint")
    parser.add_argument("--demo", action="store_true",
                        help="Run live narration demo after training")
    parser.add_argument("--collect-only", action="store_true",
                        help="Only collect dataset, no training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-collect", type=int, default=3000,
                        help="Episodes to collect for dataset")
    args = parser.parse_args()

    print("═"*60)
    print("Language Head — Phase 2")
    print(f"Checkpoint: {args.checkpoint}")
    print("═"*60)

    # ── Load frozen belief model ──────────────────────────────────────────
    belief_model = CausalBeliefModel()
    state = torch.load(args.checkpoint, map_location="cpu")
    belief_model.load_state_dict(state)
    belief_model.eval()
    for p in belief_model.parameters():
        p.requires_grad = False
    print(f"Loaded and frozen: {args.checkpoint}")
    n_params = sum(p.numel() for p in belief_model.parameters())
    print(f"Belief model parameters: {n_params:,}")

    env = CausalSimulator(hidden_dim=8, flip_mean=80)

    # ── Collect dataset ───────────────────────────────────────────────────
    Z, C, M, D = collect_dataset(belief_model, env,
                                 n_episodes=args.n_collect)

    if args.collect_only:
        torch.save({"Z": Z, "C": C, "M": M, "D": D}, "lang_dataset.pt")
        print("Dataset saved → lang_dataset.pt")
        exit(0)

    # ── Train language head ───────────────────────────────────────────────
    lang_head, losses, accs, final_accs = train_language_head(
        Z, C, M, D, n_epochs=args.epochs
    )

    torch.save(lang_head.state_dict(), "language_head.pt")
    print(f"\nLanguage head saved → language_head.pt")
    n_lang = sum(p.numel() for p in lang_head.parameters())
    print(f"Language head parameters: {n_lang:,}")

    # ── Diagnostics ───────────────────────────────────────────────────────
    plot_language_diagnostics(losses, accs, final_accs, Z, C, M, D, lang_head)

    # ── Final accuracy report ─────────────────────────────────────────────
    print("\n" + "═"*60)
    print("FINAL ACCURACY REPORT")
    print("═"*60)
    labels = ["Confidence", "Majority", "Dynamics"]
    for name, acc in zip(labels, final_accs):
        bar   = "█" * int(acc * 20)
        color = "✓" if acc > 0.60 else ("~" if acc > 0.45 else "✗")
        print(f"  {color} {name:12s}: {acc:.1%}  {bar}")
    print(f"\n  Random baseline: 33.3% per dimension")
    print(f"  Majority accuracy is the key metric — "
          f"directly measures causal world-state decoding")

    # ── Demo ──────────────────────────────────────────────────────────────
    if args.demo:
        run_demo(belief_model, lang_head, env, n_episodes=5)

    print("\n" + "═"*60)
    if final_accs[1] > 0.60:
        print("✓ Language head successfully decodes causal beliefs from z")
        print("✓ The belief state encodes interpretable world-model content")
        print("\n→ READY: model can narrate its own causal reasoning")
    else:
        print(f"~ Majority accuracy {final_accs[1]:.1%} — above chance but improvable")
        print("  Try --epochs 100 or --n-collect 5000")
    print("═"*60)
