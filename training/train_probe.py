"""
Non-linear Probe for z_w — does the world model encode C?
==========================================================

Trains a small MLP classifier on (z_w → majority_label) pairs collected
from the frozen v3 model. If PCA shows weak linear correlation (0.128)
but the intervention test shows 80% accuracy, the encoding is non-linear.
This probe tests that hypothesis directly.

Protocol:
  1. Run the frozen model for 2000 episodes, collect (z_w, label) at each step
  2. Split 80/20 train/test
  3. Train a 2-layer MLP: Linear(128, 32) → ReLU → Linear(32, 1) → Sigmoid
  4. Report accuracy on held-out test set

If probe accuracy > 90%, z_w encodes C non-linearly.
If probe accuracy < 60%, z_w does NOT encode C.

Usage:
  python train_probe.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys, time


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIMULATOR + MODEL (copies for standalone execution)
# ══════════════════════════════════════════════════════════════════════════════
class CausalSimulator:
    def __init__(self, hidden_dim=8, flip_mean=80, max_steps=200):
        self.hidden_dim  = hidden_dim
        self.flip_mean   = flip_mean
        self.max_steps   = max_steps
        self.step_cost   = -0.01
        self.pull_cost   = -0.05

    def reset(self):
        self.C = np.random.randint(0, 2, self.hidden_dim).astype(float)
        self.steps = 0
        self.flip_timers = np.random.geometric(1.0 / self.flip_mean, self.hidden_dim)
        self._last_correct = 0.0
        return self._get_obs()

    def step(self, action):
        reward = self.step_cost
        done = False

        if action == 2:
            self.steps += 1
            if self.steps >= self.max_steps: done = True
            return self._get_obs(), reward, done

        majority = 1 if np.sum(self.C) > self.hidden_dim // 2 else 0
        corr_flip = (self.C[0] == 1 and self.C[1] == 1)
        correct_action = 1 - majority if corr_flip else majority
        is_correct = (action == correct_action)
        self._last_correct = 1.0 if is_correct else 0.0

        reward += self.pull_cost + (0.2 if is_correct else -0.2)
        self.steps += 1
        self.flip_timers -= 1
        for i in range(self.hidden_dim):
            if self.flip_timers[i] <= 0:
                self.C[i] = 1 - self.C[i]
                self.flip_timers[i] = np.random.geometric(1.0 / self.flip_mean)

        if self.steps >= self.max_steps: done = True
        return self._get_obs(), reward, done

    def _get_obs(self):
        return np.array([0.0, self._last_correct,
                         min(self.steps / 50.0, 1.0),
                         float(np.random.randn())], dtype=np.float32)

    def get_majority(self):
        return 1 if np.sum(self.C) > self.hidden_dim // 2 else 0


class SeparatedCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.world_gru = nn.GRUCell(4, 128)
        self.policy_gru = nn.GRUCell(132, 64)
        self.action_head = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 3))
        self.value_head = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs_t, z_w_prev, z_p_prev):
        z_w_new = self.world_gru(obs_t, z_w_prev)
        z_w_d = z_w_new.detach()
        z_p_new = self.policy_gru(torch.cat([z_w_d, obs_t], -1), z_p_prev)
        combined = torch.cat([z_w_d, z_p_new], -1)
        return (self.action_head(combined),
                self.value_head(combined).squeeze(-1),
                z_w_new, z_p_new)

    def init_world_belief(self):  return torch.zeros(1, 128)
    def init_policy_belief(self): return torch.zeros(1, 64)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROBE ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class NonlinearProbe(nn.Module):
    """Small MLP: 128 → 32 → 1. Tests non-linear decodability of C from z_w."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, z_w):
        return self.net(z_w).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
def collect_z_w_data(model, n_episodes=2000):
    """Run frozen model, collect (z_w, majority_label) pairs at every step."""
    env = CausalSimulator(hidden_dim=8, flip_mean=80)
    z_w = model.init_world_belief()
    z_p = model.init_policy_belief()

    all_z_w = []
    all_labels = []

    for ep in range(n_episodes):
        obs_np = env.reset()
        done = False
        last_action = 0

        while not done:
            obs_np[0] = last_action / 2.0
            obs_t = torch.FloatTensor(obs_np).unsqueeze(0)

            with torch.no_grad():
                logits, _, z_w_new, z_p_new = model(obs_t, z_w, z_p)

            action = logits.argmax(-1).item()
            obs_np, _, done = env.step(action)
            last_action = action

            z_w = z_w_new.detach()
            z_p = z_p_new.detach()

            # Record z_w and current majority
            all_z_w.append(z_w.squeeze(0).numpy().copy())
            all_labels.append(float(env.get_majority()))

        if (ep + 1) % 500 == 0:
            print(f"  Collected {len(all_z_w):,} samples from {ep+1} episodes")

    return np.array(all_z_w), np.array(all_labels)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_probe(z_w_data, labels, n_epochs=100, lr=1e-3):
    """Train non-linear probe and report accuracy."""
    # Split 80/20
    n = len(z_w_data)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train = torch.FloatTensor(z_w_data[train_idx])
    y_train = torch.FloatTensor(labels[train_idx])
    X_test  = torch.FloatTensor(z_w_data[test_idx])
    y_test  = torch.FloatTensor(labels[test_idx])

    probe = NonlinearProbe()
    opt = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCELoss()

    batch_size = 256

    for epoch in range(n_epochs):
        probe.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            x_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            pred = probe(x_batch)
            loss = criterion(pred, y_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            probe.eval()
            with torch.no_grad():
                train_pred = (probe(X_train) > 0.5).float()
                train_acc = (train_pred == y_train).float().mean()
                test_pred = (probe(X_test) > 0.5).float()
                test_acc = (test_pred == y_test).float().mean()
            print(f"  Epoch {epoch+1:3d} | loss {epoch_loss/n_batches:.4f} | "
                  f"train_acc {train_acc:.1%} | test_acc {test_acc:.1%}")

    # Final evaluation
    probe.eval()
    with torch.no_grad():
        test_pred = (probe(X_test) > 0.5).float()
        test_acc = (test_pred == y_test).float().mean()
        test_probs = probe(X_test)

    # Also report linear baseline for comparison
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(z_w_data[train_idx], labels[train_idx])
    linear_acc = lr_model.score(z_w_data[test_idx], labels[test_idx])

    return probe, float(test_acc), float(linear_acc)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("═" * 60)
    print("NON-LINEAR PROBE — Does z_w encode C?")
    print("═" * 60)

    # Load frozen model
    model = SeparatedCausalModel()
    ckpt = torch.load("causal_belief_v3_final.pt", map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("Loaded frozen model: causal_belief_v3_final.pt")

    # Collect data
    print(f"\nCollecting z_w samples (2000 episodes)...")
    z_w_data, labels = collect_z_w_data(model, n_episodes=2000)
    print(f"Total samples: {len(z_w_data):,}")
    print(f"Label balance: {labels.mean():.1%} majority=1")

    # Train probe
    print(f"\nTraining non-linear probe (128 → 32 → 1)...")
    probe, mlp_acc, linear_acc = train_probe(z_w_data, labels)

    # Results
    print(f"\n{'═'*60}")
    print("RESULTS")
    print(f"{'═'*60}")
    print(f"  Linear probe (logistic regression): {linear_acc:.1%}")
    print(f"  Non-linear probe (MLP):             {mlp_acc:.1%}")
    print(f"  PCA correlation (from training):     12.8%")
    print(f"  Chance baseline:                     50.0%")

    if mlp_acc > 0.90:
        print(f"\n✓ z_w encodes C NON-LINEARLY (MLP acc = {mlp_acc:.1%})")
        print("  → The world model learned a non-linear representation of hidden state")
        print("  → PCA misses it because it's a linear method")
    elif mlp_acc > 0.70:
        print(f"\n~ z_w partially encodes C (MLP acc = {mlp_acc:.1%})")
        print("  → Some hidden state information present but not fully decodable")
    else:
        print(f"\n✗ z_w does NOT encode C (MLP acc = {mlp_acc:.1%})")
        print("  → The world model's good intervention performance comes from another mechanism")

    if mlp_acc > linear_acc + 0.05:
        print(f"\n  Non-linear > Linear by {mlp_acc - linear_acc:.1%}")
        print("  → Confirms non-linear encoding (PCA/linear methods are insufficient)")
    else:
        print(f"\n  Non-linear ≈ Linear (diff = {mlp_acc - linear_acc:.1%})")
        print("  → Encoding is approximately linear (PCA should have found it)")

    # Save probe alongside model
    torch.save({
        'probe_state_dict': probe.state_dict(),
        'mlp_accuracy': mlp_acc,
        'linear_accuracy': linear_acc,
        'n_samples': len(z_w_data),
    }, "v3_probe.pt")
    print(f"\nSaved → v3_probe.pt")


if __name__ == "__main__":
    main()
