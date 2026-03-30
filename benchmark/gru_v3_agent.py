"""
GRU v3 Agent Adapter for the CLW Benchmark
============================================
Wraps the SeparatedCausalModel (world_gru 4→128, policy_gru 132→64)
to conform to the BenchmarkAgent protocol.

Key difference from gru_agent.py:
  get_representation() extracts from z_w (world model state) via PCA,
  not from the action distribution. This gives Test B-full a direct
  signal from the world model rather than a behavioral proxy.

Usage:
    from benchmark.gru_v3_agent import GRUV3BenchmarkAgent
    agent = GRUV3BenchmarkAgent('causal_belief_v3_final.pt')
    # Now pass to any evaluate_agent() function
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class SeparatedCausalModel(nn.Module):
    """Exact copy of the trained model architecture."""
    def __init__(self):
        super().__init__()
        self.world_gru = nn.GRUCell(4, 128)
        self.policy_gru = nn.GRUCell(128 + 4, 64)
        self.action_head = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.value_head = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs_t, z_w_prev, z_p_prev):
        z_w_new = self.world_gru(obs_t, z_w_prev)
        z_w_detached = z_w_new.detach()
        policy_input = torch.cat([z_w_detached, obs_t], dim=-1)
        z_p_new = self.policy_gru(policy_input, z_p_prev)
        combined = torch.cat([z_w_detached, z_p_new], dim=-1)
        return (
            self.action_head(combined),
            self.value_head(combined).squeeze(-1),
            z_w_new,
            z_p_new,
        )

    def init_world_belief(self):
        return torch.zeros(1, 128)

    def init_policy_belief(self):
        return torch.zeros(1, 64)


class NonlinearProbe(nn.Module):
    """Small MLP: 128 → 32 → 1. Decodes P(majority=1) from z_w."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, z_w):
        return self.net(z_w).squeeze(-1)


class GRUV3BenchmarkAgent:
    """
    Wraps a trained SeparatedCausalModel for benchmark evaluation.

    Representation priority for get_representation():
      1. Non-linear probe (if v3_probe.pt exists) — 70.3% accuracy
      2. PCA projection (if checkpoint has PCA data) — 12.8% correlation
      3. Action distribution fallback
    """

    def __init__(self, checkpoint_path: str, seed: Optional[int] = None,
                 probe_path: Optional[str] = None):
        import os

        self._model = SeparatedCausalModel()
        # weights_only=False needed: checkpoint contains numpy PCA arrays
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self._model.load_state_dict(ckpt['model_state_dict'])
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

        # --- Load non-linear probe (preferred) ---
        self._has_probe = False
        if probe_path is None:
            # Auto-discover probe next to checkpoint
            ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            probe_path = os.path.join(ckpt_dir, 'v3_probe.pt')
        if os.path.exists(probe_path):
            self._probe = NonlinearProbe()
            probe_ckpt = torch.load(probe_path, map_location='cpu',
                                    weights_only=True)
            self._probe.load_state_dict(probe_ckpt['probe_state_dict'])
            self._probe.eval()
            for p in self._probe.parameters():
                p.requires_grad = False
            self._has_probe = True
            print(f"  Loaded non-linear probe: {probe_path} "
                  f"(acc={probe_ckpt.get('mlp_accuracy', '?'):.1%})")

        # --- Load PCA data (fallback) ---
        self._pca_data = ckpt.get('pca', {})
        self._has_pca = bool(self._pca_data)
        if self._has_pca:
            self._pca_components = self._pca_data['components']  # (2, 128)
            self._pca_mean       = self._pca_data['mean']        # (128,)
            self._pca_sign       = self._pca_data['sign']        # +1 or -1
            self._pca_scale      = self._pca_data['scale']       # float

        self._rng = np.random.RandomState(seed)
        self._z_w = self._model.init_world_belief()
        self._z_p = self._model.init_policy_belief()
        self._last_action = 0
        self._last_logits = None

    def reset(self) -> None:
        """Reset both hidden states for a new episode."""
        self._z_w = self._model.init_world_belief()
        self._z_p = self._model.init_policy_belief()
        self._last_action = 0
        self._last_logits = None

    def observe(self, obs: np.ndarray) -> None:
        """
        Feed observation through both GRUs.
        z_w updates from obs only (world model).
        z_p reads z_w (detached) + obs (policy).
        """
        obs_modified = obs.copy()
        obs_modified[0] = self._last_action / 2.0

        obs_t = torch.FloatTensor(obs_modified).unsqueeze(0)
        with torch.no_grad():
            logits, _, z_w_new, z_p_new = self._model(obs_t, self._z_w, self._z_p)

        self._z_w = z_w_new.detach()
        self._z_p = z_p_new.detach()
        self._last_logits = logits.squeeze(0)

    def act(self) -> int:
        """Greedy action from policy head."""
        if self._last_logits is None:
            return 0
        action = int(self._last_logits.argmax().item())
        self._last_action = action
        return action

    def get_representation(self) -> Optional[np.ndarray]:
        """
        Extract representation from z_w.

        Priority: probe (non-linear, 70% acc) > PCA (linear) > action dist.
        Returns [P(C=0), P(C=1), H, confidence].
        """
        if self._has_probe:
            return self._get_probe_representation()
        if self._has_pca:
            return self._get_pca_representation()
        return self._get_action_representation()

    def _get_probe_representation(self) -> np.ndarray:
        """Use trained MLP probe to decode P(majority=1) from z_w."""
        with torch.no_grad():
            p_c1 = float(self._probe(self._z_w).item())
        p_c0 = 1.0 - p_c1

        eps = 1e-10
        entropy = float(-p_c0 * np.log2(p_c0 + eps) - p_c1 * np.log2(p_c1 + eps))
        confidence = abs(p_c0 - 0.5) * 2

        return np.array([p_c0, p_c1, entropy, confidence], dtype=np.float32)

    def _get_pca_representation(self) -> np.ndarray:
        """Project z_w through PCA → probability estimate."""
        z_w_np = self._z_w.squeeze(0).numpy()

        # Project onto first PC
        centered = z_w_np - self._pca_mean
        pc1 = float(np.dot(centered, self._pca_components[0]))

        # Apply sign and scale → sigmoid → P(C=1)
        x = self._pca_sign * pc1 * self._pca_scale
        p_c1 = 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
        p_c0 = 1.0 - p_c1

        # Entropy of the belief
        eps = 1e-10
        entropy = float(-p_c0 * np.log2(p_c0 + eps) - p_c1 * np.log2(p_c1 + eps))

        # Confidence
        confidence = abs(p_c0 - 0.5) * 2

        return np.array([p_c0, p_c1, entropy, confidence], dtype=np.float32)

    def _get_action_representation(self) -> np.ndarray:
        """Fallback: same as gru_agent.py — derive from action distribution."""
        if self._last_logits is None:
            return np.array([0.5, 0.5, 1.0, 0.0], dtype=np.float32)

        probs = torch.softmax(self._last_logits, dim=-1).numpy()
        p_c0 = float(probs[0])
        p_c1 = float(probs[1])

        total = p_c0 + p_c1
        if total > 1e-6:
            p_c0_norm = p_c0 / total
            p_c1_norm = p_c1 / total
        else:
            p_c0_norm = p_c1_norm = 0.5

        nonzero = probs[probs > 1e-10]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        confidence = abs(p_c0_norm - 0.5) * 2

        return np.array([p_c0_norm, p_c1_norm, entropy, confidence], dtype=np.float32)
