"""
GRU Agent Adapter for the CLW Benchmark
========================================
Wraps the trained CausalBeliefModel (GRUCell 4→128) to conform to
the BenchmarkAgent protocol.

The GRU was trained on a CLW-1-style task (single hidden variable,
obs=[action_norm, last_correct, steps_norm, noise]). Running it on
CLW-2/3 measures cross-environment generalisation — how well a system
trained on one causal structure handles different ones.

Usage:
    from benchmark.gru_agent import GRUBenchmarkAgent
    agent = GRUBenchmarkAgent('causal_belief_v2_final.pt')
    # Now pass to any evaluate_agent() function
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class CausalBeliefModel(nn.Module):
    """Exact copy of the trained model architecture."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(4, 128)
        self.action_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs_t, z_prev):
        z_new = self.gru(obs_t, z_prev)
        return (
            self.action_head(z_new),
            self.value_head(z_new).squeeze(-1),
            z_new,
        )

    def init_belief(self):
        return torch.zeros(1, 128)


class GRUBenchmarkAgent:
    """
    Wraps a trained CausalBeliefModel for benchmark evaluation.
    
    The GRU maintains a persistent hidden state z ∈ R^128 that updates
    on every observation via the GRU cell. Actions are chosen greedily
    (argmax of the policy head) for deterministic scoring.
    
    get_representation() returns [P(C=0)_proxy, P(C=1)_proxy, entropy, confidence]
    estimated from the last action distribution.
    """
    
    def __init__(self, checkpoint_path: str, seed: Optional[int] = None):
        self._model = CausalBeliefModel()
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self._model.load_state_dict(ckpt)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False
        
        self._rng = np.random.RandomState(seed)
        self._z = self._model.init_belief()
        self._last_obs = np.zeros(4, dtype=np.float32)
        self._last_action = 0
        self._last_logits = None
    
    def reset(self) -> None:
        """Reset GRU hidden state for a new episode."""
        self._z = self._model.init_belief()
        self._last_action = 0
        self._last_obs = np.zeros(4, dtype=np.float32)
        self._last_logits = None
    
    def observe(self, obs: np.ndarray) -> None:
        """
        Feed observation through the GRU to update hidden state z.
        
        The base_env provides obs=[action_norm, last_outcome, steps_norm, noise].
        We override obs[0] with our own last_action/2.0 to match training protocol.
        """
        # Match training protocol: obs[0] = last_action / 2.0
        obs_modified = obs.copy()
        obs_modified[0] = self._last_action / 2.0
        
        obs_t = torch.FloatTensor(obs_modified).unsqueeze(0)
        with torch.no_grad():
            logits, _, z_new = self._model(obs_t, self._z)
        
        self._z = z_new.detach()
        self._last_logits = logits.squeeze(0)
        self._last_obs = obs.copy()
    
    def act(self) -> int:
        """Greedy action from policy head."""
        if self._last_logits is None:
            return 0  # first step before any observation
        
        action = int(self._last_logits.argmax().item())
        self._last_action = action
        return action
    
    def get_representation(self) -> Optional[np.ndarray]:
        """
        Extract a representation from the action distribution.
        
        Maps the 3-way softmax to [P(C=0)_proxy, P(C=1)_proxy, entropy, confidence].
        This is a behavioral proxy — the true internal representation is the
        128-dim z vector, but we compress it to the expected 4-dim format.
        """
        if self._last_logits is None:
            return np.array([0.5, 0.5, 1.0, 0.0], dtype=np.float32)
        
        probs = torch.softmax(self._last_logits, dim=-1).numpy()
        # P(C=0) ~ P(action=0), P(C=1) ~ P(action=1)
        p_c0 = float(probs[0])
        p_c1 = float(probs[1])
        
        # Normalise to sum to 1 (excluding wait probability)
        total = p_c0 + p_c1
        if total > 1e-6:
            p_c0_norm = p_c0 / total
            p_c1_norm = p_c1 / total
        else:
            p_c0_norm = p_c1_norm = 0.5
        
        # Action entropy  
        nonzero = probs[probs > 1e-10]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        
        # Confidence: how certain the agent is about C
        confidence = abs(p_c0_norm - 0.5) * 2
        
        return np.array([p_c0_norm, p_c1_norm, entropy, confidence], dtype=np.float32)
