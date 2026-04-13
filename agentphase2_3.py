import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class DQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


_model = None


def _load_once():
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. "
            "Include agent.py and weights.pth in submission.zip."
        )

    model = DQN(in_dim=18, n_actions=5, hidden_dim=256)

    state = torch.load(wpath, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()
    _model = model


@torch.no_grad()
def policy(obs, rng=None):
    _load_once()

    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 1:
        obs = obs.reshape(-1)

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q_values = _model(x).squeeze(0).cpu().numpy()
    action_idx = int(np.argmax(q_values))
    return ACTIONS[action_idx]
