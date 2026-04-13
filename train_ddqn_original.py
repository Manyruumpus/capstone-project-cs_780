"""Offline trainer: Double DQN + replay buffer (GPU-aware) for OBELIX.

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
  python train_dqn.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --difficulty 0 --wall_obstacles

                        ALGORITHM: DOUBLE DEEP Q-NETWORK (DDQN)


Double DQN is one of the most widely used and reliable improvements over the original Deep Q-Network (DQN).

Main problems it solves:
Vanilla DQN often overestimates true action values.
This happens because the same network is used twice:
   1. to pick the best-looking action in the next state (max)
   2. to evaluate how good that action actually is

When Q-values are noisy (which they almost always are early in training),
this double usage creates optimistic bias â†’ the agent thinks some
actions are much better than they really are â†’ leads to unstable learning.

Double DQN solution:
Split the responsibilities:
â€¢ Use the online / main Q-network  to SELECT which action looks best
â€¢ Use the target Q-network to EVALUATE (give the actual value)

So instead of:

    target = r + Î³ Ã— max_a Q_target(s', a)

We do:

    target = r + Î³ Ã— Q_target( s',   argmax_a Q_online(s', a)   )

This small change dramatically reduces overestimation and makes learning
much more stable â€” especially in environments with large action spaces
or noisy rewards.

For More Details please refer to https://arxiv.org/pdf/1509.06461 .


"""

from __future__ import annotations
import argparse, random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class DQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=64):
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

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class Replay:
    def __init__(self, cap: int = 100_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)
    def add(self, t: Transition):
        self.buf.append(t)
    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d
    def __len__(self): return len(self.buf)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--updates_per_step", type=int, default=1)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=2000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="compute device for training (auto picks cuda when available)",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    OBELIX = import_obelix(args.obelix_py)

    q = DQN(hidden_dim=args.hidden_dim).to(device)
    tgt = DQN(hidden_dim=args.hidden_dim).to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    steps = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )
        s = env.reset(seed=args.seed + ep)
        ep_ret = 0.0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    qs = q(s_t).squeeze(0).detach().cpu().numpy()
                a = int(np.argmax(qs))

            s2, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += float(r)
            replay.add(Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done)))
            s = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                for _ in range(args.updates_per_step):
                    sb, ab, rb, s2b, db = replay.sample(args.batch)
                    sb_t = torch.tensor(sb, device=device)
                    ab_t = torch.tensor(ab, device=device)
                    rb_t = torch.tensor(rb, device=device)
                    s2b_t = torch.tensor(s2b, device=device)
                    db_t = torch.tensor(db, device=device)

                    with torch.no_grad():
                        next_q = q(s2b_t)
                        next_a = torch.argmax(next_q, dim=1)
                        next_q_tgt = tgt(s2b_t)
                        next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                        y = rb_t + args.gamma * (1.0 - db_t) * next_val

                    pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                    loss = nn.functional.smooth_l1_loss(pred, y)

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                    opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        if (ep + 1) % 50 == 0:
            if device.type == "cuda":
                mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(
                    f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} eps={eps_by_step(steps):.3f} replay={len(replay)} "
                    f"gpu_mem={mem_gb:.2f}GB peak={max_mem_gb:.2f}GB"
                )
            else:
                print(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} eps={eps_by_step(steps):.3f} replay={len(replay)}")

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()