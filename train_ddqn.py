"""Offline trainer: Double DQN + replay buffer (GPU-aware) for OBELIX.

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
  python train_ddqn.py --obelix_py ./obelix.py --out ./submission_versions/run1/weights.pth --episodes 2000 --difficulty 0 --wall_obstacles

ALGORITHM: DOUBLE DEEP Q-NETWORK (DDQN)

This version keeps the same DDQN learning logic and adds only training hygiene:
- auto-create output directory
- save periodic checkpoints
- write episode CSV logs
- save an interrupt checkpoint on Ctrl+C
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class DQN(nn.Module):
    def __init__(self, in_dim: int = 18, n_actions: int = 5, hidden_dim: int = 64):
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

    def __len__(self):
        return len(self.buf)


def import_obelix(obelix_py: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load OBELIX env from {obelix_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def ensure_parent_dir(path_str: str) -> Path:
    path = Path(path_str)
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def maybe_close_env(env) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


def save_weights(model: nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--init_weights", type=str, default=None, help="optional checkpoint to initialize model from")
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

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="save checkpoint every N episodes; set 0 to disable periodic checkpoints",
    )
    ap.add_argument(
        "--log_csv",
        type=str,
        default=None,
        help="optional CSV log path; default is next to --out as <stem>_train_log.csv",
    )

    args = ap.parse_args()

    out_path = ensure_parent_dir(args.out)
    if args.log_csv is None:
        log_csv_path = out_path.with_name(f"{out_path.stem}_train_log.csv")
    else:
        log_csv_path = ensure_parent_dir(args.log_csv)

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

    print(f"Using device: {device}", flush=True)
    print(f"Final output: {out_path}", flush=True)
    print(f"CSV log: {log_csv_path}", flush=True)

    OBELIX = import_obelix(args.obelix_py)

    q = DQN(hidden_dim=args.hidden_dim).to(device)
    tgt = DQN(hidden_dim=args.hidden_dim).to(device)

    if args.init_weights is not None:
        state = torch.load(args.init_weights, map_location=device)
        q.load_state_dict(state)
        print(f"Initialized from: {args.init_weights}", flush=True)

    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    steps = 0
    train_start = time.time()

    def eps_by_step(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    def append_log(row: dict) -> None:
        file_exists = log_csv_path.exists()
        with log_csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "episode_seed",
                    "return",
                    "epsilon",
                    "replay_size",
                    "total_steps",
                    "episode_updates",
                    "mean_loss",
                    "elapsed_sec",
                    "difficulty",
                    "wall_obstacles",
                    "max_steps",
                    "box_speed",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    try:
        for ep in range(args.episodes):
            episode_seed = args.seed + ep
            env = OBELIX(
                scaling_factor=args.scaling_factor,
                arena_size=args.arena_size,
                max_steps=args.max_steps,
                wall_obstacles=args.wall_obstacles,
                difficulty=args.difficulty,
                box_speed=args.box_speed,
                seed=episode_seed,
            )

            s = np.asarray(env.reset(seed=episode_seed), dtype=np.float32)
            ep_ret = 0.0
            ep_loss_sum = 0.0
            ep_updates = 0

            try:
                last_s = s 
                for _ in range(args.max_steps):
                    current_obs = s if np.any(s[:17]) else last_s
                    eps = eps_by_step(steps)

                    if np.random.rand() < eps:
                        a = np.random.randint(len(ACTIONS))
                    else:
                        with torch.no_grad():
                            s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                            qs = q(s_t).squeeze(0).detach().cpu().numpy()
                        a = int(np.argmax(qs))
                        if np.any(s2[:17]): last_s = s2

                    s2, r, done = env.step(ACTIONS[a], render=False)
                    s2 = np.asarray(s2, dtype=np.float32)

                    ep_ret += float(r)
                    replay.add(Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done)))
                    s = s2
                    steps += 1

                    if len(replay) >= max(args.warmup, args.batch):
                        for _ in range(args.updates_per_step):
                            sb, ab, rb, s2b, db = replay.sample(args.batch)

                            sb_t = torch.as_tensor(sb, dtype=torch.float32, device=device)
                            ab_t = torch.as_tensor(ab, dtype=torch.int64, device=device)
                            rb_t = torch.as_tensor(rb, dtype=torch.float32, device=device)
                            s2b_t = torch.as_tensor(s2b, dtype=torch.float32, device=device)
                            db_t = torch.as_tensor(db, dtype=torch.float32, device=device)

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

                            ep_loss_sum += float(loss.item())
                            ep_updates += 1

                        if steps % args.target_sync == 0:
                            tgt.load_state_dict(q.state_dict())

                    if done:
                        break
            finally:
                maybe_close_env(env)

            mean_loss = (ep_loss_sum / ep_updates) if ep_updates > 0 else float("nan")
            elapsed_sec = time.time() - train_start
            eps_now = eps_by_step(steps)

            append_log(
                {
                    "episode": ep + 1,
                    "episode_seed": episode_seed,
                    "return": f"{ep_ret:.6f}",
                    "epsilon": f"{eps_now:.6f}",
                    "replay_size": len(replay),
                    "total_steps": steps,
                    "episode_updates": ep_updates,
                    "mean_loss": f"{mean_loss:.8f}" if ep_updates > 0 else "",
                    "elapsed_sec": f"{elapsed_sec:.2f}",
                    "difficulty": args.difficulty,
                    "wall_obstacles": int(args.wall_obstacles),
                    "max_steps": args.max_steps,
                    "box_speed": args.box_speed,
                }
            )

            if (ep + 1) % args.log_every == 0:
                if device.type == "cuda":
                    mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    print(
                        f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} "
                        f"eps={eps_now:.3f} replay={len(replay)} updates={ep_updates} "
                        f"mean_loss={mean_loss:.6f} gpu_mem={mem_gb:.2f}GB peak={peak_mem_gb:.2f}GB",
                        flush=True,
                    )
                else:
                    print(
                        f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} "
                        f"eps={eps_now:.3f} replay={len(replay)} updates={ep_updates} "
                        f"mean_loss={mean_loss:.6f}",
                        flush=True,
                    )

            if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                ckpt_path = out_path.with_name(f"{out_path.stem}_ep{ep+1:04d}.pth")
                save_weights(q, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}", flush=True)

    except KeyboardInterrupt:
        interrupted_path = out_path.with_name(f"{out_path.stem}_interrupted.pth")
        save_weights(q, interrupted_path)
        print(f"\nInterrupted. Saved checkpoint: {interrupted_path}", flush=True)
        raise

    save_weights(q, out_path)
    print("Saved:", out_path, flush=True)


if __name__ == "__main__":
    main()