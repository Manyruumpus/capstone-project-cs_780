"""
Offline trainer shell for tabular RL experiments on OBELIX.

Step 1 only:
- keeps trainer structure
- removes DDQN / neural-network pieces
- preserves env creation, seeding, logging, and checkpoint path handling

Main Q-learning logic is intentionally left as TODO placeholders.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

def obs_to_state(obs) -> tuple[int, ...]:
    obs_arr = np.asarray(obs, dtype=np.int8).reshape(-1)
    if obs_arr.shape[0] != 18:
        raise ValueError(f"Expected observation of length 18, got shape {obs_arr.shape}")
    return tuple(int(x) for x in obs_arr)

def ensure_state_in_q_table(q_table: dict, state: tuple[int, ...]) -> None:
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS), dtype=np.float32)

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


def save_q_table(q_table: dict, path: Path) -> None:
    serializable_q_table = {
        ",".join(map(str, state)): values.tolist()
        for state, values in q_table.items()
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable_q_table, f, indent=2)


def eps_by_step(step: int, eps_start: float, eps_end: float, eps_decay_steps: int) -> float:
    if step >= eps_decay_steps:
        return eps_end
    frac = step / eps_decay_steps
    return eps_start + frac * (eps_end - eps_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="q_table.json")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=200000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)

    ap.add_argument("--seed", type=int, default=0)
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
    alpha = args.alpha
    gamma = args.gamma

    out_path = ensure_parent_dir(args.out)
    if args.log_csv is None:
        log_csv_path = out_path.with_name(f"{out_path.stem}_train_log.csv")
    else:
        log_csv_path = ensure_parent_dir(args.log_csv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Final output: {out_path}", flush=True)
    print(f"CSV log: {log_csv_path}", flush=True)

    OBELIX = import_obelix(args.obelix_py)

    # Step 1 placeholder: actual Q-table structure will be defined in Step 3.
    q_table = {}

    total_steps = 0
    train_start = time.time()

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
                    "total_steps",
                    "visited_states",
                    "elapsed_sec",
                    "difficulty",
                    "wall_obstacles",
                    "max_steps",
                    "box_speed",
                    "zero_state_steps",
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

            ep_ret = 0.0
            zero_state_steps = 0


            try:
                obs = env.reset(seed=episode_seed)
                obs = np.asarray(obs, dtype=np.int8)
                state = obs_to_state(obs)
                if state == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0):
                    zero_state_steps += 1

                ensure_state_in_q_table(q_table, state)

                for _ in range(args.max_steps):
                    eps = eps_by_step(total_steps, args.eps_start, args.eps_end, args.eps_decay_steps)

                    if np.random.rand() < eps:
                        action_idx = np.random.randint(len(ACTIONS))
                    else:
                        action_idx = int(np.argmax(q_table[state]))

                    next_obs, reward, done = env.step(ACTIONS[action_idx], render=False)
                    next_obs = np.asarray(next_obs, dtype=np.int8)
                    next_state = obs_to_state(next_obs)
                    ensure_state_in_q_table(q_table, next_state)
                    
                    # if reward <= -100:
                    #     print(
                    #         f"step={total_steps} action={ACTIONS[action_idx]} reward={reward} "
                    #         f"done={done} state={state} next_state={next_state}",
                    #         flush=True,
                    #     )


                    ep_ret += float(reward)
                    total_steps += 1

                    old_val   = q_table[state][action_idx]
                    bootstrap = max(q_table[next_state])
                    target    = reward if done else reward + gamma * bootstrap
                    q_table[state][action_idx] = old_val + alpha * (target - old_val)

                    obs = next_obs
                    state = next_state

                    if done:
                        break
            finally:
                maybe_close_env(env)        


            elapsed_sec = time.time() - train_start

            append_log(
                {
                    "episode": ep + 1,
                    "episode_seed": episode_seed,
                    "return": f"{ep_ret:.6f}",
                    "epsilon": f"{eps_by_step(total_steps, args.eps_start, args.eps_end, args.eps_decay_steps):.6f}",
                    "total_steps": total_steps,
                    "visited_states": len(q_table),
                    "elapsed_sec": f"{elapsed_sec:.2f}",
                    "difficulty": args.difficulty,
                    "wall_obstacles": int(args.wall_obstacles),
                    "max_steps": args.max_steps,
                    "box_speed": args.box_speed,
                    "zero_state_steps": zero_state_steps,

                }
            )

            if (ep + 1) % args.log_every == 0:
                eps_now = eps_by_step(total_steps, args.eps_start, args.eps_end, args.eps_decay_steps)
                print(
                    f"Episode {ep+1}/{args.episodes} "
                    f"return={ep_ret:.1f} eps={eps_now:.3f} "
                    f"states={len(q_table)} total_steps={total_steps}",
                    flush=True,
                )


            if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                ckpt_path = out_path.with_name(f"{out_path.stem}_ep{ep+1:04d}.json")
                save_q_table(q_table, ckpt_path)

                print(f"Checkpoint saved: {ckpt_path}", flush=True)

    except KeyboardInterrupt:
        interrupted_path = out_path.with_name(f"{out_path.stem}_interrupted.json")
        save_q_table(q_table, interrupted_path)
        print(f"\nInterrupted. Saved checkpoint: {interrupted_path}", flush=True)
        raise

    save_q_table(q_table, out_path)
    print("Saved:", out_path, flush=True)


if __name__ == "__main__":
    main()
