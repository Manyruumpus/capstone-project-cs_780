"""
run_live.py — Live viewer for OBELIX using a submission-style DQN agent.

This version is updated for an agent.py that:
- loads weights.pth internally,
- expects policy(obs, rng),
- maintains its own 4-frame stack,
- does greedy evaluation on CPU.

Usage:
  python run_live.py --difficulty 0
  python run_live.py --difficulty 2
  python run_live.py --difficulty 3
  python run_live.py --difficulty 0 --wall_obstacles
  python run_live.py --difficulty 0 --seed 42 --episodes 3
"""

import argparse
import importlib.util
import os
import sys
import numpy as np


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


def import_obelix(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"obelix.py not found: {path}")
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "OBELIX"):
        raise AttributeError("obelix.py does not expose OBELIX")
    return mod.OBELIX


def import_agent(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"agent.py not found: {path}")
    spec = importlib.util.spec_from_file_location("agent_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "policy"):
        raise AttributeError("agent.py must expose policy(obs, rng)")

    return mod


def maybe_preload_agent(agent_mod):
    if hasattr(agent_mod, "_load_once"):
        agent_mod._load_once()


def step_env(env, action: str, render: bool = True):
    out = env.step(action, render=render)

    if isinstance(out, tuple):
        if len(out) == 3:
            obs, reward, done = out
            info = {}
            return obs, reward, done, info
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, done, info

    raise ValueError("env.step(...) must return (obs, reward, done) or (obs, reward, done, info)")


def reset_env(env, seed: int):
    out = env.reset(seed=seed)
    if isinstance(out, tuple):
        return out[0]
    return out


def format_status(done: bool, cumulative_reward: float):
    if not done:
        return "running"
    return "SUCCESS ✓" if cumulative_reward > 0 else "timeout/fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", default="./obelix.py")
    ap.add_argument("--agent_py", default="./agent.py")
    ap.add_argument("--difficulty", type=int, default=0,
                    help="0=static, 2=blinking, 3=moving+blinking")
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--delay_ms", type=int, default=30,
                    help="milliseconds between frames (lower=faster)")
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--box_speed", type=int, default=2)
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    agent = import_agent(args.agent_py)
    maybe_preload_agent(agent)

    import cv2

    diff_label = {
        0: "Static box",
        2: "Blinking box",
        3: "Moving + Blinking box",
    }
    wall_label = " + Wall" if args.wall_obstacles else ""

    print(f"\n{'=' * 64}")
    print(f"OBELIX Live Eval — {diff_label.get(args.difficulty, str(args.difficulty))}{wall_label}")
    print(f"Episodes: {args.episodes} | Seed start: {args.seed} | Max steps: {args.max_steps}")
    print(f"Delay: {args.delay_ms} ms/frame | Press Q or ESC to stop early")
    print(f"Agent file: {os.path.abspath(args.agent_py)}")
    print(f"Env file:   {os.path.abspath(args.obelix_py)}")
    print(f"{'=' * 64}\n")

    total_rewards = []
    total_steps = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep

        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=ep_seed,
        )

        obs = np.asarray(reset_env(env, ep_seed), dtype=np.float32)
        cumulative_reward = 0.0
        step = 0
        quit_early = False

        print(f"Episode {ep + 1}/{args.episodes} | seed={ep_seed} | starting...")

        while True:
            try:
                action = agent.policy(obs, rng=env.rng)
            except TypeError:
                action = agent.policy(obs)

            if action not in ACTIONS:
                raise ValueError(f"Agent returned invalid action: {action}")

            obs, reward, done, info = step_env(env, action, render=True)
            obs = np.asarray(obs, dtype=np.float32)

            cumulative_reward += float(reward)
            step += 1

            key = cv2.waitKey(args.delay_ms) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                quit_early = True
                print("  [Q pressed — stopping early]")
                break

            if done:
                status = format_status(done, cumulative_reward)
                print(f"  done at step {step} | reward={cumulative_reward:.2f} | {status}")
                break

        total_rewards.append(cumulative_reward)
        total_steps.append(step)

        if hasattr(env, "close"):
            env.close()

        if quit_early:
            break

        cv2.waitKey(800)

    cv2.destroyAllWindows()

    if total_rewards:
        print(f"\n{'=' * 64}")
        print(f"Results over {len(total_rewards)} episode(s):")
        for i, (r, s) in enumerate(zip(total_rewards, total_steps), start=1):
            print(f"  Episode {i}: reward={r:.2f} | steps={s}")
        print(f"Mean reward: {np.mean(total_rewards):.2f}")
        print(f"Std reward:  {np.std(total_rewards):.2f}")
        print(f"Mean steps:  {np.mean(total_steps):.2f}")
        print(f"{'=' * 64}\n")
    else:
        print("\nNo episodes completed.\n")


if __name__ == "__main__":
    main()