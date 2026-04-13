"""
run_live.py — Watch your trained OBELIX agent live with cv2 rendering.
Usage:
  python run_live.py --difficulty 0               # static box
  python run_live.py --difficulty 2               # blinking box
  python run_live.py --difficulty 3               # moving + blinking box
  python run_live.py --difficulty 0 --wall_obstacles
  python run_live.py --difficulty 0 --seed 42 --episodes 3
"""

import argparse
import importlib.util
import sys
import os
import time
import numpy as np

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# ── load obelix env ──────────────────────────────────────────────────────────
def import_obelix(path: str):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

# ── load your agent ──────────────────────────────────────────────────────────
def import_agent(path: str):
    spec = importlib.util.spec_from_file_location("agent_mod", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod          # exposes mod.policy(obs, rng)

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",   default="./obelix.py")
    ap.add_argument("--agent_py",    default="./agent.py")
    ap.add_argument("--difficulty",  type=int,   default=0,
                    help="0=static, 2=blinking, 3=moving+blinking")
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--episodes",    type=int,   default=5)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--max_steps",   type=int,   default=1000)
    ap.add_argument("--delay_ms",    type=int,   default=30,
                    help="milliseconds between frames (lower = faster)")
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size",     type=int, default=500)
    ap.add_argument("--box_speed",      type=int, default=2)
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    agent  = import_agent(args.agent_py)

    diff_label = {0: "Static box", 2: "Blinking box", 3: "Moving+Blinking"}
    wall_label = " + Wall" if args.wall_obstacles else ""
    print(f"\n{'='*55}")
    print(f"  OBELIX Live Eval — {diff_label.get(args.difficulty, str(args.difficulty))}{wall_label}")
    print(f"  Episodes: {args.episodes}  |  Seed start: {args.seed}  |  Max steps: {args.max_steps}")
    print(f"  Delay: {args.delay_ms}ms/frame   (press Q in window to quit early)")
    print(f"{'='*55}\n")

    total_rewards = []

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
        obs = env.reset(seed=ep_seed)
        cumulative_reward = 0.0
        step = 0
        quit_early = False

        print(f"  Episode {ep+1}/{args.episodes} | seed={ep_seed} | starting...")

        while True:
            # get action from your agent (passes rng for episode-reset detection)
            action = agent.policy(obs, rng=env.rng)

            # step with render=True so cv2.imshow fires inside obelix.py
            obs, reward, done = env.step(action, render=True)
            cumulative_reward += reward
            step += 1

            # waitKey controls frame speed; returns -1 if no key pressed
            # Q or ESC quits early
            import cv2
            key = cv2.waitKey(args.delay_ms) & 0xFF
            if key in (ord('q'), ord('Q'), 27):   # Q or ESC
                quit_early = True
                print("  [Q pressed — stopping early]")
                break

            if done:
                status = "SUCCESS ✓" if cumulative_reward > 0 else "timeout/fail"
                print(f"  Episode {ep+1} done at step {step} | reward={cumulative_reward:.1f} | {status}")
                break

        total_rewards.append(cumulative_reward)

        if quit_early:
            break

        # short pause between episodes so you can see the final frame
        import cv2
        cv2.waitKey(800)

    import cv2
    cv2.destroyAllWindows()

    print(f"\n{'='*55}")
    print(f"  Results over {len(total_rewards)} episode(s):")
    for i, r in enumerate(total_rewards):
        print(f"    Episode {i+1}: {r:.1f}")
    print(f"  Mean: {np.mean(total_rewards):.2f}  |  Std: {np.std(total_rewards):.2f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()