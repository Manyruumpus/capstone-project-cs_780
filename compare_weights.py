import os
import csv
import torch
import numpy as np
import shutil
from pathlib import Path
from agent import DQN, ACTIONS, policy  # Direct import is faster
from obelix import OBELIX

# --- Configuration ---
BASE_DIR = Path(os.getcwd())
OUTPUT_CSV = "phase3_only_results2.csv"  # New file for Phase 3
WEIGHTS_PTH = "weights.pth"

def get_processed():
    processed = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["Checkpoint"])
    return processed

def evaluate_level_3(pth_path):
    """
    Directly runs the Level 3 evaluation with the exact parameters 
    from evaluate_on_codabench.py.
    """
    # Official Level 3 settings
    runs = 10
    base_seed = 0
    scaling_factor = 5
    arena_size = 500
    max_steps = 1000
    wall_obstacles = True
    difficulty = 3
    box_speed = 2

    # Force the agent to load THIS specific checkpoint
    # We clear the global model cache from agent.py if it exists
    import agent
    agent._model = None 
    shutil.copy(pth_path, BASE_DIR / "weights.pth") 

    scores = []
    env = OBELIX(
        scaling_factor=scaling_factor, arena_size=arena_size,
        max_steps=max_steps, wall_obstacles=wall_obstacles,
        difficulty=difficulty, box_speed=box_speed, seed=base_seed
    )

    for i in range(runs):
        seed = base_seed + i
        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        total = 0.0
        done = False
        while not done:
            action = policy(obs, rng)
            obs, reward, done = env.step(action, render=False)
            total += float(reward)
        scores.append(total)
    
    return np.mean(scores), np.std(scores)

def main():
    pth_files = list(BASE_DIR.rglob("*.pth"))
    # Filter out the generic weights.pth in root
    pth_files = [f for f in pth_files if not (f.name == "weights.pth" and f.parent == BASE_DIR)]
    
    already_done = get_processed()
    to_run = [f for f in pth_files if f.name not in already_done]

    print(f"Phase 3 Sweep: {len(to_run)} files remaining.")

    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Checkpoint", "Folder", "Mean_L3", "Std_L3"])
        if not file_exists:
            writer.writeheader()

        for i, pth in enumerate(to_run):
            print(f"[{i+1}/{len(to_run)}] Testing Level 3: {pth.name}...", end=" ", flush=True)
            try:
                mean_l3, std_l3 = evaluate_level_3(pth)
                writer.writerow({
                    "Checkpoint": pth.name,
                    "Folder": str(pth.parent.relative_to(BASE_DIR)),
                    "Mean_L3": f"{mean_l3:.3f}",
                    "Std_L3": f"{std_l3:.3f}"
                })
                csvfile.flush()
                print(f"Score: {mean_l3:.2f}")
            except Exception as e:
                print(f"FAILED: {e}")

    # Restore the original weights.pth if you had one
    print(f"\nPhase 3 sweep complete. Results in {OUTPUT_CSV}")

if __name__ == "__main__":
    main()