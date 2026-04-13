import os
import csv
import torch
import numpy as np
import shutil
from pathlib import Path
from agent import DQN, ACTIONS, policy 
from obelix import OBELIX

# --- Configuration ---
BASE_DIR = Path(os.getcwd())
OUTPUT_CSV = "level3_wall_only_results.csv"
WEIGHTS_PTH = "weights.pth"

# Final Phase Parameters
EVAL_CONFIG = {
    "runs": 10,
    "base_seed": 0,
    "scaling_factor": 5,
    "arena_size": 500,
    "max_steps": 1000, 
    "box_speed": 2
}

def get_processed():
    processed = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["Checkpoint"])
    return processed

def evaluate_l3_wall(pth_path):
    """
    Evaluates only Level 3 with Wall Obstacles.
    """
    import agent
    agent._model = None # Force reload of the model for every file
    shutil.copy(pth_path, BASE_DIR / "weights.pth") 

    scores = []
    # Difficulty 3 + Wall Obstacles = True
    env = OBELIX(
        scaling_factor=EVAL_CONFIG["scaling_factor"],
        arena_size=EVAL_CONFIG["arena_size"],
        max_steps=EVAL_CONFIG["max_steps"],
        wall_obstacles=True,
        difficulty=3,
        box_speed=EVAL_CONFIG["box_speed"],
        seed=EVAL_CONFIG["base_seed"]
    )

    for i in range(EVAL_CONFIG["runs"]):
        seed = EVAL_CONFIG["base_seed"] + i
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
    # Filter out generic root files
    pth_files = [f for f in pth_files if not (f.name == "weights.pth" and f.parent == BASE_DIR)]
    
    already_done = get_processed()
    to_run = [f for f in pth_files if f.name not in already_done]

    print(f"Targeted Wall Evaluation: {len(to_run)} files remaining.")

    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Checkpoint", "Folder", "Mean_L3_Wall", "Std_L3_Wall"])
        if not file_exists:
            writer.writeheader()

        for i, pth in enumerate(to_run):
            print(f"[{i+1}/{len(to_run)}] Testing L3 Wall: {pth.name}...", end=" ", flush=True)
            try:
                mean, std = evaluate_l3_wall(pth)
                writer.writerow({
                    "Checkpoint": pth.name,
                    "Folder": str(pth.parent.relative_to(BASE_DIR)),
                    "Mean_L3_Wall": f"{mean:.3f}",
                    "Std_L3_Wall": f"{std:.3f}"
                })
                csvfile.flush()
                print(f"Score: {mean:.2f}")
            except Exception as e:
                print(f"FAILED: {e}")

    print(f"\nEvaluation complete. Check results in {OUTPUT_CSV}")

if __name__ == "__main__":
    main()