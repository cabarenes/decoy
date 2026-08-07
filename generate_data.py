
import numpy as np
import pandas as pd
from typing import List, Tuple
from dec import make_radar, make_decoy, run_single_trial
from decoy_optimization import single_attribute_utility, ObjectiveSpec

# --- Configuration ---
N_MAX = 5                     # Maximum number of decoys
R_MIN, R_MAX = 100.0, 600.0   # Deployment annulus limits
N_SAMPLES = 5000              # Number of configurations in the dataset
MC_REPS = 20                  # Low replication count for noisy but fast utility estimates
RADAR_XY = (0.0, 0.0)
DECISION_POINT = (0.0, 900.0, 200.0)

# Weights and Objectives (matching your existing setup)
W1, W2 = 0.2, 0.8  
OBJ1 = ObjectiveSpec(x_star=0, x_best=1000, rl=200, ru=800)
OBJ2 = ObjectiveSpec(x_star=0, x_best=1000, rl=200, ru=800)

def generate_random_config(rng: np.random.Generator) -> np.ndarray:
    """Generates a flat vector: [active, r, theta] for each of the N_MAX slots."""
    config = []
    for _ in range(N_MAX):
        # 70% chance a decoy slot is active to ensure variety in n_decoys
        active = 1.0 if rng.uniform() > 0.3 else 0.0
        r = rng.uniform(R_MIN, R_MAX)
        theta = rng.uniform(0.0, 2 * np.pi)
        config.extend([active, r, theta])
    return np.array(config)

def evaluate_config(config: np.ndarray, rng: np.random.Generator, radar, seed_pool: List[np.random.SeedSequence]) -> float:
    """Evaluates the configuration vector using a fixed number of replications."""
    # Decode configuration
    decoys = []
    for i in range(N_MAX):
        idx = i * 3
        active, r, theta = config[idx], config[idx+1], config[idx+2]
        if active > 0.5:
            x = RADAR_XY[0] + r * np.cos(theta)
            y = RADAR_XY[1] + r * np.sin(theta)
            decoys.append(make_decoy(x, y, f"d{i}"))
    
    # If no decoys were active, utility is 0 (or baseline)
    if not decoys:
        return 0.0

    utilities = []
    for rep in range(MC_REPS):
        rep_rng = np.random.default_rng(seed_pool[rep])
        res = run_single_trial(radar, decoys, DECISION_POINT, rep_rng)
        det = res["detonation"]
        
        if det is None:
            continue
            
        x_det, y_det = det
        x1_val = float(np.hypot(x_det - radar.x, y_det - radar.y))
        x2_val = float(min(np.hypot(x_det - d.x, y_det - d.y) for d in decoys))
        
        u1 = single_attribute_utility(x1_val, OBJ1.x_star, OBJ1.x_best, OBJ1.rl, OBJ1.ru)
        u2 = single_attribute_utility(x2_val, OBJ2.x_star, OBJ2.x_best, OBJ2.rl, OBJ2.ru)
        utilities.append(W1 * u1 + W2 * u2)

    return float(np.mean(utilities)) if utilities else 0.0

if __name__ == "__main__":
    print(f"Generating dataset of {N_SAMPLES} configurations...")
    rng = np.random.default_rng(42)
    master_seed = np.random.SeedSequence(42)
    crn_pool = master_seed.spawn(MC_REPS) # Common Random Numbers for stability
    radar = make_radar(RADAR_XY[0], RADAR_XY[1])
    
    dataset = []
    for i in range(N_SAMPLES):
        if i % 500 == 0:
            print(f"  Processed {i}/{N_SAMPLES} samples...")
        cfg = generate_random_config(rng)
        utility = evaluate_config(cfg, rng, radar, crn_pool)
        
        row = cfg.tolist() + [utility]
        dataset.append(row)
        
    # Columns: a1, r1, t1, a2, r2, t2... utility
    cols = []
    for i in range(N_MAX):
        cols.extend([f"active_{i}", f"r_{i}", f"theta_{i}"])
    cols.append("utility")
    
    df = pd.DataFrame(dataset, columns=cols)
    df.to_csv("decoy_dataset.csv", index=False)
    print("Dataset saved to decoy_dataset.csv.")

https://tn13b0v6.r.eu-central-1.awstrack.me/L0/https:%2F%2Fimplicit.harvard.edu%2Fimplicit%2Fuser%2Fagg%2Fblindspot%2Findexrk.htm/1/010701985a9e29a1-8574f6fc-388a-4050-9f1e-c4974fd20a8c-000000/_7g73elQHZ8qn8F0bt2GMh2c3fs=217
