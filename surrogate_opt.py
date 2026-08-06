# surrogate_opt.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Tuple
from dec import make_radar
from decoy_optimization import DecoyRSOptimizer, ObjectiveSpec, weight_bounds_from_ratio

# --- Configuration ---
N_MAX = 5
R_MIN, R_MAX = 100.0, 600.0
RADAR_XY = (0.0, 0.0)
DECISION_POINT = (0.0, 900.0, 200.0)

# --- 1. Define the Surrogate Neural Network ---
class UtilitySurrogate(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Utility is bounded between 0 and 1
        )

    def forward(self, x):
        return self.net(x)

def train_surrogate(df: pd.DataFrame) -> UtilitySurrogate:
    print("Training Surrogate Model...")
    X = torch.tensor(df.drop("utility", axis=1).values, dtype=torch.float32)
    y = torch.tensor(df["utility"].values, dtype=torch.float32).unsqueeze(1)
    
    model = UtilitySurrogate(input_dim=N_MAX * 3)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | MSE Loss: {loss.item():.4f}")
            
    return model

# --- 2. Optimize Inputs against the Surrogate (Gradient Ascent) ---
def optimize_configurations(model: UtilitySurrogate, n_candidates: int = 20) -> List[np.ndarray]:
    print("\nRunning Gradient Ascent to find optimal layouts...")
    model.eval()
    
    # Start with random noise configurations
    z = torch.rand((n_candidates, N_MAX * 3), requires_grad=True)
    
    # Scale initial R values to valid range roughly
    with torch.no_grad():
        for i in range(N_MAX):
            z[:, i*3+1] = z[:, i*3+1] * (R_MAX - R_MIN) + R_MIN
            z[:, i*3+2] = z[:, i*3+2] * (2 * np.pi)
            
    z.requires_grad_(True)
    optimizer = optim.Adam([z], lr=5.0) # High learning rate for input space
    
    steps = 150
    for _ in range(steps):
        optimizer.zero_grad()
        utilities = model(z)
        
        # We want to MAXIMIZE utility, so we MINIMIZE negative utility
        loss = -utilities.mean() 
        loss.backward()
        optimizer.step()
        
        # Soft-clamp values to physical reality during optimization
        with torch.no_grad():
            for i in range(N_MAX):
                z[:, i*3] = torch.clamp(z[:, i*3], 0.0, 1.0) # Active flags
                z[:, i*3+1] = torch.clamp(z[:, i*3+1], R_MIN, R_MAX) # Radius bounds
                z[:, i*3+2] = z[:, i*3+2] % (2 * np.pi) # Wrap angles
                
    # Extract the top mathematically proposed candidates
    optimized_layouts = z.detach().numpy()
    candidate_utilities = model(z).detach().numpy().flatten()
    
    # Sort by predicted utility descending
    sorted_idx = np.argsort(candidate_utilities)[::-1]
    best_layouts = optimized_layouts[sorted_idx]
    
    # Convert back to (x, y) coordinate arrays for DecoyRSOptimizer
    final_candidates = []
    for layout in best_layouts:
        pts = []
        for i in range(N_MAX):
            active, r, theta = layout[i*3], layout[i*3+1], layout[i*3+2]
            if active > 0.5: # Hard threshold the active flag
                x = RADAR_XY[0] + r * np.cos(theta)
                y = RADAR_XY[1] + r * np.sin(theta)
                pts.append([x, y])
        if len(pts) > 0:
            final_candidates.append(np.array(pts))
            
    return final_candidates

# --- 3. Verification using existing R&S machinery ---
if __name__ == "__main__":
    # 1. Load Data and Train
    df = pd.read_csv("decoy_dataset.csv")
    surrogate = train_surrogate(df)
    
    # 2. Extract Top 10 novel configurations via Surrogate
    proposed_candidates = optimize_configurations(surrogate, n_candidates=10)
    
    # 3. Verify statistically via OCBA
    print("\nVerifying Neural Network proposals via DecoyRSOptimizer...")
    radar = make_radar(RADAR_XY[0], RADAR_XY[1])
    
    optimizer = DecoyRSOptimizer(
        radar=radar,
        decision_point=DECISION_POINT,
        candidates=proposed_candidates,
        obj1=ObjectiveSpec(x_star=0, x_best=1000, rl=200, ru=800),
        obj2=ObjectiveSpec(x_star=0, x_best=1000, rl=200, ru=800),
        weight_bounds=weight_bounds_from_ratio(1, 4),
        seed=42
    )

    results = optimizer.run(n0=10, total_budget=1000, round_increment=100, verbose=True)
    
    print("\n=== FINAL STATISTICALLY VERIFIED RESULTS ===")
    best_candidate_idx = int(results.iloc[0]["candidate"])
    print(results.to_string(index=False))
    
    print(f"\nBest Verified Decoy Coordinates (Candidate {best_candidate_idx}):")
    print(proposed_candidates[best_candidate_idx])