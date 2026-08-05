
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Sequence
from dataclasses import dataclass

from dec import Transmitter, make_radar, make_decoy, run_single_trial


# ==========================================================================
# 1. Candidate generation
# ==========================================================================

def _min_pairwise_dist(points: np.ndarray) -> float:
    if len(points) < 2:
        return np.inf
    d = np.inf
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = min(d, np.hypot(*(points[i] - points[j])))
    return d


def generate_candidates(n_decoys: int, radar_xy: Tuple[float, float],
                         r_min: float, r_max: float, K: int,
                         rng: np.random.Generator,
                         min_sep: Optional[float] = None,
                         max_tries: int = 200) -> List[np.ndarray]:
    """
    Generate K candidate decoy layouts. Each decoy position is sampled in
    polar coordinates (radius in [r_min, r_max], angle in [0, 2*pi)) around
    the radar, which keeps decoys within a plausible deployment annulus
    (not on top of the radar, not arbitrarily far away). If `min_sep` is
    given, layouts are rejection-sampled to keep decoys at least that far
    apart from one another.

    Returns a list of (n_decoys, 2) arrays of (x, y) coordinates.
    """
    candidates = []
    for _ in range(K):
        pts = None
        for _attempt in range(max_tries):
            radii = rng.uniform(r_min, r_max, size=n_decoys)
            angles = rng.uniform(0.0, 2 * np.pi, size=n_decoys)
            xs = radar_xy[0] + radii * np.cos(angles)
            ys = radar_xy[1] + radii * np.sin(angles)
            pts = np.column_stack([xs, ys])
            if min_sep is None or _min_pairwise_dist(pts) >= min_sep:
                break
        candidates.append(pts)
    return candidates


# 2. Additive MAU: single-attribute utilities + feasible weight region


def single_attribute_utility(x, x_star: float, x_best: float,
                              rl: float, ru: float) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    mid = 0.5 * (rl + ru)
    k = np.log(19.0) * 2.0 / max(ru - rl, 1e-9)

    def sig(v):
        return 1.0 / (1.0 + np.exp(-k * (v - mid)))

    raw = sig(x)
    raw_lo = sig(x_star)
    raw_hi = sig(x_best)
    u = (raw - raw_lo) / (raw_hi - raw_lo)
    return np.clip(u, 0.0, 1.0)


def weight_bounds_from_ratio(ratio_lo: float, ratio_hi: float) -> Tuple[float, float]:
    """
    Convert a DM preference statement on the ratio w1/w2 (radar-protection
    importance relative to decoy-protection importance), e.g. "radar
    protection is 1x to 4x as important as decoy protection", into bounds
    on w1 for the two-objective case (w1 + w2 = 1).
    """
    w1_lo = ratio_lo / (1.0 + ratio_lo)
    w1_hi = ratio_hi / (1.0 + ratio_hi)
    return w1_lo, w1_hi


# 3. R&S optimizer with OCBA budget allocation

@dataclass
class ObjectiveSpec:
    
    x_star: float   # least desired value (0 by construction)
    x_best: float   # most desired value ("beyond this, no impact")
    rl: float       # lower bound of the lethal-radius transition zone
    ru: float       # upper bound of the lethal-radius transition zone


class DecoyRSOptimizer:

    def __init__(self,
                 radar: Transmitter,
                 decision_point: Tuple[float, float, float],
                 candidates: List[np.ndarray],
                 obj1: ObjectiveSpec,
                 obj2: ObjectiveSpec,
                 weight_bounds: Tuple[float, float],
                 nominal_weights: Optional[Tuple[float, float]] = None,
                 seed: int = 0):
        self.radar = radar
        self.decision_point = decision_point
        self.candidates = candidates
        self.K = len(candidates)
        self.obj1, self.obj2 = obj1, obj2
        self.w1_lo, self.w1_hi = weight_bounds
        self.nominal_weights = nominal_weights or (
            0.5 * (self.w1_lo + self.w1_hi), 1.0 - 0.5 * (self.w1_lo + self.w1_hi)
        )

        self._ss = np.random.SeedSequence(seed)
        self._seed_pool: List[np.random.SeedSequence] = []

        self.n_reps = np.zeros(self.K, dtype=int)
        self.X1_data: List[List[float]] = [[] for _ in range(self.K)]
        self.X2_data: List[List[float]] = [[] for _ in range(self.K)]
        self.U_data: List[List[float]] = [[] for _ in range(self.K)]

    # ---- common-random-number seed pool -------------------------------

    def _ensure_seeds(self, n_needed: int) -> None:
        if n_needed <= len(self._seed_pool):
            return
        additional = n_needed - len(self._seed_pool)
        self._seed_pool.extend(self._ss.spawn(additional))

    # ---- simulate a candidate up to `target_n` replications ------------

    def _run_up_to(self, k: int, target_n: int) -> None:
        current = self.n_reps[k]
        if target_n <= current:
            return
        self._ensure_seeds(target_n)
        decoy_xy = self.candidates[k]
        for r in range(current, target_n):
            rng = np.random.default_rng(self._seed_pool[r])
            decoys = [make_decoy(x, y, f"d{i}") for i, (x, y) in enumerate(decoy_xy)]
            res = run_single_trial(self.radar, decoys, self.decision_point, rng)
            det = res["detonation"]
            if det is None:
                continue  
            x_det, y_det = det
            X1 = float(np.hypot(x_det - self.radar.x, y_det - self.radar.y))
            X2 = float(min(np.hypot(x_det - dx, y_det - dy) for dx, dy in decoy_xy))
            u1 = single_attribute_utility(X1, self.obj1.x_star, self.obj1.x_best,
                                           self.obj1.rl, self.obj1.ru)
            u2 = single_attribute_utility(X2, self.obj2.x_star, self.obj2.x_best,
                                           self.obj2.rl, self.obj2.ru)
            U = float(self.nominal_weights[0] * u1 + self.nominal_weights[1] * u2)
            self.X1_data[k].append(X1)
            self.X2_data[k].append(X2)
            self.U_data[k].append(U)
        self.n_reps[k] = target_n

    def _stats(self, k: int) -> Tuple[float, float]:
        arr = np.asarray(self.U_data[k])
        if len(arr) == 0:
            return 0.0, 1.0
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 1.0
        return mean, max(std, 1e-6)

    # ---- OCBA allocation round ------------------------------------------

    def _ocba_round(self, increment: int) -> None:
  
        means = np.array([self._stats(k)[0] for k in range(self.K)])
        stds = np.array([self._stats(k)[1] for k in range(self.K)])
        b = int(np.argmax(means))

        ratios = np.zeros(self.K)
        for i in range(self.K):
            if i == b:
                continue
            delta_i = max(means[b] - means[i], 1e-9)
            ratios[i] = (stds[i] / delta_i) ** 2

        nonbest = [i for i in range(self.K) if i != b]
        if ratios[nonbest].sum() <= 0:
            props = np.ones(self.K) / self.K
        else:
            inner = np.sqrt(sum((ratios[i] / stds[i]) ** 2 for i in nonbest))
            raw = ratios.copy()
            raw[b] = stds[b] * inner
            props = raw / raw.sum()

        target_total = int(self.n_reps.sum()) + increment
        target_alloc = np.maximum(self.n_reps, np.round(props * target_total).astype(int))
        for k in range(self.K):
            self._run_up_to(k, int(target_alloc[k]))

    # ---- public API -------------------------------------------------------

    def run(self, n0: int = 20, total_budget: int = 3000,
            round_increment: int = 150, verbose: bool = True) -> pd.DataFrame:
        
        for k in range(self.K):
            self._run_up_to(k, n0)

        round_no = 0
        while self.n_reps.sum() < total_budget:
            round_no += 1
            remaining = total_budget - int(self.n_reps.sum())
            inc = min(round_increment, remaining)
            self._ocba_round(inc)
            if verbose:
                best = int(np.argmax([self._stats(k)[0] for k in range(self.K)]))
                print(f"  round {round_no:3d} | total reps = {int(self.n_reps.sum()):5d} "
                      f"| current best candidate = {best} "
                      f"(mean U = {self._stats(best)[0]:.4f})")
        return self.summary()

    def summary(self) -> pd.DataFrame:
        rows = []
        for k in range(self.K):
            mean_u, std_u = self._stats(k)
            X1 = np.asarray(self.X1_data[k])
            X2 = np.asarray(self.X2_data[k])
            rows.append({
                "candidate": k,
                "n_reps": int(self.n_reps[k]),
                "mean_dist_radar": X1.mean() if len(X1) else np.nan,
                "mean_dist_decoy": X2.mean() if len(X2) else np.nan,
                "mean_utility": mean_u,
                "se_utility": std_u / np.sqrt(max(self.n_reps[k], 1)),
            })
        df = pd.DataFrame(rows).sort_values("mean_utility", ascending=False).reset_index(drop=True)
        return df

    # ---- SMAA-style rank acceptability under incomplete weights -----------

    def rank_acceptability(self, n_draws: int = 4000, seed: int = 12345) -> pd.DataFrame:

        rng = np.random.default_rng(seed)
        win_counts = np.zeros(self.K)
        for _ in range(n_draws):
            w1 = rng.uniform(self.w1_lo, self.w1_hi)
            w2 = 1.0 - w1
            utils = np.full(self.K, -np.inf)
            for k in range(self.K):
                X1 = self.X1_data[k]
                X2 = self.X2_data[k]
                if not X1:
                    continue
                idx = rng.integers(0, len(X1))
                u1 = single_attribute_utility(X1[idx], self.obj1.x_star, self.obj1.x_best,
                                               self.obj1.rl, self.obj1.ru)
                u2 = single_attribute_utility(X2[idx], self.obj2.x_star, self.obj2.x_best,
                                               self.obj2.rl, self.obj2.ru)
                utils[k] = w1 * u1 + w2 * u2
            win_counts[int(np.argmax(utils))] += 1
        rai = win_counts / n_draws
        df = pd.DataFrame({"candidate": np.arange(self.K), "rank_acceptability_index": rai})
        return df.sort_values("rank_acceptability_index", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    radar = make_radar(0, 0)
    decision_point = (0, 900, 200)  # x, y, altitude

    candidates = generate_candidates(
        n_decoys=3,
        radar_xy=(radar.x, radar.y),
        r_min=100,
        r_max=600,
        K=10,
        rng=rng,
        min_sep=50,
    )

    optimizer = DecoyRSOptimizer(
        radar=radar,
        decision_point=decision_point,
        candidates=candidates,
        obj1=ObjectiveSpec(x_star=0, x_best=1_000, rl=200, ru=800),
        obj2=ObjectiveSpec(x_star=0, x_best=1_000, rl=200, ru=800),
        weight_bounds=weight_bounds_from_ratio(1, 4),
        seed=42,
    )

    results = optimizer.run(
        n0=5,
        total_budget=100,
        round_increment=25,
        verbose=True,
    )
    best = results.iloc[0]  # results are sorted by mean_utility, best first
    best_index = int(best["candidate"])

    print("\nBest candidate:")
    print(best.to_string())

    print("\nDecoy coordinates for the best candidate:")
    print(optimizer.candidates[best_index])
    print("\nOptimization results:")
    print(results.to_string(index=False))

    print("\nRank acceptability:")
    print(optimizer.rank_acceptability(n_draws=1_000).to_string(index=False))