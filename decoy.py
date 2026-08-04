from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

#constants

WAVELENGTH = 0.1                 # lambda, 
INTEGRATION_GAIN = 1.0           # I
NOISE_TEMPERATURE = 290.0        # T
TOTAL_LOSSES = 1.0               # L
BOLTZMANN_K = 1.380649e-23       # k

# Seeker parameters
SEEKER_GAIN = 3.0                # Gr
SEEKER_BANDWIDTH_HZ = 1.0e6      # B
SEEKER_NOISE_FIGURE = 30.0       # F
SEEKER_BEAMWIDTH_DEG = 36.0      # theta
K_M = 1.6                        # k_M

# Transmitter peak power
RADAR_PEAK_POWER_W = 50_000.0    # W
DECOY_PEAK_POWER_W = 4_000.0     # W

# Antenna peak gains
RADAR_PEAK_GAIN = 1585.0         
DECOY_GAIN_AZ = 2.0              
DECOY_GAIN_EL = 1.5              

RADAR_BEAMWIDTH_AZ_DEG = 3.0
RADAR_BEAMWIDTH_EL_DEG = 3.0




@dataclass
class Transmitter:
    
    name: str
    x: float
    y: float
    kind: str                 # "radar" or "decoy"
    peak_power: float = None  # W
    z: float = 0.0            # ground level by default 

    def __post_init__(self):
        if self.kind not in ("radar", "decoy"):
            raise ValueError("kind must be 'radar' or 'decoy'")
        if self.peak_power is None:
            self.peak_power = (
                RADAR_PEAK_POWER_W if self.kind == "radar" else DECOY_PEAK_POWER_W
            )


def make_radar(x: float, y: float, name: str = "radar") -> Transmitter:
    return Transmitter(name=name, x=x, y=y, kind="radar")


def make_decoy(x: float, y: float, name: str) -> Transmitter:
    return Transmitter(name=name, x=x, y=y, kind="decoy")


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) in radians to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_true_los(tx_x: float, tx_y: float,
                      dec_x: float, dec_y: float, dec_z: float
                      ) -> Tuple[float, float, float]:
    """
    Noiseless LOS (azimuth, elevation, range) from the decision point
    (dec_x, dec_y, dec_z) to a ground-level transmitter at (tx_x, tx_y, 0).
    """
    dx = tx_x - dec_x
    dy = tx_y - dec_y
    dz = 0.0 - dec_z
    horiz = np.hypot(dx, dy)
    R = np.sqrt(dx**2 + dy**2 + dec_z**2)
    az_true = np.arctan2(dy, dx)
    el_true = np.arctan2(dz, horiz)
    return az_true, el_true, R


def project_to_ground(dec_x: float, dec_y: float, dec_z: float,
                       az_est: float, el_est: float) -> Optional[Tuple[float, float]]:

    dx = np.cos(el_est) * np.cos(az_est)
    dy = np.cos(el_est) * np.sin(az_est)
    dz = np.sin(el_est)
    if dz >= 0:
        return None
    t = -dec_z / dz
    x_det = dec_x + t * dx
    y_det = dec_y + t * dy
    return x_det, y_det

# Antenna gain pattern (Schelkunoff-style squared-sinc), SNR, error model

_BEAMWIDTH_SCALE = 2.783099

def _sinc2(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    out = np.ones_like(u)
    mask = np.abs(u) > 1e-9
    out[mask] = (np.sin(u[mask]) / u[mask]) ** 2
    return out


def directional_pattern_gain(delta_angle_rad: float, beamwidth_rad: float) -> float:
    u = _BEAMWIDTH_SCALE * delta_angle_rad / beamwidth_rad
    return _sinc2(u)


def radar_antenna_gain(seeker_az_from_tx: float, seeker_el_from_tx: float,
                        radar_bearing_rad: float) -> float:
    d_az = wrap_to_pi(seeker_az_from_tx - radar_bearing_rad)
    d_el = seeker_el_from_tx  # boresight elevation = 0
    g_az = directional_pattern_gain(d_az, np.radians(RADAR_BEAMWIDTH_AZ_DEG))
    g_el = directional_pattern_gain(d_el, np.radians(RADAR_BEAMWIDTH_EL_DEG))
    return RADAR_PEAK_GAIN * g_az * g_el


def decoy_antenna_gain() -> float:
    return DECOY_GAIN_AZ * DECOY_GAIN_EL


def compute_snr(peak_power_w: float, gain_tx: float, gain_seeker: float,
                 range_m: float) -> float:
    numerator = peak_power_w * gain_tx * gain_seeker * WAVELENGTH**2 * INTEGRATION_GAIN
    denominator = (
        (4 * np.pi * range_m) ** 2
        * BOLTZMANN_K * NOISE_TEMPERATURE * SEEKER_BANDWIDTH_HZ
        * SEEKER_NOISE_FIGURE * TOTAL_LOSSES
    )
    return numerator / denominator


def measurement_error_std(snr: float,
                           theta_deg: float = SEEKER_BEAMWIDTH_DEG,
                           k_m: float = K_M) -> float:
    theta_rad = np.radians(theta_deg)
    return theta_rad / (k_m * np.sqrt(2.0 * snr))


def weighted_los_estimate(los_values: np.ndarray, snrs: np.ndarray) -> float:
    los_values = np.asarray(los_values, dtype=float)
    snrs = np.asarray(snrs, dtype=float)
    return float(np.sum(los_values * snrs) / np.sum(snrs))

# Single-trial and Monte Carlo simulation

def run_single_trial(radar: Transmitter, decoys: List[Transmitter],
                      decision_point: Tuple[float, float, float],
                      rng: np.random.Generator,
                      radar_bearing_rad: Optional[float] = None
                      ) -> Dict[str, Any]:

    dec_x, dec_y, dec_z = decision_point
    transmitters = [radar] + list(decoys)

    if radar_bearing_rad is None:
        radar_bearing_rad = rng.uniform(0.0, 2 * np.pi)

    az_true = np.empty(len(transmitters))
    el_true = np.empty(len(transmitters))
    R = np.empty(len(transmitters))
    G = np.empty(len(transmitters))
    P = np.empty(len(transmitters))

    for i, tx in enumerate(transmitters):
        az_t, el_t, r = compute_true_los(tx.x, tx.y, dec_x, dec_y, dec_z)
        az_true[i], el_true[i], R[i] = az_t, el_t, r
        P[i] = tx.peak_power

        if tx.kind == "radar":
            az_from_tx = wrap_to_pi(az_t + np.pi)
            el_from_tx = -el_t
            G[i] = radar_antenna_gain(az_from_tx, el_from_tx, radar_bearing_rad)
        else:
            G[i] = decoy_antenna_gain()

    SNR = compute_snr(P, G, SEEKER_GAIN, R)
    sigma_n = measurement_error_std(SNR)

    az_ref = az_true[0]
    az_true_unwrapped = az_ref + wrap_to_pi(az_true - az_ref)

    noisy_az = az_true_unwrapped + rng.normal(0.0, sigma_n)
    noisy_el = el_true + rng.normal(0.0, sigma_n)

    az_est = weighted_los_estimate(noisy_az, SNR)
    el_est = weighted_los_estimate(noisy_el, SNR)

    detonation = project_to_ground(dec_x, dec_y, dec_z, az_est, el_est)

    return {
        "radar_bearing_rad": radar_bearing_rad,
        "names": [tx.name for tx in transmitters],
        "az_true": az_true, "el_true": el_true, "R": R, "G": G, "SNR": SNR,
        "sigma_n": sigma_n,
        "az_est": az_est, "el_est": el_est,
        "detonation": detonation,
    }


def run_monte_carlo(radar: Transmitter, decoys: List[Transmitter],
                     decision_point: Tuple[float, float, float],
                     n_trials: int = 10_000,
                     seed: Optional[int] = None) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [run_single_trial(radar, decoys, decision_point, rng)
            for _ in range(n_trials)]


def summarize_results(results: List[Dict[str, Any]],
                       radar: Transmitter,
                       decoys: List[Transmitter]) -> "pd.DataFrame":
                       
    import pandas as pd

    rows = []
    for r in results:
        det = r["detonation"]
        if det is None:
            rows.append({
                "x_det": np.nan, "y_det": np.nan,
                "dist_radar": np.nan, "dist_nearest_decoy": np.nan,
                "radar_bearing_deg": np.degrees(r["radar_bearing_rad"]),
            })
            continue
        x_det, y_det = det
        dist_radar = np.hypot(x_det - radar.x, y_det - radar.y)
        dist_decoys = [np.hypot(x_det - d.x, y_det - d.y) for d in decoys]
        dist_nearest_decoy = min(dist_decoys) if dist_decoys else np.nan
        rows.append({
            "x_det": x_det, "y_det": y_det,
            "dist_radar": dist_radar,
            "dist_nearest_decoy": dist_nearest_decoy,
            "radar_bearing_deg": np.degrees(r["radar_bearing_rad"]),
        })
    return pd.DataFrame(rows)

def plot_simulation(results, radar, decoys, decision_point):
    import matplotlib.pyplot as plt

    # Keep only trials that produced a ground detonation point
    detonations = [r["detonation"] for r in results if r["detonation"] is not None]
    x_det, y_det = zip(*detonations) if detonations else ([], [])

    fig, ax = plt.subplots(figsize=(9, 7))

    # Detonation points: +
    ax.scatter(x_det, y_det, marker="+", s=35, alpha=0.55,
               color="tab:red", label="Detonation")

    # Radar: diamond
    ax.scatter(radar.x, radar.y, marker="D", s=100,
               color="black", label="Radar", zorder=3)

    # Decoys: circles
    ax.scatter([d.x for d in decoys], [d.y for d in decoys],
               marker="o", s=80, color="tab:blue",
               label="Decoy", zorder=3)

    # Decision point: shown as a green square
    ax.scatter(decision_point[0], decision_point[1], marker="s", s=90,
               color="tab:green", label="Decision point", zorder=3)

    ax.set_title("Decoy-location simulation")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

    
if __name__ == "__main__":
    radar = make_radar(0, 0)
    decoys = [
        make_decoy(100, 450, "decoy_1"),
        make_decoy(-200, 400, "decoy_2"),
    ]
    decision_point = (0, 800, 500)

    results = run_monte_carlo(
        radar, decoys, decision_point,
        n_trials=100, seed=42
    )
    print(summarize_results(results, radar, decoys).describe())
    plot_simulation(results, radar, decoys, decision_point)


