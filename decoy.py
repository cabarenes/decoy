"""
decoy_simulation.py
====================

Stochastic simulation model for the *Decoy Location Problem* (DLP): estimating
where an anti-radiation missile (ARM) detonates when its passive seeker's
line-of-sight (LOS) estimate to a surface-based radar is corrupted by decoy
transmitters deployed around the radar.

This module implements exactly the five-step model logic described in the
brief:

    1. Set the locations of the radar, decoys, and the decision point.
    2. Calculate the noiseless (true) LOS from the decision point to each
       transmitter (radar + decoys).
    3. Determine the noisy LOS for each transmitter (SNR-dependent Gaussian
       error).
    4. Combine the noisy LOS values into a single SNR-weighted LOS estimate.
    5. Project the LOS estimate to the ground -> detonation point.

--------------------------------------------------------------------------
ONE MODELING ASSUMPTION WORTH FLAGGING EXPLICITLY
--------------------------------------------------------------------------
The brief cites a directional-antenna gain pattern "according to the model
of a transmitter pattern (Schelkunoff 1943)" but does not give the closed
form. Schelkunoff's 1943 array theory shows that a uniformly illuminated
linear array/aperture produces a pattern whose continuous-angle limit is the
classic squared-sinc ("Dirichlet kernel") shape, with a first sidelobe about
13 dB below the main lobe. That is the standard, textbook Schelkunoff-style
pattern, and is what `directional_pattern_gain()` below implements:

    F(delta) = [ sin(u) / u ]^2 ,   u = c * delta / theta_3dB

where `theta_3dB` is the antenna's -3 dB beamwidth and c is chosen so that
F(theta_3dB / 2) = 0.5 exactly. The radar's total gain in a given direction
is modeled as the peak gain times this pattern evaluated separately in the
azimuth and elevation planes (a standard separable-pattern approximation):

    G_radar(d_az, d_el) = G_peak * F(d_az; theta_az) * F(d_el; theta_el)

The two -3 dB beamwidths (`RADAR_BEAMWIDTH_AZ_DEG`, `RADAR_BEAMWIDTH_EL_DEG`)
are NOT stated in the brief, so they are exposed as tunable constants below
(defaulted to a narrow, high-gain pencil beam consistent with the stated
32 dB peak gain). If your source paper gives an explicit pattern formula or
beamwidth/sidelobe numbers, replace `directional_pattern_gain()` and the two
constants accordingly -- everything else in the model is independent of that
choice.

The decoys are explicitly stated to be omnidirectional with constant gain,
so no pattern function is applied to them.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


# ==========================================================================
# 1. Fixed physical / model constants (as specified in the brief)
# ==========================================================================

WAVELENGTH = 0.1                 # lambda, m
INTEGRATION_GAIN = 1.0           # I, linear (0 dB)
NOISE_TEMPERATURE = 290.0        # T, K
TOTAL_LOSSES = 1.0               # L, linear (0 dB)
BOLTZMANN_K = 1.380649e-23       # k, J/K

# Seeker (missile receiver) parameters
SEEKER_GAIN = 3.0                # Gr, linear (4.8 dB)
SEEKER_BANDWIDTH_HZ = 1.0e6      # B, Hz (1 MHz)
SEEKER_NOISE_FIGURE = 30.0       # F, linear (14.8 dB)
SEEKER_BEAMWIDTH_DEG = 36.0      # theta, -3dB beamwidth used in error model
K_M = 1.6                        # k_M, constant factor in error model

# Transmitter peak (radiated) power
RADAR_PEAK_POWER_W = 50_000.0    # W
DECOY_PEAK_POWER_W = 4_000.0     # W

# Antenna peak gains
RADAR_PEAK_GAIN = 1585.0         # linear (32.0 dB), shared peak of az & el cuts
DECOY_GAIN_AZ = 2.0              # linear (3.0 dB), constant (omnidirectional)
DECOY_GAIN_EL = 1.5              # linear (1.8 dB), constant (omnidirectional)

# Radar directional-pattern beamwidths -- ASSUMPTION, see module docstring.
# Tune these if your reference gives explicit values.
RADAR_BEAMWIDTH_AZ_DEG = 3.0
RADAR_BEAMWIDTH_EL_DEG = 3.0


# ==========================================================================
# 2. Geometry primitives
# ==========================================================================

@dataclass
class Transmitter:
    """A ground-based transmitter: either the radar or a single decoy."""
    name: str
    x: float
    y: float
    kind: str                 # "radar" or "decoy"
    peak_power: float = None  # W; defaults set in __post_init__ if omitted
    z: float = 0.0            # ground level by default (flat terrain)

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

    Azimuth is measured in the ground (x, y) plane via atan2(dy, dx).
    Elevation is measured from the local horizontal at the decision point;
    it is negative when the transmitter lies below the decision point
    (the usual case, since the missile is elevated above the target array).

    Returns
    -------
    az_true, el_true : float (radians)
    R : float, slant range (m), R = sqrt(dx^2 + dy^2 + dec_z^2)
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
    """
    Project the estimated LOS (azimuth, elevation) ray from the decision
    point onto the ground plane z = 0, giving the detonation point.

    Returns None if the estimated ray does not point downward (a degenerate
    case that should be rare given realistic geometries / errors).
    """
    dx = np.cos(el_est) * np.cos(az_est)
    dy = np.cos(el_est) * np.sin(az_est)
    dz = np.sin(el_est)
    if dz >= 0:
        return None
    t = -dec_z / dz
    x_det = dec_x + t * dx
    y_det = dec_y + t * dy
    return x_det, y_det


# ==========================================================================
# 3. Antenna gain pattern (Schelkunoff-style squared-sinc), SNR, error model
# ==========================================================================

# Solves [sin(c/2)/(c/2)]^2 = 0.5  ->  c/2 = 1.391549...  (standard sinc-2 result)
_BEAMWIDTH_SCALE = 2.783099

def _sinc2(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    out = np.ones_like(u)
    mask = np.abs(u) > 1e-9
    out[mask] = (np.sin(u[mask]) / u[mask]) ** 2
    return out


def directional_pattern_gain(delta_angle_rad: float, beamwidth_rad: float) -> float:
    """
    Normalized (peak = 1 at delta_angle = 0) directional antenna pattern,
    following the continuous-aperture limit of Schelkunoff's (1943) uniform
    array factor (squared sinc / Dirichlet-kernel shape). See module
    docstring for the reasoning and how to swap in an exact formula.
    """
    u = _BEAMWIDTH_SCALE * delta_angle_rad / beamwidth_rad
    return _sinc2(u)


def radar_antenna_gain(seeker_az_from_tx: float, seeker_el_from_tx: float,
                        radar_bearing_rad: float) -> float:
    """
    Total radar antenna gain toward the seeker, given the seeker's direction
    as seen FROM the radar (azimuth/elevation) and the radar's antenna
    boresight bearing (elevation boresight assumed 0, i.e. pointed at the
    horizon).
    """
    d_az = wrap_to_pi(seeker_az_from_tx - radar_bearing_rad)
    d_el = seeker_el_from_tx  # boresight elevation = 0
    g_az = directional_pattern_gain(d_az, np.radians(RADAR_BEAMWIDTH_AZ_DEG))
    g_el = directional_pattern_gain(d_el, np.radians(RADAR_BEAMWIDTH_EL_DEG))
    return RADAR_PEAK_GAIN * g_az * g_el


def decoy_antenna_gain() -> float:
    """Decoys are omnidirectional: constant gain regardless of viewing angle."""
    return DECOY_GAIN_AZ * DECOY_GAIN_EL


def compute_snr(peak_power_w: float, gain_tx: float, gain_seeker: float,
                 range_m: float) -> float:
    """
    Equation (1): one-way link SNR at the seeker from a transmitter with
    peak power `peak_power_w`, transmit gain `gain_tx`, at slant range
    `range_m`, received by the seeker with gain `gain_seeker`.
    """
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
    """
    Equation (3): standard deviation (radians) of the SNR-dependent LOS
    measurement error for a single transmitter, sigma_n = theta / (k_M * sqrt(2*SNR)).
    """
    theta_rad = np.radians(theta_deg)
    return theta_rad / (k_m * np.sqrt(2.0 * snr))


def weighted_los_estimate(los_values: np.ndarray, snrs: np.ndarray) -> float:
    """Equation (4): SNR-weighted average of the noisy LOS values."""
    los_values = np.asarray(los_values, dtype=float)
    snrs = np.asarray(snrs, dtype=float)
    return float(np.sum(los_values * snrs) / np.sum(snrs))


# ==========================================================================
# 4. Single-trial and Monte Carlo simulation
# ==========================================================================

def run_single_trial(radar: Transmitter, decoys: List[Transmitter],
                      decision_point: Tuple[float, float, float],
                      rng: np.random.Generator,
                      radar_bearing_rad: Optional[float] = None
                      ) -> Dict[str, Any]:
    """
    Run one Monte Carlo replication of the DLP simulation model (steps 1-5).

    Parameters
    ----------
    radar : Transmitter (kind="radar")
    decoys : list of Transmitter (kind="decoy")
    decision_point : (xdec, ydec, zdec)
    rng : numpy Generator, source of randomness for this trial
    radar_bearing_rad : optional fixed radar bearing; if None, drawn
        uniformly from [0, 2*pi) as specified (the first source of
        uncertainty in the model).

    Returns
    -------
    dict with per-transmitter intermediate values, the LOS estimate, and
    the resulting detonation point (None if the projected ray does not
    point at the ground).
    """
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
            # Direction of the seeker AS SEEN FROM the radar is the reverse
            # of the decision-point-to-radar LOS.
            az_from_tx = wrap_to_pi(az_t + np.pi)
            el_from_tx = -el_t
            G[i] = radar_antenna_gain(az_from_tx, el_from_tx, radar_bearing_rad)
        else:
            G[i] = decoy_antenna_gain()

    SNR = compute_snr(P, G, SEEKER_GAIN, R)
    sigma_n = measurement_error_std(SNR)

    # Unwrap azimuth values relative to the radar's true azimuth before
    # weighted-averaging, to avoid +-pi wraparound artifacts.
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
    """Run `n_trials` independent replications and return the raw results."""
    rng = np.random.default_rng(seed)
    return [run_single_trial(radar, decoys, decision_point, rng)
            for _ in range(n_trials)]


def summarize_results(results: List[Dict[str, Any]],
                       radar: Transmitter,
                       decoys: List[Transmitter]) -> "pd.DataFrame":
    """
    Turn raw Monte Carlo trial results into a tidy pandas DataFrame with one
    row per trial: detonation coordinates, distance from radar, and distance
    from the nearest decoy.
    """
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

plot_simulation(results, radar, decoys, decision_point)