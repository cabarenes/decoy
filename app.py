from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

app = Flask(__name__)

# --- ORIGINAL SIMULATION CODE START ---
WAVELENGTH = 0.1                 
INTEGRATION_GAIN = 1.0           
NOISE_TEMPERATURE = 290.0        
TOTAL_LOSSES = 1.0               
BOLTZMANN_K = 1.380649e-23       
SEEKER_GAIN = 3.0                
SEEKER_BANDWIDTH_HZ = 1.0e6      
SEEKER_NOISE_FIGURE = 30.0       
SEEKER_BEAMWIDTH_DEG = 36.0      
K_M = 1.6                        
RADAR_PEAK_POWER_W = 50_000.0    
DECOY_PEAK_POWER_W = 4_000.0     
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
    kind: str                 
    peak_power: float = None  
    z: float = 0.0            

    def __post_init__(self):
        if self.kind not in ("radar", "decoy"):
            raise ValueError("kind must be 'radar' or 'decoy'")
        if self.peak_power is None:
            self.peak_power = (RADAR_PEAK_POWER_W if self.kind == "radar" else DECOY_PEAK_POWER_W)

def make_radar(x: float, y: float, name: str = "radar") -> Transmitter:
    return Transmitter(name=name, x=x, y=y, kind="radar")

def make_decoy(x: float, y: float, name: str, peak_power: float = None) -> Transmitter:
    return Transmitter(name=name, x=x, y=y, kind="decoy", peak_power=peak_power)

def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_true_los(tx_x: float, tx_y: float, dec_x: float, dec_y: float, dec_z: float) -> Tuple[float, float, float]:
    dx, dy, dz = tx_x - dec_x, tx_y - dec_y, 0.0 - dec_z
    horiz = np.hypot(dx, dy)
    R = np.sqrt(dx**2 + dy**2 + dec_z**2)
    az_true = np.arctan2(dy, dx)
    el_true = np.arctan2(dz, horiz)
    return az_true, el_true, R

def project_to_ground(dec_x: float, dec_y: float, dec_z: float, az_est: float, el_est: float) -> Optional[Tuple[float, float]]:
    dx = np.cos(el_est) * np.cos(az_est)
    dy = np.cos(el_est) * np.sin(az_est)
    dz = np.sin(el_est)
    if dz >= 0: return None
    t = -dec_z / dz
    x_det = dec_x + t * dx
    y_det = dec_y + t * dy
    return x_det, y_det

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

def radar_antenna_gain(seeker_az_from_tx: float, seeker_el_from_tx: float, radar_bearing_rad: float) -> float:
    d_az = wrap_to_pi(seeker_az_from_tx - radar_bearing_rad)
    g_az = directional_pattern_gain(d_az, np.radians(RADAR_BEAMWIDTH_AZ_DEG))
    g_el = directional_pattern_gain(seeker_el_from_tx, np.radians(RADAR_BEAMWIDTH_EL_DEG))
    return RADAR_PEAK_GAIN * g_az * g_el

def decoy_antenna_gain() -> float:
    return DECOY_GAIN_AZ * DECOY_GAIN_EL

def compute_snr(peak_power_w: float, gain_tx: float, gain_seeker: float, range_m: float) -> float:
    numerator = peak_power_w * gain_tx * gain_seeker * WAVELENGTH**2 * INTEGRATION_GAIN
    denominator = ((4 * np.pi * range_m) ** 2 * BOLTZMANN_K * NOISE_TEMPERATURE * SEEKER_BANDWIDTH_HZ * SEEKER_NOISE_FIGURE * TOTAL_LOSSES)
    return numerator / denominator

def measurement_error_std(snr: float, theta_deg: float = SEEKER_BEAMWIDTH_DEG, k_m: float = K_M) -> float:
    return np.radians(theta_deg) / (k_m * np.sqrt(2.0 * snr))

def weighted_los_estimate(los_values: np.ndarray, snrs: np.ndarray) -> float:
    return float(np.sum(los_values * snrs) / np.sum(snrs))

def run_single_trial(radar: Transmitter, decoys: List[Transmitter], decision_point: Tuple[float, float, float], rng: np.random.Generator, radar_bearing_rad: Optional[float] = None) -> Dict[str, Any]:
    dec_x, dec_y, dec_z = decision_point
    transmitters = [radar] + list(decoys)
    if radar_bearing_rad is None: radar_bearing_rad = rng.uniform(0.0, 2 * np.pi)

    az_true, el_true, R, G, P = (np.empty(len(transmitters)) for _ in range(5))

    for i, tx in enumerate(transmitters):
        az_t, el_t, r = compute_true_los(tx.x, tx.y, dec_x, dec_y, dec_z)
        az_true[i], el_true[i], R[i] = az_t, el_t, r
        P[i] = tx.peak_power
        if tx.kind == "radar":
            G[i] = radar_antenna_gain(wrap_to_pi(az_t + np.pi), -el_t, radar_bearing_rad)
        else:
            G[i] = decoy_antenna_gain()

    SNR = compute_snr(P, G, SEEKER_GAIN, R)
    sigma_n = measurement_error_std(SNR)
    az_ref = az_true[0]
    
    noisy_az = az_ref + wrap_to_pi(az_true - az_ref) + rng.normal(0.0, sigma_n)
    noisy_el = el_true + rng.normal(0.0, sigma_n)

    az_est = weighted_los_estimate(noisy_az, SNR)
    el_est = weighted_los_estimate(noisy_el, SNR)

    return {
        "radar_bearing_rad": radar_bearing_rad, "names": [tx.name for tx in transmitters],
        "detonation": project_to_ground(dec_x, dec_y, dec_z, az_est, el_est),
    }

def run_monte_carlo(radar: Transmitter, decoys: List[Transmitter], decision_point: Tuple[float, float, float], n_trials: int = 100, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [run_single_trial(radar, decoys, decision_point, rng) for _ in range(n_trials)]

def summarize_results(results: List[Dict[str, Any]], radar: Transmitter, decoys: List[Transmitter]) -> pd.DataFrame:
    rows = []
    for r in results:
        det = r["detonation"]
        if det is None: continue
        x_det, y_det = det
        dist_radar = np.hypot(x_det - radar.x, y_det - radar.y)
        dist_decoys = [np.hypot(x_det - d.x, y_det - d.y) for d in decoys]
        rows.append({
            "x_det": x_det, "y_det": y_det,
            "dist_radar": dist_radar,
            "dist_nearest_decoy": min(dist_decoys) if dist_decoys else np.nan,
            "radar_bearing_deg": np.degrees(r["radar_bearing_rad"]),
        })
    return pd.DataFrame(rows)
# --- ORIGINAL SIMULATION CODE END ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    data = request.json
    
    # Parse decision point
    dec_x = float(data.get('dec_x', 0))
    dec_y = float(data.get('dec_y', 800))
    dec_z = float(data.get('dec_z', 200))
    decision_point = (dec_x, dec_y, dec_z)

    # Parse Decoy ERP
    decoy_erp = float(data.get('decoy_erp', 4000.0))

    # Parse decoys
    raw_decoys = data.get('decoys', [])
    decoys = []
    for i, d in enumerate(raw_decoys):
        decoys.append(make_decoy(float(d['x']), float(d['y']), f"decoy_{i+1}", peak_power=decoy_erp))

    radar = make_radar(0, 0)
    
    # Run Simulation
    results = run_monte_carlo(radar, decoys, decision_point, n_trials=500, seed=42)
    
    # Extract Detonation Points
    detonations = [{"x": r["detonation"][0], "y": r["detonation"][1]} for r in results if r["detonation"] is not None]
    
    # Generate Data Table
    df = summarize_results(results, radar, decoys)
    if not df.empty:
        # 1. Drop the radar_bearing_deg column
        df = df.drop(columns=['radar_bearing_deg'], errors='ignore')
        
        # 2. Get the describe statistics
        desc = df.describe()
        
        # 3. Drop count and percentile rows
        desc = desc.drop(['count', '25%', '50%', '75%'], errors='ignore')
        
        # 4. Rename the columns to clean titles
        desc = desc.rename(columns={
            'x_det': 'X Detonation (m)',
            'y_det': 'Y Detonation (m)',
            'dist_radar': 'Distance to Radar (m)',
            'dist_nearest_decoy': 'Distance to Nearest Decoy (m)'
        })
        
        # 5. Capitalize the index row names (mean -> Mean, std -> Std, etc.)
        desc.index = [str(idx).capitalize() for idx in desc.index]
        
        # 6. Format all floats to 2 decimal places for readability
        table_html = desc.to_html(classes="dataframe", float_format="%.2f")
    else:
        table_html = "<p>No detonations recorded on the ground.</p>"
    
    return jsonify({
        "detonations": detonations,
        "table_html": table_html
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)
