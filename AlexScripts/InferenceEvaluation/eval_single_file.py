import numpy as np
import pandas as pd
import h5py
import json
import os
from tkinter import Tk, filedialog
from scipy.ndimage import convolve1d, gaussian_filter1d

# --- 1. Missing Data & Gap Profiling ---
def measure_nan_gaps(signal_1d):
    is_nan = np.isnan(signal_1d)
    starts = np.where(is_nan & ~np.roll(is_nan, 1))[0]
    stops = np.where(~is_nan & np.roll(is_nan, 1))[0]
    if is_nan[0]: starts = np.insert(starts, 0, 0)
    if is_nan[-1]: stops = np.append(stops, len(is_nan))
    return list(zip(starts, stops)), stops - starts

def convolve_nans(nan_mask, window_size=60):
    window = np.ones(window_size)
    return convolve1d(nan_mask.astype(float), window, mode="constant", cval=0.0)

def get_gap_displacement(coords, intervals):
    displacements = []
    for start, stop in intervals:
        if start > 0 and stop < len(coords):
            p0, pf = coords[start - 1], coords[stop]
            if not np.isnan(p0).any() and not np.isnan(pf).any():
                displacements.append(np.linalg.norm(pf - p0))
    return displacements

# --- 2. Kinematic & Spatial Computations ---
def calc_kinematics(coords):
    dx = np.diff(coords[:, :, 0], axis=0)
    dy = np.diff(coords[:, :, 1], axis=0)
    velocity = np.sqrt(dx**2 + dy**2)
    acceleration = np.abs(np.diff(velocity, axis=0))
    return velocity, acceleration

def to_local_coordinates(coords, origin_idx, basis_idx):
    p_origin = coords[:, origin_idx, :]
    p_basis = coords[:, basis_idx, :]
    centered = coords - p_origin[:, np.newaxis, :]
    b1 = p_basis - p_origin
    norms = np.linalg.norm(b1, axis=1, keepdims=True)
    u = b1 / np.where(norms == 0, 1.0, norms)
    R = np.empty((coords.shape[0], 2, 2))
    R[:, 0, 0], R[:, 0, 1] = u[:, 0], u[:, 1]
    R[:, 1, 0], R[:, 1, 1] = -u[:, 1], u[:, 0]
    return np.einsum("fij, fnj -> fni", R, centered)

def get_angle(p_origin, p1, p2):
    v1, v2 = p1 - p_origin, p2 - p_origin
    v1_n = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_n = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    dot_product = np.sum(v1_n * v2_n, axis=1)
    return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

# --- 3. Outlier Detection & Anomaly Categorization ---
def remove_anatomical_outliers(coords, head_idx, nose_idx, neck_idx):
    angles = get_angle(coords[:, head_idx], coords[:, nose_idx], coords[:, neck_idx])
    q25, q75 = np.nanpercentile(angles, [25, 75])
    iqr = q75 - q25
    outlier_mask = (angles < (q25 - 3 * iqr)) | (angles > (q75 + 3 * iqr))
    coords_clean = np.copy(coords)
    coords_clean[outlier_mask, nose_idx, :] = np.nan
    return coords_clean, outlier_mask

def flag_bone_length_outliers(coords, n1, n2, iqr_mult=1.5):
    edge_vects = coords[:, n1] - coords[:, n2]
    lengths = np.linalg.norm(edge_vects, axis=1)
    q25, q75 = np.nanpercentile(lengths, [25, 75])
    iqr = q75 - q25
    return (lengths < (q25 - iqr_mult * iqr)) | (lengths > (q75 + iqr_mult * iqr))

# --- 4. Smoothing & Interpolation Testing ---
def interpolate_cubic_bounded(coords, max_gap_frames=15):
    interp_coords = np.copy(coords)
    for node in range(coords.shape[1]):
        for axis in range(2):
            series = pd.Series(coords[:, node, axis])
            is_nan = series.isna()
            blocks = (~is_nan).cumsum()
            gap_sizes = is_nan.groupby(blocks).transform('sum')
            valid_gaps = is_nan & (gap_sizes <= max_gap_frames)
            full_interp = series.interpolate(method="pchip", limit_area="inside")
            interp_coords[valid_gaps, node, axis] = full_interp[valid_gaps]
    return interp_coords

def filter_gaussian(coords, sigma=3):
    smoothed = np.copy(coords)
    for node in range(coords.shape[1]):
        smoothed[:, node, 0] = gaussian_filter1d(coords[:, node, 0], sigma)
        smoothed[:, node, 1] = gaussian_filter1d(coords[:, node, 1], sigma)
    return smoothed

# --- Execution Block ---
if __name__ == "__main__":
    PREFS_FILE = "script1_prefs.json"
    prefs = {}
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f:
            prefs = json.load(f)

    root = Tk()
    root.withdraw()
    
    h5_path = filedialog.askopenfilename(
        title="Select Inference .h5 File",
        initialdir=prefs.get("last_h5_dir", os.getcwd()),
        filetypes=[("HDF5 Files", "*.h5")]
    )
    
    if h5_path:
        prefs["last_h5_dir"] = os.path.dirname(h5_path)
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f)
            
        print(f"Loading {h5_path}...")
        with h5py.File(h5_path, "r") as f:
            coords = f['tracks'][:].T
            if coords.ndim == 4: coords = coords[..., 0]
            node_names = [n.decode('utf-8') for n in f['node_names'][:]]
            
        print("Executing Analyses...")
        
        # 1. Missing Data (Using dynamic index for 'nose' or defaulting to 0)
        target_node = node_names.index("nose") if "nose" in node_names else 0
        x_signal_target = coords[:, target_node, 0]
        gaps, gap_lengths = measure_nan_gaps(x_signal_target)
        density = convolve_nans(np.isnan(x_signal_target))
        
        # 2. Kinematics & Spatial 
        vel, acc = calc_kinematics(coords)
        if "spine_2" in node_names and "spine_1" in node_names:
            local_coords = to_local_coordinates(
                coords, 
                origin_idx=node_names.index("spine_2"), 
                basis_idx=node_names.index("spine_1")
            )
        
        # 3. Outliers
        if all(n in node_names for n in ["nose", "implant", "neck"]):
            clean_coords, out_mask = remove_anatomical_outliers(
                coords, 
                head_idx=node_names.index("implant"), 
                nose_idx=node_names.index("nose"), 
                neck_idx=node_names.index("neck")
            )
            
        # 4. Smoothing & Interpolation
        interp_coords = interpolate_cubic_bounded(coords)
        smooth_coords = filter_gaussian(interp_coords)
        
        print("Analysis complete. Processed tensors kept in memory.")