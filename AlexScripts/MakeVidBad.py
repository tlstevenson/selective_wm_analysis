import cv2
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
import sleap_io as sio

# ==========================================
# 1. Frame Rendering Helper
# ==========================================
def render_frame_buffer(frame_idx, labels, video_cap, s_data, i_data):
    """
    Renders a single frame with overlays into a numpy buffer.
    """
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = video_cap.read()
    if not ret:
        return None

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # --- Original SLP (The "Before" state) ---
    for lf in labels.find(video=labels.videos[0], frame_idx=frame_idx):
        for inst in lf.instances:
            for node in inst.skeleton.nodes:
                try:
                    pt = inst[node]
                    if getattr(pt, 'visible', True) and not np.isnan(pt.x):
                        ax.scatter(pt.x, pt.y, c='lightgray', marker='o', s=60, alpha=0.5, edgecolors='black')
                except: continue

    # --- Interpolated H5 (The "Fixed" state) ---
    # Shape check: (instances, dims, nodes, frames)
    int_x = i_data[0, 0, :, frame_idx]
    int_y = i_data[0, 1, :, frame_idx]
    ax.scatter(int_x, int_y, c='red', marker='*', s=150, label='Interpolated')

    # --- Stripped H5 (The "Confident" state) ---
    strp_x = s_data[0, 0, :, frame_idx]
    strp_y = s_data[0, 1, :, frame_idx]
    ax.scatter(strp_x, strp_y, c='cyan', marker='o', s=25, label='Confident')

    ax.set_title(f"QC Video | Frame: {frame_idx}", fontsize=14)
    ax.axis('off')
    plt.tight_layout()

    # Convert plot to image buffer
    fig.canvas.draw()
    
    # Grab the RGBA buffer directly as a numpy array
    img = np.asarray(fig.canvas.buffer_rgba())
    
    # Drop the Alpha (transparency) channel so it is pure RGB
    img = img[:, :, :3] 
    
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# ==========================================
# 2. Main Logic: Find 5 Peaks & Export 5 Videos
# ==========================================
def export_top_5_qc_videos(slp_path, stripped_h5, interp_h5, node_priorities, window_size=15):
    print("Analyzing tracking gaps and finding top 5 resolved maxima...")
    
    # --- A. Scoring ---
    with h5py.File(stripped_h5, 'r') as f:
        s_data = f['tracks'][:]
        node_names = [n.decode('utf-8') for n in f['node_names'][:]]
    with h5py.File(interp_h5, 'r') as f:
        i_data = f['tracks'][:]

    missing_mask = np.isnan(s_data[0, 0, :, :])
    weights = np.array([2 if node_priorities.get(n, 'low') == 'high' else 1 for n in node_names])
    raw_labels = np.sum(missing_mask * weights[:, None], axis=0)
    smoothed = pd.Series(raw_labels).rolling(window=window_size, center=True).mean().to_numpy()

    # Find peaks where the interpolated data is actually successful (no NaNs)
    peaks, _ = find_peaks(smoothed)
    resolved_peaks = [p for p in peaks if not np.isnan(i_data[0, 0, :, p]).any()]
    
    if not resolved_peaks:
        print("Error: Could not find any fully resolved gaps.")
        return

    # Sort resolved peaks by their severity
    resolved_peaks.sort(key=lambda p: smoothed[p], reverse=True)
    top_5_centers = resolved_peaks[:5]

    # --- B. Video Generation Loop ---
    labels = sio.load_slp(str(slp_path))
    video_file = labels.videos[0].filename
    
    for i, center_frame in enumerate(top_5_centers):
        rank = i + 1
        output_name = slp_path.parent / f"QC_Area_{rank}_Worst.mp4"
        
        start_f = max(0, center_frame - 15)
        end_f = min(s_data.shape[-1], center_frame + 15)

        # Initialize Video Capture and Writer
        cap = cv2.VideoCapture(video_file)
        
        # Test render to get frame size
        sample = render_frame_buffer(start_f, labels, cap, s_data, i_data)
        h, w, _ = sample.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_name), fourcc, 10.0, (w, h))

        print(f"\nRendering Video {rank}/5: {output_name.name}")
        for f_idx in range(start_f, end_f):
            img = render_frame_buffer(f_idx, labels, cap, s_data, i_data)
            if img is not None:
                out.write(img)
                print(f"  Progress: {f_idx - start_f + 1}/30 frames", end='\r')

        cap.release()
        out.release()
    
    print("\n" + "="*40 + "\nAll 5 QC videos have been exported.")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    #Can also use base_dir / filename
    base_dir = Path(r"C:\Users\cns-th-lab\SLEAP_Labels_4_29_26\198\Videos")
    slp_file = Path(r"C:\Users\cns-th-lab\SLEAP_Labels_4_29_26\198\Videos\198.2025-08-01.mov_0005.proj.slp")
    strp_h5 = Path(r"C:\Users\cns-th-lab\SLEAP_Labels_4_29_26\198\Videos\198.2025-08-01.mov_0005.proj_STRIPPED.h5")
    intp_h5 = Path(r"C:\Users\cns-th-lab\SLEAP_Labels_4_29_26\198\Videos\198.2025-08-01.mov_0005.proj_INTERPOLATED.h5")

    # Define your priority nodes
    my_priorities = {
        'nose': 'high',
        'implant': 'high',
        'ear_l': 'low',
        'ear_r': 'low',
        'cheek_l': 'low',
        'cheek_r': 'low',
        'body_end': 'low',
        'contour_1_l': 'high',
        'contour_1_r': 'high',
        'contour_2_l': 'high',
        'contour_2_r': 'high',
        'contour_3_l': 'high',
        'contour_3_r': 'high',
        'spine_1': 'high',
        'spine_2': 'high',
        'spine_3': 'high',
        'tail_base': 'high',
        'tail_mid': 'low',
        'tail_end': 'low'
    }

    if slp_file.exists() and strp_h5.exists() and intp_h5.exists():
        export_top_5_qc_videos(slp_file, strp_h5, intp_h5, my_priorities)
    else:
        print("Check file paths. Required files not found.")