import sleap_io as sio
import os
import glob
import yaml
import re
import numpy as np

# --- STEP 1: DEFINE PROJECT PATHS ---
project_dir = r"C:\Users\cns-th-lab\DeepLabCut_Projects\AllRatsBulky-AITapus-2026-04-08"

config_path = os.path.join(project_dir, "config.yaml")
labeled_data_dir = os.path.join(project_dir, "labeled-data")
base_vid_dir = r"C:\Users\cns-th-lab\Tanner_Alex_Vids"

# --- STEP 2: AUTO-PARSE DLC CONFIG ---
print("Reading DeepLabCut config.yaml...")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

node_names = config.get('bodyparts', [])
dlc_edges = config.get('skeleton', [])

# Build SLEAP Master Skeleton dynamically
node_map = {name: sio.Node(name=name) for name in node_names}
edges = []
for edge in dlc_edges:
    if len(edge) == 2 and edge[0] in node_map and edge[1] in node_map:
        edges.append(sio.Edge(source=node_map[edge[0]], destination=node_map[edge[1]]))

# We generate the skeleton and track ONCE so they are mathematically identical across all files
master_skeleton = sio.Skeleton(nodes=list(node_map.values()), edges=edges)
master_track = sio.Track(name="Rat_1")

# --- STEP 3: PROCESS AND SAVE INDIVIDUAL CSVs ---
csv_files = glob.glob(os.path.join(labeled_data_dir, "**", "*.csv"), recursive=True)
print(f"Found {len(csv_files)} labeled datasets.\n")
print("="*50)

for csv_file in csv_files:
    video_name = os.path.basename(os.path.dirname(csv_file))
    print(f"Processing: {video_name}")
    
    # 1. Initialize a BRAND NEW Labels object for this specific video
    labels = sio.Labels(skeletons=[master_skeleton])
    labels.tracks.append(master_track)
    
    # 2. Search for the video
    search_pattern = os.path.join(base_vid_dir, "**", f"{video_name}.*")
    potential_videos = glob.glob(search_pattern, recursive=True)
    
    real_video_path = None
    vid_extensions = [".mp4", ".avi", ".mov"]
    
    for pv in potential_videos:
        if any(pv.lower().endswith(ext) for ext in vid_extensions):
            real_video_path = pv
            break
            
    if not real_video_path:
        print(f"  [!] WARNING: Could not find full video for {video_name}. Skipping.\n")
        continue
        
    print(f"  -> Linked video: {real_video_path}")
    real_video = sio.Video.from_filename(real_video_path)
    labels.videos.append(real_video)
    
    # 3. Load the DLC CSV 
    dlc_labels = sio.load_file(csv_file)
    
    frames_added = 0
    for dlc_lf in dlc_labels.labeled_frames:
        
        vid_file = dlc_lf.video.filename
        if isinstance(vid_file, (tuple, list)):
            image_filename = vid_file[dlc_lf.frame_idx]
        else:
            image_filename = str(vid_file)
            
        match = re.search(r'\d+', os.path.basename(image_filename))
        real_frame_idx = int(match.group()) if match else dlc_lf.frame_idx 
            
        new_instances = []
        for inst in dlc_lf.instances:
            old_pts = inst.numpy()[:, :2] 
            new_pts = np.full((len(master_skeleton.nodes), 2), np.nan)
            
            for i, old_node in enumerate(inst.skeleton.nodes):
                if old_node.name in node_map:
                    new_idx = master_skeleton.nodes.index(node_map[old_node.name])
                    new_pts[new_idx] = old_pts[i]
                    
            user_inst = sio.Instance.from_numpy(
                new_pts,
                skeleton=master_skeleton,
                track=master_track
            )
            new_instances.append(user_inst)
            
        new_lf = sio.LabeledFrame(
            video=real_video,
            frame_idx=real_frame_idx,
            instances=new_instances
        )
        labels.labeled_frames.append(new_lf)
        frames_added += 1
        
    print(f"  -> Mapped {frames_added} frames.")
    
    # 4. SAVE IMMEDIATELY
    # This saves the .slp file directly next to the original .csv file!
    output_filename = csv_file.replace(".csv", "_SLEAP.slp")
    
    try:
        sio.save_file(labels, output_filename)
        print(f"  -> [SUCCESS] Saved to: {os.path.basename(output_filename)}\n")
    except Exception as e:
        print(f"  -> [ERROR] Failed to save {video_name}. Details: {e}\n")

print("="*50)
print("ALL FILES SUCCESSFULLY CONVERTED!")