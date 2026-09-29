# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:59:01 2026

@author: cns-th-lab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% IMPORTS & DIRECTORIES
import sys
import json
import os
import math
import itertools
import random
import h5py
from pathlib import Path
import keypoint_moseq as kpms
from jax_moseq.utils import set_mixed_map_iters

# Define root directories
keypoint_master_dir = "/Users/cns-th-lab/keypoint_tapus"
data_config_dir = os.path.join(keypoint_master_dir, "data_config")
master_video_dir = "/Users/cns-th-lab/TannerVidsRenamed"


# %% =====================================================================
# HELPER FUNCTIONS 
# =====================================================================
def extract_h5_metadata(filepath):
    """Transforms the h5 file at filepath into a python dictionary for further use."""
    with h5py.File(filepath, "r") as f:
        labels_dict = {
            "node_names": [n.decode("utf-8") for n in f["node_names"][:]],
            "edge_names": [[n1.decode("utf-8"), n2.decode("utf-8")] for n1, n2 in f["edge_names"][:]],
        }
        vid_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(filepath))), os.path.basename(filepath))
        labels_dict["vid_path"] = os.path.splitext(vid_path)[0] + ".mp4"
        labels_dict["sess"] = str.removeprefix(os.path.splitext(os.path.basename(filepath))[0], "mov_")
        labels_dict["subj_id"] = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filepath)))))
        labels_dict["model_name"] = os.path.basename(os.path.dirname(filepath))
        return labels_dict


def assign_nodes_interactively(bodyparts):
    """Helper function to prompt the user for node assignments."""
    print("\n" + "="*50)
    print("BODYPART ASSIGNMENT")
    print("="*50)
    print("For each node, type 'a' for anterior, 'p' for posterior, or press Enter to skip.")
    
    anterior_nodes, posterior_nodes = [], []
    for node in bodyparts:
        while True:
            choice = input(f"Node '{node}': [a/p/skip] > ").strip().lower()
            if choice == 'a':
                anterior_nodes.append(node)
                break
            elif choice == 'p':
                posterior_nodes.append(node)
                break
            elif choice == '':
                break
            else:
                print("  Invalid input. Type 'a', 'p', or just press Enter.")
    return anterior_nodes, posterior_nodes


def update_config_nodes(json_filepath):
    """Reads an existing config, prompts the user to re-assign nodes, and saves it."""
    with open(json_filepath, 'r') as file:
        config = json.load(file)
        
    bodyparts = config["bodyparts"].get("_AVAILABLE_NODES_REFERENCE", [])
    if not bodyparts:
        print("Error: Could not find '_AVAILABLE_NODES_REFERENCE' in the config.")
        sys.exit(1)
        
    print(f"\nRe-assigning nodes for {os.path.basename(json_filepath)}...")
    ant_nodes, post_nodes = assign_nodes_interactively(bodyparts)
    
    config["bodyparts"]["anterior"] = ant_nodes
    config["bodyparts"]["posterior"] = post_nodes
    
    with open(json_filepath, 'w') as file:
        json.dump(config, file, indent=4)
        
    print(f"Config successfully updated: {json_filepath}\n")


def create_config_template(h5_paths, config_name):
    """Generates a sweep-compatible JSON template with interactive node assignment."""
    os.makedirs(data_config_dir, exist_ok=True)
    json_path = os.path.join(data_config_dir, f"{config_name}.json")
    
    sample_h5 = h5_paths[0]
    print(f"\nExtracting metadata from {sample_h5}...")
    try:
        metadata = extract_h5_metadata(sample_h5)
        bodyparts = metadata["node_names"]
    except Exception as e:
        print(f"Error loading h5 file {sample_h5}: {e}")
        sys.exit(1)

    anterior_nodes, posterior_nodes = assign_nodes_interactively(bodyparts)

    config_template = {
        "video_dir": master_video_dir,
        "keypoint_files": h5_paths,
        "bodyparts": {
            "_AVAILABLE_NODES_REFERENCE": bodyparts,
            "anterior": anterior_nodes,
            "posterior": posterior_nodes,
            "use": bodyparts
        },
        "base_parameters": {
            "fps": 30,
            "ar_iters": 50,
            "full_iters": 500,
            "latent_dim": 7,
            "ar_kappa": 2000,
            "full_kappa": 10000
        },
        "sweep_parameters": {
            "full_kappa": [5000, 10000, 20000],
            "latent_dim": [7, 10]
        }
    }

    with open(json_path, 'w') as file:
        json.dump(config_template, file, indent=4)
        
    print(f"\nSuccess! Config saved to '{json_path}'.")
    return json_path


def generate_all_rat_config():
    """Scans the directory structure, picks one random video per rat, and creates the config."""
    videos_base = Path(master_video_dir)
    if not videos_base.exists():
        print(f"Error: Base video directory {master_video_dir} does not exist.")
        sys.exit(1)

    print("\n" + "="*50)
    print("PREDICTION MODEL SELECTION")
    print("="*50)
    print("Example: 260731_198_199x_234x_235x_237x_238x_274x_400x_402x_419x_421x_422x_424x_483x_occin")
    chosen_model = input("Enter the exact name of the prediction model folder:\n> ").strip()
    
    if not chosen_model:
        print("Error: You must specify a model folder name.")
        sys.exit(1)
        
    print(f"\nEnforcing prediction model: {chosen_model}")
    selected_h5_files = []
    
    for rat_dir in [d for d in videos_base.iterdir() if d.is_dir()]:
        videos_dir = rat_dir / "Videos"
        if videos_dir.exists() and videos_dir.is_dir():
            mp4_files = list(videos_dir.glob("*.mp4"))
            if mp4_files:
                random.shuffle(mp4_files)
                for chosen_mp4 in mp4_files:
                    sessid = chosen_mp4.stem
                    target_model_dir = videos_dir / "predictions" / chosen_model
                    
                    matched_h5 = None
                    if target_model_dir.exists():
                        for h5_name in [f"{sessid}.h5", f"mov_{sessid}.h5"]:
                            potential_h5 = target_model_dir / h5_name
                            if potential_h5.exists():
                                matched_h5 = potential_h5
                                break
                    if matched_h5:
                        selected_h5_files.append(str(matched_h5))
                        break 
    
    if not selected_h5_files:
        print(f"Error: Could not find any valid .mp4 / .h5 pairs for model '{chosen_model}'.")
        sys.exit(1)
        
    print(f"Randomly selected {len(selected_h5_files)} videos (1 per rat).")
    return create_config_template(selected_h5_files, "all_rat_data_config")


# %% 1. CONFIGURATION & PROJECT SETUP
target_json = os.path.join(data_config_dir, "all_rat_data_config.json")
if not os.path.exists(target_json):
    target_json = generate_all_rat_config()
    
with open(target_json, 'r') as file:
    config = json.load(file)
    
basename = os.path.splitext(os.path.basename(target_json))[0]
project_dir = os.path.join(keypoint_master_dir, f"{basename}_project")
os.makedirs(project_dir, exist_ok=True)
force_setup = not os.path.exists(os.path.join(project_dir, "config.yml"))
kpms.setup_project(project_dir, sleap_file=config["keypoint_files"][0], overwrite=force_setup)
#%%
bp = config["bodyparts"]
kpms.update_config(
    project_dir,
    video_dir=config["video_dir"],
    anterior_bodyparts=bp["anterior"],
    posterior_bodyparts=bp["posterior"],
    use_bodyparts=bp["use"],
    fps=config["base_parameters"]["fps"]
)
get_config = lambda: kpms.load_config(project_dir)

# %% 2. LOAD KEYPOINTS
print("Loading keypoints...")
coordinates, confidences, _ = kpms.load_keypoints(config["keypoint_files"], format="sleap", extension='h5')

num_batches = math.ceil(len(config["keypoint_files"]) / 4)
set_mixed_map_iters(num_batches)

# %% 3. REMOVE OUTLIERS & CALIBRATE NOISE
print("Removing outlier keypoints...")
kpms.update_config(project_dir, outlier_scale_factor=6.0) # You can adjust this if needed
get_config = lambda: kpms.load_config(project_dir) # Reload updated config
coordinates, confidences = kpms.outlier_removal(coordinates, confidences, project_dir, **get_config())
# %%
print("Calibrating observation noise...")
kpms.noise_calibration(project_dir, coordinates, confidences, **get_config())
get_config = lambda: kpms.load_config(project_dir) # Reload config modified by calibration

# %% 4. FORMAT DATA & FIT PCA
print("Formatting data and fitting PCA...")
data, metadata = kpms.format_data(coordinates, confidences, **get_config())

pca = kpms.fit_pca(**data, **get_config())
kpms.save_pca(pca, project_dir)

kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **get_config())

# %% 5. INITIALIZE MODEL
interactive_params = config["base_parameters"]
interactive_model_name = "interactive_test_model"

kpms.update_config(project_dir, latent_dim=interactive_params["latent_dim"])
model = kpms.init_model(data, pca=pca, **get_config())

# %% 6. FIT AR-HMM (Autoregressive Phase)
model = kpms.update_hypparams(model, kappa=interactive_params["ar_kappa"])
model, _ = kpms.fit_model(
    model, data, metadata, project_dir, 
    model_name=interactive_model_name, 
    ar_only=True, 
    num_iters=interactive_params["ar_iters"]
)

# %% 7. FIT FULL MODEL
model, temp_data, temp_meta, current_iter = kpms.load_checkpoint(
    project_dir, interactive_model_name, iteration=interactive_params["ar_iters"]
)
model = kpms.update_hypparams(model, kappa=interactive_params["full_kappa"])
model = kpms.fit_model(
    model, temp_data, temp_meta, project_dir, interactive_model_name,
    ar_only=False,
    start_iter=current_iter,
    num_iters=current_iter + interactive_params["full_iters"]
)[0]

# %% 8. EXTRACT RESULTS
kpms.reindex_syllables_in_checkpoint(project_dir, interactive_model_name)
model, _, _, _ = kpms.load_checkpoint(project_dir, interactive_model_name)

results = kpms.extract_results(model, metadata, project_dir, interactive_model_name)
kpms.save_results_as_csv(results, project_dir, interactive_model_name)
kpms.generate_trajectory_plots(coordinates, results, project_dir, interactive_model_name, **get_config())


# %% =====================================================================
# BATCH SWEEP PIPELINE (Automated Execution)
# =====================================================================
def run_pipeline(json_filepath):
    """Executes the KPMS pipeline, looping over parameter grids."""
    with open(json_filepath, 'r') as file:
        config = json.load(file)
        
    if not config["bodyparts"].get("anterior") or not config["bodyparts"].get("posterior"):
         print(f"\nNotice: Missing anterior/posterior definitions in {os.path.basename(json_filepath)}.")
         update_config_nodes(json_filepath)
         with open(json_filepath, 'r') as file:
             config = json.load(file)

    print(f"\nLoading configuration from {json_filepath}...")
    basename = os.path.splitext(os.path.basename(json_filepath))[0]
    project_dir = os.path.join(keypoint_master_dir, f"{basename}_project")

    os.makedirs(project_dir, exist_ok=True)
    print(f"\n=== Setting up Project: {project_dir} ===")
    
    force_setup = not os.path.exists(os.path.join(project_dir, "config.yml"))
    kpms.setup_project(project_dir, sleap_file=config["keypoint_files"][0], overwrite=force_setup)

    bp = config["bodyparts"]
    kpms.update_config(
        project_dir,
        video_dir=config["video_dir"],
        anterior_bodyparts=bp["anterior"],
        posterior_bodyparts=bp["posterior"],
        use_bodyparts=bp["use"],
        fps=config["base_parameters"]["fps"]
    )
    
    get_config = lambda: kpms.load_config(project_dir)

    # 1. LOAD KEYPOINTS
    print("Loading keypoints...")
    coordinates, confidences, _ = kpms.load_keypoints(config["keypoint_files"], format="sleap", extension='h5')
    
    num_videos = len(config["keypoint_files"])
    batch_size = 4
    num_batches = math.ceil(num_videos / batch_size)
    print(f"Setting JAX map iterations to {num_batches} ({num_videos} videos, max {batch_size}/batch)...")
    set_mixed_map_iters(num_batches)
    
    # 2. REMOVE OUTLIERS & CALIBRATE NOISE
    print("Removing outlier keypoints...")
    kpms.update_config(project_dir, outlier_scale_factor=6.0) 
    get_config = lambda: kpms.load_config(project_dir)
    coordinates, confidences = kpms.outlier_removal(coordinates, confidences, project_dir, **get_config())

    print("Calibrating observation noise...")
    kpms.noise_calibration(project_dir, coordinates, confidences, **get_config())
    get_config = lambda: kpms.load_config(project_dir) # Reload config modified by calibration

    # 3. FORMAT DATA & FIT PCA
    print("Formatting data...")
    data, metadata = kpms.format_data(coordinates, confidences, **get_config())

    print("Fitting global PCA...")
    pca = kpms.fit_pca(**data, **get_config())
    kpms.save_pca(pca, project_dir)

    # 4. PREPARE PARAMETER GRID
    base_params = config["base_parameters"]
    sweep_params = config.get("sweep_parameters", {})
    if not sweep_params:
        sweep_params = {"dummy": ["dummy"]}
        
    keys, values = zip(*sweep_params.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"\nFound {len(combinations)} parameter combination(s) to test.")

    # 5. EXECUTE SWEEP
    for combo in combinations:
        current_params = base_params.copy()
        if "dummy" in combo:
            model_name = "default_model"
        else:
            current_params.update(combo)
            name_parts = [f"{k}{v}" for k, v in combo.items()]
            model_name = "_".join(name_parts)
            
        print(f"\n--- Training Model: {model_name} ---")
        print(f"Parameters: {current_params}")

        kpms.update_config(project_dir, latent_dim=current_params["latent_dim"])
        
        model = kpms.init_model(data, pca=pca, **get_config())

        print(f"Fitting AR HMM ({current_params['ar_iters']} iters)...")
        model = kpms.update_hypparams(model, kappa=current_params["ar_kappa"])
        model, _ = kpms.fit_model(
            model, data, metadata, project_dir, 
            model_name=model_name, 
            ar_only=True, 
            num_iters=current_params["ar_iters"]
        )

        print(f"Fitting Full Model ({current_params['full_iters']} iters)...")
        model, temp_data, temp_meta, current_iter = kpms.load_checkpoint(
            project_dir, model_name, iteration=current_params["ar_iters"]
        )
        model = kpms.update_hypparams(model, kappa=current_params["full_kappa"])
        model = kpms.fit_model(
            model, temp_data, temp_meta, project_dir, model_name,
            ar_only=False,
            start_iter=current_iter,
            num_iters=current_iter + current_params["full_iters"]
        )[0]

        print("Extracting results and generating media...")
        kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
        model, _, _, _ = kpms.load_checkpoint(project_dir, model_name)
        
        results = kpms.extract_results(model, metadata, project_dir, model_name)
        kpms.save_results_as_csv(results, project_dir, model_name)
        
        kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **get_config())
        kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **get_config())

    print("\nAll parameter combinations completed successfully!")


def check_and_run(target_json):
    """Helper to ask the user if they want to edit nodes before running an existing config."""
    if os.path.exists(target_json):
        edit_choice = input(f"\nFound config: {os.path.basename(target_json)}\nPress 'e' to edit node assignments, or Enter to continue: ").strip().lower()
        if edit_choice == 'e':
            update_config_nodes(target_json)
        run_pipeline(target_json)
    else:
        print(f"Error: Config not found at {target_json}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_arg = sys.argv[1]
        if not config_arg.endswith('.json'):
            config_arg += '.json'
        target_json = os.path.join(data_config_dir, config_arg)
        check_and_run(target_json)
    else:
        default_config_path = os.path.join(data_config_dir, "all_rat_data_config.json")
        if os.path.exists(default_config_path):
            check_and_run(default_config_path)
        else:
            print("No config specified and 'all_rat_data_config.json' not found.")
            print("Generating new 'all_rat_data_config.json'...")
            new_config_path = generate_all_rat_config()
            run_pipeline(new_config_path)