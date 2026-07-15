#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import os
import keypoint_moseq as kpms

def create_config_template(h5_path, json_path):
    """Extracts node names from an h5 file and generates a JSON template."""
    print(f"\nExtracting node names from {h5_path}...")
    
    # We use kpms to load just one file to safely grab the exact bodyparts list
    try:
        _, _, bodyparts = kpms.load_keypoints(
            filepath_pattern=[h5_path], 
            format="sleap", 
            extension="h5"
        )
    except Exception as e:
        print(f"Error loading h5 file: {e}")
        sys.exit(1)

    # Build the template dictionary
    config_template = {
        "project_dir": "ENTER_PROJECT_DIRECTORY_HERE",
        "video_dir": "ENTER_VIDEO_DIRECTORY_HERE",
        "keypoint_files": [
            h5_path,
            "ADD_MORE_FILE_PATHS_HERE.h5"
        ],
        "bodyparts": {
            "_AVAILABLE_NODES_REFERENCE": bodyparts,  # Keeping this here so you can copy/paste easily
            "anterior": [],                           # PASTE ANTERIOR NODES HERE
            "posterior": [],                          # PASTE POSTERIOR NODES HERE
            "use": bodyparts                          # Defaults to using all nodes
        },
        "parameters": {
            "fps": 30,
            "latent_dim": 7,
            "ar_iters": 50,
            "ar_kappa": 2000,
            "full_iters": 500,
            "full_kappa": 10000
        }
    }

    # Write it to the JSON file
    with open(json_path, 'w') as file:
        json.dump(config_template, file, indent=4)
        
    print(f"\nSuccess! A configuration template has been saved to '{json_path}'.")
    print("=======================================================================")
    print("ACTION REQUIRED:")
    print("1. Open config.json in a text editor.")
    print("2. Copy nodes from '_AVAILABLE_NODES_REFERENCE' into 'anterior' and 'posterior'.")
    print("3. Update your project/video directories and add any other .h5 files.")
    print("4. Run this script again to execute the pipeline.")
    print("=======================================================================")


def run_pipeline(json_filepath):
    """Executes the KPMS pipeline using the parameters in the JSON file."""
    print(f"Loading configuration from {json_filepath}...")
    with open(json_filepath, 'r') as file:
        config_data = json.load(file)

    project_dir = config_data["project_dir"]
    video_dir = config_data["video_dir"]
    keypoint_files = config_data["keypoint_files"]
    bp = config_data["bodyparts"]
    params = config_data["parameters"]

    # Sanity check to make sure the user actually edited the template
    if project_dir == "ENTER_PROJECT_DIRECTORY_HERE":
        print("Error: You need to edit the config.json with your actual paths before running the pipeline.")
        sys.exit(1)

    os.makedirs(project_dir, exist_ok=True)

    print("Setting up project and updating configuration...")
    kpms.setup_project(project_dir, sleap_file=keypoint_files[0], overwrite=False)

    kpms.update_config(
        project_dir,
        video_dir=video_dir,
        anterior_bodyparts=bp["anterior"],
        posterior_bodyparts=bp["posterior"],
        use_bodyparts=bp["use"],
        fps=params["fps"]
    )

    get_config = lambda: kpms.load_config(project_dir)

    print(f"Loading keypoints for {len(keypoint_files)} files...")
    coordinates, confidences, bodyparts = kpms.load_keypoints(
        filepath_pattern=keypoint_files, 
        format="sleap",
        extension='h5'
    )

    print("Formatting data...")
    data, metadata = kpms.format_data(coordinates, confidences, **get_config())

    print("Fitting PCA...")
    pca = kpms.fit_pca(**data, **get_config())
    kpms.save_pca(pca, project_dir)
    
    kpms.plot_scree(pca, project_dir=project_dir)
    kpms.plot_pcs(pca, project_dir=project_dir, **get_config())

    print(f"Initializing model with latent_dim = {params['latent_dim']}...")
    kpms.update_config(project_dir, latent_dim=params["latent_dim"])
    model = kpms.init_model(data, pca=pca, **get_config())

    print(f"Fitting AR HMM for {params['ar_iters']} iterations...")
    model = kpms.update_hypparams(model, kappa=params["ar_kappa"])
    model, model_name = kpms.fit_model(
        model, data, metadata, project_dir, 
        ar_only=True, 
        num_iters=params["ar_iters"]
    )

    print(f"Fitting Full Model for {params['full_iters']} iterations...")
    model, data, metadata, current_iter = kpms.load_checkpoint(
        project_dir, model_name, iteration=params["ar_iters"]
    )

    model = kpms.update_hypparams(model, kappa=params["full_kappa"])
    model = kpms.fit_model(
        model, data, metadata, project_dir, model_name,
        ar_only=False,
        start_iter=current_iter,
        num_iters=current_iter + params["full_iters"]
    )[0]

    print("Reindexing syllables and extracting results...")
    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
    model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)
    
    results = kpms.extract_results(model, metadata, project_dir, model_name)
    kpms.save_results_as_csv(results, project_dir, model_name)

    print("Generating plots and grid movies...")
    kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **get_config())
    kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **get_config())
    kpms.generate_grid_movies(results, project_dir, model_name, coordinates=coordinates, **get_config())

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    # Define standard JSON filename
    json_path = "ant_pos.json"
    
    # Allow overriding json path via command line argument (e.g., `python run_kpms.py my_custom_config.json`)
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        
    # Check if the config file exists
    if not os.path.exists(json_path):
        print(f"Configuration file '{json_path}' not found.")
        # Ask the user for a sample .h5 file to build the template
        h5_file = input("Please enter the full path to a sample .h5 file to extract node names:\n> ").strip()
        
        # Clean up path formatting (remove quotes if user dragged-and-dropped file into terminal)
        h5_file = h5_file.strip('\'"')
        
        if os.path.exists(h5_file):
            create_config_template(h5_file, json_path)
        else:
            print("Error: The .h5 file path you provided does not exist. Exiting.")
            sys.exit(1)
    else:
        # If the file exists, execute the main analysis
        run_pipeline(json_path)