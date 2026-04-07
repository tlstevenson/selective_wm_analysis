#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:47:09 2026

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import keypoint_moseq as kpms

def get_syllable_transitions(moseq_df):
    """
    Finds all syllable transitions in the dataset.
    
    Returns:
    A dictionary mapping recording names to a list of transitions.
    Each transition is structured as: (frame_index, (prev_syllable, current_syllable))
    """
    print("Extracting syllable transitions...")
    transitions_dict = {}
    
    # Ensure dataframe is sorted by recording and time
    df = moseq_df.sort_values(by=['name', 'frame_index']).copy()
    
    # Shift the syllable column by 1 within each recording to find the previous syllable
    df['prev_syllable'] = df.groupby('name')['syllable'].shift(1)
    
    # A transition occurs where prev_syllable is not NaN and differs from the current syllable
    is_transition = (df['prev_syllable'].notna()) & (df['syllable'] != df['prev_syllable'])
    transitions_df = df[is_transition]
    
    # Package into the requested data structure
    for name, group in transitions_df.groupby('name'):
        trans_list = []
        for _, row in group.iterrows():
            frame = int(row['frame_index'])
            prev_syll = int(row['prev_syllable'])
            curr_syll = int(row['syllable'])
            trans_list.append((frame, (prev_syll, curr_syll)))
            
        transitions_dict[name] = trans_list
        
    return transitions_dict


def run_analysis(project_dir, model_name):
    print(f"Starting analysis for project: {project_dir} | model: {model_name}")
    
    # Ensure save directory exists
    save_dir = os.path.join(project_dir, model_name, "analysis_outputs")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Generate moseq_df (Frame-by-frame kinematics and syllables)
    print("Computing moseq_df...")
    moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)
    moseq_df_path = os.path.join(save_dir, 'moseq_df.csv')
    moseq_df.to_csv(moseq_df_path, index=False)
    print(f"Saved moseq_df to {moseq_df_path}")

    # 2. Extract transitions using the custom function
    transitions = get_syllable_transitions(moseq_df)
    
    # Save the transitions to a JSON file for easy loading later
    transitions_path = os.path.join(save_dir, 'syllable_transitions.json')
    with open(transitions_path, 'w') as f:
        json.dump(transitions, f, indent=4)
    print(f"Saved custom syllable transitions to {transitions_path}")

    # 3. Generate stats_df (Summary statistics for each syllable)
    print("Computing stats_df...")
    # NOTE: If your JSON used a different FPS, make sure to update `fps=30` below.
    stats_df = kpms.compute_stats_df(
        project_dir, 
        model_name, 
        moseq_df, 
        min_frequency=0.005, 
        groupby=["group", "name"], 
        fps=30 
    )
    stats_df_path = os.path.join(save_dir, 'stats_df.csv')
    stats_df.to_csv(stats_df_path, index=False)
    print(f"Saved stats_df to {stats_df_path}")

    # 4. Generate Transition Matrices
    print("Computing and visualizing transition matrices...")
    normalize = "bigram" 
    trans_mats, usages, groups, syll_include = kpms.generate_transition_matrices(
        project_dir, 
        model_name, 
        normalize=normalize, 
        min_frequency=0.005
    )
    
    # Heatmap of transition bigrams
    kpms.visualize_transition_bigram(
        project_dir, 
        model_name, 
        groups, 
        trans_mats, 
        syll_include, 
        normalize=normalize, 
        show_syllable_names=True
    )
    
    # 5. Generate Syllable Transition Graphs
    print("Plotting transition graphs...")
    # Render transition rates in graph form (nodes = syllables, edges = transitions)
    kpms.plot_transition_graph_group(
        project_dir, 
        model_name, 
        groups, 
        trans_mats, 
        usages, 
        syll_include, 
        layout="circular", 
        show_syllable_names=True 
    )

    print("\nAnalysis complete! Figures have been saved to your model's 'figures' folder.")
    print(f"Dataframes and custom transition dicts are saved in: {save_dir}")


if __name__ == "__main__":
    # You can hardcode these, or pass them in via the command line.
    # To use command line: python analyze_kpms.py /path/to/project your_model_name
    
    if len(sys.argv) >= 3:
        PROJECT_DIR = sys.argv[1]
        MODEL_NAME = sys.argv[2]
    else:
        # Fallback to reading the config.json we made in the previous step
        config_path = "config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config_data = json.load(file)
            PROJECT_DIR = config_data.get("project_dir")
            
            # Keypoint MoSeq outputs models with a timestamp name (e.g. 2024_03_05-14_30_00).
            # If you know the name, enter it here. Otherwise, the script attempts to find the most recent one.
            models = [d for d in os.listdir(PROJECT_DIR) if os.path.isdir(os.path.join(PROJECT_DIR, d)) and d.startswith("20")]
            if models:
                models.sort() # Sorts by timestamp naturally
                MODEL_NAME = models[-1] # Grabs the most recent model
                print(f"Auto-detected most recent model: {MODEL_NAME}")
            else:
                print("Could not automatically find a model directory. Please provide it as an argument.")
                sys.exit(1)
        else:
            print("Usage: python analyze_kpms.py <project_dir> <model_name>")
            sys.exit(1)
            
    run_analysis(PROJECT_DIR, MODEL_NAME)