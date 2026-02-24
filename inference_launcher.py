#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 20:14:44 2026

@author: alex
"""

import subprocess
import sys
import os
import inference_config_setup as ics

def run_inference(video_list, write_path_list):
    """Launches the SLEAP inference pipeline, streaming real-time feedback to the console."""
    if not video_list:
        print("No videos provided for inference. Skipping.")
        return False

    config = ics.load_or_create_config()        
    conda_env_path = config.get("conda_env_path")

    print(f"\nLaunching SLEAP inference on {len(video_list)} videos...\n")
    print("=" * 50)
    
    single_model_path = config["single_model_path"]
    centroid_model_path = config["centroid_model_path"]
    centered_model_path = config["centered_model_path"]
    
    for i in range(len(video_list)):
        command = []
        if os.path.exists(single_model_path):
            #print(conda_env_path)
            #print(os.path.exists(conda_env_path))
            #print(video_list[i])
            #print(os.path.exists(video_list[i]))
            #print(single_model_path)
            #print(os.path.exists(single_model_path))
            #print(write_path_list[i])
            #print(os.path.exists(write_path_list[i]))
            command = ["conda", "run", "--no-capture-output", "-p", conda_env_path, "sleap-track", 
                       video_list[i], "-m", single_model_path, "-o", write_path_list[i]]
        elif os.path.exists(centered_model_path) and os.path.exists(centroid_model_path):
            command = ["conda", "run", "--no-capture-output", "-p", conda_env_path, "sleap-track", 
                       video_list[i], "-m", centroid_model_path, 
                       "-m", centered_model_path, "-o", write_path_list[i]]
        else:
            raise Exception("The path to the models does not exist. Please replace it with a valid path.")
        try:
            """#Create a text file at the location with each video's name
            write_base_without_ext, ext = os.path.splitext(os.path.basename(write_path_list[i]))
            text_file_path = os.path.join(os.path.dirname(write_path_list[i]), f"{write_base_without_ext}.txt")
            # Ensure the output directory exists before FFmpeg tries to write to it
            os.makedirs(os.path.dirname(text_file_path), exist_ok=True)
            print(text_file_path)
            with open(text_file_path, "a") as f:
                f.write("Now the file has more content!")
            #return True#TEMPORARY STOPGAP"""
            os.makedirs(os.path.dirname(write_path_list[i]), exist_ok=True)
            # Popen streams the output line-by-line
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, shell=True) as process:
                # Iterate through the output as it is generated and print to console
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    
            # Ensure the process is fully complete before checking the exit code
            process.wait()
                    
            if process.returncode == 0:
                print("=" * 50)
                print("Inference completed successfully!")
                return True
            else:
                print("=" * 50)
                print(f"Inference failed with exit code {process.returncode}.")
                return False
        except Exception as e:
            print(f"Failed to launch subprocess: {e}")
            return False