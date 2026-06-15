#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 20:14:44 2026

@author: alex
"""

import subprocess
import sys
import os

def run_inference(video_list, write_path_list, model_path):
    """Launches the SLEAP inference pipeline, streaming real-time feedback to the console."""
    if not video_list:
        print("No videos provided for inference. Skipping.")
        return False

    print(f"\nLaunching SLEAP inference on {len(video_list)} videos...\n")
    print("=" * 50)
    
    for i in range(len(video_list)):
        command = []
        if(len(model_path)==2):
            centroid_model = model_path[0]
            centered_instance = model_path[1]
            if os.path.exists(centroid_model) and os.path.exists(centered_instance):
                command = ["sleap", "track", "-i", video_list[i], "-m", centroid_model, "-m", centered_instance, "-o", write_path_list[i], "--tracking"]
        elif len(model_path) == 1:
            single_instance_model = model_path[0]
            if os.path.exists(single_instance_model):
                command = ["sleap", "track", "-i", video_list[i], "-m", single_instance_model, "-o", write_path_list[i], "--tracking"]
        else:
            raise Exception("The path to the models does not exist. Please replace it with a valid path.")
        #Run the inference command
        try:
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
            else:
                print("=" * 50)
                print(f"Inference failed with exit code {process.returncode}.")
        except Exception as e:
            print(f"Failed to launch subprocess: {e}")
            return False
    print("\nAll videos processed!")
    return True