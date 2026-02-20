#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 20:14:44 2026

@author: alex
"""

import subprocess
import json

import subprocess
import json
import sys

def run_inference(config_path, video_list):
    """Launches the SLEAP inference pipeline, streaming real-time feedback to the console."""
    if not video_list:
        print("No videos provided for inference. Skipping.")
        return False

    with open(config_path, 'r') as file:
        config = json.load(file)
        
    conda_env_path = config.get("conda_env_path")
    script_path = config.get("inference_script_path")

    # Command structure: conda run -p <env> python -u <script> <config> <video1> <video2> ...
    command = [
        "conda", "run", "-p", conda_env_path, 
        "python", "-u", script_path, config_path
    ] + video_list

    print(f"\nLaunching SLEAP inference on {len(video_list)} videos...\n")
    print("=" * 50)
    
    try:
        # Popen streams the output line-by-line
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
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