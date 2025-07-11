#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 13:23:22 2025

@author: alex
"""
import init
#from pyutils import file_select_ui as fsu
#from pyutils import cluster_utils
from sys_neuro_tools import sleap_utils
import json
import subprocess
import os
from pyutils import file_select_ui as fsui

#Settings for running inference
print("The following files need to be in the same folder: ")
print("vid_to_inference_main.py")
print("vid_to_inference_sub.py")
print("vid_to_inference_lib.py")
print("inference_paths.json")

subj_id=""
vid_path=""
#to_do: Replace with something that returns video path for given params

path_settings = "inference_paths.json"
#Setup instructions for changing setting file
new_model = False #Do you have a new centroid or centered model
change_python_loc = False #Do the environment's python location change
new_video = False #Upload new video (will be automated by entering subject+date)
new_write_loc = False #Do you want to write it to a new directory
json_exists = True #False if you deleted inference_paths or first run


sleap_utils.update_sleap_settings(path=None, new_model=new_model, change_python_loc = change_python_loc, new_video = new_video, new_write_loc = new_write_loc, json_exists = json_exists)

#Get python location from file
#home_dir = utils.get_user_home() #NEEDS TO BE C:\ NOT C:\\ ? FIX NECESSARY
script_path = fsui.GetFile("Select sleap_utils_env.py")#os.path.join(home_dir, "repos\python\sys_neuro_tools\sleap_utils_env.py")
env_python = sleap_utils.load_sleap_settings()["sleap_python"]
sleap_settings = sleap_utils.load_sleap_settings()
settings_path = sleap_utils.get_settings_path() #MAKE SURE TO SET THIS (JSON MANAGEMENT FUNCTION)

#sleap_utils_env.RunInference(vid_path, centroid_path, center_path, write_dir)
#Run inference with environment's python version
#command = [env_python, script_path, vid_path, single_path, centroid_path, center_path, write_dir]
command = [env_python, script_path, settings_path]
print(command)
try:
    result = subprocess.run(command, check=True, capture_output=True)
    print("STDOUT:", result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Subprocess failed with exit code {e.returncode}")
    print("STDERR:", e.stderr)