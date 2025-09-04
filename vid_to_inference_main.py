#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 13:23:22 2025

@author: alex

READ ME: Before running this script, make sure all the videos you need to run 
are all in the same directory. Make sure you ran sleap_vid_reformat.py with that
directory. The naming convention (all reformatted videos end in _r.mp4) is
necessary. 
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

path_settings = "inference_paths.json"
#Setup instructions for changing setting file
new_model = False #Do you have a new centroid or centered model
change_python_loc = False #Do the environment's python location change
new_video_dir = False #Upload new video (will be automated by entering subject+date)
new_write_dir = False #Do you want to write it to a new directory
json_exists = True #False if you deleted inference_paths or first run
changed_script_loc = False #Did you move sleap_utils_env.py and need to reset

#Rewrites the json file with new settings
sleap_utils.update_sleap_settings(path=None, new_model=new_model, 
                                  change_python_loc = change_python_loc, 
                                  new_vid_dir = new_video_dir, 
                                  new_write_dir = new_write_dir, 
                                  json_exists = json_exists, 
                                  changed_script_loc=changed_script_loc)
print("SLEAP SETTINGS UPDATED: Turn everything except json_exists back to false")

#Get python location from file
#home_dir = utils.get_user_home() #NEEDS TO BE C:\ NOT C:\\ ? FIX NECESSARY
sleap_settings = sleap_utils.load_sleap_settings()
env_python = sleap_settings["sleap_python"]
script_path = sleap_settings["script_loc"]
settings_path = sleap_utils.get_settings_path() #MAKE SURE TO SET THIS (JSON MANAGEMENT FUNCTION)

#sleap_utils_env.RunInference(vid_path, centroid_path, center_path, write_dir)
#Run inference with environment's python version
#command = [env_python, script_path, vid_path, single_path, centroid_path, center_path, write_dir]
#command = [env_python, script_path, settings_path]
#print(command)
#commands_string = f"'conda activate sleap'; 'python {script_path} {settings_path}'"
#print(commands_string)
try:
    #setup_command = "set QT_API=pyside2" #Windows
    #export QT_API=pyside2 #Mac/Linux
    #setup = subprocess.run(setup_command, check=True, capture_output=True)
    #result = subprocess.run(command, check=True, capture_output=True)
    #result = subprocess.run(commands_string, shell=True, check=True, capture_output=True)
    result = subprocess.run(["conda", "run", "-n", "sleapUpdated", "python", script_path, settings_path], check=True)
    print("STDOUT:", result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Subprocess failed with exit code {e.returncode}")
    if e.stderr:
        print("STDERR:", e.stderr.decode("utf-8"))
    else:
        print("No STDERR output.")