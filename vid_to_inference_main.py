#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 13:23:22 2025

@author: alex
"""
import init
import file_select_ui as fsu
from pyutils import cluster_utils
import json
import subprocess
import os

#Settings for running inference
print("The following files need to be in the same folder: ")
print("vid_to_inference_main.py")
print("vid_to_inference_sub.py")
print("vid_to_inference_lib.py")
print("inference_paths.json")

path_settings = "inference_paths.json"
#Setup instructions for changing setting file
new_model = False #Do you have a new centroid or centered model
change_python_loc = True #Do the environment's python location change
new_video = False #Upload new video (will be automated by entering subject+date)
new_write_loc = False #Do you want to write it to a new directory
json_exists = False #False if you deleted inference_paths or first run


data = {}
if json_exists:
    #Read current path settings
    try:
        with open(path_settings, "r") as file:
            data = json.load(file)
            print(data)
    except Exception as e:
        print(e)
    #Make necessary edits
    if new_model:
        #Print because directory dialogues dont have titles
        print("Select the Centroid Model Parent Directory")
        data["centroid_path"] = fsu.GetDirectory("Select the Centroid Model Parent Directory")
        print("Select the Center Model Parent Directory")
        data["center_path"] = fsu.GetDirectory("Select the Center Model Parent Directory")
    if new_write_loc:
        print("Select Analysis File Write Directory")
        data["write_dir"] = fsu.GetDirectory("Select Analysis File Write Directory") + "/" + input("Name file(no .hdf5): ") + ".hdf5"
    else:
        print("Select Analysis File Write Directory")
        data["write_dir"] = data["write_dir"][:data["write_dir"].rfind("/")] + "/" + input("Name file(no .hdf5): ") + ".hdf5"        
    if change_python_loc:
        data["sleap_python"] = fsu.GetFile("Select SLEAP Python Location")
        #data["sleap_python"] = fsu.GetDirectory("Select SLEAP Python Location") + "/python"
        print("Changing python location")
    if new_video:
        print("Adding new video")
        data["vid_path"] = fsu.GetFile("Select Video Location")
    #Push to file
    try:
        with open(path_settings, "w") as file:
            json.dump(data, file)
    except Exception as e:
        print(e)
else:
    #Add all entries
    print("Select the Centroid Model Parent Directory")
    data["centroid_path"] = fsu.GetDirectory("Select the Centroid Model Parent Directory")
    print("Select the Center Model Parent Directory")
    data["center_path"] = fsu.GetDirectory("Select the Center Model Parent Directory")
    print("Select Analysis File Write Directory")
    data["write_dir"] = fsu.GetDirectory("Select Analysis File Write Directory") + "/" + input("Name file(no .hdf5): ") + ".hdf5"
    print(data["write_dir"])
    data["sleap_python"] = fsu.GetFile("Select SLEAP Python Location")
    #data["sleap_python"] = fsu.GetDirectory("Select SLEAP Python Location") + "/python"
    data["vid_path"] = fsu.GetFile("Select Video Location")
    #Push to file
    try:
        with open(path_settings, "w") as file:
            json.dump(data, file)
    except Exception as e:
        print(e)

#Get python location from file
script_path = "vid_to_inference_sub.py"
env_python = None
try:
    with open(path_settings, "r") as file:
        data = json.load(file)
        env_python = data["sleap_python"]
except Exception as error:
    print(error)
print(env_python)
print(script_path)
settings_path = "" #MAKE SURE TO SET THIS (JSON MANAGEMENT FUNCTION)
#Run inference with environment's python version
command = [env_python, script_path, settings_path]
try:
    subprocess.run(command, check=True)
except Exception as e:
    print(e)