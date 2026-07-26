# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:24:18 2026

@author: cns-th-lab

MUST RUN IN NEW ENVIRONMENT: conda env config vars set MPLBACKEND=Agg -n disk 
"""

import subprocess
import init
import file_select_ui as fsui
import sys
import os
import random

#%%Get all directories
use_list = True
#%%
analysis_dir_list = [r"C:/Users/cns-th-lab/TannerVidsRenamed/198/Videos/predictions/260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
                     r"C:/Users/cns-th-lab/TannerVidsRenamed/199/Videos/predictions/260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
                     r"C:/Users/cns-th-lab/TannerVidsRenamed/237/Videos/predictions/260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
                     r"C:/Users/cns-th-lab/TannerVidsRenamed/238/Videos/predictions/260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
                     r"C:/Users/cns-th-lab/TannerVidsRenamed/274/Videos/predictions/260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
                     r"C:/Users/cns-th-lab/TannerVidsRenamed/400/Videos/predictions/260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
                     r"C:/Users/cns-th-lab/TannerVidsRenamed/402/Videos/predictions/260523_198_199x_237x_238x_274x_400x_402x_424x_483x", 
                     r"C:/Users/cns-th-lab/TannerVidsRenamed/424/Videos/predictions/260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
                     r"C:/Users/cns-th-lab/TannerVidsRenamed/483/Videos/predictions/260523_198_199x_237x_238x_274x_400x_402x_424x_483x"]

if not use_list:
    my_ans = "y"
    while my_ans != "n":
        new_dir = fsui.GetDirectory("Select Label Parent Folder")
        if new_dir != None:
            analysis_dir_list.append(new_dir)
        my_ans = input("Would you like to select another directory? (y/n)")
#%% All files    
analysis_path_list = [os.path.join(my_dir,h5_name) 
                      for my_dir in analysis_dir_list for h5_name in os.listdir(my_dir) 
                      if os.path.splitext(h5_name)[1]==".h5"]
#%% Random sample without replacement
analysis_path_list = []
for my_dir in analysis_dir_list:
    subfiles = os.listdir(my_dir)
    valid_files = [os.path.join(my_dir, filename) for filename in subfiles if os.path.splitext(filename)[1]==".h5"]
    chosen_files = random.sample(valid_files, 5)
    analysis_path_list = analysis_path_list + chosen_files
#TODO: All h5 files one level under parent
#%%
def ErrorPronePrintableCommand(command, working_directory):
    #Try to run the command
    try:
        # Popen streams the output line-by-line
        with subprocess.Popen(command, stdin=subprocess.PIPE, 
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=1, 
                              shell=True, cwd=working_directory) as process:
            try:
                process.stdin.write("n\n")
                process.stdin.flush()
            except Exception as e:
                print(f"Could not send input: {e}")
            #stdout, stderr = process.communicate(input=create_skeleton)
            # Iterate through the output as it is generated and print to console
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
            
            # Ensure the process is fully complete before checking the exit code
            process.wait()
            if process.returncode == 0:
                print("=" * 50)
                print("Command completed successfully!")
                return True
            else:
                print("=" * 50)
                print(f"Command failed with exit code {process.returncode}.")
                return False
    except Exception as e:
        print(f"Failed to launch subprocess: {e}")
        return False
#%% Project creation
project_path = r"C:\Users\cns-th-lab\FullTannerVidsDataset"
programs_folder = r"C:\Users\cns-th-lab\Github_Repos\selective_wm_analysis"
#%%
dataset_name = "dataset_length30_stride15"
#%%
model_name = "dataset_30_15_DISK"
env_name = r"C:\Users\cns-th-lab\AppData\Local\miniconda3\envs\disk"
command_create = ["DISK-create-project", "--project_path", project_path, "--file_format", "sleap_h5", "--data_files"]
command_create = ["conda", "run", "-p", env_name, "--no-capture-output"] + command_create + analysis_path_list
concat_command = ""
for part in command_create:
    concat_command = concat_command + part
print(len(concat_command))
ErrorPronePrintableCommand(command_create, programs_folder)

#%% Prepare Dataset for DISK model
# prepare a dataset for DISK model with a given sample length
length_segment = 30
command_prep = ["DISK-prepare-data", "--project_path", project_path, "--length", f"{length_segment}"]
command_prep = ["conda", "run", "-p", env_name, "--no-capture-output"] + command_prep
print(command_prep)
ErrorPronePrintableCommand(command_prep, programs_folder)

#%% Train a DISK model on the previsouly created datase
command_train = ["DISK-train", "--project_path", project_path, "--dataset_name", dataset_name, "--training_batch_size", 16]
command_train = ["conda", "run", "-p", env_name, "--no-capture-output"] + command_train
ErrorPronePrintableCommand(command_train, programs_folder)

#%% Use model for imputation
command_impute = ["DISK-impute", "--project_path", project_path, "--dataset_name", dataset_name, "--model_name", model_name]
command_impute = ["conda", "run", "-p", env_name, "--no-capture-output"] + command_impute
ErrorPronePrintableCommand(command_train, programs_folder)