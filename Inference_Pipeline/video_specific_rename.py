# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:07:59 2026

@author: cns-th-lab
"""
#%%
import os
import datetime
video_dir = r"C:\Users\cns-th-lab\Tanner_Alex_Vids"
subj_folders = [f.path for f in os.scandir(video_dir) if f.is_dir()]
for subj_folder in subj_folders:
    print()
    video_folder_path = os.path.join(subj_folder,"Videos")
    for file in os.listdir(video_folder_path):
        file_path = os.path.join(video_folder_path, file)
        #dir_path = os.path.dirname(file_path)

        # 1. Get the modification time directly
        timestamp = os.path.getmtime(file_path)
    
        # 2. Convert date to a readable format
        last_modified_datetime = datetime.datetime.fromtimestamp(timestamp)    
        formatted_date = last_modified_datetime.date().isoformat()        
        
        #Rename with new naming convention [subjid.date.moviename.ext]
        subj_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        name_without_ext, ext = os.path.splitext(file_path)
        new_name = f"{subj_id}.{formatted_date}.{os.path.basename(name_without_ext)}{ext}"
        new_path = os.path.join(video_folder_path, new_name)
        print(new_path)
        #If name is mov0001.mp4, there is no longer a .
        #If name is subj_id.date-move0001.mp4 there are still periods
        #Make sure it is not renamed to something with more than one period
        if "." not in os.path.splitext(os.path.basename(file_path))[0]:
            print(f"Would have renamed to {new_path}")
            #os.rename(file_path, new_path)
        else:
            print(f"Filepath {file_path} skipped as it was already renamed.")
"""

#%%
for subj_folder in subj_folders:
    print()
    print(f"Videos subfolder: {os.path.join(subj_folder,'Videos')}")
    video_folder_path = os.path.join(subj_folder,"Videos")
    for file in os.listdir(video_folder_path):
        file_path = os.path.join(video_folder_path, file)
        path_root, ext = os.path.splitext(file_path)
        path_root, movie_name = os.path.splitext(path_root)
        path_root, date = os.path.splitext(path_root)
        path_root, subjid = os.path.splitext(path_root)
        new_path = f"{path_root}{movie_name}{ext}"
        print(new_path)
        os.rename(file_path, new_path)

        if file_path != new_path:
            os.rename(file_path, new_path)
        else:
            print(f"Filepath {file_path} skipped as it was already renamed.")"""