#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 12:48:46 2025

@author: alexandru
"""
#activate sleap before running
import sleap 
import tkinter as tk
from tkinter import filedialog

def RunInference(vid_path, centroid_path, centered_path, write_path):
    predictor = sleap.load_model([centroid_path, centered_path], batch_size=16)
    video = sleap.load_video(vid_path)
    print(video.shape, video.dtype)

    # Load frames
    imgs = video[:10]
    print(f"imgs.shape: {imgs.shape}")

    # Predict on nthe array.
    predictions = predictor.predict(imgs)
    predictions.export(write_path)
      
def InitializeTkinter():
    root = tk.Tk()
    root.withdraw()  # Hides the main window
    
def GetFile():
    path = filedialog.askopenfilename()
    if path:  # Check if a file was selected (user didn't cancel)
        print(f"Selected file path: {path}")
        return path
    else:
        return None
    
def GetDirectory():
    path = filedialog.askdirectory()
    if path:  # Check if a file was selected (user didn't cancel)
        print(f"Selected file path: {path}")
        return path
    else:
        return None
 
