#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 12:48:46 2025

@author: alexandru
"""
#activate sleap before running
import sleap 
#Placeholders (Eventually read from the vector processing script)
predictor = sleap.load_model(["centroid_model.zip", "centered_instance_id_model.zip"], batch_size=16)
file_loc = "~\..\..\repos\hankslabdb\___.mov" #path relative to the sleap conda environment
video = sleap.load_video(file_loc)
print(video.shape, video.dtype)

# Load frames
imgs = video[:100]
print(f"imgs.shape: {imgs.shape}")

# Predict on nthe array.
predictions = predictor.predict(imgs)
try:
    predictions.export("ResultInference.hdf5")
except:
    predictions.export("ResultInference")
