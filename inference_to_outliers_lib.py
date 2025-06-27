#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 12:13:51 2025

@author: alex
"""
import numpy as np
import inference_to_clusters_lib as itcl
import file_select_ui as fsui
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

def VelocityOutlierDetection(locations, z_score=2):
    '''Calculates differences in position over time and marks any suspiciously 
    rapid movement.
    
    Parameters
    ---
    locations: locations data from hdf5 file(frames x nodes x (x,y) x 1)
    confidence: percentage of data that should be found within
    
    Returns
    ---
    flags: same shape with 0 for normal and 1 for outliers'''
    mask = np.zeros(np.shape(locations))
    print("Checking for outliers")
    for n_idx in range(locations.shape[1]):
        print(n_idx)
        x_list = locations[:,n_idx,0,0]
        y_list = locations[:,n_idx,1,0]
        x_list = pd.Series(x_list)
        y_list = pd.Series(y_list)
        
        #print("num nan raw")
        #print(np.sum(np.isnan(x_list)))
        #print(np.sum(np.isnan(y_list)))
        
        x_list = x_list.interpolate(method='linear')
        y_list = y_list.interpolate(method='linear')
        
        #print("num nan lin")
        #print(np.sum(np.isnan(x_list)))
        #print(np.sum(np.isnan(y_list)))


        
        x_diff = np.diff(x_list)
        y_diff = np.diff(y_list)
        
        x_std = np.std(x_diff)
        y_std = np.std((y_diff))
        #print("std")
        #print(x_std)
        #print(y_std)
        
        x_mean = np.mean(x_diff)
        y_mean = np.mean(y_diff)
        #print("Mean")
        #print(x_mean)
        #print(y_mean)
        
        #Creates bounds num_std standard deviations above/below the mean
        #Currently uses z_scores for outlier detection
        #Could use 1.5*IQR in the future if it works better
        x_bounds = [x_mean-x_std*z_score, x_mean+x_std*z_score]
        y_bounds = [y_mean-y_std*z_score, y_mean+y_std*z_score]
        
        #Checks those bounds at all points
        for frame_idx in range(1,locations.shape[0]):
            mask[frame_idx, n_idx, 0,0] = (x_diff[frame_idx-1] < x_bounds[0] or x_diff[frame_idx-1] > x_bounds[1])
            mask[frame_idx, n_idx, 1,0] = (y_diff[frame_idx-1] < y_bounds[0] or y_diff[frame_idx-1] > y_bounds[1])
    plt.imshow(mask[:,:,0,0], aspect='auto')
    plt.colorbar()
    plt.show()
    plt.imshow(mask[:,:,1,0], aspect='auto')
    plt.colorbar()
    plt.show()
    return mask

def GetDiff(vect, plot=False):
    '''Returns and plots the differences between neighboring entries of an array.
    Parameters
    ---
    vect: the vector to take a derivative of
    plot: should the returned vector be plotted? default:True
    Returns
    ---
    vect_diff: an array of differences between values (one less entry than vect)'''
    #Reshapes vect if it has a second dimension of one
    vect=np.reshape(vect, (np.shape(vect)[0],))
    print(np.shape(vect))
    #Removes nan by interpolation
    vect = pd.Series(vect)
    vect = vect.interpolate(method='linear')
    vect_diff = np.diff(vect)
    
    #Plots the vector if needed
    if plot:
        plt.plot(range(len(vect_diff)), vect_diff)
        plt.show()
        print(np.mean(vect))
        print(np.std(vect))
    return vect_diff

def PlotOutliers(frame_bounds=[0,60]):
    '''Takes an hdf5 file, processes it, and plots areas near outliers by velocity
    Additionally, it returns the total outliers in whole file
    Parameters
    ---
    frame_bounds: default parameter that lets you visualize a subset of one video
    
    Returns
    ---
    Number of outliers'''
    

