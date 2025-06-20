#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:55:33 2025

@author: alex
"""

#%% Imports

import inference_to_clusters_lib as itcl
import numpy as np
import matplotlib.pyplot as plt

filename = r"/Users/alex/Downloads/ResultInference.hdf5"
# Process the hdf5 to a dictionary
new_dict = itcl.process_hdf5_data(filename)

#itcl.NodePositionsLocal(new_dict)

port_pos_list = np.array([[969, 1068.5, 974],
                          [330.5, 484.5, 641]])
overallAngles = np.zeros((port_pos_list.shape[1], len(new_dict["locations"])))

print(itcl.AngleToPorts(new_dict["locations"][0], new_dict, port_pos_list))


for frame_idx in range(len(new_dict["locations"])):
    itcl.PlotSkeleton(new_dict["locations"][frame_idx], new_dict) #REMOVE ANGLES MAKES IT UNUSABLE
    itcl.PlotPorts(port_pos_list)
    angles = itcl.AngleToPorts(new_dict["locations"][frame_idx], new_dict, port_pos_list)
    offset = [20,0]
    itcl.PlotAnglesToPorts(angles, port_pos_list, offset)
    plt.show()
    