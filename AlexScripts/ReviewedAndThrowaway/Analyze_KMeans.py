#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 15:46:14 2025

@author: alex
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import inference_to_clusters_lib as itcl


df = pickle.load(open("DataframeKMeans.bin", "rb"))
print(df.head())

#plt.plot(range(len(df)), df["cluster"])
#plt.show()

processed_dict=itcl.ReformatToOriginal(df)
processed_dict["edge_inds"] = pickle.load(open("Skeleton_Edges.bin", "rb"))
for f_idx in range(len(processed_dict["locations"])):
    itcl.PlotSkeleton(processed_dict["locations"][f_idx], processed_dict)
    plt.show()

