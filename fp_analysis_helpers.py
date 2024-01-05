# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:58:30 2023

@author: tanne
"""
import init

import pyutils.utils as utils
from sys_neuro_tools import plot_utils
import numpy as np
import matplotlib.pyplot as plt

def plot_aligned_signals(signal_dict, title, sub_titles_dict, x_label, y_label, cmap = 'viridis', outlier_thresh=10):
    ''' Plot aligned signals in the given nested dictionary where the first set of keys define the rows and the
        second set of keys in the dictionaries indexed by the first set of keys define the columns

        In each 'cell' will plot a heatmap of the aligned signals above an average signal trace
    '''

    outer_keys = [k for k in signal_dict.keys() if k != 't']
    n_rows = len(outer_keys)
    n_cols = len(signal_dict[outer_keys[0]])
    t = signal_dict['t']

    fig, axs = plt.subplots(n_rows*2, n_cols, height_ratios=np.tile([3,2], n_rows),
                            figsize=(4*n_cols, 4*n_rows), layout='constrained')

    plt.suptitle(title)

    for i, key in enumerate(outer_keys):

        # remove outliers in z-score space
        z_signals = [utils.z_score(signal) for signal in signal_dict[key].values() if len(signal) > 0]
        signals_no_outliers = {}
        for j, (name, signal) in enumerate(signal_dict[key].items()):
            if len(signal) > 0:
                signal_no_outliers = signal.copy()
                signal_no_outliers[np.abs(z_signals[j]) > outlier_thresh] = np.nan
                signals_no_outliers[name] = signal_no_outliers
            else:
                signals_no_outliers[name] = signal

        max_act = np.max([np.max(signal[~np.isnan(signal)]) for signal in signals_no_outliers.values() if len(signal) > 0])
        min_act = np.min([np.min(signal[~np.isnan(signal)]) for signal in signals_no_outliers.values() if len(signal) > 0])

        # plot average signals first to get the x axis labels for the heatmap
        for j, (name, signal) in enumerate(signals_no_outliers.items()):
            im, _ = plot_utils.plot_stacked_heatmap_avg(signal, t, axs[i*2, j], axs[i*2+1, j],
                                             x_label=x_label, y_label=y_label,
                                             title='{} {}'.format(key, sub_titles_dict[name]),
                                             show_cbar=False, cmap=cmap, vmax=max_act, vmin=min_act)

        # share y axis for all average rate plots
        # find largest y range
        min_y = np.min([ax.get_ylim()[0] for ax in axs[i*2+1, :]])
        max_y = np.max([ax.get_ylim()[1] for ax in axs[i*2+1, :]])
        for j in range(axs.shape[1]):
            axs[i*2+1, j].set_ylim(min_y, max_y)

        # create color bar legend for heatmap plots
        fig.colorbar(im, ax=axs[i*2,:].ravel().tolist(), label=y_label)
