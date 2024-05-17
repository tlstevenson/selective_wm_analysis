# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:58:30 2023

@author: tanne
"""
import init

import pyutils.utils as utils
from sys_neuro_tools import plot_utils, fp_utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def get_all_processed_signals(raw_lig, raw_iso):
    ''' Gets all possible processed signals and intermediaries for the given raw signals '''
    
    # use isosbestic correction
    dff_iso, fitted_iso = fp_utils.calc_iso_dff(raw_lig, raw_iso)

    # use ligand-signal baseline with polynomial
    # nans = np.isnan(raw_lig)
    # baseline = np.full_like(raw_lig, np.nan)
    # baseline[~nans] = peakutils.baseline(raw_lig[~nans])

    # dff_baseline = ((raw_lig - baseline)/baseline)*100

    # use ligand-signal baseline with linear & exponential decay function to approximate photobleaching
    baseline_lig = fp_utils.fit_baseline(raw_lig)
    dff_baseline = ((raw_lig - baseline_lig)/baseline_lig)*100

    # use baseline and iso together to calculate fluorescence residual (dF instead of dF/F)

    # first subtract the baseline fit to each signal to correct for photobleaching
    baseline_corr_lig = raw_lig - baseline_lig

    baseline_iso = fp_utils.fit_baseline(raw_iso)
    baseline_corr_iso = raw_iso - baseline_iso

    # scale the isosbestic signal to best fit the ligand-dependent signal
    fitted_baseline_iso = fp_utils.fit_signal(baseline_corr_iso, baseline_corr_lig)

    # then use the baseline corrected signals to calculate dF, which is a residual fluorescence
    df_baseline_iso = baseline_corr_lig - fitted_baseline_iso

    return {'raw_lig': raw_lig,
            'raw_iso': raw_iso,
            'baseline_lig': baseline_lig,
            'baseline_iso': baseline_iso,
            'baseline_corr_lig': baseline_corr_lig,
            'baseline_corr_iso': baseline_corr_iso,
            'fitted_iso': fitted_iso,
            'dff_iso': dff_iso,
            'z_dff_iso': utils.z_score(dff_iso),
            'dff_baseline': dff_baseline,
            'z_dff_baseline': utils.z_score(dff_baseline),
            'df_baseline_iso': df_baseline_iso,
            'z_df_baseline_iso': utils.z_score(df_baseline_iso)}


def view_processed_signals(processed_signals, t, dec=10, title='Full Signals', vert_marks=[], 
                           filter_outliers=False, outlier_zthresh=10, t_min=0, t_max=np.inf):

    if utils.is_dict(list(processed_signals.values())[0]):
        n_panel_stacks = len(processed_signals.values())
    else:
        n_panel_stacks = 1
        # make a temporary outer dictionary for ease of use with for loop
        processed_signals = {'temp': processed_signals}
        
    t = t[::dec]
    
    t_min_idx = np.argmax(t > t_min)
    t_max_idx = np.argwhere(t < t_max)[-1,0]

    # plot the raw signals and their baseline fits, baseline corrected signals, raw ligand and fitted iso, dff and baseline corrected df
    fig, axs = plt.subplots(2*n_panel_stacks, 2, layout='constrained', figsize=[20,6*n_panel_stacks])
    plt.suptitle(title)
    
    vert_marks = vert_marks[(vert_marks > t_min) & (vert_marks < t_max)]
    
    for i, (sub_key, sub_signals) in enumerate(processed_signals.items()):
        
        # remove outliers in z-score space
        filt_sel = np.full(t.shape, True)
        if filter_outliers:
            # filter based on raw signals
            z_lig = utils.z_score(sub_signals['raw_lig'][::dec])
            z_iso = utils.z_score(sub_signals['raw_iso'][::dec])
            filt_sel = filt_sel & (np.abs(z_lig) < outlier_zthresh)
            filt_sel = filt_sel & (np.abs(z_iso) < outlier_zthresh)
        
        # repurpose outlier filter for time filter as well
        filt_sel[:t_min_idx] = False
        filt_sel[t_max_idx:] = False
            
        filt_t = t[filt_sel]
        
        gen_sub_title = sub_key + ' {}' if sub_key != 'temp' else '{}'

        # plot raw signals and baseline
        ax = axs[i,0]
        color1 = next(ax._get_lines.prop_cycler)['color']
        color2 = next(ax._get_lines.prop_cycler)['color']
        l3 = ax.plot(filt_t, sub_signals['raw_iso'][::dec][filt_sel], label='Raw Iso', color=color2, alpha=0.5)
        l4 = ax.plot(filt_t, sub_signals['baseline_iso'][::dec][filt_sel], '--', label='Iso Baseline', color=color2)
        l1 = ax.plot(filt_t, sub_signals['raw_lig'][::dec][filt_sel], label='Raw Lig', color=color1, alpha=0.5)
        l2 = ax.plot(filt_t, sub_signals['baseline_lig'][::dec][filt_sel], '--', label='Lig Baseline', color=color1)
        plot_utils.plot_dashlines(vert_marks, ax=ax)
        ax.set_title(gen_sub_title.format('Raw Signals'))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescent Signal (V)')
        ls = [l1[0], l2[0], l3[0], l4[0]]
        labs = [l.get_label() for l in ls]
        ax.legend(ls, labs)
        
        # plot baseline corrected signals
        ax = axs[i,1]
        l2 = ax.plot(filt_t, sub_signals['baseline_corr_iso'][::dec][filt_sel], label='Baseline Corrected Iso', color=color2, alpha=0.5)
        l1 = ax.plot(filt_t, sub_signals['baseline_corr_lig'][::dec][filt_sel], label='Baseline Corrected Lig', color=color1, alpha=0.5)
        plot_utils.plot_dashlines(vert_marks, ax=ax)
        ax.set_title(gen_sub_title.format('Baseline Subtracted Signals'))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescent Signal (dV)')
        ls = [l1[0], l2[0]]
        labs = [l.get_label() for l in ls]
        ax.legend(ls, labs)
        
        # plot raw ligand and fitted iso
        ax = axs[n_panel_stacks+i,0]
        l2 = ax.plot(filt_t, sub_signals['fitted_iso'][::dec][filt_sel], label='Fitted Iso', color=color2, alpha=0.5)
        l1 = ax.plot(filt_t, sub_signals['raw_lig'][::dec][filt_sel], label='Raw Lig', color=color1, alpha=0.5)
        plot_utils.plot_dashlines(vert_marks, ax=ax)
        ax.set_title(gen_sub_title.format('Iso dF/F Signals'))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescent Signal (V)')
        ls = [l1[0], l2[0]]
        labs = [l.get_label() for l in ls]
        ax.legend(ls, labs)
        
        # plot iso dFF and baseline corrected dF
        ax = axs[n_panel_stacks+i,1]
        ax2 = ax.twinx()
        l2 = ax2.plot(filt_t, sub_signals['df_baseline_iso'][::dec][filt_sel], label='Baseline Corrected dF', color=color2, alpha=0.5)
        l1 = ax.plot(filt_t, sub_signals['dff_iso'][::dec][filt_sel], label='Iso dF/F', color=color1, alpha=0.5)
        plot_utils.plot_dashlines(vert_marks, ax=ax)
        ax.set_title(gen_sub_title.format('Iso Corrected Ligand Signals'))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dF/F')
        ax2.set_ylabel('dF')
        ax.tick_params(axis='y', labelcolor=color1)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.zorder = 1
        ax.zorder = 2
        ax.set_frame_on(False)
        
        ls = [l1[0], l2[0]]
        labs = [l.get_label() for l in ls]
        ax.legend(ls, labs)
        

def view_signal(processed_signals, t, signal_type, title=None, dec=10, vert_marks=[], 
                filter_outliers=False, outlier_zthresh=10, t_min=0, t_max=np.inf, figsize=None,
                ylabel=''):
    
    if utils.is_dict(list(processed_signals.values())[0]):
        n_panel_stacks = len(processed_signals.values())
    else:
        n_panel_stacks = 1
        # make a temporary outer dictionary for ease of use with for loop
        processed_signals = {'temp': processed_signals}
        
    t = t[::dec]
    
    t_min_idx = np.argmax(t > t_min)
    t_max_idx = np.argwhere(t < t_max)[-1,0]
    
    if title is None:
        title = signal_type
    
    if figsize is None:
        figsize = (7, 3*n_panel_stacks)
    
    _, axs = plt.subplots(n_panel_stacks, 1, layout='constrained', figsize=figsize)
    plt.suptitle(title)

    if len(vert_marks) > 0:
        vert_marks = vert_marks[(vert_marks > t_min) & (vert_marks < t_max)]

    for i, (sub_key, sub_signals) in enumerate(processed_signals.items()):
        # remove outliers in z-score space
        filt_sel = np.full(t.shape, True)
        if filter_outliers:
            # filter based on raw signals
            z_sig = utils.z_score(sub_signals[signal_type][::dec])
            filt_sel = filt_sel & (np.abs(z_sig) < outlier_zthresh)
        
        # repurpose outlier filter for time filter as well
        filt_sel[:t_min_idx] = False
        filt_sel[t_max_idx:] = False
            
        filt_t = t[filt_sel]
    
        # plot raw signals and baseline
        ax = axs[i]
        ax.plot(filt_t, sub_signals[signal_type][::dec][filt_sel])
        plot_utils.plot_dashlines(vert_marks, ax=ax)
        ax.set_title(sub_key)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
    

def plot_aligned_signals(signal_dict, t, title, sub_titles_dict, x_label, y_label,
                         cmap = 'viridis', outlier_thresh=10, trial_markers=None):
    ''' Plot aligned signals in the given nested dictionary where the first set of keys define the rows and the
        second set of keys in the dictionaries indexed by the first set of keys define the columns

        In each 'cell' will plot a heatmap of the aligned signals above an average signal trace
    '''

    outer_keys = [k for k in signal_dict.keys() if k != 't']
    n_rows = len(outer_keys)
    n_cols = len(signal_dict[outer_keys[0]])

    fig, axs = plt.subplots(n_rows*2, n_cols, height_ratios=np.tile([3,2], n_rows),
                            figsize=(5*n_cols, 4*n_rows), layout='constrained')

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
            hm_ax = axs[i*2, j]
            hm, _ = plot_utils.plot_stacked_heatmap_avg(signal, t, hm_ax, axs[i*2+1, j],
                                             x_label=x_label, y_label=y_label,
                                             title='{} {}'.format(key, sub_titles_dict[name]),
                                             show_cbar=False, cmap=cmap, vmax=max_act, vmin=min_act)

            # plot any horizontal trial markers
            if not trial_markers is None:
                if name in trial_markers:
                    for t_idx in trial_markers[name]:
                        hm_ax.axhline(t_idx - 0.5, dashes=[4, 4], c='k', lw=1)

        # share y axis for all average rate plots
        # find largest y range
        min_y = np.min([ax.get_ylim()[0] for ax in axs[i*2+1, :]])
        max_y = np.max([ax.get_ylim()[1] for ax in axs[i*2+1, :]])
        for j in range(axs.shape[1]):
            axs[i*2+1, j].set_ylim(min_y, max_y)

        # create color bar legend for heatmap plots
        fig.colorbar(hm, ax=axs[i*2,:].ravel().tolist(), label=y_label)
    
        
def get_signal_type_labels(signal_type):
    ''' Get signal titles and labels based on the type of signal '''
    match signal_type: 
        case 'dff_iso':
            title = 'Isosbestic Normalized'
            ax_label = 'dF/F'
        case 'z_dff_iso':
            title = 'Isosbestic Normalized'
            ax_label = 'z-scored dF/F'
        case 'dff_baseline':
            title = 'Baseline Normalized'
            ax_label = 'dF/F'
        case 'z_dff_baseline':
            title = 'Baseline Normalized'
            ax_label = 'z-scored dF/F'
        case 'df_baseline_iso':
            title = 'Baseline-Subtracted Isosbestic Residuals'
            ax_label = 'dF residuals'
        case 'z_df_baseline_iso':
            title = 'Baseline-Subtracted Isosbestic Residuals'
            ax_label = 'z-scored dF residuals'
        case 'baseline_corr_lig':
            title = 'Baseline-Subtracted Ligand Signal'
            ax_label = 'dF'
        case 'baseline_corr_iso':
            title = 'Baseline-Subtracted Isosbestic Signal'
            ax_label = 'dF'
        case 'raw_lig':
            title = 'Raw Ligand Signal'
            ax_label = 'dF/F' if trial_normalize else 'V'
        case 'raw_iso':
            title = 'Raw Isosbestic Signal'
            ax_label = 'dF/F' if trial_normalize else 'V'
            
    return title, ax_label


def plot_power_spectra(signal, dt, f_max=40, title=''):
    #signal = signal - np.mean(signal)
    # ps = np.abs(np.fft.rfft(signal))**2
    # freqs = np.fft.rfftfreq(signal.size, dt)
    
    freqs, ps = sig.welch(signal, fs = 1/dt, nperseg = round(1/dt)*60, scaling='spectrum')
    
    #freqs, ps = sig.periodogram(signal, fs = 1/dt, scaling='spectrum')
    # same as FFT but simpler

    _, ax = plt.subplots(1)
    ax.plot(freqs, ps)
    ax.set_yscale('log')
    #ax.set_xscale('log')
    #ax.set_xlim([0.01, f_max])
    ax.set_xlim([1, f_max])
    ax.set_title(title)

    
def remove_outliers(mat, outlier_thresh):
    mat[np.abs(mat) > outlier_thresh] = np.nan
        
    return mat


def stack_fp_mats(mat_dict, regions, sess_ids, subjects, signal_type, filter_outliers=False, outlier_thresh=20, groups=None):
    stacked_mats = {region: {} for region in regions}
    if groups is None:
        groups = mat_dict[sess_ids[subjects[0]][0]][signal_type][regions[0]].keys()

    for region in regions:
        for group in groups:
            for subj_id in subjects:
                mat = np.vstack([mat_dict[sess_id][signal_type][region][group] for sess_id in sess_ids[subj_id]])
                if filter_outliers:
                    mat = remove_outliers(mat, outlier_thresh)
                    
                stacked_mats[region][group] = mat

    return stacked_mats


# def combine_group_mats(data_dict, grouping=None):
#     if grouping is None:
#         grouping = {'all': list(data_dict[regions[0]].keys())}
        
#     for region in regions:
#         for name, groups in grouping.items():
#             data_dict[region][name] = np.vstack([data_dict[region][group] for group in groups])
        
#     return data_dict