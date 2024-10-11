# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:16:38 2024

@author: tanne
"""

# %% imports

import init
from hankslab_db import db_access
import hankslab_db.basicRLtasks_db as rl_db
import hankslab_db.tonecatdelayresp_db as wm_db
from pyutils import utils
import numpy as np
import fp_analysis_helpers as fpah
from fp_analysis_helpers import Alignment as Align
from sys_neuro_tools import plot_utils, fp_utils
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import pandas as pd
import pickle
import os.path as path
import seaborn as sb
from scipy import stats
import copy


# %% Declare subject information

# declare the variant names and subject ids for comparison
variant_subj = {'3.6 CAG': [182, 202], '3.8 CAG': [179, 180, 188], '3.8 Syn': [191, 207]}
subj_variant = {v: k for k, vs in variant_subj.items() for v in vs}
subj_ids = utils.flatten(variant_subj)

wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
rl_loc_db = rl_db.LocalDB_BasicRLTasks()

protocol_db = {'ClassicRLTasks': rl_loc_db, 'ToneCatDelayResp': wm_loc_db, 'ToneCatDelayResp2': wm_loc_db}

sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)
sess_info = db_access.get_sess_protocol_stage(utils.flatten(sess_ids))
# update stage number for rat 188 since different numbers are used for the same stage due to adding a stage in the middle of recording
sess_info.loc[(sess_info['subjid'] == 188) & (sess_info['protocol'] == 'ToneCatDelayResp2'), 'startstage'] = 9
sess_info['proto_stage'] = sess_info.apply(lambda x: '{}_{}'.format(x['protocol'], x['startstage']), axis=1)

sess_info_dict = {subjid: sess_info[sess_info['subjid'] == subjid][['sessid', 'protocol', 'startstage', 'proto_stage']].set_index('sessid').to_dict('index')
             for subjid in subj_ids}

# %% Plot session information
proto_stage_labels = {'ClassicRLTasks_1': 'Operant Conditioning', 'ClassicRLTasks_2': 'Two-armed Bandit', 'ClassicRLTasks_3': 'Intertemporal Choice',
                       'ClassicRLTasks_4': 'Foraging', 'ToneCatDelayResp2_10': 'Working Memory', 'ToneCatDelayResp2_7': 'Working Memory',
                       'ToneCatDelayResp2_9': 'Working Memory', 'ToneCatDelayResp_8': 'Working Memory'}

plot_protos = list(proto_stage_labels.keys()) #['ClassicRLTasks_1', 'ClassicRLTasks_2', 'ClassicRLTasks_3', 'ClassicRLTasks_4']

beh_order = ['Operant Conditioning', 'Two-armed Bandit', 'Intertemporal Choice', 'Foraging', 'Working Memory']

sub_proto_info = sess_info[sess_info['proto_stage'].isin(plot_protos)].copy()

sub_proto_info['stage_labels'] = sub_proto_info['proto_stage'].apply(lambda x: proto_stage_labels[x])

sub_proto_info = sub_proto_info.groupby(['stage_labels', 'subjid']).agg('count')['sessid'].reset_index().pivot(index='stage_labels', columns='subjid', values='sessid')

sub_proto_info = sub_proto_info.reindex(beh_order)

ax = sub_proto_info.plot(kind='barh', stacked=True, colormap='Pastel2', legend=False, figsize=(5,3))
ax.set_ylabel('Behavior')
ax.set_xlabel('Session Counts')
ax.invert_yaxis()

fpah.save_fig(ax.get_figure(), fpah.get_figure_save_path('Session Counts', '', 'RL Tasks'), format='pdf')

# %% Set up variables
alignments = [Align.cue, Align.reward]
regions = ['DMS', 'PL']
xlims = {'DMS': [-1,2], 'PL': [-3,20]}
signal_type = 'z_dff_iso'
recalculate = False

filename = 'dLight_comparison_data'

save_path = path.join(utils.get_user_home(), 'db_data', signal_type+'_'+filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
        aligned_signals = data['aligned_signals']
        trial_sels = data['trial_sels']
else:
    aligned_signals = {subjid: {sessid: {align: {region: [] for region in regions} for align in alignments} for sessid in sess_ids[subjid]} for subjid in subj_ids}
    trial_sels = {subjid: {sessid: {align: [] for align in alignments} for sessid in sess_ids[subjid]} for subjid in subj_ids}

# %% Build signal matrices aligned to alignment points

reward_vol_thresh = 20
# choose 405 over 420 when there are sessions with both for 3.6
isos = {182: ['405', '420'], 202: ['405', '420'], 179: ['420', '405'],
        180: ['420', '405'], 188: ['420', '405'], 191: ['420', '405'], 207: ['420', '405']}

for subj_id in subj_ids:
    if not subj_id in aligned_signals:
        aligned_signals[subj_id] = {sessid: {align: {region: [] for region in regions} for align in alignments} for sessid in sess_ids[subj_id]}

    for sess_id in sess_ids[subj_id]:
        if sess_id in fpah.__sess_ignore:
            continue

        if not sess_id in aligned_signals[subj_id]:
            aligned_signals[subj_id][sess_id] = {align: {region: [] for region in regions} for align in alignments}

        if any([isinstance(aligned_signals[subj_id][sess_id][a][r], list) for a in alignments for r in regions]):

            protocol = sess_info_dict[subj_id][sess_id]['protocol']
            stage = sess_info_dict[subj_id][sess_id]['startstage']
            loc_db = protocol_db[protocol]
            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, isos=isos[subj_id], fit_baseline=False)
            fp_data = fp_data[subj_id][sess_id]

            # plot power spectrum
            # title = 'Subject {}, Session {}, {} Stage {}'.format(subj_id, sess_id, protocol, stage)
            # signals = [fp_data['processed_signals'][region][signal_type] for region in fp_data['processed_signals'].keys()]
            # fig = fpah.plot_power_spectra(signals, fp_data['dec_info']['decimated_dt'], title=title, signal_names=list(fp_data['processed_signals'].keys()))
            # fpah.save_fig(fig, fpah.get_figure_save_path('Power Spectra', subj_id, title.replace(',', '')))
            # plt.close(fig)

            sess_data = loc_db.get_behavior_data(sess_id)
            if not 'reward_time' in sess_data.columns:
                if sess_id < 95035:
                    sess_data['reward_time'] = sess_data['response_time']
                else:
                    sess_data['reward_time'] = sess_data['response_time'] + 0.5

            ts = fp_data['time']
            trial_start_ts = fp_data['trial_start_ts'][:-1]
            cue_ts = trial_start_ts + sess_data['response_cue_time']
            if 'cpoke_out_time' in sess_data.columns:
                # fix bug from earlier versions where early cpoke out was empty
                sess_data['cpoke_out_time'] = sess_data['cpoke_out_time'].apply(lambda x: np.nan if isinstance(x, list) else x)
                cpoke_out_ts = trial_start_ts + sess_data['cpoke_out_time']
            else:
                cpoke_out_ts = np.zeros_like(cue_ts) + np.nan
            reward_ts = trial_start_ts + sess_data['reward_time']

            for align in alignments:

                if not align in aligned_signals[subj_id][sess_id]:
                    aligned_signals[subj_id][sess_id][align] = {region: [] for region in regions}

                for region in fp_data['processed_signals'].keys():
                    if region in regions:
                        signal = fp_data['processed_signals'][region][signal_type]

                        match align:
                            case Align.cue:
                                resp_sel = ~np.isnan(sess_data['response_time'])
                                unreward_sel = sess_data['reward'] < reward_vol_thresh
                                # only look at trials where the cue happened before poking out
                                # do it this way so that if there is no cpoke out (i.e. pavlovian and foraging) this will evaluate to False
                                # poke_out_before_cue_sel = cpoke_out_ts < cue_ts
                                # trial_sel = resp_sel & ~poke_out_before_cue_sel
                                trial_sel = resp_sel
                                trial_after_sel = np.insert(trial_sel, 0, False)
                                trial_after_unreward_sel = np.insert(unreward_sel, 0, False)

                                align_ts = cue_ts[trial_sel].to_numpy()

                                # mask out reward-related signal
                                mask_ts = reward_ts.to_numpy()
                                # if current trial is unrewarded, mask on next reward
                                mask_ts[unreward_sel & trial_sel] = np.append(reward_ts, np.inf)[trial_after_sel & trial_after_unreward_sel]
                                mask_ts = mask_ts[trial_sel]

                                if region == 'PL':
                                    mask_lims = np.hstack((np.zeros_like(align_ts)[:, None], mask_ts[:, None]))
                                elif region == 'DMS':
                                    mask_lims = np.hstack((np.zeros_like(align_ts)[:, None], mask_ts[:, None]))
                            case Align.reward:
                                trial_sel = sess_data['reward'] > reward_vol_thresh
                                align_ts = reward_ts[trial_sel].to_numpy()
                                # for masking
                                trial_after_reward_sel = np.insert(trial_sel, 0, False)

                                if region == 'PL':
                                    mask_lims = np.hstack((np.insert(align_ts[:-1], 0, 0)[:, None], np.append(align_ts[1:], np.inf)[:, None]))
                                elif region == 'DMS':
                                    if protocol == 'ClassicRLTasks' and stage in [1,4]:
                                        mask_cue_ts = np.append(cue_ts, np.inf)
                                        mask_lims = np.hstack((np.zeros_like(align_ts)[:, None], mask_cue_ts[trial_after_reward_sel][:, None]))
                                    else:
                                        cport_on_ts = np.append(trial_start_ts + sess_data['cport_on_time'], np.inf)
                                        mask_lims = np.hstack((np.zeros_like(align_ts)[:, None], cport_on_ts[trial_after_reward_sel][:, None]))

                        lims = xlims[region]

                        mat, t = fp_utils.build_signal_matrix(signal, ts, align_ts, -lims[0], lims[1], mask_lims=mask_lims)

                        aligned_signals[subj_id][sess_id][align][region] = mat
                        trial_sels[subj_id][sess_id][align] = trial_sel

aligned_signals['t'] = {region: [] for region in regions}
dt = fp_data['dec_info']['decimated_dt']
for region in regions:
    aligned_signals['t'][region] = np.arange(xlims[region][0], xlims[region][1]+dt, dt)

with open(save_path, 'wb') as f:
    pickle.dump({'aligned_signals': aligned_signals, 'trial_sels': trial_sels}, f)

# %% Analyze aligned signals

ignored_signals = {'PL': [96556, 101853, 101906, 101958, 102186, 102235, 102288, 102604],
                   'DMS': [96556, 102604]}

ignored_subjects = [] # [179]

alignments = [Align.cue, Align.reward]
regions = ['PL', 'DMS']

plot_ind_sess = False
plot_beh_avg = False
save_plots = False
show_plots = False
analyze_peaks = True
normalize_to_cue = True

filter_props = {'DMS': {'filter': True, 'cutoff_f': 10},
                'PL': {'filter': True, 'cutoff_f': 1}}
peak_find_props = {'DMS': {'min_dist': 0.05, 'peak_tmax': 0.4, 'tau_tmax': 1},
                   'PL': {'min_dist': 1, 'peak_tmax': 3, 'tau_tmax': 10}}


def save_plot(fig, folder_name, plot_name, subj_id):
    if save_plots:
        fpah.save_fig(fig, fpah.get_figure_save_path(folder_name, subj_id, plot_name))

    if not show_plots:
        plt.close(fig)

def calc_error(mat, use_se):
    if use_se:
        return utils.stderr(mat, axis=0)
    else:
        return np.nanstd(mat, axis=0, ddof=1)

def plot_avg_signals(data_mat_dict, t_dict, title, x_label, y_label, peak_metrics,
                     xlims=None, dashlines=0, use_se=True, ph=3.5, pw=7):

    if not save_plots and not show_plots:
        return None

    regions = list(data_mat_dict.keys())
    n_rows = len(regions)
    fig, axs = plt.subplots(n_rows, 1, figsize=(pw, ph*n_rows), layout='constrained')

    if n_rows == 1:
        axs = [axs]

    fig.suptitle(title)

    for i, region in enumerate(regions):

        t = t_dict[region]

        # limit x axis like this so that the y is scaled to what is plotted
        if xlims is None:
            t_sel = np.full(t.shape, True)
        else:
            t_sel = (t > xlims[region][0]) & (t < xlims[region][1])

        ax = axs[i]
        ax.set_title(region)
        if not dashlines is None:
            plot_utils.plot_dashlines(dashlines, ax=ax)

        act = data_mat_dict[region]
        if len(act) > 0:
            avg_signal = np.nanmean(act, axis=0)[t_sel]
            #avg_signal = utils.zscore(avg_signal)
            t_lim = t[t_sel]
            plot_utils.plot_psth(avg_signal, t_lim, calc_error(act, use_se)[t_sel], ax)

            if len(peak_metrics[region]) > 0:
                # get peak and decay values
                peak_t = peak_metrics[region]['peak_time']
                peak_t_idx = np.argmin(np.abs(t_lim - peak_t))
                peak_y = avg_signal[peak_t_idx]
                peak_height = peak_metrics[region]['peak_height']
                width_info = peak_metrics[region]['peak_width_info']
                width_y = width_info['y']

                t_tau = t_lim[(t_lim >= peak_t) & (t_lim <= peak_metrics[region]['peak_end_time'])]
                decay_fit = peak_metrics[region]['decay_form'](t_tau - peak_t, *peak_metrics[region]['decay_params'])

                # plot peak marker, height, width, and decay fit
                ax.plot(peak_t, peak_y, marker=7, markersize=10, color='C1')
                ax.vlines(peak_t, peak_y-peak_height, peak_y, color='C2', linestyles='dotted')
                width_info = peak_metrics[region]['peak_width_info']
                ax.hlines(width_y, *width_info['t_lims'], color='C1')
                ax.plot(t_tau, decay_fit, '--', color='C1', linewidth=2)

        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

    return fig


t = aligned_signals['t']
trial_stacked_signals = {v: {a: {r: np.zeros((0,len(t[r]))) for r in regions} for a in alignments} for v in variant_subj.keys()}
beh_stacked_signals = {v: {a: {r: np.zeros((0,len(t[r]))) for r in regions} for a in alignments} for v in variant_subj.keys()}
beh_variance_info = {v: {a: {r: {'var': [], 'n':[]} for r in regions} for a in alignments} for v in variant_subj.keys()}
session_peak_metrics = {subjid: {sessid: {a: {r: {} for r in regions} for a in alignments} for sessid in sess_ids[subjid]} for subjid in subj_ids}
beh_peak_metrics = {subjid: {} for subjid in subj_ids}

for variant, subjids in variant_subj.items():
    for subj_id in subjids:
        if subj_id in ignored_subjects:
            continue

        subj_behaviors = sess_info[sess_info['subjid'] == subj_id][['protocol', 'startstage', 'proto_stage']].drop_duplicates().to_numpy()

        for (protocol, stage, proto_key) in subj_behaviors:
            beh_sess_ids = sess_info[(sess_info['subjid'] == subj_id) & (sess_info['proto_stage'] == proto_key)]['sessid']
            beh_peak_metrics[subj_id][proto_key] = {a: {r: {} for r in regions} for a in alignments}

            for align in alignments:
                beh_signal = {r: np.zeros((0,len(t[r]))) for r in regions}

                match align:
                    case Align.cue:
                        folder_name = 'Response Cue Comparison'
                        x_label = 'Time from response cue (s)'
                    case Align.reward:
                        folder_name = 'Reward Comparison'
                        x_label = 'Time from reward (s)'

                for sess_id in beh_sess_ids:
                    if not align in aligned_signals[subj_id][sess_id]:
                        continue

                    mat = aligned_signals[subj_id][sess_id][align]
                    norm_mat = copy.deepcopy(mat)

                    for region in regions:
                        if len(mat[region]) > 0:
                            t_r = t[region]
                            baseline_sel = (t_r >= -0.1) & (t_r < 0)

                            # shift baseline to so cue baseline is 0 at t=0
                            if normalize_to_cue:
                                cue_trial_sels = trial_sels[subj_id][sess_id][Align.cue]
                                current_trial_sels = trial_sels[subj_id][sess_id][align]
                                baseline = np.nanmean(aligned_signals[subj_id][sess_id][Align.cue][region][current_trial_sels[cue_trial_sels],:][:,baseline_sel], axis=1)
                            else:
                                # shift to match signals at t=0 fo this alignment
                                baseline = np.nanmean(mat[region][:,baseline_sel], axis=1)
                                
                            norm_mat[region] = mat[region] - baseline[:,None]

                            if not sess_id in ignored_signals[region]:
                                trial_stacked_signals[variant][align][region] = np.vstack((trial_stacked_signals[variant][align][region], norm_mat[region]))
                                beh_signal[region] = np.vstack((beh_signal[region], norm_mat[region]))

                                if analyze_peaks:
                                    # calculate peak properties
                                    session_peak_metrics[subj_id][sess_id][align][region] = fpah.calc_peak_properties(np.nanmean(norm_mat[region], axis=0), t_r, 
                                                                                                                 filter_params=filter_props[region],
                                                                                                                 peak_find_params=peak_find_props[region])

                    if plot_ind_sess:
                        title = 'Subject {}, Session {}, {} Stage {}'.format(subj_id, sess_id, protocol, stage)
                        fig = plot_avg_signals(norm_mat, t, title, x_label, '% dF/F', session_peak_metrics[subj_id][sess_id][align])#, correct_peak_offset=True)
                        save_plot(fig, folder_name, title.replace(',', ''), subj_id)

                # handle behavior averages
                for region in regions:
                    t_r = t[region]
                    avg_signal = np.nanmean(beh_signal[region], axis=0)
                    beh_stacked_signals[variant][align][region] = np.vstack((beh_stacked_signals[variant][align][region], avg_signal))
                    if analyze_peaks:
                        beh_peak_metrics[subj_id][proto_key][align][region] = fpah.calc_peak_properties(avg_signal, t_r, 
                                                                                                   filter_params=filter_props[region],
                                                                                                   peak_find_params=peak_find_props[region])

                    beh_variance_info[variant][align][region]['var'].append(np.nanvar(beh_signal[region], axis=0, ddof=1))
                    beh_variance_info[variant][align][region]['n'].append(utils.nancount(beh_signal[region]))

                if plot_beh_avg:
                    title = 'Subject {}, {} Stage {}, All'.format(subj_id, protocol, stage)
                    fig = plot_avg_signals(beh_signal, t, title, x_label, '% dF/F', beh_peak_metrics[subj_id][proto_key][align])
                    save_plot(fig, folder_name, title.replace(',', ''), subj_id)


# %% Plot average across variants

regions = ['DMS']
alignments = [Align.cue, Align.reward]

grouping = 'trials' # 'behavior' #

stacked_signals = trial_stacked_signals

plot_lims = {Align.cue: {'DMS': [-0.1,0.6], 'PL': [-1,1]},
             Align.reward: {'DMS': [-0.1,1], 'PL': [-0.5,10]}}

#width_ratios = [2,10.5]
width_ratios = [0.7,1.1]

plot_name = '{}_cue_reward'.format('_'.join(regions))

n_rows = len(regions)
n_cols = len(alignments)
t = aligned_signals['t']

fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(7, 4*n_rows), sharey='row', width_ratios=width_ratios)
if n_rows == 1 and n_cols == 1:
    axs = np.array(axs)

axs = axs.reshape((n_rows, n_cols))

for i, region in enumerate(regions):
    for j, align in enumerate(alignments):
        ax = axs[i,j]
        
        match align:
            case Align.cue:
                label = 'Response Cue'
            case Align.reward:
                label = 'Reward Delivery'

        act = np.vstack([stacked_signals[v][align][region] for v in stacked_signals.keys()])
        t_r = t[region]
        t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
        error = calc_error(act, True)
        
        plot_utils.plot_dashlines(0, ax=ax)

        plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, color='#08AB36')

        ax.set_title('{} {}'.format(region, label))
        if j == 0:
            ax.set_ylabel('Normalized Z-dF/F')
        ax.set_xlabel('Time (s)')

fpah.save_fig(fig, fpah.get_figure_save_path('Example Signals', '', plot_name), format='pdf')

# %% Plot average for each variant

def calc_combined_error(variance_info, use_se=True):
    var = np.array(variance_info['var'])
    n = np.array(variance_info['n'])
    combined_var = np.sum((n - 1)*var, axis=0)/(np.sum(n, axis=0)-len(variance_info['n']))
    combined_std = np.sqrt(combined_var)

    if use_se:
        return combined_std/np.sqrt(np.sum(n, axis=0)-len(variance_info['n']))
    else:
        return combined_std

grouping = 'trials' # 'behavior' #

match grouping:
    case 'trials':
        stacked_signals = trial_stacked_signals
    case 'behavior':
        stacked_signals = beh_stacked_signals

n_rows = len(regions)
t = aligned_signals['t']

for align in alignments:
    match align:
        case Align.cue:
            title = 'Response Cue'
            x_label = 'Time from response cue (s)'
            plot_lims = {'DMS': [-0.1,0.8], 'PL': [-1,8]}
        case Align.reward:
            title = 'Reward Delivery'
            x_label = 'Time from reward (s)'
            plot_lims = {'DMS': [-0.1,1.1], 'PL': [-1,11]}

    fig, axs = plt.subplots(n_rows, 1, layout='constrained', figsize=(6, 3*n_rows))
    fig.suptitle(title)

    for i, region in enumerate(regions):
        ax = axs[i]

        for variant in stacked_signals.keys():
            act = stacked_signals[variant][align][region]
            t_r = t[region]
            t_sel = (t_r > plot_lims[region][0]) & (t_r < plot_lims[region][1])
            match grouping:
                case 'trials':
                    error = calc_error(act, True)
                case 'behavior':
                    error = calc_combined_error(beh_variance_info[variant][align][region])

            plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=variant)

        ax.set_title(region)
        plot_utils.plot_dashlines(0, ax=ax)

        ax.set_ylabel('Normalized %dF/F')
        ax.set_xlabel(x_label)
        ax.legend()


# %% Set up peak property analysis

# reformat the peak metric dictionaries into a flat table
proto_stage_labels = {'ClassicRLTasks_1': 'Pavlovian', 'ClassicRLTasks_2': 'Two-armed Bandit', 'ClassicRLTasks_3': 'Temporal Choice',
       'ClassicRLTasks_4': 'Foraging', 'ToneCatDelayResp2_10': 'Sel WM - Two Tones', 'ToneCatDelayResp2_7': 'Sel WM - Grow Poke',
       'ToneCatDelayResp2_9': 'Sel WM - Grow Delay', 'ToneCatDelayResp_8': 'Sel WM - Two Tones'}

sess_peak_metrics_df = pd.DataFrame([dict([('subjid', subjid), ('sessid', sessid), ('region', region), ('align', align.name),
                                           ('proto_stage', proto_stage_labels[sess_info_dict[subjid][sessid]['proto_stage']]),
                                           *session_peak_metrics[subjid][sessid][align][region].items()])
                              for subjid in subj_ids for sessid in sess_ids[subjid] for region in regions for align in alignments])

beh_peak_metrics_df = pd.DataFrame([dict([('subjid', subjid), ('sessid', 'all'), ('region', region), ('align', align.name),
                                          ('proto_stage', proto_stage_labels[proto_stage]),
                                          *beh_peak_metrics[subjid][proto_stage][align][region].items()])
                              for subjid in subj_ids for proto_stage in beh_peak_metrics[subjid].keys() for region in regions for align in alignments])

# add variant info
sess_peak_metrics_df['variant'] = sess_peak_metrics_df['subjid'].apply(lambda x: subj_variant[x])
beh_peak_metrics_df['variant'] = beh_peak_metrics_df['subjid'].apply(lambda x: subj_variant[x])

# make subject ids categories
sess_peak_metrics_df['subjid'] = sess_peak_metrics_df['subjid'].astype('category')
beh_peak_metrics_df['subjid'] = beh_peak_metrics_df['subjid'].astype('category')


parameters = ['peak_time', 'peak_height', 'peak_width', 'decay_tau']
parameter_labels = {'peak_time': 'Time to peak (s)', 'peak_height': 'Peak height (% dF/F)',
                    'peak_width': 'Peak FWHM (s)', 'decay_tau': 'Decay τ (s)'}

subj_order = utils.flatten(variant_subj)
beh_order = ['Pavlovian', 'Foraging', 'Two-armed Bandit', 'Temporal Choice',
             'Sel WM - Grow Poke', 'Sel WM - Grow Delay', 'Sel WM - Two Tones']

def calc_iqr_multiple(table, group_by_cols, parameters):
    table = table.copy()
    if not utils.is_list(group_by_cols):
        group_by_cols = [group_by_cols]

    if not utils.is_list(parameters):
        parameters = [parameters]

    iqr_mult_keys = {param: 'iqr_mult_' + param for param in parameters}
    # initialize iqr columns
    for param in parameters:
        table[iqr_mult_keys[param]] = np.nan

    groupings = table[group_by_cols].drop_duplicates().to_numpy()
    for group in groupings:
        sel = np.all(np.array([(table[group_col] == group_val) for group_col, group_val in zip(group_by_cols, group)]), axis=0)

        sub_table = table.loc[sel]
        for param in parameters:
            q1, q3 = np.nanquantile(sub_table[param], [0.25, 0.75])
            iqr = q3 - q1
            below_sel = sub_table[param] < q1
            above_sel = sub_table[param] > q3
            sub_table.loc[below_sel, iqr_mult_keys[param]] = (sub_table.loc[below_sel, param] - q1)/iqr
            sub_table.loc[above_sel, iqr_mult_keys[param]] = (sub_table.loc[above_sel, param] - q3)/iqr

        table.loc[sel, iqr_mult_keys.values()] = sub_table[iqr_mult_keys.values()]

    return table

# %% plot the all session peak property values organized by variant, subject, and behavior

ignore_outliers = True
outlier_thresh = 10

parameters = ['peak_time', 'peak_height', 'peak_width', 'decay_tau']

for align in alignments:
    match align:
        case Align.cue:
            base_title = 'Response Cue Transients - {}'
        case Align.reward:
            base_title = 'Reward Transients - {}'

    for region in regions:
        region_metrics = sess_peak_metrics_df[(sess_peak_metrics_df['region'] == region) & (sess_peak_metrics_df['align'] == align)]

        # remove outliers on a per-subject basis:
        if ignore_outliers:
            region_metrics = calc_iqr_multiple(region_metrics, 'subjid', parameters)
            for param in parameters:
                outlier_sel = np.abs(region_metrics['iqr_mult_'+param]) >= outlier_thresh
                if any(outlier_sel):
                    region_metrics.loc[outlier_sel, param] = np.nan

        # group by variant and subject, colored by task
        fig = plt.figure(figsize=(12,8), layout='constrained')
        gs = GridSpec(2, 3, width_ratios=[1,1,0.4], figure=fig)
        fig.suptitle(base_title.format(region))

        for i, param in enumerate(parameters):
            ax = fig.add_subplot(gs[int(np.floor(i/2)), i%2])
            sb.stripplot(data=region_metrics, x='subjid', y=param, hue='proto_stage', ax=ax,
                         alpha=0.6, order=subj_order, hue_order=beh_order, legend=False, jitter=0.25)
            ax.set_xlabel('')
            ax.set_ylabel(parameter_labels[param])

            # Add labels for the variants and lines to separate them
            variant_ax = ax.secondary_xaxis(location=0)
            variant_ax.set_xticks([0.5, 3, 5.5], labels=variant_subj.keys())
            variant_ax.tick_params('x', length=20, bottom=False)

            variant_sep_ax = ax.secondary_xaxis(location=0)
            variant_sep_ax.set_xticks([-0.5, 1.5, 4.5, 6.5], ['', '', '', ''])
            variant_sep_ax.tick_params('x', length=35)

        # add legend
        ax = fig.add_subplot(gs[:, 2])
        colors = sb.color_palette()
        patches = [Patch(label=beh, color=colors[i], alpha=0.8) for i, beh in enumerate(beh_order)]
        ax.legend(patches, beh_order, loc='center', frameon=False, title='Tasks')
        ax.set_axis_off()
        
        # group by task, colored by subject
        fig = plt.figure(figsize=(12,8), layout='constrained')
        gs = GridSpec(2, 3, width_ratios=[1,1,0.4], figure=fig)
        fig.suptitle(base_title.format(region))

        for i, param in enumerate(parameters):
            ax = fig.add_subplot(gs[int(np.floor(i/2)), i%2])
            sb.stripplot(data=region_metrics, x='proto_stage', y=param, hue='subjid', ax=ax,
                         alpha=0.6, order=beh_order, hue_order=subj_order, legend=False, jitter=0.25)
            ax.set_xlabel('')
            ax.set_ylabel(parameter_labels[param])
            plt.xticks(rotation = 45)

        # add legend
        ax = fig.add_subplot(gs[:, 2])
        colors = sb.color_palette()
        patches = [Patch(label=subj, color=colors[i], alpha=0.8) for i, subj in enumerate(subj_order)]
        ax.legend(patches, subj_order, loc='center', frameon=False, title='Subjects')
        ax.set_axis_off()

# %% make combined comparison figures per peak property across variants

ignore_outliers = True
outlier_thresh = 10

parameters = ['peak_time', 'peak_height', 'peak_width', 'decay_tau']
parameter_titles = {'peak_time': 'Time to Peak', 'peak_height': 'Peak Amplitude',
                    'peak_width': 'Peak Width', 'decay_tau': 'Decay τ'}

regions = ['DMS', 'PL']
align_order = ['Response Cue', 'Reward Delivery']
align_labels = {'cue': 'Response Cue', 'reward': 'Reward Delivery'}
peak_metrics_df = sess_peak_metrics_df.copy()
peak_metrics_df['align_label'] = peak_metrics_df['align'].apply(lambda x: align_labels[x])
subj_ids = np.unique(peak_metrics_df['subjid'])

for param in parameters:
    
    plot_name = 'cue_reward_peak_comp_{}'.format(param)

    # Compare responses across regions grouped by alignment
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout='constrained', sharey=True)
    fig.suptitle(parameter_titles[param])

    if ignore_outliers:
        peak_metrics_df = calc_iqr_multiple(peak_metrics_df, ['subjid', 'region'], param)
        outlier_sel = np.abs(peak_metrics_df['iqr_mult_'+param]) >= outlier_thresh
        if any(outlier_sel):
            peak_metrics_df.loc[outlier_sel, param] = np.nan

    subj_avgs = peak_metrics_df.groupby(['subjid', 'region', 'align_label']).agg({param: np.nanmean}).reset_index()

    # plot sensor averages in boxplots
    sb.boxplot(data=peak_metrics_df, x='align_label', y=param, hue='region',
               order=align_order, hue_order=regions, ax=ax, showfliers=False)
    ax.set_ylabel(parameter_labels[param])
    ax.set_xlabel('')
    ax.set_yscale('log')

    # add subject averages for each alignment with lines connecting them
    dodge = 0.2
    noise = 0.06
    n_neg = int(np.floor(len(subj_ids)/2))
    n_pos = int(np.ceil(len(subj_ids)/2))
    jitters = np.concatenate([np.random.uniform(-1, -0.1, n_neg), np.random.uniform(0.1, 1, n_pos)]) * noise
    
    for i, align in enumerate(align_order):
        x = np.array([i - dodge, i + dodge])

        for subj_id, jitter in zip(subj_ids, jitters):
            subj_avg = subj_avgs[(subj_avgs['subjid'] == subj_id) & (subj_avgs['align_label'] == align)]

            y = [subj_avg.loc[subj_avg['region'] == r, param] for r in regions]

            ax.plot(x+jitter, y, color='black', marker='o', linestyle='dashed', alpha=0.75)
            

    fpah.save_fig(fig, fpah.get_figure_save_path('Example Signals', '', plot_name), format='pdf')

# %% make combined comparison figures per peak property grouped by variant

ignore_outliers = True
outlier_thresh = 10

parameters = ['peak_time', 'peak_height', 'peak_width', 'decay_tau']
parameter_titles = {'peak_time': 'Time to Peak', 'peak_height': 'Peak Height',
                    'peak_width': 'Peak Width', 'decay_tau': 'Decay τ'}

n_regions = len(regions)
variant_palette = sb.color_palette()
variant_labels = list(variant_subj.keys())
align_order = ['cue', 'reward']
align_labels = {'cue': 'Response Cue', 'reward': 'Reward Delivery'}
hatch_order = ['//\\\\', '']

for param in parameters:

    # Compare responses across alignments grouped by region
    fig, axs = plt.subplots(1, n_regions, figsize=(4*n_regions, 4), layout='constrained')
    fig.suptitle(parameter_titles[param])

    for i, region in enumerate(regions):

        region_metrics = sess_peak_metrics_df[sess_peak_metrics_df['region'] == region]

        if ignore_outliers:
            region_metrics = calc_iqr_multiple(region_metrics, ['subjid', 'align'], param)
            outlier_sel = np.abs(region_metrics['iqr_mult_'+param]) >= outlier_thresh
            if any(outlier_sel):
                region_metrics.loc[outlier_sel, param] = np.nan

        subj_avgs = region_metrics.groupby(['subjid', 'align']).agg({param: np.nanmean}).reset_index()

        ax = axs[i]
        ax.set_title(region)
        # plot sensor averages in boxplots
        sb.boxplot(data=region_metrics, x='variant', order=variant_labels, y=param, hue='align',
                   hue_order=align_order, ax=ax, showfliers=False, legend=False)
        ax.set_ylabel(parameter_labels[param])
        ax.set_xlabel('')

        # update colors and fills of boxes
        for j, patch in enumerate(ax.patches):
            # Left boxes first, then right boxes
            if j < len(variant_labels):
                # add hatch to cues
                patch.set_hatch(hatch_order[int(j/len(variant_labels))])

            patch.set_facecolor(variant_palette[j % len(variant_labels)])
            patch.set_alpha(0.6)


        # add subject averages for each alignment with lines connecting them
        dodge = 0.2
        for j, variant in enumerate(variant_subj.keys()):
            x = [j - dodge, j + dodge]

            for subj_id in variant_subj[variant]:
                subj_avg = subj_avgs[subj_avgs['subjid'] == subj_id]
                y = [subj_avg.loc[subj_avg['align'] == a, param] for a in align_order]

                ax.plot(x, y, color='black', marker='o', linestyle='dashed', alpha=0.75)

    # Add the custom legend to the figure (or to one of the subplots)
    legend_patches = [Patch(facecolor='none', edgecolor='black', hatch=hatch_order[i], label=align_labels[a]) for i, a in enumerate(align_order)]
    ax.legend(handles=legend_patches, frameon=False)


    # Compare responses across regions grouped by alignment
    fig, axs = plt.subplots(1, n_regions, figsize=(4*n_regions, 4), layout='constrained', sharey=True)
    fig.suptitle(parameter_titles[param])

    for i, align in enumerate(align_order):

        align_metrics = sess_peak_metrics_df[sess_peak_metrics_df['align'] == align]

        if ignore_outliers:
            align_metrics = calc_iqr_multiple(align_metrics, ['subjid', 'region'], param)
            outlier_sel = np.abs(align_metrics['iqr_mult_'+param]) >= outlier_thresh
            if any(outlier_sel):
                align_metrics.loc[outlier_sel, param] = np.nan

        subj_avgs = align_metrics.groupby(['subjid', 'region']).agg({param: np.nanmean}).reset_index()

        ax = axs[i]
        ax.set_title(align_labels[align])
        # plot sensor averages in boxplots
        sb.boxplot(data=align_metrics, x='variant', order=variant_labels, y=param, hue='region',
                   hue_order=regions, ax=ax, showfliers=False, legend=False)
        ax.set_ylabel(parameter_labels[param])
        ax.set_xlabel('')
        ax.set_yscale('log')

        # update colors and fills of boxes
        for j, patch in enumerate(ax.patches):
            # Left boxes first, then right boxes
            if j < len(variant_labels):
                # add hatch to cues
                patch.set_hatch(hatch_order[int(j/len(variant_labels))])

            patch.set_facecolor(variant_palette[j % len(variant_labels)])
            patch.set_alpha(0.6)


        # add subject averages for each alignment with lines connecting them
        dodge = 0.2
        for j, variant in enumerate(variant_subj.keys()):
            x = [j - dodge, j + dodge]

            for subj_id in variant_subj[variant]:
                subj_avg = subj_avgs[subj_avgs['subjid'] == subj_id]
                y = [subj_avg.loc[subj_avg['region'] == r, param] for r in regions]

                ax.plot(x, y, color='black', marker='o', linestyle='dashed', alpha=0.75)

    # Add the custom legend to the figure (or to one of the subplots)
    legend_patches = [Patch(facecolor='none', edgecolor='black', hatch=hatch_order[i], label=regions[i]) for i in range(len(regions))]
    ax.legend(handles=legend_patches, frameon=False)

# %% Perform statistical tests on the peak properties

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm

ignore_outliers = True
outlier_thresh = 10

# look at each region and property separately
parameters = ['peak_time', 'peak_height', 'peak_width', 'decay_tau'] # ['peak_height', 'peak_width'] #
regions = ['DMS', 'PL']

for region in regions:
    region_metrics = sess_peak_metrics_df[sess_peak_metrics_df['region'] == region]
    
    for param in parameters:
        
        if ignore_outliers:
            region_metrics = calc_iqr_multiple(region_metrics, ['subjid', 'align'], param)
            outlier_sel = np.abs(region_metrics['iqr_mult_'+param]) >= outlier_thresh
            if any(outlier_sel):
                region_metrics.loc[outlier_sel, param] = np.nan
                
        # mem = sm.MixedLM.from_formula(param+' ~ C(variant)', groups='subjid', re_formula='~C(align)', data=region_metrics, missing='drop')
        # print('{}: {} fixed variants, random subjects and alignments:\n {}\n'.format(param, region, mem.fit().summary()))

        alignments = np.unique(region_metrics['align'])
        
        for align in alignments:
            align_metrics = region_metrics[region_metrics['align'] == align]
            
    #     for param in parameters:
            
    #         # variant_lm = ols(param+' ~ C(variant)', data=align_metrics).fit()
    #         # subj_lm = ols(param+' ~ C(subjid)', data=align_metrics).fit()
    #         # variant_subj_lm = ols(param+' ~ C(variant)*C(subjid)', data=align_metrics).fit()
    #         # subj_variant_lm = ols(param+' ~ C(subjid)*C(variant)', data=align_metrics).fit()
            
    #         # # print('{} {}-aligned variant model fit:\n {}\n'.format(region, align, variant_lm.summary()))
    #         # # print('{} {}-aligned variant & subject model fit:\n {}\n'.format(region, align, variant_subj_lm.summary()))
            
    #         # print('{}: {} {}-aligned variant model ANOVA:\n {}\n'.format(param, region, align, anova_lm(variant_lm)))
    #         # print('{}: {} {}-aligned subject model ANOVA:\n {}\n'.format(param, region, align, anova_lm(subj_lm)))
    #         # print('{}: {} {}-aligned variant & subject model ANOVA:\n {}\n'.format(param, region, align, anova_lm(variant_subj_lm)))
    #         # print('{}: {} {}-aligned subject & variant model ANOVA:\n {}\n'.format(param, region, align, anova_lm(subj_variant_lm)))
            
    #         # print('{}: {} {}-aligned comparison between variant & variant/subject models:\n {}\n'.format(param, region, align, anova_lm(variant_lm, variant_subj_lm)))
    #         # print('{}: {} {}-aligned comparison between subject & subject/variant models:\n {}\n'.format(param, region, align, anova_lm(subj_lm, subj_variant_lm)))
            
            
            mem = sm.MixedLM.from_formula(param+' ~ C(variant)', groups='subjid', data=align_metrics, missing='drop')
            print('{}: {} {}-aligned fixed variants, random subjects:\n {}\n'.format(param, region, align, mem.fit().summary()))

            mem = sm.MixedLM.from_formula(param+' ~ C(proto_stage)', groups='subjid', data=align_metrics, missing='drop')
            print('{}: {} {}-aligned fixed behavior, random subjects:\n {}\n'.format(param, region, align, mem.fit().summary()))
    
        

