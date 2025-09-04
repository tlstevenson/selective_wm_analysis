# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:41:01 2024

@author: tanne
"""

# %% imports

import init
import hankslab_db.basicRLtasks_db as rl_db
import hankslab_db.tonecatdelayresp_db as wm_db
from pyutils import utils
import numpy as np
import fp_analysis_helpers as fpah
from sys_neuro_tools import plot_utils, fp_utils
import matplotlib.pyplot as plt
import numpy as  np


# %% Load data
wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
rl_loc_db = rl_db.LocalDB_BasicRLTasks('twoArmBandit')

#sess_ids = {180: [101447]}
#sess_ids = {191: [101617]}

#fp_data, implant_info = fpah.load_fp_data(wm_loc_db, sess_ids, iso_lpf=10)
#sess_data = wm_loc_db.get_behavior_data(utils.flatten(sess_ids))

# sess_ids = {207: [100752]}
# fp_data, implant_info = fpah.load_fp_data(rl_loc_db, sess_ids)
# sess_data = rl_loc_db.get_behavior_data(utils.flatten(sess_ids))

sess_ids = {198: [116607,117242,117442], 199: [117450]}
fp_data, implant_info = fpah.load_fp_data(wm_loc_db, sess_ids, filter_dropout_outliers=False)
#sess_data = wm_loc_db.get_behavior_data(utils.flatten(sess_ids))

# %% View preprocessing

gen_title = 'Subject {}, Session {}'
sub_t = [0, np.inf] # [1100, 1120] #
dec = 10

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        t = fp_data[subj_id][sess_id]['time']

        fpah.view_processed_signals(fp_data[subj_id][sess_id]['processed_signals'], t, t_min=sub_t[0], t_max=sub_t[1], dec=dec,
                                    title=gen_title.format(subj_id, sess_id))

# %% define plotting routine

def plot_signal_details(processed_signals, t, title, lines_dict={}, dec=10, signal_types=None):

    regions = processed_signals.keys()

    t = t[::dec].copy()

    if signal_types is None:
        signal_types = ['dff_iso']
        
    for signal_type in signal_types:
        
        fig, axs = plt.subplots(len(regions), 1, layout='constrained', figsize=[18,6*len(regions)], sharex=True)
        if len(regions) == 1:
            axs = [axs]
        
        plt.suptitle(title + ' - ' + signal_type)
    
        for i, region in enumerate(regions):

            ax = axs[i]
            ax.plot(t, processed_signals[region][signal_type][::dec], label='_')

            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']

            for j, (name, lines) in enumerate(lines_dict.items()):
                ax.vlines(lines, 0, 1, label=name, color=colors[j], linestyles='dashed', transform=ax.get_xaxis_transform())

            ax.set_title(region)
            _, y_label = fpah.get_signal_type_labels(signal_type)
            ax.set_ylabel(y_label)
            ax.set_xlabel('Time (s)')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
            ax.legend()


# %% create plots
subj_ids = list(sess_ids.keys())
dec=2
signal_types = ['z_dff_iso']

for subj_id in subj_ids:
    for sess_id in sess_ids[subj_id]:

        sess_fp = fp_data[subj_id][sess_id]
        trial_data = sess_data[sess_data['sessid'] == sess_id]

        ts = sess_fp['time']
        trial_start_ts = sess_fp['trial_start_ts'][:-1]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        first_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[0] if utils.is_list(x) else x)
        second_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[1] if utils.is_list(x) else np.nan)
        tone_ts = np.unique([*first_tone_ts, *second_tone_ts])
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']
        reward_ts = trial_start_ts + trial_data['reward_time']
        reward_ts = reward_ts[trial_data['reward'] > 0]

        lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, 'Tone': tone_ts,
                      'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
                      'Reward': reward_ts}

        plot_signal_details(sess_fp['processed_signals'], sess_fp['time'], 'Subject {}, Session {}'.format(subj_id, sess_id), lines_dict, dec=dec, signal_types=signal_types)


# %% Create cleaned up example data plot

subj_id = 207
sess_id = 100752
# subj_id = 180
# sess_id = 101447
sess_ids = {subj_id: [sess_id]}
sess_fp = fp_data[subj_id][sess_id]
trial_data = sess_data[sess_data['sessid'] == sess_id]
t_range = [1420, 1470] #[0, np.inf] #[2730, 2780] #[1155, 1195] # [1150, 1250] # 
regions = ['DMS', 'PL']
signal_type = 'dff_iso'
dec = 2
filt_f = {'DMS': 8, 'PL': 4}

region_colors = {'DMS': '#53C43B', 'PL': '#BB6ED8'}

t = sess_fp['time']

trial_start_ts = sess_fp['trial_start_ts'][:-1]
cue_ts = trial_start_ts + trial_data['response_cue_time']
reward_ts = trial_start_ts + trial_data['reward_time']
reward_ts = reward_ts[trial_data['reward'] > 0]

lines_dict = {'Response Cue': cue_ts, 'Reward Delivery': reward_ts}

t_sel = (t > t_range[0]) & (t < t_range[1])

fig, axs = plt.subplots(len(regions), 1, layout='constrained', figsize=[8,4*len(regions)], sharex=True)

axs = np.array(axs).reshape((len(regions)))

for i, region in enumerate(regions):
    signal = sess_fp['processed_signals'][region][signal_type]
    filt_signal = fp_utils.filter_signal(signal, filt_f[region], 1/sess_fp['dec_info']['decimated_dt'])
    
    ax = axs[i]
    ax.plot(t[t_sel][::dec], filt_signal[t_sel][::dec], label='_', color=region_colors[region])
    
    for j, (name, lines) in enumerate(lines_dict.items()):
        lines = lines[(lines > t_range[0]) & (lines < t_range[1])]
        ax.vlines(lines, 0, 1, label=name, color='C'+str(j+1), linestyles='dashed', transform=ax.get_xaxis_transform())
    
    _, y_label = fpah.get_signal_type_labels(signal_type)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Time (s)')
    ax.legend()

fpah.save_fig(fig, fpah.get_figure_save_path('Example Signals', '', '_'.join(regions)), format='pdf')

# %%
regions = ['PL', 'DMS', 'DLS', 'TS']
signal_type = 'dff_iso'
signals = [sess_fp['processed_signals'][region][signal_type] for region in regions]
fpah.plot_power_spectra(signals, sess_fp['dec_info']['decimated_dt'], title='{}'.format(signal_type), signal_names=regions)
