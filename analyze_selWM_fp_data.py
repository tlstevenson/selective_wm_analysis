# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:33:24 2023

@author: tanne
"""

# %% imports

import init

import os.path as path
import pandas as pd
import pyutils.utils as utils
from sys_neuro_tools import plot_utils, fp_utils
import hankslab_db.tonecatdelayresp_db as db
import fp_analysis_helpers as fpah
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import pickle
import copy

# %% Load behavior data

sess_ids = [93412] #92692, 92562, 92600

loc_db = db.LocalDB_ToneCatDelayResp()  # reload=True
sess_data = loc_db.get_behavior_data(sess_ids)

# %% Load fiber photometry data

data_path = path.join(utils.get_user_home(), 'fp_data')
fp_data = {}
for sess_id in sess_ids:
    load_path = path.join(data_path, 'Session_{}.pkl'.format(sess_id))
    with open(load_path, 'rb') as f:
        fp_data[sess_id] = pickle.load(f)

# %% Process photometry data in different ways

regions = ['DMS', 'PFC'] #

# iso_pfc = 'PFC_405'
# iso_dms = 'DMS_405'
iso_pfc = 'PFC_420'
iso_dms = 'DMS_420'
lig_pfc = 'PFC_490'
lig_dms = 'DMS_490'
# lig_pfc = 'PFC_465'
# lig_dms = 'DMS_465'

signals_by_region = {region: {'iso': iso_pfc if region == 'PFC' else iso_dms,
                              'lig': lig_pfc if region == 'PFC' else lig_dms}
                     for region in regions}

for sess_id in sess_ids:
    signals = fp_data[sess_id]['signals']

    fp_data[sess_id]['processed_signals'] = {}

    for region, signal_names in signals_by_region.items():
        raw_lig = signals[signal_names['lig']]
        raw_iso = signals[signal_names['iso']]

        # use isosbestic correction
        dff_iso = fp_utils.calc_iso_dff(raw_lig, raw_iso)

        # use ligand-signal baseline with polynomial
        # nans = np.isnan(raw_lig)
        # baseline = np.full_like(raw_lig, np.nan)
        # baseline[~nans] = peakutils.baseline(raw_lig[~nans])

        # dff_baseline = ((raw_lig - baseline)/baseline)*100

        # use ligand-signal baseline with linear & exponential decay function to approximate photobleaching
        baseline_lig = fp_utils.fit_baseline(raw_lig)
        dff_baseline = ((raw_lig - baseline_lig)/baseline_lig)*100

        # use baseline and iso together to calculate fluorescence residual (dF instead of dF/F)
        nans = np.isnan(raw_lig) | np.isnan(raw_iso)

        # first subtract the baseline fit to each signal to correct for photobleaching
        baseline_corr_lig = raw_lig - baseline_lig

        baseline_iso = fp_utils.fit_baseline(raw_iso)
        baseline_corr_iso = raw_iso - baseline_iso

        # scale the isosbestic signal to best fit the ligand-dependent signal
        fitted_baseline_iso = fp_utils.fit_signal(baseline_corr_iso, baseline_corr_lig)

        # then use the baseline corrected signals to calculate dF
        df_baseline_iso = baseline_corr_lig - fitted_baseline_iso

        fp_data[sess_id]['processed_signals'][region] = {
            'raw_lig': raw_lig,
            'raw_iso': raw_iso,
            'baseline_corr_lig': baseline_corr_lig,
            'baseline_corr_iso': baseline_corr_iso,
            'dff_iso': dff_iso,
            'z_dff_iso': utils.z_score(dff_iso),
            'dff_baseline': dff_baseline,
            'z_dff_baseline': utils.z_score(dff_baseline),
            'df_baseline_iso': df_baseline_iso,
            'z_df_baseline_iso': utils.z_score(df_baseline_iso)}


# %% Average signal to the tone presentations, and response, all split by trial outcome

signal_types = ['dff_iso'] # 'baseline_corr_iso',  'dff_iso', 'z_dff_iso', 'dff_baseline', 'z_dff_baseline', 'df_baseline_iso', 'z_df_baseline_iso'

# whether to nomarlize aligned signals on a trial-by-trial basis to the average of the signal in the window before poking in
trial_normalize = False
pre_poke_norm_window = np.array([-0.6, -0.1])

# whether to look at all alignments or only the main ones
main_alignments = True

title_suffix = '420 Iso'

outlier_thresh = 7 # z-score threshold

def create_mat(signal, ts, align_ts, pre, post, windows, sel=[]):
    if trial_normalize:
        return fp_utils.build_trial_dff_signal_matrix(signal, ts, align_ts, pre, post, windows, sel)
    else:
        return fp_utils.build_signal_matrix(signal, ts, align_ts, pre, post, sel)

for sess_id in sess_ids:
    trial_data = sess_data[sess_data['sessid'] == sess_id]
    sess_fp = fp_data[sess_id]

    ts = sess_fp['time']
    trial_start_ts = sess_fp['trial_start_ts'][:-1]
    hit_sel = trial_data['hit'] == True
    miss_sel = trial_data['hit'] == False
    bail_sel = trial_data['bail'] == True
    single_tone_sel = trial_data['n_tones'] == 1
    two_tone_sel = trial_data['n_tones'] == 2

    abs_poke_ts = trial_start_ts + trial_data['cpoke_in_time']
    abs_pre_poke_windows = abs_poke_ts.to_numpy()[:, None] + pre_poke_norm_window[None, :]

    for signal_type in signal_types:

        data_dict = {region: {} for region in regions}
        single_tone_outcome = copy.deepcopy(data_dict)
        two_tone_outcome = copy.deepcopy(data_dict)
        response = copy.deepcopy(data_dict)
        response_cue = copy.deepcopy(data_dict)
        poke_out = copy.deepcopy(data_dict)
        poke_in_prev_outcome = copy.deepcopy(data_dict)
        poke_in_post_outcome = copy.deepcopy(data_dict)
        delay_first_tone = copy.deepcopy(data_dict)
        cport_light_on_prev_outcome = copy.deepcopy(data_dict)
        hits_by_next_resp = copy.deepcopy(data_dict)
        miss_by_next_resp = copy.deepcopy(data_dict)
        miss_by_prev_resp = copy.deepcopy(data_dict)
        miss_by_prev_outcome = copy.deepcopy(data_dict)
        first_tone_prev_outcome = copy.deepcopy(data_dict)

        for region in regions:
            signal = sess_fp['processed_signals'][region][signal_type]

            # aligned to tones by trial outcome
            # single tones
            pre = 1
            post = 2
            single_hit_sel = single_tone_sel & hit_sel
            single_miss_sel = single_tone_sel & miss_sel

            align_ts = trial_start_ts[single_hit_sel] + trial_data.loc[single_hit_sel, 'abs_tone_start_times']
            windows = abs_pre_poke_windows[single_hit_sel,:] - align_ts.to_numpy()[:, None]
            single_tone_outcome[region]['hit'], t = create_mat(signal, ts, align_ts, pre, post, windows)

            align_ts = trial_start_ts[single_miss_sel] + trial_data.loc[single_miss_sel, 'abs_tone_start_times']
            windows = abs_pre_poke_windows[single_miss_sel,:] - align_ts.to_numpy()[:, None]
            single_tone_outcome[region]['miss'], t = create_mat(signal, ts, align_ts, pre, post, windows)

            single_tone_outcome['t'] = t

            # two tones
            pre = 1
            post = 2
            two_hit_sel = two_tone_sel & hit_sel
            two_miss_sel = two_tone_sel & miss_sel

            first_align_ts = trial_start_ts[two_hit_sel] + trial_data.loc[two_hit_sel, 'abs_tone_start_times'].apply(lambda x: x[0])
            last_align_ts = trial_start_ts[two_hit_sel] + trial_data.loc[two_hit_sel, 'abs_tone_start_times'].apply(lambda x: x[1])
            first_windows = abs_pre_poke_windows[two_hit_sel,:] - first_align_ts.to_numpy()[:, None]
            last_windows = abs_pre_poke_windows[two_hit_sel,:] - last_align_ts.to_numpy()[:, None]
            two_tone_outcome[region]['first hit'], t = create_mat(signal, ts, first_align_ts, pre, post, first_windows)
            two_tone_outcome[region]['last hit'], t = create_mat(signal, ts, last_align_ts, pre, post, last_windows)

            first_align_ts = trial_start_ts[two_miss_sel] + trial_data.loc[two_miss_sel, 'abs_tone_start_times'].apply(lambda x: x[0])
            last_align_ts = trial_start_ts[two_miss_sel] + trial_data.loc[two_miss_sel, 'abs_tone_start_times'].apply(lambda x: x[1])
            first_windows = abs_pre_poke_windows[two_miss_sel,:] - first_align_ts.to_numpy()[:, None]
            last_windows = abs_pre_poke_windows[two_miss_sel,:] - last_align_ts.to_numpy()[:, None]
            two_tone_outcome[region]['first miss'], t = create_mat(signal, ts, first_align_ts, pre, post, first_windows)
            two_tone_outcome[region]['last miss'], t = create_mat(signal, ts, last_align_ts, pre, post, last_windows)

            two_tone_outcome['t'] = t

            # response
            pre = 2
            post = 3

            response_ts = trial_start_ts + trial_data['response_time']
            response_windows = abs_pre_poke_windows - response_ts.to_numpy()[:, None]
            response[region]['hit'], t = create_mat(signal, ts, response_ts[hit_sel], pre, post, response_windows[hit_sel])
            response[region]['miss'], t = create_mat(signal, ts, response_ts[miss_sel], pre, post, response_windows[miss_sel])
            response['t'] = t

            # response cue
            pre = 1
            post = 3

            response_cue_ts = trial_start_ts + trial_data['response_cue_time']
            windows = abs_pre_poke_windows - response_cue_ts.to_numpy()[:, None]
            response_cue[region]['hit'], t = create_mat(signal, ts, response_cue_ts[hit_sel], pre, post, windows[hit_sel])
            response_cue[region]['miss'], t = create_mat(signal, ts, response_cue_ts[miss_sel], pre, post, windows[miss_sel])
            response_cue['t'] = t

            # delay period after first tone
            pre = 1
            post = 3

            first_tone_times = trial_data['abs_tone_start_times'].apply(lambda x: x if utils.is_scalar(x) else x[0])
            first_tone_ts = trial_start_ts + first_tone_times
            first_tone_windows = abs_pre_poke_windows - first_tone_ts.to_numpy()[:, None]
            delay_first_tone[region]['hit'], t = create_mat(signal, ts, first_tone_ts[hit_sel], pre, post, first_tone_windows[hit_sel])
            delay_first_tone[region]['miss'], t = create_mat(signal, ts, first_tone_ts[miss_sel], pre, post, first_tone_windows[miss_sel])

            # for bails, only choose trials where first tone is heard
            bail_after_tone_sel = bail_sel & (trial_data['cpoke_out_time'] > first_tone_times + 0.2)
            delay_first_tone[region]['bail'], t = create_mat(signal, ts, first_tone_ts[bail_after_tone_sel], pre, post, first_tone_windows[bail_after_tone_sel])

            delay_first_tone['t'] = t

            # poke in/out based on outcome of current trial
            pre = 3
            post = 3

            poke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
            poke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
            poke_in_windows = abs_pre_poke_windows - poke_in_ts.to_numpy()[:, None]
            poke_out_windows = abs_pre_poke_windows - poke_out_ts.to_numpy()[:, None]

            poke_in_post_outcome[region]['hit'], t = create_mat(signal, ts, poke_in_ts[hit_sel], pre, post, poke_in_windows[hit_sel])
            poke_in_post_outcome[region]['miss'], t = create_mat(signal, ts, poke_in_ts[miss_sel], pre, post, poke_in_windows[miss_sel])
            poke_in_post_outcome[region]['bail'], t = create_mat(signal, ts, poke_in_ts[bail_sel], pre, post, poke_in_windows[bail_sel])
            poke_in_post_outcome['t'] = t

            poke_out[region]['hit'], t = create_mat(signal, ts, poke_out_ts[hit_sel], pre, post, poke_out_windows[hit_sel])
            poke_out[region]['miss'], t = create_mat(signal, ts, poke_out_ts[miss_sel], pre, post, poke_out_windows[miss_sel])
            poke_out[region]['bail'], t = create_mat(signal, ts, poke_out_ts[bail_sel], pre, post, poke_out_windows[bail_sel])
            poke_out['t'] = t

            # poke in and center light on based on previous trial
            pre = 3
            post = 3

            prev_hit_sel = np.insert(hit_sel[:-1].to_numpy(), 0, False)
            prev_miss_sel = np.insert(miss_sel[:-1].to_numpy(), 0, False)
            prev_bail_sel = np.insert(bail_sel[:-1].to_numpy(), 0, False)

            poke_in_prev_outcome[region]['hit'], t = create_mat(signal, ts, poke_in_ts[prev_hit_sel], pre, post, poke_in_windows[prev_hit_sel])
            poke_in_prev_outcome[region]['miss'], t = create_mat(signal, ts, poke_in_ts[prev_miss_sel], pre, post, poke_in_windows[prev_miss_sel])
            poke_in_prev_outcome[region]['bail'], t = create_mat(signal, ts, poke_in_ts[prev_bail_sel], pre, post, poke_in_windows[prev_bail_sel])
            poke_in_prev_outcome['t'] = t

            cport_on_ts = trial_start_ts
            windows = abs_pre_poke_windows - cport_on_ts[:, None]
            cport_light_on_prev_outcome[region]['hit'], t = create_mat(signal, ts, cport_on_ts[prev_hit_sel], pre, post, windows[prev_hit_sel])
            cport_light_on_prev_outcome[region]['miss'], t = create_mat(signal, ts, cport_on_ts[prev_miss_sel], pre, post, windows[prev_miss_sel])
            cport_light_on_prev_outcome[region]['bail'], t = create_mat(signal, ts, cport_on_ts[prev_bail_sel], pre, post, windows[prev_bail_sel])
            cport_light_on_prev_outcome['t'] = t

            if main_alignments:
                continue

            # hits/misses by next response
            pre = 1
            post = 3
            miss_with_next_resp = np.append(miss_sel[:-1].to_numpy() & ~bail_sel[1:].to_numpy(), False)
            hit_with_next_resp = np.append(hit_sel[:-1].to_numpy() & ~bail_sel[1:].to_numpy(), False)
            next_correct_same = np.append(trial_data['correct_port'][:-1].to_numpy() == trial_data['correct_port'][1:].to_numpy(), False)
            next_choice_same = np.append(trial_data['choice'][:-1].to_numpy() == trial_data['choice'][1:].to_numpy(), False)

            miss_by_next_resp[region]['same stay'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (miss_with_next_resp & next_correct_same & next_choice_same))
            miss_by_next_resp[region]['same switch'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (miss_with_next_resp & next_correct_same & ~next_choice_same))
            miss_by_next_resp[region]['diff stay'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (miss_with_next_resp & ~next_correct_same & next_choice_same))
            miss_by_next_resp[region]['diff switch'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (miss_with_next_resp & ~next_correct_same & ~next_choice_same))
            miss_by_next_resp['t'] = t

            hits_by_next_resp[region]['same stay'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (hit_with_next_resp & next_correct_same & next_choice_same))
            hits_by_next_resp[region]['same switch'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (hit_with_next_resp & next_correct_same & ~next_choice_same))
            hits_by_next_resp[region]['diff stay'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (hit_with_next_resp & ~next_correct_same & next_choice_same))
            hits_by_next_resp[region]['diff switch'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (hit_with_next_resp & ~next_correct_same & ~next_choice_same))
            hits_by_next_resp['t'] = t

            # misses by previous response
            pre = 1
            post = 3

            miss_with_prev_resp = np.insert(miss_sel[1:].to_numpy() & ~bail_sel[:-1].to_numpy(), 0, False)
            prev_correct_same = np.insert(trial_data['correct_port'][:-1].to_numpy() == trial_data['correct_port'][1:].to_numpy(), 0, False)
            prev_choice_same = np.insert(trial_data['choice'][:-1].to_numpy() == trial_data['choice'][1:].to_numpy(), 0, False)

            miss_by_prev_resp[region]['same stay'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (miss_with_prev_resp & prev_correct_same & prev_choice_same))
            miss_by_prev_resp[region]['same switch'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (miss_with_prev_resp & prev_correct_same & ~prev_choice_same))
            miss_by_prev_resp[region]['diff stay'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (miss_with_prev_resp & ~prev_correct_same & prev_choice_same))
            miss_by_prev_resp[region]['diff switch'], t = create_mat(signal, ts, response_ts, pre, post, response_windows, (miss_with_prev_resp & ~prev_correct_same & ~prev_choice_same))
            miss_by_prev_resp['t'] = t

            # misses by previous outcome
            pre = 1
            post = 3

            miss_by_prev_outcome[region]['hit'], t = create_mat(signal, ts, response_ts[miss_sel & prev_hit_sel], pre, post, response_windows[miss_sel & prev_hit_sel])
            miss_by_prev_outcome[region]['miss'], t = create_mat(signal, ts, response_ts[miss_sel & prev_miss_sel], pre, post, response_windows[miss_sel & prev_miss_sel])
            miss_by_prev_outcome[region]['bail'], t = create_mat(signal, ts, response_ts[miss_sel & prev_bail_sel], pre, post, response_windows[miss_sel & prev_bail_sel])
            miss_by_prev_outcome['t'] = t

            # first tone by previous outcome and future choice
            pre = 2
            post = 2

            prev_choice_same = np.insert(trial_data['choice'][:-1].to_numpy() == trial_data['choice'][1:].to_numpy(),
                                         0, False)
            prev_tone_same = np.insert(trial_data['correct_port'][:-1].to_numpy() == trial_data['correct_port'][1:].to_numpy(),
                                         0, False)

            first_tone_prev_outcome[region]['stay | same hit'], t = create_mat(signal, ts, first_tone_ts, pre, post, first_tone_windows, (prev_hit_sel & prev_choice_same & prev_tone_same & ~bail_sel))
            first_tone_prev_outcome[region]['stay | diff hit'], t = create_mat(signal, ts, first_tone_ts, pre, post, first_tone_windows, (prev_hit_sel & prev_choice_same & ~prev_tone_same & ~bail_sel))
            first_tone_prev_outcome[region]['stay | same miss'], t = create_mat(signal, ts, first_tone_ts, pre, post, first_tone_windows, (prev_miss_sel & prev_choice_same & prev_tone_same & ~bail_sel))
            first_tone_prev_outcome[region]['stay | diff miss'], t = create_mat(signal, ts, first_tone_ts, pre, post, first_tone_windows, (prev_miss_sel & prev_choice_same & ~prev_tone_same & ~bail_sel))

            first_tone_prev_outcome[region]['switch | same hit'], t = create_mat(signal, ts, first_tone_ts, pre, post, first_tone_windows, (prev_hit_sel & ~prev_choice_same & prev_tone_same & ~bail_sel))
            first_tone_prev_outcome[region]['switch | diff hit'], t = create_mat(signal, ts, first_tone_ts, pre, post, first_tone_windows, (prev_hit_sel & ~prev_choice_same & ~prev_tone_same & ~bail_sel))
            first_tone_prev_outcome[region]['switch | same miss'], t = create_mat(signal, ts, first_tone_ts, pre, post, first_tone_windows, (prev_miss_sel & ~prev_choice_same & prev_tone_same & ~bail_sel))
            first_tone_prev_outcome[region]['switch | diff miss'], t = create_mat(signal, ts, first_tone_ts, pre, post, first_tone_windows, (prev_miss_sel & ~prev_choice_same & ~prev_tone_same & ~bail_sel))
            first_tone_prev_outcome['t'] = t

        # plot alignment results

        # get appropriate labels
        match signal_type: # noqa <-- to disable incorrect error message
            case 'dff_iso':
                signal_type_title = 'Isosbestic Normalized'
                signal_type_label = 'dF/F'
            case 'z_dff_iso':
                signal_type_title = 'Isosbestic Normalized'
                signal_type_label = 'z-scored dF/F'
            case 'dff_baseline':
                signal_type_title = 'Baseline Normalized'
                signal_type_label = 'dF/F'
            case 'z_dff_baseline':
                signal_type_title = 'Baseline Normalized'
                signal_type_label = 'z-scored dF/F'
            case 'df_baseline_iso':
                signal_type_title = 'Baseline-Subtracted Isosbestic Residuals'
                signal_type_label = 'dF residuals'
            case 'z_df_baseline_iso':
                signal_type_title = 'Baseline-Subtracted Isosbestic Residuals'
                signal_type_label = 'z-scored dF residuals'
            case 'baseline_corr_lig':
                signal_type_title = 'Baseline-Subtracted Ligand Signal'
                signal_type_label = 'dF'
            case 'baseline_corr_iso':
                signal_type_title = 'Baseline-Subtracted Isosbestic Signal'
                signal_type_label = 'dF'
            case 'raw_lig':
                signal_type_title = 'Raw Ligand Signal'
                signal_type_label = 'dF/F' if trial_normalize else 'V'
            case 'raw_iso':
                signal_type_title = 'Raw Isosbestic Signal'
                signal_type_label = 'dF/F' if trial_normalize else 'V'

        if trial_normalize:
            signal_type_title += ' - Trial Normalized'

        if title_suffix != '':
            signal_type_title += ' - ' + title_suffix

        all_sub_titles = {'hit': 'Hits', 'miss': 'Misses', 'bail': 'Bails',
                          'first hit': 'First Tone Hits', 'first miss': 'First Tone Misses',
                          'last hit': 'Last Tone Hits', 'last miss': 'Last Tone Misses',
                          'same stay': 'Same Tone, Same Choice', 'same switch': 'Same Tone, Different Choice',
                          'diff stay': 'Different Tone, Same Choice', 'diff switch': 'Different Tone, Different Choice',
                          'stay | same hit': 'Same Response, Previous Hit, Same Tone', 'stay | diff hit': 'Same Response, Previous Hit, Different Tone',
                          'stay | same miss': 'Same Response, Previous Miss, Same Tone', 'stay | diff miss': 'Same Response, Previous Miss, Different Tone',
                          'switch | same hit': 'Different Response, Previous Hit, Same Tone', 'switch | diff hit': 'Different Response, Previous Hit, Different Tone',
                          'switch | same miss': 'Different Response, Previous Miss, Same Tone', 'switch | diff miss': 'Different Response, Previous Miss, Different Tone'}

        fpah.plot_aligned_signals(single_tone_outcome, 'Single Tone Trials - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from tone start (s)', signal_type_label, outlier_thresh=outlier_thresh)

        fpah.plot_aligned_signals(two_tone_outcome, 'Two Tone Trials - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from tone start (s)', signal_type_label, outlier_thresh=outlier_thresh)

        fpah.plot_aligned_signals(response_cue, 'Response Cue Aligned - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from response cue (s)', signal_type_label, outlier_thresh=outlier_thresh)

        fpah.plot_aligned_signals(poke_out, 'Poke Out - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from poke out (s)', signal_type_label, outlier_thresh=outlier_thresh)

        fpah.plot_aligned_signals(response, 'Response Aligned - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh)

        fpah.plot_aligned_signals(poke_in_post_outcome, 'Poke In, Future Outcome - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from poke in (s)', signal_type_label, outlier_thresh=outlier_thresh)

        fpah.plot_aligned_signals(poke_in_prev_outcome, 'Poke In, Previous Outcome - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from poke in (s)', signal_type_label, outlier_thresh=outlier_thresh)

        fpah.plot_aligned_signals(cport_light_on_prev_outcome, 'Light On, Previous Outcome - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from center light on (s)', signal_type_label, outlier_thresh=outlier_thresh)

        fpah.plot_aligned_signals(delay_first_tone, 'Delay After First Tone - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from tone start (s)', signal_type_label, outlier_thresh=outlier_thresh)

        if not main_alignments:

            fpah.plot_aligned_signals(miss_by_next_resp, 'Misses Grouped By Next Tone/Response - {} (session {})'.format(signal_type_title, sess_id),
                                 all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh)

            fpah.plot_aligned_signals(hits_by_next_resp, 'Hits Grouped By Next Tone/Response - {} (session {})'.format(signal_type_title, sess_id),
                                 all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh)

            fpah.plot_aligned_signals(miss_by_prev_resp, 'Misses Grouped By Previous Tone/Response - {} (session {})'.format(signal_type_title, sess_id),
                                 all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh)

            fpah.plot_aligned_signals(miss_by_prev_outcome, 'Misses Grouped By Previous Outcome - {} (session {})'.format(signal_type_title, sess_id),
                                 all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh)

            fpah.plot_aligned_signals(first_tone_prev_outcome, 'First Tone By Choice & Previous Outcome - {} (session {})'.format(signal_type_title, sess_id),
                                 all_sub_titles, 'Time from tone start (s)', signal_type_label, outlier_thresh=outlier_thresh)

# %% Look at some behavioral metrics

sess_metrics = {}

for sess_id in sess_ids:
    sess_metrics[sess_id] = {}

    trial_data = sess_data[sess_data['sessid'] == sess_id]
    hit_sel = trial_data['hit'] == True
    miss_sel = trial_data['hit'] == False
    bail_sel = trial_data['bail'] == True

    prev_hit_sel = np.insert(hit_sel[:-1].to_numpy(), 0, False)
    prev_miss_sel = np.insert(miss_sel[:-1].to_numpy(), 0, False)
    prev_bail_sel = np.insert(bail_sel[:-1].to_numpy(), 0, False)

    prev_choice_same = np.insert(trial_data['choice'][:-1].to_numpy() == trial_data['choice'][1:].to_numpy(),
                                 0, False)
    prev_tone_same = np.insert(trial_data['correct_port'][:-1].to_numpy() == trial_data['correct_port'][1:].to_numpy(),
                                 0, False)

    # probability of outcome given previous outcome
    sess_metrics[sess_id]['p(hit|prev hit)'] = np.sum(hit_sel & prev_hit_sel)/np.sum(prev_hit_sel)
    sess_metrics[sess_id]['p(hit|prev miss)'] = np.sum(hit_sel & prev_miss_sel)/np.sum(prev_miss_sel)
    sess_metrics[sess_id]['p(hit|prev bail)'] = np.sum(hit_sel & prev_bail_sel)/np.sum(prev_bail_sel)

    sess_metrics[sess_id]['p(miss|prev hit)'] = np.sum(miss_sel & prev_hit_sel)/np.sum(prev_hit_sel)
    sess_metrics[sess_id]['p(miss|prev miss)'] = np.sum(miss_sel & prev_miss_sel)/np.sum(prev_miss_sel)
    sess_metrics[sess_id]['p(miss|prev bail)'] = np.sum(miss_sel & prev_bail_sel)/np.sum(prev_bail_sel)

    sess_metrics[sess_id]['p(bail|prev hit)'] = np.sum(bail_sel & prev_hit_sel)/np.sum(prev_hit_sel)
    sess_metrics[sess_id]['p(bail|prev miss)'] = np.sum(bail_sel & prev_miss_sel)/np.sum(prev_miss_sel)
    sess_metrics[sess_id]['p(bail|prev bail)'] = np.sum(bail_sel & prev_bail_sel)/np.sum(prev_bail_sel)

    # stay and switch require animals to make responses on consecutive trials, so they cant bail
    sess_metrics[sess_id]['p(stay|prev hit)'] = np.sum(prev_choice_same & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
    sess_metrics[sess_id]['p(stay|prev miss)'] = np.sum(prev_choice_same & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)

    sess_metrics[sess_id]['p(switch|prev hit)'] = np.sum(~prev_choice_same & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
    sess_metrics[sess_id]['p(switch|prev miss)'] = np.sum(~prev_choice_same & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)

    sess_metrics[sess_id]['p(stay|prev hit & same tone)'] = np.sum(prev_choice_same & prev_hit_sel & ~bail_sel & prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & prev_tone_same)
    sess_metrics[sess_id]['p(stay|prev hit & diff tone)'] = np.sum(prev_choice_same & prev_hit_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & ~prev_tone_same)
    sess_metrics[sess_id]['p(stay|prev miss & same tone)'] = np.sum(prev_choice_same & prev_miss_sel & ~bail_sel & prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & prev_tone_same)
    sess_metrics[sess_id]['p(stay|prev miss & diff tone)'] = np.sum(prev_choice_same & prev_miss_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & ~prev_tone_same)

    sess_metrics[sess_id]['p(switch|prev hit & same tone)'] = np.sum(~prev_choice_same & prev_hit_sel & ~bail_sel & prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & prev_tone_same)
    sess_metrics[sess_id]['p(switch|prev hit & diff tone)'] = np.sum(~prev_choice_same & prev_hit_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & ~prev_tone_same)
    sess_metrics[sess_id]['p(switch|prev miss & same tone)'] = np.sum(~prev_choice_same & prev_miss_sel & ~bail_sel & prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & prev_tone_same)
    sess_metrics[sess_id]['p(switch|prev miss & diff tone)'] = np.sum(~prev_choice_same & prev_miss_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & ~prev_tone_same)

    # probability of hit/miss given previous outcome when responding multiple times in a row
    sess_metrics[sess_id]['p(hit|prev hit & no bail)'] = np.sum(hit_sel & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
    sess_metrics[sess_id]['p(hit|prev miss & no bail)'] = np.sum(hit_sel & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)

    sess_metrics[sess_id]['p(miss|prev hit & no bail)'] = np.sum(miss_sel & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
    sess_metrics[sess_id]['p(miss|prev miss & no bail)'] = np.sum(miss_sel & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)


# %% Plot signals for hits and misses at poke out and response in both regions

signal_type = 'dff_iso' # 'dff_iso', 'df_baseline_iso', 'raw_lig'
regions = ['DMS', 'PFC']

# whether to normalize aligned signals on a trial-by-trial basis to the average of the signal in the window before poking in
trial_normalize = False
pre_poke_norm_window = np.array([-0.6, -0.1])

def create_mat(signal, ts, align_ts, pre, post, windows, sel=[]):
    if trial_normalize:
        return fp_utils.build_trial_dff_signal_matrix(signal, ts, align_ts, pre, post, windows, sel)
    else:
        return fp_utils.build_signal_matrix(signal, ts, align_ts, pre, post, sel)

def stack_mat(old_mat, new_mat):
    if len(old_mat) == 0:
        return new_mat
    else:
        return np.vstack((old_mat, new_mat))

# build data matrices

data_dict = {region: {'hit': [], 'miss': []} for region in regions}
poke_out = copy.deepcopy(data_dict)
response = copy.deepcopy(data_dict)

for sess_id in sess_ids:
    trial_data = sess_data[sess_data['sessid'] == sess_id]
    sess_fp = fp_data[sess_id]

    ts = sess_fp['time']
    trial_start_ts = sess_fp['trial_start_ts'][:-1]
    hit_sel = trial_data['hit'] == True
    miss_sel = trial_data['hit'] == False

    abs_poke_ts = trial_start_ts + trial_data['cpoke_in_time']
    abs_pre_poke_windows = abs_poke_ts.to_numpy()[:, None] + pre_poke_norm_window[None, :]

    poke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
    poke_out_windows = abs_pre_poke_windows - poke_out_ts.to_numpy()[:, None]

    response_ts = trial_start_ts + trial_data['response_time']
    response_windows = abs_pre_poke_windows - response_ts.to_numpy()[:, None]

    for region in regions:
        signal = sess_fp['processed_signals'][region][signal_type]

        # poke out based on outcome of current trial
        pre = 2
        post = 2

        hit_mat, t = create_mat(signal, ts, poke_out_ts, pre, post, poke_out_windows, hit_sel)
        miss_mat, t = create_mat(signal, ts, poke_out_ts, pre, post, poke_out_windows, miss_sel)
        poke_out[region]['hit'] = stack_mat(poke_out[region]['hit'], hit_mat)
        poke_out[region]['miss'] = stack_mat(poke_out[region]['miss'], miss_mat)
        poke_out['t'] = t

        hit_mat, t = create_mat(signal, ts, response_ts, pre, post, response_windows, hit_sel)
        miss_mat, t = create_mat(signal, ts, response_ts, pre, post, response_windows, miss_sel)
        response[region]['hit'] = stack_mat(response[region]['hit'], hit_mat)
        response[region]['miss'] = stack_mat(response[region]['miss'], miss_mat)
        response['t'] = t

# %% Plot Averaged Signals

use_se = True
def calc_error(mat):
    if use_se:
        return utils.stderr(mat, axis=0)
    else:
        return np.std(mat, axis=0)

fig, axs = plt.subplots(2, 2, figsize=(6, 4), layout='constrained')
fig.suptitle(signal_type)

ax = axs[0,0]
region = 'DMS'
ax.set_title(region)
hit_act = utils.z_score(poke_out[region]['hit'])
miss_act = utils.z_score(poke_out[region]['miss'])
plot_utils.plot_psth(np.nanmean(hit_act, axis=0), t, calc_error(hit_act), ax, label='Hits')
plot_utils.plot_psth(np.nanmean(miss_act, axis=0), t, calc_error(miss_act), ax, label='Misses')
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from poke out (s)')

ax = axs[1,0]
ax.sharey(axs[0,0])
hit_act = utils.z_score(response[region]['hit'])
miss_act = utils.z_score(response[region]['miss'])
plot_utils.plot_psth(np.nanmean(hit_act, axis=0), t, calc_error(hit_act), ax, label='Hits')
plot_utils.plot_psth(np.nanmean(miss_act, axis=0), t, calc_error(miss_act), ax, label='Misses')
ax.axvline(0, dashes=[4, 4], c='k', lw=1)
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from response (s)')


ax = axs[0,1]
region = 'PFC'
ax.set_title(region)
hit_act = utils.z_score(poke_out[region]['hit'])
miss_act = utils.z_score(poke_out[region]['miss'])
plot_utils.plot_psth(np.nanmean(hit_act, axis=0), t, calc_error(hit_act), ax, label='Hits')
plot_utils.plot_psth(np.nanmean(miss_act, axis=0), t, calc_error(miss_act), ax, label='Misses')
ax.set_xlabel('Time from poke out (s)')

ax.legend(loc='upper left')

ax = axs[1,1]
ax.sharey(axs[0,1])
hit_act = utils.z_score(response[region]['hit'])
miss_act = utils.z_score(response[region]['miss'])
plot_utils.plot_psth(np.nanmean(hit_act, axis=0), t, calc_error(hit_act), ax, label='Hits')
plot_utils.plot_psth(np.nanmean(miss_act, axis=0), t, calc_error(miss_act), ax, label='Misses')
ax.axvline(0, dashes=[4, 4], c='k', lw=1)
ax.set_xlabel('Time from response (s)')

plt.rcParams["svg.fonttype"] = 'none'
fig.savefig(path.join(utils.get_user_home(), 'downloads', 'dlight_pilot_data.svg'), format='svg')