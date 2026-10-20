# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:45:51 2023

@author: tanne
"""

import init

from sys_neuro_tools import plot_utils
from pyutils import utils
import numpy as np
import pandas as pd


def limit_sess_ids(sess_id_dict, n_back=np.inf, last_idx=0):
    if last_idx == 0:
        sess_ids = {k: ids[-n_back:]
                    if len(ids) >= n_back
                    else ids for k, ids in sess_id_dict.items()}
    else:
        sess_ids = {k: ids[-n_back+last_idx:last_idx]
                    if len(ids) >= n_back-last_idx
                    else ids[:last_idx] for k, ids in sess_id_dict.items()}

    return sess_ids

def get_count_dict(data, groupby_col, count_cols, normalize=False):
    count_dict = {}

    if not utils.is_list(count_cols):
        count_cols = [count_cols]

    pivot_vals = 'proportion' if normalize else 'count'

    groups = data.groupby(groupby_col, as_index=False)
    for col in count_cols:
        count_dict[col] = groups[col].value_counts(normalize=normalize).pivot(
            index=groupby_col, columns=col, values=pivot_vals)

    return count_dict


def get_avg_value_dict(data, value_col, groupby_cols):
    avg_dict = {}

    for col in groupby_cols:
        if type(col) is list:
            # first get individual column rates
            for ind_col in col:
                if not ind_col in avg_dict:
                    avg_dict[ind_col] = calc_avg_info(data, value_col, ind_col)

            # then get joined rates
            key = ' x '.join(col)
            avg_dict[key] = calc_avg_info(data, value_col, col)

        else:
            avg_dict[col] = calc_avg_info(data, value_col, col)

    return avg_dict


def calc_avg_info(data, value_col, groupby_col):

    return data.groupby(groupby_col).agg(avg=(value_col, 'mean'),
           std=(value_col, np.std), se=(value_col, utils.stderr)).infer_objects().reset_index()


def get_rate_dict(data, rate_col, groupby_cols, ci_level=0.95):
    rate_dict = {}

    for col in groupby_cols:
        if type(col) is list:
            # first get individual column rates
            for ind_col in col:
                if not ind_col in rate_dict:
                    rate_dict[ind_col] = calc_rate_info(data, rate_col, ind_col, ci_level)

            # then get joined rates
            key = ' x '.join(col)
            rate_dict[key] = calc_rate_info(data, rate_col, col, ci_level)

        else:
            rate_dict[col] = calc_rate_info(data, rate_col, col, ci_level)

    return rate_dict


def calc_rate_info(data, rate_col, groupby_col, ci_level=0.95):

    # remove any NA or nan values that may be in the rate column and cast to int
    data = data.dropna(subset=rate_col)
    data[rate_col] = data[rate_col].astype(int)

    rate_info = data.groupby(groupby_col).agg(
        n=(rate_col, 'count'), sum=(rate_col, 'sum'), rate=(rate_col, 'mean')).infer_objects().reset_index()

    # compute confidence intervals
    rate_info['ci'] = rate_info.apply(
        lambda r: utils.binom_cis(r['sum'], r['n'], ci_level), axis=1)

    # convert cis to lower + upper error bounds
    rate_info['err'] = rate_info.apply(lambda r: abs(r['ci'] - r['rate']), axis=1)

    return rate_info


def convert_rate_err_to_mat(rate_data):
    return np.asarray(rate_data['err'].to_list()).T


def get_rate_avg_err(bin_mat):

    avg = np.nanmean(bin_mat, axis=0)
    err = np.asarray([abs(utils.binom_cis(np.nansum(bin_mat[:,i]), np.sum(~np.isnan(bin_mat[:,i]))) - avg[i]) for i in range(bin_mat.shape[1])]).T

    return avg, err


def plot_counts(counts, ax, title, y_label, stack, legend_cols=1):
    x_labels = counts.index.to_list()
    val_labels = counts.columns.tolist()
    vals = [counts[k].tolist() for k in val_labels]

    # make sure value labels are strings
    val_labels = [str(l) for l in val_labels]

    plot_utils.plot_stacked_bar(vals, val_labels, x_labels, stack, ax)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(ncols=legend_cols)


def plot_rate_heatmap(rate_dict, column_key, column_name, row_key, row_name, ax=None,
                      fmt='.3f', row_summary=True, col_summary=True, cbar=False,
                      x_rot=0, y_rot=0):

    if ax is None:
        ax = plot_utils.get_axes()

    keys = list(rate_dict.keys())
    key = keys[np.where([column_key in key and row_key in key and 'x' in key for key in keys])[0][0]]
    values = rate_dict[key].reset_index().rename(
        columns={column_key: column_name, row_key: row_name}
    ).pivot(index=row_name, columns=column_name, values='rate')

    # Remove any rows and columns that are all nans
    cols_to_remove = [col_name for col_name, col_values in values.items() if col_values.isnull().values.all()]
    rows_to_remove = [row_name for row_name, row_values in values.iterrows() if row_values.isnull().values.all()]

    values = values.drop(columns=cols_to_remove, index=rows_to_remove)

    # Add in summary statistics for each row and column
    if row_summary:
        values['all'] = rate_dict[row_key].set_index(row_key)['rate']  # Across columns

    if col_summary:
        values.loc['all'] = rate_dict[column_key].set_index(column_key)['rate']  # Down rows

    plot_utils.plot_value_matrix(values, ax=ax, fmt=fmt, cbar=cbar, x_rot=x_rot, y_rot=y_rot)

    # add dividing lines for summaries
    if row_summary:
        ax.axvline(len(values.columns)-1, linewidth=4, color='w')
    if col_summary:
        ax.axhline(len(values)-1, linewidth=4, color='w')


# %% Trial History Methods

def calc_rew_rate_hist(sess_data, n_back=5, kernel='uniform', exclude_bails=True, col_suffix=None):

    hist_cols = ['rew_rate_hist_all', 'rew_rate_hist_left_all', 'rew_rate_hist_right_all',
                 'rew_rate_hist_left_only', 'rew_rate_hist_right_only']

    if not col_suffix is None:
        hist_cols = [c+'_'+col_suffix for c in hist_cols]

    for col in hist_cols:
        sess_data[col] = None

    sess_ids = np.unique(sess_data['sessid'])

    for sess_id in sess_ids:
        sess_sel = sess_data['sessid'] == sess_id
        ind_sess_data = sess_data[sess_sel]
        resp_sel = ind_sess_data['choice'] != 'none'
        if not exclude_bails and 'bail' in sess_data.columns:
            resp_sel = resp_sel | (ind_sess_data['bail'] == True)

        chose_left_sel = ind_sess_data['chose_left']
        chose_right_sel = ind_sess_data['chose_right']

        all_rewards = ind_sess_data['rewarded'].to_numpy()
        right_rewards = all_rewards.copy()
        left_rewards = all_rewards.copy()
        right_rewards[chose_left_sel] = False
        left_rewards[chose_right_sel] = False

        left_only_rewards = left_rewards[chose_left_sel]
        right_only_rewards = right_rewards[chose_right_sel]

        # limit all side reward rates to responses
        all_rewards = all_rewards[resp_sel]
        right_rewards = right_rewards[resp_sel]
        left_rewards = left_rewards[resp_sel]

        all_rew_rate = np.zeros(len(all_rewards))
        left_rew_rate_all = all_rew_rate.copy()
        right_rew_rate_all = all_rew_rate.copy()
        left_only_rew_rate = np.zeros(np.sum(chose_left_sel))
        right_only_rew_rate = np.zeros(np.sum(chose_right_sel))

        # build weighting kernel
        x = np.arange(n_back, 0, -1)-1
        match kernel:
            case 'uniform':
                weights = np.ones_like(x)
            case 'exp':
                weights = np.exp(-x*4/n_back)
        weights = weights/np.sum(weights)

        for i in np.arange(1, len(all_rewards)+1):
            if i >= n_back:
                all_rew_rate[i-1] = np.sum(all_rewards[i-n_back:i]*weights)
                left_rew_rate_all[i-1] = np.sum(left_rewards[i-n_back:i]*weights)
                right_rew_rate_all[i-1] = np.sum(right_rewards[i-n_back:i]*weights)
            else:
                sub_weights = weights[n_back-i:]
                sub_weights = sub_weights/np.sum(sub_weights)
                all_rew_rate[i-1] = np.sum(all_rewards[:i]*sub_weights)
                left_rew_rate_all[i-1] = np.sum(left_rewards[:i]*sub_weights)
                right_rew_rate_all[i-1] = np.sum(right_rewards[:i]*sub_weights)

        for i in np.arange(1, len(left_only_rewards)+1):
            if i >= n_back:
                left_only_rew_rate[i-1] = np.sum(left_only_rewards[i-n_back:i]*weights)
            else:
                sub_weights = weights[n_back-i:]
                sub_weights = sub_weights/np.sum(sub_weights)
                left_only_rew_rate[i-1] = np.sum(left_only_rewards[:i]*sub_weights)

        for i in np.arange(1, len(right_only_rewards)+1):
            if i >= n_back:
                right_only_rew_rate[i-1] = np.sum(right_only_rewards[i-n_back:i]*weights)
            else:
                sub_weights = weights[n_back-i:]
                sub_weights = sub_weights/np.sum(sub_weights)
                right_only_rew_rate[i-1] = np.sum(right_only_rewards[:i]*sub_weights)

        # to fill in the missing values, first fill in a dataframe copy then use ffill
        hist_data = ind_sess_data[hist_cols].copy()
        hist_data.loc[resp_sel, 'rew_rate_hist_all'] = all_rew_rate
        hist_data.loc[resp_sel, 'rew_rate_hist_left_all'] = left_rew_rate_all
        hist_data.loc[resp_sel, 'rew_rate_hist_right_all'] = right_rew_rate_all

        hist_data.loc[chose_left_sel, 'rew_rate_hist_left_only'] = left_only_rew_rate
        hist_data.loc[chose_right_sel, 'rew_rate_hist_right_only'] = right_only_rew_rate

        # fill missing values with the value before it
        hist_data.ffill(inplace=True)
        # fill any remaining nans at the beginning with 0s
        hist_data.fillna(0, inplace=True)

        # shift the rows by one so that current rate is from previous n trials and doesn't include the current outcome
        # this order of operations necessary to correctly propogate missing values for side-only reward rates
        hist_data.iloc[1:,:] = hist_data.iloc[:-1,:]
        hist_data.iloc[0,:] = 0

        # update the original table
        sess_data.loc[sess_sel, hist_cols] = hist_data

    # calculate differences between side rew histories
    sess_data['rew_rate_hist_diff_all'] = sess_data['rew_rate_hist_left_all'] - sess_data['rew_rate_hist_right_all']
    sess_data['rew_rate_hist_diff_only'] = sess_data['rew_rate_hist_left_only'] - sess_data['rew_rate_hist_right_only']


def calc_trial_hist(sess_data, n_back=5, exclude_bails=False, col_suffix=None):

    hist_cols = ['choice_hist', 'rew_hist']

    if not col_suffix is None:
        hist_cols = [c+'_'+col_suffix for c in hist_cols]

    for col in hist_cols:
        sess_data[col] = None
        sess_data[col] = sess_data[col].astype(object)

    sess_ids = np.unique(sess_data['sessid'])

    for sess_id in sess_ids:
        sess_sel = sess_data['sessid'] == sess_id
        ind_sess_data = sess_data[sess_sel]
        resp_sel = ind_sess_data['choice'] != 'none'
        if not exclude_bails and 'bail' in sess_data.columns:
            resp_sel = resp_sel | (ind_sess_data['bail'] == True)

        resp_data = ind_sess_data[resp_sel]

        rewards = resp_data['rewarded'].to_numpy().astype(int)
        choices = resp_data['choice'].to_numpy()

        # make buffered vectors
        buffer = np.full(n_back, np.nan)
        buff_rewards = np.concatenate((buffer, rewards))
        buff_choices = np.concatenate((buffer, choices))
        choice_hist = [buffer]
        rew_hist = [buffer]

        for i in range(sum(resp_sel)-1):
            choice_hist.append(np.flip(buff_choices[i+1:i+n_back+1]))
            rew_hist.append(np.flip(buff_rewards[i+1:i+n_back+1]))

        # to fill in the missing values, first fill in a dataframe copy then use ffill
        hist_data = ind_sess_data[hist_cols].copy()
        # have to use to_numpy to ignore indexes because pandas is too particular
        hist_data.loc[resp_sel, hist_cols[0]] = pd.Series(choice_hist).to_numpy()
        hist_data.loc[resp_sel, hist_cols[1]] = pd.Series(rew_hist).to_numpy()

        # fill missing values with the value before it
        hist_data.ffill(inplace=True)
        # fill any remaining nans at the beginning with the buffer
        nan_sel = hist_data[hist_cols[0]].isna()
        for col in hist_cols:
            hist_data.loc[nan_sel, col] = pd.Series([buffer]).repeat(sum(nan_sel)).to_numpy()

        # update the original table
        sess_data.loc[sess_sel, hist_cols] = hist_data

def trial_hist_exists(sess_data):
    return set(['choice_hist', 'rew_hist']).issubset(sess_data.columns)

# %% Other behavior parsing

def get_beh_trial_start_ts(sess_data):
    ''' Gets psuedo trial start times relative to the first trial start for each session '''

    sess_ids = sess_data['sessid'].unique()

    trial_start_ts = {}

    for sess_id in sess_ids:
        trial_data = sess_data[sess_data['sessid'] == sess_id]

        start_times = trial_data['trialtime']
        trial_start_ts[sess_id] = start_times.apply(lambda x: (x - start_times.iloc[0]).total_seconds()).to_numpy()

    return trial_start_ts


def get_wm_trial_outcome(sess_data):
    sess_data['outcome_type'] = ''
    sess_data.loc[sess_data['hit'] == True, 'outcome_type'] = 'hit'
    sess_data.loc[sess_data['hit'] == False, 'outcome_type'] = 'miss'
    sess_data.loc[sess_data['bail'] == True, 'outcome_type'] = 'bail'

def get_bandit_trial_outcome(sess_data):
    sess_data['outcome_type'] = ''
    sess_data.loc[sess_data['rewarded'] == True, 'outcome_type'] = 'rewarded'
    sess_data.loc[sess_data['rewarded'] == False, 'outcome_type'] = 'unrewarded'

def calc_trial_creation_latency(sess_data, trial_start_ts):
    # calculate the latency between the last state and the start of the next trial
    sess_data['next_trial_start_latency'] = np.nan
    sess_ids = np.unique(sess_data['sessid'])

    for sess_id in sess_ids:
        sess_sel = sess_data['sessid'] == sess_id
        ind_sess_data = sess_data[sess_sel]
        sess_trial_start_ts = trial_start_ts[sess_id]

        for i in range(len(ind_sess_data)):
            current_trial = ind_sess_data.iloc[i]
            current_states = current_trial['parsed_events']['States']
            trial_start_dt = sess_trial_start_ts[i+1] - sess_trial_start_ts[i]

            last_event = 0
            for state_times in current_states.values():
                if not all([s is None for s in state_times]):
                    last_state = max(state_times)
                    if last_state > last_event:
                        last_event = last_state

            ind_sess_data.iloc[i, ind_sess_data.columns.get_loc('next_trial_start_latency')] = trial_start_dt - last_event

        # update the original table
        sess_data.loc[sess_sel, ['next_trial_start_latency']] = ind_sess_data[['next_trial_start_latency']]


def fill_missing_cpoke_times(sess_data, trial_start_ts, latency_thresh=0.3):
    # fill missing cpoke in and out times due to pokes in and out spread across trials

    if not 'next_trial_start_latency' in sess_data.columns:
        calc_trial_creation_latency(sess_data, trial_start_ts)

    sess_ids = np.unique(sess_data['sessid'])

    for sess_id in sess_ids:
        sess_sel = sess_data['sessid'] == sess_id
        ind_sess_data = sess_data[sess_sel]
        sess_trial_start_ts = trial_start_ts[sess_id]

        for i in range(len(ind_sess_data)):
            current_trial = ind_sess_data.iloc[i]
            current_peh = current_trial['parsed_events']

            # attempt to fill missing cpoke in times, this happens if they poke in from the previous trial or during the gap between trials
            if pd.isna(current_trial['cpoke_in_time']) and not pd.isna(current_trial['cpoke_out_time']) and i > 0:
                prev_trial = ind_sess_data.iloc[i-1]

                if prev_trial['next_trial_start_latency'] > latency_thresh:
                    continue

                prev_peh = prev_trial['parsed_events']
                trial_start_dt = sess_trial_start_ts[i] - sess_trial_start_ts[i-1]
                prev_cpoke_in = np.atleast_1d(prev_peh['Events'].get('Port2In', [])) - trial_start_dt
                prev_cpoke_out = np.atleast_1d(prev_peh['Events'].get('Port2Out', [])) - trial_start_dt

                valid_cpoke_in = []
                if len(prev_cpoke_out) > 0:
                    valid_cpoke_in = prev_cpoke_in[prev_cpoke_in > prev_cpoke_out[-1]]

                if len(valid_cpoke_in) > 0:
                    ind_sess_data.iloc[i, ind_sess_data.columns.get_loc('cpoke_in_time')] = valid_cpoke_in[0]

            # fill missing cpoke out time, this happens if they poke out almost immediately after poking in or poke out on the next trial
            if pd.isna(current_trial['cpoke_out_time']) and not pd.isna(current_trial['cpoke_in_time']):
                cpoke_in = current_trial['cpoke_in_time']
                cpoke_out = np.atleast_1d(current_peh['Events'].get('Port2Out', []))

                valid_cpoke_out = cpoke_out[cpoke_out > cpoke_in]

                if len(valid_cpoke_out) == 0 and i < len(ind_sess_data) and current_trial['next_trial_start_latency'] <= latency_thresh:

                    # see if there is a cpoke out on the next trial
                    trial_start_dt = sess_trial_start_ts[i+1] - sess_trial_start_ts[i]
                    next_trial = ind_sess_data.iloc[i+1]
                    next_peh = next_trial['parsed_events']
                    next_cpoke_in = trial_start_dt + np.atleast_1d(next_peh['Events'].get('Port2In', []))
                    next_cpoke_out = trial_start_dt + np.atleast_1d(next_peh['Events'].get('Port2Out', []))

                    if len(next_cpoke_in) > 0:
                        valid_cpoke_out = next_cpoke_out[next_cpoke_out < next_cpoke_in[-1]]
                    else:
                        valid_cpoke_out = next_cpoke_out

                if len(valid_cpoke_out) > 0:
                    ind_sess_data.iloc[i, ind_sess_data.columns.get_loc('cpoke_out_time')] = valid_cpoke_out[0]

        # update the original table
        sess_data.loc[sess_sel, ['cpoke_in_time', 'cpoke_out_time']] = ind_sess_data[['cpoke_in_time', 'cpoke_out_time']]


def get_bail_response_info(sess_data, trial_start_ts, resp_time_thresh=10, latency_thresh=0.3):
    ''' Parse bail events to determine the side and time of a response poke after a bail '''

    if not 'next_trial_start_latency' in sess_data.columns:
        calc_trial_creation_latency(sess_data, trial_start_ts)

    sess_ids = np.unique(sess_data['sessid'])

    for sess_id in sess_ids:
        sess_sel = sess_data['sessid'] == sess_id
        ind_sess_data = sess_data[sess_sel]
        sess_trial_start_ts = trial_start_ts[sess_id]

        bail_sel = ind_sess_data['bail'] == True

        for i in range(len(bail_sel)-1):
            if bail_sel.iloc[i]:
                current_trial = ind_sess_data.iloc[i]
                next_trial = ind_sess_data.iloc[i+1]

                # time difference between trial starts
                trial_start_dt = sess_trial_start_ts[i+1] - sess_trial_start_ts[i]

                response_time = np.nan
                choice = 'none'

                current_peh = current_trial['parsed_events']
                next_peh = next_trial['parsed_events']
                bail_time = current_trial['cpoke_out_time']

                # see if there is a response poke on the current trial
                left_pokes = np.atleast_1d(current_peh['Events'].get('Port1In', []))
                right_pokes = np.atleast_1d(current_peh['Events'].get('Port3In', []))

                post_bail_left_pokes = left_pokes[left_pokes > bail_time]
                post_bail_right_pokes = right_pokes[right_pokes > bail_time]

                if len(post_bail_left_pokes) == 0 and len(post_bail_right_pokes) == 0 and current_trial['next_trial_start_latency'] <= latency_thresh:

                    # see if there is a response poke on the next trial
                    left_pokes = trial_start_dt + np.atleast_1d(next_peh['Events'].get('Port1In', []))
                    right_pokes = trial_start_dt + np.atleast_1d(next_peh['Events'].get('Port3In', []))

                    next_cpoke = trial_start_dt + next_trial['cpoke_in_time']
                    resp_poke_thresh = min(bail_time + resp_time_thresh, next_cpoke)
                    post_bail_left_pokes = left_pokes[left_pokes < resp_poke_thresh]
                    post_bail_right_pokes = right_pokes[right_pokes < resp_poke_thresh]

                if len(post_bail_left_pokes) > 0 and len(post_bail_right_pokes) > 0:
                    first_left = post_bail_left_pokes[0]
                    first_right = post_bail_right_pokes[0]

                    if first_left < first_right:
                        choice = 'left'
                        response_time = first_left
                    else:
                        choice = 'right'
                        response_time = first_right

                elif len(post_bail_left_pokes) > 0:
                    choice = 'left'
                    response_time = post_bail_left_pokes[0]

                elif len(post_bail_right_pokes) > 0:
                    choice = 'right'
                    response_time = post_bail_right_pokes[0]

                # update the sub table
                ind_sess_data.iloc[i, ind_sess_data.columns.get_loc('choice')] = choice
                ind_sess_data.iloc[i, ind_sess_data.columns.get_loc('response_time')] = response_time

        # update the original table
        sess_data.loc[sess_sel, ['choice', 'response_time']] = ind_sess_data[['choice', 'response_time']]


def calc_lick_duration(sess_data, trial_start_ts):

    sess_ids = np.unique(sess_data['sessid'])
    sess_data['lick_dur'] = np.nan
    sess_data['lick_stop_time'] = np.nan

    for sess_id in sess_ids:
        sess_sel = sess_data['sessid'] == sess_id
        ind_sess_data = sess_data[sess_sel]
        sess_trial_start_ts = trial_start_ts[sess_id]

        resp_sel = ~pd.isna(ind_sess_data['response_time'])

        for i in range(len(resp_sel)-1):
            if resp_sel.iloc[i]:
                current_trial = ind_sess_data.iloc[i]
                next_trial = ind_sess_data.iloc[i+1]

                # time difference between trial starts
                trial_start_dt = sess_trial_start_ts[i+1] - sess_trial_start_ts[i]

                choice = current_trial['choice']
                resp_time = current_trial['response_time']
                outcome_time = current_trial['reward_time']
                current_peh = current_trial['parsed_events']
                next_peh = next_trial['parsed_events']
                next_cpoke = trial_start_dt + next_trial['cpoke_in_time']

                left_poke_ins = np.atleast_1d(current_peh['Events'].get('Port1In', [np.inf]))
                right_poke_ins = np.atleast_1d(current_peh['Events'].get('Port3In', [np.inf]))
                left_poke_outs = np.atleast_1d(current_peh['Events'].get('Port1Out', []))
                right_poke_outs = np.atleast_1d(current_peh['Events'].get('Port3Out', []))

                left_poke_ins = left_poke_ins[left_poke_ins >= resp_time]
                right_poke_ins = right_poke_ins[right_poke_ins >= resp_time]
                left_poke_outs = left_poke_outs[left_poke_outs > resp_time]
                right_poke_outs = right_poke_outs[right_poke_outs > resp_time]

                left_poke_ins = np.concatenate((left_poke_ins, trial_start_dt + np.atleast_1d(next_peh['Events'].get('Port1In', [np.inf]))))
                right_poke_ins = np.concatenate((right_poke_ins, trial_start_dt + np.atleast_1d(next_peh['Events'].get('Port3In', [np.inf]))))
                left_poke_outs = np.concatenate((left_poke_outs, trial_start_dt + np.atleast_1d(next_peh['Events'].get('Port1Out', []))))
                right_poke_outs = np.concatenate((right_poke_outs, trial_start_dt + np.atleast_1d(next_peh['Events'].get('Port3Out', []))))

                # find the last poke out before a poke in any other port

                if choice == 'left':
                    next_poke = min(min(right_poke_ins), next_cpoke)
                    valid_poke_outs = left_poke_outs[left_poke_outs < next_poke]
                else:
                    next_poke = min(min(left_poke_ins), next_cpoke)
                    valid_poke_outs = right_poke_outs[right_poke_outs < next_poke]

                # if no valid poke outs (they poke out while next trial state machine is being constructed), set lick duration to 0
                if len(valid_poke_outs) == 0:
                    valid_poke_outs = [resp_time]

                last_poke_out = max(valid_poke_outs)

                if 'bail' in current_trial.index and current_trial['bail'] == True:
                    # bail lick durations are from response to last poke out
                    lick_dur = max(0, last_poke_out - resp_time)
                else:
                    # response lick durations are from outcome cue to last poke out
                    lick_dur = max(0, last_poke_out - outcome_time)

                # update the sub table
                ind_sess_data.iloc[i, ind_sess_data.columns.get_loc('lick_dur')] = lick_dur
                ind_sess_data.iloc[i, ind_sess_data.columns.get_loc('lick_stop_time')] = last_poke_out

        # update the original table
        sess_data.loc[sess_sel, ['lick_dur', 'lick_stop_time']] = ind_sess_data[['lick_dur', 'lick_stop_time']]

def calc_action_cpoke_latency(sess_data, trial_start_ts):
    # calculate the cpoke latency from the last action (lick end, cpoke out, oucome tone)

    if not 'lick_stop_time' in sess_data.columns:
        calc_lick_duration(sess_data, trial_start_ts)

    sess_ids = np.unique(sess_data['sessid'])
    sess_data['action_cpoke_in_latency'] = np.nan
    sess_data['next_action_cpoke_in_latency'] = np.nan

    for sess_id in sess_ids:
        sess_sel = sess_data['sessid'] == sess_id
        ind_sess_data = sess_data[sess_sel]
        sess_trial_start_ts = trial_start_ts[sess_id]

        for i in range(len(ind_sess_data)-1):
            current_trial = ind_sess_data.iloc[i]
            next_trial = ind_sess_data.iloc[i+1]

            # time difference between trial starts
            trial_start_dt = sess_trial_start_ts[i+1] - sess_trial_start_ts[i]

            resp_time = current_trial['response_time']
            cpoke_out_time = current_trial['cpoke_out_time']
            bail_tone_time = current_trial.get('bail_tone_time', default=np.nan)
            outcome_time = current_trial['reward_time']
            lick_stop_time = current_trial['lick_stop_time']
            if pd.isna(next_trial['cpoke_in_time']):
                next_cpoke = np.nan
            else:
                next_cpoke = trial_start_dt + next_trial['cpoke_in_time']

            if not pd.isna(lick_stop_time):
                cpoke_latency = next_cpoke - lick_stop_time
            elif not pd.isna(outcome_time):
                cpoke_latency = next_cpoke - outcome_time
            elif not pd.isna(resp_time):
                cpoke_latency = next_cpoke - resp_time
            elif not pd.isna(bail_tone_time):
                cpoke_latency = next_cpoke - bail_tone_time
            elif not pd.isna(cpoke_out_time):
                cpoke_latency = next_cpoke - cpoke_out_time
            else:
                cpoke_latency = np.nan

            # update the sub table
            ind_sess_data.iloc[i, ind_sess_data.columns.get_loc('next_action_cpoke_in_latency')] = cpoke_latency
            ind_sess_data.iloc[i+1, ind_sess_data.columns.get_loc('action_cpoke_in_latency')] = cpoke_latency

        # update the original table
        sess_data.loc[sess_sel, ['action_cpoke_in_latency', 'next_action_cpoke_in_latency']] = ind_sess_data[['action_cpoke_in_latency', 'next_action_cpoke_in_latency']]


def calc_wm_stay_switch(sess_data):

    fields = ['chose_left', 'chose_right', 'prev_choice', 'next_choice',
              'switch', 'stay', 'next_switch', 'next_stay']
    sess_ids = np.unique(sess_data['sessid'])

    for sess_id in sess_ids:
        sess_sel = sess_data['sessid'] == sess_id
        ind_sess_data = sess_data[sess_sel]

        choices = ind_sess_data['choice'].to_numpy()
        ind_sess_data['chose_left'] = choices == 'left'
        ind_sess_data['chose_right'] = choices == 'right'

        # Previous and future trial information

        ind_sess_data['prev_choice'] = None
        ind_sess_data.iloc[1:, ind_sess_data.columns.get_loc('prev_choice')] = choices[:-1]

        ind_sess_data['next_choice'] = None
        ind_sess_data.iloc[:-1, ind_sess_data.columns.get_loc('next_choice')] = choices[1:]

        ind_sess_data['switch'] = False
        ind_sess_data['stay'] = False
        ind_sess_data['next_switch'] = False
        ind_sess_data['next_stay'] = False

        resp_sel = choices != 'none'
        repeat_resp = resp_sel[1:] & resp_sel[:-1]
        stays = repeat_resp & (choices[1:] == choices[:-1])
        switches = repeat_resp & (choices[1:] != choices[:-1])
        ind_sess_data.iloc[1:, ind_sess_data.columns.get_loc('switch')] = switches
        ind_sess_data.iloc[1:, ind_sess_data.columns.get_loc('stay')] = stays
        ind_sess_data.iloc[:-1, ind_sess_data.columns.get_loc('next_switch')] = switches
        ind_sess_data.iloc[:-1, ind_sess_data.columns.get_loc('next_stay')] = stays

        # update the original table
        sess_data.loc[sess_sel, fields] = ind_sess_data[fields]