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


def limit_sess_ids(sess_id_dict, n_back, last_idx=0):
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

    pivot_vals = 'proportion' if normalize else 'count'

    groups = data.groupby(groupby_col, as_index=False)
    for col in count_cols:
        count_dict[col] = groups[col].value_counts(normalize=normalize).pivot(
            index=groupby_col, columns=col, values=pivot_vals)

    return count_dict


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

    rate_info = data.groupby(groupby_col).agg(
        n=(rate_col, 'count'), sum=(rate_col, 'sum'), rate=(rate_col, 'mean')).infer_objects()
    
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


def plot_counts(counts, ax, title, y_label, stack):
    x_labels = counts.index.to_list()
    val_labels = counts.columns.tolist()
    vals = [counts[k].tolist() for k in val_labels]

    plot_utils.plot_stacked_bar(vals, val_labels, x_labels, stack, ax)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def plot_rate_heatmap(rate_dict, column_key, column_name, row_key, row_name, ax=None,
                      fmt='.3f', row_summary=True, col_summary=True, cbar=False,
                      x_rot=0, y_rot=0):
    
    if ax is None:
        ax = plot_utils.get_axes()
        
    keys = list(rate_dict.keys())
    key = keys[np.where([column_key in key and row_key in key for key in keys])[0][0]]
    values = rate_dict[key].reset_index().rename(
        columns={column_key: column_name, row_key: row_name}
    ).pivot(index=row_name, columns=column_name, values='rate')

    # Remove any rows and columns that are all nans
    cols_to_remove = [col_name for col_name, col_values in values.items() if col_values.isnull().values.all()]
    rows_to_remove = [row_name for row_name, row_values in values.iterrows() if row_values.isnull().values.all()]
        
    values = values.drop(columns=cols_to_remove, index=rows_to_remove)

    # Add in summary statistics for each row and column
    if row_summary:
        values['all'] = rate_dict[row_key]['rate']  # Across columns

    if col_summary:
        values.loc['all'] = rate_dict[column_key]['rate']  # Down rows

    plot_utils.plot_value_matrix(values, ax=ax, fmt=fmt, cbar=cbar, x_rot=x_rot, y_rot=y_rot)

    # add dividing lines for summaries
    if row_summary:
        ax.axvline(len(values.columns)-1, linewidth=4, color='w')
    if col_summary:
        ax.axhline(len(values)-1, linewidth=4, color='w')


