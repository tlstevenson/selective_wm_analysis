# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:59:38 2024

@author: tanne
"""

# %%
import init
import os.path as path
import fp_analysis_helpers as fpah
from fp_analysis_helpers import Alignment as Align
import pyutils.utils as utils

#fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'All Sessions'), group_by=['alignment', 'behavior', 'subject'], alignments='sess')

# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), '179 By Alignment'), group_by=['subject', 'alignment', 'behavior'], subjects=179)
# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), '179 By Behavior'), group_by=['subject', 'behavior', 'alignment'], subjects=179)
# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), '179 By Filename'), group_by=['subject', 'alignment', 'filename', 'behavior'], subjects=179)
# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Two-armed Bandit - 179'), group_by=['subject','behavior', 'alignment', 'filename'], behaviors='Two-armed Bandit', subjects=179)

# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Two-armed Bandit - 182'), group_by=['subject','behavior', 'alignment', 'filename'], behaviors='Two-armed Bandit', subjects=182)
# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Two-armed Bandit - 191'), group_by=['subject','behavior', 'alignment', 'filename'], behaviors='Two-armed Bandit', subjects=191)

# By Behavior, all subjects where subjects are adjacent
# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'SelWM - Grow Delay - all'), group_by=['behavior', 'alignment', 'filename', 'subject'], behaviors='SelWM - Grow Delay')
# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'SelWM - Grow Nosepoke - all'), group_by=['behavior', 'alignment', 'filename', 'subject'], behaviors='SelWM - Grow Nosepoke')
# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'SelWM - Two Tones - all'), group_by=['behavior', 'alignment', 'filename', 'subject'], behaviors='SelWM - Two Tones')
# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Two-armed Bandit - all'), group_by=['behavior', 'alignment', 'filename', 'subject'], behaviors='Two-armed Bandit')
# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Intertemporal Choice - all'), group_by=['behavior', 'alignment', 'filename', 'subject'], behaviors='Intertemporal Choice')
# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Foraging - all'), group_by=['behavior', 'alignment', 'filename', 'subject'], behaviors='Foraging')
# fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Operant Conditioning - all'), group_by=['behavior', 'alignment', 'filename', 'subject'], behaviors='Operant Pavlovian Conditioning')

# By Alignment across all subjects and behaviors are adjacent
# for a in Align:
#     fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), '{} - all by subj'.format(fpah.get_align_title(a))), group_by=['alignment', 'subject', 'behavior', 'filename'], alignments = a)
#     fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), '{} - all by beh filename'.format(fpah.get_align_title(a))), group_by=['alignment', 'behavior', 'filename', 'subject'], alignments = a)

# By subject across all behaviors and alignments
subjects = [179, 180, 182, 188, 191, 202, 207]
for sub in subjects:
    fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), '{}'.format(sub)), group_by=['subject', 'alignment', 'filename', 'behavior'], subjects = sub)

#%%
interest_fnames = {
    Align.cpoke_in: ['stay_switch_side', 'prev_future_outcome', 'prev_side_prev_outcome', 'prev_outcome_choice_side'],
    Align.tone: ['choice_side', 'rew_tone_vol_by_rew', 'rew_tone_vol_by_vol', 'tone_type_side', 'tone_type_choice_side', 'tone_vol_choice_by_choice',
                 'tone_vol_tone_type_choice', 'position_side_*', 'one_tone_side_outcome_*', 'two_tone_side_outcome_*', 'two_tone_side_outcome_no_offset_*'],
    Align.cue: ['stay_switch_side', 'prev_outcome_choice_side', 'rew_hist_diff_all', 'rew_hist_all_prev_side', 'rew_hist_side_only_prev_side', 'rew_hist_diff_only_prev_side',
                'side_prev_side_prev_outcome', 'choice_block_prob_side', 'reward_vol_side', 'reward_rate_trial_length_side', 'prev_side_prev_reward',
                'choice_prev_choice', 'prev_rew_side', 'prev_rew_harvest_init_vol', 'choice_side', 'side_outcome', 'cor_side_repeat_stay_switch_side',
                'prev_outcome_stay_switch_side'],
    Align.cpoke_out: ['stay_switch_side', 'prev_future_outcome', 'side_outcome', 'prev_outcome_choice_side', 'rew_hist_diff_all',
                      'side_prev_side_prev_outcome'],
    Align.resp: ['stay_switch_side','choice_prev_choice', 'rew_prev_choice', 'rew_side', 'reward_rate_trial_length', 'tone_type_side',
                 'tone_type_side_averse', 'tone_type_side_both_conditions', 'side_outcome', 'side_stay_switch_outcome', 'prev_future_outcome',
                 'resp_delay_outcome', 'resp_delay_side_outcome', 'cor_side_repeat_stay_switch_outcome', 'choice_block_prob_outcome',
                 'choice_block_prob_side_outcome', 'choice_block_prob_side_rewarded_norm', 'rew_hist_all_outcome', 'rew_hist_side_only_outcome',
                 'side_outcome_future_stay_switch', 'side_outcome_prev_outcome', 'side_prev_side_outcome_prev_outcome'],
    Align.reward: ['stay_switch_side', 'reward_delay_side', 'reward_rate_trial_length', 'reward_rate_trial_length_side', 'reward_vol_side',
                   'trial_length_stay_switch_or_side']}

alignments = list(interest_fnames.keys())

fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Interesting groupings by subject'),
                         group_by=['alignment', 'behavior', 'subject', 'filename'], alignments=alignments, filenames=utils.flatten(interest_fnames))

fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Interesting groupings by filename'),
                         group_by=['alignment', 'behavior', 'filename', 'subject'], alignments=alignments, filenames=utils.flatten(interest_fnames))

# %%

fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Power Spectra'),
                         group_by=['behavior', 'subject', 'filename'], behaviors='Power Spectra', alignments='')

fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Reward Responses - masked'),
                         group_by=['behavior', 'subject', 'filename'], behaviors='Reward Comparison', alignments='')

fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'Response Cue Responses - masked'),
                         group_by=['behavior', 'subject', 'filename'], behaviors='Response Cue Comparison', alignments='')
