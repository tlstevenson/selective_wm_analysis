# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:59:38 2024

@author: tanne
"""

# %%
import os.path as path
import fp_analysis_helpers as fpah
from fp_analysis_helpers import Alignment as Align
import pyutils.utils as utils

fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), 'All Sessions'), group_by=['alignment', 'behavior', 'subject'], alignments='sess')

fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), '179 By Alignment'), group_by=['subject', 'alignment', 'behavior'], subj_ids=179)
fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), '179 By Behavior'), group_by=['subject', 'behavior', 'alignment'], subj_ids=179)
fpah.generate_figure_ppt(path.join(fpah.get_base_figure_save_path(), '179 By Filename'), group_by=['subject', 'alignment', 'filename', 'behavior'], subj_ids=179)
