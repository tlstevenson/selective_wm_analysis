# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:36:32 2026

@author: cns-th-lab
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
df = pd.read_csv(r"C:\Users\cns-th-lab\Downloads\RatsLabelGoals - FrameByFrame.csv")
print(df.head())

def get_macro_beh(row):
    print(row)
    micro_beh = row["Behavior"]
    micro_macro_dict = {"Face Forward": "Face Forward",
                        "Grooming (curled)": "Grooming",
                        "Grooming (ears)": "Grooming",
                        "Head Up": "Head Tilt",
                        "Head Down": "Head Tilt",
                        "Head Left": "Head Tilt",
                        "Head Right": "Head Tilt",
                        "Poke Center": "Port Poke",
                        "Poke Left": "Port Poke",
                        "Poke Right": "Port Poke",
                        "Stand Up": "Stand Up",
                        "Walk Left": "Walk",
                        "Walk Right": "Walk"}
    if micro_beh not in micro_macro_dict:
        raise IndexError(f"Value {micro_beh} not in micro_macro_dict with keys {list(micro_macro_dict.keys())}")
    return micro_macro_dict[micro_beh]
df["macro_beh"] = df.apply(lambda row: get_macro_beh(row), axis=1)
#%%
# ---------------------------------------------------------
# 2. Calculate Cross-Tabulations (The Math)
# ---------------------------------------------------------
rat_order = ["198", "402", "237", "199", "238", "274", "400", "424", "483", "234", "419", "421", "422", "235"]
#Cleanup possible space errors and order
df["Subj_id"] = df["Subj_id"].astype(str).str.replace('.0', '', regex=False).str.strip()
df["Subj_id"] = pd.Categorical(df["Subj_id"], categories=rat_order, ordered=True)

rat_vs_beh_abs = pd.crosstab(df['Subj_id'], df['Behavior'])
rat_vs_beh_prop = rat_vs_beh_abs.div(rat_vs_beh_abs.sum(axis=1), axis=0) # Normalize to 1.0

beh_vs_rat_abs = pd.crosstab(df['Behavior'], df['Subj_id'])
beh_vs_rat_prop = beh_vs_rat_abs.div(beh_vs_rat_abs.sum(axis=1), axis=0) # Normalize to 1.0

rat_vs_macro_beh_abs = pd.crosstab(df['Subj_id'], df['macro_beh'])
rat_vs_macro_beh_prop = rat_vs_macro_beh_abs.div(rat_vs_macro_beh_abs.sum(axis=1), axis=0)

macro_beh_vs_rat_abs = pd.crosstab(df['macro_beh'], df['Subj_id'])
macro_beh_vs_rat_prop = macro_beh_vs_rat_abs.div(macro_beh_vs_rat_abs.sum(axis=1), axis=0)
#%% Overall Behavior Plots
df["macro_beh"].value_counts().plot(kind="bar", ylabel="Frame Count", title="Frame Count By Behavior")
plt.xlabel("")
plt.show()

#%%
#%% Overall Behavior Plots
df["Behavior"].value_counts().plot(kind="bar", ylabel="Frame Count", title="Frame Count By Behavior")
plt.xlabel("")
plt.show()
#%% Micro Behavior Plots
#%%% Behavior on x-axis, by rat (in axes)
#Q: Given a rat, are there any behaviors overrepresented?
fig, axes = plt.subplots(7, 2, figsize=(14, 12))
fig.suptitle("Proportion of Labels in Behavior per Rat")
for i in range(14):
    print(rat_vs_beh_prop)
    filtered_rat_beh_prop = rat_vs_beh_prop.loc[rat_order[i]]
    filtered_rat_beh_prop.plot(kind="bar", ax=axes[i%7,i//7])
    axes[i%7,i//7].set_ylabel(f"Proportion Rat:{rat_order[i]}")
    if i%7 !=6:
        axes[i%7,i//7].tick_params(labelbottom=False)
    axes[i%7,i//7].set_xlabel("")
    #axes[i%7,i//7].legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
#%%% Rats on x-axis, by Behavior (Line)
#Q: Are there patterns in labeling
fig, axes = plt.subplots(figsize=(14, 12))
rat_vs_beh_prop.plot(kind='line', ax=axes, colormap='tab20')
axes.set_xticks(range(len(rat_order)))
axes.set_xticklabels(rat_order)
axes.set_title('Proportion of Behaviors per Rat')
axes.set_ylabel('Proportion (0 to 1)')
axes.set_xlabel('Subject ID')

#%%% Behaviors on x-axis, by Rat (Line)
#Which behaviors correspond to which rat?
fig, axes = plt.subplots(figsize=(14, 12))
beh_vs_rat_prop.plot(kind='line', ax=axes, colormap='tab20')
axes.set_title('Proportion of Rats per Behavior')
axes.set_ylabel('Proportion (0 to 1)')
axes.set_xlabel('Behavior')
axes.set_xticks(range(len(beh_vs_rat_prop)))
axes.set_xticklabels(beh_vs_rat_prop.index, rotation=45)
axes.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')

# Clean up layout so legends don't overlap with charts
plt.tight_layout()
plt.xlabel("")
plt.show()

#%% Macro Behavior Plots
#%%% Behavior on x-axis, by rat (in axes)
#Q: Given a rat, are there any behaviors overrepresented?
fig, axes = plt.subplots(7, 2, figsize=(14, 12))
fig.suptitle("Proportion of Labels in Behavior per Rat")
for i in range(14):
    print(rat_vs_beh_prop)
    filtered_rat_macro_beh_prop = rat_vs_macro_beh_prop.loc[rat_order[i]]
    filtered_rat_macro_beh_prop.plot(kind="bar", ax=axes[i%7,i//7])
    axes[i%7,i//7].set_ylabel(f"Proportion Rat:{rat_order[i]}")
    axes[i%7,i//7].set_xlabel("")
    if i%7 !=6:
        axes[i%7,i//7].tick_params(labelbottom=False)
    axes[i%7,i//7].set_xlabel("")
plt.tight_layout()
plt.show()
#%%% Rats on x-axis, by Behavior (Line)
#Q: Are there patterns in labeling
fig, axes = plt.subplots(figsize=(14, 12))
rat_vs_macro_beh_prop.plot(kind='line', ax=axes, colormap='tab20')
axes.set_xticks(range(len(rat_order)))
axes.set_xticklabels(rat_order)
axes.set_title('Proportion of Labels in Behavior per Rat')
axes.set_ylabel('Proportion (0 to 1)')
axes.set_xlabel('Subject ID')

#%%% Behaviors on x-axis, by Rat (Line)
#Which behaviors correspond to which rat?
fig, axes = plt.subplots(figsize=(14, 12))
macro_beh_vs_rat_prop.plot(kind='line', ax=axes, colormap='tab20')
axes.set_title('Behavior Rat Representation')
axes.set_ylabel('Proportion (0 to 1)')
axes.set_xlabel('Behavior')
axes.set_xticks(range(len(macro_beh_vs_rat_prop)))
axes.set_xticklabels(macro_beh_vs_rat_prop.index, rotation=45)
axes.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')

# Clean up layout so legends don't overlap with charts
plt.tight_layout()
plt.show()
#%% Balancing check
# 1. Get the current count for every Rat + Behavior combination
counts = df.groupby(['Subj_id', 'Behavior']).size()

# 2. Find the MAXIMUM number of frames currently existing in any single combination
target_frames = counts.max()

print(f"The target number of frames for an even representation is: {target_frames}")

# 3. Calculate how many frames need to be ADDED to each group to reach that target
frames_to_add = target_frames - counts

# 4. Convert it into a clean, readable DataFrame
df_to_add = frames_to_add.reset_index(name='Frames_to_Add')

print("\nNumber of frames you need to add for each combination:")
print(df_to_add)

#%%
print(np.sum(df_to_add["Frames_to_Add"]))