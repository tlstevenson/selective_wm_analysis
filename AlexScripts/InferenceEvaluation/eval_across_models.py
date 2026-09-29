import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tkinter import Tk, filedialog
import sleap_nn

# --- 1. Missing Data & Gap Profiling (Comparative) ---
def plot_comparative_nans(nan_df):
    plt.figure(figsize=(14, 6))
    sns.barplot(data=nan_df, x="node", y="prop_nan", hue="model_name")
    plt.title("Comparative Proportion of NaNs by Node")
    plt.ylabel("Proportion of Missing Frames")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --- 2. Model Evaluation vs. Ground Truth (Comparative) ---
def get_five_num_summary(scores_df):
    summary = scores_df.groupby(['model_name', 'node'])['score'].describe(
        percentiles=[.25, .5, .75]
    ).reset_index()
    return summary[['model_name', 'node', 'min', '25%', '50%', '75%', 'max']]

def plot_confidence_distributions(scores_df):
    plt.figure(figsize=(16, 8))
    sns.violinplot(
        data=scores_df, x="node", y="score", hue="model_name",
        inner="quartile", density_norm="width", linewidth=1
    )
    plt.title("Confidence Score Distributions Across Models")
    plt.ylabel("Model Confidence Score")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_cv_metrics_comparison(cv_metrics_dict):
    df = pd.DataFrame(cv_metrics_dict).T.reset_index().rename(columns={'index': 'Model'})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.barplot(data=df, x='Model', y='mAP', ax=axes[0], color='steelblue')
    axes[0].set_title('Mean Average Precision (mAP)')
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=df, x='Model', y='mAR', ax=axes[1], color='mediumseagreen')
    axes[1].set_title('Mean Average Recall (mAR)')
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=df, x='Model', y='p90_error_px', ax=axes[2], color='indianred')
    axes[2].set_title('Localization Error (p90 in pixels)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# --- Execution Block ---
if __name__ == "__main__":    
    PREFS_FILE = "script3_prefs.json"
    prefs = {}
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f:
            prefs = json.load(f)

    root = Tk()
    root.withdraw()
    
    # 1. Select Multiple CSVs for NaNs and Scores
    nan_csv_paths = filedialog.askopenfilenames(
        title="Select compiled NaN CSVs for all models",
        initialdir=prefs.get("last_csv_dir", os.getcwd()),
        filetypes=[("CSV Files", "*.csv")]
    )
    
    score_csv_paths = filedialog.askopenfilenames(
        title="Select compiled Score CSVs for all models",
        initialdir=prefs.get("last_csv_dir", os.getcwd()),
        filetypes=[("CSV Files", "*.csv")]
    )
    
    # 2. Select Multiple NPZ files for CV Metrics
    npz_paths = filedialog.askopenfilenames(
        title="Select validation_metrics.npz for all models",
        initialdir=prefs.get("last_npz_dir", os.getcwd()),
        filetypes=[("NPZ Files", "*.npz")]
    )

    if nan_csv_paths and score_csv_paths:
        prefs["last_csv_dir"] = os.path.dirname(nan_csv_paths[0])
        if npz_paths: prefs["last_npz_dir"] = os.path.dirname(npz_paths[0])
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f)
            
        print("Executing Comparative Visualizations...")
        
        # Combine selected CSVs into master dataframes
        master_nan_df = pd.concat([pd.read_csv(p) for p in nan_csv_paths], ignore_index=True)
        master_score_df = pd.concat([pd.read_csv(p) for p in score_csv_paths], ignore_index=True)
        
        plot_comparative_nans(master_nan_df)
        print("\n5-Number Summary:")
        print(get_five_num_summary(master_score_df).head())
        plot_confidence_distributions(master_score_df)
            
    # Load and process actual SLEAP npz metrics instead of mocking
    if npz_paths:
        print("\nExtracting and Plotting Real CV Metrics...")
        real_cv_data = {}
        for path in npz_paths:
            # Assumes directory name is the model name
            model_name = os.path.basename(os.path.dirname(path))
            metrics = sleap_nn.evaluation.load_metrics(path)
            
            real_cv_data[model_name] = {
                'mAP': metrics["voc_metrics"]["oks_voc.mAP"],
                'mAR': metrics["voc_metrics"]["oks_voc.mAR"],
                'p90_error_px': metrics["distance_metrics"]["p90"]
            }
            
        plot_cv_metrics_comparison(real_cv_data)