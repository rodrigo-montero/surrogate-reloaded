"""
The purpose of this file is to create a series of box plots to analyse the metrics obtained from running the BNN and the MLP
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = "grid_search_results_THE_ONE.csv"
mlp_baseline = {
    "precision": 0.110,
    "recall": 0.477,
    "f_measure": 0.178,
    "auc_roc": 0.697
}

df = pd.read_csv(csv_path)

group_cols = ['model', 'layers', 'learning_rate', 'under', 'oversample',
              'hidden_layer_size', 'test_split', 'batch_size', 'augment', 'weight_loss']

config_means = df.groupby(group_cols)['val_auc'].mean().reset_index()
top_configs = config_means.sort_values(by='val_auc', ascending=False).head(10)

def match_top(row):
    for _, top_row in top_configs.iterrows():
        if all(row[col] == top_row[col] for col in group_cols):
            return True
    return False

df_top = df[df.apply(match_top, axis=1)]

metrics = ['precision', 'recall', 'f_measure', 'auc_roc']
titles = {
    'precision': 'Precision',
    'recall': 'Recall',
    'f_measure': 'F1-score',
    'auc_roc': 'AUC-ROC'
}


fig, axs = plt.subplots(1, len(metrics), figsize=(16, 5))

for i, metric in enumerate(metrics):
    sns.boxplot(data=df_top, y=metric, ax=axs[i], color='skyblue')
    axs[i].axhline(mlp_baseline[metric], color='red', linestyle='--', label='MLP Baseline')
    axs[i].set_title(titles[metric])

    if metric == 'recall':
        axs[i].set_ylim(0.1, 1.05)  # The legend looked weird so added some 'minimum' limits

    axs[i].legend(loc='upper right')

plt.tight_layout()
plt.show()