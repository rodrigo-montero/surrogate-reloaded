"""
This file is responsible for showing the grid search results in an ordered table in the terminal
Write: 'indago/avf/show_grid_search_results.py' in the terminal
"""

import pandas as pd

def show_grid_search_results(file_path="grid_search_results.csv", sort_by="f_measure"):
    df = pd.read_csv(file_path)

    print("\n=== Grid Search Results (sorted by {}) ===\n".format(sort_by))
    df_sorted = df.sort_values(by=sort_by, ascending=False)
    print(df_sorted.to_string(index=False))

    print("\n=== Summary Statistics ===\n")
    metrics = ["accuracy", "precision", "recall", "f_measure", "auc_roc", "test_loss"]
    for metric in metrics:
        if metric in df.columns:
            best_row = df.loc[df[metric].idxmax()]
            print(f"{metric.capitalize()}:")
            print(f" - Mean: {df[metric].mean():.4f}")
            print(f" - Max: {df[metric].max():.4f} (config: layers={best_row['layers']}, lr={best_row['learning_rate']}, oversample={best_row['oversample']}, seed={best_row['seed']})")
            print(f" - Min: {df[metric].min():.4f}")
            print("")

if __name__ == "__main__":
    show_grid_search_results()
