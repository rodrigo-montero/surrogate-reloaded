"""
This file is responsible for showing the grid search results in an ordered table in the terminal
Write: 'indago/avf/show_grid_search_results.py' in the terminal
"""

import pandas as pd

def show_grid_search_results(file_path="grid_search_results.csv", sort_by="auc_roc"):
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

def show_grid_search_results2(file_path="grid_search_results.csv", sort_by="auc_roc"):
    df = pd.read_csv(file_path)

    print("\n=== Aggregated Grid Search Results (mean ± std) grouped by hyperparameters ===\n")

    group_cols = ['model', 'layers', 'learning_rate', 'under', 'oversample', 'hidden_layer_size', 'test_split', 'batch_size', 'augment', 'weight_loss']

    # Aggregate means and standard deviations
    agg_funcs = {
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f_measure': ['mean', 'std'],
        'auc_roc': ['mean', 'std'],
        'test_loss': ['mean', 'std'],
        'best_epochs': ['mean', 'std'],
    }

    grouped = df.groupby(group_cols).agg(agg_funcs).reset_index()

    # Flatten MultiIndex columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

    # Sort by the chosen metric mean
    sort_col = f"{sort_by}_mean"
    grouped_sorted = grouped.sort_values(by=sort_col, ascending=False)

    # Format display with mean ± std
    def format_metric(mean, std):
        return f"{mean:.4f} ± {std:.4f}"

    # Display the table
    metrics = ['accuracy', 'precision', 'recall', 'f_measure', 'auc_roc', 'test_loss']
    display_cols = group_cols + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]

    print(grouped_sorted[display_cols].to_string(index=False))

    # Summary stats (best config per metric)
    print("\n=== Best Config per Metric ===\n")
    for metric in metrics:
        best_row = grouped_sorted.loc[grouped_sorted[f"{metric}_mean"].idxmax()]
        print(f"{metric.capitalize()}: {best_row[f'{metric}_mean']:.4f} ± {best_row[f'{metric}_std']:.4f}")
        print(f" - Config: layers={best_row['layers']}, lr={best_row['learning_rate']}, oversample={best_row['oversample']}, hidden_layer_size={best_row['hidden_layer_size']}, augment={best_row['augment']}, weight_loss={best_row['weight_loss']}")
        print("")

def show_grid_search_results3(file_path="grid_search_results_THE_ONE.csv", sort_by="val_auc", top_n=50):
    df = pd.read_csv(file_path)

    print(f"\n=== Top {top_n} Grid Search Results by {sort_by}_mean (Grouped by config, averaged over seeds) ===\n")

    # Grouping by all hyperparameters except 'seed'
    group_cols = ['model', 'layers', 'learning_rate', 'under', 'oversample', 
                  'hidden_layer_size', 'test_split', 'batch_size', 'augment', 'weight_loss']

    # Metrics to aggregate
    agg_funcs = {
        'val_auc': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f_measure': ['mean', 'std'],
        'auc_roc': ['mean', 'std'],
        'test_loss': ['mean', 'std'],
        'best_epochs': ['mean', 'std'],
    }

    grouped = df.groupby(group_cols).agg(agg_funcs).reset_index()

    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

    # Sort by chosen metric (default = val_auc_mean)
    sort_col = f"{sort_by}_mean"
    grouped_sorted = grouped.sort_values(by=sort_col, ascending=False)

    # Only show top N results
    grouped_top = grouped_sorted.head(top_n)

    # Metrics to display
    metrics = ['val_auc', 'accuracy', 'precision', 'recall', 'f_measure', 'auc_roc', 'test_loss']

    display_cols = group_cols + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]

    print(grouped_top[display_cols].to_string(index=False))

    # Summary of best config per metric
    print("\n=== Best Config per Metric ===\n")
    for metric in metrics:
        best_row = grouped_sorted.loc[grouped_sorted[f"{metric}_mean"].idxmax()]
        print(f"{metric.capitalize()}: {best_row[f'{metric}_mean']:.4f} ± {best_row[f'{metric}_std']:.4f}")
        print(f" - Config: layers={best_row['layers']}, lr={best_row['learning_rate']}, oversample={best_row['oversample']}, hidden_layer_size={best_row['hidden_layer_size']}, augment={best_row['augment']}, weight_loss={best_row['weight_loss']}")
        print("")

if __name__ == "__main__":
    show_grid_search_results3()
