import pandas as pd

# Load the CSV file
csv_path = "grid_search_with_ga_Combination.csv"  # Replace with your actual file path
df = pd.read_csv(csv_path)

# Sort by GA_mean in descending order
df_sorted = df.sort_values(by="ga_mean", ascending=False)

# Display top X entries
top_n = 20  # Change this number to display more or fewer rows
top_results = df_sorted.head(top_n)

# Print the top results
print(top_results)

# Optional: Save to CSV
top_results.to_csv("top_grid_search_results.csv", index=False)
