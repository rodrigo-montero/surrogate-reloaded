import pandas as pd

csv_path = "grid_search_with_ga.csv"
df = pd.read_csv(csv_path)

df_sorted = df.sort_values(by="ga_mean", ascending=False)

top_n = 20 
top_results = df_sorted.head(top_n)

print(top_results)
#top_results.to_csv("top_grid_search_results.csv", index=False)
