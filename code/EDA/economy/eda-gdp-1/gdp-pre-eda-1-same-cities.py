import pandas as pd
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Read files
file_path_traffic = r'dft_traffic_counts_raw_counts.csv'
df_traffic = pd.read_csv(file_path_traffic, low_memory=False)

file_path_eco = r"Regional gross domestic product(all ITL).xlsx" 
df_eco = pd.read_excel(file_path_eco, sheet_name="Table 5", header=1) # GDP at current market prices

# Get unique city names from traffic and economic data
unique_authorities = df_traffic["local_authority_name"].dropna().unique()
region_names = df_eco["Region name"].dropna().unique()

# Set operations to find common and unique values
set_traffic = set(unique_authorities)
set_eco = set(region_names)

common = set_traffic.intersection(set_eco)
only_traffic = set_traffic.difference(set_eco)
only_eco = set_eco.difference(set_traffic)

# Set similarity threshold
threshold = 0.8 

# Use list comprehension to find similar name pairs
similar_pairs = [
    (name_traffic, name_eco, similar(name_traffic, name_eco))
    for name_traffic in only_traffic
    for name_eco in only_eco
    if similar(name_traffic, name_eco) >= threshold
]

# Output similar but not exactly matched name pairs
# print("Name pairs that are similar but not identified as identical (similarity >= 0.8):")
# for name_traffic, name_eco, ratio in similar_pairs:
#     print(f"traffic set: '{name_traffic}'  <-->  economic set: '{name_eco}'  similarity: {ratio:.2f}")

# Get the matching results
common_traffic = common.union({pair[0] for pair in similar_pairs})
common_eco = common.union({pair[1] for pair in similar_pairs})

# Extract corresponding data and form new tables
df_traffic_common = df_traffic[df_traffic["local_authority_name"].isin(common_traffic)]
df_eco_common = df_eco[df_eco["Region name"].isin(common_eco)]

print("\nTraffic new table row count:", df_traffic_common.shape[0])
print("Economic new table row count:", df_eco_common.shape[0])
print(f"Datasets with same city names created.")

# Save the new tables as CSV files
df_traffic_common.to_csv("Traffic_common_data.csv", index=False)
df_eco_common.to_csv("Economic_common_data.csv", index=False)
