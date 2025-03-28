import pandas as pd

"""
1. create columns for month and day from count_date
2. traffic data cleaning:
    2.1 drop columns with NaN values over 50 raws: link_length_km, link_length_miles, start_junction_road_name, end_junction_road_name
    2.2 fill NaN values with the mean of the columns for the rest of the columns

NaN total: 7169454
start_junction_road_name        1793112
end_junction_road_name          1793040
link_length_km                  1791636
link_length_miles               1791636
"""


# read traffic data
file_path = r"Traffic_common_data.csv"
traffic = pd.read_csv(file_path, low_memory=False)

# create month and day columns
traffic["count_date"] = pd.to_datetime(traffic["count_date"], errors="coerce")
traffic.insert(
    traffic.columns.get_loc("count_date") + 1, "month", traffic["count_date"].dt.month
)
traffic.insert(
    traffic.columns.get_loc("count_date") + 2, "day", traffic["count_date"].dt.day
)
# print(traffic.columns)

# check if there are NaN values in the traffic dataset
nan_count = traffic.isna().sum().sum()
# print(f"NaN total: {nan_count}")
nan_per_column = traffic.isna().sum()
nan_per_column = nan_per_column[nan_per_column > 0].sort_values(ascending=False)
# print(f"NaN count per column: \n{nan_per_column}")

# drop columns with NaN values over 50 raws
traffic = traffic.dropna(axis=1, thresh=len(traffic) - 50)
nan_per_column_cleaned = traffic.isna().sum()
nan_per_column_cleaned = nan_per_column_cleaned[nan_per_column_cleaned > 0].sort_values(
    ascending=False
)
# print(f"NaN count per column: \n{nan_per_column_cleaned}")

# after dropping columns with NaN values over 50 raws
# NaN count per column:
# all_motor_vehicles              12
# all_HGVs                         6
# buses_and_coaches                5
# HGVs_4_or_more_rigid_axle        2
# cars_and_taxis                   1
# HGVs_3_rigid_axle                1
# HGVs_2_rigid_axle                1
# HGVs_6_articulated_axle          1
# HGVs_3_or_4_articulated_axle     1

# fill NaN values with the mean of the columns
traffic_filled = traffic.fillna(traffic.mean(numeric_only=True))

# check if there are NaN values in the traffic dataset
nan_count_filled = traffic_filled.isna().sum().sum()
# print(f"NaN total after filling: {nan_count_filled}")

# save cleaned traffic data
traffic_filled.to_csv("Traffic_common_data_cleaned.csv", index=False)
print("CSV file saved successfully!")
