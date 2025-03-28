

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

df = pd.read_excel("dft_traffic_counts_raw_counts.xlsx")

vehicle_columns = [
    "pedal_cycles", "two_wheeled_motor_vehicles", "cars_and_taxis",
    "buses_and_coaches", "LGVs", "HGVs_2_rigid_axle", "HGVs_3_rigid_axle",
    "HGVs_4_or_more_rigid_axle", "HGVs_3_or_4_articulated_axle",
    "HGVs_5_articulated_axle", "HGVs_6_articulated_axle", "all_HGVs", "all_motor_vehicles"
]

vehicle_type = "cars_and_taxis"
region_filter = None

# Filter data (by region)
if region_filter:
    df = df[df["region_name"] == region_filter]

# Calculate the latitude and longitude traffic distribution per year
geo_data = df.groupby(["year", "latitude", "longitude"])[vehicle_columns].sum().reset_index()

#Draw a dynamic map
fig = px.scatter_mapbox(
    geo_data,
    lat="latitude",
    lon="longitude",
    color=vehicle_type if vehicle_type else "all_motor_vehicles",
    size=vehicle_type if vehicle_type else "all_motor_vehicles",
    animation_frame="year",  #按年份自动播放
    mapbox_style="open-street-map",
    title=f"Traffic Distribution Over the Years ({region_filter if region_filter else 'UK'})"
)

# Set the map center
fig.update_layout(
    mapbox=dict(
        center=dict(lat=54.0, lon=-2.0),
        zoom=5
    )
)

fig.show()

