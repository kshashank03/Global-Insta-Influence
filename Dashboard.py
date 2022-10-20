from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pandas as pd
import plotly.express as px
import os
import numpy as np
from math import radians, cos, sin, asin, sqrt
import plotly.graph_objects as go


# from matplotlib import pyplot as plt
import streamlit as st

st.set_page_config(layout="wide", page_title="Global Centers Insta")

full_account_data_file_path = os.path.join("dashboard_data", "full_account_data.pkl")
data = pd.read_pickle(filepath_or_buffer=full_account_data_file_path)
data.dropna(inplace=True)
st.title("Global Insta Centers of Influence")

continent_list = list(data["continent_names"].unique())
continent_list.sort()
continent_list = ["World"] + continent_list

country_list = list(data["country_names"].unique())
country_list.sort()
country_list = ["World"] + country_list

metrics_list = [
    "account_count",
    "followers",
    "region",
    "avg_engagement",
    "is_verified",
    "posts_count",
]

with st.sidebar:
    continent = st.selectbox(label="Continent", options=continent_list)
    country = st.selectbox(label="Country", options=country_list)
    clusters = st.select_slider(label="Clusters", options=range(2, 11), value=6)
    metrics = st.selectbox(label="Metrics", options=metrics_list)

# Model
if continent != "World":
    data = data[data["continent_names"] == continent]
if country != "World":
    data = data[data["country_names"] == country]

weightedkmeanModel = KMeans(n_clusters=clusters).fit(
    X=data[["latitude", "longitude"]],
    sample_weight=data[metrics],
)

# Plot Data
plot_dataframe = data.copy()
plot_dataframe["colors"] = weightedkmeanModel.labels_
# fig = px.scatter_mapbox(
#     plot_dataframe, lat="latitude", lon="longitude", zoom=1, height=1000, color="colors"
# )
# fig.update_layout(mapbox_style="carto-positron", coloraxis_showscale=False)

# New Implementation
counter = 0
color_scales = ["algae", "ice", "PuBu", "OrRd", "Greys", "Brwnyl", "Tealgrn", "Agsunset", "Purp", "amp", "YlGn"]

for color in plot_dataframe["colors"].unique():
    lat = plot_dataframe[plot_dataframe["colors"] == color]["latitude"]
    long = plot_dataframe[plot_dataframe["colors"] == color]["longitude"]
    if counter == 0:
        fig = go.Figure(
                data=[go.Densitymapbox(lat=lat, lon=long, radius=3, colorscale=color_scales[counter], name="Group 1", showscale=False, hoverinfo="skip")],
                layout=go.Layout(height=1000,
                )
            )
        counter += 1
    else:
        fig.add_trace(go.Densitymapbox(lat=lat, lon=long, radius=3, colorscale=color_scales[counter], showscale=False, hoverinfo="skip"))
        counter += 1
fig.update_layout(mapbox_style="carto-positron")
# fig.show()

center_dataframes = pd.DataFrame(
    weightedkmeanModel.cluster_centers_, columns=["center_lat", "center_long"]
)

groups = ["Insta Center " + str(i + 1) for i in range(len(center_dataframes))]
center_dataframes["groups"] = groups # range(len(center_dataframes))
center_dataframes["sizes"] = 1
colors = px.colors.qualitative.Safe

fig_centers = px.scatter_mapbox(
    center_dataframes,
    lat="center_lat",
    lon="center_long",
    zoom=1,
    height=1000,
    color=colors[:len(center_dataframes)],
    text="groups",
    # labels="groups",
    size="sizes",
)
fig_centers.update_layout(mapbox_style="carto-positron", coloraxis_showscale=False, showlegend=False)

world_city_data_filepath = os.path.join("dashboard_data", "world_city_data.pkl")
world_city_data = pd.read_pickle(filepath_or_buffer=world_city_data_filepath)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


all_centers_and_cities = pd.merge(
    left=world_city_data, right=center_dataframes, how="cross"
)

append_distances = []
for i in range(len(all_centers_and_cities)):
    append_distances.append(
        haversine(
            all_centers_and_cities["center_long"][i],
            all_centers_and_cities["center_lat"][i],
            all_centers_and_cities["longitude"][i],
            all_centers_and_cities["latitude"][i],
        )
    )

all_centers_and_cities["Distances"] = append_distances

temp = all_centers_and_cities.groupby("groups")["Distances"].min().reset_index()
centers_presentation = pd.merge(
    left=all_centers_and_cities, right=temp, how="inner", on=["groups", "Distances"]
)[["Name", "Country", "Population", "groups"]]


tab1, tab2 = st.tabs(["All Points", "Capital Points"])
with tab1:
    st.plotly_chart(fig, use_container_width=True)
    pass

with tab2:
    st.plotly_chart(fig_centers, use_container_width=True)
    st.table(centers_presentation)
