# %% [markdown]
# # Public transport analysis
# By Elina Yancheva

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np


# %%
df = pd.read_csv("data/transport_data.csv")
display(df.head())
print(df.columns)

# %% [markdown]
# # Data cleaning and preparation

# %%
# converting millisecond timestamps to datetime
df["dest_datetime"] = pd.to_datetime(df["dest_ts"], unit="ms")
df["hour_of_day"] = (
    df["dest_datetime"].dt.hour if "dest_datetime" in df.columns else None
)

df["trip_duration_min"] = df["boarding_t"] / 60  # assuming boarding_t is in seconds

for col in df.columns:
    missing_pct = df[col].isnull().mean() * 100
    if missing_pct > 0:
        print(f"{col}: {missing_pct:.2f}% missing")

# filter out invalid coordinates
print(f"In total, {df.shape[0]} records")
df = df[
    (df["tap_lat"].between(42, 43)) & (df["tap_lon"].between(23, 24))
]  # Bounds of Sofia region
print(f"{df.shape[0]} records in Sofia region")

df.head()

# %%
# Head of lines with missing og_line_id
missing_og_line = df[df["og_line_id"].isnull()]
print(missing_og_line["transport_type"].value_counts())
print("---")
print(
    f"All metro records don't have og_line_id: \
       {df[df['transport_type'] == 'metro']['og_line_id'].isnull().sum() == df[df['transport_type'] == 'metro'].shape[0]}"
)

# %%
# For each column with significant missing data, check transport type distribution
for col in ["media_id", "product_id", "pan", "og_line_id", "task_id", "boarding_dist"]:
    print(f"\nMissing {col} by transport type:")
    print(df[df[col].isnull()]["transport_type"].value_counts())

# %% [markdown]
# Aparently, metro doesn't have information about boarding distance, task_id and line id.

# %%
# Check for negative values in numeric columns where negative values would be problematic
negative_value_checks = {}

# Time-related variables shouldn't be negative
for col in ["boarding_t", "transfer_t", "origin_ts", "dest_ts"]:
    if col in df.columns:
        neg_count = (df[col] < 0).sum()
        neg_pct = (df[col] < 0).mean() * 100 if neg_count > 0 else 0
        negative_value_checks[col] = {"count": neg_count, "percentage": neg_pct}

# Distance-related variables shouldn't be negative
for col in ["boarding_dist", "transfer_dist"]:
    if col in df.columns:
        neg_count = (df[col] < 0).sum()
        neg_pct = (df[col] < 0).mean() * 100 if neg_count > 0 else 0
        negative_value_checks[col] = {"count": neg_count, "percentage": neg_pct}

# Coordinate variables (latitudes and longitudes) shouldn't be zero (not necessarily negative)
# For this specific region, coordinates should be in a certain range
for col in [
    "tap_lat",
    "tap_lon",
    "origin_stop_lat",
    "origin_stop_lon",
    "dest_stop_lat",
    "dest_stop_lon",
]:
    if col in df.columns:
        zero_count = (df[col] == 0).sum()
        zero_pct = (df[col] == 0).mean() * 100 if zero_count > 0 else 0
        negative_value_checks[f"{col}_zero"] = {
            "count": zero_count,
            "percentage": zero_pct,
        }

# Print the results
print("Negative Value Checks:")
for col, stats in negative_value_checks.items():
    if stats["count"] > 0:
        print(f"{col}: {stats['count']} negative values ({stats['percentage']:.2f}%)")

# For transfer time, also check for extreme negative values (potential severe data issues)
if "transfer_t" in df.columns and (df["transfer_t"] < -600).sum() > 0:
    print(
        f"\nExtreme negative transfer times (< -10 min): {(df['transfer_t'] < -600).sum()} records"
    )
    # Show a few examples of these problematic records
    print("\nSample of records with extremely negative transfer times:")
    print(
        df[df["transfer_t"] < -600][
            ["transport_type", "transfer_t", "is_transfer"]
        ].head()
    )

# Check for logical inconsistencies
# Transfers with zero transfer time
if "transfer_t" in df.columns and "is_transfer" in df.columns:
    zero_transfer_time = ((df["is_transfer"] == True) & (df["transfer_t"] == 0)).sum()
    if zero_transfer_time > 0:
        print(f"\nTransfers with zero transfer time: {zero_transfer_time} records")

# Destination timestamp earlier than origin timestamp
if "origin_ts" in df.columns and "dest_ts" in df.columns:
    time_inconsistency = (df["dest_ts"] < df["origin_ts"]).sum()
    if time_inconsistency > 0:
        print(
            f"\nTrips where destination time is earlier than origin time: {time_inconsistency} records"
        )
        time_diff_minutes = (df["origin_ts"] - df["dest_ts"]).where(
            df["dest_ts"] < df["origin_ts"]
        ) / 60000  # convert ms to minutes
        print(f"Average time inconsistency: {time_diff_minutes.mean():.2f} minutes")

# %% [markdown]
# strange why date and dest_ts are one year apart?
# # Exploratory data analysis

# %% [markdown]
# ## Transport type distribution

# %%
df["transport_type"].value_counts()

# %%
# Removed 3 records (0.005% of dataset) with "unknown" transport type to ensure accurate transport-specific analysis.
df = df[df["transport_type"] != "unknown"]

# %%
# So we'll convert all transport types to lower case to merge 'Tram' and 'tram' etc.
df["transport_type"] = df["transport_type"].str.lower()

# %%
transport_counts = df["transport_type"].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=transport_counts.index, y=transport_counts.values)
plt.title("Distribution of Transport Types")
plt.ylabel("Number of Trips")
plt.xlabel("Transport Type")
plt.xticks(rotation=45)
# Add bar values
for i, v in enumerate(transport_counts.values):
    plt.text(i, v + 5, str(v), ha="center", va="bottom")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Trip timing analysis

# %%
if "hour_of_day" in df.columns and df["hour_of_day"] is not None:
    hourly_trips = (
        df.groupby(["hour_of_day", "transport_type"]).size().unstack().fillna(0)
    )
    plt.figure(figsize=(12, 6))
    hourly_trips.plot(kind="line", marker="o")
    plt.title("Hourly Trip Distribution by Transport Type")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Trips")
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show("hourly_trips.png")

# %% [markdown]
# ### Transport Usage Analysis Summary
#
# **Transport Type Distribution:**
# - Metro dominates with ~37,000 trips (70%+), followed by bus (~5,800 trips)
#
# **Hourly Usage Patterns:**
# - Clear morning peak (8-10) and evening peak (16-19) for all transport modes
# - Metro volumes significantly exceed all other transportation types

# %% [markdown]
# # Transfer analysis

# %%
transfer_stats = df.groupby("transport_type")["is_transfer"].mean() * 100
plt.figure(figsize=(10, 6))
transfer_stats.plot(kind="bar", color="orange")
plt.title("Percentage of Transfers by Transport Type")
plt.ylabel("Transfer Percentage (%)")
plt.xlabel("Transport Type")
plt.axhline(
    y=df["is_transfer"].mean() * 100,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f'Overall Average: {df["is_transfer"].mean()*100:.1f}%',
)
plt.legend()
plt.tight_layout()
plt.show()

# Transfer time analysis
if "transfer_t" in df.columns:
    # Filter for trips with transfers and avoid invalid transfer times
    transfer_trips = df[df["is_transfer"] == True & (df["transfer_t"] > 0)]

    # Transfer time by transport type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="transport_type", y="transfer_t", data=transfer_trips)
    plt.title("Transfer Time by Transport Type")
    plt.ylabel("Transfer Time (seconds)")
    plt.xlabel("Transport Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Calculate efficiency metrics
    avg_transfer_time = df[df["is_transfer"]]["transfer_t"].mean() / 60  # minutes
    print(f"Average transfer waiting time: {avg_transfer_time:.2f} minutes")

# Transfer disance analysis
if "transfer_dist" in df.columns:
    # Filter for trips with transfers
    transfer_trips = df[df["is_transfer"] == True]

    # Transfer distance by transport type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="transport_type", y="transfer_dist", data=transfer_trips)
    plt.title("Transfer Distance by Transport Type")
    plt.ylabel("Transfer Distance (meters)")
    plt.xlabel("Transport Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Calculate efficiency metrics
    avg_transfer_distance = (
        df[df["is_transfer"]]["transfer_dist"].mean() / 1000
    )  # kilometers
    print(f"Average transfer distance: {avg_transfer_distance:.2f} kilometers")

# %% [markdown]
# # Transfer Analysis Insights
#
# ## Transfer Frequency
# - **Bus** has the highest transfer rate at ~44%, significantly above the overall average of 36.2%.
# - **Metro** has the lowest transfer rate at ~34%, suggesting it may serve more direct routes or complete journeys.
# - **Tram** and **Trolleybus** show transfer rates close to or slightly below the overall average.
#
# ## Transfer Time
# - **Transfer times** typically range from ~150-750 seconds (2.5-12.5 minutes) across all transport types.
# - **Tram** shows the highest median transfer time, suggesting potentially less frequent service.
# - **Metro** displays notable outliers with some negative transfer times, indicating potential data quality issues or system timing anomalies.
#
# ## Transfer Distance
# - **Trolleybus** transfers involve the longest distances (median ~100m).
# - **Metro** shows a weird distribution with many transfers happening at very short distances (0-1m) but also having numerous outliers stretching to 500m.
# - **Bus** and **Tram** show similar distance distributions with medians around 75-100m.
#
# These patterns suggest that while metro is the dominant transport mode, bus connections play a crucial role in the overall network connectivity. The presence of negative transfer times and the unusual metro distance distribution suggest further investigation into data quality and how transfers are recorded in the system.

# %% [markdown]
# # Geospatial analysis

# %%
# create a map of all trip origins
m = folium.Map(location=[df["tap_lat"].mean(), df["tap_lon"].mean()], zoom_start=12)

# heatmap layer
heat_data = [[row["tap_lat"], row["tap_lon"]] for _, row in df.iterrows()]
HeatMap(heat_data).add_to(m)
m.save("trip_origin_heatmap.html")

transport_colors = {
    "bus": "blue",
    "metro": "red",
    "trolleybus": "green",
    "tram": "orange",
}

m2 = folium.Map(location=[df["tap_lat"].mean(), df["tap_lon"].mean()], zoom_start=12)

# add SMALL SAMPLE of points colored by transport type
sample_size = min(1000, len(df))
for _, row in df.sample(sample_size).iterrows():
    color = transport_colors.get(row["transport_type"], "gray")
    folium.CircleMarker(
        location=[row["tap_lat"], row["tap_lon"]],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=row["transport_type"],
    ).add_to(m2)

legend_html = """
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
padding: 10px; border: 2px solid grey; border-radius: 5px;">
<p><b>Transport Type</b></p>
"""
for transport, color in transport_colors.items():
    legend_html += (
        f'<p><i class="fa fa-circle" style="color:{color}"></i> {transport}</p>'
    )
legend_html += "</div>"

m2.get_root().html.add_child(folium.Element(legend_html))
m2.save("transport_type_map.html")

display(m)
display(m2)

# %%
from folium.plugins import HeatMapWithTime

m4 = folium.Map(
    location=[df["origin_stop_lat"].mean(), df["origin_stop_lon"].mean()], zoom_start=12
)

# Create a column for hour if it doesn't exist yet
if "hour" not in df.columns and "dest_datetime" in df.columns:
    df["hour"] = df["dest_datetime"].dt.hour
elif "hour" not in df.columns and "origin_ts" in df.columns:
    df["origin_datetime"] = pd.to_datetime(df["origin_ts"], unit="ms")
    df["hour"] = df["origin_datetime"].dt.hour

heat_data_by_hour = []
hour_labels = []

# Process data for each hour
for hour in range(24):
    hour_data = df[df["hour"] == hour]

    hour_heat_data = [
        [row["origin_stop_lat"], row["origin_stop_lon"], 1]
        for _, row in hour_data.iterrows()
    ]
    heat_data_by_hour.append(hour_heat_data)

    hour_labels.append(f"{hour}:00")

HeatMapWithTime(
    heat_data_by_hour,
    index=hour_labels,
    auto_play=True,
    max_opacity=0.8,
    radius=15,
    gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
    min_opacity=0.5,
    use_local_extrema=True,
).add_to(m4)

title_html = """
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px; z-index: 1000;">
<h3>Public Transport Usage by Hour (origin locations)</h3>
</div>
"""
m4.get_root().html.add_child(folium.Element(title_html))

m4.save("hourly_heatmap.html")
display(m4)

# %% [markdown]
# # Clustering analysis

# %%
if "trip_duration" not in df.columns:
    df["trip_duration_min"] = (
        df["dest_ts"] - df["origin_ts"]
    ) / 60_000  # 1 min = 60 000 ms


# %%
# Calculate distance between origin and destination using Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points in kilometers"""
    R = 6371  # Earth radius in kilometers

    # convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c

    return distance


# %%
# calculate direct distance between origin and destination
df["direct_distance_km"] = df.apply(
    lambda row: haversine_distance(
        row["origin_stop_lat"],
        row["origin_stop_lon"],
        row["dest_stop_lat"],
        row["dest_stop_lon"],
    ),
    axis=1,
)

# calculate travel speed (km/h)
df["avg_speed_kmh"] = df["direct_distance_km"] / (df["trip_duration_min"] / 60)
# N.B the distance is direct, not the actual travel distance

# convert transport_type to numeric
transport_type_mapping = {"bus": 0, "metro": 1, "trolleybus": 2, "tram": 3}
df["transport_type_code"] = df["transport_type"].map(transport_type_mapping)

df["is_transfer"] = df["is_transfer"].astype(int)

# subset of features for clustering
features = df[
    [
        "direct_distance_km",
        "trip_duration_min",
        "avg_speed_kmh",
        "transport_type_code",
        "is_transfer",
        "hour_of_day",
    ]
]

# %%
print("\nMissing values in features:")
print(features.isnull().sum())

# %% [markdown]
# ## Standardize the features
# For example, without scaling, a 5-minute difference in trip duration (small relative to the ~24 minute range) would outweigh a change from no transfer to transfer (which is the maximum possible change in that feature).
#
# After standardization, all features are expressed in the same unit: standard deviations from the mean. This makes them directly comparable and ensures no feature dominates the distance calculations in algorithms like K-means simply because it has larger values.

# %%
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# %% [markdown]
# ## K-means Clustering
#
# ### Determining the Optimal Number of Clusters

# %%
# Elbow method
inertia = []
silhouette_avg = []
k_range = range(2, 10)

for k in k_range:
    print(f"Calculating KMeans for k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

    # Calculate Silhouette Score
    labels = kmeans.labels_
    silhouette_avg.append(silhouette_score(features_scaled, labels))

# %%
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, "bo-")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_avg, "ro-")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal k")
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# From the **silhouette score plot**, we can see that:
#
# The highest silhouette score occurs at k=4 clusters. There's a secondary peak at k=6 clusters. The scores generally decrease after k=6
#
# From the **elbow method** plot:
#
# There's no sharp "elbow" point, but there's a gradual decrease in inertia. The rate of decrease noticeably slows around k=4 to k=5
#
# Given these observations, **k=4 appears to be the optimal choice**

# %%
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(features_scaled)

cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_), columns=features.columns
)
print("\nCluster Centers:")
display(cluster_centers)

# %% [markdown]
# 1. **transport_type_code**: This column was originally encoded as 0 (bus), 1 (metro), and 2 (trolleybus), 3 (tram). After clustering, we're seeing decimal values like 0.902816, which indicates that Cluster 0 contains a mix of transport types, leaning slightly toward metro (1). Cluster 3 with 2.684802 suggests it contains predominantly trolleybus trips.
#
# 2. **is_transfer**: This was originally binary (0 or 1). The value 2.034468e-01 (0.2034) in Cluster 0 means approximately 20% of trips in this cluster involve transfers. The value 2.986500e-14 is effectively zero, meaning Cluster 1 has almost no transfers. Cluster 2 with exactly 1.0 consists entirely of transfers.
#
# 3. **hour_of_day**: The values (around 9-11) seem reasonable for morning/midday trips.
#
# These "strange" values occur because cluster centers represent the average of all points in a cluster.
#
# ### Cluster intrpretation
#
# - **Cluster 0**: Long-distance trips (7.3 km), longest duration (22.7 min), fastest speed (20.2 km/h), mix of transport types but mostly metro, 20% transfers, early morning (9.8 hour = 9:48 AM)
#
# - **Cluster 1**: Medium-distance trips (2.6 km), medium duration (10 min), medium speed (16.7 km/h), mostly metro, virtually no transfers, mid-morning (10.7 hour)
#
# - **Cluster 2**: Medium-distance trips (3.1 km), medium duration (11.1 min), medium-fast speed (17.3 km/h), mostly metro, 100% transfers, late morning (11.3 hour)
#
# - **Cluster 3**: Short-distance trips (2 km), shortest duration (9.2 min), slowest speed (14 km/h), predominantly trolley and tram, 36% transfers, mid-morning (11 hour)
