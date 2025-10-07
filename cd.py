# merged_dbscan_streamlit.py (with comparison mode)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.tsa.arima.model import ARIMA
import folium
from folium.plugins import HeatMap
from streamlit.components.v1 import html
from rtree import index  # For CD-DBSCAN
import os
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="DBSCAN Simulator", layout="wide")

st.title("📍 Spatiotemporal DBSCAN Simulator")

# Sidebar Navigation
st.sidebar.title("Navigation")

# --- Sidebar Setup ---
menu_options = [
    "1. Upload & Preprocess", 
    "2. Run Clustering", 
    "3. Visualizations", 
    "4. Evaluation",
    "5. Forecasting",
    "6. Comparison",
    "7. Download Output"
]

# CRITICAL FIX: Use the 'key' argument to connect the widget to st.session_state.menu
st.sidebar.radio(
    "Select Step", 
    menu_options, 
    key='menu' # <--- THIS ENSURES THE WIDGET REFLECTS SESSION STATE
)

# Now, retrieve the menu variable from session state for your main logic
menu = st.session_state.menu

# Sidebar Parameters
st.sidebar.header("⚙️ Parameters")

# Default recommended values
rec_eps_spatial = 0.05        # degrees (≈1 km at equator)
rec_eps_temporal = 3600       # seconds (1 hour)
rec_min_samples = 5           # typical MinPts for DBSCAN 

eps_spatial = st.sidebar.number_input(
    "Spatial Epsilon (degrees)", min_value=0.001,
    max_value=0.1,
    value=0.05, 
    step=0.001,
    format="%.4f",
    help = "Distance threshold for spatial neighbors (in coordinate units)"
)

eps_temporal = st.sidebar.number_input(
    "Temporal Epsilon (seconds)", 
    min_value=60,
    max_value=86400*7,
    value=3600,
    step=60,
    format="%d",
    help = "Distance threshold for temporal neighbors (in seconds)"
)

min_samples = st.sidebar.number_input(
    "Minimum Samples (MinPts)", 
    min_value=1,
    max_value=100,
    value=5,
    step=1,
    format="%d",
    help = "Minimum number of points to form a dense region"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"Created by **Jake & Reymart**  \n📧 dbscan@isu.edu.ph")

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'spatial_index' not in st.session_state:
    st.session_state.spatial_index = None

if 'menu' not in st.session_state:
    st.session_state.menu = "1. Upload & Preprocess"

# --- CALLBACKS FOR ALL STEPS ---

def go_to_step_2():
    """Sets the menu state to '2. Run Clustering'."""
    st.session_state.menu = "2. Run Clustering"

def go_to_step_3():
    """Sets the menu state to '3. Visualizations'."""
    st.session_state.menu = "3. Visualizations"

def go_to_step_4():
    """Sets the menu state to '4. Evaluation'."""
    st.session_state.menu = "4. Evaluation"

def go_to_step_5():
    """Sets the menu state to '5. Forecasting'."""
    st.session_state.menu = "5. Forecasting"

def go_to_step_6():
    """Sets the menu state to '6. Comparison'."""
    st.session_state.menu = "6. Comparison"

def go_to_step_7():
    """Sets the menu state to '7. Download Output' (the final step)."""
    st.session_state.menu = "7. Download Output"

# --- END OF CALLBACKS ---


# ---------------- CLUSTERING FUNCTIONS ----------------
def run_st_dbscan(df):
    coords = df[['LAT', 'LON']].values
    times = df['seconds'].values
    visited = set()
    labels = [-1] * len(df)
    cluster_id = 0

    query_times = []  # Store query times

    def get_neighbors(i):
        import time
        start_q = time.perf_counter()
        lat, lon, t = coords[i][0], coords[i][1], times[i]
        neighbors = []
        for j in range(len(df)):
            if i == j:
                continue
            if abs(times[j] - t) <= eps_temporal:
                dist = np.sqrt((lat - coords[j][0])**2 + (lon - coords[j][1])**2)
                if dist <= eps_spatial:
                    neighbors.append(j)
        end_q = time.perf_counter()
        query_times.append((end_q - start_q) * 1000)  # ms
        return neighbors

    for i in range(len(df)):
        if i in visited:
            continue
        neighbors = get_neighbors(i)
        if len(neighbors) < min_samples:
            visited.add(i)
            continue
        labels[i] = cluster_id
        seeds = set(neighbors)
        while seeds:
            current = seeds.pop()
            if current not in visited:
                visited.add(current)
                current_neighbors = get_neighbors(current)
                if len(current_neighbors) >= min_samples:
                    seeds.update(current_neighbors)
            if labels[current] == -1:
                labels[current] = cluster_id
        cluster_id += 1

    avg_query_time = np.mean(query_times) if query_times else 0
    df['cluster'] = labels
    return df, cluster_id, avg_query_time


def run_cd_dbscan(df):
    coords = df[['LAT', 'LON']].values
    times = df['seconds'].values
    visited = set()
    labels = [-1] * len(df)
    cluster_id = 0

    spatial_index = index.Index()
    for i, row in df.iterrows():
        spatial_index.insert(i, (row['LON'], row['LAT'], row['LON'], row['LAT']))

    query_times = []  # Store query times

    def get_neighbors(i):
        import time
        start_q = time.perf_counter()
        lat, lon, t = coords[i][0], coords[i][1], times[i]
        box = (lon - eps_spatial, lat - eps_spatial, lon + eps_spatial, lat + eps_spatial)
        spatial_candidates = list(spatial_index.intersection(box))
        neighbors = [j for j in spatial_candidates if abs(times[j] - t) <= eps_temporal]
        end_q = time.perf_counter()
        query_times.append((end_q - start_q) * 1000)  # ms
        return neighbors

    for i in range(len(df)):
        if i in visited:
            continue
        neighbors = get_neighbors(i)
        if len(neighbors) < min_samples:
            visited.add(i)
            continue
        labels[i] = cluster_id
        seeds = set(neighbors)
        while seeds:
            current = seeds.pop()
            if current not in visited:
                visited.add(current)
                current_neighbors = get_neighbors(current)
                if len(current_neighbors) >= min_samples:
                    seeds.update(current_neighbors)
            if labels[current] == -1:
                labels[current] = cluster_id
        cluster_id += 1

    avg_query_time = np.mean(query_times) if query_times else 0
    df['cluster'] = labels
    return df, cluster_id, avg_query_time

# --- NEW FUNCTION FOR SAVING (Place this function above the main menu logic) ---

def save_clustered_csv_to_folder(df):
    """Saves the clustered DataFrame to a local folder with a timestamp."""
    # Define the local folder for history
    SAVE_FOLDER = 'dbscan_history' 
    
    # 1. Create the directory if it doesn't exist
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # 2. Generate a unique, timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dbscan_clustered_output_{timestamp}.csv"
    save_path = os.path.join(SAVE_FOLDER, filename)

    try:
        # 3. Save the DataFrame
        df.to_csv(save_path, index=False)
        return {"status": "success", "path": save_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ---------------- MAIN APP ----------------

# Step 1: Upload & Preprocess
if menu == "1. Upload & Preprocess":
    st.subheader("📁 Upload & Preprocess Data")
    file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if file:
        # Load data
        df = pd.read_csv(file)
        original_size = len(df)

        # Basic cleaning
        df.dropna(subset=['DATE OCC', 'LAT', 'LON'], inplace=True)
        df.drop_duplicates(inplace=True)
        cleaned_size = len(df)
        removed_rows = original_size - cleaned_size

        # Timestamp conversion
        if 'DATE OCC' in df.columns:
            df['timestamp'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
        else:
            st.error("Missing 'DATE OCC' column.")
            st.stop()

        # Seconds since first event
        df['seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

        # Success message with stats
        st.success("✅ Data preprocessed successfully!")
        st.write(f"Original dataset size: {original_size} rows")
        st.write(f"After cleaning: {cleaned_size} rows ({removed_rows} removed)")

        # Additional info
        col1, col2, col3 = st.columns(3)
        time_span_days = (df['timestamp'].max() - df['timestamp'].min()).days
        st.subheader(f"📊 Data Preview")
        col1.metric("**Total Records**", cleaned_size)
        col2.metric("**Time Span (days)**", time_span_days)

        # Approximate area coverage
        if 'LAT' in df.columns and 'LON' in df.columns:
            lat_range = df['LAT'].max() - df['LAT'].min()
            lon_range = df['LON'].max() - df['LON'].min()
            # Rough km² approximation (1° ~ 111 km)
            area_km2 = (lat_range * 111) * (lon_range * 111)
            col3.metric("**Area Coverage (km²)**", int(area_km2))

        # Show dataframe preview
        st.dataframe(df.head())

        # Save to session states
        st.session_state.df = df

        if st.button("Proceed to Step 2: Run Clustering 🚀", on_click=go_to_step_2):
            pass # No more manual state change or st.rerun() needed here.


# Step 2: Run Clustering
elif menu == "2. Run Clustering":
    st.subheader("🔄 Run Clustering")

    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please preprocess the data first.")
    else:
        df = st.session_state.df.copy()
        dataset_size = len(df)

        algorithm = st.radio("Select Algorithm", ["ST-DBSCAN (Standard)", "CD-DBSCAN (Enhanced)"])

        st.write(f"**Dataset Size**: {dataset_size} rows")
        st.write(f"**Algorithm**: {algorithm}")
        st.write(f"**Parameters**: ε-distance={eps_spatial}, ε-temporal={eps_temporal}, min_samples={min_samples}")

        if st.button("🚀 Start Clustering"):
            with st.spinner(f"Running {algorithm}..."):
                # Simulated progress bar
                progress_bar = st.progress(0)
                for percent_complete in range(0, 101, 20):
                    time.sleep(0.2)  # short delay for effect
                    progress_bar.progress(percent_complete)

                start_time = time.time()
                # Run clustering
                if algorithm == "ST-DBSCAN (Standard)":
                    df, clusters_found, avg_query_time = run_st_dbscan(df)
                else:
                    df, clusters_found, avg_query_time = run_cd_dbscan(df)
                end_time = time.time()

            execution_time = end_time - start_time
            noise_points = (df['cluster'] == -1).sum()
            clustered_points = dataset_size - noise_points
            clustered_percentage = (clustered_points / dataset_size) * 100
            noise_percentage = 100 - clustered_percentage
            average_cluster_size = clustered_points / clusters_found if clusters_found > 0 else 0
            processing_rate = dataset_size / execution_time if execution_time > 0 else 0

            # Save results to session state
            st.session_state.df = df
            st.session_state.labels = df['cluster']
            st.session_state.avg_query_time = avg_query_time

            st.success("✅ Clustering complete!")

            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Clusters Found", clusters_found)
            col2.metric("Noise Points", noise_points)
            col3.metric("Execution Time (s)", round(execution_time, 2))
            col4.metric("Query Responsiveness (ms/query)", f"{avg_query_time:.4f}")

            # Performance summary in a clean block
            st.subheader("📊 Performance Summary")
            col1, col2 = st.columns(2)
            col1.write(f"Clustered Data: {clustered_points} points ({clustered_percentage:.1f}%)")
            col1.write(f"Average Cluster Size: {average_cluster_size:.1f} points")
            col2.write(f"Noise Ratio: {noise_points} points ({noise_percentage:.1f}%)")
            col2.write(f"Processing Rate: {int(processing_rate)} points/second")

            # Optional: preview clustered data
            st.dataframe(df.head())

            if 'labels' in st.session_state:
            # The button now uses the callback function to set the state.
                if st.button("Proceed to Step 3: Visualizations 🗺️", on_click=go_to_step_3):
                    pass # No more manual state change or st.rerun() needed here.

# Step 3: Visualizations
elif menu == "3. Visualizations":
    st.subheader("🗺️ Cluster Visualizations")
    if st.session_state.df is None:
        st.warning("Please run clustering first.")
    else:
        df = st.session_state.df

        fig, ax = plt.subplots(figsize=(10, 6))

        # Remove (0,0) noise points if any
        df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

        sns.scatterplot(
            data=df.sample(min(10000, len(df))),
            x='LON', y='LAT', hue='cluster',
            palette='tab20', s=10, ax=ax, legend=False
        )

        # Auto-zoom based on data range
        ax.set_xlim(df['LON'].min() - 0.01, df['LON'].max() + 0.01)
        ax.set_ylim(df['LAT'].min() - 0.01, df['LAT'].max() + 0.01)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("📊 Cluster Visualization")

        st.pyplot(fig)

        st.subheader("🌍 Interactive Folium Map")
        map_center = [df['LAT'].mean(), df['LON'].mean()]
        m = folium.Map(location=map_center, zoom_start=12)
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['LAT'], row['LON']],
                radius=3,
                color='red' if row['cluster'] == -1 else 'blue',
                fill=True,
                fill_opacity=0.5,
                popup=f"Cluster: {row['cluster']}"
            ).add_to(m)
        html(m._repr_html_(), height=600)

        st.subheader("🔥 Crime Density Heatmap")
        m2 = folium.Map(location=map_center, zoom_start=12)
        heat_data = [[row['LAT'], row['LON']] for _, row in df.iterrows()]
        HeatMap(heat_data).add_to(m2)
        html(m2._repr_html_(), height=600)

        if st.button("Proceed to Step 4: Evaluation 📈", on_click=go_to_step_4):
            pass # State change handled by callback

# Step 5: Evaluation
elif menu == "4. Evaluation":
    st.subheader("📈 Cluster Evaluation")

    start_time = time.perf_counter()

    # Check if clustering has been run
    if st.session_state.df is None or 'cluster' not in st.session_state.df.columns:
        st.warning("Please run clustering first.")
    else:
        df = st.session_state.df
        labels = df['cluster'].values

        # --- Basic Metrics ---
        total_points = len(labels)
        noise_points = np.sum(labels == -1)
        clustered_points = total_points - noise_points
        unique_clusters = len(np.unique(labels[labels != -1]))
        noise_ratio = noise_points / total_points

        # Measure latency end (stop timer here for clustering evaluation)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000  # convert to ms

        # --- Dashboard-style horizontal metrics ---
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.markdown("**Total Points**")
        col1.metric("", f"{total_points:,}")

        col2.markdown("**Clusters Found**")
        col2.metric("", unique_clusters)

        col3.markdown("**Clustered Points**")
        col3.metric("", f"{clustered_points:,}")

        col4.markdown("**Noise Ratio**")
        col4.metric("", f"{noise_ratio:.1%}")

        # New latency metric
        col5.markdown("**Latency (ms)**")
        col5.metric("", f"{latency_ms:.2f}")

        # Average query responsiveness from session state
        avg_query_time = st.session_state.get('avg_query_time', 0)
        col6.markdown("**Query Responsiveness (ms/query)**")
        col6.metric("", f"{avg_query_time:.4f}")

        # --- Detailed Cluster Analysis ---
        st.write("### 🔍 Detailed Cluster Analysis")
        if unique_clusters > 0:
            cluster_metrics = []
            for cluster_id in np.unique(labels[labels != -1]):
                cluster_data = df[df['cluster'] == cluster_id]

                # Centroid
                centroid_lat = cluster_data['LAT'].mean()
                centroid_lon = cluster_data['LON'].mean()

                # Distances from centroid
                distances = np.sqrt((cluster_data['LAT'] - centroid_lat)**2 +
                                    (cluster_data['LON'] - centroid_lon)**2)
                avg_distance = distances.mean()
                max_distance = distances.max()

                # Temporal span (hours)
                time_span = (cluster_data['seconds'].max() - cluster_data['seconds'].min()) / 3600

                # Density (points/km²/hr)
                spatial_area = np.pi * (max_distance * 111)**2  # rough km²
                density = len(cluster_data) / (spatial_area * max(time_span, 1)) if spatial_area > 0 else 0

                cluster_metrics.append({
                    'Cluster ID': cluster_id,
                    'Size': len(cluster_data),
                    'Centroid (Lat, Lon)': f"({centroid_lat:.4f}, {centroid_lon:.4f})",
                    'Avg Radius (km)': f"{avg_distance * 111:.2f}",
                    'Max Radius (km)': f"{max_distance * 111:.2f}",
                    'Time Span (hrs)': f"{time_span:.1f}",
                    'Density (pts/km²/hr)': f"{density:.2f}"
                })

            cluster_df = pd.DataFrame(cluster_metrics)
            st.dataframe(cluster_df, use_container_width=True)
        else:
            st.info("No clusters found. All points may be noise.")

        # --- Cluster Quality Assessment ---
        st.write("### 🏆 Cluster Quality Assessment")
        quality_msgs = []

        if 0.10 <= noise_ratio <= 0.25:
            quality_msgs.append(("✅ Optimal noise ratio (10-25%)", "green"))
        elif 0.05 <= noise_ratio <= 0.35:
            quality_msgs.append(("🟡 Acceptable noise ratio", "orange"))
        else:
            if noise_ratio < 0.05:
                quality_msgs.append(("🔴 Very low noise - possible over-clustering", "red"))
            else:
                quality_msgs.append(("🔴 High noise - consider parameter adjustment", "red"))

        # Display quality messages safely
        if quality_msgs:
            cols = st.columns(len(quality_msgs))
            for i, (msg, color) in enumerate(quality_msgs):
                cols[i].markdown(f"<div style='color:{color}; font-weight:bold;'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.write("No quality messages available.")

        if st.button("Proceed to Step 5: Forecasting 📅", on_click=go_to_step_5):
            pass # No more manual state change or st.rerun() needed here.



# Step 5: Forecasting
elif menu == "5. Forecasting":
    st.subheader("📅 Forecasting with ARIMA")
    if st.session_state.df is None:
        st.warning("Please run clustering first.")
    else:
        df = st.session_state.df
        cluster_counts = df.groupby(df['timestamp'].dt.date)['cluster'].count()
        if cluster_counts.empty:
            st.warning("No data available for forecasting.")
        else:
            model = ARIMA(cluster_counts, order=(1, 1, 1))
            results = model.fit()
            forecast = results.forecast(steps=7)

            fig2, ax2 = plt.subplots()
            cluster_counts.plot(ax=ax2, label='Observed')
            forecast.plot(ax=ax2, label='Forecast', style='--')
            ax2.set_title("Crime Incident Forecast")
            ax2.legend()
            st.pyplot(fig2)

            if st.button("Proceed to Step 6: Comparison ⚖️", on_click=go_to_step_6):
                pass # State change handled by callback

# Step 6: Comparison
elif menu == "6. Comparison":
    st.subheader("⚖️ Performance Comparison: ST-DBSCAN vs CD-DBSCAN")

    if st.session_state.df is None:
        st.warning("Please upload and preprocess data first.")
    else:
        df = st.session_state.df.copy()
        total_points = len(df)

        import time

        # --- Run ST-DBSCAN ---
        start = time.perf_counter()
        st_df, st_clusters, st_avg_query_time = run_st_dbscan(df.copy())
        st_time = time.perf_counter() - start
        st_latency_ms = st_time * 1000  # convert to ms
        st_noise_points = np.sum(st_df['cluster'] == -1)
        st_clustered_points = total_points - st_noise_points
        st_noise_ratio = st_noise_points / total_points

        # --- Run CD-DBSCAN ---
        start = time.perf_counter()
        cd_df, cd_clusters, cd_avg_query_time = run_cd_dbscan(df.copy())
        cd_time = time.perf_counter() - start
        cd_latency_ms = cd_time * 1000  # convert to ms
        cd_noise_points = np.sum(cd_df['cluster'] == -1)
        cd_clustered_points = total_points - cd_noise_points
        cd_noise_ratio = cd_noise_points / total_points

        # --- Display metrics side by side ---
        st.write("### 📊 Performance Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**ST-DBSCAN**")
            st.metric("Clusters Found", st_clusters)
            st.metric("Clustered Points", f"{st_clustered_points:,}")
            st.metric("Noise Points", f"{st_noise_points:,}")
            st.metric("Noise Ratio", f"{st_noise_ratio:.1%}")
            st.metric("Execution Time (s)", f"{st_time:.2f}")
            st.metric("Latency (ms)", f"{st_latency_ms:.2f}")
            st.metric("Query Responsiveness (ms/query)", f"{st_avg_query_time:.4f}")

        with col2:
            st.write("**CD-DBSCAN**")
            st.metric("Clusters Found", cd_clusters)
            st.metric("Clustered Points", f"{cd_clustered_points:,}")
            st.metric("Noise Points", f"{cd_noise_points:,}")
            st.metric("Noise Ratio", f"{cd_noise_ratio:.1%}")
            st.metric("Execution Time (s)", f"{cd_time:.2f}")
            st.metric("Latency (ms)", f"{cd_latency_ms:.2f}")
            st.metric("Query Responsiveness (ms/query)", f"{cd_avg_query_time:.4f}")

        # --- Side-by-side cluster plots ---
        st.write("### 📊 Cluster Visualization Comparison")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.scatterplot(
            data=st_df.sample(min(10000, len(st_df))),
            x='LON', y='LAT', hue='cluster',
            palette='tab20', s=10, ax=axes[0], legend=False
        )
        axes[0].set_title("ST-DBSCAN")

        sns.scatterplot(
            data=cd_df.sample(min(10000, len(cd_df))),
            x='LON', y='LAT', hue='cluster',
            palette='tab20', s=10, ax=axes[1], legend=False
        )
        axes[1].set_title("CD-DBSCAN")

        st.pyplot(fig)

        if st.button("Proceed to Step 7: Download Output 📄", on_click=go_to_step_7):
            pass # State change handled by callback

elif menu == "7. Download Output":
    st.subheader("💾 Output & Save Clustered Data")
    
    if st.session_state.df is None:
        st.warning("Please upload and run clustering first to see the data.")
    else:
        df = st.session_state.df
        
        if 'cluster' not in df.columns:
            st.warning("Data is uploaded but clustering has not been run. Please go to '2. Run Clustering'.")
        else:
            st.write(f"Displaying the first {min(100, len(df))} rows of the clustered data.")
            st.dataframe(df.head(100))
            
            st.markdown("---")
            
            # Create the save button
            if st.button("📁 Save to Output  History Folder"):
                
                # Call the new saving function
                with st.spinner("Saving file to system folder..."):
                    save_result = save_clustered_csv_to_folder(df)

                if save_result['status'] == "success":
                    # Success message with the path
                    st.success(f"✅ **Results successfully saved!** You can find your output history here:")
                    st.code(save_result['path'])
                else:
                    # Error message
                    st.error(f"❌ **Failed to save file!** Error: {save_result['message']}")
            
            st.info("This is the final step. Your data is ready to be saved to your local history folder.")

