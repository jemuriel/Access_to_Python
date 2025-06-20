# Streamlit app with Consist Optimisation using Interval Trees and Compatibility Matrix
import itertools

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from intervaltree import IntervalTree
import streamlit.components.v1 as components

# --- Supabase Config ---
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}

# --- Helper Functions ---
def fetch_table(table_name, chunk_size=1000):
    all_rows = []
    offset = 0
    while True:
        url = f"{SUPABASE_URL}/rest/v1/{table_name}?select=*"
        headers = {**HEADERS, "Range-Unit": "items", "Range": f"{offset}-{offset+chunk_size-1}"}
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        chunk = res.json()
        if not chunk:
            break
        all_rows.extend(chunk)
        if len(chunk) < chunk_size:
            break
        offset += chunk_size
    return pd.DataFrame(all_rows)

def upload_table(df: pd.DataFrame, table_name: str):
    df_serialized = df.copy()
    df_serialized = df_serialized.loc[:, ~df_serialized.columns.str.startswith("_")]
    for col in df_serialized.select_dtypes(include=["datetime"]):
        df_serialized[col] = df_serialized[col].apply(lambda x: x.strftime("%d/%m/%Y %H:%M") if pd.notnull(x) else "EMPTY")
    df_serialized = df_serialized.fillna("EMPTY")
    url = f"{SUPABASE_URL}/rest/v1/{table_name}"
    res = requests.post(url, headers={**HEADERS, "Prefer": "resolution=merge-duplicates"},
                        json=df_serialized.to_dict(orient="records"))
    if res.status_code >= 400:
        st.error(f"Upload failed with status {res.status_code}: {res.text}")
    res.raise_for_status()

def apply_version_selection_preference(key="compare_version"):
    if "next_version_to_select" in st.session_state:
        st.session_state[key] = st.session_state.pop("next_version_to_select")

# --- Compatibility Matrix ---
def build_compatibility_matrix(train_meta_df, feature='WAGON_TYPE'):
    matrix = pd.DataFrame(0, index=train_meta_df.index, columns=train_meta_df.index)
    for i, row_i in train_meta_df.iterrows():
        for j, row_j in train_meta_df.iterrows():
            if row_i[feature] == row_j[feature]:
                matrix.loc[i, j] = 1
    return matrix

# --- Optimiser ---
from datetime import timedelta
import pandas as pd

def consist_optimiser(long_train_df, buffer_minutes=720, max_cycle_days=14):
    df = long_train_df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=True)
    df = df.sort_values(by="Datetime")

    # Get endpoints for each train
    endpoints = df.groupby("Train", group_keys=False).apply(lambda g: pd.concat([g.head(1), g.tail(1)])).reset_index(drop=True)
    services = endpoints[endpoints["Activity"] == "DEP"]
    arrivals = endpoints[endpoints["Activity"] == "ARR"]

    arr_terminal = arrivals.set_index("Train")["Terminal"].to_dict()
    arr_time = arrivals.set_index("Train")["Datetime"].to_dict()
    dep_terminal = services.set_index("Train")["Terminal"].to_dict()
    dep_time = services.set_index("Train")["Datetime"].to_dict()

    trains = list(services["Train"].unique())
    used_trains = set()
    consists = []

    # Precompute feasible next train links
    successors = {train: [] for train in trains}
    for t1 in trains:
        for t2 in trains:
            if t1 == t2 or t2 in used_trains:
                continue
            if arr_terminal[t1] == dep_terminal[t2]:
                if arr_time[t1] + timedelta(minutes=buffer_minutes) <= dep_time[t2]:
                    successors[t1].append(t2)

    def build_consist_path(start_train):
        path = [start_train]
        visited = {start_train}
        origin = dep_terminal[start_train]
        cycle_start_time = dep_time[start_train]

        def dfs(current_train):
            for next_train in successors[current_train]:
                if next_train in visited or next_train in used_trains:
                    continue

                if arr_time[next_train] - cycle_start_time > timedelta(days=max_cycle_days):
                    continue

                path.append(next_train)
                visited.add(next_train)

                # If we return to origin, and the path is at least 2 trains long, stop
                if arr_terminal[next_train] == origin and len(path) > 1:
                    return True

                if dfs(next_train):
                    return True

                # backtrack
                path.pop()
                visited.remove(next_train)

            return False

        if dfs(start_train):
            return path
        return None

    for train in trains:
        if train in used_trains:
            continue
        path = build_consist_path(train)
        if path:
            consists.append(path)
            used_trains.update(path)

    # Assign consist IDs
    train_to_consist = {}
    assignments = []
    for cid, train_list in enumerate(consists, start=1):
        for t in train_list:
            train_to_consist[t] = cid
            assignments.append((cid, t, dep_time[t], arr_time[t]))

    short_df = pd.DataFrame(assignments, columns=["Consist_ID", "Train", "Departure", "Arrival"])
    long_train_df["Consist_ID"] = long_train_df["Train"].map(train_to_consist)
    long_train_df = long_train_df.sort_values(by=['Train', 'Datetime'])

    return short_df, long_train_df

def plotly_consist_service_distribution(short_train_df):
    service_counts = short_train_df['Consist_ID'].value_counts().sort_index()
    avg_services = service_counts.mean()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=service_counts.index.astype(str),
        y=service_counts.values,
        name='Services per Consist',
        marker_color='skyblue',
        hovertemplate = 'Consist ID: %{x}<br>Services: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=service_counts.index.astype(str),
        y=[avg_services] * len(service_counts),
        mode='lines',
        name=f'Average: {avg_services:.1f}',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title="📊 Services per Consist",
        xaxis_title="Consist ID",
        yaxis_title="Number of Services",
        height=400,
        showlegend=True
    )

    return fig

# --- Main App ---
st.set_page_config(layout="wide")
st.title("📅 Train Timetable with Consist Optimisation")

base_df = fetch_table("train_timetable_new")
version_df = fetch_table("train_timetable_versions_new")
all_versions = pd.concat([base_df, version_df], ignore_index=True)

for col in ['Datetime', 'Saved_at']:
    all_versions[col] = pd.to_datetime(all_versions.get(col), errors="coerce")

all_versions["Version_label"] = all_versions["Version_label"].fillna("Original")
version_options = all_versions["Version_label"].unique().tolist()
selected_version = 'Original'

current_df = all_versions[all_versions["Version_label"] == selected_version].copy()

corridor_mapping = {
    'MB': 'MB-BM', 'BM': 'MB-BM',
    'MP': 'MP-PM', 'PM': 'MP-PM',
    'SP': 'SP-PS', 'PS': 'SP-PS',
    'MS': 'MS-SM', 'SM': 'MS-SM',
    'SB': 'SB-BS', 'BS': 'SB-BS',
}
current_df['Corridor Group'] = current_df['Corridor'].map(corridor_mapping)

st.sidebar.markdown("### Display Settings")
tra_percent = st.sidebar.slider("Transit Terminal Granularity", 1, 31, 10, step=5)
# train_buffer = st.sidebar.slider("Train transit buffer", 30, 120, 30, step=30)

st.subheader("🚉 Filters")
available_corridors = current_df['Corridor Group'].dropna().unique()
selected_corridor = st.selectbox("Select Corridor", available_corridors)

valid_corridor_values = [k for k, v in corridor_mapping.items() if v == selected_corridor]
corridor_df = current_df[current_df['Corridor'].isin(valid_corridor_values)].copy()

available_trains = corridor_df['Train'].dropna().unique()
selected_trains = st.sidebar.multiselect("🚂 Select Train(s)", available_trains, default=list(available_trains))

filtered = corridor_df[corridor_df['Train'].isin(selected_trains)].copy()
activity_summary = filtered.groupby(['Terminal', 'Activity']).size().unstack(fill_value=0)
required_terminals = set(activity_summary.index[(activity_summary.get('ARR', 0) > 0)
                                                | (activity_summary.get('DEP', 0) > 0)])
tra_only = activity_summary.index[(activity_summary.get('ARR', 0) == 0) & (activity_summary.get('DEP', 0) == 0)]
sampled_tra = set(np.random.choice(tra_only, size=max(1, int((tra_percent / 100) * len(tra_only))),
                                   replace=False)) if len(tra_only) > 0 else set()

filtered_opti = filtered[filtered['Terminal'].isin(required_terminals.union(sampled_tra))].copy()
ordered_stations = filtered_opti['Terminal'].drop_duplicates().tolist()
station_pos_map = {station: i for i, station in enumerate(reversed(ordered_stations))}
filtered_opti['StationPosition'] = filtered_opti['Terminal'].map(station_pos_map)

min_datetime = filtered_opti['Datetime'].min()
max_datetime = filtered_opti['Datetime'].max()
start_date = min_datetime
end_date = max_datetime + timedelta(days=1)

# --- Timetable Plot ---
st.subheader("🚆 Train Timetable")
fig = go.Figure()
for train, group in filtered_opti.groupby('Train'):
    fig.add_trace(go.Scatter(
        x=group['Datetime'],
        y=group['StationPosition'],
        mode='lines+markers',
        name=train,
        line=dict(width=2),
        marker=dict(size=6),
        customdata=group[['Terminal', 'Train']],
        hovertemplate=(
                "Train: %{customdata[1]}<br>" +
                "Terminal: %{customdata[0]}<br>" +
                "Time: %{x|%d/%m/%Y %H:%M}<extra></extra>"
        )
    ))

tickvals = list(station_pos_map.values())
ticktext = [f"<b><i>{station}</i></b>" if (station in activity_summary.index and
              (activity_summary.loc[station].get('ARR', 0) > 0 or activity_summary.loc[station].get('DEP', 0) > 0))
            else station for station in station_pos_map.keys()]
fig.update_layout(
    title=f"🚆 Train Timetable — {selected_corridor}",
    xaxis=dict(title="Datetime", range=[start_date, end_date], tickformat="%a\n%d/%m\n%H:%M", showgrid=True),
    yaxis=dict(title="Station Name", tickmode="array", tickvals=tickvals, ticktext=ticktext,
                             autorange="reversed", showgrid=True),
    height=800, margin=dict(l=120, r=30, t=60, b=60), legend=dict(font=dict(size=10)))

st.plotly_chart(fig, use_container_width=True)

# --- Optimise Consists ---
st.subheader("🚄 Optimise Consist Schedule")

# Initialize buffer if first run
if 'train_buffer' not in st.session_state:
    st.session_state['train_buffer'] = 720  # default

# Temporary slider input (does not auto-update session state)
buffer_input = st.sidebar.slider("Train transit buffer (2 to 12 hours)", 120, 720,
                                 st.session_state['train_buffer'], step=120)

# Run optimiser only when button is clicked
if st.button("🚄 Optimise Consists"):
    st.session_state['train_buffer'] = buffer_input  # Save chosen buffer
    short_df, long_df = consist_optimiser(long_train_df=current_df, buffer_minutes=st.session_state['train_buffer'])
    st.session_state['short_df'] = short_df
    st.session_state['long_df'] = long_df
    st.success(f"Minimum consists required: {long_df['Consist_ID'].nunique()}")

# Render optimisation results if available
if 'long_df' in st.session_state and 'short_df' in st.session_state:
    long_df = st.session_state['long_df']
    short_df = st.session_state['short_df']

    available_consists = short_df['Consist_ID'].dropna().unique()
    selected_consist = st.sidebar.multiselect("🚄 Select Consist(s)", available_consists,
                                              default=list(available_consists))
    st.dataframe(short_df[short_df['Consist_ID'].isin(selected_consist)])

    filtered_opti = (long_df[(long_df['Train'].isin(selected_trains)) &
                            (long_df['Consist_ID'].isin(selected_consist))]
                     .copy())
    activity_summary = filtered_opti.groupby(['Terminal', 'Activity']).size().unstack(fill_value=0)
    required_terminals = set(activity_summary.index[(activity_summary.get('ARR', 0) > 0)
                                                    | (activity_summary.get('DEP', 0) > 0)])
    tra_only = activity_summary.index[(activity_summary.get('ARR', 0) == 0) & (activity_summary.get('DEP', 0) == 0)]
    sampled_tra = set(np.random.choice(tra_only, size=max(1, int((tra_percent / 100) * len(tra_only))),
                                       replace=False)) if len(tra_only) > 0 else set()

    filtered_opti = filtered_opti[filtered_opti['Terminal'].isin(required_terminals.union(sampled_tra))].copy()
    ordered_stations = filtered_opti['Terminal'].drop_duplicates().tolist()
    station_pos_map = {station: i for i, station in enumerate(reversed(ordered_stations))}
    filtered_opti['StationPosition'] = filtered_opti['Terminal'].map(station_pos_map)

    min_datetime = filtered_opti['Datetime'].min()
    max_datetime = filtered_opti['Datetime'].max()
    start_date = min_datetime
    end_date = max_datetime + timedelta(days=1)

    fig3 = go.Figure()
    for cid, group in filtered_opti.groupby('Consist_ID'):
        fig3.add_trace(go.Scatter(
            x=group['Datetime'],
            y=group['StationPosition'],
            mode='lines+markers',
            name=f"Consist {cid}",
            line=dict(width=3),
            marker=dict(size=8),
            customdata=group[['Terminal', 'Train']],
            hovertemplate=(
                    "Consist ID: " + str(cid) + "<br>" +
                    "Train: %{customdata[1]}<br>" +
                    "Terminal: %{customdata[0]}<br>" +
                    "Time: %{x|%d/%m/%Y %H:%M}<extra></extra>"
            )
        ))

    tickvals = list(station_pos_map.values())
    ticktext = []
    for station in station_pos_map.keys():
        row = activity_summary.loc[station] if station in activity_summary.index else {}
        if ('ARR' in row and row['ARR'] > 0) or ('DEP' in row and row['DEP'] > 0):
            ticktext.append(f"<b><i>{station}</i></b>")
        else:
            ticktext.append(station)

    fig3.update_layout(
        title=f"Train Timetable — {selected_corridor}",
        xaxis=dict(title="Datetime", range=[start_date, end_date], tickformat="%a%d/%m%H:%M", showgrid=True),
        yaxis=dict(title="Station Name", tickmode="array", tickvals=tickvals, ticktext=ticktext,
                   autorange="reversed", showgrid=True),
        height=800, margin=dict(l=120, r=30, t=60, b=60), legend=dict(font=dict(size=10))
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📊 Services per Consist")
    fig_bar = plotly_consist_service_distribution(short_df)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Editable Plan ---
st.subheader("✏️ Edit and Save Train Plan")
edited_df = st.data_editor(filtered_opti.drop(columns=['StationPosition']), num_rows="dynamic", use_container_width=True)
if st.button("📅 Save Modified Version"):
    Version_label = f"Modified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    full_df = current_df.copy()
    keys = ["Train", "Terminal", "Datetime", "Activity", "Corridor"]
    for col in full_df.columns:
        if col not in edited_df.columns:
            edited_df[col] = full_df[col]
    full_df = full_df[~full_df[keys].apply(tuple, axis=1).isin(edited_df[keys].apply(tuple, axis=1))]
    updated_df = pd.concat([full_df, edited_df], ignore_index=True)
    updated_df["Version_label"] = Version_label
    updated_df["Saved_at"] = datetime.now().isoformat()
    upload_table(updated_df, "train_timetable_versions_new")
    st.session_state.next_version_to_select = Version_label
    st.success(f"Saved new version: {Version_label}")
    st.rerun()

# --- Compare Version ---
st.subheader("📊 Compare with Another Version")
apply_version_selection_preference("compare_version")
compare_version = st.selectbox("Select Version to Compare", [v for v in version_options if v != selected_version], key="compare_version")
df_compare = all_versions[all_versions["Version_label"] == compare_version].copy()
df_compare = df_compare[df_compare['Corridor'].isin(valid_corridor_values)]
df_compare = df_compare[df_compare['Train'].isin(selected_trains)]
df_compare = df_compare[df_compare['Terminal'].isin(required_terminals.union(sampled_tra))]
df_compare['StationPosition'] = df_compare['Terminal'].map(station_pos_map)

fig2 = go.Figure()
for train, group in df_compare.groupby('Train'):
    group = group.sort_values("Datetime")
    fig2.add_trace(go.Scatter(
        x=group['Datetime'],
        y=group['StationPosition'],
        mode='lines+markers',
        name=train,
        line=dict(width=2),
        marker=dict(size=6),
        customdata=group[['Terminal', 'Train']],
        hovertemplate=(
                "Train: %{customdata[1]}<br>" +
                "Terminal: %{customdata[0]}<br>" +
                "Time: %{x|%d/%m/%Y %H:%M}<extra></extra>"
        )
    ))

fig2.update_layout(
    title=f"📊 Compared Version — {compare_version}",
    xaxis=dict(title="Datetime", range=[start_date, end_date], tickformat="%a\n%d/%m\n%H:%M", showgrid=True),
    yaxis=dict(title="Station Name", tickmode="array", tickvals=tickvals, ticktext=ticktext,
                              autorange="reversed", showgrid=True),
    height=800, margin=dict(l=120, r=30, t=60, b=60), legend=dict(font=dict(size=10)))

st.plotly_chart(fig2, use_container_width=True)
