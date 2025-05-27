import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import random

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
    res = requests.post(url, headers={**HEADERS, "Prefer": "resolution=merge-duplicates"}, json=df_serialized.to_dict(orient="records"))
    if res.status_code >= 400:
        st.error(f"Upload failed with status {res.status_code}: {res.text}")
    res.raise_for_status()

def delete_versions(table_name):
    url = f"{SUPABASE_URL}/rest/v1/{table_name}?Version_label=neq.null"
    res = requests.delete(url, headers={**HEADERS, "Prefer": "return=representation"})
    res.raise_for_status()
    return res.status_code in [200, 204]

def apply_version_selection_preference(key="compare_version"):
    if "next_version_to_select" in st.session_state:
        st.session_state[key] = st.session_state.pop("next_version_to_select")

# --- App Setup ---
st.set_page_config(layout="wide")
st.title("📅 Train Timetable from Supabase")

# --- Load Data ---
base_df = fetch_table("train_timetable_new")
version_df = fetch_table("train_timetable_versions_new")
all_versions = pd.concat([base_df, version_df], ignore_index=True)

for col in ['Datetime', 'Saved_at']:
    all_versions[col] = pd.to_datetime(all_versions.get(col), errors="coerce")

all_versions["Version_label"] = all_versions["Version_label"].fillna("Original")
version_options = all_versions["Version_label"].unique().tolist()
selected_version = 'Original'

current_df = all_versions[all_versions["Version_label"] == selected_version].copy()

# --- Corridor Mapping ---
corridor_mapping = {
    'MB': 'MB-BM', 'BM': 'MB-BM',
    'MP': 'MP-PM', 'PM': 'MP-PM',
    'SP': 'SP-PS', 'PS': 'SP-PS',
    'MS': 'MS-SM', 'SM': 'MS-SM',
    'SB': 'SB-BS', 'BS': 'SB-BS',
}
current_df['Corridor Group'] = current_df['Corridor'].map(corridor_mapping)

# --- TRA Terminal Slider ---
st.sidebar.markdown("### Display Settings")
tra_percent = st.sidebar.slider("Transit Terminal Granularity", 1, 31, 10, step=5)

# --- Corridor and Train Filtering ---
st.subheader("🚉 Filters")
available_corridors = current_df['Corridor Group'].dropna().unique()
selected_corridor = st.selectbox("Select Corridor", available_corridors)

valid_corridor_values = [k for k, v in corridor_mapping.items() if v == selected_corridor]
corridor_df = current_df[current_df['Corridor'].isin(valid_corridor_values)].copy()
available_trains = corridor_df['Train'].dropna().unique()
selected_trains = st.sidebar.multiselect("🚂 Select Train(s)", available_trains, default=list(available_trains))

filtered_df = corridor_df[corridor_df['Train'].isin(selected_trains)].copy()

# --- TRA Terminal Filtering ---
activity_summary = filtered_df.groupby(['Terminal', 'Activity']).size().unstack(fill_value=0)
required_terminals = set(activity_summary.index[(activity_summary.get('ARR', 0) > 0) | (activity_summary.get('DEP', 0) > 0)])
tra_only = activity_summary.index[(activity_summary.get('ARR', 0) == 0) & (activity_summary.get('DEP', 0) == 0)]
sampled_tra = set(np.random.choice(tra_only, size=max(1, int((tra_percent / 100) * len(tra_only))), replace=False)) if len(tra_only) > 0 else set()

filtered_df = filtered_df[filtered_df['Terminal'].isin(required_terminals.union(sampled_tra))].copy()

# Assign station position
ordered_stations = filtered_df['Terminal'].drop_duplicates().tolist()
station_pos_map = {station: i for i, station in enumerate(reversed(ordered_stations))}
filtered_df['StationPosition'] = filtered_df['Terminal'].map(station_pos_map)

# Calculate week start (Monday) and end (next Monday)
min_datetime = filtered_df['Datetime'].min()
max_datetime = filtered_df['Datetime'].max()
# start_monday = min_datetime - timedelta(days=min_datetime.weekday())
# start_monday = start_monday.replace(hour=0, minute=0, second=0, microsecond=0)
# end_monday = start_monday + timedelta(days=7)

start_date = min_datetime
end_date = max_datetime + timedelta(days=1)

# --- Plotting Primary ---
st.subheader("🚆 Train Timetable")
fig = go.Figure()
for train, group in filtered_df.groupby('Train'):
    fig.add_trace(go.Scatter(
        x=group['Datetime'],
        y=group['StationPosition'],
        mode='lines+markers',
        name=train,
        line=dict(width=2),
        marker=dict(size=6)
    ))

tickvals = list(station_pos_map.values())
ticktext = []
for station in station_pos_map.keys():
    row = activity_summary.loc[station] if station in activity_summary.index else {}
    if ('ARR' in row and row['ARR'] > 0) or ('DEP' in row and row['DEP'] > 0):
        # ticktext.append(f"<span style='color:red'><b><i>{station}</i></b></span>")
        ticktext.append(f"<b><i>{station}</i></b>")
    else:
        ticktext.append(station)

fig.update_layout(
    title=f"Train Timetable — {selected_corridor}",
    xaxis=dict(
        title="Datetime",
        range=[start_date, end_date],
        tickformat="%a\n%d/%m\n%H:%M",
        showgrid=True
    ),
    yaxis=dict(
        title="Station Name",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        autorange="reversed",
        showgrid=True
    ),
    height=1400,
    margin=dict(l=120, r=30, t=60, b=60),
    legend=dict(font=dict(size=10))
)

st.plotly_chart(fig, use_container_width=True)

# --- Editable Train Plan ---
st.subheader("✏️ Edit and Save Train Plan")
edited_df = st.data_editor(filtered_df.drop(columns=['StationPosition']), num_rows="dynamic", use_container_width=True)

if st.button("💾 Save Modified Version"):
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

# --- Comparison Version ---
st.subheader("📊 Compare with Another Version")
apply_version_selection_preference("compare_version")

compare_version = st.selectbox("Select Version to Compare", [v for v in version_options if v != selected_version], key="compare_version")

df_compare = all_versions[all_versions["Version_label"] == compare_version].copy()
df_compare = df_compare[df_compare['Corridor'].isin(valid_corridor_values)]
df_compare = df_compare[df_compare['Train'].isin(selected_trains)].copy()
df_compare = df_compare[df_compare['Terminal'].isin(required_terminals.union(sampled_tra))].copy()
df_compare['StationPosition'] = df_compare['Terminal'].map(station_pos_map)

st.dataframe(df_compare)

fig2 = go.Figure()
for train, group in df_compare.groupby('Train'):
    group = group.sort_values("Datetime")
    fig2.add_trace(go.Scatter(
        x=group['Datetime'],
        y=group['StationPosition'],
        mode='lines+markers',
        name=train,
        line=dict(width=2),
        marker=dict(size=6)
    ))

fig2.update_layout(
    title=f"Compared Version — {compare_version}",
    xaxis=dict(
        title="Datetime",
        range=[start_date, end_date],
        tickformat="%a\n%d/%m\n%H:%M",
        showgrid=True
    ),
    yaxis=dict(
        title="Station Name",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        autorange="reversed",
        showgrid=True
    ),
    height=1400,
    margin=dict(l=120, r=30, t=60, b=60),
    legend=dict(font=dict(size=10))
)

st.plotly_chart(fig2, use_container_width=True)


