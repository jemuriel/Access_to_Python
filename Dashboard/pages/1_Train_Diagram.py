import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random
import requests

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

    # Drop internal Streamlit or metadata columns
    df_serialized = df_serialized.loc[:, ~df_serialized.columns.str.startswith("_")]

    # Convert datetime to desired format or "EMPTY"
    for col in df_serialized.select_dtypes(include=["datetime", "datetimetz", "datetime64[ns]"]):
        df_serialized[col] = df_serialized[col].apply(
            lambda x: x.strftime("%d/%m/%Y %H:%M") if pd.notnull(x) else "EMPTY"
        )

    # Replace any other NaNs with "EMPTY" (optional fallback)
    df_serialized = df_serialized.fillna("EMPTY")

    url = f"{SUPABASE_URL}/rest/v1/{table_name}"
    res = requests.post(
        url,
        headers={**HEADERS, "Prefer": "resolution=merge-duplicates"},
        json=df_serialized.to_dict(orient="records")
    )
    if res.status_code >= 400:
        st.error(f"Upload failed with status {res.status_code}: {res.text}")
    res.raise_for_status()


def delete_versions(table_name):
    url = f"{SUPABASE_URL}/rest/v1/{table_name}?version_label=neq.null"
    res = requests.delete(url, headers={**HEADERS, "Prefer": "return=representation"})
    res.raise_for_status()
    return res.status_code in [200, 204]

def apply_version_selection_preference(key="compare_version"):
    if "next_version_to_select" in st.session_state:
        st.session_state[key] = st.session_state.pop("next_version_to_select")

def plot_train_diagram(df, title="Train Time-Distance Diagram"):
    df['ARR_TIME'] = pd.to_datetime(df['ARR_TIME'], errors='coerce')
    df['DEP_TIME'] = pd.to_datetime(df['DEP_TIME'], errors='coerce')
    df['TERMINAL'] = pd.Categorical(df['TERMINAL'], categories=sorted(df['TERMINAL'].dropna().unique()), ordered=True)
    fig = go.Figure()
    color_map = {train: color for train, color in zip(df['TRAIN_ID'].dropna().unique(), px.colors.qualitative.Alphabet)}

    for train in df['TRAIN_ID'].dropna().unique():
        train_data = df[df['TRAIN_ID'] == train].sort_values(by='LEG_NUM')
        color = color_map.get(train, random.choice(px.colors.qualitative.Alphabet))
        x_vals, y_vals = [], []
        for _, row in train_data.iterrows():
            if pd.notnull(row['ARR_TIME']) and pd.notnull(row['DEP_TIME']):
                x_vals.extend([row['ARR_TIME'], row['DEP_TIME'], None])
                y_vals.extend([row['TERMINAL'], row['TERMINAL'], None])
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name=train, line=dict(color=color)))

    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Terminal", height=600, width=1000)
    return fig

# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("Train Timetable")

# --- Load Data from Supabase ---
base_df = fetch_table("train_timetable_base")
version_df = fetch_table("train_timetable_versions")
all_versions = pd.concat([base_df, version_df], ignore_index=True)

for col in ['ARR_TIME', 'DEP_TIME', 'saved_at']:
    all_versions[col] = pd.to_datetime(all_versions.get(col), errors="coerce")

all_versions["version_label"] = all_versions["version_label"].fillna("Original")
version_options = all_versions["version_label"].unique().tolist()

# Turn off selector
# selected_version = st.sidebar.selectbox("Select Train Timetable Version", version_options)
selected_version = 'Original'

current_df = all_versions[all_versions["version_label"] == selected_version].copy()

# --- Corridor & Train Filters (Preserved) ---
st.subheader("🚉 Filters")
available_corridors = current_df['CORRIDOR'].dropna().unique()
selected_corridor = st.selectbox("Select Corridor", available_corridors)

filtered_by_corridor = current_df[current_df['CORRIDOR'] == selected_corridor].copy()

available_trains = filtered_by_corridor['TRAIN_ID'].dropna().unique()
preselected_trains = ['2BM4', '3BM4', '4MB4', '5BM4']
selected_trains = st.multiselect("Select Trains", available_trains, default=preselected_trains)

filtered_df = filtered_by_corridor[filtered_by_corridor['TRAIN_ID'].isin(selected_trains)].copy()

# --- Editable Table ---
st.subheader("✏️ Edit Train Plan")
edited_df = st.data_editor(filtered_df, num_rows="dynamic", use_container_width=True)
st.session_state.train_df = edited_df

# --- Save Modified Version (Properly Merge Edits) ---
if st.button("💾 Save Modified Version"):
    version_label = f"Modified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Start with full current version
    full_df = current_df.copy()

    # Identify keys to match edited rows back into full_df
    keys = ["TRAIN_ID", "TERMINAL", "ARR_TIME", "DEP_TIME", "LEG_NUM", "CORRIDOR"]

    # Step 2: Ensure edited_df has all columns (especially ARR_TIME, DEP_TIME)
    for col in full_df.columns:
        if col not in edited_df.columns:
            edited_df[col] = full_df[col]

    # Drop conflicting rows and insert the edited ones
    full_df = full_df[~full_df[keys].apply(tuple, axis=1).isin(edited_df[keys].apply(tuple, axis=1))]
    updated_df = pd.concat([full_df, edited_df], ignore_index=True)

    # Add version metadata
    updated_df["version_label"] = version_label
    updated_df["saved_at"] = datetime.now().isoformat()

    try:
        upload_table(updated_df, "train_timetable_versions")
        st.session_state.next_version_to_select = version_label
        st.success(f"Saved new version: {version_label}")
        st.rerun()
    except Exception as e:
        st.error(f"Upload failed: {e}")

# Optional: fallback if no save was triggered yet
csv_export_df = None
if "updated_df" in locals():
    csv_export_df = updated_df
else:
    # fallback: just export what's visible (in case no save was triggered)
    full_df = current_df.copy()
    keys = ["TRAIN_ID", "LEG_NUM"]
    full_df = full_df[~full_df[keys].apply(tuple, axis=1).isin(edited_df[keys].apply(tuple, axis=1))]
    csv_export_df = pd.concat([full_df, edited_df], ignore_index=True)

# Create download button
csv_bytes = csv_export_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Modified Train Plan (CSV)",
    data=csv_bytes,
    file_name="modified_train_plan.csv",
    mime="text/csv"
)

# --- Delete All Modified Versions ---
if st.button("🗑️ Delete All Modified Versions"):
    try:
        delete_versions("train_timetable_versions")
        st.success("Deleted all modified versions.")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Plot Current Version ---
# --- 🚂 Train Time-Distance Diagram (Original Logic Restored) ---
st.subheader("🚂 Train Time-Distance Diagram")

df_filtered = edited_df.copy()

# Station Logic (same as your original logic)
if selected_corridor in ["MEL-BRI", "MEL-SYD"]:
    stations_sorted = ['MFT', 'ETT', 'SFT', 'MOR', 'BFT']
    emphasis_labels = {'MFT': 20, 'SFT': 20, 'BFT': 20}
elif selected_corridor in ["MEL-PER", "MEL-ADL"]:
    stations_sorted = ['MFT', 'SPG', 'TLB', 'APD', 'AFT', 'PAD', 'SCJ', 'COK', 'WEK', 'KWK', 'PFT']
    emphasis_labels = {'MFT': 20, 'AFT': 20, 'PFT': 20}
else:
    stations_sorted = sorted(df_filtered['TERMINAL'].dropna().unique())
    emphasis_labels = {}

df_filtered['TERMINAL'] = pd.Categorical(df_filtered['TERMINAL'], categories=stations_sorted, ordered=True)
color_map = {train: color for train, color in zip(df_filtered['TRAIN_ID'].dropna().unique(),
                                                  px.colors.qualitative.Alphabet)}

fig = go.Figure()

for train in df_filtered['TRAIN_ID'].unique():
    train_data = df_filtered[df_filtered['TRAIN_ID'] == train].sort_values(by='LEG_NUM')
    color = color_map.get(train)
    if color is None:
        color = random.choice(list(color_map.values()))

    x_vals, y_vals = [], []
    for _, row in train_data.iterrows():
        if pd.notnull(row['ARR_TIME']) and pd.notnull(row['DEP_TIME']):
            x_vals.extend([row['ARR_TIME'], row['DEP_TIME'], None])
            y_vals.extend([row['TERMINAL'], row['TERMINAL'], None])
        elif pd.notnull(row['DEP_TIME']):
            x_vals.extend([row['DEP_TIME'], row['DEP_TIME'], None])
            y_vals.extend([row['TERMINAL'], row['TERMINAL'], None])

    last_row = train_data.iloc[-1]
    last_time = last_row['ARR_TIME'] if pd.notnull(last_row['ARR_TIME']) else last_row['DEP_TIME']
    if pd.notnull(last_time):
        fig.add_trace(go.Scatter(
            x=[last_time],
            y=[last_row['TERMINAL']],
            mode='markers',
            marker=dict(color=color, size=10, symbol='circle'),
            name=f"{train} end",
            legendgroup=train,
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines+markers',
        line=dict(color=color, width=1, dash='dashdot'),
        marker=dict(color=color),
        name=train,
        legendgroup=train,
        showlegend=True
    ))

    for i in range(len(train_data) - 1):
        prev_time = train_data.iloc[i]['DEP_TIME'] if pd.notnull(train_data.iloc[i]['DEP_TIME']) else train_data.iloc[i]['ARR_TIME']
        next_time = train_data.iloc[i + 1]['ARR_TIME'] if pd.notnull(train_data.iloc[i + 1]['ARR_TIME']) else train_data.iloc[i + 1]['DEP_TIME']
        y1, y2 = train_data.iloc[i]['TERMINAL'], train_data.iloc[i + 1]['TERMINAL']

        if pd.notnull(prev_time) and pd.notnull(next_time):
            fig.add_trace(go.Scatter(
                x=[prev_time, next_time],
                y=[y1, y2],
                mode='lines',
                line=dict(color=color, width=1, dash='dashdot'),
                name=f"{train} travel",
                legendgroup=train,
                showlegend=False
            ))

# Optional: day separators (disabled for now)
all_times = pd.concat([df_filtered['ARR_TIME'], df_filtered['DEP_TIME']]).dropna()
if not all_times.empty:
    min_time = all_times.min().normalize()
    max_time = all_times.max()
    current_day = min_time
    # while current_day <= max_time:
    #     fig.add_vline(
    #         x=current_day,
    #         line=dict(color='gray', dash='dash'),
    #         opacity=0.4
    #     )
    #     current_day += timedelta(days=1)

fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Terminal',
    yaxis=dict(
        autorange='reversed',
        categoryorder='array',
        categoryarray=stations_sorted,
        tickmode='array',
        tickvals=stations_sorted,
        ticktext=[
            f'<span style="font-size:{emphasis_labels.get(label,14)}px">{label}</span>' for label in stations_sorted
        ]
    ),
    height=800,
    width=1200,
    xaxis=dict(tickformat="%a %H:%M")
)

st.plotly_chart(fig, use_container_width=False)

# --- Compare Two Versions Side-by-Side ---
# --- 📊 Compare a Version Against Current ---
st.subheader("📊 Compare Another Version to Current")

apply_version_selection_preference("compare_version")

# compare_version = st.selectbox("Select a Version to Compare", [v for v in version_options if v != selected_version], key="compare_version")
compare_version = st.selectbox(
    "Select a Version to Compare",
    version_options,
    index=version_options.index(st.session_state.get("compare_version", "Original")),
    key="compare_version"
)

df_compare = all_versions[all_versions["version_label"] == compare_version].copy()

# Ensure ARR_TIME and DEP_TIME are parsed as datetime
df_compare["ARR_TIME"] = pd.to_datetime(df_compare["ARR_TIME"], errors="coerce")
df_compare["DEP_TIME"] = pd.to_datetime(df_compare["DEP_TIME"], errors="coerce")

# Use same corridor for consistency
df_compare = df_compare[df_compare['CORRIDOR'] == selected_corridor]
df_compare = df_compare[df_compare['TRAIN_ID'].isin(selected_trains)].copy()

# Use same station formatting
df_compare['TERMINAL'] = pd.Categorical(df_compare['TERMINAL'], categories=stations_sorted, ordered=True)
color_map = {train: color for train, color in zip(df_compare['TRAIN_ID'].dropna().unique(),
                                                  px.colors.qualitative.Alphabet)}


st.dataframe(df_compare)
fig_compare = go.Figure()

for train in df_compare['TRAIN_ID'].unique():
    train_data = df_compare[df_compare['TRAIN_ID'] == train].sort_values(by='LEG_NUM')
    color = color_map.get(train)
    if color is None:
        color = random.choice(list(color_map.values()))

    x_vals, y_vals = [], []
    for _, row in train_data.iterrows():
        if pd.notnull(row['ARR_TIME']) and pd.notnull(row['DEP_TIME']):
            x_vals.extend([row['ARR_TIME'], row['DEP_TIME'], None])
            y_vals.extend([row['TERMINAL'], row['TERMINAL'], None])
        elif pd.notnull(row['DEP_TIME']):
            x_vals.extend([row['DEP_TIME'], row['DEP_TIME'], None])
            y_vals.extend([row['TERMINAL'], row['TERMINAL'], None])

    last_row = train_data.iloc[-1]
    last_time = last_row['ARR_TIME'] if pd.notnull(last_row['ARR_TIME']) else last_row['DEP_TIME']
    if pd.notnull(last_time):
        fig_compare.add_trace(go.Scatter(
            x=[last_time],
            y=[last_row['TERMINAL']],
            mode='markers',
            marker=dict(color=color, size=10, symbol='circle'),
            name=f"{train} end",
            legendgroup=train,
            showlegend=False
        ))

    fig_compare.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines+markers',
        line=dict(color=color, width=1, dash='dashdot'),
        marker=dict(color=color),
        name=train,
        legendgroup=train,
        showlegend=True
    ))

    for i in range(len(train_data) - 1):
        prev_time = train_data.iloc[i]['DEP_TIME'] if pd.notnull(train_data.iloc[i]['DEP_TIME']) else train_data.iloc[i]['ARR_TIME']
        next_time = train_data.iloc[i + 1]['ARR_TIME'] if pd.notnull(train_data.iloc[i + 1]['ARR_TIME']) else train_data.iloc[i + 1]['DEP_TIME']
        y1, y2 = train_data.iloc[i]['TERMINAL'], train_data.iloc[i + 1]['TERMINAL']

        if pd.notnull(prev_time) and pd.notnull(next_time):
            fig_compare.add_trace(go.Scatter(
                x=[prev_time, next_time],
                y=[y1, y2],
                mode='lines',
                line=dict(color=color, width=1, dash='dashdot'),
                name=f"{train} travel",
                legendgroup=train,
                showlegend=False
            ))
# Optional: day separators (disabled for now)
all_times = pd.concat([df_compare['ARR_TIME'], df_compare['DEP_TIME']]).dropna()
if not all_times.empty:
    min_time = all_times.min().normalize()
    max_time = all_times.max()
    current_day = min_time

fig_compare.update_layout(
    title=f"Compared Version: {compare_version}",
    xaxis_title='Time',
    yaxis_title='Terminal',
    yaxis=dict(
        autorange='reversed',
        categoryorder='array',
        categoryarray=stations_sorted,
        tickmode='array',
        tickvals=stations_sorted,
        ticktext=[
            f'<span style="font-size:{emphasis_labels.get(label,14)}px">{label}</span>' for label in stations_sorted
        ]
    ),
    height=800,
    width=1200,
    xaxis=dict(tickformat="%a %H:%M")
)

st.plotly_chart(fig_compare, use_container_width=False)

