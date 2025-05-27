import io
import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from Wagon_Planning.New_Time_Table_Run import NewTimeTable

# -------------------------------------
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
        res = requests.get(url, headers={**HEADERS, "Range-Unit": "items", "Range": f"{offset}-{offset+chunk_size-1}"})
        res.raise_for_status()
        chunk = res.json()
        if not chunk:
            break
        all_rows.extend(chunk)
        if len(chunk) < chunk_size:
            break
        offset += chunk_size
    return pd.DataFrame(all_rows)

# -------------------------------------

# Adjust path for local module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Page configuration
st.set_page_config(page_title="Inventory Balancer", layout="wide")
st.title("📦 Wagon Balance")

# File inputs
timetable_file = "csv_files/6_train_services.csv"
wagon_mapping = "csv_files/8_wagon_mapping.csv"

# --- Load Wagon Plans from Supabase ---
wagon_base = fetch_table("base_wagon_plan")
wagon_versions = fetch_table("wagon_plan_versions")
all_wagon_plans = pd.concat([wagon_base, wagon_versions], ignore_index=True)
all_wagon_plans["version_time"] = pd.to_datetime(all_wagon_plans.
                                    get("saved_at", pd.Timestamp.now()), errors="coerce")

# Allow reload from session
all_wagon_plans = st.session_state.get("all_wagon_plans", all_wagon_plans)
version_labels = all_wagon_plans["version_label"].fillna("Original").unique().tolist()

# Preselect version if one was just saved
if "next_version_to_select" in st.session_state:
    st.session_state.selected_version_label = st.session_state.pop("next_version_to_select")

# Track current selection
if "selected_version_label" not in st.session_state:
    st.session_state.selected_version_label = "Original"

selected_version_label = st.sidebar.selectbox(
    "Select Wagon Plan Version",
    version_labels,
    index=version_labels.index(st.session_state.selected_version_label),
    key="selected_version_label"
)

# Get current plan
wagon_plan_df = all_wagon_plans[
    all_wagon_plans["version_label"].fillna("Original") == st.session_state.selected_version_label
]

st.subheader("📋 Wagon Plan Preview")
st.dataframe(wagon_plan_df, use_container_width=True)

# Sidebar settings
time_buckets = st.sidebar.selectbox("Time Granularity (minutes)", [15, 30, 60], index=0)

# Run Simulation
if st.button("Run Inventory Balance"):
    st.info("Initializing inventory balance...")

    # Instantiate model
    timetable = NewTimeTable(
        time_buckets, timetable_file, wagon_plan_df, wagon_mapping
    )
    st.success("✅ Inventory balance completed.")
    st.session_state["inventory_df"], st.session_state["adjustment_df"] = timetable.balance_inventory()

# Show output
if "inventory_df" in st.session_state:
    st.subheader("📊 Inventory Levels Over Time")
    inv_df = st.session_state["inventory_df"]
    adjustments_df = st.session_state['adjustment_df']

    terminals = inv_df['Terminal'].unique().tolist()
    default_terminal = "MFT"
    default_index = terminals.index(default_terminal) if default_terminal in terminals else 0

    selected_terminal = st.selectbox("🏗️ Select Terminal", terminals, index=default_index)

    selected_wagon_types = st.multiselect("🚛 Select Wagon Types",
                                          inv_df['Wagon_Type'].unique(),
                                          default=inv_df['Wagon_Type'].unique())

    filtered = inv_df[
        (inv_df['Terminal'] == selected_terminal) &
        (inv_df['Wagon_Type'].isin(selected_wagon_types))
    ].copy()

    # Clean Day_Time and ensure it's a valid string
    filtered = filtered.dropna(subset=['Day', 'Day_Time'])
    filtered = filtered[filtered['Day_Time'].str.match(r"^\d{2}:\d{2}$", na=False)]

    # Base Sunday date
    base_date = pd.Timestamp("2023-12-31")  # Sunday

    # Create datetime using Day and Day_Time
    filtered['Synthetic_Datetime'] = filtered.apply(
        lambda row: base_date + pd.Timedelta(days=int(row['Day'])) + pd.to_timedelta(row['Day_Time'] + ":00"),
        axis=1
    )

    fig = px.line(
        filtered,
        x='Synthetic_Datetime',
        y='Inventory',
        color='Wagon_Type',
        title=f"Inventory Trend from Sunday to Saturday - {selected_terminal}",
        labels={"Synthetic_Datetime": "Day & Time", "Inventory": "Wagons Available"}
    )
    fig.update_layout(
        xaxis_tickformat="%a %H:%M",
        xaxis_title="Day of Week & Time",
        yaxis_title="Wagon Inventory"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(filtered, use_container_width=True)

    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            buffer = io.StringIO()
            inv_df.to_csv(buffer, index=False)
            st.download_button(
                label="📦 Download Inventory Balance",
                data=buffer.getvalue(),
                file_name="inventory_levels.csv",
                mime="text/csv"
            )

        with col2:
            buffer = io.StringIO()
            adjustments_df.to_csv(buffer, index=False)
            st.download_button(
                label="🔄 Download Wagon Adjustments",
                data=buffer.getvalue(),
                file_name="inventory_adjustments.csv",
                mime="text/csv"
            )

# Optional: show adjustments
# if "adjustment_df" in st.session_state:
#     st.subheader("📉 Inventory Adjustments")
#     st.dataframe(st.session_state["adjustment_df"], use_container_width=True)


