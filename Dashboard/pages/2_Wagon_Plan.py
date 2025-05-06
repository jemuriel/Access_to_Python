import streamlit as st
import pandas as pd
import requests
import tempfile
import os
from fpdf import FPDF
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Wagon Plan Editor")

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

def upload_table(df, table_name):
    df_serialized = df.copy()
    for col in df_serialized.select_dtypes(include=["datetime", "datetimetz"]):
        df_serialized[col] = df_serialized[col].astype(str)
    url = f"{SUPABASE_URL}/rest/v1/{table_name}"
    res = requests.post(
        url,
        headers={**HEADERS, "Prefer": "return=representation"},  # Changed from merge-duplicates
        json=df_serialized.to_dict(orient="records")
    )
    res.raise_for_status()

def delete_versions(table_name):
    url = f"{SUPABASE_URL}/rest/v1/{table_name}?version_label=neq.null"
    res = requests.delete(url, headers={**HEADERS, "Prefer": "return=representation"})
    res.raise_for_status()
    return res.status_code in [200, 204]

def create_pdf(dataframe):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Wagon Plan", ln=True, align='L')
    pdf.ln(5)
    for train in dataframe['TRAIN_NAME'].unique():
        train_data = dataframe[dataframe['TRAIN_NAME'] == train]
        primary_train = train[:4]
        departing_location = train[-3:]
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(40, 10, "Primary Train")
        pdf.set_font("Arial", '', 12)
        pdf.cell(40, 10, f"{primary_train}", ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.cell(40, 10, f"Train {train}")
        pdf.cell(70, 10, f"Departing Location {departing_location}", ln=True)
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(65, 10, "SERVICE", 1)
        pdf.cell(65, 10, "WAGON CLASS", 1)
        pdf.cell(60, 10, "NUM WAGONS", 1, ln=True)
        total = 0
        for _, row in train_data.iterrows():
            pdf.set_font("Arial", '', 10)
            pdf.cell(65, 10, str(row['SERVICE']), 1)
            pdf.cell(65, 10, str(row['WAGON_CLASS']), 1)
            pdf.cell(60, 10, str(row['NUM_WAGONS']), 1, ln=True)
            total += row['NUM_WAGONS']
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(255, 255, 0)
        pdf.cell(130, 10, "TOTAL", 1, 0, 'L', fill=True)
        pdf.cell(60, 10, f"{total:.2f}", 1, 1, 'R', fill=True)
        pdf.ln(5)
    return pdf

# --- Load Wagon Plans ---
wagon_base = fetch_table("base_wagon_plan")
wagon_versions = fetch_table("wagon_plan_versions")
all_wagon_plans = pd.concat([wagon_base, wagon_versions], ignore_index=True)
all_wagon_plans["version_time"] = pd.to_datetime(all_wagon_plans.get("saved_at", pd.Timestamp.now()), errors="coerce")

# Reload if new version was saved
all_wagon_plans = st.session_state.get("all_wagon_plans", all_wagon_plans)
version_labels = all_wagon_plans["version_label"].fillna("Original").unique().tolist()

# Pre-assign version label before selectbox
if "next_version_to_select" in st.session_state:
    st.session_state.selected_version_label = st.session_state.pop("next_version_to_select")

# Track current version selection
if "selected_version_label" not in st.session_state:
    st.session_state.selected_version_label = "Original"

selected_version_label = st.sidebar.selectbox(
    "Select Wagon Plan Version",
    version_labels,
    index=version_labels.index(st.session_state.selected_version_label),
    key="selected_version_label"
)

# Filter selected version
current_plan_df = all_wagon_plans[all_wagon_plans["version_label"].fillna("Original") == st.session_state.selected_version_label]

# Upload New Wagon Plan
uploaded_file = st.file_uploader("Upload New Wagon Plan (CSV)", type="csv")
if uploaded_file:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        required_cols = {'TRAIN_NAME', 'WAGON_CLASS', 'NUM_WAGONS'}
        if not required_cols.issubset(uploaded_df.columns):
            raise ValueError("Uploaded file is missing required columns.")
        current_plan_df = uploaded_df
        st.success("New wagon plan uploaded.")
    except Exception as e:
        st.error(f"Upload failed: {e}")

# --- Origin Filter ---
st.header("\U0001F683 Filter by Origin")
origin_options = current_plan_df['ORIGIN'].unique()
selected_origin = st.selectbox("Select Origin:", options=origin_options)
filtered_df = current_plan_df[current_plan_df['ORIGIN'] == selected_origin]

# --- Editable Wagon Plan ---
st.subheader("\u270F\ufe0f Editable Wagon Plan")
if "editable_df" not in st.session_state or (
    st.session_state.get("origin_tracker") != selected_origin or
    st.session_state.get("version_tracker") != st.session_state.selected_version_label
):
    st.session_state.editable_df = filtered_df.copy()
    st.session_state.editable_df['NUM_WAGONS'] = st.session_state.editable_df['NUM_WAGONS'].astype(int)
    st.session_state.origin_tracker = selected_origin
    st.session_state.version_tracker = st.session_state.selected_version_label

st.session_state.editable_df = st.data_editor(
    st.session_state.editable_df,
    num_rows="dynamic",
    use_container_width=True
)

# --- Save Modified Version ---
if st.button("\U0001F4BE Save Modified Wagon Plan"):
    version_label = f"Modified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load full current version before update
    full_current_plan = all_wagon_plans[
        all_wagon_plans["version_label"].fillna("Original") == st.session_state.selected_version_label
    ].copy()

    # Replace edited origin rows in full plan
    edited_origin = selected_origin.upper().strip()
    full_current_plan = full_current_plan[
        full_current_plan["ORIGIN"].astype(str).str.upper().str.strip() != edited_origin
    ]

    updated_plan = pd.concat([full_current_plan, st.session_state.editable_df.copy()], ignore_index=True)
    updated_plan['version_label'] = version_label
    updated_plan['saved_at'] = datetime.now().isoformat()

    try:
        upload_table(updated_plan, "wagon_plan_versions")
        st.success(f"\u2705 Saved new wagon plan version: {version_label}")
        st.session_state.all_wagon_plans = pd.concat([all_wagon_plans, updated_plan], ignore_index=True)
        st.session_state.next_version_to_select = version_label
        st.rerun()
    except Exception as e:
        st.error(f"\u274C Failed to save version: {e}")

# --- Delete All Modified Versions ---
if st.button("\U0001F5D1\ufe0f Delete All Modified Versions"):
    try:
        delete_versions("wagon_plan_versions")
        st.success("\u2705 All modified wagon plan versions deleted.")
    except Exception as e:
        st.error(f"Error deleting modified versions: {e}")

# --- Detailed Train-wise Display ---
st.subheader("\U0001F682 Detailed Train-wise Wagon Plan")
with st.expander("View Detailed Wagon Plan", expanded=True):
    editable_df = st.session_state.editable_df.copy()
    for train in editable_df['TRAIN_NAME'].unique():
        train_data = editable_df[editable_df['TRAIN_NAME'] == train]
        primary_train = train[:4]
        departing_location = train[-3:]
        st.markdown(f"### Primary Train {primary_train}")
        st.markdown(f"**Train**: {train} &nbsp;&nbsp; **Departing Location**: {departing_location}")
        st.dataframe(train_data[['SERVICE', 'WAGON_CLASS', 'NUM_WAGONS']])
        st.markdown(f"**TOTAL**: {train_data['NUM_WAGONS'].sum():.2f}")
        st.markdown("---")

# --- Wagon Class Summary ---
st.subheader("\U0001F6A9 Wagon Class Summary")
summary = st.session_state.editable_df.groupby('WAGON_CLASS').agg({'NUM_WAGONS': 'sum'}).reset_index()
st.dataframe(summary, use_container_width=True)

# --- Download Section ---
st.subheader("\U0001F4E5 Download Modified Wagon Plan")
pdf = create_pdf(st.session_state.editable_df)
temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
pdf.output(temp_pdf.name)

st.download_button("Download as CSV", st.session_state.editable_df.to_csv(index=False).
                   encode('utf-8'), "modified_wagon_plan.csv", "text/csv")
with open(temp_pdf.name, "rb") as f:
    st.download_button("Download as PDF", f, "wagon_plan.pdf", "application/pdf")

# --- Compare Two Versions ---
st.subheader("\U0001F197 Compare Two Wagon Plan Versions (Color Highlighted)")
col1, col2 = st.columns(2)
with col1:
    version_1 = st.selectbox("Select First Version", options=version_labels, key="version_1")
with col2:
    version_2 = st.selectbox("Select Second Version", options=version_labels, key="version_2")

df_v1 = all_wagon_plans[all_wagon_plans["version_label"].fillna("Original") == version_1].copy()
df_v2 = all_wagon_plans[all_wagon_plans["version_label"].fillna("Original") == version_2].copy()

common_cols = ["TRAIN_NAME", "TRAIN", "ORIGIN", "SERVICE", "WAGON_CLASS", "NUM_WAGONS", "LEG_NUM"]
df_v1 = df_v1[common_cols]
df_v2 = df_v2[common_cols]

# Merge and compare
df_compare = df_v1.merge(df_v2, on=["TRAIN_NAME", "TRAIN", "ORIGIN", "SERVICE", "WAGON_CLASS", "LEG_NUM"], how="outer",
                         suffixes=("_v1", "_v2"), indicator=True)
df_compare["change"] = df_compare["NUM_WAGONS_v2"].fillna(0) - df_compare["NUM_WAGONS_v1"].fillna(0)
diff_df = df_compare[(df_compare["_merge"] != "both") | (df_compare["change"] != 0)]
diff_df = diff_df.sort_values("change", ascending=False)

# Prepare colors per cell (white or red)
def color_if_diff(val1, val2):
    return "red" if val1 != val2 else "white"

highlight_colors = {
    "NUM_WAGONS_v1": [color_if_diff(r1, r2) for r1, r2 in zip(diff_df["NUM_WAGONS_v1"], diff_df["NUM_WAGONS_v2"])],
    "NUM_WAGONS_v2": [color_if_diff(r2, r1) for r1, r2 in zip(diff_df["NUM_WAGONS_v1"], diff_df["NUM_WAGONS_v2"])],
    "change": ["red" if c != 0 else "white" for c in diff_df["change"]],
    "_merge": ["red" if m != "both" else "white" for m in diff_df["_merge"]],
}

fig = go.Figure(data=[go.Table(
    header=dict(values=["TRAIN_NAME", "SERVICE", "WAGON_CLASS", "ORIGIN", "NUM_WAGONS_v1",
                        "NUM_WAGONS_v2", "Change", "Status"],
                # fill_color='black',
                font=dict(color='white'), align='left'),
    cells=dict(
        values=[
            diff_df["TRAIN_NAME"],
            diff_df["SERVICE"],
            diff_df["WAGON_CLASS"],
            diff_df["ORIGIN"],
            diff_df["NUM_WAGONS_v1"],
            diff_df["NUM_WAGONS_v2"],
            diff_df["change"],
            diff_df["_merge"]
        ],
        # fill_color='black',
        font=dict(
            color=[
                "white", "white", "white", "white",
                highlight_colors["NUM_WAGONS_v1"],
                highlight_colors["NUM_WAGONS_v2"],
                highlight_colors["change"],
                highlight_colors["_merge"]
            ]
        ),
        align='left'
    )
)])
fig.update_layout(width=1200, height=600)
st.plotly_chart(fig, use_container_width=True)

added = (diff_df["_merge"] == "right_only").sum()
removed = (diff_df["_merge"] == "left_only").sum()
changed = ((diff_df["_merge"] == "both") & (diff_df["change"] != 0)).sum()
st.info(f"\U0001F4CA Summary: {changed} modified, {added} added, {removed} removed.")