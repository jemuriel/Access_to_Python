import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
import io
import re

from Forecast_Disag.Improved_Prob_Model import Probabilistic_Model

st.set_page_config(layout="wide")
st.title("Raw Forecast")

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
        headers = {
            **HEADERS,
            "Range-Unit": "items",
            "Range": f"{offset}-{offset + chunk_size - 1}"
        }
        url = f"{SUPABASE_URL}/rest/v1/{table_name}?select=*"
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

def upload_table(df, table_name, replace=False):
    df_serialized = df.copy()
    for col in df_serialized.select_dtypes(include=["datetime", "datetimetz"]):
        df_serialized[col] = df_serialized[col].astype(str)
    url = f"{SUPABASE_URL}/rest/v1/{table_name}"
    method = requests.put if replace else requests.post
    res = method(
        url,
        headers={**HEADERS, "Prefer": "resolution=merge-duplicates"},
        json=df_serialized.to_dict(orient="records")
    )
    res.raise_for_status()

def extract_timestamp(label):
    match = re.search(r'\d{8}_\d{6}', str(label))
    return datetime.strptime(match.group(), '%Y%m%d_%H%M%S') if match else datetime.min

def reload_combined_forecast():
    # raw_forecast = fetch_table("raw_forecast")
    versioned_forecast = fetch_table("forecast_versions")
    combined = pd.concat([raw_forecast, versioned_forecast], ignore_index=True)
    combined["DATE"] = pd.to_datetime(combined["DATE"], format="%d/%m/%Y", errors="coerce")
    combined["Month"] = combined["DATE"].dt.strftime("%Y-%m")
    combined["version_time"] = combined["version_label"].apply(extract_timestamp)
    return combined


# --- Initial Load ---
# combined_forecast = reload_combined_forecast()

# --- Select version label later ---
selected_version_label = None
version_data = pd.DataFrame()

# --- Filters (Preliminary) ---
raw_forecast = fetch_table("raw_forecast")
raw_forecast["DATE"] = pd.to_datetime(raw_forecast["DATE"], format="%d/%m/%Y", errors="coerce")
raw_forecast["Month"] = raw_forecast["DATE"].dt.strftime("%Y-%m")

month_options = sorted(raw_forecast["Month"].dropna().unique())
od_pair_options = sorted(raw_forecast["OD_PAIR"].dropna().unique())
preselected_od_pairs = ['AFT-BFT', 'BFT-AFT', 'MFT-AFT', 'MFT-PFT', 'MFT-BFT', 'MFT-SFT']
selected_months = st.multiselect("🕕️ Select Month(s)", month_options, default=month_options)
selected_od_pairs = st.multiselect("🚛 Select OD Pairs", od_pair_options, default=preselected_od_pairs)

# --- Edit Area ---
st.subheader("✏️ Edit Monthly Forecast")
filtered_raw = raw_forecast[
    raw_forecast["Month"].isin(selected_months) &
    raw_forecast["OD_PAIR"].isin(selected_od_pairs)
]

edited_df = st.data_editor(
    filtered_raw,
    use_container_width=True,
    num_rows="dynamic",
    key="editable_monthly"
)

# --- Buttons ---
# col_delete, col_update = st.columns([1, 1])
# with col_delete:
if st.button("🗑️ Delete All Modified Forecasts", key="delete_modified_forecasts_button"):
    url = f"{SUPABASE_URL}/rest/v1/forecast_versions?version_label=neq.null"
    res = requests.delete(url, headers={**HEADERS, "Prefer": "return=representation"})
    if res.status_code in [200, 204]:
        st.session_state.pop("modified_forecast_df", None)
        st.markdown("<p style='color: green; font-size: 0.85rem;'>✅ Modified versions deleted.</p>", unsafe_allow_html=True)
    else:
        st.error(f"Failed to delete modified forecasts: {res.status_code}")
        st.text(res.text)

# with col_update:
st.write("")
update_triggered = st.button("📊 Update Forecast", key="show_update_forecast_button")

# --- Forecast Charts ---
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Original Forecast**")
    fig_original = px.line(
        filtered_raw,
        x="DATE", y="NUM_BOXES", color="OD_PAIR", markers=True
    )
    fig_original.update_layout(xaxis_title="Month", yaxis_title="TEUs", xaxis_tickformat="%b %Y")
    st.plotly_chart(fig_original, use_container_width=True, key="original_forecast_chart")

with col2:
    if "modified_forecast_df" in st.session_state:
        modified_df = st.session_state["modified_forecast_df"]
    else:
        modified_df = raw_forecast.copy()

    if update_triggered:
        updated_rows = edited_df.copy()
        index_cols = ["DATE", "OD_PAIR"]
        modified_df.set_index(index_cols, inplace=True)
        updated_rows.set_index(index_cols, inplace=True)
        modified_df.update(updated_rows)
        modified_df.reset_index(inplace=True)

        version_label = f"Modified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        modified_df['version_label'] = version_label

        forecast_versions_df = modified_df[["DATE", "OD_PAIR", "NUM_BOXES", "version_label"]]
        forecast_versions_df['DATE'] = forecast_versions_df['DATE'].dt.strftime('%d/%m/%Y')

        upload_table(forecast_versions_df, "forecast_versions")

        st.session_state["selected_forecast_version"] = version_label
        st.session_state["modified_forecast_df"] = modified_df.copy()
        st.markdown(f"<p style='color: green; font-size: 0.85rem;'>✅ Forecast updated: {version_label}</p>", unsafe_allow_html=True)

    df_chart = modified_df[
        modified_df["Month"].isin(selected_months) & modified_df["OD_PAIR"].isin(selected_od_pairs)
    ]
    st.markdown("**Modified Forecast**")
    fig_modified = px.line(
        df_chart,
        x="DATE",
        y="NUM_BOXES",
        color="OD_PAIR",
        # title="Modified Forecast",
        markers=True
    )
    fig_modified.update_layout(xaxis_title="Month", yaxis_title="TEUs", xaxis_tickformat="%b %Y")
    st.plotly_chart(fig_modified, use_container_width=True, key="modified_forecast_chart")

# --- Sidebar Forecast Version Selector (at the end) ---
combined_forecast = reload_combined_forecast()
all_versions_df = combined_forecast.sort_values("version_time", ascending=False).groupby("version_label").first().reset_index()
all_versions_sorted = all_versions_df["version_label"].tolist()
default_version_label = st.session_state.get("selected_forecast_version", all_versions_sorted[0])

st.sidebar.header("📜 Forecast Version")
selected_version_label = st.sidebar.selectbox(
    "Select Forecast Version to Use",
    all_versions_sorted,
    index=all_versions_sorted.index(default_version_label)
    if default_version_label in all_versions_sorted else 0
)

# Clear modified forecast if version changed
if st.session_state.get("selected_forecast_version") != selected_version_label:
    st.session_state.pop("modified_forecast_df", None)

# --- Run Disaggregation Model -----------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading historical data...")
def get_historical_data():
    return fetch_table("historical_data")

@st.cache_data(show_spinner=False)
def run_disaggregation_model(version_df, label):
    # historical_df = get_historical_data()
    historical_df = pd.read_csv('csv_files/historical_data.csv')
    box_types_df = pd.read_csv('csv_files/box_types.csv')

    model = Probabilistic_Model(
        historical_file=historical_df,
        forecast_file=version_df,
        box_types_file=box_types_df,
        output_file=f"temp_disagg_output_{label.replace(' ', '_').replace(':', '')}.csv"
    )

    result_df = model.run_complete_model()
    result_df["version_label"] = label
    result_df["saved_at"] = datetime.now().isoformat()
    upload_table(result_df, "disag_forecast")
    return result_df, "disag_forecast"


st.subheader("▶️ Run Disaggregation Model")
if st.button("Run Disaggregation Model"):
    version_df = combined_forecast[combined_forecast["version_label"] == selected_version_label].copy()
    if version_df.empty:
        st.warning("Selected version not found. Defaulting to the first available.")
        version_df = combined_forecast[combined_forecast["version_label"] == default_version_label].copy()

    st.info(f"Running disaggregation model for forecast version: {selected_version_label}")
    result_df, version_table = run_disaggregation_model(version_df, selected_version_label)
    st.success(f"Results saved to: {version_table}")

    # --- Download Button ---
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Disaggregated Forecast", csv_buffer.getvalue().encode(), "Disaggregated_forecast.csv")

    # --- Chart Filters ---
    st.sidebar.header("📊 Filters (Disaggregated)")
    result_df["DATE"] = pd.to_datetime(result_df["DATE"], errors="coerce")
    result_df = result_df.dropna(subset=["DATE"])

    od_opts = sorted(result_df["OD_PAIR"].dropna().unique())
    train_opts = sorted(result_df["TRAIN_NUM"].dropna().unique())
    box_opts = sorted(result_df["BOX_TYPE"].dropna().unique())

    dr = st.sidebar.date_input("Date Range", [result_df["DATE"].min(), result_df["DATE"].max()])
    od_sel = st.sidebar.multiselect("OD Pair", od_opts, default=['AFT-BFT', 'BFT-AFT', 'MFT-BFT'])
    train_sel = st.sidebar.multiselect("Train Number", train_opts, default=train_opts)
    box_sel = st.sidebar.multiselect("Box Type", box_opts, default=box_opts)

    df_filtered = result_df[
        (result_df["DATE"] >= pd.to_datetime(dr[0])) &
        (result_df["DATE"] <= pd.to_datetime(dr[1])) &
        (result_df["OD_PAIR"].isin(od_sel)) &
        (result_df["TRAIN_NUM"].isin(train_sel)) &
        (result_df["BOX_TYPE"].isin(box_sel))
    ]

    st.dataframe(df_filtered[["DATE", "TRAIN_NUM", "BOX_TYPE", "OD_PAIR", "CALCULATED_TEUS", "NUM_BOXES"]],
                 use_container_width=True)

    st.subheader("📦 Boxes by OD Pair")
    st.plotly_chart(px.line(
        df_filtered.groupby(["DATE", "OD_PAIR"], as_index=False).sum(),
        x="DATE", y="NUM_BOXES", color="OD_PAIR", markers=True),
        use_container_width=True)

    st.subheader("🚆 Boxes by Train")
    st.plotly_chart(px.line(
        df_filtered.groupby(["DATE", "TRAIN_NUM"], as_index=False).sum(),
        x="DATE", y="NUM_BOXES", color="TRAIN_NUM", markers=True),
        use_container_width=True)

    st.subheader("📦 Boxes by Box Type")
    st.plotly_chart(px.line(
        df_filtered.groupby(["DATE", "BOX_TYPE"], as_index=False).sum(),
        x="DATE", y="NUM_BOXES", color="BOX_TYPE", markers=True),
        use_container_width=True)



