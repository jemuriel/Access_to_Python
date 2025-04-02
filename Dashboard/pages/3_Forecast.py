import streamlit as st
import pandas as pd
import plotly.express as px
import io

st.set_page_config(layout="wide")

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:1rem; padding-bottom:0 rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)

st.title("Raw Forecast")

# --- Load Main Dataset ---
df = pd.read_csv('csv_files/5_disag_forecasted_flows_MBM.csv')
df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')
df = df.dropna(subset=["DATE"])

# --- Load Monthly Forecast Dataset ---
df_monthly = pd.read_csv('csv_files/4_PN_Forecast.csv')
df_monthly["DATE"] = pd.to_datetime(df_monthly["DATE"], dayfirst=True, errors='coerce')
df_monthly = df_monthly.dropna(subset=["DATE"])
df_monthly["MONTH"] = df_monthly["DATE"].dt.to_period("M").astype(str)

# Sidebar Filters
st.sidebar.header("🔍 Filters")

od_pair_options = sorted(df["OD_PAIR"].unique())
selected_od_pairs = st.sidebar.multiselect("OD Pair", od_pair_options, default=od_pair_options)

min_date = df["DATE"].min()
max_date = df["DATE"].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

train_options = sorted(df["TRAIN_NUM"].unique())
selected_trains = st.sidebar.multiselect("Train Number", train_options, default=train_options)

box_type_options = sorted(df["BOX_TYPE"].unique())
selected_box_types = st.sidebar.multiselect("Box Type", box_type_options, default=box_type_options)

month_options = sorted(df_monthly["MONTH"].unique())
selected_months = st.multiselect("📅 Select Month(s) for Monthly Chart", month_options, default=month_options)

# Filter Monthly Chart Data
df_monthly_chart = df_monthly[df_monthly["OD_PAIR"].isin(selected_od_pairs)]
df_monthly_chart = df_monthly_chart[df_monthly_chart["MONTH"].isin(selected_months)]

# Editable Forecast Table
st.subheader("✏️ Edit Monthly Forecast Data")
edited_df = st.data_editor(df_monthly_chart, use_container_width=True, num_rows="dynamic", key="editable_monthly")

# Update Button
if st.button("🔄 Update Forecast Chart"):
    df_monthly_chart = edited_df.copy()

# Download Button
csv_buffer = io.StringIO()
edited_df.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode('utf-8')

st.download_button(
    label="📥 Download Modified Forecast",
    data=csv_bytes,
    file_name="modified_monthly_forecast.csv",
    mime="text/csv"
)

# Redraw Chart with possibly updated data
st.subheader("📈 Monthly Forecast by OD Pair")
fig0 = px.line(df_monthly_chart, x="DATE", y="NUM_BOXES", color="OD_PAIR", markers=True,
               title="Monthly Forecast - Boxes by OD Pair")
fig0.update_layout(xaxis_title="Month", yaxis_title="Number of TEUs",
                   xaxis_tickformat="%b %Y")
st.plotly_chart(fig0, use_container_width=True)

st.title("Disaggregated Forecast Results")

# Apply Filters on Detailed Forecast
df_filtered = df.copy()
if len(date_range) == 2:
    df_filtered = df_filtered[(df_filtered["DATE"] >= pd.to_datetime(date_range[0])) &
                              (df_filtered["DATE"] <= pd.to_datetime(date_range[1]))]
df_filtered = df_filtered[df_filtered["OD_PAIR"].isin(selected_od_pairs)]
df_filtered = df_filtered[df_filtered["TRAIN_NUM"].isin(selected_trains)]
df_filtered = df_filtered[df_filtered["BOX_TYPE"].isin(selected_box_types)]

# Aggregations
df_od_chart = df_filtered.groupby(["DATE", "OD_PAIR"], as_index=False)["NUM_BOXES"].sum()
df_train_chart = df_filtered.groupby(["DATE", "TRAIN_NUM"], as_index=False)["NUM_BOXES"].sum()
df_box_chart = df_filtered.groupby(["DATE", "BOX_TYPE"], as_index=False)["NUM_BOXES"].sum()

# Detailed Forecast Charts
st.subheader("📦 Number of Boxes by OD Pair over Time")
fig1 = px.line(df_od_chart, x="DATE", y="NUM_BOXES", color="OD_PAIR", markers=True,
               title="Boxes by OD Pair")
fig1.update_layout(xaxis_title="Date", yaxis_title="Number of Boxes", xaxis_tickformat="%Y-%m-%d")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🚆 Number of Boxes by Train Number over Time")
fig2 = px.line(df_train_chart, x="DATE", y="NUM_BOXES", color="TRAIN_NUM", markers=True,
               title="Boxes by Train Number")
fig2.update_layout(xaxis_title="Date", yaxis_title="Number of Boxes", xaxis_tickformat="%Y-%m-%d")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📦 Number of Boxes by Box Type over Time")
fig3 = px.line(df_box_chart, x="DATE", y="NUM_BOXES", color="BOX_TYPE", markers=True,
               title="Boxes by Box Type")
fig3.update_layout(xaxis_title="Date", yaxis_title="Number of Boxes", xaxis_tickformat="%Y-%m-%d")
st.plotly_chart(fig3, use_container_width=True)
