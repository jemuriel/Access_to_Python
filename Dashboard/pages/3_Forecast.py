import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Disaggregated Forecast Results")

# Load your dataset
df = pd.read_csv('csv_files/5_disag_forecasted_flows_MBM.csv')

# Parse DATE properly
df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')
df = df.dropna(subset=["DATE"])

# Sidebar Filters
st.sidebar.header("🔍 Filters")
min_date = df["DATE"].min()
max_date = df["DATE"].max()

# Safe fallback for empty dates
if pd.isna(min_date) or pd.isna(max_date):
    st.warning("⚠ No valid dates found in your data. Please check the 'DATE' column.")
    min_date = max_date = pd.to_datetime("today")

date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Apply date filter globally
df_filtered = df.copy()
if len(date_range) == 2:
    df_filtered = df_filtered[(df_filtered["DATE"] >= pd.to_datetime(date_range[0])) &
                              (df_filtered["DATE"] <= pd.to_datetime(date_range[1]))]

# OD Pair filter
od_pair_options = sorted(df["OD_PAIR"].unique())
selected_od_pairs = st.sidebar.multiselect("OD Pair", od_pair_options, default=od_pair_options)

# Train Number filter
train_options = sorted(df["TRAIN_NUM"].unique())
selected_trains = st.sidebar.multiselect("Train Number", train_options, default=train_options)

# Box Type filter
box_type_options = sorted(df["BOX_TYPE"].unique())
selected_box_types = st.sidebar.multiselect("Box Type", box_type_options, default=box_type_options)

# ---------------- Aggregate Data ---------------- #

# Aggregate total NUM_BOXES per DATE for each chart requirement

# Chart 1: Aggregate only by DATE and OD_PAIR (OD Pair should not be affected by other filters)
df_od_chart = df_filtered[df_filtered["OD_PAIR"].isin(selected_od_pairs)]
df_od_chart = df_od_chart.groupby(["DATE", "OD_PAIR"], as_index=False)["NUM_BOXES"].sum()

# Chart 2 & 3: Apply all filters and aggregate per DATE
df_train_chart = df_filtered[
    (df_filtered["OD_PAIR"].isin(selected_od_pairs)) &
    (df_filtered["TRAIN_NUM"].isin(selected_trains)) &
    (df_filtered["BOX_TYPE"].isin(selected_box_types))
]
df_train_chart = df_train_chart.groupby(["DATE", "TRAIN_NUM"], as_index=False)["NUM_BOXES"].sum()

df_box_chart = df_filtered[
    (df_filtered["OD_PAIR"].isin(selected_od_pairs)) &
    (df_filtered["TRAIN_NUM"].isin(selected_trains)) &
    (df_filtered["BOX_TYPE"].isin(selected_box_types))
]
df_box_chart = df_box_chart.groupby(["DATE", "BOX_TYPE"], as_index=False)["NUM_BOXES"].sum()


# ---------------- Charts ---------------- #

# Chart 1: Number of Boxes by OD Pair
st.subheader("📦 Number of Boxes by OD Pair over Time")
fig1 = px.line(df_od_chart, x="DATE", y="NUM_BOXES", color="OD_PAIR", markers=True,
               title="Boxes by OD Pair")
fig1.update_layout(xaxis_title="Date", yaxis_title="Number of Boxes",
                   xaxis_tickformat="%Y-%m-%d")
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Number of Boxes by Train Number
st.subheader("🚆 Number of Boxes by Train Number over Time")
fig2 = px.line(df_train_chart, x="DATE", y="NUM_BOXES", color="TRAIN_NUM", markers=True,
               title="Boxes by Train Number")
fig2.update_layout(xaxis_title="Date", yaxis_title="Number of Boxes",
                   xaxis_tickformat="%Y-%m-%d")
st.plotly_chart(fig2, use_container_width=True)

# Chart 3: Number of Boxes by Box Type
st.subheader("📦 Number of Boxes by Box Type over Time")
fig3 = px.line(df_box_chart, x="DATE", y="NUM_BOXES", color="BOX_TYPE", markers=True,
               title="Boxes by Box Type")
fig3.update_layout(xaxis_title="Date", yaxis_title="Number of Boxes",
                   xaxis_tickformat="%Y-%m-%d")
st.plotly_chart(fig3, use_container_width=True)

