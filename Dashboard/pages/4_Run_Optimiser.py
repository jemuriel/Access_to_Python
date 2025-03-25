import os
import sys

# 🔧 Fix path for Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ✅ NOW import local modules
from Wagon_Planning.New_Time_Table_Run import NewTimeTable
from Wagon_Planning.Wagon_Assigner import WagonAssigner

import streamlit as st
import pandas as pd
import plotly.express as px


# Set page configuration
st.set_page_config(page_title="Run Assignment", layout="wide")
st.title("🚆 Run Yearly Wagon Assignment")

# Configuration paths
timetable_file = "csv_files/6_train_services.csv"
wagon_mapping = "csv_files/8_wagon_mapping.csv"
disag_forecast_file = "csv_files/5_disag_forecasted_flows_MBM.csv"

inventory_output = "inventory_levels.csv"
adjustments_output = "inventory_adjustments.csv"
wagon_output = "9_wagon_config_transformed_MBM.csv"
train_summary = "10_train_summary_MBM.csv"
unassigned_boxes = "11_unassigned_containers_MBM.csv"
wagon_assignments = "11.1_wagon_assignments_MBM.csv"

# Load forecast data
disag_forecast = pd.read_csv(disag_forecast_file)

# Sidebar parameters
# st.sidebar.header("⚙️ Model Parameters")
# plan_days = st.sidebar.slider("Planning Horizon (Days)", min_value=7, max_value=90, value=30)
# verbose_mode = st.sidebar.selectbox("Verbose Output", options=[0, 1], index=1)

# Force use of default file even if a modified version exists
use_default_checkbox = st.sidebar.checkbox("✅ Force use of default wagon plan file", value=False)

# ===================== 🔍 Wagon Plan Source Decision =====================
st.subheader("📋 Wagon Plan Preview")

if "modified_wagon_plan" in st.session_state and not use_default_checkbox:
    wagon_plan_df = st.session_state["modified_wagon_plan"]
    st.success("Using modified wagon plan from the editor.")
else:
    default_file_path = "csv_files/7.1_wagon_plan_2025.csv"
    wagon_plan_df = pd.read_csv(default_file_path)
    if use_default_checkbox:
        st.info("You have chosen to force use of the default wagon plan file.")
    else:
        st.warning("No modified plan found — using default file instead.")

st.dataframe(wagon_plan_df, use_container_width=True)

# ===================== 🚀 Run Assignment =====================
if st.button("Run Yearly Assignment Model"):
    st.info("Running assignment...")

    # Save temp version for backend usage
    temp_wagon_path = "temp_wagon_plan.csv"
    wagon_plan_df.to_csv(temp_wagon_path, index=False)

    train_timetable = NewTimeTable(30, timetable_file, temp_wagon_path, wagon_mapping,
                                   inventory_output, adjustments_output, wagon_output, 1)

    WagonAssigner.run_yearly_assignment(disag_forecast, train_timetable,
                                        train_summary, wagon_assignments, unassigned_boxes)

    st.success("✅ Assignment completed successfully!")

    # Load and store train summary in session state
    train_summary_df = pd.read_csv(train_summary)
    train_summary_df.columns = [
        'Train_Number', 'Leg_OD', 'Leg_ID', 'Date',
        'Total_Capacity', 'Total_Utilised', 'Occupancy',
        'Total_Packs', 'Packs_Used', 'Unassigned_Containers', 'Unassigned_Length'
    ]
    train_summary_df['Date'] = pd.to_datetime(train_summary_df['Date'], format='mixed', dayfirst=True, errors='coerce').dt.date
    train_summary_df['Occupancy (%)'] = (train_summary_df['Occupancy'] * 100).round(2)

    st.session_state["train_summary_df"] = train_summary_df

    # Download buttons
    with open(train_summary, "rb") as f1, open(unassigned_boxes, "rb") as f2, open(wagon_assignments, "rb") as f3:
        st.download_button("📥 Download Train Summary", f1, file_name="train_summary.csv")
        st.download_button("📥 Download Unassigned Containers", f2, file_name="unassigned_containers.csv")
        st.download_button("📥 Download Wagon Assignments", f3, file_name="wagon_assignments.csv")

# ===================== 📊 Show Chart + Table Persistently =====================
if "train_summary_df" in st.session_state:
    train_summary_df = st.session_state["train_summary_df"]
    st.subheader("📊 Filtered Train Summary with Occupancy Chart")

    train_list = train_summary_df['Train_Number'].unique().tolist()
    selected_trains = st.multiselect("🚆 Select Train Number(s) to Visualise:", train_list, default=train_list)

    filtered_summary = train_summary_df[train_summary_df['Train_Number'].isin(selected_trains)]

    st.dataframe(filtered_summary.drop(columns=['Occupancy']), use_container_width=True)

    fig = px.bar(
        filtered_summary,
        x='Date',
        y='Occupancy (%)',
        color='Train_Number',
        barmode='group',
        title="Occupancy % Per Train Over Time",
        labels={'Occupancy (%)': 'Occupancy (%)', 'Date': 'Date', 'Train_Number': 'Train Number'}
    )
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Occupancy (%)',
        legend_title='Train',
        yaxis_tickformat='.0f'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("ℹ️ Run the assignment to view Train Summary and charts.")
