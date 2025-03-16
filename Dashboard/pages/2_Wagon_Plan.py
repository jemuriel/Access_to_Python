import streamlit as st
import pandas as pd

# Load Data
file_path = 'csv_files/7.1_wagon_plan_2025.csv'
df = pd.read_csv(file_path)

st.title("Wagon Plan")

# Filter by Train Name
train_options = df['TRAIN_NAME'].unique()
selected_trains = st.multiselect("Select Train Name(s):", options=train_options, default=train_options)
filtered_df = df[df['TRAIN_NAME'].isin(selected_trains)]

# Editable Data Section
st.subheader("Editable Wagon Plan")
editable_df = filtered_df.copy()
editable_df['NUM_WAGONS'] = editable_df['NUM_WAGONS'].astype(int)
editable_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

# Group and display data similar to the image style
for train in editable_df['TRAIN_NAME'].unique():
    train_data = editable_df[editable_df['TRAIN_NAME'] == train]
    primary_train = train[:4]  # assuming format like 1431ENF -> 1431
    departing_location = train[-3:]  # assuming ENF

    st.markdown(f"### Primary Train {primary_train}")
    st.markdown(f"**Train**: {train} &nbsp;&nbsp;&nbsp;&nbsp; **Departing Location**: {departing_location}")

    display_data = train_data[['SERVICE', 'WAGON_CLASS', 'NUM_WAGONS']]
    st.dataframe(display_data.reset_index(drop=True), use_container_width=True)

    total_wagons = display_data['NUM_WAGONS'].sum()
    st.markdown(f"**TOTAL**: {total_wagons:.2f}")
    st.markdown("---")

# Second table - Filter by Wagon Class
st.subheader("Wagon Class Summary")
wagon_classes = df['WAGON_CLASS'].unique()
selected_classes = st.multiselect("Select Wagon Class(es):", options=wagon_classes, default=wagon_classes)
class_filtered_df = editable_df[editable_df['WAGON_CLASS'].isin(selected_classes)]

summary = class_filtered_df.groupby(['WAGON_CLASS']).agg({'NUM_WAGONS': 'sum'}).reset_index()
st.dataframe(summary, use_container_width=True)

# Optional: subtotal by wagon class and train
subtotal = class_filtered_df.groupby(['TRAIN_NAME', 'WAGON_CLASS']).agg({'NUM_WAGONS': 'sum'}).reset_index()
st.dataframe(subtotal, use_container_width=True)

# Export Modified Plan
st.subheader("Download Modified Wagon Plan")
modified_csv = editable_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Modified Plan as CSV",
    data=modified_csv,
    file_name='modified_wagon_plan.csv',
    mime='text/csv'
)
