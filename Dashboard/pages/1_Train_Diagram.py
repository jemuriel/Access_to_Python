import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
from io import StringIO
import random

st.set_page_config(layout="wide")

# Style Tweaks
reduce_header_height_style = """
    <style>
        div.block-container {padding-top:1rem; padding-bottom:0 rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)

st.markdown("""
<style>
span[data-baseweb="tag"] {
  background-color: #7f7f7f !important;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Train Timetable")

# File Upload
st.subheader("⬆️ Upload a Wagon Plan File")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

# Load into session state
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        if df_uploaded['ARR_TIME'].notna().any():
            df_uploaded['ARR_TIME'] = pd.to_datetime(df_uploaded['ARR_TIME'], errors='coerce')
        if df_uploaded['DEP_TIME'].notna().any():
            df_uploaded['DEP_TIME'] = pd.to_datetime(df_uploaded['DEP_TIME'], errors='coerce')
        st.session_state.train_df = df_uploaded
        st.success("Wagon plan uploaded and loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
elif 'train_df' not in st.session_state:
    path = 'csv_files/filtered_long_train_services.csv'
    st.session_state.train_df = pd.read_csv(path)

df = st.session_state.train_df.copy()

# Ensure datetime format
df['ARR_TIME'] = pd.to_datetime(df['ARR_TIME'], errors='coerce')
df['DEP_TIME'] = pd.to_datetime(df['DEP_TIME'], errors='coerce')

# Corridor & Train Filters
st.subheader("🚉 Filters")
available_corridors = df['CORRIDOR'].dropna().unique()
selected_corridor = st.selectbox("Select Corridor", available_corridors)

filtered_by_corridor = df[df['CORRIDOR'] == selected_corridor].copy()

available_trains = filtered_by_corridor['TRAIN_ID'].dropna().unique()
selected_trains = st.multiselect("Select Trains", available_trains, default=available_trains)

filtered_df = filtered_by_corridor[filtered_by_corridor['TRAIN_ID'].isin(selected_trains)].copy()

# Editable Table
st.subheader("✏️ Edit Train Plan")
edited_df = st.data_editor(filtered_df, num_rows="dynamic")
st.session_state.train_df.update(edited_df)

# Download Edited Plan
csv_buffer = StringIO()
st.session_state.train_df.to_csv(csv_buffer, index=False)
st.download_button("Download Edited Plan", csv_buffer.getvalue(), "edited_train_plan.csv", "text/csv")

# Station Logic
if selected_corridor in ["MEL-BRI", "MEL-SYD"]:
    stations_sorted = ['MFT', 'ETT', 'SFT', 'MOR', 'BFT']
    emphasis_labels = {'MFT': 20, 'SFT': 20, 'BFT': 20}
elif selected_corridor in ["MEL-PER", "MEL-ADL"]:
    stations_sorted = ['MFT', 'SPG', 'TLB', 'APD', 'AFT', 'PAD', 'SCJ', 'COK', 'WEK', 'KWK', 'PFT']
    emphasis_labels = {'MFT': 20, 'AFT': 20, 'PFT': 20}
else:
    stations_sorted = sorted(df['TERMINAL'].unique())
    emphasis_labels = {}

# Plotting
st.subheader("🚂 Train Time-Distance Diagram")
df_filtered = edited_df.copy()
df_filtered['TERMINAL'] = pd.Categorical(df_filtered['TERMINAL'], categories=stations_sorted, ordered=True)

color_map = {train: color for train, color in zip(df['TRAIN_ID'].dropna().unique(), px.colors.qualitative.Alphabet)}

fig = go.Figure()

for train in df_filtered['TRAIN_ID'].unique():
    train_data = df_filtered[df_filtered['TRAIN_ID'] == train].sort_values(by='LEG_NUM')
    # color = color_map.get(train, 'gray')
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

# Day separators
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
    # title='Train Time-Distance Diagram',
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
    height=1000,
    width=2000,
    xaxis=dict(tickformat="%a %H:%M")
)

st.plotly_chart(fig, use_container_width=False)

# Overwrite file section
st.subheader("⚠️ Overwrite Original Train Plan File")

confirm_overwrite = st.checkbox("I understand that this will permanently overwrite the train plan file.")

if confirm_overwrite:
    if st.button("Overwrite 'filtered_long_train_services.csv'"):
        try:
            st.session_state.train_df.to_csv('csv_files/filtered_long_train_services.csv', index=False)
            st.success("Train plan file successfully overwritten.")
        except Exception as e:
            st.error(f"Error saving file: {e}")
else:
    st.info("Check the box above to enable the overwrite button.")
