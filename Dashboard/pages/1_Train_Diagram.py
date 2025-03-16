import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta

st.set_page_config(layout="wide")

reduce_header_height_style = """
    <style>
        div.block-container {padding-top:1rem; padding-bottom:0 rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)


# Change color of multiselect options
st.markdown(
    """
<style>
span[data-baseweb="tag"] {
  background-color: #7f7f7f !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Load Data
path = 'csv_files/filtered_long_train_services.csv'
df = pd.read_csv(path)

# Convert datetime
if df['ARR_TIME'].notna().any():
    df['ARR_TIME'] = pd.to_datetime(df['ARR_TIME'], format='%d/%m/%Y %H:%M', errors='coerce')
if df['DEP_TIME'].notna().any():
    df['DEP_TIME'] = pd.to_datetime(df['DEP_TIME'], format='%d/%m/%Y %H:%M', errors='coerce')

# UI Filters
st.title("Train Time-Distance Diagram")
selected_corridor = st.selectbox("Select Corridor", df['CORRIDOR'].unique())

# Define y-axis order based on corridor
if selected_corridor == "East":
    stations_sorted = ['MFT', 'ETT', 'SFT', 'MOR', 'BFT']
    emphasis_labels = {'MFT': 20, 'SFT': 20, 'BFT': 20}
elif selected_corridor == "West":
    stations_sorted = ['MFT', 'SPG', 'TLB', 'APD', 'AFT', 'PAD', 'SCJ', 'COK', 'WEK', 'KWK', 'PFT']
    emphasis_labels = {'MFT': 20, 'AFT': 20, 'PFT': 20}
else:
    stations_sorted = sorted(df['TERMINAL'].unique())

# Filter by corridor before any terminal processing
corridor_df = df[df['CORRIDOR'] == selected_corridor].copy()

# Set TERMINAL as categorical with all expected categories
corridor_df['TERMINAL'] = pd.Categorical(corridor_df['TERMINAL'], categories=stations_sorted, ordered=True)

# Multiselect for TRAIN_ID
selected_trains = st.multiselect("Select Trains", corridor_df['TRAIN_ID'].dropna().unique(),
                                 default=corridor_df['TRAIN_ID'].dropna().unique())
df_filtered = corridor_df[corridor_df['TRAIN_ID'].isin(selected_trains)].copy()

# Assign colors to TRAIN_IDs
color_map = {train: color for train, color in zip(df['TRAIN_ID'].dropna().unique(), px.colors.qualitative.Alphabet)}

# Plot
fig = go.Figure()

for train in df_filtered['TRAIN_ID'].unique():
    train_data = df_filtered[df_filtered['TRAIN_ID'] == train].sort_values(by='LEG_NUM')
    color = color_map.get(train, 'gray')

    # Plot continuous lines and markers for each TRAIN_ID
    x_vals = []
    y_vals = []
    for i, row in train_data.iterrows():
        if pd.notnull(row['ARR_TIME']) and pd.notnull(row['DEP_TIME']):
            x_vals.extend([row['ARR_TIME'], row['DEP_TIME'], None])
            y_vals.extend([row['TERMINAL'], row['TERMINAL'], None])
        elif pd.notnull(row['DEP_TIME']):  # Show start terminal if only departure is available
            x_vals.extend([row['DEP_TIME'], row['DEP_TIME'], None])
            y_vals.extend([row['TERMINAL'], row['TERMINAL'], None])

    # Add scatter point at final terminal
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
        line=dict(color=color),
        marker=dict(color=color),
        name=train,
        legendgroup=train,
        showlegend=True
    ))

    # Travel time
    for i in range(len(train_data) - 1):
        prev_time = train_data.iloc[i]['DEP_TIME'] if pd.notnull(train_data.iloc[i]['DEP_TIME']) else \
            train_data.iloc[i]['ARR_TIME']
        next_time = train_data.iloc[i + 1]['ARR_TIME'] if pd.notnull(train_data.iloc[i + 1]['ARR_TIME']) else \
            train_data.iloc[i + 1]['DEP_TIME']
        y1 = train_data.iloc[i]['TERMINAL']
        y2 = train_data.iloc[i + 1]['TERMINAL']

        if pd.notnull(prev_time) and pd.notnull(next_time):
            fig.add_trace(go.Scatter(
                x=[prev_time, next_time],
                y=[y1, y2],
                mode='lines',
                line=dict(color=color),
                name=f"{train} travel",
                legendgroup=train,
                showlegend=False
            ))

# Add vertical dashed lines to separate days
all_times = pd.concat([df_filtered['ARR_TIME'], df_filtered['DEP_TIME']]).dropna()
if not all_times.empty:
    min_time = all_times.min().normalize()
    max_time = all_times.max()
    current_day = min_time
    while current_day <= max_time:
        fig.add_vline(
            x=current_day,
            line=dict(color='gray', dash='dash'),
            opacity=0.4
        )
        current_day += timedelta(days=1)

# Layout
fig.update_layout(
        title='Train Time-Distance Diagram',
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

