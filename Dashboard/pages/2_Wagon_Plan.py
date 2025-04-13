import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os


st.set_page_config(layout="wide")
st.title("Wagon Plan Editor")

# # Session state initialization
# if 'wagon_plan_df' not in st.session_state:
#     st.session_state.wagon_plan_df = pd.read_csv('csv_files/7.1_wagon_plan_2025.csv')
#
# # Upload new wagon plan file
# uploaded_file = st.file_uploader("Upload New Wagon Plan (CSV)", type="csv")
# if uploaded_file:
#     st.session_state.wagon_plan_df = pd.read_csv(uploaded_file)
#     st.success("New wagon plan uploaded and loaded for current session.")

uploaded_file = st.file_uploader("Upload New Wagon Plan (CSV)", type="csv")

if uploaded_file:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        required_cols = {'TRAIN_NAME', 'WAGON_CLASS', 'NUM_WAGONS'}
        if not required_cols.issubset(df_uploaded.columns):
            raise ValueError("Uploaded file is missing required columns.")
        st.session_state.wagon_plan_df = df_uploaded
        st.success("New wagon plan uploaded and loaded for current session.")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
elif 'wagon_plan_df' not in st.session_state:
    st.session_state.wagon_plan_df = pd.read_csv('csv_files/7.1_wagon_plan_2025.csv')


# Origin filter at the top
st.header("🚉 Filter by Origin")
origin_options = st.session_state.wagon_plan_df['ORIGIN'].unique()
selected_origin = st.selectbox("Select Origin:", options=origin_options)

# Filter based on selected origin
df = st.session_state.wagon_plan_df
filtered_df = df[df['ORIGIN'] == selected_origin]

# Editable Data Section
st.subheader("✏️ Editable Wagon Plan")
editable_df = filtered_df.copy()
editable_df['NUM_WAGONS'] = editable_df['NUM_WAGONS'].astype(int)
editable_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

# Store modified version in session state
st.session_state["modified_wagon_plan"] = editable_df.copy()
st.success("Modified wagon plan is now stored in memory and ready for assignment run.")

# Detailed Train-wise Display
st.subheader("🚂 Detailed Train-wise Wagon Plan")
with st.expander("View Detailed Wagon Plan", expanded=True):
    scroll_container = st.container()
    for train in editable_df['TRAIN_NAME'].unique():
        with scroll_container:
            train_data = editable_df[editable_df['TRAIN_NAME'] == train]
            primary_train = train[:4]  # e.g., 1431ENF -> 1431
            departing_location = train[-3:]  # e.g., ENF

            st.markdown(f"### Primary Train {primary_train}")
            st.markdown(f"**Train**: {train} &nbsp;&nbsp;&nbsp;&nbsp; **Departing Location**: {departing_location}")

            display_data = train_data[['SERVICE', 'WAGON_CLASS', 'NUM_WAGONS']]
            st.dataframe(display_data.reset_index(drop=True), use_container_width=True)

            total_wagons = display_data['NUM_WAGONS'].sum()
            st.markdown(f"**TOTAL**: {total_wagons:.2f}")
            st.markdown("---")

# Wagon Class Summary Table
st.subheader("🚩 Wagon Class Summary")
wagon_classes = df['WAGON_CLASS'].unique()
# selected_classes = st.multiselect("Select Wagon Class(es):", options=wagon_classes, default=wagon_classes)
# class_filtered_df = editable_df[editable_df['WAGON_CLASS'].isin(selected_classes)]

# summary = class_filtered_df.groupby(['WAGON_CLASS']).agg({'NUM_WAGONS': 'sum'}).reset_index()
summary = editable_df.groupby(['WAGON_CLASS']).agg({'NUM_WAGONS': 'sum'}).reset_index()
st.dataframe(summary, use_container_width=True)

# Optional CSV Export
st.subheader("📥 Download Modified Wagon Plan")
modified_csv = editable_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Modified Plan as CSV",
    data=modified_csv,
    file_name='modified_wagon_plan.csv',
    mime='text/csv'
)

# Export to PDF Function
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
        pdf.cell(60, 10, "No OF WAGONS", 1, ln=True)

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

# Generate and Download PDF
pdf = create_pdf(editable_df)
temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
pdf.output(temp_pdf.name)
with open(temp_pdf.name, "rb") as f:
    st.download_button(
        label="Download Wagon Plan as PDF",
        data=f,
        file_name="wagon_plan.pdf",
        mime="application/pdf"
    )

# Overwrite Button with confirmation dialog
st.subheader("⚠️ Overwrite Original Wagon Plan File")

confirm_overwrite = st.checkbox("I understand that this will permanently overwrite the saved wagon plan file.")

if confirm_overwrite:
    if st.button("Overwrite Saved Wagon Plan"):
        try:
            st.session_state["modified_wagon_plan"].to_csv('csv_files/7.1_wagon_plan_2025.csv', index=False)
            st.success("Wagon plan successfully overwritten.")
        except Exception as e:
            st.error(f"Error saving file: {e}")
else:
    st.info("Check the box above to confirm overwriting the original file.")

