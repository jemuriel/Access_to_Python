import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Writing to the screen -------------------------------------------------------------------------

st.write('Hello world')
st.write({"key":"value"})
3+4
"hello world" if False else "bye"
# on terminal streamlit run .\main.py
# Ctr + C to close the application

# when something changes on the code, streamlit reruns the enture file

# Create a button - behaviour of each button changes with each press ---------------------------
print('Run')
pressed = st.button("First button")
print('First:', pressed)

pressed2 = st.button("Second button")
print('Second', pressed2)

# Text elements ---------------------------------------------------------------------------------
st.title("Super simple tittle")
st.header("this is a header")
st. subheader('Subheader')
st.markdown("This is _Markdown_")
st.caption("small text")

code_example = """
def green(name):
    print('hello', name)

"""
st.code(code_example, language='python')
st.divider()

# Images ----------------------------------------------------------------------------------------
st.image(os.path.join(os.getcwd(), "static", "bluff_knull.jpg")) #option width=95

# Pandas df -------------------------------------------------------------------------------------
df = pd.DataFrame({
    'Name' : ['Alice', 'Bob', 'Charles', 'David'],
    'Age' : [25, 32, 37, 45],
    'Occupation': ['Engineer', 'Doctor', 'Artist', 'Chef']
})

st.dataframe(df)

# Data editor section ######
st.subheader('Data Editor')
editable_df = st.data_editor(df) # Changes editable_df when edited
print(editable_df)

# Static table section #######
st.subheader('Static Table')
st.table(df)

# Metrics section
st.subheader("Metrics")
st.metric(label="Total Rows", value=len(df))
st.metric(label='Average Age', value=round(df['Age'].mean(),1))

# JSON and Dict section ------------------------------------------------------------------------------
st.subheader("JSON and Dictionary")
sample_dict = {
    "name": "Alice",
    'age': 25,
    'skills': ['Python', 'Data Science', 'Machine learning']
}
st.json(sample_dict)

# Also show as dictionary
st.write('Dinctionary view:', sample_dict)


# Charts elements ------------------------------------------------------------------------------------
chart_data = pd.DataFrame(
    np.random.randn(20,3),
    columns=['A','B', 'C']
)

# Area chart subsection
st.subheader('Area chart')
st.area_chart(chart_data)

# Bar Chart section
st.subheader('Bar Chart')
st.bar_chart(chart_data)

# Line chart section
st.subheader('Line Chart')
st.line_chart(chart_data)

# Scatter chart section
st.subheader('Scatter Chart')
scatter_data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})
st.scatter_chart(scatter_data)

# Map section displaying random points in a map --------------------------------------------------------
st.subheader('Map')
map_data = pd.DataFrame(
    np.random.randn(100,2)/[50,50]+[37.76, -122.4],
    columns= ['lat','lon']
)
st.map(map_data)

# Pyplot section ---------------------------------------------------------------------------------------
st.subheader('Pyplot Chart')
fig, ax = plt.subplots()
ax.plot(chart_data['A'], label="A")
ax.plot(chart_data['B'], label="A")
ax.plot(chart_data['C'], label="A")
ax.set_title("Pyplot Line Chart")
ax.legend()
st.pyplot(fig)

# Forms -----------------------------------------------------------------------------------------------
# Form creation avoids the script to tbe rerun constantly
# its only run here when the submit button is pressed
with st.form(key='sample_form'):

    # Text input
    st.subheader("Text Inputs")
    name = st.text_input('Enter your name')
    feedback = st.text_area("Provide your feedback")
    # st.number_input

    # Date and time
    st.subheader("Data and time inputs")
    dob = st.date_input("Select your date of birth")
    time = st.time_input("Choose a preferred time")

    # Selectors
    st.subheader("Selectors")
    choice = st.radio("Choose an option", ['Option 1', 'Option 2', 'Option 3'])
    gender = st.selectbox('Select your gender', ['Male', 'Female'])
    slider_value = st.select_slider('Select a range', options = [1, 2, 3, 4, 5])

    # Toggles and checkbox
    st.subheader('Toggles & checkboxes')
    notifications = st.checkbox('Receive notifications?')
    toggle_value = st.checkbox('Enable dark mode?', value=False)

    # Submit button for the form #######################
    submit_button = st.form_submit_button(label='Submit')

# Store input values in a dictionary
st.title('User information form')
form_values = {
    'name' : None,
    'height': None,
    'gender': None,
    'dob': None
}

min_date = datetime(1990,1 ,1)
max_date = datetime.now()

with st.form(key='user_info_form', clear_on_submit=True):
    form_values['name'] = st.text_input('Enter your name')
    form_values['height'] = st.number_input('Enter your height')
    form_values['gender'] = st.selectbox('Gender:', ['Male', 'Female'])
    form_values['dob'] = st.date_input('Enter your birthdate', max_value=max_date, min_value=min_date)

    if form_values['dob']:
        birth_date = form_values['dob']
        age = max_date.year - birth_date.year
        st.write(f'Your calculated age is {age} years')

    submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        if not all(form_values.values()):
            st.warning('Please fill all the fields')
        else:
            st.balloons()
            st.write('## Info')
            for key,value in form_values.items():
                st.write(f'{key}:{value}')