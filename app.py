# Last Updated on 10-07-2024

"""
This file contains the main Streamlit app.
It defines the sidebar and the main content of the app.
The app is divided into two columns:
1. Sidebar: Contains the navigation buttons for the app.
2. Main Content: Displays the content of the selected page.
    - Side Column: Displays the logo and the navigation buttons.
        - Logo: Displays the logo of the app.
        - Theme Toggle: Allows the user to toggle the theme of the app from light to dark or vice versa.
        - Contents: Displays the contents of the app.
    - Main Column: Displays the content of the selected page.
"""

# Import Streamlit for creating web-based app interface
import streamlit as st
st.set_page_config(
    page_title="Data Visualization & Insights",
    page_icon="custom_css\\data-analytics_12489648.png", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)
st.logo("custom_css\\data-analytics_12489648.png")

# Import all custom modules
import global_streamlit as gs
import app_pages as pg

if gs.selected_page is None:
    gs.selected_page = pg.initial_page

# Main sidebar container
with st.sidebar:
    st.sidebar.markdown("""# Data Visualization & Insights\n#""")
    if st.sidebar.button("Introduction"):
        gs.selected_page = pg.initial_page

    if st.sidebar.button("Project Overview"):
        gs.selected_page = pg.project_overview_page

    st.sidebar.divider()
    st.sidebar.markdown("### Data Sets")
    if st.sidebar.button("Overview of Datasets"):
        gs.selected_page = pg.dataset_overview_page

    if st.sidebar.button("EDA & ML Modelling"):
        gs.selected_page = pg.eda_ml_modelling_page()


# Two column layout - One column for page contents and another one for main page
col1, col2 = st.columns([0.25, 0.75], gap='medium')

# Load the page
# Load the page contents list
with col1:
    # Logo
    st.image("custom_css\\data-analytics_12489648.png")

    # Theme toggle container
    toggle_container = st.container()
    
    # Page contents list
    if not isinstance(gs.selected_page, pg.eda_ml_modelling_page):
        gs.selected_page.page_content()
    else:
        # If EDA Page is shown then show a selectbox to select the listed datasets
        gs.eda_ml_modelling_option = st.selectbox(
            label="Select your Dataset",
            options=gs.module_dict.keys(),
            label_visibility='hidden',
            index=None,
            placeholder='Select your Dataset'
        )
        gs.selected_page.page_content()

    
    toggle = toggle_container.toggle(":first_quarter_moon:", key='Theme', value=True)
    if toggle:
        with open("custom_css\\dark_theme.css", "r") as f:
            custom_css = f.read()
        st.html("<style>%s</style>" % custom_css)
    else:
        with open("custom_css\\light_theme.css", "r") as f:
            custom_css = f.read()
        st.html("<style>%s</style>" % custom_css)

# Load the main page
with col2:
    gs.selected_page.main_page()

