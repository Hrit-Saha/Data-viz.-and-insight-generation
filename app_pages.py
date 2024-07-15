# Last Updated on 14-07-2024

"""
This file contains all the pages of the Streamlit App.
The pages are defined as classes and each class has two methods:
1. `page_content()`: This method displays the content of the page in the side column.
2. `main_page()`: This method displays the main content of the page in the main column.

### The pages defined in this file:
1. initial_page
2. project_overview_page
3. dataset_overview_page
4. eda_ml_modelling_page
"""

# All imports
# Import Streamlit for creating web-based app interface
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

# Import utilities
import global_streamlit as gs

# Inital page
class initial_page:
    def __init__(self) -> None:
        return
    
    def page_content() -> None:
        st.markdown("""
            ## How to Use:
            * ##### Use the above toggle to switch between light and dark theme.
            * ##### All the in-page content cross-references can be found in this section.
            * ##### Access the sidebar on the top-right corner to navigate through the app.
        """)
    
    def main_page() -> None:
        '''Displays the very first page of the app when all no app-page has been selected from the Sidebar.'''
        try:
            with open("document_files\\initial.md", 'r') as f:
                markdown_text = f.read()
            with stylable_container(
                key='initial',
                css_styles=[
                    "blockquote {border :1px solid #888; border-radius:20px; text-align:center; padding: 0px 10px; transition: transform 0.2s;}",
                    "blockquote strong {font-size:3.2rem;}",
                    "blockquote:hover {transform :scale(1.02);}"
                ]
            ):
                st.markdown(markdown_text)
        except FileNotFoundError:
            st.error(f"Markdown file not found: {"document_files\\initial.md"}")

# Project Overview page
class project_overview_page:
    def __init__(self) -> None:
        return None
    
    def page_content() -> None:
        pass

    def main_page() -> None:
        '''Displays the 'Project Overview' Page.
        The core details of the Project has been presented in this page with abstraction.'''
        try:
            with open("document_files\\project_overview.md", 'r') as f:
                markdown_text = f.read()
            st.write(markdown_text)
        except FileNotFoundError:
            st.error(f"Markdown file not found: {"document_files\\project_overview.md"}")

# Dataset Overview page
class dataset_overview_page:
    def __init__(self) -> None:
        return None
    
    def page_content() -> None:
        st.markdown("""
            #### Content: 
            1. [Social Network Ads](#1-social-network-ads)
            2. [E-Commerce Shipment Data](#2-e-commerce-shipment-data)
            3. [Bank Marketing Dataset](#3-bank-marketing-dataset)    
        """)

    def main_page() -> None:
        '''Displays the details of all the datsets which are used in this Project.'''
        try:
            with open("document_files\\overview_of_datasets.md", 'r') as f:
                markdown_text = f.read()
            st.write(markdown_text)
        except FileNotFoundError:
            st.error(f"Markdown file not found: {"document_files\\overview_of_datasets.md"}")

# Exploratory Data Analysis page
class eda_ml_modelling_page:

    def __init__(self) -> None:
        return

    target = ""
    model_names = {
        'log_reg': "Logistic Regression",
        'lin_reg': "Linear Regression",
        'dtc': "Decision Tree Classifier",
        'dtr': "Decision Tree Regressor",
        'rfc': "Random Forest Classifier",
        'rfr': "Random Forest Regressor",
        'svc': "Support Vector Classifier",
        'svr': "Support Vector Regressor",
        'knc': "K-Nearest Neighbors Classifier",
        'gauss_nb': "Gaussian Naive Bayes",
    }

    def page_content(self) -> None:

        if gs.eda_ml_modelling_option is not None:
            self.target = gs.module_dict[gs.eda_ml_modelling_option].target

            st.divider()

            # All the page contents for navigation
            st.markdown("#### Content: ")
            with st.expander("Exploratory Data Analysis"):
                st.markdown(f"""
                    - [The Dataset](#the-dataset)
                    - [Small deatails about the Dataset](#small-deatails-about-the-dataset)
                    - [Correlation](#correlation-heatmap)
                    - [Pair Plot](#pair-plot)
                    - [Distribution Plots](#distribution-plots)
                    - [Categorical Plots](#categorical-plots)
                    - [{self.target.capitalize()} vs Other Features](#{self.target}-vs-other-features)
                    - [Other Plots](#other-plots)
                """)
            with st.expander("Data Preprocessing"):
                st.markdown(f"""
                    - [Independent & Dependent Features](#independent-dependent-features)
                    - [Train-Test Split](#train-test-split)
                    - [Train-Test Split (Scaled & Encoded)](#train-test-split-scaled-encoded)
                """)
            with st.expander("ML Modelling"):
                model_names = ["- " + f'[{self.model_names[method]}]' + f'(#{self.model_names[method].lower().replace(" ", "-")})' + '\n' for method in gs.module_dict[gs.eda_ml_modelling_option].model_kwargs.keys()]
                st.markdown(f"""{"".join(model_names)}""")
            with st.expander("ANN Modelling"):
                st.markdown(f"""
                    - [Artificial Neural Network](#artificial-neural-network)
                    - [Training the ANN Model](#training-the-ann-model)
                    - [Evaluation Metrics](#evaluation-metrics)
                """)

    def main_page(self) -> None:
        
        st.markdown("# EDA and ML Modelling")
        Exploratory_Data_Analysis, Insights_Patterns, Data_Preprocessing, ML_Modelling, ANN_Modelling, Predict = st.tabs(["Exploratory Data Analysis", "Insights & Patterns", "Data Preprocessing", "ML Modelling", "ANN Modelling", "Predict"])

        if gs.eda_ml_modelling_option is not None:

            # Exploratory Data Analysis
            with Exploratory_Data_Analysis:
                st.markdown("## Exploratory Data Analysis (EDA)")
                st.divider()
                
                gs.module_dict[gs.eda_ml_modelling_option].data_details()
                gs.module_dict[gs.eda_ml_modelling_option].plotting()

            # Insights & Patterns
            with Insights_Patterns:
                st.markdown("## Insights & Patterns")
                st.divider()

                gs.module_dict[gs.eda_ml_modelling_option].insights()

            # Data preprocessing
            with Data_Preprocessing:
                st.markdown("## Data Preprocessing")
                st.divider()

                gs.module_dict[gs.eda_ml_modelling_option].data_preprocessing()

            # ML Modelling & Evaluation
            with ML_Modelling:
                st.markdown("## Machine Learning Modelling")
                st.divider()

                gs.module_dict[gs.eda_ml_modelling_option].ml_modelling_evaluation()

            # ANN Modelling & Evaluation
            with ANN_Modelling:
                st.markdown("## Artificial Neural Network Modelling")
                st.divider()

                gs.module_dict[gs.eda_ml_modelling_option].ann_modelling_evaluation()

            # Interactive Prediction
            with Predict:
                st.markdown("## Interactive Prediction")
                st.divider()

                gs.module_dict[gs.eda_ml_modelling_option].prediction()

        else:
            st.error("Please select a dataset from the sidebar")