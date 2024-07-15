"""
## Calories Dataset

This file contains the code for performing Exploratory Data Analysis, Data Preprocessing, ML Modelling and Evaluation, and ANN Modelling on the Calories dataset.
It containsall the required functions for the following steps:
1. Loading the dataset
2. Exploratory Data Analysis
3. Data Preprocessing
4. ML Modelling and Evaluation
5. ANN Modelling and Evaluation
"""

# All Imports
# Import Data processing and visualization libraries
import pandas as pd
import plotly.graph_objects as go

# Import streamlit API
import streamlit as st

# Dependencies
from data_analysis.data_analysis_utils import exploratory_data_analysis as EDA
from data_analysis.data_analysis_utils import data_preprocessing as DP
from data_analysis.data_analysis_utils import ml_model as ML
from data_analysis.data_analysis_utils import ANN


# Loading the dataset
@st.cache_resource
def load_data() -> pd.DataFrame:
    df1 = pd.read_csv("data_analysis\\Calories\\exercise.csv")
    df2 = pd.read_csv("data_analysis\\Calories\\calories.csv")
    df = pd.merge(df1, df2, on="User_ID")
    return df

df = load_data() # Loaded the main dataset into the variable df
numerical_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
categorical_cols = ["Gender"]
target = "Calories"


# Exploratory Data Analysis and Data Preprocessing
eda = EDA(data=df)
dp = DP(data=df)

# Showing Dataset Details
@st.cache_resource
def data_details() -> None:
    eda.df_details()

    # Data Cleaning
    dp.data_cleaning(insignificant_columns=['User_ID'])
    eda.data = dp.data

    st.divider()

@st.cache_resource
def plotting() -> None:
    """
    Plots graphs for data analysis
    """

    eda.fig_count = 0
    eda.save_folder = f'data_analysis\\Calories\\figs\\'

    eda.preliminary_plots(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        target='Calories',
        target_is_categorical=False,
        corr_kwargs = dict(annot=True, fmt='.2f', cmap='plasma'),
        pair_plot_kwargs = dict(hue='Gender', diag_kind='kde', palette='plasma'),
        categorical_plot_kwargs = dict(palette='plasma'),
        other_plot_kwargs = dict(hue='Gender', palette='plasma')
    )

    st.markdown("#### `Other plots`")
    with st.expander("Plots"):

        eda.scatter_3d(x='Age', y='Height', z='Weight', color='Gender', color_discrete_map={'female':'orange', 'male':'deepskyblue'})

@st.cache_resource
def data_preprocessing() -> None:
    """
    Data preprocessing
    """

    save_folder = f'data_analysis\\Calories\\figs\\'

    dp.independent_dependent_split(target_feature='Calories', histplot_fig_folder=save_folder)
    dp.train_test_split()

    gaussian_cols = ["Height", "Weight", "Heart_Rate"]

    dp.data_transform(
        scale_columns={
            "std": gaussian_cols,
            "min_max": [col for col in numerical_cols if col not in gaussian_cols if col != 'Calories']
        },
        encode_columns={"label": ['Gender']},
        target_scaler='min_max'
    )


best_model = None

# ML Modelling and Evaluation
model_kwargs = {
    'lin_reg': dict(n_jobs=-1),
    'dtr': dict(criterion='squared_error', random_state=42),
    'rfr': dict(n_estimators=20, criterion='squared_error', random_state=42, n_jobs=-1),
    'svr': dict(kernel='rbf', max_iter=120)
}

@st.cache_resource
def ml_modelling_evaluation() -> None:

    global best_model

    save_folder = f'data_analysis\\Calories\\figs\\'
    fig_count = 0

    for method in model_kwargs.keys():

        fig_count += 1
        model = ML.create_train_predict_eval(
            method=method,
            X_train=dp.X_train.to_numpy(),
            y_train=dp.y_train.to_numpy().ravel(), 
            X_test=dp.X_test.to_numpy(), 
            y_test=dp.y_test.to_numpy().ravel(), 
            y_scaler=dp.y_scaler,
            classification=False,
            metrics_fig_path=f"{save_folder}ml_metrics_{fig_count}.png",
            **model_kwargs[method]
        )

        st.markdown("# ") # For Spacing between sections

    ML.k_means_train_predict(
        # Using X_train, X_test combined as X, since X is not scaled or encode. Similarly, for y,
        data=pd.concat([pd.concat([dp.X_train, dp.X_test], axis=0), pd.concat([dp.y_train, dp.y_test], axis=0)], axis=1),
        n_clusters=4,
        x_axis=['Calories'],
        y_axis=['Weight', 'Height', 'Age'],
        save_folder=save_folder,
    )


# ANN Modelling
@st.cache_resource
def ann_modelling_evaluation() -> None:

    global best_model

    best_model = ANN.create_train_predict_eval(
        X_train=dp.X_train.to_numpy(), 
        y_train=dp.y_train.to_numpy().ravel(),
        X_test=dp.X_test.to_numpy(), 
        y_test=dp.y_test.to_numpy().ravel(), 
        n_layers=3, n_units=[16, 10, 8], n_epochs=50,
        metrics=['mse', 'r2_score'],
        y_scaler=dp.y_scaler,
        classification=False,
        fit_kwargs=dict(batch_size=32)
    )

    st.success("Best Performance")

# Insights
def insights() -> None:
    try:
        with open("data_analysis\\Calories\\insights.md", 'r') as file:
            st.markdown(file.read())
    except FileNotFoundError:
        st.error("Insights not available")


# Prediction
def prediction() -> None:
    dp.prediction(numerical_cols, categorical_cols, target, best_model)

