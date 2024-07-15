"""
## Social Network Ads Dataset

This file contains the code for performing Exploratory Data Analysis, Data Preprocessing, ML Modelling and Evaluation, and ANN Modelling on the Social Network Ads Dataset.
It contains all the required functions for the following steps:
1. Loading the dataset
2. Exploratory Data Analysis
3. Data Preprocessing
4. ML Modelling and Evaluation
5. ANN Modelling and Evaluation
"""

# All Imports
# Import Data processing and visualization libraries
import pandas as pd

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
    df = pd.read_csv("data_analysis\\Social_Network_Ads\\Social_network_ads.csv")
    return df

df = load_data() # Loaded the main dataset into the variable df
numerical_cols = ['Age','EstimatedSalary']
categorical_cols = ['Gender', 'Purchased']
target = 'Purchased'


# Exploratory Data Analysis and Data Preprocessing
eda = EDA(data=df)
dp = DP(data=df)

# Showing Dataset Details
@st.cache_resource
def data_details() -> None:
    eda.df_details()

    # Data cleaning
    dp.data_cleaning(insignificant_columns=['User ID'])
    eda.data = dp.data

    st.divider()

@st.cache_resource
def plotting() -> None:
    """
    Plots graphs for data analysis
    """

    eda.fig_count = 0
    eda.save_folder = f'data_analysis\\Social_Network_Ads\\figs\\'

    eda.preliminary_plots(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        target='Purchased',
        target_is_categorical=True,
        corr_kwargs = dict(annot=True, fmt='.2f', cmap='plasma'),
        pair_plot_kwargs = dict(palette={1: 'green', 0: 'red'}),
        categorical_plot_kwargs = dict(palette='plasma'),
        other_plot_kwargs = dict(palette={1: 'green', 0: 'red'})
    )

    st.markdown("#### `Other plots`")
    with st.expander("Plots"):
        
        eda.scatter_3d(x='Age', y='EstimatedSalary', z='Purchased', color='Gender', symbol='Purchased', color_discrete_map={1: 'green', 0: 'red'})

        eda.parallel_coordinates(dimensions=numerical_cols+['Purchased'], color='Purchased', color_continuous_scale='viridis')

        eda.plot(plot_kind='jointplot', markdown_title='Estimated Salary vs Age', x="EstimatedSalary", y="Age", hue='Purchased', palette={1: 'green', 0: 'red'})

@st.cache_resource
def data_preprocessing() -> None:
    """
    Data Preprocessing
    """

    save_folder = f'data_analysis\\Social_Network_Ads\\figs\\'

    dp.independent_dependent_split(target_feature='Purchased', histplot_fig_folder=save_folder)
    dp.train_test_split(test_size=0.25)
    dp.data_transform(scale_columns={"std": ['Age', 'EstimatedSalary']}, encode_columns={"label": ['Gender']})


best_model = None

# ML Modelling and Evaluation
model_kwargs = {
    'log_reg': dict(random_state=42, n_jobs=-1),
    'dtc': dict(criterion='entropy', random_state=42),
    'rfc': dict(n_estimators=80, criterion='entropy', random_state=42, n_jobs=-1),
    'knc': dict(n_neighbors=5, n_jobs=-1),
    'svc': dict(kernel='rbf', random_state=42, max_iter=100),
    'gauss_nb': dict()
}

@st.cache_resource
def ml_modelling_evaluation() -> None:

    global best_model

    save_folder = f'data_analysis\\Social_Network_Ads\\figs\\'
    fig_count = 0
    
    for method in model_kwargs.keys():
        
        fig_count += 1
        model = ML.create_train_predict_eval(
            method=method,
            X_train=dp.X_train.to_numpy(),
            y_train=dp.y_train.to_numpy().ravel(),
            X_test=dp.X_test.to_numpy(),
            y_test=dp.y_test.to_numpy().ravel(),
            metrics_fig_path=f"{save_folder}ml_metrics_{fig_count}.png",
            **model_kwargs[method]
        )
        
        if method == "gauss_nb":
            best_model = model
            st.success("Best Performance")

        st.markdown("# ") # For Spacing between sections

    ML.k_means_train_predict(
        # Using X_train, X_test combined as X, since X is not scaled or encode. Similarly, for y,
        data=pd.concat([pd.concat([dp.X_train, dp.X_test], axis=0), pd.concat([dp.y_train, dp.y_test], axis=0)], axis=1),
        n_clusters=4,
        x_axis=['EstimatedSalary'],
        y_axis=['Age'],
        save_folder=save_folder
    )


# ANN Modelling
@st.cache_resource
def ann_modelling_evaluation() -> None:

    ANN.create_train_predict_eval(
        X_train=dp.X_train.to_numpy(),
        y_train=dp.y_train.to_numpy().ravel(),
        X_test=dp.X_test.to_numpy(),
        y_test=dp.y_test.to_numpy().ravel(),
        n_layers=3, n_classes=2,
        n_units=[16, 10, 8], n_epochs=80,
        metrics=['accuracy', 'recall'],
        classification=True,
        fit_kwargs=dict(batch_size=25),
    )


# Insights
def insights() -> None:
    try:
        with open("data_analysis\\Social_Network_Ads\\insights.md", 'r') as file:
            st.markdown(file.read())
    except FileNotFoundError:
        st.error("Insights not available")


# Prediction
def prediction() -> None:
    dp.prediction(numerical_cols, categorical_cols, target, best_model)

