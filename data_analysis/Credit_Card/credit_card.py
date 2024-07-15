"""
## Credit Card Fraud Detection Dataset

This file contains the code for performing Exploratory Data Analysis, Data Preprocessing, ML Modelling and Evaluation, and ANN Modelling on the Credit Card Fraud Detection Dataset.
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
    df = pd.read_csv("data_analysis\\Credit_Card\\Credit_Card.csv")
    return df

df = load_data() # Loaded the main dataset into the variable df
numerical_cols = [col for col in df.columns if col != 'Class']
categorical_cols = ['Class']
target = 'Class'


# Exploratory data analysis and Data Preprocessing
eda = EDA(data=df)
dp = DP(data=df)

# Showing Dataset Details
@st.cache_resource
def data_details() -> None:
    eda.df_details()
    st.divider()

@st.cache_resource
def plotting() -> None:
    
    eda.fig_count = 0
    eda.save_folder = f'data_analysis\\Credit_Card\\figs\\'

    eda.preliminary_plots(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        target='Class',
        target_is_categorical=True,
        corr_kwargs = dict(annot=False, cmap='plasma'),
        pair_plot_exclude_cols=[col for col in numerical_cols if col not in ['Time', 'Amount']],
        dist_plot_kwargs=dict(bins=40),
        categorical_plot_kwargs = dict(palette={0: 'green', 1: 'red'}),
        other_plot_kwargs = dict(palette={0: 'green', 1: 'red'})
    )

    st.markdown("#### `Other plots`")
    with st.expander("Plots"):
        
        eda.scatter_3d(x='Time', y='Amount', z='Class', color='Class', color_discrete_map={0: 'green', 1: 'red'})

@st.cache_resource
def data_preprocessing() -> None:
    """
    Data preprocessing
    """

    save_folder = f'data_analysis\\Credit_Card\\figs\\'

    dp.independent_dependent_split(target_feature='Class', histplot_fig_folder=save_folder)
    dp.train_test_split()

    dp.data_transform(scale_columns={"quantile": numerical_cols})


best_model = None

# ML Modelling and Evaluation
model_kwargs = {
    'log_reg': dict(random_state=42, n_jobs=-1),
    'dtc': dict(criterion='entropy', random_state=42),
    'rfc': dict(n_estimators=40, criterion='entropy', random_state=42, n_jobs=-1),
    'knc': dict(n_neighbors=5, n_jobs=-1),
    'svc': dict(kernel='rbf', random_state=42, max_iter=100),
    'gauss_nb': dict()
}

@st.cache_resource
def ml_modelling_evaluation() -> None:

    global best_model
    
    save_folder = f'data_analysis\\Credit_Card\\figs\\'
    fig_count = 0
    
    for method in model_kwargs.keys():

        fig_count += 1
        model = ML.create_train_predict_eval(
            method=method,
            X_train=dp.X_train.to_numpy(),
            y_train=dp.y_train.to_numpy().ravel(),
            X_test=dp.X_test.to_numpy(),
            y_test=dp.y_test.to_numpy().ravel(),
            classification=True,
            metrics_fig_path=f"{save_folder}ml_metrics_{fig_count}.png",
            **model_kwargs[method]
        )
        
        if method == "gauss_nb":
            best_model = model
            st.success("Best Performance")

        st.markdown("# ") # For Spacing between sections


# ANN Modelling
@st.cache_resource
def ann_modelling_evaluation() -> None:

    ANN.create_train_predict_eval(
        X_train=dp.X_train.to_numpy(),
        y_train=dp.y_train.to_numpy().ravel(),
        X_test=dp.X_test.to_numpy(),
        y_test=dp.y_test.to_numpy().ravel(),
        n_layers=3, n_classes=2,
        n_units=[16, 10, 8], n_epochs=42,
        metrics=['accuracy', 'recall'],
        y_encoder=dp.y_encoder,
        classification=True,
        fit_kwargs=dict(batch_size=800),
    )


# Insights
def insights() -> None:
    try:
        with open("data_analysis\\Credit_Card\\insights.md", 'r') as file:
            st.markdown(file.read())
    except FileNotFoundError:
        st.error("Insights not available")


# Prediction
def prediction() -> None:
    dp.prediction(numerical_cols, categorical_cols, target, best_model)

