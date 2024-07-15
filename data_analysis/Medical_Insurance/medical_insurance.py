"""
## Medical Insurance Cost Dataset

This file contains the code for performing Exploratory Data Analysis, Data Preprocessing, ML Modelling and Evaluation, and ANN Modelling on the Medical Insurance Cost Dataset.
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
    df = pd.read_csv("data_analysis\\Medical_Insurance\\Medical_Insurance.csv")
    return df

df = load_data() # Loaded the main dataset into the variable df
numerical_cols = [col for col in df.columns if col not in ['sex', 'smoker', 'region']]
categorical_cols = ['sex', 'region', 'smoker']
target = 'charges'


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
    eda.save_folder = f'data_analysis\\Medical_Insurance\\figs\\'

    eda.preliminary_plots(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        target='charges',
        target_is_categorical=False,
        corr_kwargs = dict(annot=True, fmt='.2f', cmap='plasma'),
        pair_plot_kwargs = dict(hue='smoker', palette={'no': 'green', 'yes': 'red'}),
        other_plot_kwargs = dict(palette='plasma')
    )

    st.markdown("#### `Other plots`")
    with st.expander("Plots"):
        
        eda.scatter_3d(x='age', y='bmi', z='charges', color='smoker', symbol='sex', color_discrete_map={"no": "green", "yes": "red"})

        eda.parallel_categories(dimensions=categorical_cols, color='smoker', color_continuous_scale='viridis')

        eda.parallel_categories(dimensions=categorical_cols, color='sex', color_continuous_scale='viridis')

        eda.parallel_coordinates(dimensions=numerical_cols, color='smoker', color_continuous_scale='viridis')

@st.cache_resource
def data_preprocessing() -> None:
    """
    Data preprocessing
    """

    save_folder = f'data_analysis\\Medical_Insurance\\figs\\'

    dp.independent_dependent_split(target_feature='charges', histplot_fig_folder=save_folder)
    dp.train_test_split()

    dp.data_transform(
        scale_columns={"std": ['bmi'], "min_max": ['age', 'children']},
        encode_columns={"label": ['sex', 'smoker'], "one_hot": ['region']},
        target_scaler='quantile'
    )


best_model = None

# ML Modelling and Evaluation
model_kwargs = {
    'lin_reg': dict(n_jobs=-1),
    'dtr': dict(criterion='squared_error', random_state=42),
    'rfr': dict(n_estimators=100, criterion='squared_error', random_state=42, n_jobs=-1),
    'svr': dict(kernel='rbf', max_iter=100)
}

@st.cache_resource
def ml_modelling_evaluation() -> None:

    global best_model

    save_folder = f'data_analysis\\Medical_Insurance\\figs\\'
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
                
        if method == "rfr":
            best_model = model
            st.success("Best Performance")
        
        st.markdown("# ") # For Spacing between sections

    ML.k_means_train_predict(
        # Using X_train, X_test combined as X, since X is not scaled or encode. Similarly, for y,
        data=pd.concat([pd.concat([dp.X_train, dp.X_test], axis=0), pd.concat([dp.y_train, dp.y_test], axis=0)], axis=1),
        n_clusters=2,
        x_axis=['charges','bmi', 'age', 'children'],
        y_axis=['bmi', 'age', 'children'],
        save_folder=save_folder,
    )


# ANN Modelling
@st.cache_resource
def ann_modelling_evaluation() -> None:

    ANN.create_train_predict_eval(
        X_train=dp.X_train.to_numpy(), 
        y_train=dp.y_train.to_numpy().ravel(),
        X_test=dp.X_test.to_numpy(), 
        y_test=dp.y_test.to_numpy().ravel(), 
        n_layers=3, n_units=[16, 10, 8], n_epochs=80,
        metrics=['mse', 'r2_score'],
        y_scaler=dp.y_scaler,
        classification=False,
        fit_kwargs=dict(batch_size=32)
    )


# Insights
def insights() -> None:
    try:
        with open("data_analysis\\Medical_Insurance\\insights.md", 'r') as file:
            st.markdown(file.read())
    except FileNotFoundError:
        st.error("Insights not available")


# Prediction
def prediction() -> None:
    dp.prediction(numerical_cols, categorical_cols, target, best_model)

