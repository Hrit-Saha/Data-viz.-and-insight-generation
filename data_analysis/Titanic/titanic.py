"""
## Titanic Dataset

This file contains the code for performing Exploratory Data Analysis, Data Preprocessing, ML Modelling and Evaluation, and ANN Modelling on the Titanic Dataset.
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
    df = pd.read_csv("data_analysis\\Titanic\\Titanic.csv")
    return df

df = load_data() # Loaded the main dataset into the variable df
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_cols = ['Pclass', 'Sex', 'Embarked', 'Survived']
target = 'Survived'


# Exploratory Data Analysis and Data preprocessing
eda = EDA(data=df)
dp = DP(data=df)

# Showing Dataset Details
@st.cache_resource
def data_details() -> None:
    eda.df_details()

    # Data Cleaning
    dp.data_cleaning(
        impute_values={"Fare": df['Fare'].mean(), "Age": df['Age'].mean()},
        dropna_subset=['Embarked'],
        insignificant_columns=['PassengerId', 'Name', 'Ticket', 'Cabin']
    )
    eda.data = dp.data

    st.divider()

@st.cache_resource
def plotting() -> None:
    """
    Plots graphs for data analysis
    """

    eda.fig_count = 0
    eda.save_folder = f'data_analysis\\Titanic\\figs\\'

    eda.preliminary_plots(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        target='Survived',
        target_is_categorical=True,
        corr_kwargs = dict(annot=True, fmt='.2f', cmap='plasma'),
        pair_plot_kwargs = dict(palette={1: 'green', 0: 'red'}),
        categorical_plot_kwargs = dict(palette='plasma'),
        other_plot_kwargs = dict(palette={1: 'green', 0: 'red'})
    )

    st.markdown("#### `Other plots`")
    with st.expander("Plots"):
        
        eda.scatter_3d(x='Age', y='Fare', z='Pclass', color='Survived', color_discrete_map={1: 'green', 0: 'red'})

        eda.scatter_3d(x='Age', y='Fare', z='SibSp', color='Parch', symbol='Survived', color_continuous_scale='viridis')

        eda.parallel_categories(dimensions=categorical_cols, color='Survived', color_continuous_scale='viridis')

        eda.parallel_coordinates(dimensions=numerical_cols, color='Survived', color_continuous_scale='viridis')

@st.cache_resource
def data_preprocessing() -> None:
    """
    Data Preprocessing
    """

    save_folder = f'data_analysis\\Titanic\\figs\\'

    dp.independent_dependent_split(
        target_feature='Survived', 
        histplot_fig_folder=save_folder
    )
    dp.train_test_split()
    dp.data_transform(
        scale_columns={"quantile": ['Age', 'SibSp', 'Parch', 'Fare']}, 
        encode_columns={"label": ['Sex'], "one_hot": ['Embarked']}, 
    )


best_model = None

# ML Modelling and Evaluation
model_kwargs = {
    'log_reg': dict(random_state=42, n_jobs=-1),
    'dtc': dict(criterion='entropy', random_state=42),
    'rfc': dict(n_estimators=95, criterion='entropy', random_state=42, n_jobs=-1),
    'knc': dict(n_neighbors=5, n_jobs=-1),
    'svc': dict(kernel='rbf', random_state=42, max_iter=100),
    'gauss_nb': dict()
}

@st.cache_resource
def ml_modelling_evaluation() -> None:

    global best_model

    save_folder = f'data_analysis\\Titanic\\figs\\'
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
                
        if method == "knc":
            best_model = model
            st.success("Best Performance")
        
        st.markdown("# ") # For Spacing between sections

    ML.k_means_train_predict(
        # Using X_train, X_test combined as X, since X is not scaled or encode. Similarly, for y,
        data=pd.concat([pd.concat([dp.X_train, dp.X_test], axis=0), pd.concat([dp.y_train, dp.y_test], axis=0)], axis=1),
        n_clusters=2,
        x_axis=['Age', 'SibSp', 'Parch'],
        y_axis=['Fare', 'SibSp', 'Parch'],
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
        n_units=[16, 10, 8], n_epochs=100,
        metrics=['accuracy', 'recall'],
        y_encoder=dp.y_encoder,
        classification=True,
        fit_kwargs=dict(batch_size=32),
    )


# Insights
def insights() -> None:
    try:
        with open("data_analysis\\Titanic\\insights.md", 'r') as file:
            st.markdown(file.read())
    except FileNotFoundError:
        st.error("Insights not available")


# Prediction
def prediction() -> None:
    dp.prediction(numerical_cols, categorical_cols, target, best_model)

