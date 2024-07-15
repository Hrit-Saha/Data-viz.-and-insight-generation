"""
## Bank Marketing Dataset

This file contains the code for performing Exploratory Data Analysis, Data Preprocessing, ML Modelling and Evaluation, and ANN Modelling on the Bank Marketing dataset.
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
    df = pd.read_csv("data_analysis\\Bank\\Bank.csv")
    return df

df = load_data() # Loaded the main dataset into the variable df
numerical_cols = ['age', 'balance', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous']
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'deposit']
target = 'deposit'

# Converting month from string to int datatype since month is a numerical distribution
month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
df['month'] = df['month'].map(month_map)


# Exploratory Data Analysis and Data Preprocessing
eda = EDA(data=df)
dp = DP(data=df)

# Showing Dataset Details
@st.cache_resource
def data_details() -> None:
    eda.df_details()
    st.divider()

@st.cache_resource
def plotting() -> None:
    """
    Plots graphs for data analysis
    """

    eda.fig_count = 0
    eda.save_folder = f'data_analysis\\Bank\\figs\\'

    eda.preliminary_plots(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        target='deposit',
        target_is_categorical=True,
        corr_kwargs = dict(annot=True, fmt='.2f', cmap='plasma'),
        pair_plot_kwargs = dict(palette={'yes': 'green', 'no': 'red'}),
        dist_plot_kwargs=dict(bins=40),
        categorical_plot_kwargs = dict(palette='plasma'),
        other_plot_kwargs = dict(palette={'yes': 'green', 'no': 'red'})
    )

    st.markdown("#### `Other plots`")
    with st.expander("Plots"):

        eda.plot(plot_kind='jointplot', markdown_title='Age vs Balance', x="age", y="balance", hue='deposit', palette={'yes': 'green', 'no': 'red'}, 
                 s=16, edgecolor="black", linewidth=0.2, legend=True, legend_properties=dict(framealpha=0.2))

        eda.plot(plot_kind='kdeplot', markdown_title='Age distribution based on Marital status', x='age', hue='marital', palette='plasma', xlabel="Age", ylabel="Density", fill=True)
        
        eda.scatter_3d(x='age', y='balance', z='duration', color='deposit', color_discrete_map={'yes': 'green', 'no': 'red'})

@st.cache_resource
def data_preprocessing() -> None:
    """
    Data Preprocessing
    """

    save_folder = f'data_analysis\\Bank\\figs\\'

    dp.independent_dependent_split(target_feature='deposit', histplot_fig_folder=save_folder)
    dp.train_test_split()
    
    # Filter columns with only 2 unique values
    cols_with_two_unique =  [col for col in categorical_cols if df[col].nunique() == 2]
    cols_with_two_unique.remove('deposit')

    dp.data_transform(
        scale_columns={"quantile": [col for col in numerical_cols if col not in ['day', 'month']], "min_max": ['day']},
        encode_columns={
            "label": cols_with_two_unique,
            "one_hot": [col for col in categorical_cols if col not in cols_with_two_unique+['deposit']]
        },
        encode_target=True
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

    save_folder = f'data_analysis\\Bank\\figs\\'
    fig_count = 0
    
    for method in model_kwargs.keys():

        fig_count += 1
        model = ML.create_train_predict_eval(
            method=method,
            X_train=dp.X_train.to_numpy(),
            y_train=dp.y_train.to_numpy().ravel(),
            X_test=dp.X_test.to_numpy(),
            y_test=dp.y_test.to_numpy().ravel(),
            y_encoder=dp.y_encoder,
            metrics_fig_path=f"{save_folder}ml_metrics_{fig_count}.png",
            pos_label = 'yes',
            **model_kwargs[method]
        )
        
        if method == "rfc":
            best_model = model
            st.success("Best Performance")

        st.markdown("# ") # For Spacing between sections

    ML.k_means_train_predict(
        # Using X_train, X_test combined as X, since X is not scaled or encode. Similarly, for y,
        data=pd.concat([pd.concat([dp.X_train, dp.X_test], axis=0), pd.concat([dp.y_train, dp.y_test], axis=0)], axis=1),
        n_clusters=3,
        x_axis=['age', 'balance', 'duration', 'campaign', 'pdays', 'previous'],
        y_axis=['campaign', 'pdays', 'previous'],
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
        n_units=[16, 10, 8], n_epochs=20,
        metrics=['accuracy', 'recall'],
        y_encoder=dp.y_encoder,
        classification=True,
        fit_kwargs=dict(batch_size=32),
        pos_label = 'yes'
    )


# insights
def insights() -> None:
    try:
        with open("data_analysis\\Bank\\insights.md", "r") as file:
            st.markdown(file.read())
    except FileNotFoundError:
        st.error("Insights not available")


# Interactive Predictions
def prediction() -> None:
    dp.prediction(numerical_cols, categorical_cols, target, best_model)

