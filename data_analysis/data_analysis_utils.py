# Last Updated on 14-07-2024

# Module Documentation
"""
This file has all the necessary utilities for performing data analysis on datasets and displaying the result in a Streamlit app.
From plotting chartsand graphs, data preprocessing to ML and ANN model training and evaluation, everything has been fine tuned for this project.
The classes and functions defined in this file provide a lot of abstarction for the data manipulation and preprocessing, exploratory data analysis, and ML model and ANN model implementation.

### The libraries used:
1. Data Manipulation & Visualization: `numpy` `pandas` `matplotlib.pyplot` `seaborn` `plotly`
2. Machine Learning: `sklearn`
3. Deep Learning: `tensorflow`

### The Classes defined in this file:
1. exploratory_data_analysis
2. data_preprocessing
3. ML_model
4. ANN_model
"""

# All Imports
# Data Manipulation Libraries
import numpy as np
import pandas as pd

# Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, OneHotEncoder, LabelEncoder  # Scalers amd Encoders
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, RFE                                               # Feature Selection Classes [FUTURE UPGRADE] 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis                             # Feature Extrartion Classes [FUTURE UPGRADE]
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression, LinearRegression       # Linear Models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor      # Decision Tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Random Forest
from sklearn.svm import SVC, SVR                                            # Support Vector Machines
from sklearn.neighbors import KNeighborsClassifier                          # K-Nearest Neighbors
from sklearn.naive_bayes import GaussianNB                                  # Naive Bayes
from sklearn.cluster import KMeans                                          # Clustering
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, recall_score, f1_score, precision_score, explained_variance_score, mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

# Deep Learning Libraries
import tensorflow as tf

# Streamlit API
import streamlit as st

# Other libraries
import io               # Used in df_details function
import importlib.util   # Used in is_package_installed function
import pathlib

# Type aliases
Machine_Learning_Model = (
    LogisticRegression | LinearRegression | DecisionTreeClassifier | DecisionTreeRegressor | 
    RandomForestClassifier | RandomForestRegressor | SVC | SVR | KNeighborsClassifier | GaussianNB
)
Encoder = LabelEncoder | OneHotEncoder
Scaler = StandardScaler | MinMaxScaler | MaxAbsScaler | QuantileTransformer


# Check for the existence of a python package
def is_package_installed(package_name) -> bool:
    """
    Checks if a package is installed using importlib.util.find_spec.
    """

    spec = importlib.util.find_spec(package_name)
    return spec is not None


# Set the theme 
# [SUCCESSFULLY TESTED 14-07-2024]
class Theme:
    """
    Class to set the theme for the Streamlit app
    """

    def __init__(self) -> None:
        return

    # Configures Matplotlib and Seaborn styles and sets it as default
    def set_matplotlib_theme(plt_plot_dpi: int=300) -> None:
        """
        Sets the matplotlib context and style, and applies the plotly layout template based on the theme.
        Also sets the default matplotlib plot resolution (dpi).

        ### Args:
            `theme` (string): One of ["light", "dark"]
            `plot_dpi` (int): Default resolution (dpi) for matplotlib plots.

        ### Returns:
            `dict`: dictionary containing matplotlib context styles 
        """

        contexts = {
            'font.size': 10.0,
            'axes.labelsize': 'medium',
            'axes.titlesize': 'large',
            'xtick.labelsize': 'medium',
            'ytick.labelsize': 'medium',
            'legend.fontsize': 'medium',
            'legend.title_fontsize': None,
            'axes.linewidth': 1.0,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.5,
            'lines.markersize': 6.0,
            'patch.linewidth': 0.5,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            'xtick.major.size': 3.5,
            'ytick.major.size': 3.5,
            'xtick.minor.size': 2.0,
            'ytick.minor.size': 2.0
        }

        styles = {
            'axes.facecolor': (0, 0, 0, 0),
            'axes.edgecolor': '#7f8288',
            'axes.grid': True,
            'axes.axisbelow': True,
            'axes.labelcolor': '#7f8288',
            'figure.facecolor': (0, 0, 0, 0),
            'grid.color': '#7f8288',
            'grid.linestyle': '-',
            'text.color': '#7f8288',
            'xtick.color': '#7f8288',
            'ytick.color': '#7f8288',
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'patch.edgecolor': '#7f8288',
            'patch.force_edgecolor': False,
            'image.cmap': 'viridis',
            'font.family': ['sans-serif'],
            'font.sans-serif': [
                'DejaVu Sans',
                'Bitstream Vera Sans',
                'Computer Modern Sans Serif',
                'Lucida Grande',
                'Verdana',
                'Geneva',
                'Lucid',
                'Arial',
                'Helvetica',
                'Avant Garde',
                'sans-serif'
            ],
            'xtick.bottom': True,
            'xtick.top': False,
            'ytick.left': True,
            'ytick.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': True,
            'axes.spines.top': True
        }

        sns.set_theme(context=contexts, style=styles)
        plt.rcParams['legend.framealpha'] = 0.12

        # Matplotlib and Seaborn figure resolution
        plt.rcParams['figure.dpi'] = plt_plot_dpi

    # Configures Plotly Template and sets it as default
    def set_plotly_theme() -> None:
        """
        Sets the plotly layout template based on the theme.
        """

        pio.templates["custom_layout"] = go.layout.Template(
            {"layout" :
                {
                    "font": {"color": "#7f8288"},
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "legend": {"bgcolor": "rgba(127, 130, 136, 0)"},
                    "scene": { # For Scatter 3D plots
                        'xaxis': dict(
                            backgroundcolor='rgba(0,0,0,0)',
                            titlefont=dict(color='#7f8288'),
                            tickfont=dict(color='#7f8288'),
                            linecolor='rgba(127, 130, 136)',
                            zerolinecolor='rgba(127, 130, 136)',
                            gridcolor='rgba(127, 130, 136)',
                            gridwidth=1,
                            ticks = ""
                        ),
                        "yaxis": dict(
                            backgroundcolor='rgba(0,0,0,0)',
                            titlefont=dict(color='#7f8288'),
                            tickfont=dict(color='#7f8288'),
                            linecolor='rgba(127, 130, 136)',
                            zerolinecolor='rgba(127, 130, 136)',
                            gridcolor='rgba(127, 130, 136)',
                            gridwidth=1,
                            ticks = ""
                        ),
                        "zaxis": dict(
                            backgroundcolor='rgba(0,0,0,0)',
                            titlefont=dict(color='#7f8288'),
                            tickfont=dict(color='#7f8288'),
                            linecolor='rgba(127, 130, 136)',
                            zerolinecolor='rgba(127, 130, 136)',
                            gridcolor='rgba(127, 130, 136)',
                            gridwidth=1,
                            ticks = ""
                        )
                    }
                }
            }
        )
        pio.templates.default = "custom_layout"


# Exploratory Data Analysis 
# [SUCCESSFULLY TESTED 14-07-2024]
class exploratory_data_analysis:
    """
    Exploratory Data Analysis from data analysis utilities. 

    ### Parameters
    `data`: pandas.DataFrame or numpy.ndarray
    """

    def __init__(self, data: pd.DataFrame|np.ndarray=None, fig_count: int=0, save_folder: str=None) -> None:
        self.data: pd.DataFrame|np.ndarray = data
        self.fig_count: int = fig_count
        self.save_folder: str = save_folder

    # Function to get column details of a DataFrame
    def get_column_details(self) -> pd.DataFrame:
        """
        This function takes a pandas DataFrame and returns a new DataFrame
        containing details about the columns of the original DataFrame.

        Returns:
            `pandas.DataFrame`: DataFrame containing column details.
        """

        data: pd.DataFrame = self.data

        dtypes = data.dtypes.to_frame(name="dtypes").T
        unique_count = pd.DataFrame([[data[col].nunique() for col in data.columns]], columns=data.columns, index=["unique count"])
        count = [data[col].count() for col in data.columns]
        data_desc = data.describe()  # Exclude categorical columns

        column_details = pd.concat([dtypes, unique_count, data_desc], axis=0)
        column_details.loc['count'] = count
        column_details.rename(index={"count" : "non-null count"}, inplace=True)
        column_details.fillna('-', inplace=True)  # Replace NaN with 'NA' (or any preferred value)

        return column_details

    # Displays the DataFrame details
    def df_details(self) -> None:
        """
        Gets the preliminary details of the loaded dataframe and displays them in the Streamlit App.
        """
        
        st.markdown("### The Dataset")
        with st.expander("Data Frame"):
            st.dataframe(self.data, use_container_width=True, hide_index=True)

        # Small details about the dataset
        st.markdown("### Small deatails about the Dataset")

        with st.expander("Details about the dataset"):
            # Column info.
            st.markdown("#### Column Details:")
            st.dataframe(self.get_column_details(), use_container_width=True)

            # df.info()
            st.markdown("#### Non-null count, Dtype, ")
            st.code("df.info()")
            buffer = io.StringIO()
            self.data.info(buf=buffer)
            s = buffer.getvalue()
            st.code(s)

    # Main plotting func for matplotlib and seaborn plots
    def plot(
        self,
        data: pd.DataFrame|np.ndarray=None,
        figure: plt.Figure=None,
        plot_kind: str=None, 
        figsize: tuple=(8,5),
        markdown_title: str=None,
        title: str=None,
        xlabel: str=None,
        ylabel: str=None,
        grid: bool=True,
        legend: bool=False,
        legend_properties: dict=None, 
        xticks_rotation=0,
        **kwargs
    ) -> None:
        """
        Main ploting function which return figures of matplotlib and seaborn plots.

        ## Args
        #### `plot_kind`: 
            The plotting function. One of
            ["countplot", "lineplot", "histplot", "barplot", "scatter", "lmplot", "heatmap", 
            "kdeplot", "violinplot", "boxplot", "jointplot", "displot", "pairplot"]
        #### `figsize` (tuple): 
            The dimension of the figure object.
        #### `markdown_title` (string):
            Title of the figure in markdown.
        #### `title` (string): 
            Title of the main plot.
        #### `xlabel` (string): 
            Label of the x-axis of the plot.
        #### `ylabel` (string): 
            Label of the y-axis of the plot.
        #### `legend_properties` (dict): 
            Properties of the legend in a dictionary.
        #### `**kwargs`: 
            All the other arguments taken by the `plot_func`

        ## Returns
            `matplotlib.figure.Figure`: Matplotlib figure object

        ## Examples
            >>> plot(data=df, plot_func='lineplot', 
                    figsize=(10,6), title="Age vs Expense",
                    legend_properties = {"title": "Group"} 
                    x='age', y='expense', 
                    hue='group', palette='viridis')
        """

        # The dictionary of plot_kind and its associated plotting function
        plots = {
            "countplot": sns.countplot,
            "lineplot": sns.lineplot,
            "histplot": sns.histplot,
            "barplot": sns.barplot,
            "scatter": sns.barplot,
            "lmplot": sns.lmplot,
            "heatmap": sns.heatmap,
            "kdeplot": sns.kdeplot,
            "violinplot": sns.violinplot,
            "boxplot": sns.boxplot,
            "jointplot": sns.jointplot,
            "displot": sns.displot,
            "pairplot": sns.pairplot
        }

        if markdown_title is not None:
            st.markdown(f"#### {markdown_title}")

        fig_path = f"{self.save_folder}eda_{self.fig_count}.png" if self.save_folder is not None else None

        if fig_path is not None and pathlib.Path(fig_path).exists():
            st.image(fig_path, use_column_width=True)
            self.fig_count += 1
            return

        Theme.set_matplotlib_theme()
        if figure == None:
            fig = plt.figure(figsize=figsize)
            if data is not None:
                g = plots[plot_kind](data=data, **kwargs)
            else:
                g = plots[plot_kind](data=self.data ,**kwargs)
            if isinstance(g, (sns.axisgrid.Grid, sns.axisgrid.FacetGrid, sns.axisgrid.JointGrid, sns.axisgrid.PairGrid)):
                fig = g.figure
            if xlabel is not None and not isinstance(g, sns.axisgrid.PairGrid):
                plt.xlabel(xlabel)
            if ylabel is not None and not isinstance(g, sns.axisgrid.PairGrid):
                plt.ylabel(ylabel)
            if title is not None:
                plt.suptitle(title)
            if legend_properties is not None and legend:
                plt.legend(**legend_properties)
            plt.xticks(rotation=xticks_rotation)
        else:
            fig = figure
        plt.grid(grid)
        st.pyplot(fig, clear_figure=False, use_container_width=True)
        
        if fig_path is not None:
            fig.savefig(fig_path, bbox_inches='tight')

        self.fig_count += 1

    # Plots the basic plots related to a dataset
    def preliminary_plots(
        self,
        numerical_cols: list=[],
        categorical_cols: list=[],
        target: str=None,
        target_is_categorical: bool=True,
        corr_kwargs: dict={},
        show_pair_plot: bool=True,
        pair_plot_exclude_cols: list=[],
        pair_plot_kwargs: dict={},
        dist_plot_kwargs: dict={},
        categorical_plot_kwargs: dict={},
        other_plot_kwargs: dict={},
    ) -> None:
        """
        Automatically plots all the basic plots related to the dataset.
        Includes all the distribution plots, categorical plots, pair plot, correlation plot and independent features vs target feature plot.

        ### Args
        `numerical_cols` (list):
            List of numerical columns in the dataset.
        `categorical_cols` (list):
            List of categorical columns in the dataset.
        `target` (string):
            Name of the target feature in the dataset.
        `target_is_categorical` (bool):
            Whether the target feature is categorical or not.
        `corr_kwargs` (dict):
            Keyword arguments for the correlation heatmap.
        `pair_plot_exclude_cols` (list):
            List of columns to be excluded from the pair plot.
        `pair_plot_hue` (string):
            Column to be used for hue in the pair plot.
        `dist_plot_kwargs` (dict):
            Keyword arguments for the distribution plots.
        `categorical_plot_kwargs` (dict):
            Keyword arguments for the categorical plots.
        `other_plot_hue` (string):
            Column to be used for hue in the other plots.
        
        ### Returns
        `None`
        
        ### Examples
        >>> preliminary_plots(
                numerical_cols=['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'],
                categorical_cols=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'deposit'],
                exclude_cols=['day', 'month', 'duration', 'campaign'],
                target='deposit', target_is_categorical=True, pair_plot_hue='deposit'
            )
        """

        # Correlation Heatmap
        if numerical_cols != []:
            st.markdown("#### `Correlation Heatmap`")
            with st.expander("Plot"):
                self.plot(data=self.data[numerical_cols].corr(), plot_kind='heatmap', markdown_title='Correlation Heatmap', xticks_rotation=90, **corr_kwargs)

        # Pairplot
        if show_pair_plot:
            st.markdown("#### `Pair Plot`")
            with st.expander("Plot"):
                if target_is_categorical:
                    self.plot(data=self.data.drop(columns=pair_plot_exclude_cols, inplace=False), plot_kind='pairplot', hue=target, **pair_plot_kwargs)
                else:
                    self.plot(data=self.data.drop(columns=pair_plot_exclude_cols, inplace=False), plot_kind='pairplot', **pair_plot_kwargs)

        # Distribution plots and Categorical plots
        if numerical_cols != [] or categorical_cols != []:
            
            # Distribution plots
            if numerical_cols != []:
                st.markdown("#### `Distribution plots`")
                with st.expander("Plots"):
                    for col in numerical_cols:
                        self.plot(plot_kind='histplot', markdown_title=f'Distribution of {col.capitalize()}', 
                                  x=col, xlabel=col.capitalize(), ylabel="Density", **dist_plot_kwargs)

            # Categorical plots
            if categorical_cols != []:
                st.markdown("#### `Categorical plots`")
                with st.expander("Plots"):
                    for col in categorical_cols:
                        self.plot(plot_kind='countplot', markdown_title=f'{col.capitalize()}', 
                                  x=col, hue=col, xlabel=col.capitalize(), ylabel="Count", **categorical_plot_kwargs)

            # Target vs Other features
            st.markdown(f"#### `{target.capitalize()} vs Other Features`")
            if target_is_categorical:
                with st.expander("Plots"):
                    for col in numerical_cols:
                        self.plot(plot_kind='kdeplot', markdown_title=f'{col.capitalize()} vs {target.capitalize()}', x=col, hue=target, 
                                  xlabel=col.capitalize(), ylabel='Density', palette=other_plot_kwargs['palette'], fill=True)
                    for col in categorical_cols:
                        if col != target:
                            self.plot(plot_kind='countplot', markdown_title=f'{col.capitalize()} vs {target.capitalize()}', 
                                      x=col, hue=target, xlabel=col.capitalize(), ylabel="Count", palette=other_plot_kwargs['palette'])
            else:
                with st.expander("Plots"):
                    for col in numerical_cols:
                        if col != target:
                            self.plot(plot_kind='jointplot', markdown_title=f'{col.capitalize()} vs {target.capitalize()}', x=col, y=target,
                                      xlabel=col.capitalize(), ylabel=target.capitalize(), legend=True if 'hue' in other_plot_kwargs.keys() else False,
                                      **other_plot_kwargs)
                    for col in categorical_cols:
                        self.plot(plot_kind='kdeplot', markdown_title=f'{col.capitalize()} vs {target.capitalize()}', x=target, hue=col,
                                  xlabel=target.capitalize(), ylabel='Density', palette=other_plot_kwargs['palette'], fill=True)

    # Plots a Scatter3D plot using plotly
    def scatter_3d(
        self,
        x: str,
        y: str,
        z: str,
        data: pd.DataFrame=None,
        color: str=None,
        symbol: str=None,
        color_continuous_scale: str=None,
        color_discrete_map: dict=None,
        size: str=None,
        size_max: int=5,
        opacity: float=1.0,
        width: int=1200,
        height: int=800
    ) -> None:
        """
        Plots a 3D scatter plots

        ### Args
        `x` (string):
            Name of the first feature in the dataset.
        `y` (string):
            Name of the second feature in the dataset.
        `z` (string):
            Name of the third feature in the dataset.
        `data` (pd.DataFrame):
            Dataset to be plotted.
        `hue` (string):
            Column to be used for hue in the plot.
        `color_dict` (dict):
            Dictionary of colors to be used for each unique value in the hue column.
        `marker_color` (string):
            Color of the markers.
        `marker_size` (int):
            Size of the markers.

        ### Examples
        >>> scatter_3d(x='age', y='balance', z='duration', 
                        data=self.data,
                        marker_color='blue',
                        marker_size=10)
                
        >>> scatter_3d(x='age', y='balance', z='duration', 
                        data=self.data, hue='deposit',
                        color_dict={'yes': 'blue', 'no': 'red'},
                        marker_size=10)
        """

        # The dataset
        df = data if data is not None else self.data

        # Selecting custom plotly template as default
        Theme.set_plotly_theme()

        st.markdown(f"#### Relation between {x.capitalize()}, {y.capitalize()} and {z.capitalize()}")
        fig = px.scatter_3d(
            df, x=x, y=y, z=z, 
            color=color, symbol=symbol, 
            size=size, size_max=size_max,
            opacity=opacity,
            color_continuous_scale=color_continuous_scale,
            color_discrete_map=color_discrete_map,
            labels=dict(x=x.capitalize(), y=y.capitalize(), z=z.capitalize()),
            width=width, height=height
        )

        layout = dict(
                    font = dict(size=16, color="#7f8288"),
                    legend = dict(
                        font=dict(size=16, color="#7f8288"),
                        title=dict(font=dict(color="#7f8288")),
                        grouptitlefont=dict(color="#7f8288"),
                        x=1, y=0.5
                    ),
                    coloraxis = dict(
                        colorbar=dict(
                            title=dict(font=dict(color="#7f8288")), 
                            tickfont = dict(color="#7f8288"),
                            orientation = "h", y = 1
                        )
                    ),
                )
        fig.update_layout(layout)
        
        st.plotly_chart(fig, use_container_width=True)

    # Plots a parallel categories plot using plotly
    def parallel_categories(
        self,
        dimensions: list=None,
        data: pd.DataFrame=None,
        color: str=None,
        color_continuous_scale: str=None,
        width: int=1200,
        height: int=400
    ) -> None:
        """
        Plots a parallel categories plot using plotly.

        ### Args
        `dimensions` (list):
            List of dimensions to be plotted.
        `data` (pd.DataFrame):
            Dataset to be plotted.
        `color` (string):
            Column to be used for hue in the plot.
        `color_continuous_scale` (str): Color scale for the plot.
        `width` (int): Width of the plot.
        `height` (int): Height of the plot.
        """

        # The dataset
        df = data.copy() if data is not None else self.data.copy()

        # Selecting custom plotly template as default
        Theme.set_plotly_theme()

        # If color column of dataframe has dtype 'object', then encode it
        # since px.parallel_categories takes a column name which has int or float dtype for its color parameter
        if color is not None and df[color].dtype == "object":
            enc = LabelEncoder()
            df[f'{color}_id'] = enc.fit_transform(df[color])
            color = f'{color}_id'

        st.markdown("#### Parallel Categories Plot")
        fig = px.parallel_categories(
            df,
            dimensions=dimensions,
            color=color,
            color_continuous_scale=color_continuous_scale,
            width=width, height=height,
            labels=dict(zip(dimensions, [col.capitalize() for col in dimensions]))
        )

        layout = dict(
                    font=dict(size=16, color="#7f8288"),
                    legend=dict( 
                        font=dict(size=16, color="#7f8288"),
                        title=dict(font=dict(color="#7f8288")),
                        grouptitlefont=dict(color="#7f8288"),
                        x=1, y=0.5
                    ),
                    coloraxis = dict(
                        colorbar=dict(
                            title=dict(font=dict(color="#7f8288")), 
                            tickfont = dict(color="#7f8288"),
                        )
                    )
                )
        fig.update_layout(layout)
        
        st.plotly_chart(fig, use_container_width=True)

    # Plots a parallel coordinates plot using plotly
    def parallel_coordinates(
        self,
        dimensions: list=None,
        data: pd.DataFrame=None,
        color: str=None,
        color_continuous_scale: str=None,
        width: int=1200,
        height: int=400
    ) -> None:
        """
        Plots a parallel coordinates plot using plotly.

        ### Args
        `dimensions` (list):
            List of dimensions to be plotted.
        `data` (pd.DataFrame):
            Dataset to be plotted.
        `color` (string):
            Column to be used for hue in the plot.
        `color_continuous_scale` (str): Color scale for the plot.
        `width` (int): Width of the plot.
        `height` (int): Height of the plot.
        """

        # The dataset
        df = data.copy() if data is not None else self.data.copy()

        # Selecting custom plotly template as default
        Theme.set_plotly_theme()

        # If color column of dataframe has dtype 'object', then encode it
        # since px.parallel_categories takes a column name which has int or float dtype for its color parameter
        if color is not None and df[color].dtype == "object":
            enc = LabelEncoder()
            df[f'{color}_id'] = enc.fit_transform(df[color])
            color = f'{color}_id'

        st.markdown("#### Parallel Coordinates Plot")
        fig = px.parallel_coordinates(
            df,
            dimensions=dimensions,
            color=color,
            color_continuous_scale=color_continuous_scale,
            width=width, height=height,
            labels=dict(zip(dimensions, [col.capitalize() for col in dimensions]))
        )

        layout = dict(
                    font=dict(size=16, color="#7f8288"),
                    legend=dict( 
                        font=dict(size=16, color="#7f8288"), 
                        title=dict(font=dict(color="#7f8288")),
                        grouptitlefont=dict(color="#7f8288"),
                        x=1, y=0.5
                    ),
                    coloraxis = dict(
                        colorbar=dict(
                            title=dict(font=dict(color="#7f8288")), 
                            tickfont = dict(color="#7f8288"),
                        )
                    )
                )
        fig.update_layout(layout)
        
        st.plotly_chart(fig, use_container_width=True)


# Data Preprocessing 
# [SUCCESSFULLY TESTED 14-07-2024]
class data_preprocessing:
    """
    Data preprocessing from data analysis utilities.

    ### Parameters
    `data`: pandas.DataFrame

    ### Attributes
    data (pandas.DataFrame): The input data which is cleaned and ready for preprocessing.
    X (pandas.DataFrame): The independent features.
    y (pandas.DataFrame): The dependent feature.
    X_train (pandas.DataFrame): The training independent features.
    y_train (pandas.DataFrame): The training dependent feature.
    X_test (pandas.DataFrame): The testing independent features.
    y_test (pandas.DataFrame): The testing dependent feature.
    X_encoders (dict): A dictionary to store the encoders for each feature.
    y_encoder (data_preprocessing.Encoder): The encoder for the dependent feature.
    X_scalers (dict): A dictionary to store the scalers for each feature.
    y_scaler (data_preprocessing.Scaler): The scaler for the dependent feature.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data
        self.X: pd.DataFrame = None
        self.y: pd.DataFrame = None
        self.X_train: pd.DataFrame = None
        self.y_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_test: pd.DataFrame = None
        self.X_encoders: dict[str, data_preprocessing.Encoder] = {}
        self.y_encoder: data_preprocessing.Encoder = None
        self.X_scalers: dict[str, data_preprocessing.Scaler] = {}
        self.y_scaler: data_preprocessing.Scaler = None

    # Managing the missing values
    def data_cleaning(self, impute_values: dict={}, dropna_subset: list=[], insignificant_columns: list=[]):
        """
        Automatically cleans the data.

        ## Args
        `impute_values` (dict):
            Dictionary that contains the column in which values must be imputed as the `key` and the values to impute as `value`.
        `dropna_subset` (list):
            The columns based on whose Null values a sample(row) will be dropped.

        ## Examples
        >>> data_cleaning(impute_values={'col1': 0, 'col2': val}) 
        """

        self.data.fillna(value=impute_values, inplace=True)
        self.data.dropna(subset=dropna_subset, inplace=True)
        self.data.drop(columns=insignificant_columns, inplace=True)

        # Displaying the data cleaning parameters
        st.markdown("### **Data Cleaning**")
        with st.expander("Imputation, Removal of Missing values and Insignificant Columns"):
            
            if impute_values != {}:
                st.markdown("##### `Impute Values`")
                st.code(f"""{impute_values}""")
            
            if dropna_subset != []:
                st.markdown("##### `Dropna Subset`")
                st.code(f"""{dropna_subset}""")

            if insignificant_columns != []:
                st.markdown("##### `Insignificant Columns`")
                st.code(f"""{insignificant_columns}""")

        # Details about cleaned data
        with st.expander("Cleaned Data"):
            # Column info.
            st.dataframe(exploratory_data_analysis(data=self.data).get_column_details(), use_container_width=True)

            # df.info()
            st.code("df.info()")
            buffer = io.StringIO()
            self.data.info(buf=buffer)
            s = buffer.getvalue()
            st.code(s)      

    # Independent & Dependent split
    def independent_dependent_split(self, target_feature: str, excluded_features: list[str]=[], histplot_fig_folder: str=None) -> None:
        """
        Splits the dataset into dependent and independent features and represents them in a DataFrame in the Streamlit app.
        
        ### Parameters
        `target_faeture` (str): Name of the target feature in the data
        `exclude_features` (list[str]): List of features to be excluded from the independent features.
        ``histplot_fig_folder` (str): Folder path to save the histogram plot.
        """

        if target_feature not in self.data.columns:
            raise ValueError(f"Target feature '{target_feature}' not found in the data.")
        
        if excluded_features:
            for feature in excluded_features:
                if feature not in self.data.columns:
                    raise ValueError(f"Excluded feature '{feature}' not found in the data.")

        self.y = self.data[[target_feature]]
        self.X = self.data[[col for col in self.data.columns if col not in excluded_features if col != target_feature]]

        st.markdown("### **Independent & Dependent Features**")
        with st.expander("Details"):
            st.markdown("#### Independent Variables (X):")
            st.dataframe(self.X, use_container_width=True, hide_index=True)

            st.markdown("#### Dependent Variables (y):")
            col1, col2, col3 = st.columns([1, 1, 3])
            col1.dataframe(self.y, use_container_width=True)
            if self.y.dtypes.to_list()[0] == 'object':
                col2.dataframe(self.y.value_counts().head(10), use_container_width=True)
            else:
                col2.dataframe(self.y.describe(), use_container_width=True)
                
            histplot_fig_path = f'{histplot_fig_folder}y_histplot.png' if histplot_fig_folder is not None else None

            if histplot_fig_path is not None and pathlib.Path(histplot_fig_path).exists():
                col3.image(histplot_fig_path, use_column_width=True)
                return
            
            Theme.set_matplotlib_theme()
            fig = plt.figure()
            if self.y[[target_feature]].nunique().iloc[0] <= 20:
                # Categorical distribution
                if self.y[[target_feature]].nunique().iloc[0] <= 5:
                    sns.histplot(data=self.y, x=self.y.columns[0], hue=self.y.columns[0], palette='viridis', legend=False)
                else:
                    sns.histplot(data=self.y, x=self.y.columns[0], hue=self.y.columns[0], palette='viridis', legend=False)
            else:
                # Numerical distribution
                sns.histplot(data=self.y, x=self.y.columns[0])
            col3.pyplot(fig, clear_figure=False, use_container_width=True)

            if histplot_fig_path is not None:
                fig.savefig(histplot_fig_path, bbox_inches='tight')

    # Train & Test split
    def train_test_split(self, test_size: float=0.2, random_state: int|float=42):
        """
        Splits the data into training and testing datasets. And represents them in the Streamlit app.

        ### Parameters
        `test_size` (float): default: 0.2
        `random_state` (int|float): default: 42
        """
        
        st.markdown("### **Train-Test Split**")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        st.code(f"""{{'arrays': (X, y), test_size={test_size}, random_state={random_state}}}""")

        # Displaying the training and testing datasets
        with st.expander("Details"):
            st.markdown("##### X_train:")
            st.dataframe(self.X_train, use_container_width=True)
            st.markdown("##### X_test:")
            st.dataframe(self.X_test, use_container_width=True)
            
            col1, col2 = st.columns([1, 1], gap='large')
            with col1:
                st.markdown("##### y_train:")
                st.dataframe(self.y_train, use_container_width=True)
            with col2:
                st.markdown("##### y_test:")
                st.dataframe(self.y_test, use_container_width=True)

    # Encoder
    def encoder(method: str, **kwargs) -> Encoder:
        """
        Returns an instance of an Encoder class.

        ### Parameters
        `method` (str): By which method will the encoder encode the data.
        `**kwargs` (dict): Additional parameters to be passed to the encoder.
        """

        encoding = {
            "label": LabelEncoder,
            "one_hot": OneHotEncoder
        }

        return encoding[method](**kwargs) # Generates an instance of the encoder

    # Scaler
    def scaler(method: str, **kwargs) -> Scaler:
        """
        Returns an instance of a Scaler class.

        ### Parameters
        `method` (str): By which method will the scaler scale the data.
        `**kwargs` (dict): Additional parameters to be passed to the scaler.
        """
        
        scaling = {
            "std": StandardScaler,
            "min_max": MinMaxScaler,
            "max_abs": MaxAbsScaler,
            'quantile': QuantileTransformer
        }

        return scaling[method](**kwargs) # Generates an instance of the scaler 

    # Scale and Encode X_train, X_test, y_train and y_test directly
    def data_transform(
        self,
        scale_columns: dict={},
        encode_columns: dict={},
        target_scaler: str="",
        encode_target: bool=False,
    ) -> pd.DataFrame:
        """
        Automatically preprocesses the data based on the parameters given.
        First feature encoding is performed and then feature scaling.

        ### Parameters
        `data`: DataFrame
        `scale_columns` (dict): 
            The sclaing method and the columns to scale in a dictionary.
        `encode_columns` (dict): 
            The encoding method and the columns to encode in a dictionary.
        `target_scaler` (str):
            On which scaler will the target feature be scaled. One of ['std', 'min_max', 'max_abs'].
        `encode_target` (bool):
            Whether or not to encode the target feature.

        ### Returns
        `pandas.DataFrame`: Encoded and Scaled DataFrame

        ## Examples
        >>> dp = data_preprocessing(data=df)
            dp.preprocess(
                scale_columns={'std': ['col1', 'col2', 'col3']},
                encode_columns={'label': ['col4', 'col5'], 'one_hot': ['col6']},
                encode_target=True,
            )

        >>> dp = data_preprocessing(data=df)
            dp.preprocess(
                scale_columns={'std': ['col1'], 'min_max': ['col2']},
                target_scaler='std',
            )
        """

        st.markdown("### **Train-Test Split (Scaled & Encoded):**")
        encoders = st.expander("Encoders")
        scalers = st.expander("Scalers")

        # Transforming the independent features
        if scale_columns != {} or encode_columns != {}:

            # Feature Encoding
            with encoders:

                for encoding_method in encode_columns.keys():

                    # Label Encoding
                    if encoding_method == "label" and encode_columns[encoding_method] is not None:
                        st.markdown("##### `Label Encoder`")
                        st.code(
                            f"""
                            # Features
                            {encode_columns[encoding_method]}
                        """)

                        for col in encode_columns[encoding_method]:
                            self.X_encoders[col] = data_preprocessing.encoder(method=encoding_method) # Updating the X_encoders dictionary
                            self.X_train[col] = self.X_encoders[col].fit_transform(self.X_train[col]) # Label encoding (fit-transform) the X_train dataset
                            self.X_test[col] = self.X_encoders[col].transform(self.X_test[col]) # Label encoding (transform) the X_test dataset 

                    # One Hot Encoding
                    if encoding_method == "one_hot" and encode_columns[encoding_method] is not None:
                        enc = data_preprocessing.encoder(method='one_hot', drop='first', sparse_output=False, handle_unknown='ignore')
                        st.markdown("##### `One Hot Encoder`")
                        st.code(
                            f"""
                            # Params
                            {{'drop': 'first', 'sparse_output': False, 'handle_unknown': 'ignore'}}

                            # Features
                            {encode_columns[encoding_method]}
                        """)

                        # One hot encoding (fit-transform) the X_train dataset
                        one_hot_encoded = enc.fit_transform(self.X_train[encode_columns[encoding_method]].to_numpy())
                        one_hot_df = pd.DataFrame(one_hot_encoded, columns=enc.get_feature_names_out(encode_columns[encoding_method]))
                        one_hot_df.set_index(self.X_train.index, inplace=True)          # Matching the indices of the X_train dataset
                        self.X_train = pd.concat([self.X_train, one_hot_df], axis=1)
                        self.X_train.drop(encode_columns[encoding_method], axis=1, inplace=True)

                        # One hot encoding (transform) the X_test dataset
                        one_hot_encoded = enc.transform(self.X_test[encode_columns[encoding_method]].to_numpy())
                        one_hot_df = pd.DataFrame(one_hot_encoded, columns=enc.get_feature_names_out(encode_columns[encoding_method]))
                        one_hot_df.set_index(self.X_test.index, inplace=True)           # Matching the indices of the X_test dataset
                        self.X_test = pd.concat([self.X_test, one_hot_df], axis=1)
                        self.X_test.drop(encode_columns[encoding_method], axis=1, inplace=True)

                        for col in encode_columns[encoding_method]:
                            self.X_encoders[col] = enc # Updating the X_encoders dictionary

            # Feature Scaling
            with scalers:

                for scaling_method in scale_columns.keys():
                    if scale_columns[scaling_method] is not None:
                        if scaling_method == "quantile":
                            st.markdown("##### `Quantile Transformer`")
                            st.code(
                                f"""
                                # Params
                                {{'output_distribution': 'normal'}}
                                
                                # Features
                                {scale_columns[scaling_method]}
                            """)
                        
                        else:
                            st.markdown(f"##### `{dict(std='Standard Scaler', min_max='Min-Max Scaler', max_abs='Max-Abs Scaler')[scaling_method]}`")
                            st.code(
                                f"""
                                # Features
                                {scale_columns[scaling_method]}
                            """)
                            
                        for col in scale_columns[scaling_method]:

                            # Updating the X_scalers dictionary
                            if scaling_method == "quantile":
                                self.X_scalers[col] = data_preprocessing.scaler(method=scaling_method, output_distribution='normal') 
                            else:
                                self.X_scalers[col] = data_preprocessing.scaler(method=scaling_method)
                             
                            self.X_train[col] = self.X_scalers[col].fit_transform(self.X_train[[col]]) # Scaling (fit-transform) the X_train dataset 
                            self.X_test[col] = self.X_scalers[col].transform(self.X_test[[col]]) # Scaling (tranform) the X_test dataset

        # Transforming the target feature
        if (target_scaler != "") ^ (encode_target != False): # Either Scaling or encoding of the target acn be done, not both.
            col = self.y_train.columns[0] # The only one column in y dataset

            # Target scaling
            if target_scaler != "":
                if target_scaler == "quantile":
                    self.y_scaler = data_preprocessing.scaler(method=target_scaler, output_distribution='normal') # Updating the y_scaler, only one instance of a Scaler
                    scalers.markdown("##### `Quantile Transformer` (y_train, y_test)")
                    scalers.code(
                        f"""
                        # Params
                        {{'output_distribution': 'normal'}}

                        # Features
                        {[col]}
                    """)
                
                else:
                    self.y_scaler = data_preprocessing.scaler(method=target_scaler) # Updating the y_scaler, only one instance of a Scaler
                    scalers.markdown(f"##### `{dict(std='Standard Scaler', min_max='Min-Max Scaler', max_abs='Max-Abs Scaler')[target_scaler]}` (y_train, y_test)")
                    scalers.code(
                        f"""
                        # Features
                        {[col]}
                    """)
                
                self.y_train[col] = self.y_scaler.fit_transform(self.y_train[[col]])
                self.y_test[col] = self.y_scaler.transform(self.y_test[[col]])
            
            # Target encoding
            if encode_target == True:
                self.y_encoder = data_preprocessing.encoder(method='label') # Updating the y_encoder, only one instance of an Encoder
                encoders.markdown("##### `Label Encoder` (y_train, y_test)")
                encoders.code(
                    f"""
                    # Features
                    {[col]}
                """)

                enc = self.y_encoder
                self.y_train[col] = enc.fit_transform(self.y_train[col])
                self.y_test[col] = enc.transform(self.y_test[col])

        # Displaying the training and testing datasets
        with st.expander("Details"):
            st.markdown("##### X_train (Scaled & Encoded):")
            st.dataframe(self.X_train, use_container_width=True)
            st.markdown("##### X_test (Scaled & Encoded):")
            st.dataframe(self.X_test, use_container_width=True)
            
            col1, col2 = st.columns([1, 1], gap='large')
            with col1:
                st.markdown("##### y_train (Encoded):")
                st.dataframe(self.y_train, use_container_width=True)
            with col2:
                st.markdown("##### y_test (Encoded):")
                st.dataframe(self.y_test, use_container_width=True)

    # Given a sample data in the form of a dictionary, transforms it.
    def sample_transform(self, X_predict: dict[str, float|int|str]) -> np.ndarray:
        """
        Transforms a sample using the Encoders and Scalers used in preprocessing.

        ### Parameters
        `X_predict` (dict[str, float|int|str]):
            The sample to be transformed.

        ### Returns
        `np.ndarray`: The transformed sample as a numpy array which will be used by the model to predict the target.
        """

        X_predict: pd.DataFrame = pd.DataFrame(X_predict, index=[0]) # Converting the dictionary to a DataFrame
        one_hot_cols = [] # All features encoded with One Hot Encoding technique share the same OneHotEncoder instance
        
        st.markdown("### Sample")
        st.dataframe(X_predict, use_container_width=True)

        for col in X_predict.columns:
            
            # Encoding
            if col in self.X_encoders.keys():
                if isinstance(self.X_encoders[col], OneHotEncoder):
                    one_hot_cols.append(col)
                else:
                    X_predict[col] = self.X_encoders[col].transform(X_predict[col])
            
            # Scaling
            if col in self.X_scalers.keys():
                X_predict[col] = self.X_scalers[col].transform(X_predict[[col]].to_numpy())
            
        if one_hot_cols != []:
            enc = self.X_encoders[one_hot_cols[0]]
            one_hot_encoded = enc.transform(X_predict[one_hot_cols].to_numpy())
            one_hot_df = pd.DataFrame(one_hot_encoded, columns=enc.get_feature_names_out(one_hot_cols))
            X_predict = pd.concat([X_predict, one_hot_df], axis=1)
            X_predict.drop(one_hot_cols, axis=1, inplace=True)

        st.markdown("### **Transformed Sample**")
        st.dataframe(X_predict, use_container_width=True)

        return X_predict.to_numpy() # 2D array with only one row

    # Prediction of target feature from user input sample data
    def prediction(self, numerical_cols: list[str], categorical_cols: list[str], target: str, best_model: Machine_Learning_Model):
        """
        Predicts the target feature for a given sample.

        ### Parameters
        `numerical_cols` (list[str]): List of numerical columns.
        `categorical_cols` (list[str]): List of categorical columns.
        `target` (str): Target feature.
        `best_model` (Machine_Learning_Model): Best model for prediction.
        """
        
        X_predict: dict[str, float|int|str] = {} # Column Name and its single sample value in a dictionary
        y_predicted = None

        container = st.empty()
        with container:
            with st.expander("Input values of features", expanded=True):
                with st.form(key='interactive_predictions', border=False):
                    for col in self.X.columns:
                        if col in numerical_cols:
                            X_predict[col] = st.number_input(label=col, value=0.000)
                        if col in categorical_cols:
                            X_predict[col] = st.selectbox(label=col, options=self.X[col].unique(), index=0)

                    submit = st.form_submit_button(label='Predict')

        if submit and best_model is not None:
            X_predict = self.sample_transform(X_predict)
            y_predicted = best_model.predict(X_predict)

            if self.y_encoder is not None:
                y_predicted = self.y_encoder.inverse_transform(y_predicted)

            if self.y_scaler is not None:
                y_predicted = self.y_scaler.inverse_transform(y_predicted.reshape(-1, 1)).ravel()

            st.markdown(f"### Predicted {target}: `{y_predicted[0]}`")


# Feature Selection
# [FUTURE UPGRADE]
class feature_selection:
    """
    Feature selection class from data analysis utilities.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
    
    @st.cache_resource
    def feature_selector(
            self, 
            method: str, 
            estimator: str,
            estimator_properties: dict={}, 
            **kwargs):
        """
        Selects the features based on Wrapper methods: Forward Selection, Backward Elimination, Recursive Selection.

        ## Args
        `method`: The method of feature selection. One of ['forward', 'backward', 'recursive']

        ## Returns
        `list`: List of all the selected features.
        """

        estimator_methods = {
            "lin_reg": LinearRegression,
            "log_reg": LogisticRegression,
            "svr": SVR,
            "svc": SVC,
            "knn": KNeighborsClassifier,
            "rfr": RandomForestRegressor,
            "rfc": RandomForestClassifier,
        }

        if method == "forward" or method == "backward":
            fs = SequentialFeatureSelector(estimator=estimator_methods[estimator](**estimator_properties), direction=method, n_jobs=-1, **kwargs)


# Feature extraction
# [FUTURE UPGRADE]
class feature_extraction:
    """
    Feature extraction class from data analysis utilities.
    """
    
    def __init__(self) -> None:
        return

    def feature_extractor(method: str, **kwargs):
        """
        Genrates the feature extractor objects
        """
        
        f_extractors = {
            "lda": LinearDiscriminantAnalysis,
            "qda": QuadraticDiscriminantAnalysis,
            "pca": PCA,
            "kernel_pca": KernelPCA
        }

        return f_extractors[method](**kwargs)


# Function for displaying performance metrics including the confusion matrix
# [SUCCESSFULLY TESTED 14-07-2024]
def display_metrics(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    classification_dataset: bool=True,
    fig_path: str=None,
    pos_label: str|int=1, 
    average: str="binary"
) -> None:
    """
    Displays the performance figures or the confusion matrix, and generates the metrics report. It is used in the ml_model class and the ANN class.

    ### Parameters
    `y_pred` (np.ndarray): Predicted values.
    `y_test` (np.ndarray): True values.
    `classification_dataset` (bool): Whether the target feature in a dataset is a classification or a distribution. If False, distribution metrics are displayed.
    """

    # Distributive target
    if classification_dataset == False:

        # Display true and predicted lineplot and distribution. Also, Absolute Error and Squared Error.
        with st.expander("Prediction vs True"):

            if fig_path is not None and pathlib.Path(fig_path).exists():
                st.image(fig_path, use_column_width=True)

            else:
                Theme.set_matplotlib_theme()

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
                # Subplot 1: Lineplot
                sns.scatterplot(x=np.arange(len(y_test)), y=y_test, ax=axes[0, 0], color='limegreen', alpha=0.8, markers='o', s=10, edgecolor='black', linewidth=0.2, label='True')
                sns.scatterplot(x=np.arange(len(y_pred)), y=y_pred, ax=axes[0, 0], color='orange', alpha=0.8, markers='o', s=10, edgecolor='black', linewidth=0.2, label='Predicted')
                axes[0, 0].legend(loc='upper right')

                # Subplot 2: Distribution
                sns.kdeplot(x=y_test, ax=axes[0, 1], color='limegreen', linewidth=1.5, alpha=0.5, fill=True, label='True')
                sns.kdeplot(x=y_pred, ax=axes[0, 1], color='orange', linewidth=1.5, alpha=0.5, fill=True, label='Predicted')
                axes[0, 1].legend(loc='upper right')

                # Subplot 3: Absolute Error
                sns.lineplot(x=np.arange(len(y_test)), y=np.abs(y_test - y_pred), ax=axes[1, 0], color='#97a2be', linewidth=0.8)
                axes[1, 0].fill_between(x=np.arange(len(y_test)), y1=np.abs(y_test - y_pred), color='#97a2be', label='Absolute Error')
                axes[1, 0].legend(loc='upper right')

                # Subplot 4: Squared Error
                sns.lineplot(x=np.arange(len(y_test)), y=(y_test - y_pred)**2, ax=axes[1, 1], color='#97a2be', linewidth=0.8)
                axes[1, 1].fill_between(x=np.arange(len(y_test)), y1=(y_test - y_pred)**2, color='#97a2be', label='Squared Error')
                axes[1, 1].legend(loc='upper right')

                st.pyplot(fig, use_container_width=True, clear_figure=False)

                if fig_path is not None:
                    fig.savefig(fig_path, bbox_inches='tight')

        # Display performance metrics
        with st.expander("Performance Metrics"):
            EVS = explained_variance_score(y_pred=y_pred, y_true=y_test)
            MAE = mean_absolute_error(y_pred=y_pred, y_true=y_test)
            MSE = mean_squared_error(y_pred=y_pred, y_true=y_test)
            R2 = r2_score(y_pred=y_pred, y_true=y_test)
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            col1.metric(label="Explained Variance Score", value=f"{EVS:.4f}")
            col2.metric(label="Mean Absolute Error", value=f"{MAE:.4f}")
            col3.metric(label="Mean Squared Error", value=f"{MSE:.4f}")
            col4.metric(label="R2 Score", value=f"{R2:.4f}")
    
    # Categorical target
    else:
    
        # Display confusion matrix
        with st.expander("Confusion Matrix"):

            if fig_path is not None and pathlib.Path(fig_path).exists():
                st.image(fig_path, use_column_width=True)
            
            else:
                Theme.set_matplotlib_theme()

                confusion_matrix_fig = ConfusionMatrixDisplay.from_predictions(y_pred=y_pred, y_true=y_test, cmap='Blues').figure_
                plt.grid(False)
                st.pyplot(confusion_matrix_fig)

                if fig_path is not None:
                    confusion_matrix_fig.savefig(fig_path, bbox_inches='tight')

        # Display performance metrics
        with st.expander("Performance Metrics"):
            accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
            f1 = f1_score(y_pred=y_pred, y_true=y_test, pos_label=pos_label, average=average)
            recall = recall_score(y_pred=y_pred, y_true=y_test, pos_label=pos_label, average=average)
            precision = precision_score(y_pred=y_pred, y_true=y_test, pos_label=pos_label, average=average)
            st.code(f"Classification Report:\n{classification_report(y_pred=y_pred, y_true=y_test)}")
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            col1.metric(label="Accuracy", value=f"{accuracy*100:.2f} %")
            col2.metric(label="Precision", value=f"{precision*100:.2f} %")
            col3.metric(label="Recall", value=f"{recall*100:.2f} %")
            col4.metric(label="F1 Score", value=f"{f1*100:.2f} %")


# ML Model Implementation
# [SUCCESSFULLY TESTED 14-07-2024]
class ml_model:
    """
    ML models cached with Streamlit in-built caching methods.
    """
    
    def __init__(self) -> None:
        return

    @st.cache_resource
    def create_train_model(
        method: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Machine_Learning_Model:
        """
        Creates the machine learning model, trains it and returns the trained ML model.

        ### Parameters
        `method` (str): Method of machine learning model.
        `X_train` (np.ndarray): Training features.
        `y_train` (np.ndarray): Training labels.
        `**kwargs` (dict): Keyword arguments for the machine learning model.
        
        ### Returns
        `Machine_Learning_Model`: Trained machine learning model.
        """

        models = {
            "log_reg": LogisticRegression,
            "lin_reg": LinearRegression,
            "dtc": DecisionTreeClassifier,
            "dtr": DecisionTreeRegressor,
            "rfc": RandomForestClassifier,
            "rfr": RandomForestRegressor,
            "svc": SVC,
            "svr": SVR,
            "knc": KNeighborsClassifier,
            "gauss_nb": GaussianNB
        }
    
        model_name = {
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

        # Display model name and parameters
        st.markdown(f"### `{model_name[method]}`")
        if kwargs != {}:
            st.code(f"""{kwargs}""")

        model = models[method](**kwargs)
        model.fit(X_train, y_train)
        
        return model
    
    def predict_evaluate(
        X_test: np.ndarray,
        y_test: np.ndarray, # 1D array
        model: Machine_Learning_Model,
        classification: bool=True,
        metrics_fig_path: str=None,
        y_scaler: Scaler=None,
        y_encoder: Encoder=None,
        pos_label: str|int=1,
        average: str="binary",
    ) -> None:
        """
        Predicts the labels for the test set and evaluates the model. Generates the confusion matrix or performance figures, and the metrics.

        ### Parameters
        `model` (Machine_Learning_Model): Trained machine learning model.
        `X_test` (np.ndarray): Test features.
        `y_test` (np.ndarray): Test labels. Must be a 1D array.
        `classification` (bool): Whether the target variable is categorical or continuous. If False, the target variable is assumed to be continuous.
        `metrics_fig_path` (str): Path to save the metrics figure.
        `y_scaler` (data_preprocessing.Scaler): Scaler for the target variable.
        `y_encoder` (data_preprocessing.Encoder): Encoder for the target variable.
        `pos_label` (str|int): Label for the positive class.
        `average` (str): Determines the type of averaging performed on the data. Curremtly, only `binary` and `macro` are supported. Default is `binary`.
        """

        y_pred = model.predict(X_test)
            
        if (y_scaler is not None) ^ (y_encoder is not None):

            if y_scaler is not None:
                y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
                y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            
            if y_encoder is not None:
                y_test = y_encoder.inverse_transform(y_test)
                y_pred = y_encoder.inverse_transform(y_pred)
                # Do not convert to column vector since y_encoder is a LabelEncoder 
                # and its inverse_transform method can take only and only 1D array as an argument for its y parameter.

        # Display metrics
        display_metrics(y_pred, y_test, classification, metrics_fig_path, pos_label, average)

    def create_train_predict_eval(
        method: str, 
        X_train: np.ndarray, 
        y_train: np.ndarray, # 1D array
        X_test: np.ndarray,
        y_test: np.ndarray,  # 1D array
        y_scaler: Scaler=None,
        y_encoder: Encoder=None,
        classification: bool=True,
        metrics_fig_path: str=None,
        pos_label: str|int=1,
        average: str="binary",
        **model_kwargs,
    ) -> Machine_Learning_Model:
        """
        Creates the machine learning model, trains it and evaluates it.
        Combination of `create_train_model` and `predict_evaluate` functions.

        ### Args
        `method` (str): Method of machine learning model.
        `X_train` (np.ndarray): Training features.
        `y_train` (np.ndarray): Training labels. Must be a 1D array.
        `X_test` (np.ndarray): Testing features.
        `y_test` (np.ndarray): Testing labels. Must be a 1D array.
        `classification` (bool): Whether the target feature in a dataset is a classification or a distribution. If False, distribution metrics are displayed.
        `metrics_fig_path` (str): Path to the figure file (.png) where the metrics figure will be saved.
        `y_scaler` (data_preprocessing.Scaler): Scaler used for the target feature.
        `y_encoder` (data_preprocessing.Encoder): Encoder used for the target feature.
        `pos_label` (str|int): Label for the positive class.
        `average` (str): Determines the type of averaging performed on the data. Curremtly, only `binary` and `macro` are supported. Default is `binary`.
        `**model_kwargs` (dict): Keyword arguments for the machine learning model.
        """

        model = ml_model.create_train_model(method, X_train, y_train, **model_kwargs)
        ml_model.predict_evaluate(X_test, y_test, model, classification, metrics_fig_path, y_scaler, y_encoder, pos_label, average)

        return model # Returns the trained ML model

    def k_means_train_predict(
        data: pd.DataFrame,
        n_clusters: int,
        x_axis: list[str]=[],
        y_axis: list[str]=[],
        random_state: int=42,
        palette: str='viridis',
        centroids_color: str='red',
        save_folder: str=None
    ) -> None:
        """
        K-Means clustering implementation.

        ### Args
        `X` (np.ndarray): Data to be clustered.
        `n_clusters` (int): Number of clusters.
        `x_axis` (list[str]): List of x-axis column names for the clustering plot.
        `y_axis` (list[str]): List of y-axis column names for the clustering plot.
        `random_state` (int): Random state for reproducibility.
        `palette` (str): Color palette for the clustering plot.
        `centroids_color` (str): Color for the centroids.
        `save_folder` (str): Path to save the k-means clustering plot.
        """
        
        st.markdown(f"### `K-Means Clustering`")
        st.code(f"""{{'n_clusters': {n_clusters}, 'init': 'k-means++', 'random_state': {random_state}}}""")

        # Elbow Method of finding the optimal k, irrespective of n_clusters
        with st.expander("Elbow Method"):
            if save_folder is not None and pathlib.Path(f'{save_folder}k_means_elbow_method.png').exists():
                st.image(f'{save_folder}k_means_elbow_method.png', use_column_width=True)
            
            else:
                Theme.set_matplotlib_theme()

                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=random_state)
                    kmeans.fit(data.to_numpy())
                    wcss.append(kmeans.inertia_)
                
                fig = plt.figure()
                sns.lineplot(x=range(1, 11), y=wcss, color='orange', linewidth=2)
                plt.xlabel('Number of clusters')
                plt.ylabel('WCSS')
                plt.title('Elbow Method for Optimal k')
                st.pyplot(fig, use_container_width=True, clear_figure=False)

                if save_folder is not None:
                    fig.savefig(f'{save_folder}k_means_elbow_method.png', bbox_inches='tight')

        # Plotting the clusters
        with st.expander("Clusters Plot"):

            kmeans = None

            axes_plotted = []
            for x_feature in x_axis:
                for y_feature in y_axis:
                    
                    # If any combination of x_feature and y_feature has not been already plotted do not plot it if such a combination appears again
                    # Do not plot cluster plot if x_feature and y_feature is same which creates an identity scatter plot with `x = y`
                    if not((x_feature, y_feature) in axes_plotted or (y_feature, x_feature) in axes_plotted) and (x_feature != y_feature):

                        axes_plotted.append((x_feature, y_feature))
            
                        if save_folder is not None and pathlib.Path(f'{save_folder}k_means_cluster_plot_{x_feature}_{y_feature}.png').exists():
                            st.image(f'{save_folder}k_means_cluster_plot_{x_feature}_{y_feature}.png', use_column_width=True)

                        else:
                            Theme.set_matplotlib_theme()

                            if kmeans is None:
                                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state).fit(data.to_numpy())

                            fig = plt.figure()
                            sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=kmeans.labels_, palette=palette, s=30)
                            plt.scatter(
                                kmeans.cluster_centers_[:, data.columns.get_loc(x_feature)], # Gets the x_feature-coordinate value of the centroids
                                kmeans.cluster_centers_[:, data.columns.get_loc(y_feature)], # Gets the y_feature-coordinate value of the centroids
                                s=50, c=centroids_color, label='Centroids'
                            )

                            # Add labels and title
                            plt.xlabel(x_feature.capitalize())
                            plt.ylabel(y_feature.capitalize())
                            plt.title('K-Means Clustering Results')
                            plt.legend()
                            st.pyplot(fig, use_container_width=True, clear_figure=False)

                            if save_folder is not None:
                                fig.savefig(f'{save_folder}k_means_cluster_plot{x_feature}_{y_feature}.png', bbox_inches='tight')


# ANN Implementation 
# [SUCCESSFULLY TESTED 14-07-2024]
class ANN:
    """
    ANN class from data analysis utilities. Provides abstraction from the tensorflow library.
    """

    def __init__(self) -> None:
        return
    
    @st.cache_resource
    def create_train_ann(
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_layers: int,
        n_units: list[int],
        n_classes: None|int=None,
        n_epochs: int=1,
        metrics: list[str]=["accuracy"],
        ann_kwargs: dict={},
        fit_kwargs: dict={}
    ) -> None | tf.keras.models.Sequential:
        """
        Creates and trains an ANN model.

        ### Parameters
        `X_train` (numpy.ndarray): Training features.
        `y_train` (numpy.ndarray): Training labels. Must be a 1D array.
        `n_layers` (int): Number of hidden layers.
        `n_classes` (int): Number of classes.
        `n_units` (list[int]): Number of units in each hidden layer.
        `n_epochs` (int): Number of epochs for which the ANN model will be trained.
        `metrics` (list[str]): List of metrics to be evaluated.
        `ann_kwargs` (dict): Keyword arguments for the ANN model.
        `fit_kwargs` (dict): Keyword arguments for the ANN model fitting.

        ### Returns
        `tf.keras.models.Sequential`: The trained and ANN model.
        """
        
        # Check if tensorflow is installed
        if not is_package_installed("tensorflow"):
            st.error("Package **tensorflow** is not installed. Cannot create ANN model. Install it to create, train and evaluate ANN model.")
            st.info("""
                To install to tensorflow run the following commands in your terminal:
                
                For GPU users
                ```
                pip install tensorflow[and-cuda]
                ```
                
                For CPU users
                ```
                pip install tensorflow
                ```
                
                Once installation is completed, restart the kernel and run the app again.
            """)
            return
        else:
            st.success("Package **tensorflow** is installed.")
            
            st.markdown("### `Artificial Neural Network`")

            # ANN model creation
            ann = tf.keras.models.Sequential(**ann_kwargs)

            # Display the ANN model arguments
            if ann_kwargs != {}:
                st.code(f"""{ann_kwargs}""")

            # Hidden layers
            for i in range(n_layers):
                ann.add(tf.keras.layers.Dense(units=n_units[i], activation='relu'))

                # Display the hidden layer arguments
                col1, col2 = st.columns([1, 4])
                col1.markdown(f"##### Hidden Layer {i+1}")
                col2.code(f"""{dict(units=n_units[i], activation='relu')}""")
            
            # Output layer and ANN Compilation
            # Regression target feature
            if n_classes is None: 
                # The output layer arguments for ANN compilation for regression target features
                ann.add(tf.keras.layers.Dense(units=1))
                ann.compile(optimizer='adam', loss='mse', metrics=metrics)

                # Display the output layer and ANN compilation arguments
                col1, col2 = st.columns([1, 4])
                col1.markdown(f"##### Output Layer")
                col2.code(f"""{dict(units=1)}""")
                col1, col2 = st.columns([1, 7])
                col1.markdown(f"##### Compile ANN")
                col2.code(f"""{dict(optimizer='adam', loss='mse', metrics=metrics)}""")
            
            # Classification target feature
            else:
                # The output layer arguments for ANN compilation for classification target features
                activation = 'softmax' if n_classes > 2 else 'sigmoid'
                loss = 'sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'
                units = n_classes if n_classes > 2 else 1

                # The output layer arguments for ANN compilation for classification target features
                ann.add(tf.keras.layers.Dense(units=units, activation=activation))
                ann.compile(optimizer='adam', loss=loss, metrics=metrics)
            
                # Display the output layer and ANN compilation arguments
                col1, col2 = st.columns([1, 4])
                col1.markdown(f"##### Output Layer")
                col2.code(f"""{dict(units=units, activation=activation)}""")
                col1, col2 = st.columns([1, 7])
                col1.markdown(f"##### Compile ANN")
                col2.code(f"""{dict(optimizer='adam', loss=loss, metrics=metrics)}""")
            
            # Display the ANN model fitting arguments
            col1, col2 = st.columns([1, 7])
            col1.markdown(f"##### Fit ANN")
            fit_kwargs_str = f"""{fit_kwargs}""".replace("{", '').replace("}", '')
            col2.code(f"""{{'x': X_train, 'y': y_train, 'epochs': {n_epochs}, {fit_kwargs_str}}}""")

            st.markdown("#### Training the ANN Model")

            with st.expander(label=f"ANN Training", expanded=True):
                st.markdown("")  # Empty line
                progress_bar = st.progress(0)

                # Train the model

                for epoch in range(-1, n_epochs):
                    col1, col2, col3, *metric_cols = st.columns([1, 2, 1] + [1] * len(metrics))  # Dynamically create columns for metrics

                    # The title of the columns of the training details
                    if epoch == -1:
                        col1.markdown("**Epoch**")
                        col2.markdown("**Training Progress**")
                        col3.markdown("**Loss**")
                        for i, metric in enumerate(metrics):
                            metric_cols[i].markdown(f"**{metric.capitalize()}**")

                    # The actual training details through every epoch
                    else:
                        col1.write(f"{epoch + 1}/{n_epochs}")
                        epoch_progress = col2.progress(0)

                        # Train the model epoch by epoch
                        history = ann.fit(X_train, y_train, epochs=1, **fit_kwargs)

                        epoch_progress.progress(100)

                        col3.write(f"{history.history['loss'][0]:.6f}")  # Display loss value

                        # Update progress bar and display training information
                        progress_bar.progress((epoch + 1) / n_epochs)
                        for i, metric in enumerate(metrics):
                            metric_cols[i].write(f"{history.history[metric][0]:.6f}")  # Display metric values

            return ann # Returns the trained ANN Model

    def predict_evaluate(
        X_test: np.ndarray,
        y_test: np.ndarray,
        ann_model: tf.keras.models.Sequential=None,
        classification: bool=True,
        y_scaler: Scaler=None,
        y_encoder: Encoder=None,
        pos_label: str|int=1,
        average: str="binary"
    ) -> None:
        """
        Predicts the labels for the X_test data using the trained ANN model.
        Also, displays the evaluation metrics and confusion matrix.

        ## Args
        `X_test` (numpy.ndarray): Test features.
        `y_test` (numpy.ndarray): Test labels. Must be a 1D array.
        `ann_model` (tf.keras.models.Sequential): Trained ANN model.
        `classification` (bool): Whether the target variable is categorical or continuous. If False, the target variable is assumed to be continuous.
        `y_scaler` (data_preprocessing.Scaler): Scaler for the target variable.
        `y_encoder` (data_preprocessing.Encoder): Encoder for the target variable.
        `pos_label` (str|int): Positive label for the target variable.
        `average` (str): Determines the type of averaging performed on the data. Curremtly, only `binary` and `macro` are supported. Default is `binary`.
        """

        if ann_model is None:
            return
        
        st.markdown("#### Evaluation Metrics")

        y_pred: np.ndarray = ann_model.predict(X_test) # Strictly a 2-D Array
        
        if (y_scaler is not None) ^ (y_encoder is not None):

            # y has been scaled only
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

            # y has been encoded only
            if y_encoder is not None:
                if y_pred.shape[1] > 1: # n_features > 1
                    y_pred = np.argmax(y_pred, axis=1).ravel()
                else:
                    y_pred = (y_pred > 0.5).astype(int).ravel()
                y_pred = y_encoder.inverse_transform(y_pred)
                y_test = y_encoder.inverse_transform(y_test)
                # Do not convert to column vector since y_encoder is a LabelEncoder 
                # and its inverse_transform method can take only and only 1D array as an argument for its y parameter.
        
        else:
            if classification == True:
                if y_pred.shape[1] > 1: # n_features > 1
                    y_pred = np.argmax(y_pred, axis=1).ravel()
                else:
                    y_pred = (y_pred > 0.5).astype(int).ravel()

        # Display metrics
        display_metrics(y_pred, y_test, classification, None, pos_label, average)
    
    def create_train_predict_eval(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_layers: int,
        n_units: list[int],
        n_classes: int=None,
        n_epochs: int=1,
        metrics: list[str]=["accuracy"],
        y_scaler: Scaler=None,
        y_encoder: Encoder=None,
        classification: bool=True,
        ann_kwargs: dict={},
        fit_kwargs: dict={},
        pos_label: str|int=1,
        average: str="binary"
    ) -> None | tf.keras.models.Sequential:
        """
        Creates, trains, and evaluates an ANN model.
        Combination of `create_train_ann` and `predict_evaluate` functions.

        ### Parameters
        `X_train` (numpy.ndarray): Training features.
        `y_train` (numpy.ndarray): Training labels. Must be a 1D array.
        `X_test` (numpy.ndarray): Testing features.
        `y_test` (numpy.ndarray): Testing labels. Must be a 1D array.
        `n_layers` (int): Number of hidden layers.
        `n_units` (list[int]): Number of units in each hidden layer.
        `n_classes` (int): Number of classes.
        `n_epochs` (int): Number of epochs for which the ANN model will be trained.
        `metrics` (list[str]): List of metrics to be evaluated.
        `y_scaler` (data_preprocessing.Scaler): Scaler used for the target feature.
        `y_encoder` (data_preprocessing.Encoder): Encoder used for the target feature.
        `classification` (bool): Whether the target feature in a dataset is a classification or a distribution. If False, distribution metrics are displayed.
        `ann_kwargs` (dict): Keyword arguments for the ANN model.
        `fit_kwargs` (dict): Keyword arguments for the ANN model fitting.
        `pos_label` (str|int): Positive label for the target variable.
        `average` (str): Determines the type of averaging performed on the data. Curremtly, only `binary` and `macro` are supported. Default is `binary`.
        """

        ann = ANN.create_train_ann(X_train, y_train, n_layers, n_units, n_classes, n_epochs, metrics, ann_kwargs, fit_kwargs)
        ANN.predict_evaluate(X_test, y_test, ann, classification, y_scaler, y_encoder, pos_label, average)

        return ann # Returns the trained ANN Model
    
