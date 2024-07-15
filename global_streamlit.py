"""
Holds all the global variables used in the Streamlit app.
This file is used to store all the global variables that are used in the Streamlit app.
This helps to keep the code organized and makes it easier to share data between different pages of the app.

### The global variables defined in this file:
1. `selected_page`: The currently selected page of the app.
2. `eda_ml_modelling_option`: The dataset selected for EDA and ML modelling.
"""

# Dependencies
from data_analysis.Bank import bank
from data_analysis.Calories import calories_data
from data_analysis.Iris import Iris
from data_analysis.Social_Network_Ads import social_network_ads
from data_analysis.Titanic import titanic
from data_analysis.Parkinsons import parkinsons
from data_analysis.Credit_Card import credit_card
from data_analysis.Mushrooms import mushrooms
from data_analysis.Medical_Insurance import medical_insurance

# Global variables
selected_page = None

# ML Moodelling page
eda_ml_modelling_option = None
module_dict = {
    "Bank.csv": bank,
    "calories.csv, exercise.csv": calories_data,
    "Iris.csv": Iris,
    "social_network_ads.csv": social_network_ads,
    "titanic.csv": titanic,
    "Parkinsons.csv": parkinsons,
    "credit_card.csv": credit_card,
    "mushrooms.csv": mushrooms,
    "medical_insurance.csv": medical_insurance
}