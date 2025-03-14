# Data Visualization & Insight Generation

This project includes data wrangling using numpy and pandas, data visualization using seaborn and plotly, ML model and ANN model implementation for identification of patterns and for predictions. All of this has been manifested into a Streamlit app.

# Python Libraries required
1. Numpy 1.26.4
2. Pandas
3. Matplotlib
4. Seaborn
5. Plotly
6. Scikit-learn
7. Tensorflow
8. Streamlit
9. Streamlit-extras

# Intructions on running the Streamlit app

### ⚠️ Important intruction:

> https://drive.google.com/file/d/1X1KzP-Ve3I1bbEmELnaNaz11JWX1NJjB/view?usp=sharing Click on this given link and download the Credit_Card.csv file before running the Streamlit app.
> Since this file has huge size it couldn't be uploaded to GitHub.
> This file must be added to this folder `data_analysis\\Credit_Card\\`.
> Even though the file name is `Credit_Card.csv` but still ensure that it is indeed so before running the Streamlit app.

Download / Import / Fork this repository to your personal system and open the main folder in the integrated terminal and execute the command `Streamlit run app.py`.
![image](https://github.com/user-attachments/assets/8c8df221-7d35-431a-b005-2d2acf5b1b4c)
You will find your Streamlit app running in the default browser.
If not then simply click on the given `Local URL`, in this case `http://localhost:8501`.

If you still facing problems then you can simply download the `PROJECT - Data Visualization & Insight Generation.zip` file and follow the above given instructions.



# The files content of this repository

## `app.py`

This is the python script for the main Streamlit app page.

## `app_pages.py`

This is the python script for implementation of various pages which are to shown in the Streamlit app. App pages incldes classes for each app page like `Introduction` page, `Project Overview` page and others.

## `global_streamlit.py`

It includes all the the global variables that the Streamlit app will require during its execution and other background processes.

## `custom_css`

This directory includes the CSS code for customization of the Streamlit app design.

## `document files`

This directory includes all the `.md` documentation of some of the pages of the Streamlit app.

## `data_analysis`

This directory includes all the python modules and scripts required for the data analysis of the given datasets in this project.
The given datasets include:
1. bank.csv
2. calories_daata.csv
3. credit_card.csv
4. iris.csv
5. parkinsons.csv
6. titanic.csv
7. mushroom.csv
8. social_network_ads.csv
9. medical_insurance.csv
All of these datasets have been collected from kaggle. For more information please go through the project report.

# Main app screenshots
![image](https://github.com/user-attachments/assets/b342b3d2-5a53-4164-8ac5-f0a91479529a)
