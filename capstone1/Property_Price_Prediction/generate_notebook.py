import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Capstone Project: Property Price Prediction\n",
                "\n",
                "## **Objective**:\n",
                "Develop a machine learning model that accurately predicts house prices based on various features such as location, property size, construction status, and more. This model will help potential buyers and real estate agents estimate property values, aiding in better decision-making and price forecasting.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Data Dictionary\n",
                "\n",
                "| Column Name | Description | Data Type |\n",
                "| :--- | :--- | :--- |\n",
                "| POSTED_BY | Person posting the listing (Owner, Dealer, etc.). | Categorical |\n",
                "| UNDER_CONSTRUCTION | Indicates if the property is under construction (1: Yes, 0: No). | Binary (0/1) |\n",
                "| RERA | Indicates RERA approval status (1: Approved, 0: Not Approved). | Binary (0/1) |\n",
                "| BHK_NO. | Number of Bedrooms, Hall, and Kitchen (BHK). | Integer (Discrete) |\n",
                "| BHK_OR_RK | Property type (BHK or RK \u2013 Room Kitchen). | Categorical |\n",
                "| SQUARE_FT | Total area of the property in square feet. | Float (Continuous) |\n",
                "| READY_TO_MOVE | Indicates if the property is ready for occupancy (1: Yes, 0: No). | Binary (0/1) |\n",
                "| RESALE | Indicates if the property is for resale (1: Yes, 0: No). | Binary (0/1) |\n",
                "| ADDRESS | Location of the property (City, Region). | Text (Alphanumeric) |\n",
                "| LONGITUDE | Longitude coordinate of the property. | Float (Geographic) |\n",
                "| LATITUDE | Latitude coordinate of the property. | Float (Geographic) |\n",
                "| TARGET(PRICE_IN_LACS) | House price in lakhs (Target variable for prediction). | Float (Continuous) |\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# !pip install pandas matplotlib seaborn plotly scikit-learn scipy statsmodels xgboost lightgbm catboost\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Tasks:\n",
                "### 1. Import and Load the Data\n",
                "- Import required libraries (pandas, numpy, matplotlib, seaborn, sklearn, xgboost, lightgbm, catboost, scipy, statsmodels, plotly, etc.)\n",
                "- Load the dataset and explore the structure using .head(), .info() and .describe()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import math\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import plotly.express as px\n",
                "from scipy import stats\n",
                "import statsmodels.api as sm\n",
                "\n",
                "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n",
                "\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.tree import DecisionTreeRegressor\n",
                "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
                "from sklearn.svm import SVR\n",
                "from sklearn.neighbors import KNeighborsRegressor\n",
                "from xgboost import XGBRegressor\n",
                "from lightgbm import LGBMRegressor\n",
                "from catboost import CatBoostRegressor\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the dataset 'House Price.csv'\n",
                "property_df = pd.read_csv('House Price.csv')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display the shape and structure of the dataset\n",
                "# TODO: Use .info(), .shape, .dtypes, and .head()\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Structural and statistical analysis of the dataset\n",
                "# TODO: Use .describe() for summary statistics\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Standardize the names of each column in the dataframe\n",
                "# TODO: Remove leading and trailing spaces from each column name, replace spaces with underscores, and convert to lowercase\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check duplicates \n",
                "# TODO: Find number of duplicate rows\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Remove duplicates\n",
                "# TODO: Remove duplicates from the dataframe and display count again\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Exploratory Data Analysis (EDA)\n",
                "- Visualize the distribution of each feature\n",
                "- Understand correlations (especially with the target variable)\n",
                "- Check for variables distributions.\n",
                "- Summarize insights from EDA\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### a. Visualize the distribution of each feature\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize the distribution of each feature\n",
                "# TODO: Use seaborn's histplot or matplotlib for numeric features, and countplot/barplots for categorical features\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### b. Understand correlations (especially with the target variable)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Find correlation matrix\n",
                "# TODO: Encode categorical variables first if necessary, then calculate and visualize the correlation matrix using sns.heatmap\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate the correlation of each feature with the target variable 'target(price_in_lacs)'\n",
                "# TODO: Sort and visualize the correlations with the target variable\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### c. Check for variables distributions. \n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Analyze distributions\n",
                "# TODO: Further inspect histograms, identify skewed variables, and possibly use plotly for interactive visualizations if helpful\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### d. Summarize insights from EDA\n",
                "\n",
                "*TODO: Write insights gathered from the EDA steps above. Mention highly correlated features, skewness, and patterns observed.*\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Missing Values & Outlier Treatment\n",
                "- a. Check for missing values and treat them if any\n",
                "- b. Check if there are any outliers.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### a. Check for missing values and treat them if any\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Analyze for missing values in the dataset\n",
                "# TODO: Use .isnull().sum() and handle any missing data appropriately (e.g., imputation or removal)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### b. Check if there are any outliers.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check for outliers using boxplots\n",
                "# TODO: Use sns.boxplot for numerical features to visualize outliers\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Treat outliers\n",
                "# TODO: Decide whether to remove or cap outliers (e.g., using quantiles or IQR methods) and implement the treatment\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Feature Engineering & Preprocessing\n",
                "- a. Encode categorical variables\n",
                "- b. Feature scaling for numerical values\n",
                "- c. Check for skewness and treat it if required.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### a. Encode categorical variables\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Encode categorical variables (e.g., posted_by, bhk_or_rk, address, etc.)\n",
                "# TODO: Use pd.get_dummies, LabelEncoder, or TargetEncoder based on the feature properties\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### b. Feature scaling for numerical values\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Feature scaling for numerical values\n",
                "# TODO: Use StandardScaler or MinMaxScaler on numerical columns (excluding the target variable if not needed)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### c. Check for skewness and treat it if required.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check for skewness and kurtosis\n",
                "# TODO: Analyze skewness/kurtosis of numerical variables and apply transformations (e.g., log, Box-Cox) where necessary\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Model Building: Try Multiple Regressors\n",
                "Use regression-based models to train and test the data:\n",
                "- Linear Regression\n",
                "- Decision Tree Regressor\n",
                "- Random Forest Regressor\n",
                "- SVR\n",
                "- KNN\n",
                "- Ensemble Learning methods (XGBoost, LightGBM, CatBoost)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare the train and test data\n",
                "# TODO: Separate features (X) and target variable (y). Then split into train and test sets using train_test_split\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize and train multiple regressors\n",
                "# TODO: Fit Linear Regression, Decision Tree, Random Forest, SVR, KNN\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize and train Gradient Boosting models\n",
                "# TODO: Fit XGBoost, LightGBM, CatBoost, and GradientBoostingRegressor\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Model Evaluation & Overfitting Check\n",
                "- Use metrics:\n",
                "    - Mean Absolute Error (MAE)\n",
                "    - Mean Squared Error (MSE)\n",
                "    - Root Mean Squared Error (RMSE)\n",
                "    - R\u00b2 Score\n",
                "    - Adjusted R\u00b2 Score\n",
                "- Compare performance on both datasets (Training and testing) to detect overfitting\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define a function to train, evaluate and collect metrics for each model\n",
                "# TODO: Calculate MAE, MSE, RMSE, R2, and Adjusted R2 for both train and test predictions\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Compare performance\n",
                "# TODO: Store results in a DataFrame and print it out to determine the best model and check for overfitting gaps\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('property_price_prediction_capstone_notebook.ipynb', 'w') as f:
    json.dump(notebook, f, indent=4)

