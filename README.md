# TCS-Stock-Analysis-ML
TCS stock data analysis and price prediction using Python and Machine Learning.

# Project Overview
This project focuses on analyzing the historical stock data of Tata Consultancy Services (TCS) to understand stock behavior, identify trends, and forecast future stock prices.

The project uses data analysis techniques and a machine learning regression model to predict stock closing prices based on market indicators.

# Objective
Analyze the historical data of TCS stock to gain insights into stock behavior, identify trends, and forecast future stock prices.

# Dataset
The dataset used in this project contains historical TCS stock data including:

Date  
Open Price  
High Price  
Low Price  
Close Price  
Volume  
Dividends  
Stock Splits  

Dataset File:
TCS_stock_history.csv

# Tech Stack
Programming Language: Python

Libraries Used:

pandas  
numpy  
matplotlib  
seaborn  
scikit-learn  

# Project Steps

## 1 Data Loading
The historical stock dataset is loaded using pandas.

## 2 Data Understanding
Basic dataset exploration is performed using:

df.head()  
df.info()  
df.describe()

## 3 Data Cleaning
Checked for missing values and ensured correct data types.

Converted the Date column to datetime format and sorted the dataset chronologically.

## 4 Trend Analysis
Visualized historical stock closing price to understand long-term growth patterns.

## 5 Moving Average Analysis
Calculated two important technical indicators:

50-Day Moving Average (MA50)  
200-Day Moving Average (MA200)

These indicators help identify stock trends.

If MA50 > MA200 → Uptrend  
If MA50 < MA200 → Downtrend

## 6 Feature Selection
Features used for prediction:

Open  
High  
Low  
Volume  
MA50  
MA200  

Target variable:

Close Price

## 7 Train Test Split
The dataset was split into training and testing sets using an 80-20 ratio.

## 8 Machine Learning Model
Linear Regression was used to predict stock closing prices.

Model training was performed using the training dataset.

## 9 Model Evaluation
The model performance was evaluated using:

Mean Absolute Error (MAE)  
Mean Squared Error (MSE)  
R² Score

## 10 Visualization
The following graphs were generated and saved:

Closing Price Trend  
Moving Average Trend  
Actual vs Predicted Stock Price

These graphs help visualize stock trends and model predictions.

# Project Structure
TCS-Stock-Analysis-ML/

TCS_Stock_Analysis.py  
TCS_stock_history.csv  

graphs/  
    closing_price_trend.png  
    moving_average_trend.png  
    actual_vs_predicted.png  

README.md

# How to Run the Project

Clone the repository

Install required libraries

pip install pandas numpy matplotlib seaborn scikit-learn

Run the Python script

python TCS_Stock_Analysis.py

The program will analyze the dataset, train the model, and save the generated graphs.

# Learning Outcomes
Understanding stock market data analysis

Using moving averages for trend detection

Building regression models for price prediction

Evaluating machine learning models

Visualizing financial data

# Author
Shruti Maruti Pawar
