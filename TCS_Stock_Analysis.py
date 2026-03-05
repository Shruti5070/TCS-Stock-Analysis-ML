# TCS Stock Data Analysis and Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create graphs folder
graph_folder = "graphs"
os.makedirs(graph_folder, exist_ok=True)

# Load dataset
df = pd.read_csv("TCS_stock_history.csv")
print("First 5 rows:")
print(df.head())
print(df.shape)
# Dataset info
print("\nDataset Info:")
print(df.info())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Closing price trend graph
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'])
plt.title("TCS Stock Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.savefig(os.path.join(graph_folder,"closing_price_trend.png"))
plt.close()

# Create moving averages
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
print("\nDataset with Moving Averages:")
print(df.head())

# Moving average graph
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label="Closing Price")
plt.plot(df['Date'], df['MA50'], label="50 Day Moving Average")
plt.plot(df['Date'], df['MA200'], label="200 Day Moving Average")
plt.title("TCS Moving Average Trend")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.savefig(os.path.join(graph_folder,"moving_average_trend.png"))
plt.close()

# Remove rows with NaN values
df = df.dropna()
print("\nDataset shape after removing NaN:", df.shape)

# Feature selection
X = df[['Open','High','Low','Volume','MA50','MA200']]
y = df['Close']

# Train-test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

# Create and train model
model = LinearRegression()
model.fit(X_train,y_train)

# Prediction
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("\nModel Evaluation:")
print("MAE:",mae)
print("MSE:",mse)
print("R2 Score:",r2)

# Actual vs predicted graph
plt.figure(figsize=(12,6))
plt.plot(y_test.values,label="Actual Price")
plt.plot(y_pred,label="Predicted Price")
plt.title("Actual vs Predicted TCS Stock Prices")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.savefig(os.path.join(graph_folder,"actual_vs_predicted.png"))
plt.close()

print("\nGraphs saved in 'graphs' folder successfully.")