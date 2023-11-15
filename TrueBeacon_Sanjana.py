#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[1]:


pip install pyarrow


# In[6]:


import pyarrow.parquet as pq

parquet_file_path = 'C:/Users/sanjana/Downloads/data.parquet'

table = pq.read_table(parquet_file_path)

df = table.to_pandas()
print(df.head())


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore


# In[8]:


df['spread'] = df['banknifty'] - df['nifty']

# Display the first few rows of the modified dataset
print(df.head())


# In[10]:


missing_values = df[['banknifty', 'nifty']].isnull().sum()
print("Missing values in relevant columns:\n", missing_values)

# Forward fill missing values
df[['banknifty', 'nifty']] = df[['banknifty', 'nifty']].fillna(method='ffill')

# Calculate the spread
df['spread'] = df['banknifty'] - df['nifty']

# Check for missing values in the spread column
print("Missing values in the spread column:\n", df['spread'].isnull().sum())


# In[11]:


missing_values = df[['banknifty', 'nifty']].isnull().sum()
print("Missing values in relevant columns:\n", missing_values)


# In[12]:


# Check z-score calculation
df['spread_zscore'] = zscore(df['spread'])

# Display the first few rows of the modified dataset
print(df.head())


# In[16]:


# Assuming 'spread_zscore' is the z-score column you calculated
# Define your z-score thresholds for long and short positions
zscore_threshold_long = -1
zscore_threshold_short = 1

# Print relevant information for debugging
print("Spread Z-Scores:\n", df[['spread_zscore']].head())
print("Spread Values:\n", df[['spread']].head())

# Calculate trading positions
df['position'] = np.where(df['spread_zscore'] > zscore_threshold_short, -1, np.where(df['spread_zscore'] < zscore_threshold_long, 1, 0))

# Print the 'position' column for debugging
print("Trading Positions:\n", df[['position']].head())

# Assuming 'pnl' is the column representing the Profit/Loss based on the provided formula
df['pnl_base_model'] = df['position'].shift(1) * df['spread']

# Display the first few rows of the modified dataset
print(df.head())


# In[18]:


df['pnl'] = df['spread'] * (df['tte'] ** 0.7)


# In[19]:


total_profit_loss = df['pnl'].sum()
print("Total Profit/Loss:", total_profit_loss)


# In[22]:


plt.figure(figsize=(12, 6))
plt.plot( df['spread'], label='spread')
plt.legend()


# In[31]:


# Calculate cumulative P/L
df['cumulative_pnl_base'] = df['pnl_base_model'].cumsum()

# Check if the denominator is zero before calculating Sharpe Ratio
if df['pnl_base_model'].std() != 0:
    sharpe_ratio_base = df['pnl_base_model'].mean() / df['pnl_base_model'].std()
else:
    sharpe_ratio_base = np.nan  # Set to NaN or handle it in a way that makes sense for your analysis
    
# Display Sharpe Ratio
print("Sharpe Ratio - Base Model:", sharpe_ratio_base)


# In[32]:



# Calculate Drawdown
df['drawdown_base'] = df['cumulative_pnl_base'] - df['cumulative_pnl_base'].cummax()

# Plot cumulative P/L and Drawdown
plt.plot(df['cumulative_pnl_base'], label='Cumulative P/L - Base')
plt.plot(df['drawdown_base'], label='Drawdown - Base')
plt.legend()
plt.show()

# Display Sharpe Ratio
print("Sharpe Ratio - Base Model:", sharpe_ratio_base)


# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = df[['spread', 'tte']]
y = df['pnl']


# Handle missing values
X = X.fillna(X.mean())  # You can use other imputation methods as needed

# Handle infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.dropna()

# Check and convert data types
X = X.astype('float64')
y = y.astype('float64')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) as a measure of model performance
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Add predicted P/L to the DataFrame
df['pnl_better_model'] = model.predict(X)

# Display the first few rows of the modified dataset
print(df.head())


# In[29]:


# Calculate cumulative P/L
df['cumulative_pnl_better'] = df['pnl_better_model'].cumsum()


# In[36]:


# Check if the denominator is zero before calculating Sharpe Ratio
if df['pnl_better_model'].std() != 0:
    sharpe_ratio_better = df['pnl_better_model'].mean() / df['pnl_better_model'].std()
else:
    sharpe_ratio_better = np.nan  # Set to NaN or handle it in a way that makes sense for your analysis
    
# Display Sharpe Ratio
print("Sharpe Ratio - Better Model:", sharpe_ratio_better)

# Calculate Drawdown
df['drawdown_better'] = df['cumulative_pnl_better'] - df['cumulative_pnl_better'].cummax()

# Plot cumulative P/L and Drawdown
plt.plot(df['cumulative_pnl_better'], label='Cumulative P/L - Better')
plt.plot(df['drawdown_better'], label='Drawdown - Better')
plt.legend()
plt.show()


# In[37]:


# Compare P/L
print("Absolute P/L - Base Model:", df['cumulative_pnl_base'].iloc[-1])
print("Absolute P/L - Better model:", df['cumulative_pnl_better'].iloc[-1])

# Compare Sharpe Ratio
print("Sharpe Ratio - Base Model:", sharpe_ratio_base)
print("Sharpe Ratio - Better Model:", sharpe_ratio_better)

# Compare Drawdown
print("Maximum Drawdown - Base Model:", df['drawdown_base'].min())
print("Maximum Drawdown - Better Model:", df['drawdown_better'].min())


# Absolute P/L: The better model has a significantly higher absolute profit/loss, indicating that it has generated higher returns compared to the base model.
# 
# Sharpe Ratio: The Sharpe Ratio measures the risk-adjusted return. A higher Sharpe Ratio is generally considered better. In this case, the better model has a much higher Sharpe Ratio, suggesting a better risk-adjusted performance.
# 
# Maximum Drawdown: The maximum drawdown represents the largest drop in P/L from a peak to a trough. A smaller maximum drawdown is generally preferred. In this case, the better model has a much smaller maximum drawdown, indicating better capital preservation during downturns.
# 
# Conclusion:
# Based on the provided metrics, the better model outperforms the base model in terms of absolute P/L, Sharpe Ratio, and maximum drawdown. Therefore, the better model appears to be more effective and robust in generating returns while managing risk.
