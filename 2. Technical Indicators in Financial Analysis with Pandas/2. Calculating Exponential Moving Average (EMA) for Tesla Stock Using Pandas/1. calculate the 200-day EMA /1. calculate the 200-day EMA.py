# Importing the pandas library and giving it the alias 'pd' to work with data frames
import pandas as pd

# Importing the 'pyplot' module from the 'matplotlib' library and assigning it the alias 'plt'
# This will be used for plotting data visualizations
import matplotlib.pyplot as plt

# Importing the 'load_dataset' function from the 'datasets' library
# This will allow loading of the Tesla historic price dataset
from datasets import load_dataset

# Load the Tesla dataset using the 'load_dataset' function from the dataset library
dataset = load_dataset('codesignal/tsla-historic-prices')

# Convert the 'train' portion of the dataset to a pandas DataFrame for easier manipulation
tesla_df = pd.DataFrame(dataset['train'])

# Convert the 'Date' column from string format to a datetime object to facilitate time-based operations
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Set the 'Date' column as the index of the DataFrame to make it the reference for time series analysis
tesla_df.set_index('Date', inplace=True)

# Sort the data by the 'Date' index to ensure chronological order
tesla_df.sort_index(inplace=True)

# Calculate the 20-day Exponential Moving Average (EMA) for the 'Close' price and store it in a new column 'EMA_20'
tesla_df['EMA_200'] = tesla_df['Close'].ewm(span=200, adjust=False).mean()

# Print the first few rows of the DataFrame, showing the 'Close' and 'EMA_20' columns to verify calculations
print(tesla_df[['Close', 'EMA_200']].head())

# Narrow down the data to only include the year 2018 for better visualization in the plot
tesla_df_small = tesla_df.loc['2018']

# Plot the 'Close' price and the 20-day EMA for the year 2018 using matplotlib
# The plot size is set to 12x6 inches with the title "TSLA Close Price and 20-day EMA (2018)"
tesla_df_small[['Close', 'EMA_200']].plot(figsize=(12, 6), title="TSLA Close Price and 200-day EMA (2018)")

# Display the plot
plt.show()

"""
               Close   EMA_200
Date                          
2010-06-29  1.592667  1.592667
2010-06-30  1.588667  1.592627
2010-07-01  1.464000  1.591347
2010-07-02  1.280000  1.588249
2010-07-06  1.074000  1.583132
"""
