import pandas as pd
import datasets
import matplotlib.pyplot as plt

# Load TSLA dataset
tesla_data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(tesla_data['train'])

# Convert 'Date' to datetime format and set as index
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])
tesla_df.set_index('Date', inplace=True)

# Sort the dataset by date
tesla_df.sort_index(inplace=True)

# Calculate the 20-day Simple Moving Average for the Close Price
tesla_df['SMA_10'] = tesla_df['Close'].rolling(window=10).mean()

# Using a smaller date range for better visualization
tesla_df_small = tesla_df.loc['2018']

# Plotting
tesla_df_small[['Close', 'SMA_10']].plot(figsize=(12, 6), title="TSLA Close Price and 10-day SMA (2018)")
plt.show()

# Print first few rows of the dataframe to check the SMA calculations
print(tesla_df_small.head())

"""
                 Open       High        Low  ...  Adj Close     Volume     SMA_10
Date                                         ...                                 
2018-01-02  20.799999  21.474001  20.733334  ...  21.368668   65283000  21.546533
2018-01-03  21.400000  21.683332  21.036667  ...  21.150000   67822500  21.402400
2018-01-04  20.858000  21.236668  20.378668  ...  20.974667  149194500  21.292533
2018-01-05  21.108000  21.149332  20.799999  ...  21.105333   68868000  21.209867
2018-01-08  21.066668  22.468000  21.033333  ...  22.427334  147891000  21.241533

[5 rows x 7 columns]
"""