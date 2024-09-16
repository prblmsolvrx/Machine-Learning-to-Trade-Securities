import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# Load Tesla dataset
dataset = load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(dataset['train'])

# Convert the 'Date' column to datetime
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Set 'Date' as index and sort
tesla_df.set_index('Date', inplace=True)
tesla_df.sort_index(inplace=True)

# TODO: Calculate the 200-day Exponential Moving Average for the Open Price
tesla_df['EMA_200'] = tesla_df['Open'].ewm(span=200 ,adjust=False).mean()

# Display the first few rows to verify
print(tesla_df[['Open', 'EMA_200']].head())

# Using a smaller date range for better visualization
tesla_df_small = tesla_df.loc['2018']

# Plotting
tesla_df_small[['Open', 'EMA_200']].plot(figsize=(12, 6), title="TSLA Open Price and 200-day EMA (2018)")
plt.show()

"""
                Open   EMA_200
Date                          
2010-06-29  1.266667  1.266667
2010-06-30  1.719333  1.271171
2010-07-01  1.666667  1.275106
2010-07-02  1.533333  1.277676
2010-07-06  1.333333  1.278230
"""