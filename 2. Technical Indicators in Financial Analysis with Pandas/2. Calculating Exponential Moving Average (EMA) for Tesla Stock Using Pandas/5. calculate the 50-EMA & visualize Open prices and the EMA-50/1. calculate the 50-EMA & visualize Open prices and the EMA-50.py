"""
calculate the 50-day Exponential Moving Average (EMA), and visualize
both the Open prices and the EMA-50 for the year 2018.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# Load the Tesla dataset
dataset = load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(dataset['train'])

# Convert the 'Date' column to datetime
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Set 'Date' as the index
tesla_df.set_index('Date', inplace=True)

# Sort the data by date
tesla_df.sort_index(inplace=True)

# Calculate the 50-day Exponential Moving Average (EMA) for the Open Price
tesla_df['50-day EMA'] = tesla_df['Open'].ewm(span=50, adjust=False).mean()

# Narrow the data to the year 2018 for better visualization
tesla_df_2018 = tesla_df.loc['2018']

# Plot the Open Price and 50-day EMA for 2018
plt.figure(figsize=(12, 6))
plt.plot(tesla_df_2018.index, tesla_df_2018['Open'], label='Open', color='blue', alpha=0.6)
plt.plot(tesla_df_2018.index, tesla_df_2018['50-day EMA'], label='50-day EMA', color='orange', linestyle='--')

# Add labels and title
plt.title('Tesla Open Price and 50-day EMA in 2018')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()
