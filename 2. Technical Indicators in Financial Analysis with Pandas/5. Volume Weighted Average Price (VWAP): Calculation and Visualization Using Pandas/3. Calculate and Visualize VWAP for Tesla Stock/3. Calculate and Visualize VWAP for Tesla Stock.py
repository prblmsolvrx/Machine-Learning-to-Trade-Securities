import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load Tesla dataset
dataset = load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(dataset['train'])

# Convert Date column to datetime format and set as index
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])
tesla_df.set_index('Date', inplace=True)

# Filter data for the year 2018 and make a copy to avoid SettingWithCopyWarning
tesla_df_small = tesla_df.loc['2018'].copy()

# Calculate VWAP
tesla_df_small['VWAP'] = (np.cumsum(tesla_df_small['Volume'] * tesla_df_small['Open']) / 
                          np.cumsum(tesla_df_small['Volume']))

# TODO: Visualize VWAP with TSLA Open Prices
# Use Matplotlib to plot the VWAP and opening prices on the same graph
tesla_df_small[['Open','VWAP']].plot(figsize = (12,6), title = "TSLA opening Price and VWAP (2018)")
plt.show()
