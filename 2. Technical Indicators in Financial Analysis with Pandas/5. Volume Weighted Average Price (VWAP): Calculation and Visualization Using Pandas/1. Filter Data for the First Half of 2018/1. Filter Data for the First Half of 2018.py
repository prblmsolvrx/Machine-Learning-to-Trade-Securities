import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load Tesla dataset
dataset = load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(dataset['train'])

# Convert Date column to datetime format and set as index
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])
tesla_df.set_index('Date', inplace=True)

# Filter data for the year 2018
tesla_df_small = tesla_df.loc['2018'].copy()

# Calculating VWAP using np.cumsum

# Initialize cumulative sum variables
cumulative_price_volume = 0
cumulative_volume = 0
vwap_list = []

# Iterate over each row and calculate cumulative sums
for index, row in tesla_df_small.iterrows():
    cumulative_price_volume += row['Close'] * row['Volume']
    cumulative_volume += row['Volume']
    vwap = cumulative_price_volume / cumulative_volume
    vwap_list.append(vwap)

tesla_df_small['VWAP'] = vwap_list

# Visualize VWAP with Close Price
tesla_df_small[['Close', 'VWAP']].plot(figsize=(12, 6), title="TSLA Close Price and VWAP (2018)")
plt.show()