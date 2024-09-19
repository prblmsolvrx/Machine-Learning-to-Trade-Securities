# Import pandas library as 'pd' for data manipulation and analysis
import pandas as pd

# Import 'load_dataset' function from the 'datasets' library to load the dataset
from datasets import load_dataset

# Import 'pyplot' module from matplotlib and alias it as 'plt' for visualization
import matplotlib.pyplot as plt

# Load the Tesla historical prices dataset from the 'codesignal/tsla-historic-prices' source
dataset = load_dataset('codesignal/tsla-historic-prices')

# Convert the 'train' split of the loaded dataset into a pandas DataFrame
tesla_df = pd.DataFrame(dataset['train'])

# Convert the 'Date' column to datetime format to handle dates properly
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Set the 'Date' column as the DataFrame index, so it's used for data access and filtering
tesla_df.set_index('Date', inplace=True)

# Filter the DataFrame to include only the data from the year 2018, and create a copy of this filtered data
tesla_df_small = tesla_df.loc['2018'].copy()

# Initialize cumulative variables to calculate VWAP (Volume Weighted Average Price)
# 'cumulative_price_volume' will store the running total of (price * volume)
cumulative_price_volume = 0
# 'cumulative_volume' will store the running total of the volume
cumulative_volume = 0
# 'vwap_list' will store the VWAP values calculated for each row
vwap_list = []

# Loop through each row in the filtered DataFrame (tesla_df_small)
for index, row in tesla_df_small.iterrows():
    # Update the cumulative price-volume sum using the 'Close' price and 'Volume' of the current row
    cumulative_price_volume += row['Close'] * row['Volume']
    # Update the cumulative volume sum by adding the 'Volume' of the current row
    cumulative_volume += row['Volume']
    # Calculate the VWAP as the ratio of cumulative price-volume sum to cumulative volume sum
    vwap = cumulative_price_volume / cumulative_volume
    # Append the calculated VWAP value to the 'vwap_list'
    vwap_list.append(vwap)

# Add the VWAP values as a new column 'VWAP' to the filtered DataFrame
tesla_df_small['VWAP'] = vwap_list

# Plot the 'Close' price and 'VWAP' values from the DataFrame in a line chart
# Set the plot size to 12x6 inches and add a title to the chart
tesla_df_small[['Close', 'VWAP']].plot(figsize=(12, 6), title="TSLA Close Price and VWAP (2018)")

# Display the plot
plt.show()
