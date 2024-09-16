"""
calculate the 50-day EMA for the Volume
"""
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# Load Tesla dataset from the codesignal repository using the datasets library
dataset = load_dataset('codesignal/tsla-historic-prices')

# Convert the dataset into a pandas DataFrame for easier manipulation
tesla_df = pd.DataFrame(dataset['train'])

# Convert the 'Date' column from string format to a pandas datetime object
# This allows for easier time-series manipulation and analysis
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Set the 'Date' column as dthe index of the DataFrame
# This is useful for time-series operations like sorting and slicing by date
tesla_df.set_index('Date', inplace=True)

# Sort the DataFrame by the 'Date' index in ascending order
# This ensures the data is in chronological order, which is essential for time-series analysis
tesla_df.sort_index(inplace=True)

# Calculate the 50-day Exponential Moving Average (EMA) for the 'Volume' column
# EMA gives more weight to recent data points, providing a smoothed trend line
# span=50 means it's a 50-day moving average, adjust=False ensures no bias at the start
tesla_df['50-day EMA Volume'] = tesla_df['Volume'].ewm(span=50, adjust=False).mean()

# Select a smaller date range for better visualization, limiting data to the year 2018
# This reduces clutter in the plot, making it easier to see the trends
tesla_df_small = tesla_df.loc['2018']

# Create a plot to visualize the original 'Volume' data and its 50-day EMA for the year 2018
plt.figure(figsize=(10, 6))  # Set the figure size for better visibility

# Plot the original 'Volume' data in blue, which represents the raw trading volume for Tesla
plt.plot(tesla_df_small.index, tesla_df_small['Volume'], label='Volume', color='blue')

# Plot the 50-day EMA of the 'Volume' in orange, which smooths out the volume data
plt.plot(tesla_df_small.index, tesla_df_small['50-day EMA Volume'], label='50-day EMA Volume', color='orange')

# Add a title to the plot to explain what the visualization represents
plt.title('Tesla Volume and 50-day EMA (2018)')

# Label the x-axis as 'Date' since it shows the timeline of trading days
plt.xlabel('Date')

# Label the y-axis as 'Volume' since it represents the trading volume
plt.ylabel('Volume')

# Add a legend to the plot to differentiate between the 'Volume' and the '50-day EMA Volume'
plt.legend()

# Enable the grid to make it easier to compare values and track trends
plt.grid(True)

# Tighten the layout to prevent the plot labels from overlapping with the edges
plt.tight_layout()

# Display the plot
plt.show()
