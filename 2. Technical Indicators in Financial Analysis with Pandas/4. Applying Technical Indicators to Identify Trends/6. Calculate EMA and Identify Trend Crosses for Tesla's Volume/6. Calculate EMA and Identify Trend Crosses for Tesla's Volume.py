# Import the pandas library, which is used for data manipulation and analysis
import pandas as pd

# Import the pyplot module from matplotlib, used for data visualization
import matplotlib.pyplot as plt

# Import the load_dataset function from the datasets library to access predefined datasets
from datasets import load_dataset

# Load the Tesla historical prices dataset from CodeSignal's dataset repository
dataset = load_dataset('codesignal/tsla-historic-prices')

# Convert the dataset's 'train' split into a pandas DataFrame for easier manipulation
tesla_df = pd.DataFrame(dataset['train'])

# Convert the 'Date' column from string to datetime format so it can be used for time-based operations
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Set the 'Date' column as the DataFrame's index to allow for time series analysis
tesla_df.set_index('Date', inplace=True)

# TODO: Calculate the 20-day and 50-day Simple Moving Averages (SMA) for the 'Volume' column
# Create a new column 'SMA_20' with the 20-day rolling mean of the 'Volume' column
tesla_df['SMA_20'] = tesla_df['Volume'].rolling(window=20).mean()

# Create a new column 'SMA_50' with the 50-day rolling mean of the 'Volume' column
tesla_df['SMA_50'] = tesla_df['Volume'].rolling(window=50).mean()

# TODO: Identify "Golden Cross" (SMA_20 crossing above SMA_50) and "Death Cross" (SMA_20 crossing below SMA_50) for the 'Volume' column
# Assign 1 (Golden Cross) to the 'Signal' column when SMA_20 is greater than SMA_50
tesla_df.loc[tesla_df['SMA_20'] > tesla_df['SMA_50'], 'Signal'] = 1

# Assign -1 (Death Cross) to the 'Signal' column when SMA_20 is less than SMA_50
tesla_df.loc[tesla_df['SMA_20'] < tesla_df['SMA_50'], 'Signal'] = -1

# TODO: Filter the DataFrame to only include data for the year 2019
# Filter rows in the DataFrame where the index (Date) belongs to 2019
tesla_df_small = tesla_df.loc['2019']

# TODO: Plot the Volume along with the SMAs for the year 2019
# Create a new figure for the plot with dimensions 12x8 inches
plt.figure(figsize=(12, 8))

# Plot the 'Volume' column for 2019 with partial transparency
plt.plot(tesla_df_small.index, tesla_df_small['Volume'], label='Volume', alpha=0.5)

# Plot the 20-day SMA (SMA_20) in orange for the same period
plt.plot(tesla_df_small.index, tesla_df_small['SMA_20'], label='SMA 20', color='orange')

# Plot the 50-day SMA (SMA_50) in blue for the same period
plt.plot(tesla_df_small.index, tesla_df_small['SMA_50'], label='SMA 50', color='blue')

# TODO: Highlight the points where Golden Cross and Death Cross occurred
# Filter the rows where 'Signal' is 1 (Golden Cross) for 2019 and store them in golden_cross
golden_cross = tesla_df_small[(tesla_df_small['Signal'] == 1)]

# Filter the rows where 'Signal' is -1 (Death Cross) for 2019 and store them in death_cross
death_cross = tesla_df_small[(tesla_df_small['Signal'] == -1)]

# Mark the Golden Cross points on the plot with gold-colored upward triangles (marker '^') of size 100
plt.scatter(golden_cross.index, golden_cross['Volume'], label='Golden Cross', color='gold', marker='^', s=100)

# Mark the Death Cross points on the plot with red-colored downward triangles (marker 'v') of size 100
plt.scatter(death_cross.index, death_cross['Volume'], label='Death Cross', color='red', marker='v', s=100)

# Set the title of the plot to describe its contents
plt.title('Tesla Volume with SMA (2019) and Golden/Death Crosses')

# Label the x-axis as 'Date' and y-axis as 'Volume'
plt.xlabel('Date')
plt.ylabel('Volume')

# Display a legend to explain which lines and markers represent the Volume, SMA, Golden Cross, and Death Cross
plt.legend()

# Add a grid to the plot for easier readability of data points
plt.grid(True)

# Show the plot with all the customizations applied
plt.show()
