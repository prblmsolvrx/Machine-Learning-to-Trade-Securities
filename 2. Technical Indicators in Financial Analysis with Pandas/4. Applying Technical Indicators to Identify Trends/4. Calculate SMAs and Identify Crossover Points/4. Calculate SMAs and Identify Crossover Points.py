# Importing the required libraries
import pandas as pd  # Pandas for data manipulation and analysis
import matplotlib.pyplot as plt  # Matplotlib for plotting the data
from datasets import load_dataset  # Function to load the dataset

# Load the Tesla historical prices dataset from CodeSignal using the load_dataset function
dataset = load_dataset('codesignal/tsla-historic-prices')

# Convert the loaded dataset (train set) into a Pandas DataFrame for easier manipulation
tesla_df = pd.DataFrame(dataset['train'])

# Convert the 'Date' column to datetime format to facilitate date-based operations
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Set the 'Date' column as the index of the DataFrame, which allows easier time series analysis
tesla_df.set_index('Date', inplace=True)

# Calculate the 50-day Simple Moving Average (SMA) of the 'Close' price
# The rolling() method creates a rolling window, and mean() computes the average over the window
tesla_df['SMA_50'] = tesla_df['Close'].rolling(window=50).mean()

# Calculate the 200-day Simple Moving Average (SMA) of the 'Close' price
tesla_df['SMA_200'] = tesla_df['Close'].rolling(window=200).mean()

# Create a 'Signal' column to identify market signals: 1 for "Golden Cross", -1 for "Death Cross"
tesla_df['Signal'] = 0  # Initialize the 'Signal' column with 0 (no signal by default)

# Set 'Signal' to 1 (Golden Cross) when the 50-day SMA is greater than the 200-day SMA
tesla_df.loc[tesla_df['SMA_50'] > tesla_df['SMA_200'], 'Signal'] = 1

# Set 'Signal' to -1 (Death Cross) when the 50-day SMA is less than the 200-day SMA
tesla_df.loc[tesla_df['SMA_50'] < tesla_df['SMA_200'], 'Signal'] = -1

# Create a new column 'Crossover' that stores the difference between consecutive 'Signal' values
# This helps in identifying the exact points where Golden or Death crosses happen
tesla_df['Crossover'] = tesla_df['Signal'].diff()

# For visualization, extract a smaller subset of the data (e.g., the year 2018) to avoid clutter in the plot
tesla_df_small = tesla_df.loc['2018']

# Create a plot to visualize the closing price and the two SMAs (50-day and 200-day) for the smaller data range
fig, ax = plt.subplots(figsize=(12, 6))  # Define the plot size
tesla_df_small[['Close', 'SMA_50', 'SMA_200']].plot(ax=ax, title="TSLA with Golden Cross and Death Cross (2018)")  # Plot the closing price and SMAs

# Identify the points where Golden and Death crosses occur (Crossover != 0 indicates a crossover)
crosses = tesla_df_small[tesla_df_small['Crossover'] != 0]

# Iterate over the rows where a crossover occurs
for idx, row in crosses.iterrows():
    # If the crossover is positive (Golden Cross), mark it with a green circle
    if row['Crossover'] == 1:
        plt.plot(idx, row['SMA_50'], 'go', markersize=10, label='Golden Cross' 
                 if 'Golden Cross' not in [text.get_text() for text in ax.get_legend().get_texts()] else "")
    # If the crossover is negative (Death Cross), mark it with a red circle
    elif row['Crossover'] == -1:
        plt.plot(idx, row['SMA_50'], 'ro', markersize=10, label='Death Cross' 
                 if 'Death Cross' not in [text.get_text() for text in ax.get_legend().get_texts()] else "")

# Add a legend to the plot that identifies the Golden Cross and Death Cross
plt.legend()

# Display the final plot with the SMAs and the crossover points
plt.show()
