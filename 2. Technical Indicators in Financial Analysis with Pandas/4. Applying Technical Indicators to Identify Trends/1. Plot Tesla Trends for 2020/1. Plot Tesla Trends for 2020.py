# Import the pandas library for data manipulation and analysis, and alias it as 'pd'
import pandas as pd

# Import the pyplot module from matplotlib for data visualization and alias it as 'plt'
import matplotlib.pyplot as plt

# Import the 'load_dataset' function from the 'datasets' library to access preloaded datasets
from datasets import load_dataset

# Load the Tesla stock price dataset ('tsla-historic-prices') using load_dataset
dataset = load_dataset('codesignal/tsla-historic-prices')

# Convert the dataset into a pandas DataFrame for easy data manipulation
tesla_df = pd.DataFrame(dataset['train'])

# Convert the 'Date' column from a string to a datetime object for proper date handling
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Set the 'Date' column as the index of the DataFrame to make time-based operations easier
tesla_df.set_index('Date', inplace=True)

# Calculate the 20-day Simple Moving Average (SMA) for the 'Close' price and store it in a new column 'SMA_20'
tesla_df['SMA_20'] = tesla_df['Close'].rolling(window=20).mean()

# Calculate the 50-day Simple Moving Average (SMA) for the 'Close' price and store it in a new column 'SMA_50'
tesla_df['SMA_50'] = tesla_df['Close'].rolling(window=50).mean()

# Create a new column 'Signal' to identify trading signals:
# 1 represents a Golden Cross (SMA_20 > SMA_50), -1 represents a Death Cross (SMA_20 < SMA_50)
tesla_df['Signal'] = 0  # Initialize all signals as 0 (neutral)
tesla_df.loc[tesla_df['SMA_20'] > tesla_df['SMA_50'], 'Signal'] = 1  # Golden Cross
tesla_df.loc[tesla_df['SMA_20'] < tesla_df['SMA_50'], 'Signal'] = -1  # Death Cross

# Create a 'Crossover' column that marks the points where the signal changes (indicating crossovers)
tesla_df['Crossover'] = tesla_df['Signal'].diff()

# Select a smaller subset of the dataset for visualization (e.g., the year 2019)
tesla_df_small = tesla_df.loc['2019']

# Create a plot of the 'Close' price along with the 20-day and 50-day SMAs
fig, ax = plt.subplots(figsize=(12, 6))
tesla_df_small[['Close', 'SMA_20', 'SMA_50']].plot(ax=ax, title="TSLA with Golden Cross and Death Cross (2019)")

# Highlight the crossover points on the plot (Golden Cross and Death Cross)
crosses = tesla_df_small[tesla_df_small['Crossover'] != 0]  # Filter crossover points
for idx, row in crosses.iterrows():
    # Plot Golden Cross points (where 'SMA_20' crosses above 'SMA_50')
    if row['Crossover'] == 2:
        plt.plot(idx, row['SMA_20'], 'go', markersize=10, label='Golden Cross' if 'Golden Cross' not in [text.get_text() for text in ax.get_legend().get_texts()] else "")
    # Plot Death Cross points (where 'SMA_20' crosses below 'SMA_50')
    elif row['Crossover'] == -2:
        plt.plot(idx, row['SMA_20'], 'ro', markersize=10, label='Death Cross' if 'Death Cross' not in [text.get_text() for text in ax.get_legend().get_texts()] else "")

# Add a legend to the plot to label the crossover points
plt.legend()

# Display the plot
plt.show()
