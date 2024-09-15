"""
figure size to (14, 7) in the plt.figure() function, and modify the
plt.plot() function to plot the Close prices instead of the Open prices.
"""
# Import the 'pyplot' module from the matplotlib library for creating plots
import matplotlib.pyplot as plt

# Import the pandas library to handle data in a tabular format
import pandas as pd

# Import the 'load_dataset' function from the datasets library to load pre-built datasets
from datasets import load_dataset

# Load the TSLA dataset (Tesla historic prices) from the 'codesignal' dataset collection
tesla_data = load_dataset('codesignal/tsla-historic-prices')

# Convert the 'train' portion of the dataset into a pandas DataFrame for easier manipulation
tesla_df = pd.DataFrame(tesla_data['train'])

# Convert the 'Date' column in the DataFrame from string format to Python's datetime format
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Set the 'Date' column as the index of the DataFrame, allowing for time-based operations
# Inplace=True means the operation modifies the DataFrame directly
tesla_df.set_index('Date', inplace=True)

# Sort the DataFrame by the 'Date' index in ascending order to ensure chronological ordering
tesla_df.sort_index(inplace=True)

# Create a new figure for the plot with a custom size (width=12, height=6)
plt.figure(figsize=(14, 7))

# Plot the 'Open' price of Tesla stock over time using a green line, with specific line style and width
plt.plot(tesla_df.index, tesla_df['Close'], color='green', linestyle='-', linewidth=2)

# Add a title to the plot
plt.title('TSLA Closing Price Over Time')

# Label the x-axis (Date) and the y-axis (Price in USD)
plt.xlabel('Date')
plt.ylabel('Price (USD)')

# Add a legend to the plot, labeling the line as 'Open Price'
plt.legend(['Close Price'])

# Enable gridlines on the plot to make it easier to read data points
plt.grid(True)

# Display the final plot to the user
plt.show()
