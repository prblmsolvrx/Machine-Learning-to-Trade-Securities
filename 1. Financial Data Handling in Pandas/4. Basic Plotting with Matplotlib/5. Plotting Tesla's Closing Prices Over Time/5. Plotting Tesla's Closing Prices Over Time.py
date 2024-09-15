import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

# TODO: Load Tesla's historic price dataset `codesignal/tsla-historic-prices` and create a DataFrame
tesla_data = load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(tesla_data['train'])

# TODO: Convert the 'Date' column to datetime
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# TODO: Set the 'Date' column as the index and sort it
tesla_df.set_index('Date',inplace=True)
tesla_df.sort_index(inplace=True)

# TODO: Create a figure of appropriate size
plt.figure(figsize = (10,5))

# TODO: Plot the 'Close' prices with custom color, line style, and width
plt.plot(tesla_df.index, tesla_df['Close'], color='blue',linestyle='-', linewidth=2)

# TODO: Add title and labels for the plot
plt.title('TSLA Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')

# TODO: Add a legend and grid to the plot
plt.legend(['Close Price'])
plt.grid(True)

# TODO: Display the plot
plt.show()