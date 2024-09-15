import pandas as pd
from datasets import load_dataset

# TODO: Load the Tesla (TSLA) dataset `codesignal/tsla-historic-prices` and create a DataFrame
dataset = load_dataset('codesignal/tsla-historic-prices')
df = pd.DataFrame(dataset['train'])

# TODO: Convert the 'Date' column to datetime type to handle date calculations
df['Date'] = pd.to_datetime(df['Date'])

# TODO: Set the converted 'Date' column as the DataFrame index to facilitate date-based querying
df.set_index('Date', inplace=True)

# TODO: Sort the DataFrame based on the date index to ensure it's ordered chronologically
df.sort_index(inplace=True)

# TODO: Filter the DataFrame to get data from January 2020 to March 2020 (Q1) and print the first few rows
q1_2020 = df.loc['2020-01-01':'2020-03-31']
print(q1_2020.head())
