"""
as a data analyst and need to filter Tesla's stock data for the years 2020 to 2022 to analyze
stock trends during that period. Your task is to identify and fix a bug in the given code.
The code is designed to filter Tesla's stock data for the years 2020 to 2022. Find what's
causing the issue and correct it.
"""

import pandas as pd
import datasets

# Load TSLA dataset
tesla_data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(tesla_data['train'])

# Convert the Date column to datetime type
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Set the Date column as the index
tesla_df.set_index('Date', inplace=True)

# Sort the DataFrame based on the index
tesla_df.sort_index(inplace=True)

# Filtering the dataset for the years 2020 to 2022
tesla_2020_2022 = tesla_df.loc['2020':'2022']
print(tesla_2020_2022.head())

"""
                 Open       High        Low      Close  Adj Close     Volume
Date                                                                        
2020-01-02  28.299999  28.713333  28.114000  28.684000  28.684000  142981500
2020-01-03  29.366667  30.266666  29.128000  29.534000  29.534000  266677500
2020-01-06  29.364668  30.104000  29.333332  30.102667  30.102667  151995000
2020-01-07  30.760000  31.441999  30.224001  31.270666  31.270666  268231500
2020-01-08  31.580000  33.232666  31.215334  32.809334  32.809334  467164500
"""