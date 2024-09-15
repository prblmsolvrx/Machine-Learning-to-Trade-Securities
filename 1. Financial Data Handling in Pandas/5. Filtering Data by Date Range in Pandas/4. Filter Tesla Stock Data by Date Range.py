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

# TODO: Filter the dataset for the years 2018, 2019, 2020
tesla_2018_2019_2020 = tesla_df.loc['2018':'2020']

print(tesla_2018_2019_2020.head())

"""
                 Open       High        Low      Close  Adj Close     Volume
Date                                                                        
2018-01-02  20.799999  21.474001  20.733334  21.368668  21.368668   65283000
2018-01-03  21.400000  21.683332  21.036667  21.150000  21.150000   67822500
2018-01-04  20.858000  21.236668  20.378668  20.974667  20.974667  149194500
2018-01-05  21.108000  21.149332  20.799999  21.105333  21.105333   68868000
2018-01-08  21.066668  22.468000  21.033333  22.427334  22.427334  147891000
"""