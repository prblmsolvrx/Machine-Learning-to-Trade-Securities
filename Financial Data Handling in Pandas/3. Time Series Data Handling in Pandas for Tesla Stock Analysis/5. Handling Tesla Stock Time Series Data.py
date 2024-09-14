"""
 Fetch Tesla's stock data, convert the date to datetime,
 set it as the index, and sort the data by date. Finally,
 display the first few rows.
"""

import pandas as pd
import datasets

# TODO: Load the Tesla stock dataset and save it as a DataFrame
tesla_data = datasets.load_dataset('codesignal/tsla-historic-prices')

# TODO: Convert the 'Date' column to datetime type
tesla_df = pd.DataFrame(tesla_data['train']) # Training data extracted
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# TODO: Set the 'Date' column as the index
tesla_df.set_index('Date', inplace=True)

# TODO: Sort the DataFrame based on the index
tesla_df.sort_index(ascending=True, inplace=True)

# TODO: Display the first few rows to verify the changes
print(tesla_df.head())

"""
               Open      High       Low     Close  Adj Close     Volume
Date                                                                    
2010-06-29  1.266667  1.666667  1.169333  1.592667   1.592667  281494500
2010-06-30  1.719333  2.028000  1.553333  1.588667   1.588667  257806500
2010-07-01  1.666667  1.728000  1.351333  1.464000   1.464000  123282000
2010-07-02  1.533333  1.540000  1.247333  1.280000   1.280000   77097000
2010-07-06  1.333333  1.333333  1.055333  1.074000   1.074000  103003500
"""