"""
 fill in the missing pieces of code to filter the Tesla stock data for Q4 of 2019.
 Follow the TODO comments and make sure you succeed.
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

# TODO: Filter data for TSLA prices only in Q4 of 2019 (Oct 2019 - Dec 2019)
# Print the first few rows
tesla_Q4_Oct_2019_Dec_2019 = tesla_df.loc['2019-10':'2019-12']
print(tesla_Q4_Oct_2019_Dec_2019.head())

"""
                 Open       High        Low      Close  Adj Close     Volume
Date                                                                        
2019-10-01  16.100000  16.396667  15.942000  16.312668  16.312668   92439000
2019-10-02  16.219334  16.309999  15.962000  16.208668  16.208668   84471000
2019-10-03  15.457333  15.632000  14.952000  15.535333  15.535333  226267500
2019-10-04  15.440667  15.652000  15.204667  15.428667  15.428667  119925000
2019-10-07  15.320000  15.904000  15.236667  15.848000  15.848000  120963000
"""