import pandas as pd
import datasets

# Load the dataset
data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])

# TODO: Create the High-Low feature by subtracting the 'Low' price from the 'High' price
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']

# Create the Price-Open feature
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# Display the new features
print(tesla_df[['High-Low', 'Price-Open']].head())

"""
   High-Low  Price-Open
0  0.497334    0.326000
1  0.474667   -0.130666
2  0.376667   -0.202667
3  0.292667   -0.253333
4  0.278000   -0.259333
"""