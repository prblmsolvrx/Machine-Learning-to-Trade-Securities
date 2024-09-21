"""
 Modify the starter code to create a new column called Daily_Change that
 calculates the difference between the closing price and the opening price.
 Don't forget to print this new column as well. Let's start our path to the stars!
"""
import pandas as pd
import datasets

# Load the dataset
data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])

# Creating the High-Low feature
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']
tesla_df['Daily_Change'] = tesla_df['Close'] - tesla_df['Open']

# Displaying the new features
print(tesla_df[['High-Low']].head())
print(tesla_df[['Daily_Change']].head())

"""
   High-Low
0  0.497334
1  0.474667
2  0.376667
3  0.292667
4  0.278000
   Daily_Change
0      0.326000
1     -0.130666
2     -0.202667
3     -0.253333
4     -0.259333
"""