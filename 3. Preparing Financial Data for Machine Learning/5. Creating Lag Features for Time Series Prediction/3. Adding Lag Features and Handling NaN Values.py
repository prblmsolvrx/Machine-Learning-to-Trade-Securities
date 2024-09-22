import pandas as pd
import datasets

# Loading the dataset
data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])

# Creating basic features
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# TODO: Create a lag feature for the 'Close' column
tesla_df['Close_lag1'] = tesla_df['Close'].shift(1)
# TODO: Drop NaN values to clean the dataset
tesla_df.dropna(inplace=True)

# Defining features and the target
features = tesla_df[['Close_lag1', 'High-Low', 'Price-Open', 'Volume']].values
target = tesla_df['Close'].values

# Displaying features and target
print("Features (first 5 rows):\n", features[:5])
print("Target (first 5 rows):\n", target[:5])

"""
Features (first 5 rows):
 [[ 1.592667e+00  4.746670e-01 -1.306660e-01  2.578065e+08]
 [ 1.588667e+00  3.766670e-01 -2.026670e-01  1.232820e+08]
 [ 1.464000e+00  2.926670e-01 -2.533330e-01  7.709700e+07]
 [ 1.280000e+00  2.780000e-01 -2.593330e-01  1.030035e+08]
 [ 1.074000e+00  1.100000e-01 -4.000000e-02  1.038255e+08]]
Target (first 5 rows):
 [1.588667 1.464    1.28     1.074    1.053333]
"""