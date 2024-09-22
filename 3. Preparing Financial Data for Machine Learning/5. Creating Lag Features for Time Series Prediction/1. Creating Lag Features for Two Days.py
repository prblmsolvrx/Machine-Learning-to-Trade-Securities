"""
create a new lag feature that captures the closing
price from two days ago instead of just one.
Change the lag from one day to two days.
"""

import pandas as pd
import datasets

# Load the dataset
data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])

# Create basic features
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# Create a lag feature
tesla_df['Close_lag2'] = tesla_df['Close'].shift(2)

# Drop NaN values resulting from the lag feature
tesla_df.dropna(inplace=True)

# Define features and the target
features = tesla_df[['Close_lag2', 'High-Low', 'Price-Open', 'Volume']].values
target = tesla_df['Close'].values

# Display the first 5 rows of features and target
print("Features (first 5 rows):\n", features[:5])
print("Target (first 5 rows):\n", target[:5])

"""
Features (first 5 rows):
 [[ 1.592667e+00  3.766670e-01 -2.026670e-01  1.232820e+08]
 [ 1.588667e+00  2.926670e-01 -2.533330e-01  7.709700e+07]
 [ 1.464000e+00  2.780000e-01 -2.593330e-01  1.030035e+08]
 [ 1.280000e+00  1.100000e-01 -4.000000e-02  1.038255e+08]
 [ 1.074000e+00  1.300000e-01  8.800000e-02  1.156710e+08]]
Target (first 5 rows):
 [1.464    1.28     1.074    1.053333 1.164   ]
"""