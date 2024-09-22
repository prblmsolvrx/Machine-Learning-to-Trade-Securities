import pandas as pd
import datasets

# TODO: Load the Tesla stock dataset 'codesignal/tsla-historic-prices' using the `datasets` library
dataset = datasets.load_dataset('codesignal/tsla-historic-prices')
df = pd.DataFrame(dataset['train'])
# TODO: Add basic features 'Daily_Return' and 'Volatility' to the DataFrame
# 'Daily_Return' is the difference between the Close price and the Open price
df['Daily_Return'] = df['Close'] - df['Open']
# 'Volatility' is the difference between the highest and the lowest price
df['Volatility'] = df['High'] - df['Low']
# TODO: Create a new lag feature 'Volume_lag1' for the 'Volume' column shifted by one day
df['Volume_lag1'] = df['Volume'].shift(1)
# TODO: Handle NaN values resulting from lag feature creation
df.dropna(inplace=True)
# TODO: Define the features and target variables for the dataset
# Features will include 'Volume_lag1', 'Daily_Return', and 'Volatility'
# The target will include 'Close' prices
features = df[['Volume_lag1', 'Daily_Return', 'Volatility']]
target = df['Close']

# TODO: Print the first 5 rows of features and target variables to verify
print("Features:\n", features.head())
print("\nTarget:\n", target.head())

"""
Features:
    Volume_lag1  Daily_Return  Volatility
1  281494500.0     -0.130666    0.474667
2  257806500.0     -0.202667    0.376667
3  123282000.0     -0.253333    0.292667
4   77097000.0     -0.259333    0.278000
5  103003500.0     -0.040000    0.110000

Target:
 1    1.588667
2    1.464000
3    1.280000
4    1.074000
5    1.053333
Name: Close, dtype: float64
"""