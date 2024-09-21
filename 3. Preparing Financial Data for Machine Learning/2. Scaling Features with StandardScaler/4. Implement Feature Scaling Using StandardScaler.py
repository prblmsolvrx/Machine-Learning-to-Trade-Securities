import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datasets

# Load the Tesla stock dataset
data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])

# Feature Engineering: creating new features
# TODO: Create a new feature with a value corresponding to a daily price change
tesla_df['Daily_Change'] = tesla_df['Close'] - tesla_df['Open']
# TODO: Create a new feature with a value equals to mean price during the day
tesla_df['Mean_Price'] = (tesla_df['High'] + tesla_df['Low']) / 2

# Defining features
features = tesla_df[['Daily_Change', 'Mean_Price', 'Volume', 'Open']].values

# Scaling the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Displaying the first few scaled features
print("Scaled features (first 5 rows):\n", features_scaled[:5])

# TODO: Check mean values and standard deviations of scaled features
scaled_means = np.mean(features_scaled, axis=0)
scaled_stds = np.std(features_scaled, axis=0)

print("\nMean values of scaled features:", scaled_means)
print("Standard deviations of scaled features:", scaled_stds)

"""
Scaled features (first 5 rows):
 [[ 0.08560547 -0.6637783   2.29693712 -0.66504963]
 [-0.02912844 -0.66005443  2.00292929 -0.66053177]
 [-0.04721815 -0.66256255  0.33325453 -0.66105741]
 [-0.0599476  -0.66402146 -0.23997882 -0.66238815]
 [-0.06145506 -0.6660133   0.08156432 -0.66438426]]

Mean values of scaled features: [ 5.57267607e-18  3.39667875e-17 -6.79335750e-17  6.79335750e-17]
Standard deviations of scaled features: [1. 1. 1. 1.]
"""