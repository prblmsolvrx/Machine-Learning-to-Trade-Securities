import pandas as pd
from sklearn.preprocessing import StandardScaler
import datasets

# Load the dataset
data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])

# Feature Engineering: creating new features
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# Defining features
# Features include new columns and 'Volume' column
features = tesla_df[['High-Low', 'Price-Open', 'Volume']]

# TODO: Initialize the StandardScaler and scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Displaying the first few scaled features
print("Scaled features (first 5 rows):\n", features_scaled[:5])

# Checking mean values and standard deviations of scaled features
scaled_means = features_scaled.mean(axis=0)
scaled_stds = features_scaled.std(axis=0)

print("\nMean values of scaled features:", scaled_means)
print("Standard deviations of scaled features:", scaled_stds)

"""
Scaled features (first 5 rows):
 [[-0.48165383  0.08560547  2.29693712]
 [-0.48579183 -0.02912844  2.00292929]
 [-0.50368231 -0.04721815  0.33325453]
 [-0.51901702 -0.0599476  -0.23997882]
 [-0.52169457 -0.06145506  0.08156432]]

Mean values of scaled features: [ 3.39667875e-17  5.57267607e-18 -6.79335750e-17]
Standard deviations of scaled features: [1. 1. 1.]
"""