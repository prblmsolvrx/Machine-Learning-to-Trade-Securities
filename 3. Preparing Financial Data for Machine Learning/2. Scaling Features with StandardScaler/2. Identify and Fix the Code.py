# Importing the pandas library, which is used for data manipulation and analysis
import pandas as pd

# Importing StandardScaler from sklearn, which is a tool used to scale/standardize features 
# by removing the mean and scaling to unit variance (mean=0, variance=1)
from sklearn.preprocessing import StandardScaler

# Importing the datasets module, which is used to load datasets (e.g., from external sources)
import datasets

# Load the Tesla historical price dataset using the datasets module
# The 'codesignal/tsla-historic-prices' dataset is fetched, and the result is stored in 'data'
data = datasets.load_dataset('codesignal/tsla-historic-prices')

# Convert the 'train' portion of the dataset into a pandas DataFrame for easier manipulation
# The dataset is now in a table format, with rows and columns, where each row corresponds to a data point
tesla_df = pd.DataFrame(data['train'])

# Feature Engineering: Create a new column 'High-Low' by subtracting the 'Low' price from the 'High' price
# This new feature captures the daily price range (difference between the highest and lowest price of the day)
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']

# Feature Engineering: Create another new column 'Price-Open' by subtracting the 'Open' price from the 'Close' price
# This feature captures how the price changed from the opening of the day to the closing (whether it went up or down)
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# Define the features (independent variables) we want to use for analysis
# We're selecting the 'High-Low', 'Price-Open', and 'Volume' columns, which we expect to be important predictors
# Convert these columns to a NumPy array using the .values attribute for further processing
features = tesla_df[['High-Low', 'Price-Open', 'Volume']].values

# Create an instance of StandardScaler, which will standardize the feature data (mean=0, std=1)
# This is useful when working with machine learning algorithms that require normalized data
scaler = StandardScaler()

# Fit the scaler to the feature data and transform the features in one step using 'fit_transform'
# It calculates the mean and standard deviation for each feature and scales the data accordingly
# The transformed data is stored in 'features_scaled'
features_scaled = scaler.fit_transform(features)

# Display the first 5 rows of the scaled features to understand how the data looks after scaling
print("Scaled features (first 5 rows):\n", features_scaled[:5])

# Calculate the mean values of each feature after scaling using numpy's mean function
# The expected result is that the means will be approximately zero for each feature since we standardized the data
scaled_means = features_scaled.mean(axis=0)

# Calculate the standard deviations of each feature after scaling
# The expected result is that the standard deviations will be approximately 1 for each feature
scaled_stds = features_scaled.std(axis=0)

# Print the mean values of the scaled features (should be around 0)
print("\nMean values of scaled features:", scaled_means)

# Print the standard deviations of the scaled features (should be around 1)
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