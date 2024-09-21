# Import the pandas library for data manipulation and analysis
import pandas as pd

# Import StandardScaler for feature scaling from scikit-learn's preprocessing module
from sklearn.preprocessing import StandardScaler

# Import train_test_split for splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import the datasets module, likely from Hugging Face's `datasets` library, for loading datasets
import datasets

# -----------------------------------------------------------
# Loading and preprocessing the dataset (revision)
# -----------------------------------------------------------

# Load the 'tsla-historic-prices' dataset from CodeSignal using the `load_dataset` function
data = datasets.load_dataset('codesignal/tsla-historic-prices')

# Convert the 'train' split of the dataset into a pandas DataFrame for easier manipulation
tesla_df = pd.DataFrame(data['train'])

# Create a new column 'High-Low' by subtracting the 'Low' price from the 'High' price for each record
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']

# Create another new column 'Price-Open' by subtracting the 'Open' price from the 'Close' price for each record
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# -----------------------------------------------------------
# Defining features and target
# -----------------------------------------------------------

# Select the feature columns 'High-Low', 'Price-Open', and 'Volume' from the DataFrame
# `.values` converts the selected DataFrame columns into a NumPy array
features = tesla_df[['High-Low', 'Price-Open', 'Volume']].values

# Select the target column 'Close' from the DataFrame
# `.values` converts the target column into a NumPy array
target = tesla_df['Close'].values

# -----------------------------------------------------------
# Scaling
# -----------------------------------------------------------

# Initialize the StandardScaler, which standardizes features by removing the mean and scaling to unit variance
scaler = StandardScaler()

# Fit the scaler to the features and transform them
# This scales the feature data so that each feature has a mean of 0 and a standard deviation of 1
features_scaled = scaler.fit_transform(features)

# -----------------------------------------------------------
# Splitting the dataset
# -----------------------------------------------------------

# Split the scaled features and target into training and testing sets
# `test_size=0.25` reserves 25% of the data for testing, and 75% for training
# `random_state=42` ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled,  # Scaled feature data
    target,           # Target data
    test_size=0.25,   # Proportion of the dataset to include in the test split
    random_state=42    # Seed used by the random number generator
)

# -----------------------------------------------------------
# Verify splits
# -----------------------------------------------------------

# Print the shape of the training features to verify the number of samples and features
print(f"Training features shape: {X_train.shape}")

# Print the shape of the testing features to verify the number of samples and features
print(f"Testing features shape: {X_test.shape}")

# Print the first 5 rows of the training features to inspect the scaled feature values
print(f"First 5 rows of training features: \n{X_train[:5]}")

# Print the first 5 training target values to inspect the target data
print(f"First 5 training targets: {y_train[:5]}\n")

# Print the first 5 rows of the testing features to inspect the scaled feature values
print(f"First 5 rows of testing features: \n{X_test[:5]}")

# Print the first 5 testing target values to inspect the target data
print(f"First 5 testing targets: {y_test[:5]}")

# -----------------------------------------------------------
# Expected Output
# -----------------------------------------------------------

"""
Training features shape: (2510, 3)
Testing features shape: (837, 3)
First 5 rows of training features: 
[[-4.66075964e-01  6.80184955e-02  3.11378946e-01]
 [ 4.01701510e+00  5.04529577e+00 -4.61555718e-02]
 [ 2.04723437e+00  3.09900603e+00  9.43022378e-04]
 [-5.30579018e-01 -2.30986178e-02 -5.67163058e-01]
 [-4.78854883e-01 -5.79376618e-02 -6.94451021e-01]]
First 5 training targets: [ 17.288    355.666656 222.419998  15.000667  13.092   ]

First 5 rows of testing features: 
[[-0.36226203  0.2087143   0.69346624]
 [ 1.27319589  1.04049732  0.58204785]
 [-0.53556882 -0.03231093 -0.86874821]
 [-0.49029475  0.07773304 -0.51784526]
 [ 3.0026057  -4.41816938 -0.31923731]]
First 5 testing targets: [ 23.209333 189.606674  14.730667  16.763332 325.733337]
"""
