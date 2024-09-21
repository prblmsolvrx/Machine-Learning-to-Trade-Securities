import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datasets

# Load the dataset
data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])

# Create new features
tesla_df['5 Day Moving Avg'] = tesla_df['Close'].rolling(window=5).mean()
tesla_df['10 Day Moving Avg'] = tesla_df['Close'].rolling(window=10).mean()

# Define features and target
# Features include 5-day moving avg, 10-day moving avg, and Volume
features = tesla_df[['5 Day Moving Avg','10 Day Moving Avg']]
target = tesla_df['Close'].values

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# TODO: Split the dataset into training and testing sets, using 25% for testing and random_state of 42
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled,  # Scaled feature data
    target,           # Target data
    test_size=0.25,   # Proportion of the dataset to include in the test split
    random_state=42    # Seed used by the random number generator
)

# Verify splits
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"First 5 rows of training features: \n{X_train[:5]}")
print(f"First 5 training targets: {y_train[:5]}\n")
print(f"First 5 rows of testing features: \n{X_test[:5]}")
print(f"First 5 testing targets: {y_test[:5]}")

"""
Training features shape: (2510, 2)
Testing features shape: (837, 2)
First 5 rows of training features: 
[[-0.51193721 -0.51843385]
 [ 2.55368267  2.55831966]
 [ 1.4770418   1.48658913]
 [-0.5289848  -0.53073417]
 [-0.54460278 -0.54279135]]
First 5 training targets: [ 17.288    355.666656 222.419998  15.000667  13.092   ]

First 5 rows of testing features: 
[[-0.45004679 -0.44517433]
 [ 1.24275895  1.13622549]
 [-0.52939556 -0.52901219]
 [-0.51294676 -0.51331802]
 [ 2.66710672  2.69746393]]
First 5 testing targets: [ 23.209333 189.606674  14.730667  16.763332 325.733337]
"""