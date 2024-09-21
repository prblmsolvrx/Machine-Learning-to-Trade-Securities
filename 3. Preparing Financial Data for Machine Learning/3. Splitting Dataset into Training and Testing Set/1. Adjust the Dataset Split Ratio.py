"""
practice splitting our dataset. Change the test_size parameter in
the train_test_split function so that 30% of the data is used for testing.
"""
# Import pandas for data manipulation
import pandas as pd

# Import StandardScaler to scale features and train_test_split to split the dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import the datasets library to load the specific dataset
import datasets

# Load the Tesla stock historic prices dataset using datasets.load_dataset method
data = datasets.load_dataset('codesignal/tsla-historic-prices')

# Convert the dataset into a pandas DataFrame for easier manipulation
tesla_df = pd.DataFrame(data['train'])

# Create a new feature 'High-Low' which is the difference between the High and Low stock prices of each day
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']

# Create another new feature 'Price-Open' which is the difference between the Close and Open prices of each day
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# Define the feature matrix (independent variables) consisting of the new features and 'Volume' column
features = tesla_df[['High-Low', 'Price-Open', 'Volume']].values

# Define the target variable (dependent variable) as the 'Close' price
target = tesla_df['Close'].values

# Initialize the StandardScaler to standardize (scale) the features
scaler = StandardScaler()

# Fit the scaler to the features and then transform them into scaled values
features_scaled = scaler.fit_transform(features)

# Split the dataset into training and testing sets
# Set test_size to 0.30 to allocate 30% of the data to testing and 70% to training
# random_state ensures reproducibility of the random split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.30, random_state=42)

# Print the shape of the training features to verify the split
print(f"Training features shape: {X_train.shape}")

# Print the shape of the testing features to verify the split
print(f"Testing features shape: {X_test.shape}")

# Print the first 5 rows of the training features to inspect the scaled data
print(f"First 5 rows of training features: \n{X_train[:5]}")

# Print the first 5 rows of the training target values to inspect the training data
print(f"First 5 training targets: {y_train[:5]}\n")

# Print the first 5 rows of the testing features to inspect the scaled data used for testing
print(f"First 5 rows of testing features: \n{X_test[:5]}")

# Print the first 5 rows of the testing target values to inspect the testing data
print(f"First 5 testing targets: {y_test[:5]}")


"""
Training features shape: (2342, 3)
Testing features shape: (1005, 3)
First 5 rows of training features: 
[[-0.41033541 -0.17937145  2.11455992]
 [-0.48360134  0.02078491 -0.19365843]
 [-0.4941894  -0.0460456  -0.39334992]
 [-0.49796247 -0.04437057 -0.62251294]
 [-0.55832735  0.01006514 -1.04174218]]
First 5 training targets: [14.068667 15.899333 15.616    24.155333  2.175333]

First 5 rows of testing features: 
[[-0.36226203  0.2087143   0.69346624]
 [ 1.27319589  1.04049732  0.58204785]
 [-0.53556882 -0.03231093 -0.86874821]
 [-0.49029475  0.07773304 -0.51784526]
 [ 3.0026057  -4.41816938 -0.31923731]]
First 5 testing targets: [ 23.209333 189.606674  14.730667  16.763332 325.733337]
"""