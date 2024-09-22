# Importing the pandas library for data manipulation and analysis
import pandas as pd

# Importing StandardScaler for scaling features to a standard normal distribution
from sklearn.preprocessing import StandardScaler

# Importing TimeSeriesSplit for splitting the data into train/test sets while maintaining the order of time series
from sklearn.model_selection import TimeSeriesSplit

# Importing load_dataset to load a dataset from the 'datasets' library
from datasets import load_dataset

# Load the Tesla stock historic prices dataset as a pandas DataFrame
tesla_df = load_dataset('codesignal/tsla-historic-prices', split='train').to_pandas()

# Feature Engineering: Creating new features based on existing columns
# Creating a 'High-Low' column representing the difference between the 'High' and 'Low' prices of the day
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']

# Creating a 'Price-Open' column representing the difference between the 'Close' and 'Open' prices of the day
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# Defining the feature set ('X') by selecting columns that represent important characteristics of the data
# Choosing 'High-Low', 'Price-Open', and 'Volume' columns as features
features = tesla_df[['High-Low', 'Price-Open', 'Volume']].values

# Defining the target variable ('y'), which in this case is the 'Close' price that we want to predict
target = tesla_df['Close'].values

# Initializing the StandardScaler to normalize the feature set
scaler = StandardScaler()

# Fitting the scaler to the features and transforming them to have a mean of 0 and standard deviation of 1
features_scaled = scaler.fit_transform(features)

# Initializing TimeSeriesSplit for splitting the data while respecting the time series order
# Setting n_splits=5 to create 5 different splits of training and test sets
tscv = TimeSeriesSplit(n_splits=5)

# Splitting the data using TimeSeriesSplit
for fold, (train_index, test_index) in enumerate(tscv.split(features_scaled)):
    # Printing the current fold number
    print(f"Fold {fold + 1}")
    
    # Displaying the first 5 indices of the training and test sets
    print(f"TRAIN indices (first 5): {train_index[:5]}, TEST indices (first 5): {test_index[:5]}")

    # Splitting the features into training and test sets based on the indices provided by TimeSeriesSplit
    X_train, X_test = features_scaled[train_index], features_scaled[test_index]

    # Splitting the target variable into training and test sets
    y_train, y_test = target[train_index], target[test_index]

    # Printing the first 2 samples of training features for a quick check
    print(f"X_train sample:\n {X_train[:2]}")
    
    # Printing the first 2 samples of training target values
    print(f"y_train sample:\n {y_train[:2]}")
    
    # Printing the first 2 samples of test features for a quick check
    print(f"X_test sample:\n {X_test[:2]}")
    
    # Printing the first 2 samples of test target values
    print(f"y_test sample:\n {y_test[:2]}")
    
    # Printing a separator for better readability between folds
    print("-" * 10)

    """
    Fold 1
TRAIN indices (first 5): [0 1 2 3 4], TEST indices (first 5): [839 840 841 842 843]
X_train sample:
 [[-0.48165383  0.08560547  2.29693712]
 [-0.48579183 -0.02912844  2.00292929]]
y_train sample:
 [1.592667 1.588667]
X_test sample:
 [[-0.4714307  -0.11890593  0.26304787]
 [-0.42092366  0.03234206  1.43036618]]
y_test sample:
 [10.857333 10.964667]
----------
Fold 2
TRAIN indices (first 5): [0 1 2 3 4], TEST indices (first 5): [1675 1676 1677 1678 1679]
X_train sample:
 [[-0.48165383  0.08560547  2.29693712]
 [-0.48579183 -0.02912844  2.00292929]]
y_train sample:
 [1.592667 1.588667]
X_test sample:
 [[-0.46169462 -0.13046308  1.57995793]
 [-0.47447336  0.07639316  0.32446706]]
y_test sample:
 [17.066    17.133333]
----------
Fold 3
TRAIN indices (first 5): [0 1 2 3 4], TEST indices (first 5): [2511 2512 2513 2514 2515]
X_train sample:
 [[-0.48165383  0.08560547  2.29693712]
 [-0.48579183 -0.02912844  2.00292929]]
y_train sample:
 [1.592667 1.588667]
X_test sample:
 [[-0.27268857 -0.19528365  0.41906266]
 [-0.34291165 -0.09059793 -0.01236106]]
y_test sample:
 [66.726669 66.288002]
----------
    """
