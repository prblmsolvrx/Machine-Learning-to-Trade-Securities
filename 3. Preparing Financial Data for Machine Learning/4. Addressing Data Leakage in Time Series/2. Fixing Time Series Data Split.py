import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datasets import load_dataset

# Load the Tesla stock dataset from the Hugging Face Datasets library
tesla_dataset = load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(tesla_dataset['train'])

# Feature Engineering: Creating new features based on existing data
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']  # Difference between high and low prices
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']  # Difference between closing and opening prices

# Defining the feature matrix and target vector
features = tesla_df[['High-Low', 'Price-Open', 'Volume']].values  # Features to be used for training
target = tesla_df['Close'].values  # The target variable we want to predict

# Scaling the feature matrix to normalize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)  # Scaled features with mean 0 and standard deviation 1

# Defining the TimeSeriesSplit object with 3 splits
tscv = TimeSeriesSplit(n_splits=3)

# Iteratively splitting the data using the TimeSeriesSplit object
for fold, (train_index, test_index) in enumerate(tscv.split(features_scaled)):
    print(f"Fold {fold + 1}")
    print(f"TRAIN indices (first 5): {train_index[:5]}, TEST indices (first 5): {test_index[:5]}")
    
    # Extracting training and testing data based on the indices
    X_train, X_test = features_scaled[train_index], features_scaled[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    # Displaying small samples of the split data
    print(f"X_train sample:\n {X_train[:2]}")
    print(f"y_train sample:\n {y_train[:2]}")
    print(f"X_test sample:\n {X_test[:2]}")
    print(f"y_test sample:\n {y_test[:2]}")
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
