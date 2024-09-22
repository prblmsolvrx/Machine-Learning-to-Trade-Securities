import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datasets import load_dataset

# Load the dataset
tesla_df = load_dataset('codesignal/tsla-historic-prices', split='train').to_pandas()

# Feature Engineering: creating new features
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# Defining features and target
features = tesla_df[['High-Low', 'Price-Open', 'Volume']].values
target = tesla_df['Close'].values

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)

# Splitting with TimeSeriesSplit
for fold, (train_index, test_index) in enumerate(tscv.split(features)):
    print(f"Fold {fold + 1}")
    print(f"TRAIN indices (first 5): {train_index[:5]}, TEST indices (first 5): {test_index[:5]}")
    
    # Split into features and target for the current fold
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    # Create a scaler for each fold to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Use the fitted scaler to transform the test data
    X_test_scaled = scaler.transform(X_test)
    
    # Print a small sample of the data
    print(f"X_train sample:\n {X_train[:2]}")
    print(f"y_train sample:\n {y_train[:2]}")
    print(f"X_test sample:\n {X_test[:2]}")
    print(f"y_test sample:\n {y_test[:2]}")
    print("-" * 10)

"""
Fold 1
TRAIN indices (first 5): [0 1 2 3 4], TEST indices (first 5): [839 840 841 842 843]
X_train sample:
 [[ 4.973340e-01  3.260000e-01  2.814945e+08]
 [ 4.746670e-01 -1.306660e-01  2.578065e+08]]
y_train sample:
 [1.592667 1.588667]
X_test sample:
 [[ 5.533340e-01 -4.880000e-01  1.176255e+08]
 [ 8.300000e-01  1.140000e-01  2.116755e+08]]
y_test sample:
 [10.857333 10.964667]
----------
Fold 2
TRAIN indices (first 5): [0 1 2 3 4], TEST indices (first 5): [1675 1676 1677 1678 1679]
X_train sample:
 [[ 4.973340e-01  3.260000e-01  2.814945e+08]
 [ 4.746670e-01 -1.306660e-01  2.578065e+08]]
y_train sample:
 [1.592667 1.588667]
X_test sample:
 [[ 6.06666e-01 -5.34000e-01  2.23728e+08]
 [ 5.36667e-01  2.89333e-01  1.22574e+08]]
y_test sample:
 [17.066    17.133333]
----------
Fold 3
TRAIN indices (first 5): [0 1 2 3 4], TEST indices (first 5): [2511 2512 2513 2514 2515]
X_train sample:
 [[ 4.973340e-01  3.260000e-01  2.814945e+08]
 [ 4.746670e-01 -1.306660e-01  2.578065e+08]]
y_train sample:
 [1.592667 1.588667]
X_test sample:
 [[ 1.641998e+00 -7.920000e-01  1.301955e+08]
 [ 1.257332e+00 -3.753280e-01  9.543600e+07]]
y_test sample:
 [66.726669 66.288002]
----------
"""
