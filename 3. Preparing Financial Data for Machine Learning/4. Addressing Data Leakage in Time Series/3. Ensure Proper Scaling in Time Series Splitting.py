import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datasets import load_dataset

# Load Tesla stock data
dataset = load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(dataset['train'])

# Feature Engineering
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# Define features and target
features = tesla_df[['High-Low', 'Price-Open', 'Volume']].values
target = tesla_df['Close'].values

# Initialize the TimeSeriesSplit instance
tscv = TimeSeriesSplit(n_splits=3)

# Iterate over each fold
for fold, (train_index, test_index) in enumerate(tscv.split(features)):
    print(f"Fold {fold + 1}")
    print(f"TRAIN indices (first & last 5): {train_index[:5]}, {train_index[-5:]}")
    print(f"TEST indices (first 5): {test_index[:5]}")
    
    # Splitting the features and target
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    # Create a scaler for each fold to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Use the fitted scaler to transform the test data
    X_test_scaled = scaler.transform(X_test)
    
    # Print a small sample of the data
    print(f"X_train sample:\n {X_train_scaled[:2]}")
    print(f"y_train sample:\n {y_train[:2]}")
    print(f"X_test sample:\n {X_test_scaled[:2]}")
    print(f"y_test sample:\n {y_test[:2]}")
    print("-" * 10)

"""
Fold 1
TRAIN indices (first & last 5): [0 1 2 3 4], [834 835 836 837 838]
TEST indices (first 5): [839 840 841 842 843]
X_train sample:
 [[ 2.2081346   2.59695874  3.5950461 ]
 [ 2.06817287 -1.03939573  3.23781582]]
y_train sample:
 [1.592667 1.588667]
X_test sample:
 [[ 2.55391733 -3.8847862   1.1237962 ]
 [ 4.26224455  0.90883843  2.54213068]]
y_test sample:
 [10.857333 10.964667]
----------
Fold 2
TRAIN indices (first & last 5): [0 1 2 3 4], [1670 1671 1672 1673 1674]
TEST indices (first 5): [1675 1676 1677 1678 1679]
X_train sample:
 [[ 0.66114297  1.33094544  3.37275239]
 [ 0.57784869 -0.52820042  3.00567805]]
y_train sample:
 [1.592667 1.588667]
X_test sample:
 [[ 1.06290456 -2.17022486  2.47759029]
 [ 0.80567969  1.18166938  0.91008616]]
y_test sample:
 [17.066    17.133333]
----------
Fold 3
TRAIN indices (first & last 5): [0 1 2 3 4], [2506 2507 2508 2509 2510]
TEST indices (first 5): [2511 2512 2513 2514 2515]
X_train sample:
 [[-0.06867952  0.60592638  2.25902771]
 [-0.10005754 -0.2559031   1.97890554]]
y_train sample:
 [1.592667 1.588667]
X_test sample:
 [[ 1.51588348 -1.50398624  0.46984314]
 [ 0.98338879 -0.71763426  0.05879508]]
y_test sample:
 [66.726669 66.288002]
----------
"""