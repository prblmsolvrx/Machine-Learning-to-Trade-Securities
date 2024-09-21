"""
Try running the code and see if you can spot and fix the error.
Remember to verify the shapes of the resulting datasets to ensure the split is correct.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datasets

# Load dataset
data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])

# Feature engineering
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# Define features and target
features = tesla_df[['High-Low', 'Price-Open', 'Volume']].values
target = tesla_df['Close'].values

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.25, random_state=42)
# Verify splits
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

print(f"First 5 rows of training features:\n{X_train[:5]}")
print(f"First 5 training targets: {y_train[:5]}\n")

print(f"First 5 rows of testing features:\n{X_test[:5]}")
print(f"First 5 testing targets: {y_test[:5]}")

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