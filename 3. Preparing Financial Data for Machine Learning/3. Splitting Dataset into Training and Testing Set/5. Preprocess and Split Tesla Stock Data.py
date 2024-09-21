import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datasets

# TODO: Load the dataset 'codesignal/tsla-historic-prices' and convert it to a DataFrame
data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])

# TODO: Create new features 'SMA20' (20-day Simple Moving Average) and 'EMA20' (20-day Exponential Moving Average)
tesla_df['SMA20'] = tesla_df['Close'].rolling(window=20).mean()  # 20-day SMA
tesla_df['EMA20'] = tesla_df['Close'].ewm(span=20, adjust=False).mean()  # 20-day EMA
# TODO: Drop NaN values that were created by moving average
tesla_df.dropna(inplace=True)
# TODO: Define features and target
# `features` include SMA20, EMA20, and Volume, `target` includes Close prices
features = tesla_df[['SMA20', 'EMA20', 'Volume']]
target = tesla_df['Close']

# TODO: Scale features using StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# TODO: Split the dataset into training and testing sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# TODO: Verify splits by printing shapes and sample rows of training and testing sets
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)
print("\nSample training features:\n", pd.DataFrame(X_train, columns=['SMA20', 'EMA20', 'Volume']).head())
print("\nSample training target:\n", y_train.head())

"""
Training features shape: (2662, 3)
Testing features shape: (666, 3)
Training target shape: (2662,)
Testing target shape: (666,)

Sample training features:
       SMA20     EMA20    Volume
0  1.011828  0.956769  1.408984
1 -0.511218 -0.509894  4.333917
2 -0.665218 -0.666568 -1.143018
3 -0.469139 -0.462270  1.177422
4 -0.439417 -0.443722  0.196877

Sample training target:
 3144    125.349998
2347     19.978666
75        1.383333
2007     23.503332
1833     22.862667
Name: Close, dtype: float64
"""