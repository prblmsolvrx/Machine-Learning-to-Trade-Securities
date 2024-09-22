import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import datasets

# Load the dataset 'codesignal/tsla-historic-prices' using `datasets` library and convert to a DataFrame
dataset = datasets.load_dataset('codesignal/tsla-historic-prices')
df = pd.DataFrame(dataset['train'])

# Engineer new features: 'Moving_Average_10' and 'Returns'
df['Moving_Average_10'] = df['Close'].rolling(window=10).mean()  # 10-day simple moving average
df['Returns'] = df['Close'] - df['Open']  # Daily return (close - open)

# Define features arrays with 'Moving_Average_10', 'Returns', and 'Volume', and the target array as 'Close'
features = df[['Moving_Average_10', 'Returns', 'Volume']].dropna()  # Removing NaN values caused by rolling window
target = df['Close'][df['Moving_Average_10'].notna()]  # Ensure alignment

# Scale the features using StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initiate TimeSeriesSplit with 3 splits
tscv = TimeSeriesSplit(n_splits=3)

# Print indices and samples for each fold
for fold, (train_index, test_index) in enumerate(tscv.split(features_scaled)):
    print(f"Fold {fold + 1}")
    print(f"TRAIN indices: {train_index}")
    print(f"TEST indices: {test_index}")
    print(f"TRAIN samples: {len(train_index)}")
    print(f"TEST samples: {len(test_index)}\n")
