import pandas as pd
import datasets
from sklearn.preprocessing import StandardScaler

# Load the Tesla dataset using the `datasets.load_dataset`
tesla_dataset = datasets.load_dataset('codesignal/tsla-historic-prices')

# Convert the dataset to a pandas DataFrame
tesla_df = pd.DataFrame(tesla_dataset['train'])

# Create new features: 'Volatility' and 'Daily_Average'
tesla_df['Volatility'] = (tesla_df['High'] - tesla_df['Low']) / tesla_df['Open']
tesla_df['Daily_Average'] = (tesla_df['High'] + tesla_df['Low']) / 2

# Define the features from the DataFrame
features = tesla_df[['Volatility', 'Daily_Average', 'Volume']]

# Initialize the StandardScaler and scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert the scaled features back to a DataFrame for easy viewing
scaled_df = pd.DataFrame(scaled_features, columns=['Volatility', 'Daily_Average', 'Volume'])

# Print the first 5 rows of the scaled features
print("Scaled features (first 5 rows):")
print(scaled_df.head())

# Check and print the mean values and standard deviations of the scaled features
print("\nMeans of scaled features:", scaler.mean())
print("Standard deviations of scaled features:", scaler.scale())
