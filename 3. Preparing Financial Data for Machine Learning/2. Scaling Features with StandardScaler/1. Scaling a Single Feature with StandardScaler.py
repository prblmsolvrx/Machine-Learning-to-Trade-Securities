"""
let's modify our code to scale only the Volume feature using StandardScaler
and include this scaled feature in our dataset as a new column Volume_Scaled.
This will help you understand feature scaling on a single column.
"""

import pandas as pd
import datasets
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])

# Feature Engineering: creating new features
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# Defining the 'Volume' feature
volume_feature = tesla_df[['Volume']].values

# Scaling the 'Volume' feature
scaler = StandardScaler()
volume_scaled = scaler.fit_transform(volume_feature)

# Adding the scaled 'Volume' as a new column to the DataFrame
tesla_df['Volume_Scaled'] = volume_scaled

# Displaying the first few rows of the updated DataFrame
print("Updated DataFrame with scaled Volume (first 5 rows):\n", tesla_df[['Volume', 'Volume_Scaled']].head())

# Checking the mean and standard deviation of the scaled 'Volume'
scaled_mean = tesla_df['Volume_Scaled'].mean()
scaled_std = tesla_df['Volume_Scaled'].std()

print("\nMean of scaled 'Volume':", scaled_mean)
print("Standard deviation of scaled 'Volume':", scaled_std)

"""
Updated DataFrame with scaled Volume (first 5 rows):
       Volume  Volume_Scaled
0  281494500       2.296937
1  257806500       2.002929
2  123282000       0.333255
3   77097000      -0.239979
4  103003500       0.081564

Mean of scaled 'Volume': -6.793357497556978e-17
Standard deviation of scaled 'Volume': 1.0001494209944837
"""