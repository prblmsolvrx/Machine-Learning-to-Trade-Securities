# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler  # For standardizing features
from sklearn.ensemble import GradientBoostingRegressor  # For building a regression model
import matplotlib.pyplot as plt  # For visualizing feature importance
import datasets  # For loading datasets

# Load the Tesla historical price dataset from the specified dataset library
tesla = datasets.load_dataset('codesignal/tsla-historic-prices')

# Convert the dataset into a DataFrame for easier manipulation
tesla_df = pd.DataFrame(tesla['train'])

# Convert the 'Date' column to datetime type to facilitate time-based analysis
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Calculate the technical indicator 'Momentum_5'
# 'Momentum_5' represents the change in the adjusted close price over the past 5 days
tesla_df['Momentum_5'] = tesla_df['Adj Close'].diff(periods=5)

# Calculate the technical indicator 'Daily_Return'
# 'Daily_Return' represents the daily percentage change in the adjusted close price
tesla_df['Daily_Return'] = tesla_df['Adj Close'].pct_change() * 100

# Calculate the technical indicator 'High_Low_Diff'
# 'High_Low_Diff' is the difference between the highest and lowest prices of the day
tesla_df['High_Low_Diff'] = tesla_df['High'] - tesla_df['Low']

# Drop any rows containing NaN values generated by the indicators to ensure data integrity
tesla_df.dropna(inplace=True)

# Define the feature columns to be used for model training
features = ['Open', 'High', 'Low', 'Volume', 'Momentum_5', 'Daily_Return', 'High_Low_Diff']

# Extract the feature data (X) and target data (y) from the DataFrame
X = tesla_df[features]  # Features: Columns used for model prediction
y = tesla_df['Adj Close']  # Target: The 'Adj Close' column that we want to predict

# Split the dataset into training (80%) and testing (20%) sets using a fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a StandardScaler to standardize the feature data
scaler = StandardScaler()

# Fit the scaler to the training data and transform it to have a mean of 0 and a standard deviation of 1
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data using the same scaler (without fitting again) to ensure consistent scaling
X_test_scaled = scaler.transform(X_test)

# Initialize a Gradient Boosting Regressor model with a fixed random state for reproducibility
gbr = GradientBoostingRegressor(random_state=42)

# Train the model using the standardized training data and the corresponding target values
gbr.fit(X_train_scaled, y_train)

# Retrieve the feature importance scores from the trained model
feature_importances = gbr.feature_importances_

# Get the indices of the feature importances in descending order to sort them
sorted_indices = feature_importances.argsort()[::-1]

# Create a bar plot to visualize the importance of each feature
plt.figure(figsize=(10, 6))  # Set the figure size for better visualization

# Plot the feature importance values, ordered by their importance
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align='center')

# Set the x-ticks to be the feature names sorted by their importance and rotate them for better readability
plt.xticks(range(len(feature_importances)), [features[i] for i in sorted_indices], rotation=45)

# Set the x-axis label
plt.xlabel('Features')

# Set the y-axis label
plt.ylabel('Importance')

# Set the title of the plot
plt.title('Feature Importance in Gradient Boosting Regressor')

# Adjust the plot layout to ensure everything fits nicely
plt.tight_layout()

# Display the plot
plt.show()
