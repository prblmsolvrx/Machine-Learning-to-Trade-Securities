# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import StandardScaler  # For feature scaling
from datasets import load_dataset  # To load datasets easily
from sklearn.model_selection import train_test_split  # To split the dataset into training and testing sets
from sklearn.ensemble import GradientBoostingRegressor  # For the regression model
from sklearn.metrics import mean_squared_error  # To evaluate the model's performance
import matplotlib.pyplot as plt  # For data visualization

# Load the Tesla stock price dataset from the 'codesignal' repository
tesla = load_dataset('codesignal/tsla-historic-prices')

# Convert the loaded dataset into a pandas DataFrame for easier manipulation
tesla_df = pd.DataFrame(tesla['train'])

# Convert the 'Date' column from string format to datetime format for proper handling
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Adding technical indicators for analysis
# Simple Moving Average (SMA) over 5 days
tesla_df['SMA_5'] = tesla_df['Adj Close'].rolling(window=5).mean()
# Simple Moving Average (SMA) over 10 days
tesla_df['SMA_10'] = tesla_df['Adj Close'].rolling(window=10).mean()
# Exponential Moving Average (EMA) over 5 days
tesla_df['EMA_5'] = tesla_df['Adj Close'].ewm(span=5, adjust=False).mean()
# Exponential Moving Average (EMA) over 10 days
tesla_df['EMA_10'] = tesla_df['Adj Close'].ewm(span=10, adjust=False).mean()

# Remove rows with NaN values that were created during the calculation of moving averages
tesla_df.dropna(inplace=True)

# Selecting features and target variable for the model
# Features include stock open, close, high, low prices, volume, and technical indicators
features = tesla_df[['Open', 'Close', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10']].values
# Target variable is the adjusted close price
target = tesla_df['Adj Close'].values

# Standardizing the feature values to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Splitting the dataset into training and testing sets
# 75% of data for training and 25% for testing, with a fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.25, random_state=42)

# Initialize the Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42)
# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)
# Calculate the Mean Squared Error to evaluate the model's performance
mse = mean_squared_error(y_test, predictions)
# Print the Mean Squared Error
print("Mean Squared Error:", mse)

# Visualizing the actual vs predicted values
plt.figure(figsize=(10, 6))  # Set the figure size for the plot
# Scatter plot for actual values
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7)
# Scatter plot for predicted values
plt.scatter(range(len(y_test)), predictions, label='Predicted', alpha=0.7)
# Set the title and labels for the plot
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
# Add a legend to the plot
plt.legend()
# Display the plot
plt.show()
