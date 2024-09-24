# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import StandardScaler  # For standardizing data
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.ensemble import GradientBoostingRegressor  # For building the regression model
from sklearn.metrics import mean_squared_error  # For evaluating the model's performance
import matplotlib.pyplot as plt  # For visualizing data
import datasets  # For loading datasets

# Load TSLA (Tesla) dataset from CodeSignal's dataset library
tesla = datasets.load_dataset('codesignal/tsla-historic-prices')

# Convert the loaded dataset to a Pandas DataFrame for easier data manipulation
tesla_df = pd.DataFrame(tesla['train'])

# Convert the 'Date' column to a datetime type for better handling of date-related operations
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Create a new feature 'Previous_Close' by shifting the 'Adj Close' column by one row to get the previous day's close price
tesla_df['Previous_Close'] = tesla_df['Adj Close'].shift(1)

# Calculate the 'Day_Pct_Change' feature as the percentage change from the previous day's adjusted close to the current day's adjusted close
tesla_df['Day_Pct_Change'] = (tesla_df['Adj Close'] - tesla_df['Previous_Close']) / tesla_df['Previous_Close'] * 100

# Create a 'High_Low_Spread' feature as the difference between the high and low prices for the day
tesla_df['High_Low_Spread'] = tesla_df['High'] - tesla_df['Low']

# Create a 'Close_Prev_Close_Ratio' feature as the ratio of the current adjusted close to the previous adjusted close
tesla_df['Close_Prev_Close_Ratio'] = tesla_df['Adj Close'] / tesla_df['Previous_Close']

# Drop rows with NaN values, which result from the shift operation that created 'Previous_Close'
tesla_df.dropna(inplace=True)

# Define the features (independent variables) to be used in the model
features = ['Previous_Close', 'High_Low_Spread', 'Close_Prev_Close_Ratio']

# Define the target variable (dependent variable) to predict
target = 'Day_Pct_Change'

# Select the feature columns and target column from the DataFrame
X = tesla_df[features]  # Features (input data)
y = tesla_df[target]    # Target (output data)

# Instantiate a StandardScaler object for standardizing the features
scaler = StandardScaler()

# Fit the scaler to the features and transform them to have zero mean and unit variance
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets with 80% for training and 20% for testing
# `shuffle=False` ensures that the data remains in its original chronological order
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

# Instantiate the Gradient Boosting Regressor model
# Setting various parameters like:
# n_estimators = 1000 (maximum number of boosting iterations)
# learning_rate = 0.1 (step size for each iteration)
# max_depth = 3 (maximum depth of individual trees)
# validation_fraction = 0.1 (fraction of training data for validation to monitor early stopping)
# n_iter_no_change = 10 (number of iterations without improvement to trigger early stopping)
# tol = 1e-4 (tolerance for improvement)
gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=3,
                                random_state=42, validation_fraction=0.1,
                                n_iter_no_change=10, tol=1e-4)

# Fit the Gradient Boosting Regressor model on the training data
gbr.fit(X_train, y_train)

# Make predictions on the test data using the fitted model
y_pred = gbr.predict(X_test)

# Calculate the Mean Squared Error (MSE) between the actual test target values and the predicted values
mse = mean_squared_error(y_test, y_pred)

# Print the MSE to evaluate the model's performance
print(f'Mean Squared Error: {mse}')

# Create a new figure for plotting the actual vs predicted values
plt.figure(figsize=(12, 6))

# Plot the actual target values from the test set
plt.plot(y_test.values, label='Actual Values', color='blue')

# Plot the predicted values
plt.plot(y_pred, label='Predicted Values', color='red', linestyle='dashed')

# Set the x-axis label
plt.xlabel('Observation Index')

# Set the y-axis label
plt.ylabel('Day Percentage Change')

# Set the title of the plot
plt.title('Actual vs Predicted Day Percentage Change')

# Display the legend to differentiate between actual and predicted values
plt.legend()

# Show the plot
plt.show()
