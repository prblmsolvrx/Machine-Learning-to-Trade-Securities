# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets
from sklearn.ensemble import GradientBoostingRegressor  # Gradient Boosting Regressor model
from sklearn.metrics import mean_squared_error  # For evaluating the model's performance
import matplotlib.pyplot as plt  # For data visualization
import numpy as np  # For numerical operations
import datasets  # For loading datasets

# Load dataset
tesla = datasets.load_dataset('codesignal/tsla-historic-prices')  # Load Tesla historic prices dataset from codesignal
tesla_df = pd.DataFrame(tesla['train'])  # Convert the loaded dataset to a pandas DataFrame

# Convert 'Date' column to datetime format
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])  # Converts the 'Date' column to datetime format for easier manipulation

# Add technical indicators:
# Calculate the price difference from the previous day
tesla_df['Price_Diff'] = tesla_df['Adj Close'] - tesla_df['Adj Close'].shift(1)  # Subtracts the previous day's adjusted close price from the current day

# Calculate volatility (standard deviation of adjusted close price over the last 5 days)
tesla_df['Volatility'] = tesla_df['Adj Close'].rolling(window=5).std()  # Uses a rolling window of 5 days to compute the standard deviation

# Calculate momentum (difference in adjusted close price compared to 5 days ago)
tesla_df['Momentum'] = tesla_df['Adj Close'] - tesla_df['Adj Close'].shift(5)  # Subtracts the adjusted close price from 5 days ago

# Calculate the logarithm of the adjusted close price
tesla_df['Log_Price'] = np.log(tesla_df['Adj Close'])  # Applies the natural logarithm to the adjusted close price

# Drop rows with NaN values
tesla_df.dropna(inplace=True)  # Removes any rows with missing values (NaNs) that may have resulted from the shift or rolling operations

# Select features and target variable
features = tesla_df[['Price_Diff', 'Volatility', 'Momentum', 'Log_Price']]  # Selects the columns to be used as features for training
target = tesla_df['Adj Close']  # Sets the adjusted close price as the target variable we want to predict

# Standardize the feature values
scaler = StandardScaler()  # Creates an instance of the StandardScaler
features_scaled = scaler.fit_transform(features)  # Fits the scaler to the features and transforms them to have a mean of 0 and standard deviation of 1

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)  
# Splits the data into training (80%) and test sets (20%) using random_state for reproducibility

# Instantiate the Gradient Boosting Regressor and fit it to the training data
gbr = GradientBoostingRegressor(random_state=42)  # Creates an instance of the Gradient Boosting Regressor with a set random state for consistency
gbr.fit(X_train, y_train)  # Fits the model to the training data

# Make predictions and calculate the Mean Squared Error (MSE)
y_pred = gbr.predict(X_test)  # Uses the trained model to make predictions on the test data
mse = mean_squared_error(y_test, y_pred)  # Calculates the mean squared error between the actual and predicted values
print(f"Mean Squared Error: {mse}")  # Prints the MSE to evaluate the model's performance

# Visualize actual vs predicted values using scatter plots
plt.figure(figsize=(10, 6))  # Sets the figure size for the plot
plt.scatter(y_test, y_pred, alpha=0.6)  # Creates a scatter plot of actual vs. predicted values with some transparency
plt.xlabel('Actual Values')  # Labels the x-axis
plt.ylabel('Predicted Values')  # Labels the y-axis
plt.title('Actual vs Predicted Values')  # Sets the plot title
plt.show()  # Displays the plot
