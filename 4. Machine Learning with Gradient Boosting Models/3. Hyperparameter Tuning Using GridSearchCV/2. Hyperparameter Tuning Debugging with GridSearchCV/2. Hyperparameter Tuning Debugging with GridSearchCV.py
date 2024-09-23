# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from datasets import load_dataset  # To load datasets from Hugging Face
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from sklearn.ensemble import GradientBoostingRegressor  # The regression model to use
from sklearn.metrics import mean_squared_error  # For evaluating the model's performance
import matplotlib.pyplot as plt  # For plotting the results

# Load the dataset
tesla = load_dataset('codesignal/tsla-historic-prices')  # Load the Tesla historical prices dataset from CodeSignal
tesla_df = pd.DataFrame(tesla['train'])  # Convert the loaded dataset into a pandas DataFrame for easier manipulation

# Feature Engineering
# Calculate Simple Moving Averages (SMA)
tesla_df['SMA_5'] = tesla_df['Adj Close'].rolling(window=5).mean()  # Calculate the 5-day SMA of the adjusted close price
tesla_df['SMA_10'] = tesla_df['Adj Close'].rolling(window=10).mean()  # Calculate the 10-day SMA of the adjusted close price

# Calculate Exponential Moving Averages (EMA)
tesla_df['EMA_5'] = tesla_df['Adj Close'].ewm(span=5, adjust=False).mean()  # Calculate the 5-day EMA of the adjusted close price
tesla_df['EMA_10'] = tesla_df['Adj Close'].ewm(span=10, adjust=False).mean()  # Calculate the 10-day EMA of the adjusted close price

# Drop NaN values created by the moving averages
tesla_df.dropna(inplace=True)  # Remove any rows with NaN values to ensure clean data for modeling

# Select features and target variable
# Define the features (input variables) for the model
features = tesla_df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10']].values  
# Define the target variable (what we want to predict) - the next day's adjusted close price
target = tesla_df['Adj Close'].shift(-1).dropna().values  # Shift the adjusted close price up by one to predict the next day's price

# Ensure alignment between features and target
# Remove the last feature row that doesn't have a corresponding target
features = features[:-1]  # Exclude the last row of features to match the length of the target variable

# Splitting the dataset into training and testing sets
# Use train_test_split to create training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)  
# 25% of the data is set aside for testing, while 75% will be used for training

# Setting up the hyperparameter grid for tuning
# Define a dictionary with the hyperparameters to be tested during the Grid Search
param_grid = {
    'learning_rate': [0.05, 0.1],  # Test different learning rates
    'n_estimators': [150, 200],     # Test different numbers of boosting stages
    'max_depth': [3, 4]             # Test different maximum depths of the individual trees
}

# Instantiate the model using GridSearchCV for hyperparameter tuning
# The model is set to use a Gradient Boosting Regressor and perform cross-validation with 4 folds
model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=4)

# Fit the model to the training data
model.fit(X_train, y_train)  # Train the model on the training dataset

# Print the best parameters found from Grid Search
print("Best parameters found:", model.best_params_)  # Output the best hyperparameters identified during tuning

# Predict with the model on the test set
predictions = model.predict(X_test)  # Generate predictions for the test dataset

# Calculate and print Mean Squared Error (MSE) to evaluate the model's performance
mse = mean_squared_error(y_test, predictions)  # Calculate the Mean Squared Error between actual and predicted values
print("Mean Squared Error with best params:", mse)  # Output the MSE

# Plotting predictions vs actual values
plt.figure(figsize=(10, 6))  # Set the figure size for the plot
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7)  # Plot actual values of the test set
plt.scatter(range(len(y_test)), predictions, label='Predicted', alpha=0.7)  # Plot predicted values from the model
plt.title('Actual vs Predicted Values with Tuned Hyperparameters')  # Title of the plot
plt.xlabel('Sample Index')  # X-axis label indicating sample indices
plt.ylabel('Value')  # Y-axis label indicating the values of the stock price
plt.legend()  # Show legend to differentiate between actual and predicted values
plt.show()  # Display the plot
