import pandas as pd  # Import the pandas library for data manipulation and analysis
from datasets import load_dataset  # Import function to load datasets from the Hugging Face library
from sklearn.model_selection import train_test_split, GridSearchCV  # Import functions for splitting data and performing grid search
from sklearn.ensemble import GradientBoostingRegressor  # Import the Gradient Boosting Regressor model
from sklearn.metrics import mean_squared_error  # Import mean squared error metric for evaluation
import matplotlib.pyplot as plt  # Import matplotlib for plotting data

# Load dataset from the 'codesignal/tsla-historic-prices' dataset
tesla = load_dataset('codesignal/tsla-historic-prices')
# Convert the loaded dataset into a pandas DataFrame for easier manipulation
tesla_df = pd.DataFrame(tesla['train'])

# Feature Engineering
# Calculate the price range by subtracting the low price from the high price
tesla_df['Price_Range'] = tesla_df['High'] - tesla_df['Low']
# Calculate daily return as the ratio of adjusted close price to the open price, minus 1
tesla_df['Daily_Return'] = tesla_df['Adj Close'] / tesla_df['Open'] - 1
# Calculate the rolling volatility over a window of 5 days using the price range
tesla_df['Volatility'] = tesla_df['Price_Range'].rolling(window=5).std()
# Calculate the rolling average volume over a window of 5 days
tesla_df['Average_Volume'] = tesla_df['Volume'].rolling(window=5).mean()

# Drop NaN values created by rolling operations, which may result in missing data
tesla_df.dropna(inplace=True)

# Select features (independent variables) for the model
features = tesla_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Range', 'Daily_Return', 'Volatility', 'Average_Volume']].values
# Shift the target variable (dependent variable) to predict the next day's adjusted close price
target = tesla_df['Adj Close'].shift(-1).dropna().values  # Predicting next day's close price
# Adjust features to match the length of target after shifting
features = features[:-1]

# Split the dataset into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

# Set up the hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],  # Number of boosting stages to be run
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinking to prevent overfitting
    'max_depth': [3, 5, 7],  # Maximum depth of the individual regression estimators
    'subsample': [0.8, 1.0]  # Fraction of samples used for fitting the individual base learners
}

# Instantiate the GridSearchCV object with the model, parameter grid, and cross-validation strategy
model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3)

# Fit the model to the training data
model.fit(X_train, y_train)

# Print the best hyperparameters found during grid search
print("Best parameters found:", model.best_params_)

# Use the best estimator to make predictions on the test set
predictions = model.best_estimator_.predict(X_test)

# Calculate the Mean Squared Error between the actual and predicted values
mse = mean_squared_error(y_test, predictions)
# Print the Mean Squared Error to evaluate model performance
print("Mean Squared Error with best params:", mse)

# Plotting predictions vs actual values
plt.figure(figsize=(10, 5))  # Set the size of the plot
plt.plot(y_test, label='Actual Prices', color='blue', alpha=0.6)  # Plot actual prices
plt.plot(predictions, label='Predicted Prices', color='red', alpha=0.6)  # Plot predicted prices
plt.title('Tesla Stock Price Prediction')  # Set the title of the plot
plt.xlabel('Sample Index')  # Label for the x-axis
plt.ylabel('Adjusted Close Price')  # Label for the y-axis
plt.legend()  # Show legend for the plot
plt.show()  # Display the plot

"""
Best parameters found:  {'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 2, 'n_estimators': 100}
Mean Squared Error with best params: 28.347421051950818
"""