import pandas as pd  # Importing the pandas library for data manipulation and analysis
from datasets import load_dataset  # Importing the function to load datasets from Hugging Face
from sklearn.model_selection import train_test_split, GridSearchCV  # Importing functions for splitting data and hyperparameter tuning
from sklearn.ensemble import GradientBoostingRegressor  # Importing the Gradient Boosting Regressor model
from sklearn.metrics import mean_squared_error  # Importing function to calculate Mean Squared Error
import matplotlib.pyplot as plt  # Importing the matplotlib library for plotting

# Load dataset
tesla = load_dataset('codesignal/tsla-historic-prices')  # Loading the Tesla historic prices dataset
tesla_df = pd.DataFrame(tesla['train'])  # Converting the training dataset into a pandas DataFrame

# Feature Engineering
tesla_df['SMA_5'] = tesla_df['Adj Close'].rolling(window=5).mean()  # Calculating the 5-day Simple Moving Average (SMA)
tesla_df['SMA_10'] = tesla_df['Adj Close'].rolling(window=10).mean()  # Calculating the 10-day Simple Moving Average (SMA)
tesla_df['EMA_5'] = tesla_df['Adj Close'].ewm(span=5, adjust=False).mean()  # Calculating the 5-day Exponential Moving Average (EMA)
tesla_df['EMA_10'] = tesla_df['Adj Close'].ewm(span=10, adjust=False).mean()  # Calculating the 10-day Exponential Moving Average (EMA)

# Drop NaN values created by moving averages
tesla_df.dropna(inplace=True)  # Removing any rows with NaN values, which may result from moving averages

# Select features and target
features = tesla_df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10']].values  # Selecting feature columns
target = tesla_df['Adj Close'].shift(-1).dropna().values  # Creating the target variable as the next day's adjusted close price
features = features[:-1]  # Adjusting features to match the target length

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)  # Splitting the data into train and test sets (25% test)

# Setting up the hyperparameter grid
param_grid = {
    'learning_rate': [0.05, 0.1],  # Possible values for the learning rate
    'n_estimators': [150, 200],  # Possible values for the number of estimators (trees)
    'max_depth': [3, 4]  # Possible values for the maximum depth of the trees
}

# Instantiate the GridSearchCV object
model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=4)  # Setting up Grid Search for hyperparameter tuning

# Fit the model to the training data
model.fit(X_train, y_train)  # Fitting the model with training data

# Print the best parameters found
print("Best parameters found:", model.best_params_)  # Displaying the best hyperparameters from Grid Search

# Predict with the best estimator
best_model = model.best_estimator_  # Retrieving the best model after hyperparameter tuning
predictions = best_model.predict(X_test)  # Making predictions on the test set

# Calculate and print Mean Squared Error
mse = mean_squared_error(y_test, predictions)  # Calculating the Mean Squared Error between actual and predicted values
print("Mean Squared Error with best params:", mse)  # Displaying the MSE

# Plotting predictions vs actual values
plt.figure(figsize=(10, 6))  # Setting the figure size for the plot
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7)  # Plotting actual values
plt.scatter(range(len(y_test)), predictions, label='Predicted', alpha=0.7)  # Plotting predicted values
plt.title('Actual vs Predicted Values with Tuned Hyperparameters')  # Adding title to the plot
plt.xlabel('Sample Index')  # Labeling the x-axis
plt.ylabel('Value')  # Labeling the y-axis
plt.legend()  # Adding a legend to the plot
plt.show()  # Displaying the plot

"""
Best parameters found: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 150}
Mean Squared Error with best params: 21.765725574204243
"""