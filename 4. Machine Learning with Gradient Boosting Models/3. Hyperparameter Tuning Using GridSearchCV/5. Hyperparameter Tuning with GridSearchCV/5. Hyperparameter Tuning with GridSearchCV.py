import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datasets import load_dataset

# Load dataset
tesla = load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(tesla['train'])

# Calculate 20-day Simple Moving Averages (SMA) on the dataset
tesla_df['20_day_SMA'] = tesla_df['Adj Close'].rolling(window=20).mean()

# Drop rows with NaN values after calculating the moving averages
tesla_df.dropna(inplace=True)

# Select relevant features and prepare the target variable
# Using `Open`, `High`, `Low`, `Close`, `Volume`, and `20_day_SMA` as features
features = ['Open', 'High', 'Low', 'Close', 'Volume', '20_day_SMA']
target = 'Adj Close'

X = tesla_df[features]
y = tesla_df[target].shift(-1).dropna()  # Predict next day's adjusted close
X = X.iloc[:-1]  # Align features with target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the hyperparameter grid
param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 4]
}

# Instantiate the GridSearchCV object with GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print best parameters found by GridSearchCV
print("Best parameters found: ", grid_search.best_params_)

# Use the best estimator to predict on the test set
best_gbr = grid_search.best_estimator_
y_pred = best_gbr.predict(X_test)

# Calculate and print the Mean Squared Error for the predictions
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot predictions against actual values using matplotlib
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red')
plt.title('Actual vs Predicted Adjusted Close Prices')
plt.xlabel('Data Points')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()
