from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# Load dataset
tesla = load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(tesla['train'])

# TODO: Convert Date column to datetime type
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# TODO: Perform feature engineering to add a shifted Adj Close column as target, and the mean and std of Adj Close over 5 and 10 days
# - Add a new column 'Target' which is the difference between the shifted 'Adj Close' and the current 'Adj Close' to use as the target variable.
# - Add columns 'Mean_5' and 'Std_5', which are the mean and standard deviation of the 'Adj Close' over the past 5 days.
# - Add columns 'Mean_10' and 'Std_10', which are the mean and standard deviation of the 'Adj Close' over the past 10 days.
tesla_df['Adj Close Shifted'] = tesla_df['Adj Close'].shift(-1)
tesla_df['Target'] = tesla_df['Adj Close Shifted'] - tesla_df['Adj Close']
tesla_df['Mean_5'] = tesla_df['Adj Close'].rolling(window=5).mean()
tesla_df['Std_5'] = tesla_df['Adj Close'].rolling(window=5).std()
tesla_df['Mean_10'] = tesla_df['Adj Close'].rolling(window=10).mean()
tesla_df['Std_10'] = tesla_df['Adj Close'].rolling(window=10).std()

# TODO: Drop NaN values created during feature engineering
tesla_df.dropna(inplace=True)

# TODO: Select features and target values for model
# The target is the 'Target' column, features to choose is up to you
features = ['Adj Close', 'Mean_5', 'Std_5', 'Mean_10', 'Std_10']
target = 'Target'

X = tesla_df[features]
y = tesla_df[target]

# TODO: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# TODO: Instantiate the Gradient Boosting model with `n_estimators = 100`, `learning_rate = 0.1`, `max_depth = 3`
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# TODO: Perform 5-fold cross-validation and print the mean cross-validation score
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f'Mean cross-validation score: {cv_scores.mean()}')

# TODO: Fit the model to the entire dataset and predict target values
model.fit(X_scaled, y)
predictions = model.predict(X_scaled)

# TODO: Plot actual vs predicted values to visualize model performance```
# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(tesla_df['Date'], y, label='Actual', color='blue')
plt.plot(tesla_df['Date'], predictions, label='Predicted', color='red', alpha=0.7)
plt.title('Actual vs Predicted Target Values')
plt.xlabel('Date')
plt.ylabel('Target Value')
plt.legend()
plt.show()