# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import StandardScaler  # For standardizing features
import datasets  # To load datasets easily
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.ensemble import GradientBoostingRegressor  # The machine learning model to be used
from sklearn.metrics import mean_squared_error  # For evaluating the model performance
import matplotlib.pyplot as plt  # For visualizing data

# Load the Tesla historical prices dataset from the CodeSignal datasets
tesla = datasets.load_dataset('codesignal/tsla-historic-prices')

# Convert the dataset into a DataFrame for easier data manipulation
tesla_df = pd.DataFrame(tesla['train'])

# Convert the 'Date' column from string format to datetime format for easier time series analysis
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Calculate technical indicators:
# Simple Moving Average (SMA) over the last 5 days
tesla_df['SMA_5'] = tesla_df['Adj Close'].rolling(window=5).mean()

# Exponential Moving Average (EMA) over the last 5 days
tesla_df['EMA_5'] = tesla_df['Adj Close'].ewm(span=5, adjust=False).mean()

# Volatility (standard deviation) of the 'Close' price over the last 5 days
tesla_df['Volatility'] = tesla_df['Close'].rolling(window=5).std()

# Drop rows with NaN values created by the rolling calculations
tesla_df.dropna(inplace=True)

# Selecting features for the model
# Features include SMA, EMA, Volatility, and Adjusted Close price
features = tesla_df[['SMA_5', 'EMA_5', 'Volatility', 'Adj Close']]

# The target variable is the 'Close' price we want to predict
target = tesla_df['Close']

# Standardize the features to have mean=0 and variance=1
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)  # Fit the scaler to the features and transform them

# Split the dataset into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.25, random_state=42)

# Instantiate the Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the target values for the test set
predictions = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) between actual and predicted values
mse = mean_squared_error(y_test, predictions)

# Print the MSE to evaluate model performance
print(f'Mean Squared Error: {mse}')

# Visualize the actual vs predicted values
plt.figure(figsize=(10, 6))  # Set the size of the figure
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7)  # Plot actual values
plt.scatter(range(len(y_test)), predictions, label='Predicted', alpha=0.7)  # Plot predicted values
plt.title('Actual vs Predicted Values')  # Set the title of the plot
plt.xlabel('Sample Index')  # Label for the x-axis
plt.ylabel('Value')  # Label for the y-axis
plt.legend()  # Show legend to differentiate between actual and predicted
plt.show()  # Display the plot
