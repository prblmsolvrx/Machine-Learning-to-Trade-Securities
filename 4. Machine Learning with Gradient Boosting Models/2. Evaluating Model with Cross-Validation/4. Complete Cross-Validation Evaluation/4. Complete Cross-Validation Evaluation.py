import pandas as pd  # Import the pandas library for data manipulation and analysis
import matplotlib.pyplot as plt  # Import matplotlib for plotting and visualization
from datasets import load_dataset  # Import the load_dataset function to load datasets
from sklearn.model_selection import cross_val_score  # Import cross_val_score for cross-validation scoring
from sklearn.ensemble import GradientBoostingRegressor  # Import the GradientBoostingRegressor model
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling

# Load dataset
tesla = load_dataset('codesignal/tsla-historic-prices')  # Load the Tesla historic prices dataset from the specified source
tesla_df = pd.DataFrame(tesla['train'])  # Convert the loaded dataset into a pandas DataFrame

# Preprocess data
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])  # Convert the 'Date' column to datetime format
tesla_df['Rolling_Mean_5'] = tesla_df['Adj Close'].rolling(window=5).mean()  # Calculate the 5-day rolling mean of 'Adj Close'
tesla_df['Rolling_Std_5'] = tesla_df['Adj Close'].rolling(window=5).std()  # Calculate the 5-day rolling standard deviation of 'Adj Close'
tesla_df['Rolling_Mean_20'] = tesla_df['Adj Close'].rolling(window=20).mean()  # Calculate the 20-day rolling mean of 'Adj Close'
tesla_df.dropna(inplace=True)  # Drop rows with missing values

# Select features and target
features = tesla_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Rolling_Mean_5', 'Rolling_Std_5', 'Rolling_Mean_20']].values  # Extract feature columns into an array
target = tesla_df['Adj Close'].shift(-1).dropna().values  # Create target array by shifting 'Adj Close' up by one row and dropping any NA values
features = features[:-1]  # Align features and target by removing the last feature row to match the target length

# TODO: Standardize features
scaler = StandardScaler()  # Create a StandardScaler instance for feature standardization
features_scaled = scaler.fit_transform(features)  # Fit the scaler to the features and transform them to a standardized format

# TODO: Instantiate model using GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)  # Initialize the Gradient Boosting Regressor with specified parameters

# TODO: Perform cross-validation with 5 folds
scores = cross_val_score(model, features_scaled, target, cv=5)  # Perform 5-fold cross-validation and store scores
mean_score = scores.mean()  # Calculate the mean of the cross-validation scores
print("Mean cross-validation score: ", mean_score)  # Print the mean cross-validation score to the console

# TODO: Fit model and generate predictions
model.fit(features_scaled, target)  # Fit the model to the entire scaled feature set to train it
predictions = model.predict(features_scaled)  # Use the fitted model to predict target values from the scaled features

# Plotting predictions vs actual values
plt.figure(figsize=(10, 6))  # Create a new figure with specified size for the plot
plt.scatter(range(len(target)), target, label='Actual', alpha=0.7)  # Scatter plot for actual values with some transparency
plt.scatter(range(len(target)), predictions, label='Predicted', alpha=0.7)  # Scatter plot for predicted values with some transparency
plt.title('Actual vs Predicted Values with Cross-Validation')  # Set the title of the plot
plt.xlabel('Sample Index')  # Label the x-axis
plt.ylabel('Value')  # Label the y-axis
plt.legend()  # Display the legend for the plot
plt.show()  # Show the plot
