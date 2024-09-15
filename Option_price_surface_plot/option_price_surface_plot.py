import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import os

# Step 1: Read data from Excel using pandas
data = pd.read_excel('option_data.xlsx')

# Assuming the data has columns 'strike', 'maturity', 'mid_price'
strike = data['strike'].values
maturity = data['maturity'].values
mid_price = data['mid_price'].values

# Step 2: Create a mesh grid for strike prices and maturities
strike_range = np.linspace(min(strike), max(strike), 100)  # Generate 100 points in strike price range
maturity_range = np.linspace(min(maturity), max(maturity), 100)  # Generate 100 points in maturity range
strike_grid, maturity_grid = np.meshgrid(strike_range, maturity_range)

# Step 3: Populate the matrix with mid-prices
# Use interpolation to fit the known data points into the mesh grid
mid_price_grid = griddata((strike, maturity), mid_price, (strike_grid, maturity_grid), method='cubic')

# Step 4: Plot the surface using matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(strike_grid, maturity_grid, mid_price_grid, cmap='viridis')

# Labels
ax.set_xlabel('Strike Price')
ax.set_ylabel('Maturity')
ax.set_zlabel('Mid Price')
plt.title('Option Mid-Price Surface')
plt.show()
print(os.getcwd())

