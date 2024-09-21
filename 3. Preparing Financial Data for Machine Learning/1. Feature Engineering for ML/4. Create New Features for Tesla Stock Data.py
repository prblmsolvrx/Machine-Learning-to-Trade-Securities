import pandas as pd
import datasets

# Load the dataset
data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])

# TODO: Create the Daily Range Percentage feature (what's the percentage of the highest daily price from the lowest)
tesla_df['Daily Range %'] = (tesla_df['High'] - tesla_df['Low']) / tesla_df['Low'] * 100

# Creating the Price Change Percentage feature
tesla_df['Price Change %'] = (tesla_df['Close'] - tesla_df['Open']) / tesla_df['Open'] * 100

# TODO: Display the new features of the Tesla dataset
print(tesla_df['Daily Range %'])
print(tesla_df['Price Change %'])

"""
0       42.531426
1       30.557968
2       27.873737
3       23.463422
4       26.342396
          ...    
3342     3.693704
3343     4.381917
3344     2.951327
3345     3.421268
3346     3.748703
Name: Daily Range %, Length: 3347, dtype: float64
0       25.736835
1       -7.599808
2      -12.160018
3      -16.521721
4      -19.449980
          ...    
3342     1.707734
3343     2.277399
3344    -1.205868
3345    -1.540399
3346    -3.005021
Name: Price Change %, Length: 3347, dtype: float64
"""