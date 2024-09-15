"""
Change the starter code to sort the DataFrame by date in descending
order instead of ascending order. This will help you see the most recent data first.
"""

import pandas as pd
import datasets

# Step 1: Load TSLA dataset
tesla_data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(tesla_data['train'])

# Step 2: Convert the Date column to datetime type
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Step 3: Set the Date column as the index
tesla_df.set_index('Date', inplace=True)

# Step 4: Sort the DataFrame based on the index
tesla_df.sort_index(ascending=False, inplace=True)

# Step 5: Display the first few rows to verify the changes
print(tesla_df.head())

"""
                  Open        High  ...   Adj Close     Volume
Date                                ...                       
2023-10-13  258.899994  259.600006  ...  251.119995  102073800
2023-10-12  262.920013  265.410004  ...  258.869995  111508100
2023-10-11  266.200012  268.600006  ...  262.989990  103706300
2023-10-10  257.750000  268.940002  ...  263.619995  122656000
2023-10-09  255.309998  261.359985  ...  259.670013  101377900

[5 rows x 6 columns]
"""