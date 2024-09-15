"""
identify and fix the bug in the code. 
"""
import pandas as pd
import datasets

# Load TSLA dataset
tesla_data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(tesla_data['train'])

# Convert the Date column to datetime type and set as index
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])
tesla_df.set_index('Date', inplace=True)

# Sort the DataFrame based on the index
tesla_df.sort_index(ascending=False,inplace=True)

# Display the first few rows to verify the changes
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