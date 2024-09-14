"""
load and display the DataFrame's information and statistics correctly.
"""

import pandas as pd
import datasets

# Load TSLA dataset
tesla_data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(tesla_data['train'])

# Print basic information about the dataset
print("\nBasic information about the Tesla DataFrame:")
# TODO: Add code to print basic information about the dataset using the appropriate method
print(tesla_df.info())

# Display summary statistics
print("\nSummary statistics of the Tesla DataFrame:")
# TODO: Add code to print the dataset's summary statistics
print(tesla_df.describe())

"""
Basic information about the Tesla DataFrame:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3347 entries, 0 to 3346
Data columns (total 7 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Date       3347 non-null   object 
 1   Open       3347 non-null   float64
 2   High       3347 non-null   float64
 3   Low        3347 non-null   float64
 4   Close      3347 non-null   float64
 5   Adj Close  3347 non-null   float64
 6   Volume     3347 non-null   int64  
dtypes: float64(5), int64(1), object(1)
memory usage: 183.2+ KB
None

Summary statistics of the Tesla DataFrame:
              Open         High  ...    Adj Close        Volume
count  3347.000000  3347.000000  ...  3347.000000  3.347000e+03
mean     67.901248    69.413435  ...    67.886520  9.643192e+07
std     100.209872   102.472746  ...   100.136888  8.058132e+07
min       1.076000     1.108667  ...     1.053333  1.777500e+06
25%      10.152000    10.432000  ...    10.081333  4.540425e+07
50%      16.793333    17.000000  ...    16.771334  8.011650e+07
75%      66.069336    67.129334  ...    65.896000  1.230548e+08
max     411.470001   414.496674  ...   409.970001  9.140820e+08

[8 rows x 6 columns]
"""