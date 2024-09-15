"""
identify the mistake and correct it.
"""

import pandas as pd
import datasets

# Load TSLA dataset
tesla_data = datasets.load_dataset('codesignal/tsla-historic-prices')
tesla_df = pd.DataFrame(tesla_data['train'])

# Display the first 5 rows of the DataFrame
print(tesla_df.head())

# Print basic information about the dataset
print(tesla_df.info())

# Display summary statistics
print(tesla_df.describe())

"""
  Date      Open      High       Low     Close  Adj Close     Volume
0  2010-06-29  1.266667  1.666667  1.169333  1.592667   1.592667  281494500
1  2010-06-30  1.719333  2.028000  1.553333  1.588667   1.588667  257806500
2  2010-07-01  1.666667  1.728000  1.351333  1.464000   1.464000  123282000
3  2010-07-02  1.533333  1.540000  1.247333  1.280000   1.280000   77097000
4  2010-07-06  1.333333  1.333333  1.055333  1.074000   1.074000  103003500
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