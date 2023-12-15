import pandas as pd
import numpy as np

# Creating a DataFrame with some missing values
data = {'Feature1': [1, 2, np.nan, 4, 5], 
        'Feature2': [np.nan, 2, 3, 4, 5], 
        'Feature3': [1, np.nan, np.nan, 4, 5]}
df = pd.DataFrame(data)

# Impute missing values with the mean of each column
df_imputed = df.fillna(df.mean())
print(df_imputed)


# Defining a function to find outliers
def find_outliers(dataframe):
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)
    IQR = Q3 - Q1

    outlier_condition = ((dataframe < (Q1 - 1.5 * IQR)) | (dataframe > (Q3 + 1.5 * IQR)))
    return outlier_condition

outliers = find_outliers(df_imputed)
print(outliers)
