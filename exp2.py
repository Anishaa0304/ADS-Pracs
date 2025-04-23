import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\anish\Downloads\ADS lab exam solutions\ADS Datasets\loan_data_set.csv")
print(df)

# Find columns with null values
na_variables = [var for var in df.columns if df[var].isnull().mean() > 0]
print(na_variables)

# Mean imputation
df1 = df.copy()
missing_col = ["LoanAmount"]
for i in missing_col:
    df1.loc[df1.loc[:, i].isnull(), i] = df1.loc[:, i].mean()  # Using .loc for assignment
print(df1)

# Median imputation
df2 = df.copy()
for i in missing_col:
    df2.loc[df2.loc[:, i].isnull(), i] = df2.loc[:, i].median()
print(df2)

# Mode imputation
df4 = df.copy()
for i in missing_col:
    df4.loc[df4.loc[:, i].isnull(), i] = df4.loc[:, i].mode()[0]
print(df4)

# Categorical to numerical using OrdinalEncoder
data = df.copy()
oe = OrdinalEncoder()
result = oe.fit_transform(data)
print(result)

# Random sample imputation
df5 = df.copy()
df5.loc[df5['LoanAmount'].isnull(), 'LoanAmount'] = df5['LoanAmount'].dropna().sample(df5['LoanAmount'].isnull().sum(), random_state=0).values
print(df5)

# Frequent category imputation
df6 = df.copy()
m = df6["Gender"].mode()
m = m.tolist()
frq_imp = df6["Gender"].fillna(m[0])
print(frq_imp.unique())
