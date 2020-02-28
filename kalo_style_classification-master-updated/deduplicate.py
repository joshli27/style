import pandas as pd

df1= pd.read_csv('style_data.csv')

print(df1['id'])
df2 = pd.read_csv('style_test.csv')

df3 =pd.merge(df1, df2, on='id', how='outer')
print(df3['id'])

df3 = df3.drop_duplicates()
print(df3['id'])
