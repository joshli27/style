import pandas as pd

df = pd.read_csv('data_tiers.csv')
locs = pd.read_csv('brands_luxury_merge.csv')

df =  df.drop('Unnamed: 9',1)

df =  df.drop('Unnamed: 10',1)
df =  df. drop('Unnamed: 11',1)

df =  df. drop('tier',1)

df.insert(0, 'ID', range(0,len(df)))

i = 0
flag =0
# [0 1] - Designer, [1 0] - Fast/Mid
for index1,row1 in df.iterrows():
    for index2,row2 in locs.iterrows():
        if i==row2[0]:
            row1['Tier']='0 1'
            row1.to_csv('data_t.csv',mode='a',header=False, index=False)
            print(row1['name'])
            print(row1['Tier'])
            flag = 1
    if flag == 0:
        row1['Tier']='1 0'
        row1.to_csv('data_t.csv',mode='a',header=False, index=False)
        print(row1['name'])
    flag = 0
    print(i)
    i = i+1

df.info(verbose=True)
locs.info(verbose=True)
df.info(verbose=True)
