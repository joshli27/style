import pandas as pd

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False 
df = pd.read_csv('style_test.csv')

df.insert(0, 'ID', range(0,len(df)))

open('style_test1.csv','w')
i = 0
flag =0
p=10
# [0 1] - Designer, [1 0] - Fast/Mid
df2 = df[0:0] 
 
for index,row in df.iterrows():
    #print(row[1])
    #print(row[10])
    #print(row[10])
    #if (isinstance(row[10],float))==0:
    if (is_number(row[10])) ==1:
        p = float(row[10])
            #print(p)
    #    elif (isinstance(row[21],float))==0:
    elif (is_number(row[21]))==1:
        p=float(row[21])
    else:
        flag=1
    if flag == 1:
        row[10]=''
    if flag ==0:
        if p < 30:
            row[10] = 'AAA'
            print("YES\n")
        if 30<= p <60:
            row[10] = 'BBB'
            print("YES\n") 
        if 60<=p<90:
            row[10] = 'CCC'
            print("YES\n")
        if 90<=p<120:
            row[10] = 'DDD'
            print("YES\n")
        if 120<=p<150:
            row[10] = 'EEE'
        if 150<=p<180:
            row[10]= 'FFF'
        if 180<=p<210:
            row[10] = 'GGG'
        if 210<=p<240:
            row[10] = 'HHH'
        if 240<=p<270:
            row[10] = 'III'
        if 270<=p<300:
            row[10] = 'JJJ'
        if 300<p:
            row[10] = 'ZZZ'
    flag = 0
    #print(row)
    df2= df2.append(row,sort=False)
   # if row2[0]:
   #     new = open('data_tiers1.csv','a')
   #     row1['Tier']='0 1'
   #     row1.to_csv('data_t.csv',mode='a',header=False, index=False)
   #     print(row1['name'])
   #     print(row1['Tier'])
   #     flag = 1
   # print(i)
    i = i+1
print(df2)
df2.to_csv('style_test1.csv',mode='w',index=None,header=True)
#df.info(verbose=True)
#locs.info(verbose=True)
#df.info(verbose=True)
