# To pre process data for the fashion tiers model

import warnings
import numpy as np
from fuzzywuzzy import fuzz
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

def price_label(p):
    label = ''
    if p < 30:
        label = 'AAA'
    if 30<= p <60:
        label = 'BBB'
    if 60<=p<90:
        label = 'CCC'
    if 90<=p<120:
        label = 'DDD'
    if 120<=p<150:
        label = 'EEE'
    if 150<=p<180:
        label= 'FFF'
    if 180<=p<210:
        label = 'GGG'
    if 210<=p<240:
        label = 'HHH'
    if 240<=p<270:
        label = 'III'
    if 270<=p<300:
        label = 'JJJ'
    if 300<p<330:
        label = 'KKK'
    if 330<p<360:
        label = 'LLL'
    if 360<p<400:
        label = 'MMM'
    return label

i=0
df = pd.read_csv('tier_training_data.csv')
dfh = pd.read_csv('brands_luxury.csv',header=None)
dfl = pd.read_csv('brands_low.csv',header=None)
brands=[]
prices=[]
sale_prices=[]
tier=[]
i = 0
flag1=0
flag2=0

for index1,row1 in df.iterrows():
    for index2,row2 in dfl.iterrows():
        if fuzz.ratio(row2[0].strip().lower(),row1[3].strip().lower())>85:
            print(fuzz.ratio(row2[0].strip().lower(),row1[3].strip().lower()))
            brands.append(str(row1[3]))
            prices.append(price_label(row1[20]))
            sale_prices.append(price_label(row1[20]))
            tier.append(['fast'])
            flag1=1
    if flag1 ==0:
        for index3,row3 in dfh.iterrows():
            if fuzz.ratio(row3[0].strip().lower(),row1[3].strip().lower())>85:
                print(fuzz.ratio(row3[0].strip().lower(),row1[3].strip().lower()))
                brands.append(str(row1[3]))
                prices.append(price_label(row1[9]))
                sale_prices.append(price_label(row1[20]))
                tier.append(['designer'])
                flag2=1
        if flag2 == 0:
            #print(row1[3])
            brands.append(str(row1[3]))
            prices.append(price_label(row1[9]))
            sale_prices.append(price_label(row1[20]))
            if row1[9] > 400:
                tier.append(['designer'])
            else:
                tier.append(['mid'])
    flag1=0
    flag2=0
    i=i+1

dfo = pd.DataFrame({'brands': brands, 'prices': prices, 'sale':sale_prices, 'tier':tier}, columns=['brands', 'prices','sale','tier']) 
dfo.loc[dfo["prices"] == '','prices'] = dfo["sale"]
dfo = dfo.drop(['sale'],axis=1)
dfo.to_csv('data_pretraining.csv')

