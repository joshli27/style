import pandas as pd
import numpy as np
import re
from statistics import median 
#df = pd.read_csv('cat_prices5.csv')


#df.drop(df.columns[[0,1]],axis=1,inplace=True)

#df.to_csv('cat_prices5.csv')
#print(df)

#f = open('cat_prices8.csv','r')
#
#F = f.readlines()
#cats=[]
#for x in F:
#    cats.append(x)
#    #x = x.split(',')
#    #for y in x:
#    #    cats.append(y+'\n')
#    print(x)
#f.close()
#cats= list(set(cats))
#f=open('cat_prices8.csv','w')
#for y in cats:
#    f.write(y)
#

#flag=0
#f=open('cat_prices7.csv','r')
#c = []
#
#for x in f:
#    x=x.replace(' ','_')
#    #x=x.replace('}','')
#    #x=x.replace('"','')
#
#    #x=x.replace('1','')
#    #x=x.replace('2','')
#    #x=x.replace('3','')
#    #x=x.replace('4','')
#    #x=x.replace('5','')
#    #x=x.replace('6','')
#    #x=x.replace('7','')
#    #x=x.replace('8','')
#    #x=x.replace('9','')
#    #x=x.replace('0','')
#    c.append(x)
#f.close()
#f=open('cat_prices8.csv','w')
#for y in c:
#    f.write(y)
#

#w = open('cat_prices_y.csv','w')
#f = open('cat_prices8.csv','r')
#for x in f:
#    print(x)
#    w.write(x)
#    g = open('cat_prices3.csv','r')
#    for y in g:
#        y=y.replace(',',' ')
#        y=y.replace('}',' ')
#        y=y.replace('{',' ')
#        if x.strip() in y:
#            print(y)
#            i = 0
#            for c in y:
#                i = i+1
#                if c ==' ':
#                    w.write(y[i:i+9]+'\n')       
#                    break
#    g.close()
#

f = open('cat_pricesf.csv','r')
w = open('cat_pricest.csv','w')
prices=[]
for x in f:
    if x[0].isdigit() == 0:
        if len(prices) >0:
            prices = list(map(float, prices)) 
            average =median(prices)
            average =round(average,2)
            w.write(str(average)+' ')
            q1 = np.quantile(prices,0.25,axis=None)
            q3 = np.quantile(prices,0.75,axis=None)
            w.write(str(round(q1,2))+' '+ str(round(q3,2))+'\n')        
        prices=[]
        w.write(x)
    if x[0].isdigit() ==1:
        prices.append(x.replace('\n','')) 
prices = list(map(float, prices)) 
average =median(prices)
average =round(average,2)
w.write(str(average)+' ')
q1 = np.quantile(prices,0.25,axis=None)
q3 = np.quantile(prices,0.75,axis=None)
w.write(str(round(q1,2))+' '+ str(round(q3,2))+'\n') 
#for x in f:
#    if x[0].isdigit() == 1:
#        x =re.sub('[a-zA-Z\s_"]', '',x )    
#        x = x+'\n'
#        print(x)
#    w.write(x)
#
