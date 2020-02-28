#identify by prices

import pandas as pd
import csv
import numpy as np

def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

#f = pd.read_csv('newdata.csv')
#
#keep_col = ['price','sale_price']
#new_f = f[keep_col]
#pd.to_numeric(new_f['price'])
#new_f.to_csv('prices.csv',index = True)

new_f = pd.read_csv('newdata.csv')
i = 0
j = 0


f_id = open('brands_low_price_id.csv','w')

for x in new_f['price']:
    if x < 150:
        print(new_f['name'][i])
        #f_id.write(to_str(new_f['id'][i]))
        #f_id.write('\n')
        j = j+1
    i = i+1
f_id.close()
print(j)
#filenames = ['brands_luxury_price_id.csv', 'brands_luxury_id.csv']
#with open('brands_luxury_merge.csv', 'w') as outfile:
#    for fname in filenames:
#        with open(fname) as infile:
#            outfile.write(infile.read())
