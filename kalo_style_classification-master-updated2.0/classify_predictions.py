import pandas as pd

df = pd.read_csv('tier_predictions_file2.csv')
dfo = pd.DataFrame(columns=['id','colors','tier'])
f2= pd.read_csv('style_test.csv')
i = 0
for index,x in df.iterrows():
    print(x[0])
    if abs(x[0]-1)<1e-9: 
        tier = 'designer'
    elif abs(x[2]-1.0)<1e-9:
        tier = 'mid'
    elif abs(x[1]-1)<1e-9:
        tier = 'fast'
    dfo=dfo.append(pd.DataFrame({"id":[f2['id'][i]], "colors":[f2['colors'][i]],"tier":[tier]}), ignore_index=True)
    #print(f2['name'][0])
    #print(dfo)
    i = i+1

dfo.to_csv('predictions.csv', index=False)
