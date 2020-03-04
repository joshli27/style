import pandas as pd
import numpy as np
import math 

df = pd.read_csv('predictions.csv')
df =df.drop(columns=['Unnamed: 0'])
# define item as loud or neutral colors

def colors_class(colors):
    loud = ['cyan','pink','red','green','orange','yellow','purple']
    neutral = ['beige','black','grey','white','brown','blue']
    colors = str(colors)
    colors=colors.replace('{','')
    colors=colors.replace('}','')
    colors=colors.split(',')
    m=0
    #print(colors)
    for x in colors:
        if x.strip().lower() in loud:
            m = m+1
        if m >=2:
            return('loud')
    m=0
    return('neutral')

def style(vec):
    s=''
    occasion = vec[0]
    tier = vec[1]
    noise = vec[2]
    #print(occasion)
    if type(occasion) == float:
        if math.isnan(occasion) == 1:
            occasion = ''
    #o=vec.str.match('Casual')
    for occasion:
        if l == 'Casual\n':
            print('adf')
            for t in tier:
                if t == 'designer':
                    for n in noise:
                        if n == 'neutral':
                            print('avant') 
    #    if tier.rstrip() == 'designer':
    #        print('asdfdsa')
    #        if noise.rstrip() == 'neutral':
    #            s = 'avant garde, dynamic'
    return 0

df['noise'] = df['colors'].apply(colors_class)
#print(df['noise'])
dfi = pd.read_csv('predictions_occ.csv',delimiter='\n')
df['occasion']=dfi
df['style'] = df[['occasion','tier','noise']].apply(style,axis=1)

df.to_csv('classify_style.csv',index=False)
dfs=pd.DataFrame(columns=['style'])

#if 'casual' in df['occasion']:
#    print('1')
#    if 'designer' in df['tier']:
#        print('2')
#        if 'loud' in df['noise']:
#            dfs=dfs.append(pd.DataFrame({"style":'{avant garde,dynamic}'}), ignore_index=True)
#            print('asdf\n')
#df['style']=dfs
