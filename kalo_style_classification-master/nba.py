f = open('dictionary_nba1.csv','r')
w = open('dictionary_nba2.csv','w')

f=f.readlines()
i= 0
for x in f:
    if i == 16:
        i =0 
    if i == 0:
        w.write(x)
    i = i+1
     
