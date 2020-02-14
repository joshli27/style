f = open('data_t.csv','r')

w = open('data_t1.csv','a')

f = f.readlines()

i = 0
merged = ''
for l in f:
    if i == 10:
        i = 0
        w.write(merged + '\n')
        merged = ''
    if i == 0:
        merged = l.strip()
    else:
        merged = merged +','+ l.strip()
    i = i+1
