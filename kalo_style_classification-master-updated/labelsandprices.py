f = open('data_t1.csv','r')

w = open('data_t2.csv','w')

i = 0

f = f.readlines()

for line in f:
    line =line.replace('Fast/Mid', '[0 1]')
   # line.replace('Designer', [1 0])
    w.write(line)
