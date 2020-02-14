import re
import unidecode
def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

names = open('brands_luxury.csv','r')
data = open('data_tiers.csv','r')
ndata = open('data_new_brands.csv','w')
#more_names = open('brandnames_more.csv','r')

n = names.readlines()
d = data.readlines()
#more_n = more_names.readlines()

i = 0
flag = 0
flag2 = 0
for x in d:
    for y in n:
        y = unidecode.unidecode(y)

        if findWholeWord(y.strip().lower())(x.strip().lower()):
            # print(y.lower().strip())
            print(x.lower().strip())
            print(i)
            ndata.write(x.strip()+', ' +y.strip()+'\n')
            i=i+1
            flag = 1
            flag2 = 1
            break
#    if flag == 0:
#        for z in more_n:
#            z = unidecode.unidecode(z)
#            # print(z)
#            # print(x)
#            if findWholeWord(z.strip().lower())(x.strip().lower()):
#                # print(y.lower().strip())
#                # print(x.lower().strip())
#                # print(i)
#                ndata.write(x.strip()+', ' +z.strip()+'\n')
#
#                i=i+1
#                flag2=1
#                break
#    flag =0
    #if flag2 == 0:
    #    print(x)
    flag2 = 0

# x = "1999,518,Burberry Small Rucksack,Burberry Small Rucksack-Handbags,['Casual'],https://s3.amazonaws.com/kalo-production/products/0004000000518.jpg"

# for y in n:
#     if 'burberry' in y.lower():
#         print('yes')
