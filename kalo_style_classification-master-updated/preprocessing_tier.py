import re
import gc
import csv
import time
import nltk
import random
import string
import pickle
import argparse
import numpy as np

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Activation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_train(text):

    """
    Lower-case, tokenize, stem,
    Removal: digits + punctuations + white spaces + stopwords
    """
    text = re.sub(r'\d+', '', text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    text = [i for i in text if i not in stop_words]
    stemmer = SnowballStemmer(language='english')
    text = [stemmer.stem(t) for t in text]
    return(' '.join(text))

#def preprocess():

with open('data_pretraining.csv', 'r', newline='', encoding='utf-8', errors="ignore") as csv2:
    raw = list(csv.reader(csv2))
    x, y= [], []
    i, percent = 1, 0
    t1 = time.time()
    total_num = len(raw)
    for p in raw[1:]:
#        print(clean_train(p[6]))
        x.append(clean_train(p[1]+ ' ' +p[2]))
        #print(p)
        y.append(eval(p[3]))
        #print(p[10])
        if i >= total_num * percent:
            percent += 0.01
            t2 = time.time()
            print("[", i, "/", total_num, "]", int(percent * 100), "%", " Estimated time remaining:",
                  int(((t2 - t1) / i) * (total_num - i + 1)), "s")
        i += 1
    del raw
    gc.collect()
mlb = MultiLabelBinarizer()
y_labeled = mlb.fit_transform(y)

with open('yl_file.csv', 'w') as f:
    writer = csv.writer(f)
    for item in y_labeled:
        writer.writerow(item)
print(y)
n_class = len(list(mlb.classes_))
#n_class=3
print(n_class)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(mlb, f)

with open('xl_file.csv', 'w') as f:
    for item in x:
        f.write("%s\n" % item)

pickle_off = open("encoder.pkl","rb")
emp = pickle.load(pickle_off)
   # return
