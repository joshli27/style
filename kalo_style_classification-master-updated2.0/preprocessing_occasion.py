# Preprocesses data to be used to train the model determining the occasion of an item of clothing

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

with open('new_testdata.csv', 'r', newline='', encoding='utf-8', errors="ignore") as csv2:
    raw = list(csv.reader(csv2))
    x, y = [], []
    i, percent = 1, 0
    t1 = time.time()
    total_num = len(raw)
    for p in raw:
        print()
        x.append(clean_train(p[3]+' '+p[4]))
        y.append(eval(p[5]))
        if i >= total_num * percent:
            percent += 0.01
            t2 = time.time()
            print("[", i, "/", total_num, "]", int(percent * 100), "%", " Estimated time remaining:",
                  int(((t2 - t1) / i) * (total_num - i + 1)), "s")
        i += 1
    del raw
    gc.collect()

# with open('/content/drive/My Drive/style1_sppl_trainingset.csv', 'r', newline='', encoding='utf-8', errors="ignore") as csv2:
#     raw = list(csv.reader(csv2))
#     i, percent = 1, 0
#     t1 = time.time()
#     total_num = len(raw)
#     for p in raw:
#         x.append(clean_train(p[0]+' '+p[1]))
#         y.append(eval(p[-1]))
#         if i >= total_num * percent:
#             percent += 0.01
#             t2 = time.time()
#             print("[", i, "/", total_num, "]", int(percent * 100), "%", " Estimated time remaining:",
#                   int(((t2 - t1) / i) * (total_num - i + 1)), "s")
#         i += 1
#     del raw
#     gc.collect()
    
mlb = MultiLabelBinarizer()
y_labeled = mlb.fit_transform(y)
n_class = len(list(mlb.classes_))
with open('encoder.pkl', 'wb') as f:
    pickle.dump(mlb, f)