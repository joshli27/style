from __future__ import print_function

from keras.models import load_model
import pickle
from style_preprocessing import clean_train
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
import pandas as pd
from keras import utils as np_utils
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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

with open('tokenizer_unigram.pickle', 'rb') as handle:
       tokenize = pickle.load(handle)
with open('encoder.pkl', 'rb') as f:
       encoder = pickle.load(f)
model = load_model('style_unigram.h5')
model.summary()

df = pd.read_csv('style_test1.csv',header=0)
print(type(df['price'][1]))
print(type(df['brand'][1]))
r_x = []
for i in range(len(df)):
    r_x.append(clean_train(str(df['price'][i]) + ' ' + str(df['brand'][i])))
r_x_train1 = tokenize.texts_to_matrix(r_x)
r_preds = model.predict(r_x_train1)
print(r_preds)

with open('tier_predictions_file1.csv', 'w') as f:
    writer = csv.writer(f)
    for item in r_preds:
        writer.writerow(item)
r_preds[r_preds>=0.4] = 1
r_preds[r_preds<0.4] = 0
predict = encoder.inverse_transform(r_preds)
print(predict)

with open('tier_predictions_file2.csv', 'w') as f:
    writer = csv.writer(f)
    for item in predict:
        writer.writerow(item)

