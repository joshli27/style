#trains the model which determines the occasion of an item of clothing

from __future__ import print_function
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

x_train, x_test, y_train, y_test = train_test_split(x_sub, np.array(y_sub), test_size=0.2,
                                                        random_state=42)
tokenize = Tokenizer(num_words=3000, char_level=False)
tokenize.fit_on_texts(x_train)
with open('tokenizer_unigram.pickle', 'wb') as handle:
    pickle.dump(tokenize, handle, protocol=pickle.HIGHEST_PROTOCOL)

x_train1 = tokenize.texts_to_matrix(x_train)
x_test1 = tokenize.texts_to_matrix(x_test)

model = Sequential()
model.add(Dense(512, input_shape=(3000,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train1, y_train,
          batch_size=128,
          epochs=5,
          verbose=1,
          validation_split=0.1)

score = model.evaluate(x_test1, y_test, batch_size=128, verbose=1)
print('Test accuracy:', score[1])
filename = 'style_unigram.h5'
model.save(filename)    

x_train2 = tokenize.texts_to_sequences(x_train) 
x_test2 = tokenize.texts_to_sequences(x_test)
x_train3 = sequence.pad_sequences(x_train2, maxlen=250)
x_test3 = sequence.pad_sequences(x_test2, maxlen=250)

model2 = Sequential()
model2.add(Embedding(3000, 128))
model2.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model2.add(Dense(n_class, activation='sigmoid'))

model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model2.fit(x_train3, y_train,
          batch_size=128,
          epochs=5,
          validation_split=0.1)

score, acc = model2.evaluate(x_test3, y_test,
                            batch_size=128)
model2.save('style_lstm_padding250.h5')  
print('Test score:', score)
print('Test accuracy:', acc)

x_t_1 = tokenize.texts_to_matrix(x_test_n[:10000])
preds = model.predict(x_t_1)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0

_t_2 = tokenize.texts_to_sequences(x_test_n[:10000]) 
x_t_3 = sequence.pad_sequences(x_t_2, maxlen=250)
preds2 = model2.predict(x_t_3)
preds2[preds2>=0.5] = 1
preds2[preds2<0.5] = 0