import re
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SpatialDropout1D, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.utils.np_utils import to_categorical

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


data = pd.read_csv('db/04-train-data.csv')
data = data[['TEXT', 'SENTIMENT']]
data = data.dropna(subset=['TEXT'])
data = data.sample(frac=1).reset_index(drop=True)


max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['TEXT'].values)


X = tokenizer.texts_to_sequences(data['TEXT'].values)
X = pad_sequences(X)
Y = pd.get_dummies(data['SENTIMENT']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(CuDNNLSTM(lstm_out, return_sequences=True))
model.add(SpatialDropout1D(0.4))
model.add(CuDNNLSTM(lstm_out))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
model.summary()


history = model.fit(X_train, Y_train, epochs = 10, batch_size=256, verbose = 1, validation_split=0.2, shuffle=True)
