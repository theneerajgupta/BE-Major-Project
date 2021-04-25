print("# import all standard libraries...")
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
import matplotlib.pyplot as plt
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


print("# import all machine learning libraries...")
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalMaxPool1D
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

print("# read the required dataset and clean it...")
train_df = pd.read_csv("db/04-train-data.csv")
train_df.isnull().values.any()
X = train_df.TEXT.values.tolist()
y = train_df.SENTIMENT.values.tolist()

print("# divide data in train and train_test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("# prepare data for embedding layers...")
# first we create a tokenizer
tokenizer = Tokenizer(num_words=15000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print("# Adding 1 because of reserved 0 index")
vocab_size = len(tokenizer.word_index) + 1
maxlen = 20

print("# creating padded pad_sequences")
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print("# get the glove dictionary")
embeddings_dictionary = dict()
glove_file = open('db/glove.6B.100d.txt', encoding='utf8')
for line in tqdm(glove_file):
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

print("# create embedding matrix")
embedding_matrix = zeros((vocab_size, 100))
for word, index in tqdm(tokenizer.word_index.items()):
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
print("# THE GENERIC DEEP LEARNING MODEL")
model_normal = Sequential()
embedding_layer = Embedding(
                vocab_size,
                100,
                weights=[embedding_matrix],
                input_length=maxlen,
                trainable=False)
model_normal.add(embedding_layer)
model_normal.add(Flatten())
model_normal.add(Dense(1, activation='sigmoid'))
model_normal.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
model_normal.summary()
history_normal = model_normal.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2)
score_normal = model_normal.evaluate(X_test, y_test, verbose=1)

print("# THE CNN DEEP LEARNING MODEL")
model_cnn = Sequential()
embedding_layer = Embedding(
                vocab_size,
                100,
                weights=[embedding_matrix],
                input_length=maxlen,
                trainable=False)
model_cnn.add(embedding_layer)
model_cnn.add(Conv1D(128, 5, activation='relu'))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dense(1, activation='sigmoid'))
model_cnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
model_cnn.summary()
history_cnn = model_cnn.fit(X_train, y_train, batch_size=216, epochs=20, verbose=1, validation_split=0.2)
score_cnn = model_cnn.evaluate(X_test, y_test, verbose=1)

print("# THE LSTM DEEP LEARNING MODEL")
model_lstm = Sequential()
mbedding_layer = Embedding(
                vocab_size,
                100,
                weights=[embedding_matrix],
                input_length=maxlen ,
                trainable=False)
model_lstm.add(embedding_layer)
model_lstm.add(LSTM(128))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
model_lstm.summary()
history_lstm = model_lstm.fit(X_train, y_train, batch_size=216, epochs=20, verbose=1, validation_split=0.2)
score_lstm = model_lstm.evaluate(X_test, y_test, verbose=1)

print("# THE BiLSTM DEEP LEARNING MODEL")
model_bilstm = Sequential()
embedding_layer = Embedding(
                vocab_size,
                100,
                weights=[embedding_matrix],
                input_length=maxlen,
                trainable=False)
model_bilstm.add(embedding_layer)
model_bilstm.add(Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)))
model_bilstm.add(Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)))
model_bilstm.add(Conv1D(100, 5, activation='relu'))
model_bilstm.add(GlobalMaxPool1D())
model_bilstm.add(Dense(16, activation='relu'))
model_bilstm.add(Dense(1, activation='sigmoid'))
model_bilstm.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
model_bilstm.summary()
history_bilstm = model_bilstm.fit(X_train, y_train, batch_size=512, epochs=20, verbose=1, validation_split=0.2)
score_bilstm = model_bilstm.evaluate(X_test, y_test, verbose=1)

print("Loading User Rating Database...")
rating = pd.read_csv("db/07-user-ratings.csv")
r_normal = rating.NORMAL.values.tolist()
r_cnn = rating.CNN.values.tolist()
r_lstm = rating.LSTM.values.tolist()
r_bilstm = rating.BILSTM.values.tolist()

print("Loading Main Tweet Database...")
dataset = pd.read_csv('db/08-shortlisted-tweets.csv')
user = dataset.USER.values.tolist()
tweet = dataset.TWEET.values.tolist()
sentiment = dataset.SENTIMENT.values.tolist()

a, b, c, d = [], [], [], []
for x in tqdm(range(len(user))) :
    index = list(rating.loc[rating['USER'] == dataset['USER'][x]].index)[0]
    a.append(rating['NORMAL'][index])
    b.append(rating['CNN'][index])
    c.append(rating['LSTM'][index])
    d.append(rating['BILSTM'][index])
    
print("## making tokenizer with 10000 words...")
maxlen = 20
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(tweet)

print("## function to prep data to feed inside dataset")
def prep(array):
    array = tokenizer.texts_to_sequences(array)
    array = pad_sequences(array, padding='post', maxlen=maxlen)
    return array

array = []
for x in tqdm(range(len(dataset))) :
    array.append(prep([tweet[x]]))
    
print("## predicting using machine learning models...")
normal = []
cnn = []
lstm = []
bilstm = []

for index in tqdm(range(len(dataset))) :
    normal.append(model_normal.predict(array[index])[0][0])
    cnn.append(model_cnn.predict(array[index])[0][0])
    lstm.append(model_lstm.predict(array[index])[0][0])
    bilstm.append(model_bilstm.predict(array[index])[0][0])
    
print("Saving Dataset...")
final = pd.DataFrame(
    list(zip(user, sentiment, tweet, normal, cnn, lstm, bilstm, a, b, c, d)),
    columns=[
        'USER', 'SENTIMENT', 'TWEET', 
        'S_NORMAL', 'S_CNN', 'S_LSTM', 'S_BILSTM',
        'R_NORMAL', 'R_CNN', 'R_LSTM', 'R_BILSTM',
    ]
)
final.to_csv("db/09-final-prediction.csv", index=False)
