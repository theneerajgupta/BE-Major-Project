print("# load required libraries")
import re
import keras
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

from tensorflow.keras.models import save_model, load_model
from keras.models import model_from_yaml

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)




print("# loading the training dataset")
data = pd.read_csv('db/04-train-data.csv')
data = data[['TEXT', 'SENTIMENT']]
data.TEXT = data.TEXT.astype('str')
data = data.dropna(subset=['TEXT'])
data = data.sample(frac=1).reset_index(drop=True)




print("creating the tokenizer")
max_features = 5000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['TEXT'].values)




print("# saving the tokenizer")
with open('sentiment-analysis-model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)




# loading the tokenizer
# with open('sentiment-analysis-model/tokenizer.pickle', 'rb') as handle:
#     new_tokeinzer = pickle.load(handle)




print("# Creating 'X' and 'Y' arrays")
X = tokenizer.texts_to_sequences(data['TEXT'].values)
X = pad_sequences(X)
Y = pd.get_dummies(data['SENTIMENT']).values




print("# Spliting 'X' and 'Y' into training and validation datasets")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)




# making cpu-based machine learning model
embed_dim = 128
lstm_out = 196
model1 = Sequential()
model1.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model1.add(SpatialDropout1D(0.4))
model1.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model1.add(SpatialDropout1D(0.4))
model1.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model1.add(Dense(2,activation='softmax'))
model1.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
model1.summary()
history1 = model1.fit(X_train, Y_train, epochs = 10, batch_size = 256, verbose = 1, validation_split=0.2)




print("# making gpu-based machine learning model")
embed_dim = 128
lstm_out = 196
model2 = Sequential()
model2.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model2.add(SpatialDropout1D(0.4))
model2.add(CuDNNLSTM(lstm_out, return_sequences=True))
model2.add(SpatialDropout1D(0.4))
model2.add(CuDNNLSTM(lstm_out))
model2.add(Dense(2,activation='softmax'))
model2.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
model2.summary()
history2 = model2.fit(X_train, Y_train, epochs = 10, batch_size = 512, verbose = 1, validation_split=0.2)




print("# function to save accuracy graphs")
def save_accuracy_plot(history, name="accuracy.png") :
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(name)
    plt.show()
    
save_accuracy_plot(history1, "graphs/cpu-accuracy.png")
save_accuracy_plot(history2, "graphs/gpu-accuracy.png")




print("# function to save loss graphs")
def save_loss_plot(history, name="loss.png") :
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(name)
    plt.show()
    
save_loss_plot(history1, "graphs/cpu-loss.png")
save_loss_plot(history2, "graphs/gpu-loss.png")




print("# function to test and save the model with weigths")
def test_and_save(model, name) :
    
    # test model
    score = model.evaluate(X_test, Y_test, verbose=1)
    print(f"Evaluation Results :")
    print(f"Loss    : {score[0]}")
    print(f"Accuracy: {score[1]}\n")
    
    # saving model as a yaml file
    yaml = model.to_yaml()
    with open("".join(["sentiment-analysis-model/model-", name, ".yaml"]), 'w') as file:
        file.write(yaml)

    # saving model weights
    model.save_weights("".join(["sentiment-analysis-model/weights-", name, ".h5"]))
    
    
test_and_save(model1, "cpu")
test_and_save(model2, "gpu")





print("# function to load the model - just in case")
def load_model(model, weight) :
    with open(model, 'r') as file:
        yaml_model = file.read()
    
    model = tf.keras.models.model_from_yaml(yaml_model)
    model.load_weights(weight)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# loaded1 = load_model(
#     "sentiment-analysis-model/model-cpu.yaml", 
#     "sentiment-analysis-model/weights-cpu.h5"
# )
# loaded1.summary()

loaded2 = load_model(
    "sentiment-analysis-model/model-gpu.yaml", 
    "sentiment-analysis-model/weights-gpu.h5"
)
loaded2.summary()



loaded1.evaluate(X_test[:1000], Y_test[:1000], verbose=1)
loaded2.evaluate(X_test[:1000], Y_test[:1000], verbose=1)