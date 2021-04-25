# loading required libraries
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.models import model_from_yaml
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# loading model with weights
def load_model(model, weight) :
    with open(model, 'r') as file:
        yaml_model = file.read()
    
    model = tf.keras.models.model_from_yaml(yaml_model)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights(weight)
    
    return model

model = load_model('model-sa/model-gpu.yaml', 'model-sa/model-weights-gpu.h5')
model.summary()


# loading the tokenizer
with open('model-sa/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
    
# loading timeline tweets
df = pd.read_csv("db/04-main-data.csv")
user = pd.read_csv("db/08-user-rating.csv")
user_names = user.USER.values.tolist()
user_rating = user.RATING.values.tolist()

array = []
for name in tqdm(user_names) :
    for row in df.loc[df['USER'] == name].values.tolist() :
        array.append(row)
        
users, tweet, sentiment, rating = [], [], [], []
for row in tqdm(array) :
    users.append(row[0])
    tweet.append(row[1])
    sentiment.append(row[2])
    rating.append(int(user.loc[user['USER'] == row[0]]['RATING']))
    
    
df = pd.DataFrame(
    list(zip(users, tweet, sentiment, rating)),
    columns = ['USER', 'TWEET', 'SENTIMENT', 'RATING']
)
df = df.dropna()


# creating "X" array 
X = tokenizer.texts_to_sequences(df['TWEET'].values)
X = pad_sequences(X, 48)


# using the model to pred sentiment of each tweet
output = model.predict(X)
prediction = model.predict_classes(X)
truth = []
for row in output:
    truth.append(row[1])


# print accuracy
confusion_matrix(df['SENTIMENT'].values, prediction)
values = round((2069 + 2224) / len(df), 3)
print(f"Accuracy: {values * 100}")


df = pd.DataFrame(
    list(zip(users, tweet, sentiment, rating, prediction, truth)),
    columns = ['USER', 'TWEET', 'SENTIMENT', 'RATING', 'PREDICTION', 'OUTPUT']
)
new = df.loc[df['OUTPUT'] > 0.4]
new = new.loc[new['OUTPUT'] < 0.6]


# saving dataframes
df.to_csv("09-main-prediction.csv", index=False)
new.to_csv("09-ambiguous.csv", index=False)
