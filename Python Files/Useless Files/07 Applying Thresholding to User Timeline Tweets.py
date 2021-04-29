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
df = pd.read_csv("db/05-shortlisted-tweets.csv")


# creating "X" array 
X = tokenizer.texts_to_sequences(df['TWEET'].values)
X = pad_sequences(X, 48)


# using the model to pred sentiment of each tweet
output = model.predict(X)
C = model.predict_classes(X)


# applying triple thresholding to avoid amibuity
triple = []
Y = []
for value in output :
    Y.append(value[1])
    if value[1] < 0.25 :
        triple.append(0)
    elif value[1] < 0.5 :
        triple.append(1)
    elif value[1] < 0.75 :
        triple.append(2)
    elif value[1] <= 1 :
        triple.append(3)
        
        
# update the main dataframe
df = pd.DataFrame(
    list(zip( df.USER.values.tolist(), df.TWEET.values.tolist(), Y, C, triple )),
    columns = [ 'USER', 'TWEET', 'PREDICTION', 'SENTIMENT',  'TRIPLE_THRESH' ]
)


# get usernames from shortlisted-username dataframe
user = pd.read_csv("db/05-shortlisted-usernames.csv")
user = user.USER.values.tolist()


# make list of sentiment and thresholded-sentiment for all the users
index = []
count = []
for name in tqdm(user) :
    array = list(df.loc[df['USER'] == name].index)
    index.append(array)
    count.append(len(array))
    
    
# filter out users with less than 100 tweets
temp = pd.DataFrame(list(zip(user, count)), columns=['USER', 'COUNT'])
temp = temp.drop(list(temp.loc[temp['COUNT'] < 100].index))


# creating final user dataframe
triple = []
sentiment = []
usernames = temp.USER.values.tolist()
for name in tqdm(usernames) :
    frame = df.loc[df['USER'] == name]
    sentiment.append(frame.SENTIMENT.values.tolist())
    triple.append(frame.TRIPLE_THRESH.values.tolist())
    
    
# saving final dataframe
final = pd.DataFrame(
    list(zip(temp.USER.values.tolist(), sentiment, triple)),
    columns = ['USER', 'SENTIMENT', 'TRIPLE_THRESH']
)
final.to_csv('db/07-timeline-tweets-with-thresholding.csv', index=False)
