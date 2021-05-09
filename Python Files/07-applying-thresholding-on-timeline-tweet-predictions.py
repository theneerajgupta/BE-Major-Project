print("# loading required libraries")
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


print("# loading model with weights")
def load_model(model, weight) :
    with open(model, 'r') as file:
        yaml_model = file.read()

    model = tf.keras.models.model_from_yaml(yaml_model)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights(weight)

    return model

model = load_model('sentiment-analysis-model/model-gpu.yaml', 'sentiment-analysis-model/weights-gpu.h5')
model.summary()


print("# loading the tokenizer")
with open('sentiment-analysis-model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
    
print("# loading timeline tweets")
df = pd.read_csv("db/05-shortlisted-tweets.csv")


print("# creating 'X' array")
df.TWEET = df.TWEET.astype(str)
X = tokenizer.texts_to_sequences(df['TWEET'].values)
X = pad_sequences(X, 48)


print("# using the model to pred sentiment of each tweet")
pred = model.predict(X)
polarity = model.predict_classes(X)
prediction = []
for value in tqdm(pred) :
    prediction.append(value[1])
    
    
print("# applying thresholding...")
thresh = []
for value in tqdm(prediction) :
    if value < 0.25 :
        thresh.append(0)
    elif value < 0.5 :
        thresh.append(1)
    elif value < 0.75 :
        thresh.append(2)
    elif value <= 1 :
        thresh.append(3)
        
        
print("# saving results...")
final = pd.DataFrame(
    list(zip( df.USER.values.tolist(), df.TWEET.values.tolist(), df.ORIGINAL.values.tolist(), prediction, polarity, thresh )),
    columns = [ 'USER', 'TWEET', 'ORIGINAL', 'PREDICTION', 'SENTIMENT', 'THRESHOLD' ]
)
final.to_csv('db/07-timeline-tweets-with-thresholding.csv', index=False)