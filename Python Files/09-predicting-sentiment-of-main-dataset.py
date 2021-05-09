print("# loading required libraries...")
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
from sklearn.metrics import confusion_matrix
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from sklearn.metrics import confusion_matrix, accuracy_score


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


print("# loading the tokenizer...")
with open('sentiment-analysis-model/tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)
    
    
print("# loading maindata...")
df = pd.read_csv("db/04-main-data.csv")
rating = pd.read_csv("db/08-user-rating.csv")


print("# dropping unnecessary rows...")
df_user = df.USER.values.tolist()
rating_user = rating.USER.values.tolist()
todrop_user = list(set(df_user) - set(rating_user))
for name in tqdm(todrop_user) :
    index = df.loc[df.USER == name].index
    df.drop(index, inplace=True)

df.reset_index(drop=True, inplace=True)
    
    
print("# prepping input for prediction...")
df.TEXT=df.TEXT.astype(str)
X = tokenizer.texts_to_sequences(df['TEXT'].values)
X = pad_sequences(X, 48)


print("# predicting sentiment...")
pred1 = []
temp = model.predict(X)
polarity = [round(value[1], 3) for value in temp]
for value in polarity :
    pred1.append(0 if value<0.5 else 1)


print("# fetching user rating...")
username = df.USER.values.tolist()
user_rating = []
for x in range(len(username)) :
    user_rating.append(int(rating.loc[rating['USER'] == username[x]]['RATING']))
    
    
print("# saving output...")    
final = pd.DataFrame(
    list(zip(
        df.USER.values.tolist(),
        user_rating,
        df.TEXT.values.tolist(),
        df.ORIGINAL.values.tolist(),
        df.SENTIMENT.values.tolist(),
        polarity,
        pred1)),
    columns = [ 'USER', 'RATING', 'TEXT', 'ORIGINAL', 'SENTIMENT', 'OUTPUT', 'PRED1' ]
)
final.to_csv("db/09-phase-1-prediction.csv", index=False)

cm = confusion_matrix(final.SENTIMENT.values.tolist(), final.PRED1.values.tolist())
acc = accuracy_score(final.SENTIMENT.values.tolist(), final.PRED1.values.tolist())
print("Printing Results:")
print(f"Accuracy: {acc}")