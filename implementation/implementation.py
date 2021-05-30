print("# Loading all Libraries")
import re
import nltk
import tweepy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from joblib import dump, load
from tweepy import OAuthHandler
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')

print("# loading model with weights")
def load_model(model, weight) :
    with open(model, 'r') as file:
        yaml_model = file.read()

    model = tf.keras.models.model_from_yaml(yaml_model)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights(weight)

    return model

model = load_model('../sentiment-analysis-model/model-cpu.yaml', '../sentiment-analysis-model/weights-cpu.h5')
# model.summary()

print("# loading the tokenizer")
with open('../sentiment-analysis-model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

classifier = load('../classification-model/clf')

consumer_key = 'e3quFb7yTv8RJBfJtcsH172ey'
consumer_secret =  'SI8hYfTDQ6t90DVzk8saJlbp3Frz9eo0IWW9qCBK5JzgLj4ofa'
access_token = '724078891384582144-KG0kZkal2PbRFiXOQva8Uatull9qVRx'
access_token_secret = 'JWZlsm0KYB4vkjzc2CuJOkVaoym0L2Ts2lK9bBhSRMm3t'

try :
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    print("Twitter API Ready!")
except :
    print("Authentication Failed!")

class Twitter_User():
    def __init__(self,username,count=200):
        self.id = api.get_user(username).id
        self.username = username
        self.count = count
        self.data = None

    def get_tweets(self):
        store_tweets = api.user_timeline(self.id, count=self.count)
        simple_list = []
        for status in store_tweets:
            array = [status._json["text"].strip()]
            simple_list.append(array)

        self.data = pd.DataFrame(simple_list, columns=["TEXT"])
        self.data = self.data[~self.data["TEXT"].str.startswith('RT')]
        return self.data


    def __repr__(self):
        id = api.get_user(self.id)
        return id.screen_name

def process_url(url) :
    url = re.findall('http[s]?://twitter.com/(?:[a-zA-Z]|[0-9])+/status/[0-9]+', url)
    try :
        url = url[0].split("/")
        user_id = url[3]
        tweet_id = url[5]
        return (user_id, tweet_id)
    except :
        return False

def tweet_info(url) :
    tweet = api.get_status(process_url(url)[1])

    tweet_keys = ['created_at', 'id', 'text', 'retweet_count', 'favorite_count']
    tweet_values = []
    for key in tweet_keys :
        tweet_values.append(tweet._json[key])

    user_keys = ['id', 'name', 'screen_name', 'description', 'followers_count', 'friends_count', 'profile_image_url_https']
    user_values = []
    for key in user_keys :
        user_values.append(tweet._json['user'][key])

    tweet_info = dict(zip(tweet_keys, tweet_values))
    user_info = dict(zip(user_keys, user_values))

    return (user_info, tweet_info)

porter = PorterStemmer()
sw = stopwords.words('english')
sw.remove('not')

def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

def remove_single_chars(text) :
    array = text.split()
    return (" ".join([w for w in array if len(w) > 1]))

def remove_stopwords(text) :
    text = " ".join([word for word in text.split() if word not in sw])
    return text

def apply_stemming(text) :
    arr1 = text.split(" ")
    arr2 = []
    for word in arr1 :
        arr2.append(porter.stem(word))
    text = " ".join(arr2)
    return text

def preprocess_text(sen) :
    sentence = remove_tags(sen)
    sentence = sentence.lower()
    sentence = re.sub('@[A-Za-z]+[A-Za-z0-9-_]+', '', sentence)
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = remove_stopwords(sentence)
    sentence = remove_single_chars(sentence)
    sentence = apply_stemming(sentence)
    return sentence

def calculate_rating(user_screen_name) :
    user = Twitter_User(user_screen_name)
    tweets = list(user.get_tweets().TEXT.values.tolist())

    # preprocess all tweets
    preprocessed = []
    for sent in tweets :
        preprocessed.append(preprocess_text(sent))

    # tokenize and pad all tweets
    X = tokenizer.texts_to_sequences(preprocessed)
    X = pad_sequences(X, 48)

    # predict sentiment
    pred = model.predict(X)
    prediction = []
    for value in pred :
        prediction.append(value[1])

    # calculate rating :
    score = 0
    for value in prediction :
        if value < 0.3 :
            score = score - 1
        elif value > 0.7 :
            score = score + 1

    return 0 if score < 0 else 1

def calculate_sentiment(tweet) :
    X = tokenizer.texts_to_sequences([tweet])
    X = pad_sequences(X, 48)
    prediction = model.predict(X)[0][1]
    return prediction

pos = pd.read_csv("../../db/10-positive-word-score.csv")
neg = pd.read_csv("../../db/10-negative-word-score.csv")
lst = pos.values.tolist()
for row in neg.values.tolist() :
    lst.append(row)

dictionary = dict(lst)

def calculate_sent_score(text) :
    arr = text.split(" ")
    score = 0
    for word in arr :
        if word in dictionary :
            score = score + dictionary[word]

    return 0 if score < 0 else 1



if __name__ == "__main__" :
    
    def predictor(url) :
        user, tweet = tweet_info(url)
        userid = user['screen_name']
        text = tweet['text']
        newtext = preprocess_text(tweet['text'])

        prediction = calculate_sentiment(newtext)
        rating = calculate_rating(userid)
        score = calculate_sent_score(newtext)

        df = pd.DataFrame([[prediction, rating, score]])
        prediction = classifier.predict(df)[0]

        print(f"USERNAME  : {user['name']}")
        print(f"USERID    : {user['screen_name']}")
        print(f"TWEET     : {tweet['text']}")
        print(f"SENTIMENT : {'POSITIVE' if prediction == 1 else 'NEGATIVE'}")

    while True :
        print("PRESS CTRL + C TO EXIT")
        print("===============================================================")
        text = input("url >> ")
        predictor(text)
        print("===============================================================")
