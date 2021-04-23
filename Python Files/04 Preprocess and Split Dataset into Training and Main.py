print("# import all standard libraries...")
import re
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
import matplotlib.pyplot as plt


print("# import all machine learning libraries...")
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


print("# read the required dataset and clean it...")
df = pd.read_csv("db/1600k-noemoticon.csv", header=None)
df.isnull().values.any()
df.rename(
    columns = {
    0: 'SENTIMENT',
    1: 'ID',
    2: 'DATE',
    3: 'QUERY',
    4: 'USERNAME',
    5: 'TWEET'
    }, inplace=True, errors='raise'
)
df = df[['USERNAME', 'SENTIMENT', 'TWEET']]
df['TEXT'] = ""
dataset = df.values.tolist()
for col in df.columns:
    print(col)


print("# define functions preprocess the dataset...")
def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

def remove_single_chars(text) :
    array = text.split()
    return (" ".join([w for w in array if len(w) > 1]))

def preprocess_text(sen) :
    sentence = remove_tags(sen)
    sentence = re.sub('@[A-Za-z]+[A-Za-z0-9-_]+', '', sentence)
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub('/\b\S\s\b/', "", sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = remove_stopwords(sentence)
    sentence = remove_single_chars(sentence)
    return sentence


for node in tqdm(dataset):
    if node[1] > 1 :
        node[1] = 1
    node[3] = preprocess_text(node[2])


df = pd.DataFrame(dataset, columns=['USER', 'SENTIMENT', 'TWEET', 'TEXT'])
df = df[['USER', 'TEXT', 'SENTIMENT']]

df_list = df.values.tolist()
df_main = df_list[:10000] + df_list[800000:810000]
df_train = df_list[10000:800000] + df_list[810000:]

final_main = pd.DataFrame(df_main, columns=['USER', 'TEXT', 'SENTIMENT'])
final_train = pd.DataFrame(df_train, columns=['USER', 'TEXT', 'SENTIMENT'])

final_main.to_csv('db/04-main-data.csv', index=None)
final_train.to_csv('db/04-train-data.csv', index=None)
