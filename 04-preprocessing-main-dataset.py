print("# loading required python libraries...")
import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


print("# loading '1600k-noemoticon.csv' dataset")
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


print("# create functions that will preprocess the dataset")
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

def preprocess_text(sen) :
    sentence = remove_tags(sen)
    sentence = sentence.lower()
    sentence = re.sub('@[A-Za-z]+[A-Za-z0-9-_]+', '', sentence)
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = remove_stopwords(sentence)
    sentence = remove_single_chars(sentence)
    return sentence


print("# apply preprocessing on all tweets")
for node in tqdm(dataset):
    if node[1] > 1 :
        node[1] = 1
    node[3] = preprocess_text(node[2])


print("# create a dataframe")
df = pd.DataFrame(dataset, columns=['USER', 'SENTIMENT', 'ORIGINAL', 'TEXT'])
# df = df[['USER', 'TEXT', 'SENTIMENT']]


print("# split the dataframe")
df_list = df.values.tolist()
df_main = df_list[:10000] + df_list[800000:810000]
df_train = df_list[10000:800000] + df_list[810000:]


print("# making final dataframes")
main = pd.DataFrame(df_main, columns=['USER', 'SENTIMENT', 'ORIGINAL', 'TEXT'])
train = pd.DataFrame(df_train, columns=['USER', 'SENTIMENT', 'ORIGINAL', 'TEXT'])


print("# saving final dataframes")
main.to_csv('db/04-main-data.csv', index=None)
train.to_csv('db/04-train-data.csv', index=None)
