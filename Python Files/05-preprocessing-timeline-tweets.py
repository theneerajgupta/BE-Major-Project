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


print("# read user timeline tweets from dataframe")
df = pd.read_csv('db/03-user-tweets-english-only.csv' )


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


print("# preprocessing timeline tweets...")
user = df.USER.values.tolist()
tweet = df.TWEET.values.tolist()
processed = []
counts = []

for index in tqdm(range(len(df))) :
    text = preprocess_text(tweet[index])
    processed.append(text)
    counts.append(len(text.split()))


temp_df = pd.DataFrame(
    list(zip(user, tweet, processed)),
    columns = ['USER', 'ORIGINAL', 'PROCESSED']
)


print("# making a list of all usernames")
username = []
for x in tqdm(range(len(user))):
    if user[x] not in username :
        username.append(user[x])


print("# counting tweets by each user")
np_user = np.array(user)
tweetcount = []
for searchval in tqdm(username) :
    lst = list(np.where(np_user == searchval)[0])
    tweetcount.append(len(lst))


print("# shortlisting users with tweet count between 100 and 200")
shortlist = []
for x in tqdm(range(len(username))) :
    if (tweetcount[x] >= 100) and (tweetcount[x] <= 200) :
        shortlist.append(username[x])


print("# making final list of tweets and users")
final_user = []
final_tweet = []
final_original = []
for x in tqdm(range(len(user))) :
    if user[x] in shortlist :
        final_user.append(user[x])
        final_tweet.append(processed[x])
        final_original.append(tweet[x])


print("# creating dataframes")
final = pd.DataFrame(list(zip(final_user, final_tweet, final_original)), columns=['USER', 'TWEET', 'ORIGINAL'])
username = pd.DataFrame(shortlist, columns=['USER'])


print("# saving dataframes")
final.to_csv('db/05-shortlisted-tweets.csv', index=False)
username.to_csv('db/05-shortlisted-usernames.csv', index=False)
