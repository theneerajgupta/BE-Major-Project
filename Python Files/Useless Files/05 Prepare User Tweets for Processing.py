print("## Loading all the required libraries...")
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

print("## Reading '03-user-tweets-english-only.csv' ...")
df = pd.read_csv('db/03-user-tweets-english-only.csv')
user = df.USER.values.tolist()
tweet = df.TWEET.values.tolist()

print("## define functions preprocess the dataset...")
def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

def preprocess_text(sen) :
    sentence = remove_tags(sen)
    sentence = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', sentence, flags=re.MULTILINE)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(' +', ' ', sentence)
    return sentence

user_new = []
tweet_new = []

for index in tqdm(range(len(df))):
    text = preprocess_text(tweet[index])
    if len(text.split()) > 2 :
        user_new.append(user[index])
        tweet_new.append(text)


username = []
for x in tqdm(range(len(user_new))):
    if user_new[x] not in username :
        username.append(user_new[x])

print("Analyze...")
np_user = np.array(user)
tweetcount = []

print("Counting Number of User Tweets...")
for searchval in tqdm(username) :
    lst = list(np.where(np_user == searchval)[0])
    tweetcount.append(len(lst))

print("Shortlisting Tweets...")
shortlist = []
for x in tqdm(range(len(username))) :
    if (tweetcount[x] >= 100) and (tweetcount[x] <= 200) :
        shortlist.append(username[x])

final_user = []
final_tweet = []

for x in tqdm(range(len(user_new))) :
    if user_new[x] in shortlist :
        final_user.append(user_new[x])
        final_tweet.append(tweet_new[x])

print("Making Database...")
final = pd.DataFrame(list(zip(final_user, final_tweet)), columns=['USER', 'TWEET'])
username = pd.DataFrame(shortlist, columns=['USER'])

print("Saving Database...")
final.to_csv('db/05-shortlisted-tweets.csv', index=False)
username.to_csv('db/05-shortlisted-usernames.csv', index=False)
