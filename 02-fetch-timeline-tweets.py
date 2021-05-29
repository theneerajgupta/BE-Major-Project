print("# Loading all Libraries")
import pandas as pd
import numpy as np
import tweepy
from tweepy import OAuthHandler
from tqdm import tqdm


print("# loading dataframe containing all usernames")
df = pd.read_csv('db/01-select-users.csv')
df['TWEET'] = None


print("# setting up Twitter API using 'Tweepy'")
consumer_key = 'e3quFb7yTv8RJBfJtcsH172ey'
consumer_secret =  'SI8hYfTDQ6t90DVzk8saJlbp3Frz9eo0IWW9qCBK5JzgLj4ofa'
access_token = '724078891384582144-KG0kZkal2PbRFiXOQva8Uatull9qVRx'
access_token_secret = 'JWZlsm0KYB4vkjzc2CuJOkVaoym0L2Ts2lK9bBhSRMm3t'
try :
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    print("Authentication Successful!")
except :
    print("Authentication Failed!")


print("# create function to fetch tweets")
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
            array = [status._json["text"].strip(), status._json["favorite_count"], status._json["created_at"],status._json["retweet_count"],[h["text"] for h in status._json["entities"]["hashtags"]]]
            simple_list.append(array)
        self.data = pd.DataFrame(simple_list, columns=["Text", "Like", "Created at","Retweet","Hashtags"])
        self.data = self.data[~self.data["Text"].str.startswith('RT')]
        return self.data


    def __repr__(self):
        id = api.get_user(self.id)
        return id.screen_name

def fetch_tweets(username) :
    user = Twitter_User(username)
    tweets = user.get_tweets()
    for tweet in tweets['Text']:
        all_tweets.loc[len(all_tweets.index)] = [username, tweet]

all_tweets = pd.DataFrame(columns = ['USERNAME', 'TWEETS'])
all_tweets = all_tweets[0:0]


print("# Fetching Tweets for Each User")
failed = []
for user in tqdm(df['USERNAME']) :
    try :
        fetch_tweets(user)
    except :
        failed.append(user)


# Analyzing Fetched Data
# print(len(failed))
# print(failed)
# counts = []
# for user in all_tweets['USERNAME']:
#     counts.append(user)
# len(counts)
# my_dict = {i:counts.count(i) for i in counts}
# print(my_dict)
# all_tweets.head()


print("# Saving Dataset")
all_tweets.to_csv('db/02-user-tweets.csv', index=False)

