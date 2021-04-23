print("Load All Liberaries...")
import pandas as pd
from tqdm import tqdm
import numpy as np

print("Loading Datasets...")
df = pd.read_csv('db/05-user-tweets-without-sentiments.csv')
normal = pd.read_csv('db/06-user-tweets-normal.csv')
cnn = pd.read_csv('db/06-user-tweets-cnn.csv')
lstm = pd.read_csv('db/06-user-tweets-lstm.csv')
bilstm = pd.read_csv('db/06-user-tweets-bilstm.csv')

print("Extracting Values...")
user = df.USER.values.tolist()
tweet = df.TWEET.values.tolist()
normal = normal.NORMAL.values.tolist()
cnn = cnn.CNN.values.tolist()
lstm = lstm.LSTM.values.tolist()
bilstm = bilstm.BILSTM.values.tolist()

print("Saving Datasets...")
table = pd.DataFrame(
    list(zip(user, tweet, normal, cnn, lstm, bilstm)),
    columns = ['USER', 'TWEET', 'NORMAL', 'CNN', 'LSTM', 'BILSTM']
)
table.to_csv('db/06-user-tweets-with-sentiments.csv', index=False)