import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('db/1600k-noemoticon.csv', header=None)
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

df.drop(['DATE', 'QUERY', 'ID'], axis = 1, inplace = True)
df['SENTIMENT'] = df['SENTIMENT'].replace([0],'NEG')
df['SENTIMENT'] = df['SENTIMENT'].replace([4],'POS')

print("PROCESSING TWEETS...")
users = []
size = 10000

for x in range(size) :
    users.append(df['USERNAME'][x])

for x in range(size) :
    users.append(df['USERNAME'][x+800000])

print("CREATING DATASET...")
user_df = pd.DataFrame(users, columns = ['USERNAME'])
user_df.to_csv('db/01-select-users.csv', index=False)
print("DATASET CREATED!")
