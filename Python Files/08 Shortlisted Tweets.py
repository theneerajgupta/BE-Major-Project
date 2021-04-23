print("Importing required libraries...")
import pandas as pd
from tqdm import tqdm

print("Loading the User Details Dataset...")
user = pd.read_csv('db/07-user-ratings.csv')
username = list(set(user.USER.values.tolist()))

print("Loading the Mega Dataset...")
main = pd.read_csv('db/04-main-data.csv')
main = main.values.tolist()

final = []
for row in tqdm(main) :
    if row[0] in username :
        final.append(row)
        
final = pd.DataFrame(final, columns=['USER', 'TWEET', 'SENTIMENT'])
final.to_csv("db/08-shortlisted-tweets.csv", index=False)