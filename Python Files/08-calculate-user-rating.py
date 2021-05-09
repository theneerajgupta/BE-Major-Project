print("# import required libraries...")
import numpy as np
import pandas as pd
from tqdm import tqdm


print("# loading preprocessed dataset...")
df = pd.read_csv("db/07-timeline-tweets-with-thresholding.csv")
username = list(set(df.USER.tolist()))


print("# make of list all the index for each tweet for each user...")
rows = []
for name in tqdm(username) :
    rows.append(df.loc[df['USER'] == name].index)
    
    
print("# function to calculate user rating...")
def calculate_rating(index) :
    index = rows[index]
    thresh = []
    for x in index :
        thresh.append(df['THRESHOLD'][x])

    calc = (thresh.count(3) - thresh.count(0))
#     calc = ((thresh.count(3) - thresh.count(0)) + (thresh.count(2) - thresh.count(1))) / 2
#     calc = ( (thresh.count(3) + thresh.count(2)) / 2 ) - ( (thresh.count(1) + thresh.count(0)) / 2 )
   
    return 0 if calc < 0 else 1


print("# calculating user rating...")
rating = []
for x in tqdm(range(len(username))) :
    rating.append(calculate_rating(x))
    
        
print("# saving rating dataframe...")
final = pd.DataFrame(
    list(zip(username, rating)),
    columns =['USER', 'RATING']
)
final.to_csv("db/08-user-rating.csv", index=False)