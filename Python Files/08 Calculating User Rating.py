# import required libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


# loading preprocessed dataset...
users = pd.read_csv("db/07-timeline-tweets-with-thresholding.csv")
users = users.values.tolist()


# convert strings into a integer array
username = []
sentiment = []
threshold = []
for row in users :
    username.append(row[0])
    sentiment.append([ int(val) for val in row[1][1:-1].split(", ") ])
    threshold.append([ int(val) for val in row[2][1:-1].split(", ") ])
    
    
# create function to rate the user
def plot_values(index) :
    senti = sentiment[index]
    thresh = threshold[index]
    matrix = [
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    for x in range(len(senti)) :
        matrix[senti[x]][thresh[x]] += 1

#     calc = sum(matrix[1]) - sum(matrix[0])
    calc = ((matrix[0][3] + matrix[1][3] - matrix[0][0] - matrix[1][0]) + (matrix[0][2] + matrix[1][2] - matrix[0][1] - matrix[1][1])) / 2.0
    
    if calc > 0 :
        return 1
    else :
        return 0
      

# caluclate user rating
results = []
for x in range(len(users)) :
    results.append(plot_values(x))
    
    
# create dataframe
final = pd.DataFrame(
    list(zip(pd.read_csv("db/07-timeline-tweets-with-thresholding.csv").USER.values.tolist(), results)),
    columns = ['USER', 'RATING']
)


# save dataframe
final.to_csv("db/08-user-rating.csv", index=False)
