# Loading all required libraries
import concurrent.futures
import math
import pandas as pd
from tqdm import tqdm
from collections import Counter
import langid
from nltk.classify.textcat import TextCat
from langdetect import detect
import pickle
import glob
import re


# load '02-user-tweets.csv' dataframe
df = pd.read_csv("db/02-user-tweets.csv")
df = df.drop_duplicates()
df1 = pd.DataFrame(columns=['user', 'tweet'])
database = df.values.tolist()
output = []


# create a function to spilt data into small chunks
def make_pairs(end=1400000, divs=10000) :
    output = []
    var = 0
    while (var < end) :
        lst = [var, var + divs]
        output.append(lst)
        var = var + divs

    if (output[-1][1] > end) :
        remove = output.pop()
        output[-1][1] = end

    return output


# divide entire 'database' into chunks of size 15000
div_size = 20000
PAIRS = make_pairs(len(database), div_size)
# print(PAIRS)

# create function to filter out all tweets which are not in english
def process(endpoints):
    id = 0
    value = endpoints[0]

    while (value > 0) :
        value = value - div_size
        id = id + 1

    array = []
    print(f"Process ID {id} : Activated")
    for index in range(endpoints[0], endpoints[1]) :
            try :
                if detect(database[index][1]) == 'en' :
                    array.append(index)
            except :
                pass
    print(len(array))
            
    # print(f"{id} Processing Done : {yes} Tweets Detected")
    with open(f'buffer/{id}.txt', 'wb') as fo:
        pickle.dump(array, fo)
        print(f"Process ID {id} : Finished")



# funtion that would use multiprocessing...
def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process, PAIRS)


# function to read all saved file and combine into one single dataframe
def not_main() :
    files = glob.glob('buffer/*.txt')
    index = []
    for file in files:
        with open(file, 'rb') as fo:
            obj = pickle.load(fo)
            for row in obj :
                index.append(row)
    index = sorted(index)
    final = []
    for num in tqdm(index):
        final.append(database[num])
    final_db = pd.DataFrame(final, columns=['USER', 'TWEET'])
    final_db.to_csv('db/03-user-tweets-english-only.csv', index=False)


# executing the program
if __name__ == "__main__" :
	main()
	not_main()
