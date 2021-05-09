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

def make_pairs(end=1400000, divs=100000) :
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

df = pd.read_csv("db/02-user-tweets.csv")
df = df.drop_duplicates()
df1 = pd.DataFrame(columns=['user', 'tweet'])
database = df.values.tolist()
output = []

div_size = 15000
PAIRS = make_pairs(len(database), div_size)

def process(endpoints):

    id = 0
    value = endpoints[0]
    while (value > 0) :
        value = value - div_size
        id = id + 1

    yes = 0
    no = 0

    array = []


    for index in range(endpoints[0], endpoints[1]) :
            try :
                # database[index][1] = preprocess_text(database[index][1])
                if detect(database[index][1]) == 'en' :
                    yes = yes + 1
                    array.append(index)
                else :
                    no = no + 1
            except :
                no = no + 1

    # print(f"{id} Processing Done : {yes} Tweets Detected")
    with open(f'dataframe/{id}.txt', 'wb') as fo:
        pickle.dump(array, fo)
        print(f"{id}.txt Created!")


def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process, PAIRS)

def not_main() :
    files = glob.glob('dataframe/*.txt')
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

if __name__ == '__main__':
    main()
    not_main()
