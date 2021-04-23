print("Importing Required Libraries...")
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

print("Loading USER database...")
users = pd.read_csv("db/05-shortlisted-usernames.csv")
users = users.USER.values.tolist()

print("Loading PREDICTION database...")
pred = pd.read_csv("db/06-user-tweets-with-sentiments.csv")
pred.head()

def fetch(idx) :
    array = list(pred.loc[pred['USER'] == users[idx]].index)
    normal, cnn, lstm, bilstm = [], [], [], []
    a, b, c, d = 0, 0, 0, 0
    for row in array :
        normal.append(float(pred.NORMAL[row]))
        cnn.append(float(pred.CNN[row]))
        lstm.append(float(pred.LSTM[row]))
        bilstm.append(float(pred.BILSTM[row]))

    return [normal, cnn, lstm, bilstm]

def calculate(idx) :
    array = fetch(idx)
    calc = [[], [], [], []]
    rating = [0, 0, 0, 0]
    for model in range(len(array)) :
        for x in range(len(array[model])) :
            if array[model][x] < 0.25 :
                calc[model].append(0)
            elif array[model][x] < 0.50 :
                calc[model].append(0.25)
            elif array[model][x] < 0.75 :
                calc[model].append(0.75)
            else :
                calc[model].append(1)
        
        sec1 = calc[model].count(0)
        sec2 = calc[model].count(0.25)
        sec3 = calc[model].count(0.75)
        sec4 = calc[model].count(1)
        a = sec4 - sec1
        b = sec3 - sec2
        avg = (a + b) / 2
        rating[model] = avg
            
    return rating

print("Processing...")
normal = []
cnn = []
lstm = []
bilstm = []

for x in tqdm(range(len(users))) :
    calc = calculate(x)
    normal.append(calc[0])
    cnn.append(calc[1])
    lstm.append(calc[2])
    bilstm.append(calc[3])
    
error = []
for x in tqdm(range(len(users))) :
    if (normal[x] > 50) or (cnn[x] > 50) or (lstm[x] > 50) or (bilstm[x] > 50) :
        error.append(x)
    if (normal[x] < -50) or (cnn[x] < -50) or (lstm[x] < -50) or (bilstm[x] < -50) :
        error.append(x)
        
dataset = []
for x in tqdm(range(len(users))) :
    if x not in error :
        dataset.append([users[x], normal[x], cnn[x], lstm[x], bilstm[x]])
        
final = pd.DataFrame(dataset, columns=['USER', 'NORMAL', 'CNN', 'LSTM', 'BILSTM'])
final.to_csv("db/07-user-ratings.csv", index=False)