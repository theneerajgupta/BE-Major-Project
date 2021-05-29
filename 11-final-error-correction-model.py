import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from joblib import dump, load

df = pd.read_csv("db/09-phase-1-prediction.csv")
pos = pd.read_csv("db/10-positive-word-score.csv")
neg = pd.read_csv("db/10-negative-word-score.csv")
df.TEXT = df.TEXT.astype('str')


lst = pos.values.tolist()
for row in neg.values.tolist() :
    lst.append(row)
    
dictionary = dict(lst)


score = []

for x in range(len(df)) :
    text = df.TEXT[x]
    array = []
    for word in text.split() :
        if word in dictionary :
            array.append(dictionary[word] * 10)
        else :
            array.append(0)
    
    total = sum(array)        
    score.append(0 if total<0 else 1)
    
    
dataset = pd.DataFrame(
    list(zip(
        range(len(df)),
        df.USER.values.tolist(),
        df.ORIGINAL.values.tolist(),
        df.TEXT.values.tolist(),
        df.RATING.values.tolist(),
        df.OUTPUT.values.tolist(),
        df.SENTIMENT.values.tolist(),
        df.PRED1.values.tolist(),
        score)),
    columns = [ 'INDEX', 'USER', 'ORIGINAL', 'TEXT', 'RATING', 'OUTPUT', 'SENTIMENT', 'PRED1', 'SENT_SCORE'],
)
dataset.to_csv('db/11-final-table.csv', index=False)


dataset = dataset[['RATING', 'OUTPUT', 'SENT_SCORE', 'PRED1', 'SENTIMENT']]
X = dataset[['OUTPUT', 'RATING', 'SENT_SCORE']]
Y = dataset['SENTIMENT']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
clf = SVC()
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)
print(f"Accuracy: {round(accuracy_score(Y_test, pred)*100, 3)}%")
dump(clf, 'classification-model/clf')