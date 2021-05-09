print("# loading required python libraries...")
import pandas as pd
import nltk
from collections import Counter


print("# loading training dataset...")
df = pd.read_csv("db/04-train-data.csv")
df = df[['SENTIMENT', 'TEXT']]
df.TEXT = df.TEXT.astype('str')


print("# seperating positive and negative labelled tweets...")
pos = df.loc[df.SENTIMENT == 1].TEXT.values.tolist()
neg = df.loc[df.SENTIMENT == 0].TEXT.values.tolist()


print("# creating counters...")
pos_counts = Counter()
neg_counts = Counter()
total_counts = Counter()


print("# counting the number of postive words...")
for i in range(len(pos)):
    for word in pos[i].lower().split(" "):
        pos_counts[word]+=1
        total_counts[word]+=1


print("# counting the number of negative words...")
for i in range(len(neg)):
    for word in neg[i].lower().split(" "):
        neg_counts[word]+=1
        total_counts[word]+=1


print("# calculating postive/negative ratio...")
pos_neg_score = Counter()
for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = pos_counts[term] / float(neg_counts[term] + 1)
        pos_neg_score[term] = pos_neg_ratio


print("# seperating positive and negative words...")
pnscore_list = pos_neg_score.most_common()
pnscore = pd.DataFrame(pnscore_list, columns = ['WORD', 'SCORE']) 
pnscore = pnscore.loc[pnscore.SCORE < 30]
pscore = pnscore[:4000].reset_index(drop=True)
nscore = pnscore[4000:].reset_index(drop=True)


print("# normalizing score...")
minimum = min(pscore.SCORE.values.tolist())
maximum = max(pscore.SCORE.values.tolist())
words = pscore.WORD.values.tolist()
score = pscore.SCORE.values.tolist()
new_score = []
for value in score :
    new_score.append((value - minimum)/(maximum - minimum))
        
pscore = pd.DataFrame(list(zip(words, new_score)), columns=['WORDS', 'SCORE'])

minimum = min(nscore.SCORE.values.tolist())
maximum = max(nscore.SCORE.values.tolist())
words = nscore.WORD.values.tolist()
score = nscore.SCORE.values.tolist()
new_score = []
for value in score :
    new_value = float(-(1 - ((value - minimum)/(maximum - minimum))))
    new_score.append(new_value)
        
nscore = pd.DataFrame(list(zip(words, new_score)), columns=['WORDS', 'SCORE'])


print("# saving datasets...")
pscore.to_csv('db/10-positive-word-score.csv', index=False)
nscore.to_csv('db/10-negative-word-score.csv', index=False)