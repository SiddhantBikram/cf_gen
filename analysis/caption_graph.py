import pickle
import os
from configs import *
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


with open(os.path.join(root_dir, 'caps_car'), 'rb') as fp:
    caps = pickle.load(fp)

# nltk.download('stopwords')
stop_words = stopwords.words('english')

def mostCommonWords(concordanceList):
    finalCount = Counter()
    for line in concordanceList:
        words = [w for w in line.split(" ") if w not in stop_words]
        finalCount.update(words)  # update final count using the words list
    return finalCount

counts = mostCommonWords(caps)
D = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
D = dict(list(D.items())[:20])

plt.bar(range(len(D)), list(D.values()), align='center')
plt.xticks(range(len(D)), list(D.keys()))

plt.show()