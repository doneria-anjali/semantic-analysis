
# coding: utf-8

# In[36]:


import nltk
import math
import sklearn
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
#C:/Users/Anu/Desktop/NCSU/CSC522 (001) ALDA/Project
inp = pd.read_csv("./train.csv")
q1 = inp['question1']
q2 = inp['question2']

final_score = []

for everyQ in range(len(q1)):
    sent1 = q1[everyQ]
    sent2 = q2[everyQ]

    sent1 = re.sub(r'\d+', '', sent1)
    sent1 = sent1.lower()
    translator = str.maketrans('', '', string.punctuation)
    sent1 = sent1.translate(translator)

    sent2 = re.sub(r'\d+', '', sent2)
    sent2 = sent2.lower()
    translator = str.maketrans('', '', string.punctuation)
    sent2 = sent2.translate(translator)

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(sent1 + " " + sent2)
    word_tokens1 = word_tokenize(sent1)
    word_tokens2 = word_tokenize(sent2)

    filtered_sentence = []
    filtered_q1 = []
    filtered_q2 = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    filtered_sentence1 = []
    for x in filtered_sentence:
        if x not in filtered_sentence1:
            filtered_sentence1.append(x)

    for w in word_tokens1:
        if w not in stop_words:
            filtered_q1.append(w)

    filtered_q11 = []
    for x in filtered_q1:
        if x not in filtered_q11:
            filtered_q11.append(x)

    for w in word_tokens2:
        if w not in stop_words:
            filtered_q2.append(w)

    filtered_q21 = []
    for x in filtered_q2:
        if x not in filtered_q21:
            filtered_q21.append(x)

    sparse_matrix = np.zeros((len(filtered_sentence1), len(filtered_sentence1)))
    for i in range(len(filtered_sentence1)):
        for j in range(len(filtered_sentence1)):
            try:
                sparse_matrix[i][j] = wn.wup_similarity(wn.synset(filtered_sentence1[i] + '.v.01'),
                                                        wn.synset(filtered_sentence1[j] + '.v.01'))
                # sparse_matrix[i][j] = wn.synset(filtered_sentence1[j])[0].wup_similarity(wn.synset(filtered_sentence1[i])[0])
            except:
                if i != j:
                    sparse_matrix[i][j] = 0
                else:
                    sparse_matrix[i][j] = 1.0

    BinaryQ1 = [1 if i in filtered_q1 else 0 for i in filtered_sentence1]
    BinaryQ2 = [1 if i in filtered_q2 else 0 for i in filtered_sentence1]

    matmult1 = np.matmul(BinaryQ1, sparse_matrix)

    matmult2 = np.matmul(matmult1, np.transpose(BinaryQ2))

    denom = math.sqrt(sum(i ** 2 for i in BinaryQ1)) * math.sqrt(sum(i ** 2 for i in BinaryQ2))
    if denom != 0:
        res = matmult2 / denom
    else:
        res = 0
    final_score.append(res)
final_score1 = [1 if x >= 0.8 else 0 for x in final_score]
print(confusion_matrix(inp['is_duplicate'], final_score1, labels=[0, 1]))
print(classification_report(inp['is_duplicate'], final_score1, target_names=['Class 0', 'Class 1']))


# In[ ]:


import nltk
import math
import sklearn
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
inp = pd.read_csv("./train.csv")
q1 = inp['question1']
q2 = inp['question2']

final_score = []

for everyQ in range(len(q1)):
    sent1 = q1[everyQ]
    sent2 = q2[everyQ]

    sent1 = re.sub(r'\d+', '', sent1)
    sent1 = sent1.lower()
    translator = str.maketrans('', '', string.punctuation)
    sent1 = sent1.translate(translator)

    sent2 = re.sub(r'\d+', '', sent2)
    sent2 = sent2.lower()
    translator = str.maketrans('', '', string.punctuation)
    sent2 = sent2.translate(translator)

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(sent1 + " " + sent2)
    word_tokens1 = word_tokenize(sent1)
    word_tokens2 = word_tokenize(sent2)

    filtered_sentence = []
    filtered_q1 = []
    filtered_q2 = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    filtered_sentence1 = []
    for x in filtered_sentence:
        if x not in filtered_sentence1:
            filtered_sentence1.append(x)

    for w in word_tokens1:
        if w not in stop_words:
            filtered_q1.append(w)

    filtered_q11 = []
    for x in filtered_q1:
        if x not in filtered_q11:
            filtered_q11.append(x)

    for w in word_tokens2:
        if w not in stop_words:
            filtered_q2.append(w)

    filtered_q21 = []
    for x in filtered_q2:
        if x not in filtered_q21:
            filtered_q21.append(x)

    sparse_matrix = np.zeros((len(filtered_sentence1), len(filtered_sentence1)))
    for i in range(len(filtered_sentence1)):
        for j in range(len(filtered_sentence1)):
            try:
                sparse_matrix[i][j] = wn.wup_similarity(wn.synset(filtered_sentence1[i] + '.v.01'),
                                                        wn.synset(filtered_sentence1[j] + '.v.01'))
                # sparse_matrix[i][j] = wn.synset(filtered_sentence1[j])[0].wup_similarity(wn.synset(filtered_sentence1[i])[0])
            except:
                if i != j:
                    sparse_matrix[i][j] = 0
                else:
                    sparse_matrix[i][j] = 1.0

    BinaryQ1 = [1 if i in filtered_q1 else 0 for i in filtered_sentence1]
    BinaryQ2 = [1 if i in filtered_q2 else 0 for i in filtered_sentence1]

    matmult1 = np.matmul(BinaryQ1, sparse_matrix)

    matmult2 = np.matmul(matmult1, np.transpose(BinaryQ2))

    denom = math.sqrt(sum(i ** 2 for i in BinaryQ1)) * math.sqrt(sum(i ** 2 for i in BinaryQ2))
    if denom != 0:
        res = matmult2 / denom
    else:
        res = 0
    final_score.append(res)
final_score1 = [1 if x >= 0.8 else 0 for x in final_score]
print(confusion_matrix(inp['is_duplicate'], final_score1, labels=[0, 1]))
print(classification_report(inp['is_duplicate'], final_score1, target_names=['Class 0', 'Class 1']))

