import numpy as np
from bs4 import BeautifulSoup
import os
from sklearn.feature_extraction.text import TfidfVectorizer
#for tokenization
from nltk.tokenize import word_tokenize
import re
import pandas as pd
import json
#import counter from collections
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score

textlist = []
dirCourse = os.listdir('./course-cotrain-data/fulltext/course')
dirNonCourse = os.listdir('./course-cotrain-data/fulltext/non-course')

# print(dir)
ps = PorterStemmer()
i=0
words = []
stop_words = set(stopwords.words('english'))
for file in dirCourse:
        with open(f'./course-cotrain-data/fulltext/course/{file}', 'r') as f:
            # f = open(f'./course-cotrain-data/fulltext/course/{file}', 'r')
            t = f.read().lower()
            soup = BeautifulSoup(t, 'html.parser')
            text = ""
            if soup.title:
                text = soup.title.text
            # text += soup.body.text
            if soup.body:
                text += soup.body.text.replace('\n', ' ')
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.compile(r'\d+').sub("", text)
            text = re.compile(r'\b' + r'\b|\b'.join(stop_words) + r'\b').sub("", text)
            words = word_tokenize(text)
            temp = []
            for word in words:
                temp.append(ps.stem(word))
                temp.append(" ")
            text = "".join(temp)
            textlist.append(text)

for file in dirNonCourse:
        with open(f'./course-cotrain-data/fulltext/non-course/{file}', 'r') as f:
            t = f.read().lower()
            soup = BeautifulSoup(t, 'html.parser')
            text = ""
            if soup.title:
                text = soup.title.text
            if soup.body:
                text += soup.body.text.replace('\n', ' ')
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.compile(r'\d+').sub("", text)
            text = re.compile(r'\b' + r'\b|\b'.join(stop_words) + r'\b').sub("", text)
            words = word_tokenize(text)
            temp = []
            for word in words:
                temp.append(ps.stem(word))
                temp.append(" ")
            text = "".join(temp)
            textlist.append(text)



# with open('text.txt', 'w') as fp:
#     for item in textlist:
#         fp.write("%s\n" % item)
#     print('Done')


tfidf = TfidfVectorizer()
response = tfidf.fit_transform(textlist)
feature_names = tfidf.get_feature_names_out()
dense = response.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
# print(feature_names)


dic = {}

for i in feature_names:
    dic[i] = df[i].tolist()
dicMag = {}
for key, value in enumerate(dic):
    # print(value,dic[value])
    vector = np.array(dic[value])
    magnitude = np.linalg.norm(vector)
    dicMag[value] = magnitude

tfidfVector = []
d = Counter(dicMag)
for value in d.most_common(100):
    tfidfVector.append(value[0])
# print(tfidfVector)

termsPresent = []





"""NAIVE BAYES ON TF-IDF"""
"""
for item in textlist:
    temp = ""
    string = ""
    string = str(item)
    for key, value in enumerate(tfidfVector):
        if value in string:
            temp += " " + value
        # else:
        #     temp.append(0)
    termsPresent.append(temp)



label = [1 if i < 230 else 0 for i in range(1051)]

df = pd.DataFrame(list(zip(termsPresent,label)), columns=['Terms', 'Label'])

X_train, X_test, y_train, y_test = train_test_split(df['Terms'], df['Label'], random_state=1)
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')

X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
top_words_df = word_freq_df.sum().sort_values(ascending=False)
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions))
print('Recall score: ', recall_score(y_test, predictions))

testing_predictions = []
for i in range(len(X_test)):
    if predictions[i] == 1:
        testing_predictions.append("Course")
    else:
        testing_predictions.append("Non-Course")
check_df = pd.DataFrame({"actual_label": list(y_test), "prediction": testing_predictions, "abstract":list(X_test)})
check_df.replace(to_replace=0, value="Non-Course", inplace=True)
check_df.replace(to_replace=1, value="Course", inplace=True)
# print(check_df)

# check_df.to_csv('df.csv', sep='\t')
"""

# with open('text.txt', 'w') as fp:
#     for item in textlist:
#         fp.write("%s\n" % item)
#     print('Done')






# docA = 'name wajahat multan'
# docB = 'name rehan sadiq'
# docC = 'name abdul rehman'
# docD = 'name subhan saleem'
# # docE = 'name saif saleem'
#
# # print(len(textlist))
#
# l = [ docB,docC,docD,docA]
# tfidf = TfidfVectorizer()
# response = tfidf.fit_transform(textlist)
# feature_names = tfidf.get_feature_names_out()
# dense = response.todense()
# denselist = dense.tolist()
# df = pd.DataFrame(denselist, columns=feature_names)
# print(df)
















# dic = {}
# # print(type(feature_names))
# for i in feature_names:
#     dic[i] = df[i].tolist()
# # print(dic)
#
# dicMag = {}
# for key, value in enumerate(dic):
#     # print(value,dic[value])
#     vector = np.array(dic[value])
#     magnitude = np.linalg.norm(vector)
#     dicMag[value] = magnitude
#
# # print(dicMag)
# # dicMag = sorted(dicMag.items(), key=lambda x: x[1], reverse=True)
# # print(dicMag)
#
#
# tdfidfVector = []
# d = Counter(dicMag)
# # d = most_common(dicMag)
# for value in d.most_common(100):
#     tdfidfVector.append(value[0])
#
# # print(tdfidfVector)

















# for i in range(10):
#
#     print("-------------------------------------------------------------\n",textlist[i])


    # text = soup.get_text()
    # print(text)
    # words = word_tokenize(text)
    # for word in words:
    #     wordlist.append(word)
# print(wordlist)

# with open('./course-cotrain-data/fulltext/course/http_^^www.cs.washington.edu^education^courses^135^',"r") as f:
#     doc = BeautifulSoup(f, 'html.parser')
# t = []
# t.append(doc.body.text.replace('\n', ' '))
#
# t.append(doc.body.text)
# print(doc.body)

# text = doc.title.text
# text = word_tokenize(text)
# text += word_tokenize(doc.body.text)
#
# x = doc.body.text.replace('\n', ' ')
# x = x.strip()
# st = "".join(x.split())
# for t in text:
#     st += ' '  + t
# print(x)




