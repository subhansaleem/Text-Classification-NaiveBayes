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
from collections import Counter,defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords,wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag



textlist = []
fullText = ""
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
            fullText += " " + text
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
            fullText += " " + text
            words = word_tokenize(text)
            temp = []
            for word in words:
                temp.append(ps.stem(word))
                temp.append(" ")
            text = "".join(temp)
            textlist.append(text)


# json.dump(fullText, open("./dumps/fullText.txt", 'w'))
# json.dump(textlist, open("./dumps/textList.txt", 'w'))



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






"""NAIVE BAYES ON TF-IDF"""
"""
termsPresent = []

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

"""NOUN SELECTION"""
"""
fullVector = []
nounVector = []
d = Counter(dicMag)
for value in d.most_common():
    fullVector.append(value[0])

tagged = nltk.pos_tag(fullVector)
for (word, tag) in tagged:
    if tag == 'NN':  # If the word is a noun
        nounVector.append(word)

nounVector2 = [nounVector[i] for i in range(len(nounVector)) if i < 50]
# print(nounVector2)

coherenceTerms = []
for item in textlist:
    temp = set()
    string = ""
    # string = str(item)
    string = word_tokenize(item)
    for key, value in enumerate(nounVector2):
        for i,j in enumerate(string):
            if j == value:
                if i != 0 and i != len(string)-1:
                    temp.add(j)
                    temp.add(string[i-1])
                    temp.add(string[i+1])
            elif i == 0:
                temp.add(j)
            elif i == len(string)-1:
                temp.add(j)
    temp2 = ' '.join(temp)
    coherenceTerms.append(temp2)
# print(nounVector2)
#coherence terms == Terms

label = [1 if i < 230 else 0 for i in range(1051)]

df = pd.DataFrame(list(zip(coherenceTerms,label)), columns=['Terms', 'Label'])

X_train, X_test, y_train, y_test = train_test_split(df['Terms'], df['Label'], random_state=1)
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')

X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names_out())
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
print(check_df)
check_df.to_csv('NounsDF.csv', sep='\t')

"""


# position = ['NN', 'NNS', 'NNP', 'NNPS']
# with open('input.txt', "r") as f:
#     input_txt = f.read()
#     f.close()
#
# sentence = nltk.sent_tokenize(fullText)
# tokenizer = RegexpTokenizer(r'\w+')
# tokens = [tokenizer.tokenize(w) for w in sentence]
# tagged = [pos_tag(tok) for tok in tokens]
# nouns = [word for i in range(len(tagged)) for word, pos in tagged[i] if pos in position]

#
# relation_list = defaultdict(list)
#
# for k in range(len(nouns)):
#     relation = []
#     for syn in wordnet.synsets(nouns[k], pos=wordnet.NOUN):
#         for l in syn.lemmas():
#             relation.append(l.name())
#             if l.antonyms():
#                 relation.append(l.antonyms()[0].name())
#     relation_list[nouns[k]].append(relation)
#
# relation = relation_list
# json.dump(relation, open("./dumps/relationList.txt", 'w'))


# lexical = []
# threshold = 0.5
# for noun in nouns:
#     flag = 0
#     for j in range(len(lexical)):
#         if flag == 0:
#             for key in list(lexical[j]):
#                 if key == noun and flag == 0:
#                     lexical[j][noun] += 1
#                     flag = 1
#                 elif key in relation_list[noun][0] and flag == 0:
#                     syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
#                     syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
#                     if syns1 and syns2:
#                         if syns1[0].wup_similarity(syns2[0]) >= threshold:
#                             lexical[j][noun] = 1
#                             flag = 1
#                 elif noun in relation_list[key][0] and flag == 0:
#                     syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
#                     syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
#                     if syns1 and syns2:
#                         if syns1[0].wup_similarity(syns2[0]) >= threshold:
#                             lexical[j][noun] = 1
#                             flag = 1
#     if flag == 0:
#         dic_nuevo = {}
#         dic_nuevo[noun] = 1
#         lexical.append(dic_nuevo)
#         flag = 1

# json.dump(lexical, open("lexical.txt", 'w'))
# lexical = json.load(open("lexical.txt"))
#
# print("done")
#
# #
# final_chain = []
# while lexical:
#     result = lexical.pop()
#     if len(result.keys()) == 1:
#         for value in result.values():
#             if value != 1:
#                 final_chain.append(result)
#     else:
#         final_chain.append(result)
#
#
# # for i in range(len(final_chain)):
# #     print("Chain " + str(i + 1) + " : " + str(final_chain[i]))
# # print(final_chain[0])
#
# chainList = []
# for i in final_chain:
#     for item in i:
#         if i[item] > 1:
#             chainList.append(item)
# print(len(chainList))

# json.dump(chainList, open("./dumps/finalChainList.txt", 'w'))


"""NAIVE BAYES ON LEXICAL CHAIN"""
chainList = json.load(open("./dumps/finalChainList.txt"))
chainPresent = []
for item in textlist:
    # print(item)
    temp = ""
    string = ""
    string = word_tokenize(item)
    for key, value in enumerate(chainList):
        if value in string:
            temp += " " + value
    chainPresent.append(temp)


label = [1 if i < 230 else 0 for i in range(1051)]

df = pd.DataFrame(list(zip(chainPresent,label)), columns=['Terms', 'Label'])

X_train, X_test, y_train, y_test = train_test_split(df['Terms'], df['Label'], random_state=1)
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')

X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names_out())
top_words_df = word_freq_df.sum().sort_values(ascending=False)
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions))
print('Recall score: ', recall_score(y_test, predictions))
#
# testing_predictions = []
# for i in range(len(X_test)):
#     if predictions[i] == 1:
#         testing_predictions.append("Course")
#     else:
#         testing_predictions.append("Non-Course")
# check_df = pd.DataFrame({"actual_label": list(y_test), "prediction": testing_predictions, "abstract":list(X_test)})
# check_df.replace(to_replace=0, value="Non-Course", inplace=True)
# check_df.replace(to_replace=1, value="Course", inplace=True)
# print(check_df)
# check_df.to_csv('NounsDF.csv', sep='\t')






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




