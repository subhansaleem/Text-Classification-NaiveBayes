import numpy as np
"""for html files parsing"""
from bs4 import BeautifulSoup
"""using os directory function"""
import os
"""for tokenization"""
from nltk.tokenize import word_tokenize
"""for regex"""
import re
"""for dataframing"""
import pandas as pd
"""for dumoing json objects for quick compilation"""
import json

"""for NAIVE BAYES CLASSIFIER"""
from collections import Counter,defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords,wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score
import nltk
from nltk.tokenize import RegexpTokenizer
"""for tagging nouns"""
from nltk.tag import pos_tag


#reading all files in the directory
textlist = []
fullText = ""
dirCourse = os.listdir('./course-cotrain-data/fulltext/course')
dirNonCourse = os.listdir('./course-cotrain-data/fulltext/non-course')

ps = PorterStemmer()
words = []
stop_words = set(stopwords.words('english'))

#preprocessing the text from the files - COURES
for file in dirCourse:
        with open(f'./course-cotrain-data/fulltext/course/{file}', 'r') as f:
            t = f.read().lower()
            soup = BeautifulSoup(t, 'html.parser')
            text = ""
            if soup.title:
                text = soup.title.text
            if soup.body:
                text += soup.body.text.replace('\n', ' ')

            """removing punctuation and stopwords"""
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

"""PREPROCESSING THE TEXT FROM THE FILES - NON-COURES"""
for file in dirNonCourse:
        with open(f'./course-cotrain-data/fulltext/non-course/{file}', 'r') as f:
            t = f.read().lower()
            soup = BeautifulSoup(t, 'html.parser')
            text = ""
            if soup.title:
                text = soup.title.text
            if soup.body:
                text += soup.body.text.replace('\n', ' ')
                """removing punctuation and stopwords"""
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

"""dumping ful text and textlist to json files"""
# json.dump(fullText, open("./dumps/fullText.txt", 'w'))
# json.dump(textlist, open("./dumps/textList.txt", 'w'))


#creating tf-idf vector for selection of 100 terms
tfidf = TfidfVectorizer()
response = tfidf.fit_transform(textlist)
feature_names = tfidf.get_feature_names_out()
dense = response.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
# df.to_csv('./dumps/tfidf.csv')


#creating a dictionary of the terms and their tf-idf scores
dic = {}
for i in feature_names:
    dic[i] = df[i].tolist()
dicMag = {}

#creating a dictionary of the terms and their magnitude for selection
for key, value in enumerate(dic):
    vector = np.array(dic[value])
    magnitude = np.linalg.norm(vector)
    dicMag[value] = magnitude

#taking out the top 100 terms
tfidfVector = []
d = Counter(dicMag)
for value in d.most_common(100):
    tfidfVector.append(value[0])


####################################################################################################
"""NAIVE BAYES ON TF-IDF"""

#checking the terms in each document for terms in df
termsPresentTFIDF = []

#uncomment this to create tf-idf vector instead of loading from dump

"""
for item in textlist:
    temp = ""
    string = str(item)
    for key, value in enumerate(tfidfVector):
        if value in string:
            temp += " " + value
        # else:
        #     temp.append(0)
    termsPresentTFIDF.append(temp)
"""

#dumping termsPresentTFIDF to json file
# json.dump(termsPresentTFIDF, open("./dumps/termsPresentTFIDF.txt", 'w'))

#loading termsPresentTFIDF from json file for quick compilatio
termsPresentTFIDF = json.load(open("./dumps/termsPresentTFIDF.txt", 'r'))



#---------------------------------------------------------------NAIVE BAYES MODEL IMPLEMENTATION ON TF-IDF TERMS ---------------------------------------------------------------

label = [1 if i < 230 else 0 for i in range(1051)]

df = pd.DataFrame(list(zip(termsPresentTFIDF,label)), columns=['Terms', 'Label'])

X_train, X_test, y_train, y_test = train_test_split(df['Terms'], df['Label'], random_state=1)
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')

X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
top_words_df = word_freq_df.sum().sort_values(ascending=False)
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)

#printing ACCURACY, PRECISION, RECALL

print("-------------------------------------------------------------------------")
print("\n\n NAIVE BAYES MODEL ON TF-IDF TERMS \n\n")
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions))
print('Recall score: ', recall_score(y_test, predictions))
print("-------------------------------------------------------------------------")

#uncomment this for checking the dataframe of the Actual vs Predicted Labels for COURSE AND NON-COURSE
#with NAIVE BAYES ON TF-IDF TERMS
"""
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
check_df.to_csv('TF-IDFdf.csv', sep='\t')
"""

###########################################################################################
"""NOUN SELECTION - TOPIC COHERENCE BASED"""


#uncommment this to create nouns - topic coherence terms

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
"""

# json.dump(coherenceTerms, open("./dumps/coherenceTerms-Nouns.txt", 'w'))

#loading terms from json dump to quickly compile
coherenceTerms = json.load(open("./dumps/coherenceTerms-Nouns.txt", 'r'))

#---------------------------------------------------------------NAIVE BAYES MODEL IMPLEMENTATION ON NOUNS - COHERENCE BASED ---------------------------------------------------------------

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

#printing ACCURACY, PRECISION, RECALL
print("-------------------------------------------------------------------------")
print("\n\n NAIVE BAYES MODEL ON NOUNS - COHERENCE TOPIC BASED \n\n")
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions))
print('Recall score: ', recall_score(y_test, predictions))
print("-------------------------------------------------------------------------")


#uncomment this for checking the dataframe of the Actual vs Predicted Labels for COURSE AND NON-COURSE
#with NAIVE BAYES ON NOUN SELECTION
"""
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

########################################################################################################
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

"""LEXICAL CHAINS"""


#uncomment this to create lexical chains
#takes a while to run because of the number of words


""" taking out nouns from the text for relation list"""

#uncomment this for creting selecting nouns for relation list
"""
position = ['NN', 'NNS', 'NNP', 'NNPS']
with open('input.txt', "r") as f:
    input_txt = f.read()
    f.close()

sentence = nltk.sent_tokenize(fullText)
tokenizer = RegexpTokenizer(r'\w+')
tokens = [tokenizer.tokenize(w) for w in sentence]
tagged = [pos_tag(tok) for tok in tokens]
nouns = [word for i in range(len(tagged)) for word, pos in tagged[i] if pos in position]
"""


""" creating a relation list from the nouns"""

#uncomment this to create relation list
"""
relation_list = defaultdict(list)

for k in range(len(nouns)):
    relation = []
    for syn in wordnet.synsets(nouns[k], pos=wordnet.NOUN):
        for l in syn.lemmas():
            relation.append(l.name())
            if l.antonyms():
                relation.append(l.antonyms()[0].name())
    relation_list[nouns[k]].append(relation)

relation = relation_list
"""

"""dumping relation list to load quickly next time"""
# json.dump(relation, open("./dumps/relationList.txt", 'w'))

"""creating lexcials from the relation list"""

#uncomment this to create lexical chains
"""
lexical = []
threshold = 0.5
for noun in nouns:
    flag = 0
    for j in range(len(lexical)):
        if flag == 0:
            for key in list(lexical[j]):
                if key == noun and flag == 0:
                    lexical[j][noun] += 1
                    flag = 1
                elif key in relation_list[noun][0] and flag == 0:
                    syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                    syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                    if syns1 and syns2:
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
                elif noun in relation_list[key][0] and flag == 0:
                    syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                    syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                    if syns1 and syns2:
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
    if flag == 0:
        dic_nuevo = {}
        dic_nuevo[noun] = 1
        lexical.append(dic_nuevo)
        flag = 1


json.dump(lexical, open("lexical.txt", 'w'))
lexical = json.load(open("lexical.txt"))


#removing chains with relations = 1
final_chain = []
while lexical:
    result = lexical.pop()
    if len(result.keys()) == 1:
        for value in result.values():
            if value != 1:
                final_chain.append(result)
    else:
        final_chain.append(result)

##printing the final chains
# for i in range(len(final_chain)):
#     print("Chain " + str(i + 1) + " : " + str(final_chain[i]))
# print(final_chain[0])


chainList = []
for i in final_chain:
    for item in i:
        if i[item] > 1:
            chainList.append(item)
# print(len(chainList))
json.dump(chainList, open("./dumps/finalChainList.txt", 'w'))
"""

###########################
"""NAIVE BAYES ON LEXICAL CHAIN"""
#uncomment this to select chains and match with documents to create terms list
"""
# chainList = json.load(open("./dumps/finalChainList.txt"))
# chainPresent = []
# for item in textlist:
#     # print(item)
#     temp = ""
#     string = word_tokenize(item)
#     for key, value in enumerate(chainList):
#         if value in string:
#             temp += " " + value
#     chainPresent.append(temp)

# json.dump(chainPresent, open("./dumps/chainsPresent.txt", 'w'))
"""

#loading chains list present from dump
chainPresent = json.load(open("./dumps/chainsPresent.txt"))

#---------------------------------------------------------------NAIVE BAYES MODEL IMPLEMENTATION ON LEXICAL CHAINS ---------------------------------------------------------------

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

print("-------------------------------------------------------------------------")
print("\n\n NAIVE BAYES MODEL ON LEXICAL CHAINS \n\n")
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions))
print('Recall score: ', recall_score(y_test, predictions))
print("-------------------------------------------------------------------------")

#uncomment this for checking the dataframe of the Actual vs Predicted Labels for COURSE AND NON-COURSE
#with NAIVE BAYES ON LEXICAL CHAINS

"""
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
check_df.to_csv('LexicalDF.csv', sep='\t')
"""


##################################################################################################################
# FINAL STEP OF DOING NAIVE BAYES WITH MIX FEATURES
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""MIX FEATURES FOR NAIVE BAYES"""
mixFeature = []

#combining all the features in a single list
for i in range(1051):
    temp = ''
    temp += termsPresentTFIDF[i]
    temp += " " + coherenceTerms[i]
    temp += " " + chainPresent[i]

    #using set to remove duplicates
    temp = set(word_tokenize(temp))
    temp2 = " ".join(temp)
    mixFeature.append(temp2)

#labeling the dataframe
# 1: COURSE, 0: NON-COURSE

label = [1 if i < 230 else 0 for i in range(1051)]

#creating dataframe
df = pd.DataFrame(list(zip(mixFeature,label)), columns=['Terms', 'Label'])

#dividing the dataframe into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df['Terms'], df['Label'], random_state=1)

#creating the count vectorizer for numeric values for the model
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')

#fitting the count vectorizer on the training data
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names_out())
top_words_df = word_freq_df.sum().sort_values(ascending=False)

#naive bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)

print("\n\n NAIVE BAYES MODEL ON MIX FEATURES \n\n")
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions))
print('Recall score: ', recall_score(y_test, predictions))
print("-------------------------------------------------------------------------")


#printing the predicted labels for the data of COURSE and NON-COURSE
testing_predictions = []
for i in range(len(X_test)):
    if predictions[i] == 1:
        testing_predictions.append("Course")
    else:
        testing_predictions.append("Non-Course")
check_df = pd.DataFrame({"actual_label": list(y_test), "prediction": testing_predictions, "terms":list(X_test)})
check_df.replace(to_replace=0, value="Non-Course", inplace=True)
check_df.replace(to_replace=1, value="Course", inplace=True)

print("-------------------------------------------------------------------------")
print(check_df)

#saving the dataframe to a csv file
check_df.to_csv('mixFeaturedf.csv', sep='\t')

