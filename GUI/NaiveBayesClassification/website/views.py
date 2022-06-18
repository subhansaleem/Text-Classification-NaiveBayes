from django.shortcuts import render
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

def home(request):
    return render(request, 'home.html',{})


def tfidf(request):
    if request.method =='POST':
        print("tfidf")
        termsPresentTFIDF = json.load(open("G:/My Drive/FAST/S6/IR/A3/GUI/NaiveBayesClassification/website/dumps/termsPresentTFIDF.txt", 'r'))

        label = [1 if i < 230 else 0 for i in range(1051)]

        df = pd.DataFrame(list(zip(termsPresentTFIDF, label)), columns=['Terms', 'Label'])

        X_train, X_test, y_train, y_test = train_test_split(df['Terms'], df['Label'], random_state=1)
        cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True,
                             stop_words='english')

        X_train_cv = cv.fit_transform(X_train)
        X_test_cv = cv.transform(X_test)
        word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
        top_words_df = word_freq_df.sum().sort_values(ascending=False)
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train_cv, y_train)
        predictions = naive_bayes.predict(X_test_cv)

        # printing ACCURACY, PRECISION, RECALL

        print("-------------------------------------------------------------------------")
        print("\n\n NAIVE BAYES MODEL ON TF-IDF TERMS \n\n")
        print('Accuracy score: ', accuracy_score(y_test, predictions))
        print('Precision score: ', precision_score(y_test, predictions))
        print('Recall score: ', recall_score(y_test, predictions))
        print("-------------------------------------------------------------------------")
        r = [accuracy_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test, predictions)]

        dict = [{
            'result': r
        }]
        return render(request,'result.html',{'dict':dict})
    return render(request,'result.html',{})

def noun(request):
    if request.method =='POST':
        print("noun")
        coherenceTerms = json.load(open("G:/My Drive/FAST/S6/IR/A3/GUI/NaiveBayesClassification/website/dumps/coherenceTerms-Nouns.txt", 'r'))

        label = [1 if i < 230 else 0 for i in range(1051)]

        df = pd.DataFrame(list(zip(coherenceTerms, label)), columns=['Terms', 'Label'])

        X_train, X_test, y_train, y_test = train_test_split(df['Terms'], df['Label'], random_state=1)
        cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True,
                             stop_words='english')

        X_train_cv = cv.fit_transform(X_train)
        X_test_cv = cv.transform(X_test)
        word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names_out())
        top_words_df = word_freq_df.sum().sort_values(ascending=False)
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train_cv, y_train)
        predictions = naive_bayes.predict(X_test_cv)

        # printing ACCURACY, PRECISION, RECALL
        print("-------------------------------------------------------------------------")
        print("\n\n NAIVE BAYES MODEL ON NOUNS - COHERENCE TOPIC BASED \n\n")
        print('Accuracy score: ', accuracy_score(y_test, predictions))
        print('Precision score: ', precision_score(y_test, predictions))
        print('Recall score: ', recall_score(y_test, predictions))
        print("-------------------------------------------------------------------------")
        r = [accuracy_score(y_test, predictions), precision_score(y_test, predictions),
             recall_score(y_test, predictions)]

        dict = [{
            'result': r
        }]
        return render(request, 'result.html', {'dict': dict})
    return render(request,'result.html',{})

def lexical(request):
    if request.method =='POST':
        print("lexical")
        chainPresent = json.load(open("G:/My Drive/FAST/S6/IR/A3/GUI/NaiveBayesClassification/website/dumps/chainsPresent.txt"))

        label = [1 if i < 230 else 0 for i in range(1051)]

        df = pd.DataFrame(list(zip(chainPresent, label)), columns=['Terms', 'Label'])

        X_train, X_test, y_train, y_test = train_test_split(df['Terms'], df['Label'], random_state=1)
        cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True,
                             stop_words='english')

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
        r = [accuracy_score(y_test, predictions), precision_score(y_test, predictions),
             recall_score(y_test, predictions)]

        dict = [{
            'result': r
        }]
        return render(request, 'result.html', {'dict': dict})
    return render(request,'result.html',{})
def mix(request):
    if request.method =='POST':
        print("mix")
        mixFeature = []
        coherenceTerms = json.load(open("G:/My Drive/FAST/S6/IR/A3/GUI/NaiveBayesClassification/website/dumps/coherenceTerms-Nouns.txt", 'r'))
        termsPresentTFIDF = json.load(open("G:/My Drive/FAST/S6/IR/A3/GUI/NaiveBayesClassification/website/dumps/termsPresentTFIDF.txt", 'r'))
        chainPresent = json.load(open("G:/My Drive/FAST/S6/IR/A3/GUI/NaiveBayesClassification/website/dumps/chainsPresent.txt"))

        for i in range(1051):
            temp = ''
            temp += termsPresentTFIDF[i]
            temp += " " + coherenceTerms[i]
            temp += " " + chainPresent[i]

            # using set to remove duplicates
            temp = set(word_tokenize(temp))
            temp2 = " ".join(temp)
            mixFeature.append(temp2)

        label = [1 if i < 230 else 0 for i in range(1051)]

        # creating dataframe
        df = pd.DataFrame(list(zip(mixFeature, label)), columns=['Terms', 'Label'])

        # dividing the dataframe into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(df['Terms'], df['Label'], random_state=1)

        # creating the count vectorizer for numeric values for the model
        cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True,
                             stop_words='english')

        # fitting the count vectorizer on the training data
        X_train_cv = cv.fit_transform(X_train)
        X_test_cv = cv.transform(X_test)
        word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names_out())
        top_words_df = word_freq_df.sum().sort_values(ascending=False)

        # naive bayes classifier
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train_cv, y_train)
        predictions = naive_bayes.predict(X_test_cv)

        print("\n\n NAIVE BAYES MODEL ON MIX FEATURES \n\n")
        print('Accuracy score: ', accuracy_score(y_test, predictions))
        print('Precision score: ', precision_score(y_test, predictions))
        print('Recall score: ', recall_score(y_test, predictions))
        print("-------------------------------------------------------------------------")
        r = [accuracy_score(y_test, predictions), precision_score(y_test, predictions),
             recall_score(y_test, predictions)]

        dict = [{
            'result': r
        }]
        return render(request, 'result.html', {'dict': dict})
    return render(request,'result.html',{})