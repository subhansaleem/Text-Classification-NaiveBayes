import os
import glob
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import spacy

nlp = spacy.load("en_core_web_sm")
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import numpy as np

array = []

file_location = os.path.join('data', 'C:/Users/Home/Downloads/course-cotrain-data/course-cotrain-data/fulltext/course/',
                             '*')
# print(file_location)

path = r"C:/Users/Home/Downloads/course-cotrain-data/course-cotrain-data/fulltext/course/http_^^cs.cornell.edu^Info^Courses^Current^CS415^CS414.html"
path2 = "C:/Users/Home/Downloads/course-cotrain-data/course-cotrain-data/fulltext/course/"
filenames = glob.glob(file_location)
# print(filenames)


for f in filenames:
    # load data from file
    with open(f, 'r') as myfile:
        array.append(myfile.read())

# make words to lowercase
array = [x.lower() for x in array]
finalarray = []
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

for row in array:
    remove_sw = re.compile(r'<[^>]+>').sub("", row)
    # removing punctiation from remove_sw
    remove_punctuation = re.compile(r'[^\w\s]').sub(" ", remove_sw)
    # removing numbers
    remove_sw = re.compile(r'\d+').sub("", remove_punctuation)
    # removing stop words from array
    remove_sw = re.compile(r'\b' + r'\b|\b'.join(stop_words) + r'\b').sub("", remove_sw)
    finalarray.append(remove_sw)


def stemSentence(sentence):
    token_words = word_tokenize(sentence)
    token_words
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


x = []
Nouns = []


def ProperNounExtractor(text):
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        for (word, tag) in tagged:
            if tag == 'NN':  # If the word is a noun
                Nouns.append(word)


for word in finalarray:
    x.append(stemSentence(word))
    ProperNounExtractor(word)

vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=100,
    max_df=0.8,
    min_df=5,
    ngram_range=(1, 3),
    stop_words="english"

)

vectors = vectorizer.fit_transform(x)
# pick top 100 ranking from vectors
feature_names = vectorizer.get_feature_names_out()

# pick top 50 most frequent nouns from NounFinder
top50 = Counter(Nouns).most_common(50)

# top_nouns = [token.text for token in NounFinder if token.pos_ == "NOUN" and token.text not in stop_words]

with open('C:/Users/Home/Downloads/course-cotrain-data/A3/CleanedDoc.txt', 'w') as myfile:
    for words in finalarray:
        myfile.write("%s\n" % words)

with open('C:/Users/Home/Downloads/course-cotrain-data/A3/StemmedDoc.txt', 'w') as myfile:
    for words in x:
        myfile.write("%s\n" % words)

with open('C:/Users/Home/Downloads/course-cotrain-data/A3/Tf-IdfScore.txt', 'w') as myfile:
    for item in vectors:
        myfile.write("%s\n" % item)

with open('C:/Users/Home/Downloads/course-cotrain-data/A3/topFeatures.txt', 'w') as myfile:
    myfile.write("%s\n" % feature_names)

with open('C:/Users/Home/Downloads/course-cotrain-data/A3/Nouns.txt', 'w') as myfile:
    for word in Nouns:
        myfile.write("%s\n" % word)

with open('C:/Users/Home/Downloads/course-cotrain-data/A3/topNoun.txt', 'w') as myfile:
    myfile.write("%s\n" % top50)

# with open('C:/Users/Home/Downloads/course-cotrain-data/A3/top_words.txt', 'w') as myfile:
#     for item in top_words:
#         myfile.write("%s\n" % item)