import nltk
import string
from heapq import nlargest
from nltk.tag import pos_tag
from string import punctuation
from inspect import getsourcefile
from collections import defaultdict
from nltk.tokenize import word_tokenize
from os.path import abspath, join, dirname
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import RegexpTokenizer

def relation_list(nouns):
    relation_list = defaultdict(list)

    for k in range(len(nouns)):
        relation = []
        for syn in wordnet.synsets(nouns[k], pos=wordnet.NOUN):
            for l in syn.lemmas():
                relation.append(l.name())
                if l.antonyms():
                    relation.append(l.antonyms()[0].name())
            for l in syn.hyponyms():
                if l.hyponyms():
                    relation.append(l.hyponyms()[0].name().split('.')[0])
            for l in syn.hypernyms():
                if l.hypernyms():
                    relation.append(l.hypernyms()[0].name().split('.')[0])
        relation_list[nouns[k]].append(relation)
    return relation_list

def create_lexical_chain(nouns, relation_list):
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
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
                    elif noun in relation_list[key][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
        if flag == 0:
            dic_nuevo = {}
            dic_nuevo[noun] = 1
            lexical.append(dic_nuevo)
            flag = 1
    return lexical

def prune(lexical):
    final_chain = []
    while lexical:
        result = lexical.pop()
        if len(result.keys()) == 1:
            for value in result.values():
                if value != 1:
                    final_chain.append(result)
        else:
            final_chain.append(result)
    return final_chain



if __name__ == "__main__":

    with open('input.txt', "r") as f:
        input_txt = f.read()
        f.close()

    position = ['NN', 'NNS', 'NNP', 'NNPS']

    sentence = nltk.sent_tokenize(input_txt)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [tokenizer.tokenize(w) for w in sentence]
    tagged = [pos_tag(tok) for tok in tokens]
    nouns = [word.lower() for i in range(len(tagged)) for word, pos in tagged[i] if pos in position]

    relation = relation_list(nouns)
    lexical = create_lexical_chain(nouns, relation)
    final_chain = prune(lexical)

    for i in range(len(final_chain)):
        print("Chain " + str(i + 1) + " : " + str(final_chain[i]))
