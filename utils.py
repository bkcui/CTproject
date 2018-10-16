from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
from konlpy.tag import Kkma
from konlpy.utils import pprint
from konlpy import jvm

jvm.init_jvm()
tagger = Kkma()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
UNKNOWN = 3

MAX_LENGTH = 20
MIN_LENGTH = 2
SKIP_P = 0.8
UNKNOWN_P = 0.00005


class Lang:  # ont-hot vector 인코딩 다른 embedding 필요 language
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UNKNOWN"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):  # tokenizer 한국어
        for word in tagger.morphs(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:  # initializer
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word]

def normalizeString(s):
    s = re.sub(r"([.!?])", r" \1", s)
    return s

def readLangs(lang):
    print("Reading lines...")
    data = []
    n = 0
    # Read the file and split into lines
    with open('data/%s.txt' % (lang), 'r',encoding='utf-8') as datafile:
        line = datafile.readline()
        while line:

            if random.random() < SKIP_P and n > 300:
                datafile.readline()
                continue
            data.append(normalizeString(line))
            line = datafile.readline()
            n+=1


    return data

MAX_CHAR = 50
MIN_CHAR = 5

def filtersen(p):
    return len(p) < MAX_CHAR and re.compile(r'[^가-힣0-9\.,?! ]').match(p) is None\
               and len(p) > MIN_CHAR    #Maxlen 보다 작거나 숫자+. 으로 끝나지 않음, Minlen 보다 큼

def filtersens(sens):
    return [sen for sen in sens if filtersen(sen)]

def prepareData(lang):   #데이터 준비
    data = readLangs(lang)
    sens = filtersens(data)#filter

    print("Trimmed to %s sentence sens" % len(sens))
    print("Counting words...")
    lan = Lang(lang)
    for sen in sens:
        lan.addSentence(sen)
    print("Counted words:")
    print(lan.name, lan.n_words)
    return lan, sens

def preparepairs(name):
    print("Reading Pairs")
    linenum = 1
    pairs = []
    with open('data/%s.txt' % (name), 'r',encoding='utf-8') as pairfile:
        line = pairfile.readline()
        while line:
            pairs.append([line, linenum])
            line = pairfile.readline()
            linenum += 1
    return pairs


#helper
def indexesFromSentence(lang, sentence):
    list = []
    for word in tagger.morphs(sentence):
        try:
            if random.random() < UNKNOWN_P:
                list.append(UNKNOWN)
            else:
                list.append(lang.word2index[word])
        except KeyError:
            list.append(UNKNOWN)

    return list

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)



