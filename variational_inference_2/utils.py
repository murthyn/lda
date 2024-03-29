import sys, csv
import re
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from scipy.special import psi
import numpy as n
import matplotlib.pyplot as plt

LISTOFDOCS = "alldocs.txt"

stop_words = set(stopwords.words('english'))
filenames = []

def get_filenames(filename):
    print("getting filenames")
    if filename == None:
        filename = LISTOFDOCS
    with open(filename, 'r') as f:
        docs = f.readlines()

        for doc in docs:
            # print str(doc).split("\n")
            filenames.append(str(doc).split("\n")[0])

    return filenames


def getfiles(filename):
    # print("getting file " + filename)
    f = open(filename, 'r')
    doc = f.read()
    # print doc
    return doc


def getalldocs(filename = None):
    files = get_filenames(filename)
    docs = []
    for file in files:
        doc = getfiles(file)
        # print doc
        docs.append(doc)
    # print docs
    return docs



def dirichlet_expectation(alpha):
    '''see onlineldavb.py by Blei et al'''
    if (len(alpha.shape) == 1):
        return (psi(alpha) - psi(n.sum(alpha)))
    return (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def beta_expectation(a, b, k):
    mysum = psi(a + b)
    Elog_a = psi(a) - mysum
    Elog_b = psi(b) - mysum
    Elog_beta = n.zeros(k)
    Elog_beta[0] = Elog_a[0]
    # print Elog_beta
    for i in range(1, k):
        Elog_beta[i] = Elog_a[i] + n.sum(Elog_b[0:i])
        # print Elog_beta
    # print Elog_beta
    return Elog_beta


def parseDocument(doc, vocab):
    wordslist = list()
    countslist = list()
    doc = doc.lower()
    doc = re.split(' |, |\n|: |(|)', doc)
    doc = [elt for elt in doc if elt is not None]
    tokens = []

    # print(vocab)
    for elt in doc:
        # master_vocab.add(elt.strip('<>,^/|."}*[{]'))
        cleaned = ''.join([i for i in elt if i.isalpha()])
        if cleaned not in stop_words and 2 < len(cleaned):
            tokens.append(cleaned)

    dictionary = dict()
    for word in tokens:
        if word in vocab:
            wordtk = vocab[word]
            if wordtk not in dictionary:
                dictionary[wordtk] = 1
            else:
                dictionary[wordtk] += 1

    wordslist.append([_ for _ in dictionary.keys()])
    countslist.append([_ for _ in dictionary.values()])
    # print((wordslist[0], countslist[0]))
    return (wordslist[0], countslist[0])

# def parseDocument(doc, vocab):
#   wordslist = list()
#   countslist = list()
#   doc = doc.lower()
#   tokens = wordpunct_tokenize(doc)

#   dictionary = dict()
#   for word in tokens:
#       word = ''.join([i for i in word if i.isalpha()])
#       if word in stop_words or len(word) <= 2:
#           continue
#       if word in vocab:
#           wordtk = vocab[word]
#           if wordtk not in dictionary:
#               dictionary[wordtk] = 1
#           else:
#               dictionary[wordtk] += 1

#   wordslist.append([_ for _ in dictionary.keys()])
#   countslist.append([_ for _ in dictionary.values()])
#   # print((wordslist[0], countslist[0]))
#   return (wordslist[0], countslist[0])

def getVocab(file):
    '''getting vocab dictionary from a csv file (nostopwords)'''
    vocab = dict()
    with open(file, 'r') as infile:
        reader = csv.reader(infile)
        idx = 0
        for row in reader:
            if row[0] not in stop_words and 2 < len(row[0]) <= 15:
                vocab[row[0].strip('<>,^/|."}*[{]')] = idx
                idx += 1
    # print(vocab)
    return vocab


def plottrace(x, Y, K, n, perp):
    for i in range(K):
        plt.plot(x, Y[i], label = "Topic %i" %(i+1))

    plt.xlabel("Number of Iterations")
    plt.ylabel("Probability of Each topic")
    # plt.legend()
    plt.title("Trace plot for topic probabilities")
    plt.savefig("temp/plot_%i_%i_%f.png" %(K, n, perp))
# 
# 
# def calcPerplexity(test_docs):
    


# getalldocs()


