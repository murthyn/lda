# from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath
from utils import *
from nltk.corpus import stopwords
import re
import logging
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from tqdm import tqdm

stop_words = set(stopwords.words('english'))
allmydocs = getalldocs("alldocs3.txt")
results = open("results.txt", 'w')

cleaned_docs = []
num_docs = 20368
for idx, doc in enumerate(allmydocs):
    # if idx > num_docs:
    #     break
    doc = doc.lower()
    doc = re.split(' |, |\n|: |(|)', doc)
    doc = [elt for elt in doc if elt is not None]
    tokens = []
    for words in doc:
        cleaned = ''.join([i for i in words if i.isalpha()])
        if cleaned not in stop_words and 2 < len(cleaned):
            tokens.append(cleaned)
    cleaned_docs.append(tokens[:])

# Create a corpus from a list of texts
common_dictionary = Dictionary(cleaned_docs)
common_corpus = [common_dictionary.doc2bow(text) for text in cleaned_docs]
random.shuffle(common_corpus)
train = common_corpus[:int(len(common_corpus)*0.8)]
test = common_corpus[int(len(common_corpus)*0.8):]

lda = LdaModel(common_corpus, num_topics=25, iterations=10000, eval_every=2, chunksize=10000, passes=10)


perplex = lda.log_perplexity(common_corpus)
print('perplex', perplex)

# Save model to disk.
temp_file = datapath("model")
lda.save(temp_file)

