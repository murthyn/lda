# from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from gensim.test.utils import datapath
from utils import *
from nltk.corpus import stopwords
import re
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



# Load a potentially pretrained model from disk.
lda = LdaModel.load(datapath("model"))

perplex = lda.log_perplexity(common_corpus)
print('perplex', perplex)

# coherence_model_lda = CoherenceModel(model=lda, texts=common_corpus, dictionary=common_dictionary, coherence='c_v')
# coherence = coherence_model_lda.get_coherence()
# print('coherence', coherence)

### print(lda.print_topics(num_topics=20, num_words=10))
for idx, elt in enumerate(lda.top_topics(corpus=common_corpus, dictionary=common_dictionary)):
    # print(elt)
    words, coherence_score = elt
    print('cluster number', idx)
    # print('perplexity', perplex)
    results.write('cluster number' + str(idx))
    for tup in words:
        prob, word = tup
        print(common_dictionary[eval(word)], prob)
        results.write(common_dictionary[eval(word)] + str(prob))
        results.write('\n')
    print('\n')
    results.write('\n')


for idx, elt in enumerate(lda.print_topics(num_topics=100, num_words=10)):
    print(elt)
    topic_id, words = elt
    print('topic number', idx)
    words = words.strip(' ')
    words = words.split('+')
    # print(words)
    for tup in words:
        # print(tup)
        val, word = tup.split('*')
        # print(type(word), eval(word), type(eval(word)), type(eval(eval(word))))
        print(common_dictionary[eval(eval(word))], val)
    print('\n')
