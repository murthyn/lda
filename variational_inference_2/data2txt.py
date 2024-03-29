import os
import re
import csv
import random
from nltk.corpus import stopwords
from tqdm import tqdm

file_names = []
stop_words = set(stopwords.words('english'))
print(stop_words)

out_file1 = open('alldocs.txt', 'w')
out_file2 = open('alldocs2.txt', 'w')
out_file3 = open('alldocs3.txt', 'w')
csv_file = open('vocab.csv', 'w')
csv_writer = csv.writer(csv_file)
master_vocab = set()
c = 0

# print('l', [_ for _ in os.walk("20newsgroups")])
for root, dirs, files in tqdm(os.walk("20newsgroups")):
    for file in files:
        if file.startswith("00"):
            with open(os.path.join(root, file)) as f:
                try:
                    read_file = re.split(' |, |\n|: |(|)', f.read())

                except:
                    print(c, os.path.join(root, file))
                    continue
                # print(read_file)
                vocab = set([elt for elt in read_file if elt is not None])
                # print(vocab)
                for elt in vocab:
                    # master_vocab.add(elt.strip('<>,^/|."}*[{]'))
                    cleaned = ''.join([i for i in elt if i.isalpha()])
                    if cleaned not in stop_words and 2 < len(cleaned):
                        master_vocab.add(cleaned)
                c += 1

            file_names.append(os.path.join(root, file))

for elt in master_vocab:
    csv_writer.writerow([elt])

random.shuffle(file_names)
print(len(file_names))
for elt in file_names[:int(len(file_names)*0.8)]:
    out_file1.write(elt + '\n')
for elt in file_names[int(len(file_names)*0.2):]:
    out_file2.write(elt + '\n')
for elt in file_names:
    out_file3.write(elt + '\n')

# for filename in os.listdir(os.getcwd()):
#   if filename == '20newsgroups':
#       for file in os.listdir(os.getcwd()):
#           print(file)