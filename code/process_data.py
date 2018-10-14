#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 15:24:14 2018

@author: chandresh
"""
import os, string
from nltk.corpus import stopwords
from collections import Counter
# specify directory to load
directory = '/home/chandresh/ckm/data/movies review/review_polarity/txt_sentoken/'
vocab_dict ='movie_dict.txt'
outfile    = 'processed_review.txt'
data=[]
url ="http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"

def process_docs(directory):
    for filename in os.listdir(directory):
		# skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue		
		# create the full path of the file to open
        path = directory + '/' + filename
		# load document
        with open(path,'r') as fid:
           data.append(fid.read())
        #print("loaded:", filename)          
    print("processed ",len(data), " docs")
"""
Step 1. Data loading
"""
process_docs(directory+'pos')
process_docs(directory+'neg')

"""
step 2. data cleaning

Remove punctuation from words (e.g. ‘what’s’).
Removing tokens that are just punctuation (e.g. ‘-‘).
Removing tokens that contain numbers (e.g. ’10/10′).
Remove tokens that have one character (e.g. ‘a’).
Remove tokens that don’t have much meaning (e.g. ‘and’)

Some ideas:

We can filter out punctuation from tokens using the string translate() function.
We can remove tokens that are just punctuation or contain numbers by using an isalpha() check on each token.
We can remove English stop words using the list loaded using NLTK.
We can filter out short tokens by checking their length.
"""
def clean_doc(data):
    # split into tokens by white space
    for i,doc in enumerate(data):
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        #print(tokens)
        data[i]=tokens

# clean the doc
clean_doc(data)

"""
Step 3. Build the vocabulary
"""
 # define vocab
vocab = Counter()

for doc in data:
    vocab.update(doc)
# print lenngth of the vocab
print(len(vocab))    
# print the top most_common words
print(vocab.most_common(50))
# print lenngth of the vocab
print(len(vocab))  
# keep vocab with words whose freq is at least 5
min_freq = 5
vocab = [w for w, c in vocab.items() if c>=min_freq]  

# save the vocab
with open(directory+vocab_dict,'w') as fid:
    lines ="\n".join(vocab)
    fid.write(lines)

"""
step 4. Save the prepared data for modeling
Q. why to do that?
A. Decouple the data preparation from modeling

Q. How to use dict to clean the reviews
A. here are the steps:
    1. Load dict
    2. process each doc, remvoing tokens not in dict
    3. save the doc
"""
lines=''
for doc in data:
    tokens = [w for w in doc if w in vocab]
    line = " ".join(tokens)
    lines+=line+'\n'
    
with open(directory+outfile,'w') as fid:
    fid.write(lines)    

#The data is ready for use in a bag-of-words or even word embedding model.
    
#Extensions
#This section lists some extensions that you may wish to explore.
#
#Stemming. We could reduce each word in documents to their stem using a stemming algorithm like the Porter stemmer.
#N-Grams. Instead of working with individual words, we could work with a vocabulary of word pairs, called bigrams. We could also investigate the use of larger groups, such as triplets (trigrams) and more (n-grams).
#Encode Words. Instead of saving tokens as-is, we could save the integer encoding of the words, where the index of the word in the vocabulary represents a unique integer number for the word. This will make it easier to work with the data when modeling.
#Encode Documents. Instead of saving tokens in documents, we could encode the documents using a bag-of-words model and encode each word as a boolean present/absent flag or use more sophisticated scoring, such as TF-IDF.
