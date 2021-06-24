#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:51:05 2020

@author: ken
"""
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import word2vec
from nltk.corpus import wordnet as wn
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


corpus = "/home/ken/leetcode_web_crawler/sentences.txt"
#Setting
seed = 0
sg = 1
window_size = 10
vector_size = 50
min_count = 1
workers = 8
epochs = 20
batch_words = 10000
train_data = word2vec.LineSentence(corpus)
model = word2vec.Word2Vec(
    train_data,
    min_count = min_count,
    size = vector_size,
    workers = workers,
    iter = epochs,
    window = window_size,
    sg = sg,
    seed = seed,
    batch_words = batch_words
    )
#output_file = GloVe_dir + 'gensim_vectors_leetcode.txt'
#glove2word2vec(input_file, output_file)
#model = KeyedVectors.load_word2vec_format(output_file, binary=False)
#word = 'return'
'''
def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

embeddings_dict = {}
with open(input_file, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

tsne = TSNE(n_components=2, random_state=0)

words = list(embeddings_dict.keys())
vectors = [embeddings_dict[word] for word in words]

Y = tsne.fit_transform(vectors)
plt.scatter(Y[:, 0], Y[:, 1])

for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
'''