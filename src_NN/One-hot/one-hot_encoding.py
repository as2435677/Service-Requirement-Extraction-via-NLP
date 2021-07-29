#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:40:14 2021

@author: ken
"""

import numpy as np
from keras.preprocessing.text import Tokenizer

file_dir = "/home/ken/glove/"
filename = "sentences.txt"

input_file = file_dir + filename

sentences = []
with open(input_file, 'r', encoding="utf-8") as f:
    for line in f:
        sentences.append(line)
fo = open("one_hot_leetcode.txt", 'w', encoding="utf-8")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
voc = len(tokenizer.word_counts)
word_w_index = tokenizer.word_index
for word, i in word_w_index.items():
    integer_to_vector = np.zeros((voc))
    integer_to_vector[i-1] = 1
    fo.write("%s " % word)
    np.savetxt(fo, integer_to_vector, fmt = '%.5f', newline=' ')
    fo.write("\n")
fo.close()
    

