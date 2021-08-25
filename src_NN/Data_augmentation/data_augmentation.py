#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:29:50 2021

@author: ken
"""

from stanfordcorenlp import StanfordCoreNLP
import random
import re

nlp = StanfordCoreNLP(r'/home/ken/stanford-corenlp-4.2.0')


#input and output keywords sets
input_relateword_sets = [
    "input",
    "given",
    "give",
    "provide",
    "receive",
    "take"
    ]

output_relateword_sets = [
    "output",
    "return",
    "obtain",
    "find"
    ]

def SentencesProcessing(raw_sentences):
    #new_sentences = raw_sentences.lower()
    new_sentences = re.sub(r'\([^)]*\)', '', raw_sentences)
    new_sentences = re.sub(r"\n","", new_sentences)
    #new_sentences = re.sub('"','', new_sentences)
    #tokens = [w for w in new_sentences.split() if not w in stop_words]
    #new_sentences = re.sub("[^a-zA-Z]", " ", new_sentences) 
    long_words=[]
    for i in new_sentences.split():
        long_words.append(i)   
    return (" ".join(long_words)).strip()


file = "./sentences_test3"
fo = open(file, "r+")
sentence_group_raw = []
x = fo.readline()
while(x != ""):
    sentence_group_raw.append(x)
    x = fo.readline()
fo.close()

sentence_group = []
for sen in sentence_group_raw:
    sentence_group.append(SentencesProcessing(sen))
  
label_file = "./NN_I_test3"
fo = open(label_file, "r+")
label_group = []
y = fo.readline()
while(y != ""):
    label_group.append(y)
    y = fo.readline()
fo.close()

label_start_index = []
for k in range(len(sentence_group)):
    sen_token = nlp.word_tokenize(sentence_group[k])
    label_token_raw = nlp.word_tokenize(label_group[k])
    label_token = []
    for token in label_token_raw:
        if token == ";":
            break
        label_token.append(token)
    
    w = 0
    while w < len(sen_token) and len(label_token) != 0:
        if len(label_token) == 1 and sen_token[w] == label_token[0]:
            label_start_index.append([k,w])
            break
        elif len(label_token) == 2 and sen_token[w] == label_token[0] and sen_token[w+1] == label_token[1]:
            label_start_index.append([k,w])
            break
        elif sen_token[w] == label_token[0] and sen_token[w+1] == label_token[1] and sen_token[w+2] == label_token[2]:
            label_start_index.append([k,w])
            break
        w += 1
    if len(label_token) == 0:
        label_start_index.append([k,-1])
        


noun_bank = []

for i in range(len(sentence_group)):
    pos_tag = nlp.pos_tag(sentence_group[i])
    pos_tag_len = len(pos_tag)
    label_len = len(nlp.word_tokenize(label_group[i]))
    k = 0
    j = label_start_index[i][1]
    while k < label_len:
        if pos_tag[j][1] == 'NN' or pos_tag[j][1] == 'NNS' or pos_tag[j][1] == 'NNP' or pos_tag[j][1] == 'NNPS':
            noun = pos_tag[j][0]
            while j < pos_tag_len and (pos_tag[j+1][1] == 'NN' or pos_tag[j+1][1] == 'NNS' or pos_tag[j+1][1] == 'NNP' or pos_tag[j+1][1] == 'NNPS'):
                noun = noun + " " + pos_tag[j+1][0]
                j += 1
                k += 1
            if noun not in noun_bank:
                noun_bank.append(noun)
        j += 1
        k += 1

def replace_noun(original_noun, noun_bank):
    random_noun = random.choice(noun_bank)
    while random_noun == original_noun:
        random_noun = random.choice(noun_bank)
    return random_noun

fo_new_sen = open("./augmentation_data", "w")
fo_new_label = open("./augmentation_data_label", "w")
augmentation_data = []
augmentation_label = []
for iteration in range(10):
    for i in range(len(sentence_group)):
        original_sen = nlp.word_tokenize(sentence_group[i])
        original_label = nlp.word_tokenize(label_group[i])
        new_sen = []
        new_label = []
        pos_tag = nlp.pos_tag(sentence_group[i])
        pos_tag_len = len(pos_tag)
        label_len = len(original_label)
        j = 0
        k = 0
        while j < pos_tag_len:
            if (i == 4 or i == 5 or i == 60) and (pos_tag[j][0] == "." or pos_tag[j][0] == ";"):
                new_sen.append(pos_tag[j][0])
                break
            if j < label_start_index[i][1] or j >= label_start_index[i][1] + label_len:
                new_sen.append(pos_tag[j][0])
                j += 1
            elif pos_tag[j][1] == 'NN' or pos_tag[j][1] == 'NNS' or pos_tag[j][1] == 'NNP' or pos_tag[j][1] == 'NNPS':
                noun = pos_tag[j][0]
                if noun in input_relateword_sets or noun in output_relateword_sets:
                    new_sen.append(noun)
                    j += 1
                    if original_label[k] != ";":
                        new_label.append(noun)
                        k += 1
                    continue
                while j < pos_tag_len and (pos_tag[j+1][1] == 'NN' or pos_tag[j+1][1] == 'NNS' or pos_tag[j+1][1] == 'NNP' or pos_tag[j+1][1] == 'NNPS'):
                    noun = noun + " " + pos_tag[j+1][0]
                    j += 1
                    k += 1
                    
                new_noun = replace_noun(noun, noun_bank)
                #if k < label_len:
                if j < label_start_index[i][1] + label_len and original_label[k] != ";":
                    new_label.append(new_noun)
                    k += 1
                new_sen.append(new_noun)
                j += 1
            else:
                #if k < label_len:
                if j < label_start_index[i][1] + label_len and original_label[k] != ";":
                    new_label.append(pos_tag[j][0])
                    k += 1
                new_sen.append(pos_tag[j][0])
                j += 1
        augmentation_data.append(" ".join(new_sen))
        augmentation_label.append(" ".join(new_label))
        fo_new_sen.write(" ".join(new_sen) + "\n")
        fo_new_label.write(" ".join(new_label) + "\n")
        #break
    #break

nlp.close()
fo_new_sen.close()
fo_new_label.close()
            