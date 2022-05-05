#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:40:07 2021

@author: ken
"""

from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pycrfsuite
import sklearn_crfsuite
from nltk.stem import WordNetLemmatizer
import sys
from collections import Counter

TrainDataPath = "/home/ken/Service-Requirement-Extraction-via-NLP/src_NN/Data_augmentation/"
TestDataPath = "/home/ken/Service-Requirement-Extraction-via-NLP/src/Leetcode_data/ExtraData/"
PreTrain_WV_Path = "/home/ken/Service-Requirement-Extraction-via-NLP/src_NN/GloVe/"

nlp = StanfordCoreNLP(r'/home/ken/stanford-corenlp-4.2.0')
lemmatizer = WordNetLemmatizer()


input_relateword_sets = [
    "input",
    "given",
    "give",
    "provide",
    "receive",
    "take"
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

def Read_Sentences(dir_path, filename = 'sentences'):
    file = open(dir_path + filename, "r+")
    sentence_group = []
    x = file.readline()
    while(x != ""):
        sentence_group.append(SentencesProcessing(x))
        x = file.readline()
    file.close()
    return sentence_group

def POS(sentence):
    return nlp.pos_tag(sentence)

def DEP(sentence):
    return nlp.dependency_parse(sentence)

def tokenize(sentence):
    return nlp.word_tokenize(sentence)

#def is_input_keyword(token):
#    return (token in input_relateword_sets)

def is_input_keyword(token):
    return (lemmatizer.lemmatize(token, pos="v").lower() in input_relateword_sets)

def is_input_keyword_woGiven(token):
    return (lemmatizer.lemmatize(token, pos="v").lower() in input_relateword_sets) and token.lower() != "given"

def is_given(token):
    if token == 'given':
        return True
    return (lemmatizer.lemmatize(token, pos="v").lower() == 'given')

def is_input(token):
    if token == 'input':
        return True
    return (lemmatizer.lemmatize(token, pos="v").lower() == 'input')

def is_give(token):
    if token == 'give':
        return True
    return (lemmatizer.lemmatize(token, pos="v").lower() == 'give')

def is_provide(token):
    if token == 'provide':
        return True
    return (lemmatizer.lemmatize(token, pos="v").lower() == 'provide')

def is_receive(token):
    if token == 'receive':
        return True
    return (lemmatizer.lemmatize(token, pos="v").lower() == 'receive')

def is_take(token):
    if token == 'take':
        return True
    return (lemmatizer.lemmatize(token, pos="v").lower() == 'take')


def is_NN(pos):
    if pos == 'NN' or pos == 'NNS' or pos =='NNP' or pos == 'NNPS':
        return True
    else:
        return False
    
def is_DT(pos):
    if pos == 'DT':
        return True
    else:
        return False

def is_prep(token):
    if token == 'of' or token == 'for' or token =='from' or token == 'to':
        return True
    else:
        return False
    
def is_of(token):
    return token == 'of'

def is_for(token):
    return token == 'for'

def is_from(token):
    return token == 'from'

def is_to(token):
    return token == 'to'

def is_cc(pos):
    return pos == 'CC'

def find_difference(anchor, target):
    difference = []
    for error in anchor:
        if not(error in target):
            difference.append(error)
    return difference

def is_uniqueWord(word):
    word_len = len(word)
    for i in range(word_len):
        if i != 0 and word[i].isupper():
            return 1
    return 0

#N_gram = 3
def word2features(sent, i, dep):
    
    features = []
    #Begin and end of sequence
    if i == 0:
        features.append('BOS')
    if i == len(sent) - 1:
        features.append('EOS')
    
    #Add bias
    features.append('bias')
    
    current_word = sent[i][0]
    current_pos = sent[i][1]
    
    #if is_given(current_word):
    #    features.append('is_given = True')
    #if is_uniqueWord(current_word):
    #    features.append('Unique_word = True')
    
    if i > 0:
        previous_word = sent[i-1][0]
        previous_pos = sent[i-1][1]
        
        if is_given(previous_word) and (is_DT(current_pos) or is_NN(current_pos)):
            features.append('First_extraction = True')
        elif is_cc(previous_pos) and (is_DT(current_pos) or is_NN(current_pos)):
            features.append('Last_extraction = True')
        
    if i < len(sent) - 1:
        next_word = sent[i+1][0]
        next_pos = sent[i+1][1]
        
        if is_NN(next_pos) and is_NN(current_pos):
            features.append('Compound_noun = True')
        elif is_NN(current_pos) and next_word == ',':
            features.append('next_is_comma = True')
        elif current_word == ',' and not(is_NN(next_pos)):
            features.append('next_word_isnt_NN = True')
        elif current_word == ',' and is_uniqueWord(next_word):
            features.append('next_word_Unique_word = True')
    
    #Dependency features
    num_root = 0
    last_period = 0
    for d in dep:
        if d[0] == 'ROOT':
            num_root +=1
        while num_root > 1:
            if last_period == len(sent):
                print(sent)
            if sent[last_period][1] == '.':
                num_root -= 1
            last_period += 1
            
        if d[0] == 'amod' and last_period + d[1]-1 == i:
            if is_given(sent[last_period + d[2]-1][0]):              
                num_given = 0
                for index in range(d[2]):
                    if is_given(sent[index + last_period][0]):
                        num_given += 1
                    elif sent[index + last_period][1] == '.':
                        num_given = 0
                if num_given > 1 or is_given(sent[0][0]):
                    features.append('Should_Extraction=False')
                    break
                features.append('Should_Extraction=True')
                break
                
    #features = [
        #'bias',
        #'word=' + current_word,
        #'word.islower={}'.format(current_word.islower()),
        #'word.isupper={}'.format(current_word.isupper()),
        #'word.is_input_keyword={}'.format(is_input_keyword(current_word)),
        #'word.is_input_keyword={}'.format(is_input_keyword_woGiven(current_word)),
        #'word.is_prep={}'.format(is_prep(current_word)),
        #'pos=' + pos,
        #'word.is_given={}'.format(is_given(current_word)),
    #]
    
   
    
    return features

def sent2features(sent, dep):
    return [word2features(sent, i, dep) for i in range(len(sent))]

def convert_label_to_CRF_label(X, Y):
    label_sequence = np.zeros(len(X))
    i = 0
    j = 0
    flag = 0
    start = -1
    s_label = -1
    e_label = -1
    if len(Y) == 0:
        return label_sequence
    while i < len(X):
        if lemmatizer.lemmatize(X[i], pos="v").lower() in input_relateword_sets:
            flag = 1
        if X[i] == Y[j]:
            start = i
            s_label = j
            k = j
            while k < len(Y):
                if Y[k] == ";" or Y[k] == ".":
                    break
                k += 1
            e_label = k
            if X[start:start + (e_label - s_label)] == Y[s_label:e_label]:
                if flag == 1:
                    label_sequence[start:start + (e_label - s_label)] = 1
                    #print(start)
                    #print(start + (e_label - s_label))
                    flag = 0
                    i = start + (e_label - s_label)
                    j = e_label
                else:
                    while X[i] != ".":
                        if lemmatizer.lemmatize(X[i], pos="v").lower() in input_relateword_sets:
                            label_sequence[start:start + (e_label - s_label)] = 1
                            i = start + (e_label - s_label)
                            j = e_label
                            break
                        i += 1
            else:
                i += 1
        else:
            i += 1
        if j >= len(Y):
            break
        if Y[j] == ";" or Y[j] == ".":
            if j == len(Y) - 1:
                break
            j += 1
    return label_sequence

def convert_int_to_str(array):
    str_seq = [str(x) for x in array]
    return str_seq

def convert_CRF2Sen(sentence, y):
    sen = ""
    for i in range(len(y)):
        if y[i] == str(1.0):
            sen += sentence[i] + " "
    return sen

def find_sen_index(sent, dataset):
    for index in range(len(dataset)):
        if sent == dataset[index]:
            return index
    return -1

raw_sentences = Read_Sentences(TrainDataPath, "sentences_test3_DT")
label_sentences = Read_Sentences(TrainDataPath, "NN_I_test3_DT")
train_x = [tokenize(sent) for sent in raw_sentences]
train_y = [tokenize(sent) for sent in label_sentences]


test_sentences = Read_Sentences(TestDataPath, "sentences_DT")
test_label = Read_Sentences(TestDataPath, "NN_I_DT")
test_x = [tokenize(sent) for sent in test_sentences]
test_y = [tokenize(sent) for sent in test_label]
testing_data = [POS(sent) for sent in test_sentences]
testing_data_dep = [DEP(sent) for sent in test_sentences]
X_test = [sent2features(sent, testing_data_dep[find_sen_index(sent, testing_data)]) for sent in testing_data]
Y_test_integer = []
for i in range(len(test_x)):
    Y_test_integer.append(convert_label_to_CRF_label(test_x[i], test_y[i]))
Y_test = [convert_int_to_str(seq) for seq in Y_test_integer]
#label_sequence = convert_label_to_CRF_label(train_x[5], train_y[5])
'''
Y_train = []
for i in range(len(train_x)):
    print(i)
    Y_train.append(convert_label_to_CRF_label(train_x[i], train_y[i]))
'''

training_data = [POS(sent) for sent in raw_sentences]
training_data_dep = [DEP(sent) for sent in raw_sentences]
training_label = [tokenize(sent) for sent in label_sentences]
#print(training_data[0][0][0].isupper())
X_train = [sent2features(sent, training_data_dep[find_sen_index(sent, training_data)]) for sent in training_data]
#Y_train = training_label
Y_train_integer = []
for i in range(len(train_x)):
    Y_train_integer.append(convert_label_to_CRF_label(train_x[i], train_y[i]))
Y_train = [convert_int_to_str(seq) for seq in Y_train_integer]



#Training Data Index (need to -1 for 0-index)
receive_obj_NN = [1,3]
provide_obj_NN = [2,4]
take_obj_NN = [4,6,66]
given_case_NN = [5,7,8,9,10,12,13,14,15,17,25,26,29,30,32,34,36,37,38,41,42,43,45,46,58,59,60,62,63,64]
given_amod_NN = [11,19,20,33,39,48,49,56,57,61,65]
inputNN_nsubj_NN = [16,35,54]
areGiven_obj_NN = [18,22,24,31,47,53,55]
NowGiven_obj_NN = [21]
NowGiven_iobj_NN = [27,28]
inputNN_nmod_NN = [23]
isGiven_nsubjpass_NN = [40]
given_mark_VB = [50]
given_acl_NN = [51]
given_obj_NN = [52]
inputNN_compound_NN = []





#Training
'''
model = pycrfsuite.Trainer(verbose=True)
for xseq, yseq in zip(X_train, Y_train):
    model.append(xseq, yseq)
'''
crf = sklearn_crfsuite.CRF(
        c1 = 0,
        c2 = 5e-2,
        max_iterations = 200,
        all_possible_transitions = True
    )

crf.fit(X_train, Y_train)

train_prediction = crf.predict(X_train)
Train_Correct = 0
train_fault = []
train_answer = []
for i in range(len(X_train)):
    if train_prediction[i] == Y_train[i]:
        Train_Correct += 1
    else:
        train_fault.append(i)
        
for i in range(len(X_train)):
    train_sen = convert_CRF2Sen(train_x[i], train_prediction[i])
    train_answer.append(train_sen)

print("Training Accuracy: " + str(Train_Correct/len(X_train)))

prediction = crf.predict(X_test)
Correct = 0
fault = []
answer = []
for i in range(108):
    if prediction[i] == Y_test[i]:
        Correct += 1
    else:
        fault.append(i)
        
for i in range(108):
    sen = convert_CRF2Sen(test_x[i], prediction[i])
    answer.append(sen)


amod_target_index = [10,18,20,30,41,47,55] # [10,18,30,47,55]
Amod_error = [crf.predict_marginals_single(X_test[index]) for index in amod_target_index]
amod_panswer = [answer[index] for index in amod_target_index]
amod_answer = [test_y[index] for index in amod_target_index]

NN_target_index = [1,40,63,70] # [1,40,63,70]
NN_error = [crf.predict_marginals_single(X_test[index]) for index in NN_target_index]
NN_panswer = [answer[index] for index in NN_target_index]
NN_answer = [test_y[index] for index in NN_target_index]

AUX_target_index = [69,76,89,91,98] # [69,76,89,91,98]
AUX_error = [crf.predict_marginals_single(X_test[index]) for index in AUX_target_index]
AUX_panswer = [answer[index] for index in AUX_target_index]
AUX_answer = [test_y[index] for index in AUX_target_index]

case_target_index = [67,105] # [67,105]
case_error = [crf.predict_marginals_single(X_test[index]) for index in case_target_index]
case_panswer = [answer[index] for index in case_target_index]
case_answer = [test_y[index] for index in case_target_index]

print("Testing Accuracy: " + str(Correct/108))
'''
model.set_params({
        'c1': 1.0,
        'c2': 1e-3,
        'max_iterations': 200,
        'feature.possible_transitions': True,
        'feature.minfreq': 3
    })
model.train(model)
#Testing
tagger = pycrfsuite.Tagger()
tagger.open(model)
'''

def fault_type(fault, prediction):
    fault_error = []
    for index in fault:
        if prediction[index] == "":
            fault_error.append(0)
        else:
            fault_error.append(1)
    return fault_error

def word_error(fault, prediction, answer):
    word_err = []
    for error in fault:
        seq = answer[error]
        p_seq = prediction[error]
        total_word = len(seq)
        error_word = 0
        for index in range(len(seq)):
            if seq[index] == '1.0' and p_seq[index] == '0.0':
                error_word += 1
            elif seq[index] == '0.0' and p_seq[index] == '1.0':
                error_word += 1
        word_err.append([error, error_word, total_word, error_word/total_word])
    return word_err
                


t_fault_error = fault_type(train_fault, train_answer)
fault_error = fault_type(fault, answer)

state_features = crf.state_features_
crffeatures = []
for key, index in state_features.items():
    crffeatures.append(key)
    
def feature_weight_extraction(X, Y, state_features, features):
    weights = []
    for word_index in range(len(X)):
        word_weight = []
        for feature in X[word_index]:
            target_pair = (feature,Y[word_index])
            if target_pair in features:
                word_weight.append([target_pair,state_features[target_pair]])
            else:
                word_weight.append([target_pair,None])
        weights.append(word_weight)
    return weights

def possible_feature_weight_extraction(X, state_features, features):
    weights_0 = []
    weights_1 = []
    for word_index in range(len(X)):
        word_weight_0 = []
        word_weight_1 = []
        for feature in X[word_index]:
            target_pair_0 = (feature,'0.0')
            target_pair_1 = (feature,'1.0')
            if target_pair_0 in features:
                word_weight_0.append([target_pair_0,state_features[target_pair_0]])
            else:
                word_weight_0.append([target_pair_0,None])
            if target_pair_1 in features:
                word_weight_1.append([target_pair_1,state_features[target_pair_1]])
            else:
                word_weight_1.append([target_pair_1,None])
        weights_0.append(word_weight_0)
        weights_1.append(word_weight_1)
    return weights_0, weights_1

train_feature_weights = []
test_feature_weights = []
prediction_train_feature_weights = []
prediction_test_feature_weights = []

weights_0 = []
weights_1 = []

train_weights_0 = []
train_weights_1 = []

for index in range(len(X_train)):
    train_feature_weights.append(feature_weight_extraction(X_train[index], Y_train[index], state_features, crffeatures))
    prediction_train_feature_weights.append(feature_weight_extraction(X_train[index], train_prediction[index], state_features, crffeatures))
    temp_0, temp_1 = possible_feature_weight_extraction(X_train[index], state_features, crffeatures)
    train_weights_0.append(temp_0)
    train_weights_1.append(temp_1)
    
for index in range(len(X_test)):
    test_feature_weights.append(feature_weight_extraction(X_test[index], Y_test[index], state_features, crffeatures))
    prediction_test_feature_weights.append(feature_weight_extraction(X_test[index], prediction[index], state_features, crffeatures))
    temp_0, temp_1 = possible_feature_weight_extraction(X_test[index], state_features, crffeatures)
    weights_0.append(temp_0)
    weights_1.append(temp_1)


#Training Data Index (need to -1 for 0-index)
receive_obj_NN = [1,3]
provide_obj_NN = [2,4]
take_obj_NN = [4,6,66]
given_case_NN = [5,7,8,9,10,12,13,14,15,17,25,26,29,30,32,34,36,37,38,41,42,43,45,46,58,59,60,62,63,64]
given_amod_NN = [11,19,20,33,39,48,49,56,57,61,65]
inputNN_nsubj_NN = [16,35,54]
areGiven_obj_NN = [18,22,24,31,47,53,55]
NowGiven_obj_NN = [21]
NowGiven_iobj_NN = [27,28]
inputNN_nmod_NN = [23]
isGiven_nsubjpass_NN = [40]
given_mark_VB = [50]
given_acl_NN = [51]
given_obj_NN = [52]
inputNN_compound_NN = []

training_classification_result = (
    receive_obj_NN,
    provide_obj_NN,
    take_obj_NN,
    given_case_NN,
    given_amod_NN,
    inputNN_nsubj_NN,
    areGiven_obj_NN,
    NowGiven_obj_NN,
    NowGiven_iobj_NN,
    inputNN_nmod_NN,
    isGiven_nsubjpass_NN,
    given_mark_VB,
    given_acl_NN,
    given_obj_NN,
    inputNN_compound_NN
    )

#Testing Data Index (need to -1 for 0-index)
t_receive_obj_NN = []
t_provide_obj_NN = []
t_take_obj_NN = [18]
t_given_case_NN = [3,6,7,9,10,13,17,20,23,26,27,30,34,36,38,40,43,44,45,46,50,53,60,61,62,68,73,83,
                 86,87,88,89,93,94,97,98,104,106]
t_given_amod_NN = [11,12,19,21,22,24,25,31,42,48,56,79,105]
t_inputNN_nsubj_NN = []
t_areGiven_obj_NN = [29,37,58,59,65,66,69,70,82,90,92,101,102,63]#63 were given
t_NowGiven_obj_NN = []
t_NowGiven_iobj_NN = []
t_inputNN_nmod_NN = []
t_isGiven_nsubjpass_NN = [77,99]
t_given_mark_VB = []
t_given_acl_NN = [81]
t_given_obj_NN = []
t_inputNN_compound_NN = [2,28,41,64,71]

testing_classification_result = (
    t_receive_obj_NN,
    t_provide_obj_NN,
    t_take_obj_NN,
    t_given_case_NN,
    t_given_amod_NN,
    t_inputNN_nsubj_NN,
    t_areGiven_obj_NN,
    t_NowGiven_obj_NN,
    t_NowGiven_iobj_NN,
    t_inputNN_nmod_NN,
    t_isGiven_nsubjpass_NN,
    t_given_mark_VB,
    t_given_acl_NN,
    t_given_obj_NN,
    t_inputNN_compound_NN
    )

def classify_error(fault, mode):
    classification_result = []
    if mode == 0:
        flag = 0
        for error in fault:
            for index in range(len(training_classification_result)):
                if (error + 1) in training_classification_result[index]:
                    flag = 1
                    classification_result.append(index)
                    break
            if flag == 0:
                classification_result.append(15)
            else:
                flag = 0
    else:
        flag = 0
        for error in fault:
            for index in range(len(testing_classification_result)):
                if (error+1) in testing_classification_result[index]:
                    flag = 1
                    classification_result.append(index)
                    break
            if flag == 0:
                classification_result.append(15)
            else:
                flag = 0
                
    return classification_result

cr = classify_error(fault, 1)
t_cr = classify_error(train_fault, 0)

classfication_index_pair = {
    0:"receive_obj_NN",
    1:"provide_obj_NN",
    2:"take_obj_NN",
    3:"given_case_NN",
    4:"given_amod_NN",
    5:"inputNN_nsubj_NN",
    6:"areGiven_obj_NN",
    7:"NowGiven_obj_NN",
    8:"NowGiven_iobj_NN",
    9:"inputNN_nmod_NN",
    10:"isGiven_nsubjpass_NN",
    11:"given_mark_VB",
    12:"given_acl_NN",
    13:"given_obj_NN",
    14:"inputNN_compound_NN",
    15:"No need output"
    }

empty_index_pair = {
    0:"empty",
    1:"Non-empty"
    }

def drawPie(classification_result, mode = 0):
    cal_total_occurence = Counter(classification_result)
    target = []
    value = []
    for key, val in cal_total_occurence.items():
        if mode == 0:
            target.append(classfication_index_pair[key])
        else:
            target.append(empty_index_pair[key])
        value.append(val)
    plt.pie(value, labels = target, autopct="%1.1f%%")
    plt.axis('equal')
    plt.show()
    
def drawBar(classification_result, mode = 0):
    cal_total_occurence = Counter(classification_result)
    target = []
    value = []
    for key, val in cal_total_occurence.items():
        if mode == 0:
            target.append(key)
        else:
            target.append(empty_index_pair[key])
        value.append(val)
    x = np.arange(len(target))
    plt.bar(x, value)
    plt.xticks(x, target)
    plt.xlabel('Error type')
    plt.ylabel('Number of errors')
    #plt.title('Final Term')
    plt.show()

drawPie(t_cr,0)
drawPie(t_fault_error,1)
drawPie(cr,0)
drawPie(fault_error,1)
drawBar(t_cr,0)
drawBar(t_fault_error,1)
drawBar(cr,0)
drawBar(fault_error,1)

def find_meaning_weights(sen_feature, target_index, threshold = 0.5):
    meaning_weights = []
    target_sen = sen_feature[target_index]
    for item in target_sen:
        if item[1] == None:
            meaning_weights.append(item)
        elif abs(float(item[1])) >= threshold:
            meaning_weights.append(item)
    return meaning_weights

error = []
for err in train_fault:
    if err+1 in given_case_NN or err+1 in given_amod_NN:
        error.append(err)


#y_pred = crf.predict(X_train)
#print(y_pred[0])


nlp.close()
