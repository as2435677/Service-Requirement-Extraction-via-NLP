#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:48:58 2020

@author: ken
"""

from stanfordcorenlp import StanfordCoreNLP
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from operator import is_not
from functools import partial 

#Tool Announcement
nlp = StanfordCoreNLP(r'/home/ken/stanford-corenlp-4.1.0')
pst = PorterStemmer()
lemmatizer = WordNetLemmatizer()
#Example sentences
sentence_group = ["Request1 specifies a request for compositions that receive as input the destCity (the name of a city destination for a trip) and DateTime entities, and return the Name of a theatre for that city and the receipt for the booking performed (TheatreName, TicketConfirmaton, PlaceNum entities)",
       "The user wishes to provide as inputs a book title and author, credit card information and the address that the book will be shipped to. The outputs of the desired composite service are a payment from the credit card for the purchase, as well as shipping dates and customs cost for the specific item.",
       "Book a Hotel and a Theatre in a city where he will arrive in a given date and from which will depart from in another given date.",
       "Request2 specifies compositions that return HotelConfirmation and RentalConfirmation (the receipt of the booking of a Hotel and a car in city that is the destination for a trip) receiving destCity (the name of the city destination of the trip), ArrivalDateTime (the arrival date), and departDateTime (the departure date) as inputs.",
       "In the scenario the user takes a picture of a barcode of a product and provides his current local position. He wants to obtain the cheapest price offered by a shop close to his position, the GPS coordinates of this shop and a review of the product.",
       "Overall response time should be less than 200 ms and overall reliability should be more than 90%.",
       "Restricting the overall response time to be less than 50s.",
       "The total price of the composite service execution should be at most $500.",
       "Sa returns the blood pressure (BP) of a patient given his PatientID (PID) and DeviceAddress (Add); Sb and Sb2 return respectively the supervisor (Person) and a physician of an organisation (Org); Sc returns a Warning level (WL) given a blood pressure; Sd returns the Emergency department given a level of Warning; Se returns the Organization given a Warning level.",
       "PatientByIDService takes as input a patient ID to return the patient’s description. CurrentTreatmentByPatientIDService takes a patient ID to return the patient’s current treatment. MedicalHistoryByPatientIDService takes a patient ID to return the patient’s previous treatments. MedicationByTreatmentIDService takes a treatment ID to return the medication involved in this treatment. DrugclassByMedicaionService takes a medication ID to return the drug class of this medication (indicates the different risks to be associated to the medication).",
       "Given the hotel name, city, and state information, findHotel returns the address and zip code of the hotel.",
       "Given the zip code and food preference, findRestaurant returns the name, phone number, and address of the restaurant with matching food preference and closest to the zip code.",
       "Given the current location and food preference, guideRestaurant returns the address of the closest restaurant and its rating.",
       "Given the start and destination addresses, findDirection returns a detailed step-by-step driving direction and a map image of the destination address."
       ]

#input and output keywords sets
input_relateword_sets = [
    "input",
    "given",
    "provide"
    ]

output_relateword_sets = [
    "output",
    "return",
    "obtain"
    ]

#Announce Service name
Service_names = []

#Announcement of tuple-based groups
I = []
O = []
P = []
E = []
                    
#Get pos_tag, dependency and token
def get_pos_dep_token(sen):
    pos_tag = nlp.pos_tag(sen)
    dep_parse = nlp.dependency_parse(sen)
    tokens = nlp.word_tokenize(sen)
    tokens.insert(0,'ROOT')
    pos_tag.insert(0,('ROOT', 'ROOT'))
    return pos_tag, dep_parse, tokens

#Recover Sentence
def recover_sentence(tokens):
    sen = ''
    #if flag = 1, need to reduce one space
    flag = 0
    for token in tokens:
        if token == 'ROOT':
            continue
        elif token == '_':
            sen = sen + token
            flag = 1
        else:
            if flag == 1:
                sen = sen + token
                flag = 0
                continue
            sen = sen + ' ' + token
    return sen

#Replace unique service name with str - Service
def modify_sentence(tokens, pos_tag):
    flag = 0
    for tag in pos_tag:
        if tag[1] == 'JJ':
            if tag[0][0].islower():
                flag = 1
            for ch in tag[0]:
                if ch.isupper() and flag == 1:
                    flag = 0
                    target_index = tokens.index(tag[0])
                    if tag[0] in Service_names:
                        tokens[target_index] = 'Service' + str(Service_names.index(tag[0]))
                        continue
                    number_of_service = len(Service_names)
                    Service_names.append(tokens[target_index])
                    tokens[target_index] = 'Service' + str(number_of_service)
    #replace - to _
    tokens = ['_' if x == '-' in tokens else x for x in tokens]
    sen = recover_sentence(tokens)
    return sen

#Find the dependency of a certain word
def find_keyword_relate_dep(word, dep_parse, tokens):
    relate_dep = []
    for dep in dep_parse:
        if dep[0] == 'ROOT':
            continue
        if pst.stem(tokens[(dep[1])]) == pst.stem(word) or pst.stem(tokens[(dep[2])]) == pst.stem(word):
            relate_dep.append(dep)
    return relate_dep

#Recover the compound
def recover_compound(token_index, dep_parse, tokens, pos_tag):
    for dep in dep_parse:
        if dep[0] == 'compound' and (dep[1] == token_index or dep[2] == token_index):
            return tokens[dep[2]] + '_' + tokens[dep[1]]
        elif pos_tag[token_index-1][1] == 'NN':
            return tokens[token_index-1] + '_' + tokens[token_index]
        elif pos_tag[token_index+1][1] == 'NN':
            return tokens[token_index] + '_' + tokens[token_index+1]
    return tokens[token_index]

#Find conjunction words
def find_conj_words(token_index, dep_parse, tokens, pos_tag, keywords):
    words_index = []
    words_index.append(token_index)
    for dep in dep_parse:
        if dep[0] == 'conj' and dep[1] == token_index:
            words_index.append(dep[2])
    #Recover the compound word
    for index in words_index:
        #keywords.append(recover_compound(index, dep_parse, tokens, pos_tag))
        find_of_information(index, dep_parse, tokens, pos_tag, keywords)
    

#Find comprehensive information of target noun
def find_of_information(token_index, dep_parse, tokens, pos_tag, keywords):
    target_index = -1
    k = 1
    extend_word = recover_compound(token_index, dep_parse, tokens, pos_tag)
    for dep in dep_parse:
        if dep[0] == 'nmod' and dep[1] == token_index:
            target_index = dep[2]
        elif dep[0] == 'case' and dep[1] == token_index and tokens[dep[2]] == 'of':
            k = 0
    #print(target_index)
    if target_index == -1 and k:
        keywords.append(extend_word)
    #Check nmod is related to of condition
    for dep in dep_parse:
        if dep[0] == 'case' and dep[1] == target_index and tokens[dep[2]] == 'of':
            extend_word = recover_compound(target_index, dep_parse, tokens, pos_tag) + "_" + extend_word
            keywords.append(extend_word)
            find_conj_words(target_index, dep_parse, tokens, pos_tag, keywords)
            

#Find the target noun
def find_noun_components(relate_dep, dep_parse, tokens, pos_tag):
    keywords  = []
    for target in relate_dep:
        #find the noun that has dependency with key verb
        if target[0] == 'case':
            target_noun_index = target[1]
            #keywords.append(find_of_information(target_noun_index, dep_parse, tokens, pos_tag))
            find_conj_words(target_noun_index, dep_parse, tokens, pos_tag, keywords)
        elif target[0] == 'obj':
            target_noun_index = target[2]
            #keywords.append(find_of_information(target_noun_index, dep_parse, tokens, pos_tag))
            find_conj_words(target_noun_index, dep_parse, tokens, pos_tag, keywords)
        elif target[0] == 'conj':
            target_noun_index = target[2]
            #keywords.append(find_of_information(target_noun_index, dep_parse, tokens, pos_tag))
            find_conj_words(target_noun_index, dep_parse, tokens, pos_tag, keywords)
    return keywords


sen = sentence_group[12]
pos_tag, dep_parse, tokens = get_pos_dep_token(sen)
sen = modify_sentence(tokens, pos_tag)
#sen = " The user wishes to provide as inputs a book title and author , credit card information and the address that the book will be shipped to ."
#pos_tag, dep_parse, tokens = get_pos_dep_token(sen)

#Sentence tokenizer
sen_token = nltk.sent_tokenize(sen)
for i in range(len(sen_token)):
    pos_tag, dep_parse, tokens = get_pos_dep_token(sen_token[i])
    for token in tokens:
        if (lemmatizer.lemmatize(token, pos="v")).lower() in input_relateword_sets and pos_tag[tokens.index(token)][1] != 'NNS':
            input_relate_dep = find_keyword_relate_dep(token, dep_parse, tokens)
            I = find_noun_components(input_relate_dep, dep_parse, tokens, pos_tag)
            #I.append(find_noun_components(input_relate_dep, dep_parse, tokens, pos_tag))
        elif (lemmatizer.lemmatize(token, pos="v")).lower() in output_relateword_sets:
            output_relate_dep = find_keyword_relate_dep(token, dep_parse, tokens)
            O = find_noun_components(output_relate_dep, dep_parse, tokens, pos_tag)
            #O.append(find_noun_components(output_relate_dep, dep_parse, tokens, pos_tag))

