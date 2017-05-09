
# coding: utf-8

# In[2]:

from __future__ import division

import pandas as pd
import datetime
import operator
import spacy


# In[3]:

DATA_PATH = "../../../data/quora_data_set/"
FILE_NAME = 'spacy_features.csv'


# In[4]:

POS_TYPES = [
    'PUNCT',
    'SYM',
    'ADJ',
    'VERB',
    'CONJ',
    'NUM',
    'DET',
    'ADV',
    'ADP',
    'NOUN',
    'PROPN',
    'PART',
    'PRON',
    'INTJ',
]

POS_FEATURES = ['Q1_' + i for i in POS_TYPES] +\
               ['Q2_' + i for i in POS_TYPES] +\
               ['SHARED_' + i for i in POS_TYPES] +\
               ['JACCARD_' + i for i in POS_TYPES]

# In[5]:

ENTITY_TYPES = [
    'PERSON',
    'NORP',
    'FACILITY',
    'ORG',
    'GPE',
    'LOC',
    'PRODUCT',
    'EVENT',
    'WORK_OF_ART',
    'LANGUAGE',
    'DATE',
    'TIME',
    'PERCENT',
    'MONEY',
    'QUANTITY',
    'ORDINAL',
    'CARDINAL'
]

ENTITY_FEATURES = ['Q1_' + i for i in ENTITY_TYPES] +\
                  ['Q2_' + i for i in ENTITY_TYPES] +\
                  ['SHARED_' + i for i in ENTITY_TYPES] +\
                  ['JACCARD_' + i for i in ENTITY_TYPES]


# In[6]:

def __jaccard_distance(A, B, I):
    if A == B == 0:
        return 1.0
    else:
        return 1.0 - ( I / ((A + B) - I))


# In[12]:

def get_pos_counts(row):

        doc1 = [] if type(row['question1']) != str else nlp(row['question1'])
        doc2 = [] if type(row['question2']) != str else nlp(row['question2'])
        pos_data = {pos: [set([]), set([])] for pos in POS_TYPES}
        pos_features = {}

        if doc1 == doc2 == []:
            return '0:0:0:2'

        for t in doc1:
            if t.pos_ in pos_data:
                pos_data[t.pos_][0].add(t.text.lower())

        for t in doc2:
            if t.pos_ in pos_data:
                pos_data[t.pos_][1].add(t.text.lower())

        for pos in POS_TYPES:
            q1 = len(pos_data[pos][0])
            q2 = len(pos_data[pos][1])
            shared = len(pos_data[pos][0].intersection(pos_data[pos][1]))
            pos_features['Q1_' + pos] = q1
            pos_features['Q2_' + pos] = q2
            pos_features['SHARED_' + pos] = shared
            pos_features['JACCARD_' + pos] = __jaccard_distance(q1, q2, shared)

        return ('{}:{}:{}:{}:'*len(POS_TYPES)).format(*[pos_features[i] for i in POS_FEATURES])


# In[23]:

def get_entity_counts(row):

        doc1 = [] if type(row['question1']) != str else nlp(row['question1'])
        doc2 = [] if type(row['question2']) != str else nlp(row['question2'])
        ent_data = {ent: [set([]), set([])] for ent in ENTITY_TYPES}
        ent_features = {}

        if doc1 == doc2 == []:
            return '0:0:0:2'

        for t in doc1:
            if str(t.ent_type_) in ent_data:
                ent_data[str(t.ent_type_)][0].add(t.text.lower())

        for t in doc2:
            if str(t.ent_type_) in ent_data:
                ent_data[str(t.ent_type_)][2].add(t.text.lower())

        for ent in ENTITY_TYPES:
            q1 = len(ent_data[ent][0])
            q2 = len(ent_data[ent][1])
            shared = len(ent_data[ent][0].intersection(ent_data[ent][1]))
            ent_features['Q1_' + ent] = q1
            ent_features['Q2_' + ent] = q2
            ent_features['SHARED_' + ent] = shared
            ent_features['JACCARD_' + ent] = __jaccard_distance(q1, q2, shared)

        return ('{}:{}:{}:{}:'*len(ENTITY_TYPES)).format(*[ent_features[i] for i in ENTITY_FEATURES])

# In[18]:

sample_train = df_train.head()
sample_test = df_test.head()


# In[24]:

if __name__ == '__main__':

    df_train = pd.read_csv(DATA_PATH + 'train.csv')
    df_test  = pd.read_csv(DATA_PATH + 'test.csv')
    nlp = spacy.load('en')

    df = pd.concat([df_train, df_test])
    # df = pd.concat([sample_train, sample_test])
    df['word_shares'] = df.apply(get_entity_counts, axis=1, raw=True)

    x = pd.DataFrame()

    for index, ent_features in enumerate(ENTITY_FEATURES):
        x[ent_features] = df['word_shares'].apply(lambda x: float(x.split(':')[ index ]))

    x_train = x[:df_train.shape[0]]
    x_test  = x[df_train.shape[0]:]
    # x_train = x[:sample_train.shape[0]]
    # x_test  = x[sample_test.shape[0]:]
    x_train.to_csv(DATA_PATH + 'train_' + FILE_NAME)
    x_test.to_csv(DATA_PATH + 'test_' + FILE_NAME)
