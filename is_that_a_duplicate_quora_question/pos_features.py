from __future__ import division

import pandas as pd
import datetime
import operator
import spacy


# TODO: THIS SCRIPT NEEDS TO BE TRANSLATED TO SPARK


DATA_PATH = '../'
FILE_NAME = 'part_of_speech_benchmark.csv'

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


def __jaccard_distance(A, B, I):
    if A == B == 0:
        return 1.0
    else:
        return 1.0 - ( I / ((A + B) - I))


if __name__ == '__main__':

    df_train = pd.read_csv(DATA_PATH + 'train.csv')
    df_test  = pd.read_csv(DATA_PATH + 'test.csv')
    nlp = spacy.load('en')

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

    df = pd.concat([df_train, df_test])
    df['word_shares'] = df.apply(get_pos_counts, axis=1, raw=True)

    x = pd.DataFrame()

    for index, pos_features in enumerate(POS_FEATURES):
        x[pos_features] = df['word_shares'].apply(lambda x: float(x.split(':')[ index ]))

    x_train = x[:df_train.shape[0]]
    x_test  = x[df_train.shape[0]:]
    x_train.to_csv(DATA_PATH + 'train_' + FILE_NAME)
    x_test.to_csv(DATA_PATH + 'test_' + FILE_NAME)
