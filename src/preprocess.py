import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from string import punctuation



if not 'nlp' in locals():
    print "Loading English Module..."
    nlp = spacy.load('en')
    print "Module loading complete."

def lemm_vec_string(doc, stop_words=ENGLISH_STOP_WORDS):
    try:
        doc = unicode(doc.translate(None, punctuation))
        doc = nlp(doc)
        tokens = [token.lemma for token in doc if token not in stop_words]
        return tokens
    except:
        print 'fail'


def vectorize_tweets(f, max_words=35):
    from keras.preprocessing import sequence
    df = pd.read_csv(f, header=None).iloc[1:,0]
    vec_tweets = df.map(lemm_vec_string)
    padded = sequence.pad_sequences(vec_tweets, maxlen=max_words)
    return padded


def lemm_string(doc, stop_words=ENGLISH_STOP_WORDS):
    try:
        doc.replace('  ', ' ')
        doc.replace('   ', ' ')
        doc = unicode(doc.translate(None, punctuation))
        doc = nlp(doc)
        tokens = [token.lemma_ for token in doc if token not in stop_words]
        return ' '.join(tokens)
    except:
        print 'fail'

def load_lemma_tweets(f):
    df = pd.read_excel(f)
    # df.iloc[:,0] = df.iloc[:,0].map(lemm_string)
    return df
