import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation


if not 'nlp' in locals():
    print "Loading English Module..."
    nlp = spacy.load('en')
    print "Module loading complete."

def lemm_vec_string(doc, stop_words=ENGLISH_STOP_WORDS):
    doc = unicode(doc)
    doc = nlp(doc)
    tokens = [token.lemma for token in doc if token not in stop_words]
    return tokens


def vectorize_tweets(f, max_words=35):
    from keras.preprocessing import sequence
    df = pd.read_csv(f, header=None).iloc[1:,:]
    tf = CountVectorizer(max_features=5000).fit_transform(df.iloc[:,0].values)
    vocab = tf.vocaulary_.keys()
    vec_tweets = df.iloc[:,0].map(lemm_vec_string())
    padded = sequence.pad_sequences(vec_tweets, maxlen=max_words)
    return padded


def vectorize_labeled_tweets(df, max_words=35):
    from keras.preprocessing import sequence
    df.dropna(inplace=True)
    tf = CountVectorizer(max_features=5000).fit_transform(df['tweets'].values)
    vocab = tf.vocaulary.keys()
    vec_tweets = df['tweets'].map(lemm_vec_string(vocab=vocab))
    padded = sequence.pad_sequences(vec_tweets, maxlen=max_words)
    return padded, df['labels']
