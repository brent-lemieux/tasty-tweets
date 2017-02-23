import pandas as pd
import spacy
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import pickle

twitter_speak = pickle.load( open( 'twitter_speak.pkl', 'rb' ) )


if not 'nlp' in locals():
    print "Loading English Module..."
    nlp = spacy.load('en')
    print "Module loading complete."

extra_stop = ['st', 'rd', 'ave', 'hiring', 'jobs']

def load_xls(f, lemma=False, pos=False, slang=False):
    df = pd.read_excel(f)
    df.dropna(inplace=True)
    if df.columns.tolist()[0] != 'tweets':
        df.columns = ['tweets']
    if slang:
        df['tweets'] = df['tweets'].map(convert_slang)
    if lemma and pos:
        df['tweets'] = df['tweets'].map(lemm_pos_string)
    elif lemma:
        df['tweets'] = df['tweets'].map(lemm_string)
    return df

def load_csv(f, lemma=False, pos=False, slang=False):
    df = pd.read_csv(f)
    df.dropna(inplace=True)
    if df.columns.tolist()[0] != 'tweets':
        df.columns = ['tweets']
    if slang:
        df['tweets'] = df['tweets'].map(convert_slang)
    if lemma and pos:
        df['tweets'] = df['tweets'].map(lemm_pos_string)
    elif lemma:
        df['tweets'] = df['tweets'].map(lemm_string)
    return df

def lemm_pos_string(doc, stop_words=ENGLISH_STOP_WORDS):
    parts_of_speech =[u'ADV', u'ADJ', u'VERB']
    doc = doc.replace('  ',' ')
    doc = doc.replace('_',' ')
    doc = unicode(doc)
    doc = nlp(doc)
    tokens = [token.lemma_ for token in doc if token not in stop_words and token.pos_ in parts_of_speech]
    return ' '.join(tokens)

def lemm_string(doc, stop_words=ENGLISH_STOP_WORDS):
    doc = doc.replace('  ',' ')
    doc = unicode(doc)
    doc = nlp(doc)
    tokens = [token.lemma_ for token in doc if token not in stop_words and token not in extra_stop]
    return ' '.join(tokens)


def convert_slang(doc, slang_dic=twitter_speak):
    for k, v in slang_dic.iteritems():
        if k in doc:
            doc = doc.replace(k, v)
    return doc
